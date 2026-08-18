use serde::Deserialize;
use std::fs;
use std::path::Path;

fn default_arrival_rate() -> f64 {
    1.0
}

/// How requests arrive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalPattern {
    /// Poisson process at `arrival_rate` (or `rate_schedule`).
    Poisson,
    /// Constant inter-arrival time `1 / arrival_rate`.
    #[serde(alias = "fixed_rate")]
    Uniform,
    /// Alternating bursts (1–10 ms gaps, 20% of the time) and lulls
    /// (0.5–2 s gaps). Ignores `arrival_rate`.
    Burst,
    /// `num_concurrent_users` users, each issuing its next request when its
    /// previous one completes.
    ClosedLoop,
    /// Every request arrives at t = 0.
    Batched,
}

impl ArrivalPattern {
    /// Whether arrivals are driven by completions rather than a clock.
    pub fn is_closed_loop(self) -> bool {
        matches!(self, ArrivalPattern::ClosedLoop)
    }
}

/// A workload file is this table at top level:
///
/// ```toml
/// arrival_pattern = "closed_loop"
/// num_concurrent_users = 256
/// num_requests = 2000
/// seed = 7
/// input_len_dist = { type = "lognormal", mean = 7.0, std_dev = 0.5 }
/// output_len_dist = { type = "lognormal", mean = 6.5, std_dev = 0.8 }
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadConfig {
    /// Path to dataset file (JSONL in OpenAI batch API format)
    /// If provided, dataset mode is used instead of synthetic workload
    #[serde(default)]
    pub dataset_path: Option<String>,

    /// Path to a session file (JSONL, one session per line; see
    /// [`crate::request::session`]). Session mode: the arrival pattern
    /// governs when sessions *start* (`arrival_rate` is sessions/sec;
    /// `closed_loop` keeps `num_concurrent_users` sessions in flight), and
    /// each later step of a session arrives at its parent's completion plus
    /// the step's gap. Mutually exclusive with `dataset_path`; the length
    /// distributions are ignored.
    #[serde(default)]
    pub sessions_path: Option<String>,

    /// Session mode: maximum number of session starts. This does not limit
    /// the total request steps emitted by those sessions; `num_requests`
    /// provides that separate bound. Sessions cycle through the file, so this
    /// may exceed the file's count.
    #[serde(default)]
    pub num_sessions: Option<usize>,

    pub arrival_pattern: ArrivalPattern,

    /// Mean arrival rate (requests per second) for the open-loop patterns.
    #[serde(default = "default_arrival_rate")]
    pub arrival_rate: f64,

    /// Optional time-varying arrival rate λ(t). When present, supplies the rate
    /// at each instant instead of the constant `arrival_rate`. Open-loop
    /// patterns only (poisson/uniform/fixed_rate); ignored for
    /// closed_loop/batched. Pair with a large `num_requests` (or
    /// `duration_secs`) to run whole cycles.
    #[serde(default)]
    pub rate_schedule: Option<RateSchedule>,

    /// Input sequence length distribution (ignored in dataset mode)
    pub input_len_dist: LengthDistribution,

    /// Output sequence length distribution (ignored in dataset mode)
    pub output_len_dist: LengthDistribution,

    /// Maximum total requests generated. In session mode every step of every
    /// session counts toward this limit; use `num_sessions` to bound session
    /// starts instead. `None` leaves the total request count unbounded.
    pub num_requests: Option<usize>,

    /// Number of concurrent users for closed-loop pattern
    /// Each user immediately sends a new request when their previous one completes
    #[serde(default)]
    pub num_concurrent_users: Option<usize>,

    /// Optional uniform jitter (in seconds) added to closed-loop request
    /// arrivals. Each replenished request arrives at `completion_time +
    /// Uniform(0, jitter)`. Used to break the synchronized-arrival regime
    /// that closed-loop with fixed ISL/OSL otherwise produces.
    #[serde(default)]
    pub closed_loop_jitter_secs: Option<f64>,

    /// Stop admitting arrivals after this many simulated seconds. Requests
    /// admitted by the deadline are allowed to finish. Without this bound,
    /// generation stops at `num_requests` (or a source-specific limit).
    pub duration_secs: Option<f64>,

    /// Random seed for reproducibility
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum LengthDistribution {
    #[serde(rename = "fixed")]
    Fixed { value: u32 },

    #[serde(rename = "uniform")]
    Uniform { min: u32, max: u32 },

    #[serde(rename = "normal")]
    Normal { mean: f64, std_dev: f64 },

    #[serde(rename = "lognormal")]
    LogNormal { mean: f64, std_dev: f64 },
}

/// Time-varying arrival rate λ(t), requests/sec. Set on the workload to drive
/// the open-loop arrival process through changing load within a single run.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum RateSchedule {
    /// Sinusoid between `min` and `max`, starting in the trough at t=0 and
    /// peaking at half-period.
    #[serde(rename = "sine")]
    Sine {
        min: f64,
        max: f64,
        period_secs: f64,
    },
    /// On/off bursts: `high` for the first `duty` fraction of each period,
    /// `low` for the rest.
    #[serde(rename = "square")]
    Square {
        low: f64,
        high: f64,
        period_secs: f64,
        duty: f64,
    },
    /// Piecewise-linear (time_secs, rate) points, linearly interpolated and
    /// held flat outside the first/last point. Replays a measured load curve.
    #[serde(rename = "trace")]
    Trace { points: Vec<[f64; 2]> },
}

impl RateSchedule {
    /// Instantaneous arrival rate λ(t) ≥ 0, requests/sec.
    pub fn rate_at(&self, t: f64) -> f64 {
        let r = match self {
            RateSchedule::Sine {
                min,
                max,
                period_secs,
            } => {
                let phase = 2.0 * std::f64::consts::PI * t / period_secs.max(1e-9);
                min + (max - min) * 0.5 * (1.0 - phase.cos())
            }
            RateSchedule::Square {
                low,
                high,
                period_secs,
                duty,
            } => {
                let frac = (t / period_secs.max(1e-9)).rem_euclid(1.0);
                if frac < *duty {
                    *high
                } else {
                    *low
                }
            }
            RateSchedule::Trace { points } => {
                if points.is_empty() {
                    return 0.0;
                }
                if t <= points[0][0] {
                    return points[0][1].max(0.0);
                }
                let last = points[points.len() - 1];
                if t >= last[0] {
                    return last[1].max(0.0);
                }
                let mut out = last[1];
                for w in points.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if t >= a[0] && t <= b[0] {
                        let f = (t - a[0]) / (b[0] - a[0]).max(1e-9);
                        out = a[1] + f * (b[1] - a[1]);
                        break;
                    }
                }
                out
            }
        };
        r.max(0.0)
    }
}

impl LengthDistribution {
    /// Sample a value from this distribution
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> u32 {
        use rand::RngExt;
        use rand_distr::Distribution;

        match self {
            LengthDistribution::Fixed { value } => *value,
            LengthDistribution::Uniform { min, max } => rng.random_range(*min..=*max),
            LengthDistribution::Normal { mean, std_dev } => {
                let normal = rand_distr::Normal::new(*mean, *std_dev).unwrap();
                normal.sample(rng).max(1.0) as u32
            }
            LengthDistribution::LogNormal { mean, std_dev } => {
                let lognormal = rand_distr::LogNormal::new(*mean, *std_dev).unwrap();
                lognormal.sample(rng).max(1.0) as u32
            }
        }
    }
}

impl WorkloadConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let contents =
            fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        toml::from_str(&contents).map_err(|e| format!("{}: {e}", path.display()).into())
    }
}
