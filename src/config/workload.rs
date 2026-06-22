use serde::Deserialize;

fn default_arrival_rate() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkloadConfig {
    /// Path to dataset file (JSONL in OpenAI batch API format)
    /// If provided, dataset mode is used instead of synthetic workload
    #[serde(default)]
    pub dataset_path: Option<String>,

    /// Arrival pattern: "poisson", "uniform", "burst", "fixed_rate", "closed_loop", "batched"
    pub arrival_pattern: String,

    /// Mean arrival rate (requests per second)
    /// Not used for "closed_loop" or "batched" patterns
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

    /// Total number of requests to simulate (None = run until duration)
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

    /// Simulation duration in seconds (None = run until num_requests)
    pub duration_secs: Option<f64>,

    /// Random seed for reproducibility
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
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
#[serde(tag = "type")]
pub enum RateSchedule {
    /// Sinusoid between `min` and `max`, starting in the trough at t=0 and
    /// peaking at half-period.
    #[serde(rename = "sine")]
    Sine { min: f64, max: f64, period_secs: f64 },
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
        use rand_distr::Distribution;

        match self {
            LengthDistribution::Fixed { value } => *value,
            LengthDistribution::Uniform { min, max } => rng.gen_range(*min..=*max),
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
