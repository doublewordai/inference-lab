use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct WorkloadConfig {
    /// Arrival pattern: "poisson", "uniform", "burst", "fixed_rate", "closed_loop", "batched"
    pub arrival_pattern: String,

    /// Mean arrival rate (requests per second)
    pub arrival_rate: f64,

    /// Input sequence length distribution
    pub input_len_dist: LengthDistribution,

    /// Output sequence length distribution
    pub output_len_dist: LengthDistribution,

    /// Total number of requests to simulate (None = run until duration)
    pub num_requests: Option<usize>,

    /// Number of concurrent users for closed-loop pattern
    /// Each user immediately sends a new request when their previous one completes
    #[serde(default)]
    pub num_concurrent_users: Option<usize>,

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

impl LengthDistribution {
    /// Sample a value from this distribution
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> u32 {
        use rand_distr::Distribution;

        match self {
            LengthDistribution::Fixed { value } => *value,
            LengthDistribution::Uniform { min, max } => {
                rng.gen_range(*min..=*max)
            }
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
