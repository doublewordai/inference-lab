use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SimulationConfig {
    /// Log progress every N iterations
    pub log_interval: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self { log_interval: 100 }
    }
}
