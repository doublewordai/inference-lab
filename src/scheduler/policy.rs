/// Scheduling policy for request ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served: requests served in arrival order
    FCFS,
    /// Priority-based: requests ordered by priority value (lower = higher priority)
    Priority,
    /// Shortest Job First: prioritize requests with smallest output length
    SJF,
    /// Shortest Input First: prioritize requests with smallest input length
    SIF,
    /// Longest Input First: prioritize requests with largest input length
    LIF,
}

impl SchedulingPolicy {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "fcfs" => Ok(SchedulingPolicy::FCFS),
            "priority" => Ok(SchedulingPolicy::Priority),
            "sjf" => Ok(SchedulingPolicy::SJF),
            "sif" => Ok(SchedulingPolicy::SIF),
            "lif" => Ok(SchedulingPolicy::LIF),
            _ => Err(format!("Unknown scheduling policy: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_from_str() {
        assert_eq!(
            SchedulingPolicy::from_str("fcfs").unwrap(),
            SchedulingPolicy::FCFS
        );
        assert_eq!(
            SchedulingPolicy::from_str("FCFS").unwrap(),
            SchedulingPolicy::FCFS
        );
        assert_eq!(
            SchedulingPolicy::from_str("priority").unwrap(),
            SchedulingPolicy::Priority
        );
        assert_eq!(
            SchedulingPolicy::from_str("sjf").unwrap(),
            SchedulingPolicy::SJF
        );
        assert_eq!(
            SchedulingPolicy::from_str("SJF").unwrap(),
            SchedulingPolicy::SJF
        );
        assert_eq!(
            SchedulingPolicy::from_str("sif").unwrap(),
            SchedulingPolicy::SIF
        );
        assert_eq!(
            SchedulingPolicy::from_str("SIF").unwrap(),
            SchedulingPolicy::SIF
        );
        assert_eq!(
            SchedulingPolicy::from_str("lif").unwrap(),
            SchedulingPolicy::LIF
        );
        assert_eq!(
            SchedulingPolicy::from_str("LIF").unwrap(),
            SchedulingPolicy::LIF
        );
        assert!(SchedulingPolicy::from_str("unknown").is_err());
    }
}
