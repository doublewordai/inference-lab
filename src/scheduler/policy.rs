use std::str::FromStr;

/// Scheduling policy for request ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served: requests served in arrival order
    FCFS,
    /// Priority-based: requests ordered by priority value (lower = higher priority)
    Priority,

    // Input-based policies (prompt length)
    /// Shortest Input First: prioritize requests with smallest input length
    SIF,
    /// Longest Input First: prioritize requests with largest input length
    LIF,

    // Output-based policies (generation length)
    /// Shortest Output First: prioritize requests with smallest output length
    SOF,
    /// Longest Output First: prioritize requests with largest output length
    LOF,

    // Total-based policies (input + output)
    /// Shortest Total First: prioritize requests with smallest total length
    STF,
    /// Longest Total First: prioritize requests with largest total length
    LTF,
}

impl FromStr for SchedulingPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fcfs" => Ok(SchedulingPolicy::FCFS),
            "priority" => Ok(SchedulingPolicy::Priority),
            "sif" => Ok(SchedulingPolicy::SIF),
            "lif" => Ok(SchedulingPolicy::LIF),
            "sof" => Ok(SchedulingPolicy::SOF),
            "lof" => Ok(SchedulingPolicy::LOF),
            "stf" => Ok(SchedulingPolicy::STF),
            "ltf" => Ok(SchedulingPolicy::LTF),
            // Backward compatibility
            "sjf" => Ok(SchedulingPolicy::SOF),
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
            "fcfs".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::FCFS
        );
        assert_eq!(
            "FCFS".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::FCFS
        );
        assert_eq!(
            "priority".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::Priority
        );
        assert_eq!(
            "sif".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::SIF
        );
        assert_eq!(
            "SIF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::SIF
        );
        assert_eq!(
            "lif".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::LIF
        );
        assert_eq!(
            "LIF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::LIF
        );
        assert_eq!(
            "sof".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::SOF
        );
        assert_eq!(
            "SOF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::SOF
        );
        assert_eq!(
            "lof".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::LOF
        );
        assert_eq!(
            "LOF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::LOF
        );
        assert_eq!(
            "stf".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::STF
        );
        assert_eq!(
            "STF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::STF
        );
        assert_eq!(
            "ltf".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::LTF
        );
        assert_eq!(
            "LTF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::LTF
        );
        // Test backward compatibility
        assert_eq!(
            "sjf".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::SOF
        );
        assert_eq!(
            "SJF".parse::<SchedulingPolicy>().unwrap(),
            SchedulingPolicy::SOF
        );
        assert!("unknown".parse::<SchedulingPolicy>().is_err());
    }
}
