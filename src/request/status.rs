/// Request status in the simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// Request is waiting to be scheduled
    Waiting,
    /// Request is holding HBM blocks while its KV cache is being promoted
    /// from a slower tier (e.g. host RAM). Becomes Waiting again once the
    /// transfer completes and the request is ready to run.
    WaitingOnTransfer,
    /// Request is currently running/being processed
    Running,
    /// Request was preempted and is waiting to be resumed
    Preempted,
    /// Request has completed successfully
    Completed,
}

impl std::fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestStatus::Waiting => write!(f, "Waiting"),
            RequestStatus::WaitingOnTransfer => write!(f, "WaitingOnTransfer"),
            RequestStatus::Running => write!(f, "Running"),
            RequestStatus::Preempted => write!(f, "Preempted"),
            RequestStatus::Completed => write!(f, "Completed"),
        }
    }
}
