/// Request status in the simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// Request is waiting to be scheduled
    Waiting,
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
            RequestStatus::Running => write!(f, "Running"),
            RequestStatus::Preempted => write!(f, "Preempted"),
            RequestStatus::Completed => write!(f, "Completed"),
        }
    }
}
