//! Bandwidth-shared transfer primitive.
//!
//! A `Link` models a fixed-bandwidth resource (HBM-tier promotion bandwidth, an
//! NVLink fabric, an inter-cluster KV handoff path) over which independent
//! byte-counted transfers compete. While `n` transfers are in flight, each
//! progresses at `bandwidth / n`. Submissions are identified by an opaque
//! string id; `advance` moves simulation time forward and returns the ids of
//! transfers that completed during the step.
//!
//! Higher-level features that the KV-cache hierarchy needs — joiners
//! (zero-cost piggybacking on a leader's transfer) and multi-tier composite
//! transfers (one request straddling multiple tiers) — are layered on top of
//! `Link` by the caller, not baked in here.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct LinkTransfer {
    bytes_remaining: u64,
    last_update: f64,
}

#[derive(Debug, Clone)]
pub struct Link {
    bandwidth: f64,
    in_flight: HashMap<String, LinkTransfer>,
}

impl Link {
    pub fn new(bandwidth: f64) -> Self {
        Self {
            bandwidth,
            in_flight: HashMap::new(),
        }
    }

    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    pub fn num_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.in_flight.contains_key(id)
    }

    /// Submit a new transfer. No-op if `id` already has an in-flight transfer.
    pub fn submit(&mut self, id: String, bytes: u64, current_time: f64) {
        self.in_flight.entry(id).or_insert(LinkTransfer {
            bytes_remaining: bytes,
            last_update: current_time,
        });
    }

    /// Advance all in-flight transfers to `current_time`. Bandwidth is divided
    /// equally among the transfers in flight at the start of this step. If a
    /// transfer completes mid-step the freed bandwidth is *not* redistributed
    /// inside the step — others pick it up on the next `advance` call. This
    /// matches the simulator's iteration-discrete time model.
    ///
    /// Returns ids of transfers that finished during this step.
    pub fn advance(&mut self, current_time: f64) -> HashSet<String> {
        let n = self.in_flight.len();
        if n == 0 {
            return HashSet::new();
        }
        let share = if self.bandwidth > 0.0 {
            self.bandwidth / n as f64
        } else {
            0.0
        };
        let mut completed = HashSet::new();
        for (id, state) in self.in_flight.iter_mut() {
            let dt = current_time - state.last_update;
            if dt > 0.0 && share > 0.0 && state.bytes_remaining > 0 {
                let consumed = (share * dt) as u64;
                if consumed >= state.bytes_remaining {
                    state.bytes_remaining = 0;
                } else {
                    state.bytes_remaining -= consumed;
                }
            }
            state.last_update = current_time;
            if state.bytes_remaining == 0 {
                completed.insert(id.clone());
            }
        }
        for id in &completed {
            self.in_flight.remove(id);
        }
        completed
    }

    /// Project remaining time for `id` assuming current contention persists.
    /// Returns 0.0 if `id` is not in flight.
    pub fn estimate_remaining(&self, id: &str) -> f64 {
        let Some(state) = self.in_flight.get(id) else {
            return 0.0;
        };
        let n = self.in_flight.len();
        if n == 0 || self.bandwidth <= 0.0 {
            return 0.0;
        }
        let share = self.bandwidth / n as f64;
        if share <= 0.0 {
            return 0.0;
        }
        state.bytes_remaining as f64 / share
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_transfer_finishes_at_full_bandwidth() {
        let mut link = Link::new(1e9); // 1 GB/s
        link.submit("a".into(), 1_000_000_000, 0.0); // 1 GB
        // Estimate alone: 1 second.
        assert!((link.estimate_remaining("a") - 1.0).abs() < 1e-9);
        let done = link.advance(1.0);
        assert!(done.contains("a"));
        assert_eq!(link.num_in_flight(), 0);
    }

    #[test]
    fn two_concurrent_transfers_share_bandwidth() {
        let mut link = Link::new(1e9);
        link.submit("a".into(), 1_000_000_000, 0.0);
        link.submit("b".into(), 1_000_000_000, 0.0);
        // Each gets 0.5 GB/s, projects 2.0s.
        assert!((link.estimate_remaining("a") - 2.0).abs() < 1e-3);
        assert!((link.estimate_remaining("b") - 2.0).abs() < 1e-3);

        // Halfway: each consumed 0.5 GB; 0.5 GB remaining each, still 2-way share.
        let done = link.advance(1.0);
        assert!(done.is_empty());
        assert!((link.estimate_remaining("a") - 1.0).abs() < 1e-3);

        let done = link.advance(2.0);
        assert!(done.contains("a"));
        assert!(done.contains("b"));
        assert_eq!(link.num_in_flight(), 0);
    }

    #[test]
    fn submitting_existing_id_is_noop() {
        let mut link = Link::new(1e9);
        link.submit("a".into(), 1_000_000_000, 0.0);
        link.submit("a".into(), 999, 0.0); // ignored
        assert_eq!(link.num_in_flight(), 1);
        let done = link.advance(1.0);
        assert!(done.contains("a")); // original 1 GB transfer completes
    }

    #[test]
    fn estimate_remaining_unknown_returns_zero() {
        let link = Link::new(1e9);
        assert_eq!(link.estimate_remaining("nope"), 0.0);
    }

    #[test]
    fn zero_bandwidth_link_makes_no_progress() {
        let mut link = Link::new(0.0);
        link.submit("a".into(), 100, 0.0);
        let done = link.advance(10.0);
        assert!(done.is_empty());
        assert_eq!(link.estimate_remaining("a"), 0.0); // we report 0, not infinity
        assert_eq!(link.num_in_flight(), 1);
    }
}
