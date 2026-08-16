//! Bandwidth-shared transfer primitive.
//!
//! A `Link` models a fixed-bandwidth resource (HBM-tier promotion bandwidth,
//! an inter-pool KV hand-off path) over which independent byte-counted
//! transfers compete. While `n` transfers are in flight, each progresses at
//! `bandwidth / n` (processor sharing). Submissions are identified by an
//! opaque string id; `advance` moves simulation time forward and returns the
//! ids of transfers that completed during the step.
//!
//! The share is fixed between `advance` calls, so a caller that wants exact
//! processor sharing advances the link at every event that changes the
//! in-flight set (each submit, each completion); `next_completion_delay`
//! tells it when the next completion is due under the current share. A
//! caller that only polls (the KV tier hierarchy, advanced once per
//! scheduler pass) gets the same model at iteration granularity.
//!
//! Higher-level features that the KV-cache hierarchy needs — joiners
//! (zero-cost piggybacking on a leader's transfer) and multi-tier composite
//! transfers (one request straddling multiple tiers) — are layered on top of
//! `Link` by the caller, not baked in here.

use std::collections::{HashMap, HashSet};

/// Remaining bytes at or below which a transfer counts as complete (absorbs
/// floating-point residue when a step lands exactly on a completion).
const DONE_EPSILON_BYTES: f64 = 1e-3;

#[derive(Debug, Clone)]
struct LinkTransfer {
    bytes_remaining: f64,
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
            bytes_remaining: bytes as f64,
            last_update: current_time,
        });
    }

    /// Per-transfer bandwidth under the current contention.
    fn share(&self) -> f64 {
        let n = self.in_flight.len();
        if n == 0 || self.bandwidth <= 0.0 {
            0.0
        } else {
            self.bandwidth / n as f64
        }
    }

    /// Advance all in-flight transfers to `current_time`. Bandwidth is divided
    /// equally among the transfers in flight at the start of this step. If a
    /// transfer completes mid-step the freed bandwidth is *not* redistributed
    /// inside the step — others pick it up on the next `advance` call.
    ///
    /// Returns ids of transfers that finished during this step.
    pub fn advance(&mut self, current_time: f64) -> HashSet<String> {
        let share = self.share();
        let mut completed = HashSet::new();
        for (id, state) in self.in_flight.iter_mut() {
            let dt = current_time - state.last_update;
            if dt > 0.0 && share > 0.0 {
                state.bytes_remaining -= share * dt;
            }
            state.last_update = current_time;
            if state.bytes_remaining <= DONE_EPSILON_BYTES {
                completed.insert(id.clone());
            }
        }
        for id in &completed {
            self.in_flight.remove(id);
        }
        completed
    }

    /// Project remaining time for `id` assuming current contention persists.
    /// Returns 0.0 if `id` is not in flight or the link has no bandwidth.
    pub fn estimate_remaining(&self, id: &str) -> f64 {
        let Some(state) = self.in_flight.get(id) else {
            return 0.0;
        };
        let share = self.share();
        if share <= 0.0 {
            return 0.0;
        }
        (state.bytes_remaining / share).max(0.0)
    }

    /// Time until the next in-flight transfer completes under the current
    /// contention, measured from the last `advance`. `None` when nothing is
    /// in flight or the link has no bandwidth.
    pub fn next_completion_delay(&self) -> Option<f64> {
        let share = self.share();
        if share <= 0.0 {
            return None;
        }
        self.in_flight
            .values()
            .map(|t| (t.bytes_remaining / share).max(0.0))
            .min_by(f64::total_cmp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_transfer_finishes_at_full_bandwidth() {
        let mut link = Link::new(1e9); // 1 GB/s
        link.submit("a".into(), 1_000_000_000, 0.0); // 1 GB
        assert!((link.estimate_remaining("a") - 1.0).abs() < 1e-9);
        assert!((link.next_completion_delay().unwrap() - 1.0).abs() < 1e-9);
        let done = link.advance(1.0);
        assert!(done.contains("a"));
        assert_eq!(link.num_in_flight(), 0);
        assert!(link.next_completion_delay().is_none());
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
    fn late_joiner_slows_the_first_and_frees_bandwidth_when_done() {
        // Event-driven use: advance at every submit and completion.
        let mut link = Link::new(1e9);
        link.submit("a".into(), 1_000_000_000, 0.0);
        // At t=0.5, a is half done; b (0.25 GB) joins.
        assert!(link.advance(0.5).is_empty());
        link.submit("b".into(), 250_000_000, 0.5);
        // Shared: b needs 0.5s at 0.5 GB/s; a needs 1.0s. Next completion b at 1.0.
        assert!((link.next_completion_delay().unwrap() - 0.5).abs() < 1e-9);
        let done = link.advance(1.0);
        assert_eq!(done.len(), 1);
        assert!(done.contains("b"));
        // a has 0.25 GB left and the full link again: 0.25s.
        assert!((link.next_completion_delay().unwrap() - 0.25).abs() < 1e-9);
        assert!(link.advance(1.25).contains("a"));
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
        assert!(link.next_completion_delay().is_none());
        assert_eq!(link.num_in_flight(), 1);
    }
}
