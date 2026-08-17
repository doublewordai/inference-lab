//! Byte transfers over a graph of directed, capacity-limited edges.
//!
//! A transfer moves `bytes` along a path (a list of edges). Every edge's
//! capacity is shared among the transfers crossing it by max-min fairness
//! (progressive filling): the most contended edge fixes its transfers'
//! rate first, the residual capacity is shared among the rest, and so on.
//! Rates are recomputed whenever the set of in-flight transfers changes;
//! between changes each transfer drains at its rate. An optional fixed
//! latency is paid before bytes flow.
//!
//! Event-driven: `next_completion_delay` says when the next transfer
//! finishes under the current rates; `advance(now)` drains everything to
//! `now` and returns what completed. Completions are also queued per owner
//! (a worker, or the hand-off path) so a worker that did not drive the
//! advance can still collect its own.

use std::collections::{HashMap, HashSet};

pub type EdgeId = usize;

/// Below this many bytes a transfer counts as done (float drift).
const DONE_EPSILON_BYTES: f64 = 1e-6;
/// A remainder that would drain in less than this counts as done: an event
/// scheduled that soon lands at the same instant in floating point and
/// would never move it.
const DONE_EPSILON_SECONDS: f64 = 1e-12;

/// Who a transfer belongs to: which completion queue it lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Owner {
    /// A worker's tier promotion (global worker id).
    Worker(usize),
    /// A prefill → decode hand-off.
    Handoff,
}

#[derive(Debug, Clone)]
pub struct Edge {
    /// Template link name this edge instantiates.
    pub name: String,
    /// Bytes/s.
    pub capacity: f64,
    /// Bytes moved over this edge so far.
    pub bytes_moved: f64,
    /// Integral of in-flight transfer count over time (for mean load).
    busy_transfer_seconds: f64,
    last_update: f64,
}

impl Edge {
    fn new(name: String, capacity: f64) -> Self {
        Self {
            name,
            capacity,
            bytes_moved: 0.0,
            busy_transfer_seconds: 0.0,
            last_update: 0.0,
        }
    }

    /// Mean number of transfers in flight on this edge over `[0, until]`.
    pub fn mean_in_flight(&self, until: f64) -> f64 {
        if until > 0.0 {
            self.busy_transfer_seconds / until
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
struct Transfer {
    owner: Owner,
    path: Vec<EdgeId>,
    latency_remaining: f64,
    bytes_remaining: f64,
    /// Current max-min rate, bytes/s (0 while latency is being paid or
    /// when the path has no capacity).
    rate: f64,
    last_update: f64,
}

#[derive(Debug, Default)]
pub struct Flows {
    edges: Vec<Edge>,
    in_flight: HashMap<String, Transfer>,
    /// Completed transfers not yet collected, per owner.
    completed: HashMap<Owner, HashSet<String>>,
    /// Bytes submitted, by owner kind, for reporting.
    pub bytes_submitted_worker: f64,
    pub bytes_submitted_handoff: f64,
    /// Time of the last `advance`; new edges start their accounting here.
    now: f64,
}

impl Flows {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an edge; returns its id.
    pub fn add_edge(&mut self, name: impl Into<String>, capacity: f64) -> EdgeId {
        let mut e = Edge::new(name.into(), capacity);
        e.last_update = self.now;
        self.edges.push(e);
        self.edges.len() - 1
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn num_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.in_flight.contains_key(id)
    }

    pub fn now(&self) -> f64 {
        self.now
    }

    /// Start moving `bytes` for `id` along `path` (edges in order) after
    /// `latency` seconds. Zero bytes complete on the next `advance`. No-op
    /// if `id` is already in flight. Rates are recomputed.
    pub fn submit(
        &mut self,
        id: String,
        owner: Owner,
        path: Vec<EdgeId>,
        bytes: u64,
        latency: f64,
        now: f64,
    ) {
        if self.in_flight.contains_key(&id) {
            return;
        }
        // Bring everything to `now` under the old rates first.
        self.drain_to(now);
        match owner {
            Owner::Worker(_) => self.bytes_submitted_worker += bytes as f64,
            Owner::Handoff => self.bytes_submitted_handoff += bytes as f64,
        }
        self.in_flight.insert(
            id,
            Transfer {
                owner,
                path,
                latency_remaining: latency.max(0.0),
                bytes_remaining: bytes as f64,
                rate: 0.0,
                last_update: now,
            },
        );
        self.recompute_rates();
    }

    /// Advance every transfer to `now`. Returns `(owner, id)` for each
    /// transfer that completed, and queues them per owner too.
    pub fn advance(&mut self, now: f64) -> Vec<(Owner, String)> {
        self.drain_to(now);
        let mut done: Vec<(Owner, String)> = self
            .in_flight
            .iter()
            .filter(|(_, t)| Self::is_done(t))
            .map(|(id, t)| (t.owner, id.clone()))
            .collect();
        done.sort_by(|a, b| a.1.cmp(&b.1));
        for (owner, id) in &done {
            self.in_flight.remove(id);
            self.completed.entry(*owner).or_default().insert(id.clone());
        }
        if !done.is_empty() {
            self.recompute_rates();
        }
        done
    }

    /// Completed transfers of `owner`, drained.
    pub fn take_completed(&mut self, owner: Owner) -> HashSet<String> {
        self.completed.remove(&owner).unwrap_or_default()
    }

    /// Owners with completions waiting to be collected.
    pub fn owners_with_completions(&self) -> Vec<Owner> {
        self.completed
            .iter()
            .filter(|(_, s)| !s.is_empty())
            .map(|(o, _)| *o)
            .collect()
    }

    /// Time until the next completion under the current rates, measured
    /// from the last advance. `None` when nothing in flight can complete.
    pub fn next_completion_delay(&self) -> Option<f64> {
        self.in_flight
            .values()
            .filter_map(Self::time_to_done)
            .min_by(f64::total_cmp)
    }

    /// Projected remaining time for `id` under the current rates; 0 if
    /// unknown.
    pub fn estimate_remaining(&self, id: &str) -> f64 {
        self.in_flight
            .get(id)
            .and_then(Self::time_to_done)
            .unwrap_or(0.0)
    }

    fn time_to_done(t: &Transfer) -> Option<f64> {
        if Self::is_done(t) {
            return Some(0.0);
        }
        if t.bytes_remaining <= DONE_EPSILON_BYTES {
            return Some(t.latency_remaining.max(0.0));
        }
        if t.rate > 0.0 {
            Some(t.latency_remaining.max(0.0) + t.bytes_remaining / t.rate)
        } else {
            None
        }
    }

    /// Whether `t` has finished within float drift: no latency left and
    /// no bytes, or a remainder that would drain in under
    /// `DONE_EPSILON_SECONDS` at its rate.
    fn is_done(t: &Transfer) -> bool {
        if t.latency_remaining > DONE_EPSILON_SECONDS {
            return false;
        }
        t.bytes_remaining <= DONE_EPSILON_BYTES
            || (t.rate > 0.0 && t.bytes_remaining / t.rate <= DONE_EPSILON_SECONDS)
    }

    /// Move every transfer forward to `now` at its current rate.
    fn drain_to(&mut self, now: f64) {
        if now < self.now {
            return;
        }
        for t in self.in_flight.values_mut() {
            let mut dt = now - t.last_update;
            t.last_update = now;
            if dt <= 0.0 {
                continue;
            }
            if t.latency_remaining > 0.0 {
                let l = t.latency_remaining.min(dt);
                t.latency_remaining -= l;
                dt -= l;
            }
            if dt > 0.0 && t.rate > 0.0 && t.bytes_remaining > 0.0 {
                let moved = (t.rate * dt).min(t.bytes_remaining);
                t.bytes_remaining -= moved;
                for &e in &t.path {
                    self.edges[e].bytes_moved += moved;
                }
            }
        }
        let dt = now - self.now;
        if dt > 0.0 {
            // Edge load: transfers holding a share of the edge.
            let mut count = vec![0usize; self.edges.len()];
            for t in self.in_flight.values() {
                for &e in &t.path {
                    count[e] += 1;
                }
            }
            for (e, c) in self.edges.iter_mut().zip(count) {
                e.busy_transfer_seconds += c as f64 * dt;
                e.last_update = now;
            }
        }
        self.now = now;
    }

    /// Max-min fair rates by progressive filling over every in-flight
    /// transfer. A transfer still paying its latency holds its share from
    /// submission (it moves no bytes until the latency is paid, so the
    /// share is reserved rather than used for those microseconds).
    fn recompute_rates(&mut self) {
        let ids: Vec<String> = self.in_flight.keys().cloned().collect();
        let mut residual: Vec<f64> = self.edges.iter().map(|e| e.capacity).collect();
        let mut load: Vec<usize> = vec![0; self.edges.len()];
        let mut unfrozen: HashSet<String> = HashSet::new();
        for id in &ids {
            let t = &self.in_flight[id];
            if t.bytes_remaining <= DONE_EPSILON_BYTES {
                continue;
            }
            if t.path.is_empty() {
                // No edges: unconstrained; treat as instantaneous.
                self.in_flight.get_mut(id).unwrap().rate = f64::INFINITY;
                continue;
            }
            unfrozen.insert(id.clone());
            for &e in &t.path {
                load[e] += 1;
            }
        }
        while !unfrozen.is_empty() {
            // Most contended edge: smallest fair share among edges with load.
            let mut best: Option<(EdgeId, f64)> = None;
            for (e, &n) in load.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let share = residual[e] / n as f64;
                if best.is_none_or(|(_, s)| share < s) {
                    best = Some((e, share));
                }
            }
            let Some((edge, share)) = best else { break };
            // Freeze every unfrozen transfer on that edge at `share`.
            let frozen: Vec<String> = unfrozen
                .iter()
                .filter(|id| self.in_flight[*id].path.contains(&edge))
                .cloned()
                .collect();
            for id in frozen {
                unfrozen.remove(&id);
                let t = self.in_flight.get_mut(&id).unwrap();
                t.rate = share.max(0.0);
                for &e in &t.path {
                    residual[e] = (residual[e] - share).max(0.0);
                    load[e] -= 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn one_transfer_runs_at_the_bottleneck() {
        let mut f = Flows::new();
        let fast = f.add_edge("fast", 100.0);
        let slow = f.add_edge("slow", 10.0);
        f.submit("a".into(), Owner::Handoff, vec![fast, slow], 100, 0.0, 0.0);
        assert!(approx(f.next_completion_delay().unwrap(), 10.0));
        assert!(f.advance(5.0).is_empty());
        assert!(approx(f.estimate_remaining("a"), 5.0));
        let done = f.advance(10.0);
        assert_eq!(done, vec![(Owner::Handoff, "a".to_string())]);
        assert_eq!(f.take_completed(Owner::Handoff).len(), 1);
        assert!(approx(f.edges()[slow].bytes_moved, 100.0));
        assert!(approx(f.edges()[fast].bytes_moved, 100.0));
    }

    #[test]
    fn two_on_one_edge_share_and_a_third_elsewhere_does_not() {
        let mut f = Flows::new();
        let e = f.add_edge("e", 10.0);
        let g = f.add_edge("g", 10.0);
        f.submit("a".into(), Owner::Worker(0), vec![e], 100, 0.0, 0.0);
        f.submit("b".into(), Owner::Worker(1), vec![e], 100, 0.0, 0.0);
        f.submit("c".into(), Owner::Worker(2), vec![g], 100, 0.0, 0.0);
        // a and b at 5 each: 20 s; c alone: 10 s.
        assert!(approx(f.next_completion_delay().unwrap(), 10.0));
        let done = f.advance(10.0);
        assert_eq!(done, vec![(Owner::Worker(2), "c".to_string())]);
        assert!(approx(f.estimate_remaining("a"), 10.0));
        let done = f.advance(20.0);
        assert_eq!(done.len(), 2);
        assert_eq!(
            f.take_completed(Owner::Worker(0)),
            HashSet::from(["a".to_string()])
        );
        assert_eq!(
            f.take_completed(Owner::Worker(1)),
            HashSet::from(["b".to_string()])
        );
    }

    #[test]
    fn max_min_gives_the_uncontended_path_the_leftover() {
        // a: [x, y]; b: [x]. x = 10, y = 4. a is bounded by y at 4, so b
        // gets x's remaining 6, not 5.
        let mut f = Flows::new();
        let x = f.add_edge("x", 10.0);
        let y = f.add_edge("y", 4.0);
        f.submit("a".into(), Owner::Worker(0), vec![x, y], 40, 0.0, 0.0);
        f.submit("b".into(), Owner::Worker(1), vec![x], 60, 0.0, 0.0);
        // a: 40 / 4 = 10 s; b: 60 / 6 = 10 s.
        assert!(approx(f.next_completion_delay().unwrap(), 10.0));
        let done = f.advance(10.0);
        assert_eq!(done.len(), 2);
    }

    #[test]
    fn a_completion_frees_capacity_for_the_rest() {
        let mut f = Flows::new();
        let e = f.add_edge("e", 10.0);
        f.submit("a".into(), Owner::Worker(0), vec![e], 50, 0.0, 0.0);
        f.submit("b".into(), Owner::Worker(0), vec![e], 100, 0.0, 0.0);
        // Both at 5: a done at 10 s (b has 50 left), then b alone at 10:
        // done at 15 s.
        assert!(approx(f.next_completion_delay().unwrap(), 10.0));
        f.advance(10.0);
        assert!(approx(f.next_completion_delay().unwrap(), 5.0));
        let done = f.advance(15.0);
        assert_eq!(done, vec![(Owner::Worker(0), "b".to_string())]);
        assert_eq!(f.take_completed(Owner::Worker(0)).len(), 2);
        // Mean load on e over 15 s: (2 × 10 + 1 × 5) / 15.
        assert!(approx(f.edges()[e].mean_in_flight(15.0), 25.0 / 15.0));
    }

    #[test]
    fn latency_is_paid_before_bytes_and_zero_bytes_complete() {
        let mut f = Flows::new();
        let e = f.add_edge("e", 10.0);
        f.submit("a".into(), Owner::Handoff, vec![e], 100, 2.0, 0.0);
        assert!(approx(f.next_completion_delay().unwrap(), 12.0));
        f.advance(1.0);
        assert!(f.edges()[e].bytes_moved == 0.0);
        f.advance(2.0);
        assert!(approx(f.estimate_remaining("a"), 10.0));
        let done = f.advance(12.0);
        assert_eq!(done.len(), 1);
        f.submit("z".into(), Owner::Handoff, vec![e], 0, 0.0, 12.0);
        assert!(approx(f.next_completion_delay().unwrap(), 0.0));
        assert_eq!(f.advance(12.0).len(), 1);
    }

    #[test]
    fn a_late_joiner_slows_the_others_from_its_arrival() {
        let mut f = Flows::new();
        let e = f.add_edge("e", 10.0);
        f.submit("a".into(), Owner::Worker(0), vec![e], 100, 0.0, 0.0);
        f.advance(5.0); // a: 50 left
        f.submit("b".into(), Owner::Worker(1), vec![e], 100, 0.0, 5.0);
        // a: 50 at 5/s → 10 s more; b: 100 at 5/s → 20 s (then alone).
        assert!(approx(f.next_completion_delay().unwrap(), 10.0));
        f.advance(15.0);
        assert!(approx(f.estimate_remaining("b"), 5.0)); // 50 left at 10/s
    }
}
