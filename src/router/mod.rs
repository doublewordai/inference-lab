//! Routing requests across the replicas of a pool.
//!
//! The engine asks the pool's [`Router`] for a replica index on every
//! arrival (and, on a disaggregated topology, again when a hand-off lands
//! on the decode pool). The router sees one [`WorkerSignal`] per replica:
//! queue depth, queued prefill work, KV occupancy and — when it asks for it
//! — an estimate of how much of this prompt that replica already holds in
//! its KV cache. That estimate is what a KV-aware front end (dynamo's KV
//! router, SGLang's cache-aware router) works from: an approximation of
//! the replica's prefix tree, not the scheduler's exact admission-time
//! lookup.

use crate::config::RouterConfig;
use crate::request::Request;

/// What a router can see about one replica at routing time.
#[derive(Debug, Clone, Default)]
pub struct WorkerSignal {
    /// Requests admitted and holding KV (prefilling or decoding).
    pub running: usize,
    /// Requests queued or parked on a KV promotion.
    pub waiting: usize,
    /// Prompt tokens still to be prefilled across the replica's queue and
    /// its in-progress prefills: the prefill work ahead of a new arrival.
    pub queued_prefill_tokens: u64,
    /// KV cache occupancy, 0..1.
    pub kv_util: f64,
    /// Estimated prompt tokens the replica holds in its KV cache (HBM,
    /// in flight, or a spillover tier). `None` when the router did not ask
    /// (`Router::wants_prefix_signal` returned false).
    pub cached_prefix_tokens: Option<u32>,
}

impl WorkerSignal {
    /// Requests in the system on this replica.
    pub fn in_system(&self) -> usize {
        self.running + self.waiting
    }
}

/// Picks a replica for each request.
pub trait Router: Send {
    /// Whether the engine should fill `WorkerSignal::cached_prefix_tokens`
    /// (a prefix lookup on every replica per arrival). Off by default.
    fn wants_prefix_signal(&self) -> bool {
        false
    }

    /// Replica index for `req`. `workers` has one entry per replica and is
    /// never empty.
    fn route(&mut self, req: &Request, workers: &[WorkerSignal]) -> usize;
}

/// Build the router a config names.
pub fn build_router(cfg: &RouterConfig) -> Box<dyn Router> {
    match cfg {
        RouterConfig::RoundRobin {} => Box::new(RoundRobin::default()),
        RouterConfig::LeastLoaded {} => Box::new(LeastLoaded),
        RouterConfig::PrefixAffinity { max_load_ratio } => Box::new(PrefixAffinity {
            max_load_ratio: *max_load_ratio,
        }),
        RouterConfig::KvAware { load_weight } => Box::new(KvAware {
            load_weight: *load_weight,
        }),
    }
}

/// Cycle through the replicas.
#[derive(Debug, Default)]
pub struct RoundRobin {
    next: usize,
}

impl Router for RoundRobin {
    fn route(&mut self, _req: &Request, workers: &[WorkerSignal]) -> usize {
        let n = workers.len().max(1);
        let idx = self.next % n;
        self.next = (idx + 1) % n;
        idx
    }
}

/// Fewest requests in the system; ties broken by queued prefill tokens,
/// then by index.
#[derive(Debug, Default)]
pub struct LeastLoaded;

fn least_loaded(workers: &[WorkerSignal]) -> usize {
    let mut best = 0usize;
    for (i, w) in workers.iter().enumerate().skip(1) {
        let b = &workers[best];
        if (w.in_system(), w.queued_prefill_tokens) < (b.in_system(), b.queued_prefill_tokens) {
            best = i;
        }
    }
    best
}

impl Router for LeastLoaded {
    fn route(&mut self, _req: &Request, workers: &[WorkerSignal]) -> usize {
        least_loaded(workers)
    }
}

/// Send the request to the replica holding the longest cached prefix. With
/// no cached prefix anywhere it behaves as [`LeastLoaded`]. With
/// `max_load_ratio = Some(r)`, a holder whose request count exceeds
/// `r × mean` is passed over for the least-loaded replica (bounded-load
/// affinity: `r = 1` keeps every replica at or under the mean).
#[derive(Debug)]
pub struct PrefixAffinity {
    pub max_load_ratio: Option<f64>,
}

impl Router for PrefixAffinity {
    fn wants_prefix_signal(&self) -> bool {
        true
    }

    fn route(&mut self, _req: &Request, workers: &[WorkerSignal]) -> usize {
        let mut best: Option<usize> = None;
        for (i, w) in workers.iter().enumerate() {
            let cached = w.cached_prefix_tokens.unwrap_or(0);
            if cached == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => {
                    let bw = &workers[b];
                    (cached, std::cmp::Reverse(w.in_system()))
                        > (
                            bw.cached_prefix_tokens.unwrap_or(0),
                            std::cmp::Reverse(bw.in_system()),
                        )
                }
            };
            if better {
                best = Some(i);
            }
        }
        let Some(holder) = best else {
            return least_loaded(workers);
        };
        if let Some(ratio) = self.max_load_ratio {
            let n = workers.len() as f64;
            let mean = workers.iter().map(|w| w.in_system() as f64).sum::<f64>() / n;
            // A holder is over its bound when it exceeds `ratio × mean` and
            // is not already the emptiest replica.
            let cap = ratio * mean;
            let load = workers[holder].in_system() as f64;
            if load > cap && holder != least_loaded(workers) {
                return least_loaded(workers);
            }
        }
        holder
    }
}

/// Minimise the estimated prefill work on the request's path: the tokens
/// it will add (prompt minus the replica's cached prefix) plus
/// `load_weight` × the prefill tokens already queued there. With
/// `load_weight = 1` the score is the replica's prefill backlog once this
/// request is on it, in tokens.
#[derive(Debug)]
pub struct KvAware {
    pub load_weight: f64,
}

impl Router for KvAware {
    fn wants_prefix_signal(&self) -> bool {
        true
    }

    fn route(&mut self, req: &Request, workers: &[WorkerSignal]) -> usize {
        let mut best = 0usize;
        let mut best_cost = f64::INFINITY;
        for (i, w) in workers.iter().enumerate() {
            let cached = w
                .cached_prefix_tokens
                .unwrap_or(0)
                .min(req.num_prompt_tokens);
            let novel = (req.num_prompt_tokens - cached) as f64;
            let cost = novel + self.load_weight * w.queued_prefill_tokens as f64;
            if cost < best_cost {
                best_cost = cost;
                best = i;
            }
        }
        best
    }
}

/// What the router did, per pool.
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    /// Requests routed to each replica.
    pub per_worker: Vec<u64>,
    /// Requests routed to a replica the router estimated held a nonzero
    /// prefix of the prompt. Only counted by routers that ask for the prefix
    /// signal.
    pub prefix_routed: u64,
    /// Requests for which some replica held a nonzero prefix (routed there
    /// or not). Only counted by routers that ask for the prefix signal.
    pub prefix_available: u64,
    /// Requests routed to a replica other than the one holding the longest
    /// cached prefix while some replica held one.
    pub prefix_forgone: u64,
}

impl RouterStats {
    pub fn new(num_workers: usize) -> Self {
        Self {
            per_worker: vec![0; num_workers],
            ..Default::default()
        }
    }

    /// Record one decision. `signals` are the per-replica signals the router
    /// saw; `chosen` its answer.
    pub fn record(&mut self, signals: &[WorkerSignal], chosen: usize) {
        if let Some(c) = self.per_worker.get_mut(chosen) {
            *c += 1;
        }
        let best = signals
            .iter()
            .map(|s| s.cached_prefix_tokens.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if best == 0 {
            return;
        }
        self.prefix_available += 1;
        if signals[chosen].cached_prefix_tokens.unwrap_or(0) > 0 {
            self.prefix_routed += 1;
        }
        if signals[chosen].cached_prefix_tokens.unwrap_or(0) < best {
            self.prefix_forgone += 1;
        }
    }

    pub fn total(&self) -> u64 {
        self.per_worker.iter().sum()
    }
}

impl std::ops::AddAssign<&RouterStats> for RouterStats {
    fn add_assign(&mut self, rhs: &RouterStats) {
        if self.per_worker.len() < rhs.per_worker.len() {
            self.per_worker.resize(rhs.per_worker.len(), 0);
        }
        for (a, b) in self.per_worker.iter_mut().zip(&rhs.per_worker) {
            *a += b;
        }
        self.prefix_routed += rhs.prefix_routed;
        self.prefix_available += rhs.prefix_available;
        self.prefix_forgone += rhs.prefix_forgone;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: u32) -> Request {
        Request::new("r".into(), 0, 0.0, prompt, 8)
    }

    fn sig(running: usize, waiting: usize, queued: u64, cached: Option<u32>) -> WorkerSignal {
        WorkerSignal {
            running,
            waiting,
            queued_prefill_tokens: queued,
            kv_util: 0.0,
            cached_prefix_tokens: cached,
        }
    }

    #[test]
    fn round_robin_cycles() {
        let mut r = RoundRobin::default();
        let ws = vec![sig(0, 0, 0, None); 3];
        let picks: Vec<usize> = (0..5).map(|_| r.route(&req(10), &ws)).collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn least_loaded_counts_requests_then_tokens() {
        let mut r = LeastLoaded;
        let ws = vec![
            sig(3, 1, 100, None),
            sig(2, 1, 900, None),
            sig(2, 1, 50, None),
        ];
        assert_eq!(r.route(&req(10), &ws), 2);
    }

    #[test]
    fn affinity_prefers_the_holder_and_falls_back_to_load() {
        let mut r = PrefixAffinity {
            max_load_ratio: None,
        };
        let ws = vec![
            sig(9, 9, 0, Some(0)),
            sig(0, 0, 0, Some(64)),
            sig(0, 0, 0, Some(512)),
        ];
        assert_eq!(r.route(&req(1024), &ws), 2);
        // Nothing cached anywhere → least loaded.
        let ws = vec![
            sig(2, 0, 0, Some(0)),
            sig(1, 0, 0, Some(0)),
            sig(3, 0, 0, Some(0)),
        ];
        assert_eq!(r.route(&req(1024), &ws), 1);
    }

    #[test]
    fn bounded_load_affinity_passes_over_a_hot_holder() {
        let mut r = PrefixAffinity {
            max_load_ratio: Some(1.5),
        };
        // Holder has 10 in system, mean is 4 → over 1.5 × 4 = 6 → least loaded.
        let ws = vec![
            sig(10, 0, 0, Some(512)),
            sig(1, 0, 0, Some(0)),
            sig(1, 0, 0, Some(0)),
        ];
        assert_eq!(r.route(&req(1024), &ws), 1);
        // Holder at 5 in system, mean 7/3 ≈ 2.33 → cap 3.5 → over → least loaded (index 1).
        let ws = vec![
            sig(5, 0, 0, Some(512)),
            sig(1, 0, 0, Some(0)),
            sig(1, 0, 0, Some(0)),
        ];
        assert_eq!(r.route(&req(1024), &ws), 1);
        // Holder at 3, mean 5/3 → cap 2.5 → over. Holder at 2, cap 2 → within.
        let ws = vec![
            sig(2, 0, 0, Some(512)),
            sig(1, 0, 0, Some(0)),
            sig(1, 0, 0, Some(0)),
        ];
        assert_eq!(r.route(&req(1024), &ws), 0);
        // Everyone loaded equally: holder is within bound.
        let ws = vec![
            sig(4, 0, 0, Some(512)),
            sig(4, 0, 0, Some(0)),
            sig(4, 0, 0, Some(0)),
        ];
        assert_eq!(r.route(&req(1024), &ws), 0);
    }

    #[test]
    fn kv_aware_trades_prefix_against_backlog() {
        let mut r = KvAware { load_weight: 1.0 };
        // Replica 0 holds 900 of 1000 tokens but has 2000 queued: cost 100 + 2000.
        // Replica 1 holds nothing, 0 queued: cost 1000. → 1.
        let ws = vec![sig(0, 0, 2000, Some(900)), sig(0, 0, 0, Some(0))];
        assert_eq!(r.route(&req(1000), &ws), 1);
        // With less backlog the holder wins: 100 + 500 < 1000.
        let ws = vec![sig(0, 0, 500, Some(900)), sig(0, 0, 0, Some(0))];
        assert_eq!(r.route(&req(1000), &ws), 0);
        // load_weight = 0 ignores backlog entirely.
        let mut r = KvAware { load_weight: 0.0 };
        let ws = vec![sig(0, 0, 1_000_000, Some(900)), sig(0, 0, 0, Some(0))];
        assert_eq!(r.route(&req(1000), &ws), 0);
    }

    #[test]
    fn stats_count_prefix_outcomes() {
        let mut s = RouterStats::new(2);
        s.record(&[sig(0, 0, 0, Some(64)), sig(0, 0, 0, Some(0))], 0);
        s.record(&[sig(0, 0, 0, Some(64)), sig(0, 0, 0, Some(0))], 1);
        s.record(&[sig(0, 0, 0, Some(0)), sig(0, 0, 0, Some(0))], 1);
        s.record(&[sig(0, 0, 0, Some(64)), sig(0, 0, 0, Some(16))], 1);
        assert_eq!(s.per_worker, vec![1, 3]);
        assert_eq!(s.prefix_available, 3);
        assert_eq!(s.prefix_routed, 2);
        assert_eq!(s.prefix_forgone, 2);
        assert_eq!(s.total(), 4);
    }
}
