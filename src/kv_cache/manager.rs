use super::flows::Owner;
use super::graph::{promotion_request, MemoryGraph, SharedMemoryGraph, WorkerId};
use super::radix::{HbmLookup, KvBytesFn, Path, Radix, SharedRadix, Span};
use crate::config::HbmEviction;
use crate::request::{Outlook, Request};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Result of looking up a request's prompt against the cache hierarchy.
/// Each contiguous block of the prompt is classified as resident in HBM,
/// currently in flight (some other request is already transferring it from
/// a slower tier), resident in spillover (needs to be transferred), or cold
/// (the prefix ends here).
#[derive(Debug, Clone, Default)]
pub struct PrefixCacheLookup {
    /// Total contiguous prefix tokens cached or in-flight across all tiers.
    pub total_cached_tokens: u32,
    /// Tokens already resident in HBM; no transfer required.
    pub hbm_tokens: u32,
    /// Tokens whose blocks are currently being transferred from a slower
    /// tier on behalf of another request. The current request can join
    /// that transfer at zero additional bandwidth cost.
    pub in_flight_tokens: u32,
    /// Per-spillover-tier tokens that need to be promoted. Indexed aligned
    /// with the worker's tiers.
    pub promote_tokens_per_tier: Vec<u32>,
    /// Per-spillover-tier bytes that need to be promoted (the KV footprint of
    /// the promoted token ranges, from the model's KV curve).
    pub promote_bytes_per_tier: Vec<u64>,
    /// Identity of the leader transfer (if any) covering some portion of
    /// the in-flight region. The scheduler uses this to call `join_transfer`
    /// against the same leader rather than starting a redundant one.
    pub join_leader: Option<String>,
    /// The tier-held spans to promote, as `(tier, span)`, in prefix order.
    pub(crate) tier_spans: Vec<(usize, Span)>,
}

impl PrefixCacheLookup {
    pub fn needs_promotion(&self) -> bool {
        self.promote_tokens_per_tier.iter().any(|&t| t > 0)
    }

    pub fn needs_join(&self) -> bool {
        self.in_flight_tokens > 0
    }
}

/// Prefix-cache lookup counters. A lookup is a hit when any prefix tokens were
/// cached (in HBM, in flight, or in a spillover tier).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Sum of cached prefix lengths (tokens) over all lookups.
    pub hit_size_sum: u64,
    /// Number of lookups (hits + misses).
    pub lookups: u64,
    /// Lookups whose tier-held prefix was recomputed instead of fetched
    /// (`source = min_time`), and the tokens that recomputed.
    pub recomputed: u64,
    pub recomputed_tokens: u64,
    /// Prefetches started ahead of an announced re-entry, and the tokens
    /// they pulled up.
    pub prefetches: u64,
    pub prefetch_tokens: u64,
}

impl PrefixCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }

    /// Mean cached prefix length per lookup, in tokens.
    pub fn mean_hit_size(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hit_size_sum as f64 / self.lookups as f64
        }
    }
}

impl std::ops::AddAssign for PrefixCacheStats {
    fn add_assign(&mut self, o: Self) {
        self.recomputed += o.recomputed;
        self.recomputed_tokens += o.recomputed_tokens;
        self.prefetches += o.prefetches;
        self.prefetch_tokens += o.prefetch_tokens;
        self.hits += o.hits;
        self.misses += o.misses;
        self.hit_size_sum += o.hit_size_sum;
        self.lookups += o.lookups;
    }
}

/// Manages KV cache blocks for one worker.
///
/// Capacity and occupancy are counted in blocks. A block is the KV footprint
/// of `block_size` tokens (`kv_bytes_at(block_size)`), and a sequence of `t`
/// tokens occupies `ceil(kv_bytes_at(t) / bytes_per_block)` content blocks.
/// For models whose KV is linear in position this is exactly
/// `ceil(t / block_size)`; for models whose footprint is not (sliding window,
/// DeepSeek-V4's window + compressed history) it charges the model's actual
/// bytes. Sequences additionally hold `state_blocks` fixed blocks for
/// length-independent per-sequence state (Mamba / GatedDeltaNet).
///
/// The blocks themselves live in the topology's [`Radix`] tree, shared with
/// the memory graph: this manager is one worker's view of it. A request's
/// KV is a path through the tree; what it holds is a count (`kv_blocks`),
/// and every operation works on ranges of the path rather than on blocks.
pub struct KVCacheManager {
    /// Block size in tokens.
    block_size: u32,

    /// KV bytes of a `t`-token sequence, from the model.
    kv_bytes_at: KvBytesFn,

    /// `kv_bytes_at(block_size)`: the unit of allocation.
    bytes_per_block: u64,

    /// Total number of blocks available.
    total_blocks: u32,

    /// The topology's KV tree and this worker's id in it.
    radix: SharedRadix,
    worker: WorkerId,

    /// Enable prefix caching.
    enable_prefix_caching: bool,

    /// The pool's KV memory beyond HBM and this worker's id in it, when the
    /// deployment has tiers. Lookups, demotion and promotion transfers go
    /// through it; `None` = HBM only.
    memory: Option<(SharedMemoryGraph, WorkerId)>,

    /// For each leader currently being promoted, the count of tiers that
    /// still have bytes in flight for it. A leader is fully done when this
    /// hits zero across every tier.
    leader_active_tiers: HashMap<String, u32>,

    /// Joiners piggybacking on each leader, keyed by leader id. Joiners
    /// contribute no bandwidth load and become ready when the leader does.
    /// Modelled after vLLM's block-ref-count sharing of in-flight prefixes.
    leader_joiners: HashMap<String, Vec<String>>,

    /// Reverse index for joiners: maps a joiner request id to the leader
    /// it's piggybacking on. Used by `estimate_remaining_time` so a joiner's
    /// projected ready time stays in sync with its leader's.
    joiner_to_leader: HashMap<String, String>,

    /// Fixed blocks reserved per running sequence for length-independent state
    /// (Mamba/GatedDeltaNet recurrent state). Zero for pure-attention models.
    state_blocks: usize,

    /// Prefix-cache lookup statistics, recorded via `record_prefix_lookup`.
    stats: PrefixCacheStats,

    /// Fresh block allocations so far, in bytes: KV written into HBM.
    bytes_written: u64,
    /// `bytes_written` plus every hit that pulled a free block back into
    /// use: the content that moved to the recently-used end of the free
    /// order, so an upper bound on the LRU stack distance in bytes.
    bytes_touched: u64,

    hbm_eviction: HbmEviction,
    backed_first: bool,
}

impl KVCacheManager {
    /// Create a manager over `kv_cache_capacity` bytes of HBM with no
    /// spillover hierarchy. `kv_bytes_at(t)` is the model's KV footprint of a
    /// `t`-token sequence; `per_seq_state_bytes` is the model's fixed
    /// per-sequence state.
    pub fn new(
        kv_cache_capacity: u64,
        block_size: u32,
        kv_bytes_at: impl Fn(u32) -> u64 + Send + Sync + 'static,
        per_seq_state_bytes: u64,
        enable_prefix_caching: bool,
    ) -> Self {
        let kv_bytes_at: KvBytesFn = Arc::new(kv_bytes_at);
        let bytes_per_block = kv_bytes_at(block_size);
        let total_blocks = kv_cache_capacity.checked_div(bytes_per_block).unwrap_or(0) as u32;
        // Fixed per-sequence state (Mamba/GDN) padded up to a whole number of
        // blocks, vLLM-style. Held for the sequence's lifetime.
        let state_blocks = if bytes_per_block == 0 {
            0
        } else {
            per_seq_state_bytes.div_ceil(bytes_per_block) as usize
        };
        let mut radix = Radix::new(block_size, kv_bytes_at.clone());
        radix.register_worker(0, total_blocks, HbmEviction::Lru {}, false);
        Self {
            block_size,
            kv_bytes_at,
            bytes_per_block,
            total_blocks,
            radix: Arc::new(Mutex::new(radix)),
            worker: 0,
            enable_prefix_caching,
            memory: None,
            leader_active_tiers: HashMap::new(),
            leader_joiners: HashMap::new(),
            joiner_to_leader: HashMap::new(),
            state_blocks,
            stats: PrefixCacheStats::default(),
            bytes_written: 0,
            bytes_touched: 0,
            hbm_eviction: HbmEviction::Lru {},
            backed_first: false,
        }
    }

    /// Set the HBM eviction policy. Only meaningful before any allocation.
    pub fn with_hbm_eviction(mut self, policy: HbmEviction) -> Self {
        self.hbm_eviction = policy;
        self.radix
            .lock()
            .unwrap()
            .set_worker_hbm_policy(self.worker, policy, self.backed_first);
        self
    }

    /// Attach the pool's KV memory graph; this manager is `worker` in it.
    /// The manager's blocks move into the graph's tree (call before any
    /// allocation).
    pub fn with_memory(mut self, graph: SharedMemoryGraph, worker: WorkerId) -> Self {
        let (radix, backed_first) = {
            let g = graph.lock().unwrap();
            (g.radix(), g.evict_backed_first(worker))
        };
        self.backed_first = backed_first;
        {
            let mut r = radix.lock().unwrap();
            r.register_worker(worker, self.total_blocks, self.hbm_eviction, backed_first);
            let tiers = graph.lock().unwrap().store_ids_of(worker);
            r.set_worker_tiers(worker, tiers);
        }
        self.radix = radix;
        self.worker = worker;
        self.memory = Some((graph, worker));
        self
    }

    /// Attach a private hierarchy for this worker alone: one tier per
    /// `(name, capacity_bytes, bandwidth_to_hbm)`, closest first. What a
    /// hardware `[memory]` of `per = "gpu"` stores gives a single worker.
    pub fn with_private_tiers(self, tiers: &[(&str, u64, f64)]) -> Self {
        let bpb = self.bytes_per_block.max(1);
        let spec: Vec<(&str, u64, f64)> = tiers
            .iter()
            .map(|&(name, bytes, bw)| (name, bytes / bpb, bw))
            .collect();
        let graph = MemoryGraph::private_with(1, &spec, self.block_size, self.kv_bytes_at.clone())
            .shared_handle();
        self.with_memory(graph, 0)
    }

    /// The pool's memory graph and this worker's id in it, if tiered.
    pub fn memory(&self) -> Option<(&SharedMemoryGraph, WorkerId)> {
        self.memory.as_ref().map(|(g, w)| (g, *w))
    }

    /// The topology's KV tree.
    pub fn radix(&self) -> &SharedRadix {
        &self.radix
    }

    /// Number of tiers below this worker's HBM.
    pub fn num_tiers(&self) -> usize {
        self.radix.lock().unwrap().worker_tiers(self.worker).len()
    }

    /// Fixed blocks each running sequence holds for per-sequence state.
    pub fn state_blocks(&self) -> usize {
        self.state_blocks
    }

    /// Block size in tokens.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Bytes one block holds (`kv_bytes_at(block_size)`).
    pub fn bytes_per_block(&self) -> u64 {
        self.bytes_per_block
    }

    /// KV bytes of a `tokens`-token sequence, per the model's curve.
    pub fn kv_bytes_for_tokens(&self, tokens: u32) -> u64 {
        (self.kv_bytes_at)(tokens)
    }

    /// Content blocks a sequence of `tokens` tokens occupies (state blocks
    /// excluded).
    pub fn content_blocks_for_tokens(&self, tokens: u32) -> usize {
        if self.bytes_per_block == 0 {
            return 0;
        }
        (self.kv_bytes_at)(tokens).div_ceil(self.bytes_per_block) as usize
    }

    /// Blocks a sequence of `total_tokens` tokens needs: its fixed state
    /// blocks plus the content blocks its bytes fill.
    pub fn blocks_for_context(&self, total_tokens: u32) -> usize {
        self.state_blocks + self.content_blocks_for_tokens(total_tokens)
    }

    /// New blocks `request` needs to grow its context by `num_new_tokens`
    /// tokens beyond `num_computed_tokens`, given what it already holds.
    pub fn blocks_needed(&self, request: &Request, num_new_tokens: u32) -> usize {
        let total_tokens = request.num_computed_tokens + num_new_tokens;
        self.blocks_for_context(total_tokens)
            .saturating_sub(request.kv_blocks.len())
    }

    /// The path of `request`'s hashed content, and how the request's held
    /// blocks split into (state, in-path content, anonymous tail).
    fn hold_of(&self, request: &Request) -> (u32, u32, u32) {
        let held = request.kv_blocks.len() as u32;
        if held == 0 {
            return (0, 0, 0);
        }
        let state = (self.state_blocks as u32).min(held);
        let content = held - state;
        let in_path = if self.enable_prefix_caching {
            content.min(request.prompt_block_hashes.len() as u32)
        } else {
            0
        };
        (state, in_path, content - in_path)
    }

    /// Allocate the blocks `request` needs to grow by `num_tokens` positions.
    /// `None` if there are not enough free blocks. Returns the blocks added.
    pub fn allocate_blocks(&mut self, request: &Request, num_tokens: u32) -> Option<u32> {
        self.allocate_inner(request, num_tokens, /*publish_to_hbm=*/ true, None)
    }

    /// Allocate blocks without publishing the request's hashes to the HBM
    /// prefix cache. Used to reserve HBM landing space for an in-flight
    /// promotion: until the transfer completes the data isn't really in HBM,
    /// so other requests looking up the same prefix should not hit HBM.
    pub fn reserve_blocks_for_transfer(
        &mut self,
        request: &Request,
        num_tokens: u32,
    ) -> Option<u32> {
        self.allocate_inner(
            request,
            num_tokens,
            /*publish_to_hbm=*/ false,
            Some(&request.request_id),
        )
    }

    fn allocate_inner(
        &mut self,
        request: &Request,
        num_tokens: u32,
        publish: bool,
        leader: Option<&str>,
    ) -> Option<u32> {
        let target_total = self.blocks_for_context(request.num_computed_tokens + num_tokens) as u32;
        let held_total = request.kv_blocks.len() as u32;
        if target_total <= held_total {
            return Some(0);
        }
        let (state_held, in_path_held, anon_held) = self.hold_of(request);
        let state_new = (self.state_blocks as u32).saturating_sub(state_held);
        let content_target = target_total - self.state_blocks as u32;
        let hashed = if self.enable_prefix_caching {
            request.prompt_block_hashes.len() as u32
        } else {
            0
        };
        let in_path_target = content_target.min(hashed);
        let anon_target = content_target - in_path_target;
        let anon_new = anon_target.saturating_sub(anon_held);

        let mut r = self.radix.lock().unwrap();
        let path = if in_path_target > 0 {
            r.insert(&request.prompt_block_hashes[..in_path_target as usize])
        } else {
            Path::default()
        };
        let got = r.acquire(
            self.worker,
            &path,
            in_path_held,
            in_path_target,
            anon_new + state_new,
            publish,
            leader,
            false,
        )?;
        drop(r);
        self.bytes_written += got.fresh_blocks as u64 * self.bytes_per_block;
        self.bytes_touched += (got.fresh_blocks + got.hits_on_free) as u64 * self.bytes_per_block;
        if let Some((g, w)) = &self.memory {
            let mut g = g.lock().unwrap();
            g.demote_batch(*w, &got.evicted);
            g.hit_batch(*w, &got.hits);
            if publish {
                g.produced_batch(*w, &got.produced);
            }
        }
        if !got.evicted.is_empty() {
            let mut r = self.radix.lock().unwrap();
            for e in &got.evicted {
                r.prune_if_empty(e.span.node);
            }
        }
        Some(target_total - held_total)
    }

    /// A completed transfer's blocks (the first `cached_blocks` content
    /// blocks of `request`) landed in HBM: subsequent lookups see them as
    /// resident.
    pub fn publish_transferred_blocks(&mut self, request: &Request, cached_blocks: usize) {
        if !self.enable_prefix_caching || cached_blocks == 0 {
            return;
        }
        let n = cached_blocks.min(request.prompt_block_hashes.len());
        let spans = {
            let mut r = self.radix.lock().unwrap();
            let path = r.resolve(&request.prompt_block_hashes[..n]);
            r.landed(self.worker, &path, n as u32);
            path_spans(&path, n as u32)
        };
        if let Some((g, w)) = &self.memory {
            g.lock().unwrap().promoted_batch(*w, &spans);
        }
    }

    /// Free `request`'s blocks (due to preemption or completion). What the
    /// request held is queued for recycling tail first, so the end of a
    /// sequence is evicted before its beginning and what survives is always
    /// a prefix.
    pub fn free_blocks(&mut self, request: &Request) {
        let (state, in_path, anon) = self.hold_of(request);
        if state + in_path + anon == 0 {
            return;
        }
        let mut r = self.radix.lock().unwrap();
        let path = if in_path > 0 {
            r.resolve(&request.prompt_block_hashes[..in_path as usize])
        } else {
            Path::default()
        };
        r.release(self.worker, &path, in_path, anon + state);
    }

    /// KV bytes written into HBM so far (every fresh block allocation).
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Bytes written plus bytes of hits that pulled free blocks back into use.
    pub fn bytes_touched(&self) -> u64 {
        self.bytes_touched
    }

    pub fn num_free_blocks(&self) -> usize {
        self.radix.lock().unwrap().free_blocks(self.worker) as usize
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks as usize
    }

    /// Fraction of blocks in use (referenced or reserved).
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.num_free_blocks() as f64 / self.total_blocks as f64)
    }

    /// Look up `request`'s prompt: what is in HBM, in flight, in a tier.
    pub fn peek_prefix_cache(&self, request: &Request) -> PrefixCacheLookup {
        if !self.enable_prefix_caching {
            return PrefixCacheLookup {
                promote_tokens_per_tier: vec![0; self.num_tiers()],
                promote_bytes_per_tier: vec![0; self.num_tiers()],
                ..Default::default()
            };
        }
        // Only the prompt (short of its last token, which is always
        // computed) can be served from the cache at admission: blocks
        // beyond it — a session's future output — must not be looked up,
        // or a request could park to promote what it can never use.
        let usable = ((request.num_prompt_tokens.saturating_sub(1) / self.block_size.max(1))
            as usize)
            .min(request.prompt_block_hashes.len());
        let lk = self
            .radix
            .lock()
            .unwrap()
            .lookup(self.worker, &request.prompt_block_hashes[..usable]);
        self.lookup_from(lk)
    }

    fn lookup_from(&self, lk: HbmLookup) -> PrefixCacheLookup {
        let bs = self.block_size;
        PrefixCacheLookup {
            total_cached_tokens: lk.cached() * bs,
            hbm_tokens: lk.hbm * bs,
            in_flight_tokens: lk.landing * bs,
            promote_tokens_per_tier: lk.tier_blocks.iter().map(|b| b * bs).collect(),
            promote_bytes_per_tier: lk.tier_bytes.clone(),
            join_leader: lk.leader.clone(),
            tier_spans: lk.tier_spans.clone(),
        }
    }

    /// Cached prefix length in tokens, for routing: HBM, in flight, or in
    /// a tier — the longest prefix held anywhere on this worker.
    pub fn cached_prefix_tokens_estimate(&self, hashes: &[u64]) -> u32 {
        if !self.enable_prefix_caching || hashes.is_empty() {
            return 0;
        }
        self.radix
            .lock()
            .unwrap()
            .lookup(self.worker, hashes)
            .cached()
            * self.block_size
    }

    /// Contiguous prompt prefix already resident in HBM, in tokens: the part
    /// of an incoming context a remote KV transfer (disaggregated hand-off)
    /// can skip. Stops at the first block that is not in HBM (in flight or
    /// in a spillover tier counts as not resident).
    pub fn hbm_prefix_tokens(&self, hashes: &[u64]) -> u32 {
        if !self.enable_prefix_caching {
            return 0;
        }
        self.radix
            .lock()
            .unwrap()
            .resident_prefix(self.worker, hashes)
            * self.block_size
    }

    /// Record a prefix-cache lookup in the hit/miss statistics.
    pub fn record_prefix_lookup(&mut self, lookup: &PrefixCacheLookup) {
        let tokens = lookup.total_cached_tokens;
        self.stats.lookups += 1;
        self.stats.hit_size_sum += tokens as u64;
        if tokens == 0 {
            self.stats.misses += 1;
        } else {
            self.stats.hits += 1;
        }
    }

    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.stats
    }

    /// Time the promotions a lookup needs would take if started now:
    /// every tier's share moves in parallel, so the slowest of them, at
    /// each fetch path's current fair share. 0 when nothing needs
    /// promoting; infinite without a memory graph.
    pub fn estimate_fetch(&self, lookup: &PrefixCacheLookup) -> f64 {
        let Some((g, w)) = &self.memory else {
            return if lookup.needs_promotion() {
                f64::INFINITY
            } else {
                0.0
            };
        };
        let g = g.lock().unwrap();
        lookup
            .promote_bytes_per_tier
            .iter()
            .enumerate()
            .filter(|(_, &b)| b > 0)
            .map(|(tier, &b)| g.estimate_promotion(*w, tier, b))
            .fold(0.0, f64::max)
    }

    /// Time promoting a `tokens`-token prefix from the first tier would
    /// take if started now (the prefix need not be there yet: what a
    /// prefetch plan assumes when it is made). Infinite without tiers.
    pub fn estimate_promotion_of(&self, tokens: u32) -> f64 {
        let Some((g, w)) = &self.memory else {
            return f64::INFINITY;
        };
        let g = g.lock().unwrap();
        if g.num_tiers(*w) == 0 {
            return f64::INFINITY;
        }
        g.estimate_promotion(*w, 0, self.kv_bytes_for_tokens(tokens))
    }

    /// A prefetch of `tokens` started: count it.
    pub fn record_prefetch(&mut self, tokens: u32) {
        self.stats.prefetches += 1;
        self.stats.prefetch_tokens += tokens as u64;
    }

    /// The tier-held part of a lookup is being recomputed rather than
    /// fetched: count it.
    pub fn record_recompute(&mut self, tokens: u32) {
        self.stats.recomputed += 1;
        self.stats.recomputed_tokens += tokens as u64;
    }

    /// The session step holding `hashes` completed with `outlook`: mark the
    /// prefix its next step re-enters with (the first `shared_tokens`
    /// worth) and unmark the rest. Free runs and tier ranges are re-keyed.
    /// `None` = the trajectory is over.
    pub fn set_outlook(&mut self, hashes: &[u64], outlook: Option<Outlook>) {
        if !self.enable_prefix_caching || hashes.is_empty() {
            return;
        }
        let shared = outlook.map_or(0, |o| o.shared_tokens / self.block_size);
        let mut r = self.radix.lock().unwrap();
        let path = r.resolve(hashes);
        r.set_outlook(&path, outlook.map(|o| o.next_arrival), shared);
    }

    /// The announced re-entry time of the sequence starting with `hash`.
    pub fn outlook_of(&self, hash: u64) -> Option<f64> {
        self.radix.lock().unwrap().outlook_of(&[hash])
    }

    /// Begin tracking an in-flight promotion for `request_id`: one transfer
    /// per tier with bytes to move (`lookup.promote_bytes_per_tier`), each
    /// along that tier's path in the memory graph, sharing every edge with
    /// whatever else is in flight. Landing blocks were reserved by
    /// `reserve_blocks_for_transfer`; this only kicks off the byte pumping.
    pub fn start_transfer(
        &mut self,
        request_id: String,
        lookup: &PrefixCacheLookup,
        _hashes: &[u64],
        current_time: f64,
    ) {
        let mut active_tiers = 0u32;
        if let Some((g, w)) = &self.memory {
            let mut g = g.lock().unwrap();
            for (i, &bytes) in lookup.promote_bytes_per_tier.iter().enumerate() {
                if bytes == 0 {
                    continue;
                }
                let spans: Vec<Span> = lookup
                    .tier_spans
                    .iter()
                    .filter(|(t, _)| *t == i)
                    .map(|(_, s)| *s)
                    .collect();
                g.submit_promotion(*w, i, &request_id, bytes, &spans, current_time);
                active_tiers += 1;
            }
        }
        if active_tiers > 0 {
            self.leader_active_tiers
                .insert(request_id.clone(), active_tiers);
            self.leader_joiners.insert(request_id, Vec::new());
        }
    }

    /// Register `joiner_id` as piggybacking on the transfer that owns the
    /// blocks covered by `lookup.in_flight_tokens`. The joiner contributes
    /// no bandwidth load and becomes ready at the same time as the leader.
    pub fn join_transfer(&mut self, joiner_id: String, lookup: &PrefixCacheLookup) {
        let Some(leader) = lookup.join_leader.clone() else {
            return;
        };
        if let Some(joiners) = self.leader_joiners.get_mut(&leader) {
            joiners.push(joiner_id.clone());
        }
        self.joiner_to_leader.insert(joiner_id, leader);
    }

    /// Advance the memory graph's transfers to `current_time` and collect
    /// this worker's finished promotions. Returns the set of request ids
    /// whose transfer has completed (leaders plus their joiners).
    pub fn advance_transfers(&mut self, current_time: f64) -> HashSet<String> {
        let mut completed: HashSet<String> = HashSet::new();
        let mine: Vec<String> = match &self.memory {
            Some((g, w)) => {
                let mut g = g.lock().unwrap();
                g.advance(current_time);
                let mut ids: Vec<String> =
                    g.take_completed(Owner::Worker(*w)).into_iter().collect();
                ids.sort();
                ids
            }
            None => Vec::new(),
        };
        for id in mine {
            let leader = promotion_request(&id).to_string();
            if let Some(active) = self.leader_active_tiers.get_mut(&leader) {
                *active = active.saturating_sub(1);
                if *active == 0 {
                    self.leader_active_tiers.remove(&leader);
                    if let Some(joiners) = self.leader_joiners.remove(&leader) {
                        for joiner in joiners {
                            self.joiner_to_leader.remove(&joiner);
                            completed.insert(joiner);
                        }
                    }
                    completed.insert(leader);
                }
            }
        }
        completed
    }

    /// Project the remaining time for an in-flight transfer at `current_time`,
    /// assuming current tier contention persists. For joiners, returns the
    /// leader's projected remaining time. Returns 0.0 for unknown ids.
    pub fn estimate_remaining_time(&self, request_id: &str) -> f64 {
        let leader = self
            .joiner_to_leader
            .get(request_id)
            .map(String::as_str)
            .unwrap_or(request_id);
        if !self.leader_active_tiers.contains_key(leader) {
            return 0.0;
        }
        match &self.memory {
            Some((g, w)) => g.lock().unwrap().estimate_promotion_remaining(*w, leader),
            None => 0.0,
        }
    }

    /// Put the block of `hash` (as a one-block sequence) straight into this
    /// worker's tier `tier` (pre-warming).
    pub fn plant_in_tier(&self, tier: usize, hash: u64) {
        self.plant_in_tier_path(tier, &[hash]);
    }

    /// Put the last block of the sequence `hashes` straight into this
    /// worker's tier `tier` (pre-warming a block at its position on a
    /// chain).
    pub fn plant_in_tier_path(&self, tier: usize, hashes: &[u64]) {
        if hashes.is_empty() {
            return;
        }
        let span = {
            let mut r = self.radix.lock().unwrap();
            let path = r.insert(hashes);
            let last = *path.segs.last().unwrap();
            Span {
                node: last.node,
                start: last.len - 1,
                end: last.len,
            }
        };
        if let Some((g, w)) = &self.memory {
            g.lock().unwrap().plant(*w, tier, span);
        }
    }

    /// `(blocks in use, references, blocks free)` — for stall diagnostics.
    pub fn ref_summary(&self) -> (usize, u64, usize) {
        let free = self.num_free_blocks();
        let used = self.total_blocks as usize - free;
        (used, used as u64, free)
    }

    #[cfg(test)]
    pub(crate) fn hbm_contains(&self, hash: u64) -> bool {
        self.radix.lock().unwrap().hbm_contains(self.worker, hash)
    }
}

/// The spans of `path` covering its first `upto` blocks.
pub(crate) fn path_spans(path: &Path, upto: u32) -> Vec<Span> {
    let mut out = Vec::with_capacity(path.segs.len());
    let mut acc = 0u32;
    for seg in &path.segs {
        if acc >= upto {
            break;
        }
        let end = (upto - acc).min(seg.len);
        out.push(Span {
            node: seg.node,
            start: 0,
            end,
        });
        acc += seg.len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EvictionPolicy, WritePolicy};

    fn create_test_request(id: &str, prompt_tokens: u32) -> Request {
        Request::new(id.to_string(), 0, 0.0, prompt_tokens, 50)
    }

    /// Allocate `tokens` for `req` and account for them on the request.
    fn alloc(m: &mut KVCacheManager, req: &mut Request, tokens: u32) -> u32 {
        let n = m.allocate_blocks(req, tokens).unwrap();
        req.kv_blocks.extend(n);
        n
    }

    fn free(m: &mut KVCacheManager, req: &mut Request) {
        m.free_blocks(req);
        req.kv_blocks.clear();
    }

    #[test]
    fn test_kv_cache_manager_creation() {
        let manager = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, false);
        assert_eq!(manager.block_size, 16);
        assert_eq!(manager.total_blocks, 10);
        assert_eq!(manager.num_free_blocks(), 10);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn content_blocks_follow_the_kv_curve() {
        // Linear curve: exactly ceil(t / block_size).
        let m = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, false);
        assert_eq!(m.bytes_per_block(), 1600);
        assert_eq!(m.content_blocks_for_tokens(0), 0);
        assert_eq!(m.content_blocks_for_tokens(1), 1);
        assert_eq!(m.content_blocks_for_tokens(16), 1);
        assert_eq!(m.content_blocks_for_tokens(17), 2);
        assert_eq!(m.content_blocks_for_tokens(160), 10);
        // Windowed curve (32-token sliding window): a long sequence still
        // occupies only two blocks' worth of bytes.
        let m = KVCacheManager::new(16000, 16, |t| 100 * t.min(32) as u64, 0, false);
        assert_eq!(m.content_blocks_for_tokens(1000), 2);
        // Per-sequence state is padded to whole blocks and added on top.
        let m = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 1700, false);
        assert_eq!(m.state_blocks(), 2);
        assert_eq!(m.blocks_for_context(16), 3);
    }

    #[test]
    fn incremental_allocation_grows_the_hold_and_frees_it_whole() {
        let mut m = KVCacheManager::new(16 * 100 * 100, 16, |t| 100 * t as u64, 0, true);
        let mut req = create_test_request("a", 64);
        req.prompt_block_hashes = vec![1, 2, 3, 4];
        assert_eq!(alloc(&mut m, &mut req, 64), 4);
        req.num_computed_tokens = 64;
        // Decode into an unhashed fifth block.
        assert_eq!(alloc(&mut m, &mut req, 1), 1);
        assert_eq!(req.kv_blocks.len(), 5);
        assert_eq!(m.num_free_blocks(), 95);
        // Bytes: 5 fresh blocks written and touched.
        assert_eq!(m.bytes_written(), 5 * 1600);
        assert_eq!(m.bytes_touched(), 5 * 1600);
        free(&mut m, &mut req);
        assert_eq!(m.num_free_blocks(), 100);
        // The hashed prefix is still hittable; the anonymous block is not.
        let probe = {
            let mut p = create_test_request("p", 64);
            p.prompt_block_hashes = vec![1, 2, 3, 4];
            p
        };
        // 3 of the 4 blocks are usable prefix (the last prompt block is
        // always computed).
        assert_eq!(m.peek_prefix_cache(&probe).hbm_tokens, 48);
    }

    #[test]
    fn prefix_hits_share_blocks_and_stop_at_the_first_miss() {
        let mut m = KVCacheManager::new(10 * 1600, 16, |t| 100 * t as u64, 0, true);
        let mut a = create_test_request("a", 48);
        a.prompt_block_hashes = vec![1, 2, 3];
        alloc(&mut m, &mut a, 48);
        // b shares [1, 2] then diverges: two hits, one fresh block.
        let mut b = create_test_request("b", 48);
        b.prompt_block_hashes = vec![1, 2, 9];
        let lk = m.peek_prefix_cache(&b);
        assert_eq!((lk.hbm_tokens, lk.total_cached_tokens), (32, 32));
        assert!(!lk.needs_promotion());
        assert_eq!(alloc(&mut m, &mut b, 48), 3);
        assert_eq!(
            m.num_free_blocks(),
            10 - 4,
            "one shared prefix, 4 distinct blocks"
        );
        // Freeing a keeps its blocks hittable for b's siblings.
        free(&mut m, &mut a);
        let mut c = create_test_request("c", 49);
        c.prompt_block_hashes = vec![1, 2, 3, 4];
        assert_eq!(m.peek_prefix_cache(&c).hbm_tokens, 48);
        assert_eq!(m.cached_prefix_tokens_estimate(&[1, 2, 3, 4]), 48);
        assert_eq!(m.hbm_prefix_tokens(&[1, 2, 3, 4]), 48);
        assert_eq!(m.hbm_prefix_tokens(&[7]), 0);
    }

    #[test]
    fn lru_recycles_the_least_recently_freed_tail_first() {
        // 6 blocks; three two-block sequences freed in order a, b, c.
        let mut m = KVCacheManager::new(6 * 1600, 16, |t| 100 * t as u64, 0, true);
        let mut seqs = Vec::new();
        for (name, hashes) in [("a", vec![1u64, 2]), ("b", vec![3, 4]), ("c", vec![5, 6])] {
            let mut r = create_test_request(name, 32);
            r.prompt_block_hashes = hashes;
            alloc(&mut m, &mut r, 32);
            seqs.push(r);
        }
        for r in seqs.iter_mut() {
            free(&mut m, r);
        }
        assert_eq!(m.num_free_blocks(), 6);
        // A new 3-block sequence takes a whole and b's tail block.
        let mut d = create_test_request("d", 48);
        d.prompt_block_hashes = vec![7, 8, 9];
        alloc(&mut m, &mut d, 48);
        assert!(!m.hbm_contains(1) && !m.hbm_contains(2));
        assert!(
            m.hbm_contains(3) && !m.hbm_contains(4),
            "b's tail went, its head survives"
        );
        assert!(m.hbm_contains(5) && m.hbm_contains(6));
        assert_eq!(m.num_free_blocks(), 3);
    }

    #[test]
    fn a_hit_on_a_free_block_cannot_be_recycled_underneath_it() {
        // 2 blocks. A caches [1], frees. B hits hash 1 and needs one fresh
        // block: the fresh block must not recycle A's (now B's) block.
        let mut m = KVCacheManager::new(2 * 16 * 100, 16, |t| 100 * t as u64, 0, true);
        let mut a = create_test_request("a", 16);
        a.prompt_block_hashes = vec![1];
        alloc(&mut m, &mut a, 16);
        free(&mut m, &mut a);
        let mut b = create_test_request("b", 32);
        b.prompt_block_hashes = vec![1, 2];
        assert_eq!(alloc(&mut m, &mut b, 32), 2);
        assert!(m.hbm_contains(1) && m.hbm_contains(2));
        assert_eq!(m.num_free_blocks(), 0);
        // With both blocks held, a third request that hits nothing can't be
        // allocated: the hit did not leave a phantom free block behind.
        let mut c = create_test_request("c", 16);
        c.prompt_block_hashes = vec![9];
        assert!(m.allocate_blocks(&c, 16).is_none());
        // A request that hits [1] but needs one more block is also refused
        // (the hit's block is not available to its own miss).
        let mut d = create_test_request("d", 32);
        d.prompt_block_hashes = vec![1, 3];
        assert!(m.allocate_blocks(&d, 32).is_none());
        // Free b's second block worth by freeing b: d fits.
        free(&mut m, &mut b);
        assert_eq!(m.num_free_blocks(), 2);
        assert!(m.allocate_blocks(&d, 32).is_some());
    }

    #[test]
    fn hbm_prefers_recycling_a_backed_block_when_asked() {
        let build = |backed_first: bool| {
            let graph = MemoryGraph::private_with(
                1,
                &[("host", 100, 1e9)],
                16,
                Arc::new(|t| 100 * t as u64),
            )
            .with_policies(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {})
            .with_hbm_evict_backed_first(backed_first)
            .shared_handle();
            let mut m = KVCacheManager::new(3 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
                .with_memory(graph.clone(), 0);
            // Three blocks with hashes 1, 2, 3, freed in that order.
            for h in [1u64, 2, 3] {
                let mut r = create_test_request("r", 16);
                r.prompt_block_hashes = vec![h];
                alloc(&mut m, &mut r, 16);
                free(&mut m, &mut r);
            }
            // Only hash 2's KV sits in the tier.
            m.plant_in_tier(0, 2);
            let mut d = create_test_request("d", 16);
            d.prompt_block_hashes = vec![4];
            alloc(&mut m, &mut d, 16);
            m
        };
        // Least recently freed (hash 1) goes by default...
        let m = build(false);
        assert!(!m.hbm_contains(1));
        assert!(m.hbm_contains(2));
        // ...but the backed block (hash 2) goes when preferred: dropping it
        // costs nothing, and hash 1's write-back is avoided.
        let m = build(true);
        assert!(m.hbm_contains(1));
        assert!(!m.hbm_contains(2));
    }

    #[test]
    fn hbm_outlook_recycles_dead_then_farthest_then_tail_first() {
        use crate::request::Outlook;
        // 6 blocks. Three two-block sessions A, B, C: A announces a re-entry
        // at t=100 over both blocks, B at t=10, C has none. Under outlook
        // C goes first, then A (farthest) tail-first, then B's tail.
        let build = |policy: HbmEviction| {
            let mut m = KVCacheManager::new(6 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
                .with_hbm_eviction(policy);
            for (name, hashes, outlook) in [
                (
                    "a",
                    vec![1u64, 2],
                    Some(Outlook {
                        next_arrival: 100.0,
                        shared_tokens: 32,
                    }),
                ),
                (
                    "b",
                    vec![3, 4],
                    Some(Outlook {
                        next_arrival: 10.0,
                        shared_tokens: 32,
                    }),
                ),
                ("c", vec![5, 6], None),
            ] {
                let mut r = create_test_request(name, 32);
                r.prompt_block_hashes = hashes.clone();
                alloc(&mut m, &mut r, 32);
                m.set_outlook(&hashes, outlook);
                free(&mut m, &mut r);
            }
            m
        };
        let recycle = |m: &mut KVCacheManager, k: u32| {
            let mut d = create_test_request("d", 16 * k);
            d.prompt_block_hashes = (100..100 + k as u64).collect();
            alloc(m, &mut d, 16 * k);
        };
        let mut lru = build(HbmEviction::Lru {});
        recycle(&mut lru, 2);
        assert!(!lru.hbm_contains(1) && !lru.hbm_contains(2), "LRU: A first");
        let mut o = build(HbmEviction::Outlook {});
        recycle(&mut o, 5);
        assert!(!o.hbm_contains(5) && !o.hbm_contains(6), "dead C first");
        assert!(!o.hbm_contains(1) && !o.hbm_contains(2), "far A next");
        assert!(
            o.hbm_contains(3) && !o.hbm_contains(4),
            "then B's tail; head survives"
        );
        // A nearer re-entry announced for A re-orders it behind B.
        let mut o = build(HbmEviction::Outlook {});
        o.set_outlook(
            &[1, 2],
            Some(Outlook {
                next_arrival: 1.0,
                shared_tokens: 32,
            }),
        );
        recycle(&mut o, 4);
        assert!(o.hbm_contains(1) && o.hbm_contains(2));
        assert!(!o.hbm_contains(3) && !o.hbm_contains(4));
        // A partial outlook marks only the shared prefix: A's tail is dead.
        let mut o = build(HbmEviction::Outlook {});
        o.set_outlook(
            &[1, 2],
            Some(Outlook {
                next_arrival: 1.0,
                shared_tokens: 16,
            }),
        );
        recycle(&mut o, 3);
        assert!(!o.hbm_contains(5) && !o.hbm_contains(6));
        assert!(!o.hbm_contains(2) && o.hbm_contains(1));
    }

    #[test]
    fn write_through_writes_every_hashed_fresh_block() {
        let graph =
            MemoryGraph::private_with(1, &[("host", 100, 1e9)], 16, Arc::new(|t| 100 * t as u64))
                .with_policies(WritePolicy::WriteThrough {}, EvictionPolicy::Fifo {})
                .shared_handle();
        let mut m = KVCacheManager::new(16 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
            .with_memory(graph.clone(), 0);
        // 48-token prompt (3 hashed blocks) plus 16 tokens of output (one
        // unhashed block): 3 × 1600 bytes are written through.
        let mut r = create_test_request("r", 48);
        r.prompt_block_hashes = vec![1, 2, 3];
        alloc(&mut m, &mut r, 48);
        r.num_computed_tokens = 48;
        alloc(&mut m, &mut r, 16);
        let g = graph.lock().unwrap();
        assert!((g.flows().bytes_submitted_write - 3.0 * 1600.0).abs() < 1e-9);
        assert!(g.holds_hash(0, 1) && g.holds_hash(0, 3));
        drop(g);
        // A hit writes nothing more.
        let mut s = create_test_request("s", 48);
        s.prompt_block_hashes = vec![1, 2, 3];
        alloc(&mut m, &mut s, 48);
        assert!((graph.lock().unwrap().flows().bytes_submitted_write - 3.0 * 1600.0).abs() < 1e-9);
    }

    #[test]
    fn selective_writes_on_the_nth_hit_and_live_only_announced_evictions() {
        use crate::request::Outlook;
        let build = |write: WritePolicy| {
            let graph = MemoryGraph::private_with(
                1,
                &[("host", 100, 1e9)],
                16,
                Arc::new(|t| 100 * t as u64),
            )
            .with_policies(write, EvictionPolicy::Fifo {})
            .shared_handle();
            let m = KVCacheManager::new(4 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
                .with_memory(graph.clone(), 0);
            (m, graph)
        };
        // Selective: the second HBM hit on [1, 2] writes those two blocks.
        let (mut m, graph) = build(WritePolicy::Selective { min_hits: 2 });
        let mut a = create_test_request("a", 32);
        a.prompt_block_hashes = vec![1, 2];
        alloc(&mut m, &mut a, 32);
        free(&mut m, &mut a);
        for _ in 0..2 {
            let mut b = create_test_request("b", 32);
            b.prompt_block_hashes = vec![1, 2];
            alloc(&mut m, &mut b, 32);
            free(&mut m, &mut b);
        }
        assert!((graph.lock().unwrap().flows().bytes_submitted_write - 2.0 * 1600.0).abs() < 1e-9);
        // Live: eviction writes only blocks whose session announced a
        // re-entry. Fill HBM (4) with a (2, announced) and c (2, not);
        // then d (2 fresh) evicts LRU order a first? a freed first → a is
        // evicted → written (announced); c stays.
        let (mut m, graph) = build(WritePolicy::Live {});
        let mut a = create_test_request("a", 32);
        a.prompt_block_hashes = vec![1, 2];
        alloc(&mut m, &mut a, 32);
        m.set_outlook(
            &[1, 2],
            Some(Outlook {
                next_arrival: 50.0,
                shared_tokens: 32,
            }),
        );
        free(&mut m, &mut a);
        let mut c = create_test_request("c", 32);
        c.prompt_block_hashes = vec![5, 6];
        alloc(&mut m, &mut c, 32);
        free(&mut m, &mut c);
        let mut d = create_test_request("d", 64);
        d.prompt_block_hashes = vec![7, 8, 9, 10];
        alloc(&mut m, &mut d, 64);
        let g = graph.lock().unwrap();
        assert!((g.flows().bytes_submitted_write - 2.0 * 1600.0).abs() < 1e-9);
        assert!(g.holds_hash(0, 1) && g.holds_hash(0, 2));
        assert!(!g.holds_hash(0, 5) && !g.holds_hash(0, 6));
    }
}

#[cfg(test)]
mod landing_tests {
    use super::*;
    use crate::config::{EvictionPolicy, WritePolicy};

    #[test]
    fn a_multi_block_promotion_lands_resident_and_is_not_promoted_again() {
        let graph = MemoryGraph::private_with(
            1,
            &[("host", 100, 1e9)],
            16,
            Arc::new(|t| 100 * t as u64),
        )
        .with_policies(WritePolicy::Selective { min_hits: 1 }, EvictionPolicy::Lru {})
        .shared_handle();
        let mut m = KVCacheManager::new(16 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
            .with_memory(graph.clone(), 0);
        // Chain [1,2,3,4] sits in the tier (planted at its positions).
        for k in 1..=4u64 {
            let hashes: Vec<u64> = (1..=k).collect();
            m.plant_in_tier_path(0, &hashes);
        }
        let mut r = Request::new("r".into(), 0, 0.0, 16 * 5, 1);
        r.prompt_block_hashes = vec![1, 2, 3, 4, 5];
        let lk = m.peek_prefix_cache(&r);
        assert_eq!((lk.hbm_tokens, lk.in_flight_tokens), (0, 0));
        assert_eq!(lk.promote_tokens_per_tier, vec![64]);
        // Park: reserve 4 landing blocks, start the transfer.
        r.num_cached_tokens = 64;
        let n = m.reserve_blocks_for_transfer(&r, 64).unwrap();
        r.kv_blocks.extend(n);
        assert_eq!(r.kv_blocks.len(), 4);
        m.start_transfer("r".into(), &lk, &r.prompt_block_hashes, 0.0);
        // Another request sees it in flight and would join.
        let mut j = Request::new("j".into(), 0, 0.0, 16 * 5, 1);
        j.prompt_block_hashes = vec![1, 2, 3, 4, 9];
        let lj = m.peek_prefix_cache(&j);
        assert_eq!((lj.hbm_tokens, lj.in_flight_tokens), (0, 64));
        assert_eq!(lj.join_leader.as_deref(), Some("r"));
        // Land it.
        let done = m.advance_transfers(10.0);
        assert!(done.contains("r"));
        m.publish_transferred_blocks(&r, 4);
        r.num_computed_tokens = 64;
        let lk2 = m.peek_prefix_cache(&r);
        assert_eq!((lk2.hbm_tokens, lk2.in_flight_tokens), (64, 0), "landed → resident");
        assert!(!lk2.needs_promotion());
        // It now allocates its fifth block and runs.
        let n = m.allocate_blocks(&r, 16).unwrap();
        assert_eq!(n, 1);
        r.kv_blocks.extend(n);
        m.free_blocks(&r);
        assert_eq!(m.num_free_blocks(), 16);
    }
}
