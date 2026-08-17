use super::block::Block;
use super::free_queue::FreeQueue;
use super::link::Link;
use crate::config::KVTier;
use crate::request::{BlockId, Request};
use std::collections::{HashMap, HashSet, VecDeque};

/// Bytes of KV a sequence of `t` tokens occupies. Supplied by the model
/// (`ModelSpec::kv_storage_bytes`); the manager quantises it into blocks.
pub type KvBytesFn = Box<dyn Fn(u32) -> u64 + Send + Sync>;

#[derive(Debug, Clone)]
struct InFlightEntry {
    leader_id: String,
    block_id: BlockId,
}

/// A spillover tier in the KV cache hierarchy. Tracks block content hashes
/// that have been evicted from a closer tier; a hash present here can be
/// "promoted" back to HBM by paying the tier's transfer bandwidth, instead
/// of being recomputed via prefill. The actual byte-pumping (bandwidth
/// sharing across concurrent transfers, time advance) lives on the embedded
/// `Link`.
#[derive(Debug)]
struct SpilloverTier {
    capacity_blocks: u32,
    members: HashSet<u64>,
    /// Insertion order; front is oldest.
    order: VecDeque<u64>,
    num_evictions: u64,
    link: Link,
}

impl SpilloverTier {
    fn new(capacity_blocks: u32, bandwidth_to_hbm: f64) -> Self {
        Self {
            capacity_blocks,
            members: HashSet::new(),
            order: VecDeque::new(),
            num_evictions: 0,
            link: Link::new(bandwidth_to_hbm),
        }
    }

    fn contains(&self, hash: u64) -> bool {
        self.members.contains(&hash)
    }

    /// Insert a hash. Returns `Some(evicted)` if at capacity.
    fn insert(&mut self, hash: u64) -> Option<u64> {
        if self.members.contains(&hash) {
            return None;
        }
        self.members.insert(hash);
        self.order.push_back(hash);
        if self.members.len() > self.capacity_blocks as usize {
            let oldest = self.order.pop_front().unwrap();
            self.members.remove(&oldest);
            self.num_evictions += 1;
            Some(oldest)
        } else {
            None
        }
    }

    /// Remove a specific hash (e.g. on promotion back to HBM).
    fn remove(&mut self, hash: u64) -> bool {
        if self.members.remove(&hash) {
            if let Some(pos) = self.order.iter().position(|&h| h == hash) {
                self.order.remove(pos);
            }
            true
        } else {
            false
        }
    }
}

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
    /// with `KVCacheManager::tiers`.
    pub promote_tokens_per_tier: Vec<u32>,
    /// Per-spillover-tier bytes that need to be promoted (the KV footprint of
    /// the promoted token ranges, from the model's KV curve).
    pub promote_bytes_per_tier: Vec<u64>,
    /// Identity of the leader transfer (if any) covering some portion of
    /// the in-flight region. The scheduler uses this to call `join_transfer`
    /// against the same leader rather than starting a redundant one. If
    /// multiple leaders covered different blocks, this is the latest of
    /// them; in practice all in-flight blocks here share the same leader.
    pub join_leader: Option<String>,
}

impl PrefixCacheLookup {
    pub fn needs_promotion(&self) -> bool {
        self.promote_tokens_per_tier.iter().any(|&t| t > 0)
    }

    pub fn needs_join(&self) -> bool {
        self.in_flight_tokens > 0
    }
}

/// Manages KV cache blocks for all requests on one worker.
///
/// Capacity and occupancy are counted in blocks. A block is the KV footprint
/// of `block_size` tokens (`kv_bytes_at(block_size)`), and a sequence of `t`
/// tokens occupies `ceil(kv_bytes_at(t) / bytes_per_block)` content blocks.
/// For models whose KV is linear in position this is exactly
/// `ceil(t / block_size)`; for models whose footprint is not (sliding window,
/// DeepSeek-V4's window + compressed history) it charges the model's actual
/// bytes. Sequences additionally hold `state_blocks` fixed blocks for
/// length-independent per-sequence state (Mamba / GatedDeltaNet).
pub struct KVCacheManager {
    /// Block size in tokens.
    block_size: u32,

    /// KV bytes of a `t`-token sequence, from the model.
    kv_bytes_at: KvBytesFn,

    /// `kv_bytes_at(block_size)`: the unit of allocation.
    bytes_per_block: u64,

    /// Total number of blocks available.
    total_blocks: u32,

    /// All blocks.
    blocks: Vec<Block>,

    /// Free blocks, least recently freed first. A freed block keeps its
    /// hash and stays hittable until it is recycled from the front of this
    /// queue, so recycling order is the prefix cache's eviction order: LRU
    /// by free time. A hit on a free block pulls it back out.
    free_blocks: FreeQueue,

    /// Enable prefix caching.
    enable_prefix_caching: bool,

    /// HBM tier: maps block hash -> block_id. The hashes are incremental
    /// hashes of all the tokens up to a certain point, so a sequence with
    /// blocks (1, 2, 3) reconstructs as cat(cache[1], cache[2], cache[3]).
    prefix_cache: HashMap<u64, BlockId>,

    /// Spillover tiers, ordered closest-to-HBM first (tier 0 = host RAM).
    /// Each tier owns the bandwidth-sharing state for its own promotion
    /// transfers via an embedded `Link`.
    tiers: Vec<SpilloverTier>,

    /// For each leader currently being promoted, the count of tiers that
    /// still have bytes in flight for it. A leader is fully done when this
    /// hits zero across every tier.
    leader_active_tiers: HashMap<String, u32>,

    /// Joiners piggybacking on each leader, keyed by leader id. Joiners
    /// contribute no bandwidth load and become ready when the leader does.
    /// Modelled after vLLM's block-ref-count sharing of in-flight prefixes.
    leader_joiners: HashMap<String, Vec<String>>,

    /// Hashes currently being promoted; maps each in-flight hash to the
    /// leader request that owns the transfer and the HBM block reserved
    /// to land it. Subsequent requests with the same prefix join the
    /// existing transfer and ref-bump the same block, matching vLLM's
    /// block-ref-count sharing.
    in_flight_cache: HashMap<u64, InFlightEntry>,

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
    /// queue, so an upper bound on the LRU stack distance in bytes.
    bytes_touched: u64,
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
        self.hits += o.hits;
        self.misses += o.misses;
        self.hit_size_sum += o.hit_size_sum;
        self.lookups += o.lookups;
    }
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
        let bytes_per_block = kv_bytes_at(block_size);
        let total_blocks = kv_cache_capacity.checked_div(bytes_per_block).unwrap_or(0) as u32;
        // Fixed per-sequence state (Mamba/GDN) padded up to a whole number of
        // blocks, vLLM-style. Held for the sequence's lifetime.
        let state_blocks = if bytes_per_block == 0 {
            0
        } else {
            per_seq_state_bytes.div_ceil(bytes_per_block) as usize
        };

        let blocks = vec![Block::default(); total_blocks as usize];
        let free_blocks = FreeQueue::new(total_blocks as usize);

        Self {
            block_size,
            kv_bytes_at: Box::new(kv_bytes_at),
            bytes_per_block,
            total_blocks,
            blocks,
            free_blocks,
            enable_prefix_caching,
            prefix_cache: HashMap::new(),
            tiers: Vec::new(),
            leader_active_tiers: HashMap::new(),
            leader_joiners: HashMap::new(),
            in_flight_cache: HashMap::new(),
            joiner_to_leader: HashMap::new(),
            state_blocks,
            stats: PrefixCacheStats::default(),
            bytes_written: 0,
            bytes_touched: 0,
        }
    }

    /// Attach a spillover hierarchy. Tier ordering is closest-to-HBM first.
    pub fn with_tiers(mut self, tiers: &[KVTier]) -> Self {
        self.tiers = tiers
            .iter()
            .map(|t| {
                let capacity_blocks = t
                    .capacity_bytes
                    .checked_div(self.bytes_per_block)
                    .unwrap_or(0) as u32;
                SpilloverTier::new(capacity_blocks, t.bandwidth_to_hbm)
            })
            .collect();
        self
    }

    /// Fixed blocks reserved per running sequence for length-independent state.
    pub fn state_blocks(&self) -> usize {
        self.state_blocks
    }

    /// Bytes per block: the KV footprint of a `block_size`-token sequence.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

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

    /// Blocks a request must hold once it has `total_tokens` tokens of context:
    /// the fixed per-sequence state plus the content blocks.
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

    /// Push a hash down through the spillover hierarchy starting at tier 0.
    /// If a tier evicts to make room, the eviction cascades to the next tier.
    /// Hashes that fall off the bottom are silently dropped.
    fn cascade_demote(&mut self, mut hash: u64) {
        for tier in &mut self.tiers {
            match tier.insert(hash) {
                None => return,
                Some(evicted) => hash = evicted,
            }
        }
    }

    /// Remove a hash from any spillover tier it lives in. Used when a block
    /// is promoted back to HBM so the spillover index stays consistent.
    fn remove_from_spillover(&mut self, hash: u64) {
        for tier in &mut self.tiers {
            if tier.remove(hash) {
                return;
            }
        }
    }

    /// Allocate the blocks `request` needs to grow by `num_tokens` positions.
    /// `None` if there are not enough free blocks.
    pub fn allocate_blocks(&mut self, request: &Request, num_tokens: u32) -> Option<Vec<BlockId>> {
        self.allocate_blocks_inner(request, num_tokens, /*publish_to_hbm=*/ true)
    }

    /// Allocate blocks without publishing the request's hashes to the HBM
    /// prefix cache or removing them from the spillover hierarchy. Used to
    /// reserve HBM landing space for an in-flight promotion: until the
    /// transfer completes the data isn't really in HBM, so other requests
    /// looking up the same prefix should not hit HBM.
    pub fn reserve_blocks_for_transfer(
        &mut self,
        request: &Request,
        num_tokens: u32,
    ) -> Option<Vec<BlockId>> {
        self.allocate_blocks_inner(request, num_tokens, /*publish_to_hbm=*/ false)
    }

    /// Publish a completed transfer's blocks into the HBM prefix cache so
    /// subsequent lookups see them as resident. Also clears in-flight
    /// registrations for the involved hashes.
    pub fn publish_transferred_blocks(&mut self, hashes: &[u64], blocks: &[BlockId]) {
        if !self.enable_prefix_caching {
            return;
        }
        for (i, &hash) in hashes.iter().enumerate() {
            if let Some(&block_id) = blocks.get(i) {
                self.prefix_cache.insert(hash, block_id);
                self.remove_from_spillover(hash);
                self.in_flight_cache.remove(&hash);
            }
        }
    }

    fn allocate_blocks_inner(
        &mut self,
        request: &Request,
        num_tokens: u32,
        publish_to_hbm: bool,
    ) -> Option<Vec<BlockId>> {
        let blocks_needed = self.blocks_needed(request, num_tokens);
        let hashes = request.prompt_block_hashes.as_slice();

        // The i-th block allocated in this call continues the request's
        // existing allocation. Its index into the prompt hashes must skip the
        // per-sequence state blocks and the content blocks already held —
        // otherwise an incremental (decode-step) allocation looks up hash 0
        // and aliases the request's own first prompt block.
        let state_blocks = self.state_blocks;
        let already_allocated = request.kv_blocks.len();
        let hash_at = |i: usize| {
            (already_allocated + i)
                .checked_sub(state_blocks)
                .and_then(|idx| hashes.get(idx))
                .copied()
        };

        // Pass 1: decide for each requested block whether to ref an
        // existing block (HBM-resident or already in-flight for someone
        // else) or to allocate a fresh one. Bail early if there aren't
        // enough fresh blocks for the misses.
        enum Decision {
            Fresh,
            RefExisting(BlockId),
        }
        let mut decisions: Vec<Decision> = Vec::with_capacity(blocks_needed);
        let mut fresh_needed = 0usize;
        for i in 0..blocks_needed {
            let dec = match hash_at(i) {
                Some(h) if self.enable_prefix_caching => {
                    if let Some(&block_id) = self.prefix_cache.get(&h) {
                        Decision::RefExisting(block_id)
                    } else if let Some(entry) = self.in_flight_cache.get(&h) {
                        Decision::RefExisting(entry.block_id)
                    } else {
                        fresh_needed += 1;
                        Decision::Fresh
                    }
                }
                _ => {
                    fresh_needed += 1;
                    Decision::Fresh
                }
            };
            decisions.push(dec);
        }

        // Hits on free-but-cached blocks leave the free queue when they are
        // referenced below, so they are not available to the misses.
        let hits_on_free = decisions
            .iter()
            .filter(|d| matches!(d, Decision::RefExisting(id) if self.free_blocks.contains(*id)))
            .count();
        if self.free_blocks.len() < fresh_needed + hits_on_free {
            return None;
        }
        self.bytes_touched += (fresh_needed + hits_on_free) as u64 * self.bytes_per_block;

        // Pass 2: execute. Reference the hits first so a fresh pop can't
        // recycle a block a later hit in this same call refers to. Track
        // newly-allocated (hash, block_id) pairs so we can publish them to
        // HBM or to in_flight_cache afterwards.
        let mut allocated: Vec<Option<BlockId>> = vec![None; blocks_needed];
        for (i, decision) in decisions.iter().enumerate() {
            if let Decision::RefExisting(block_id) = decision {
                self.free_blocks.remove(*block_id);
                self.blocks[*block_id as usize].reference();
                allocated[i] = Some(*block_id);
            }
        }
        let mut evicted_hashes = Vec::new();
        let mut newly_allocated: Vec<(u64, BlockId)> = Vec::new();
        for (i, decision) in decisions.iter().enumerate() {
            if let Decision::Fresh = decision {
                let block_id = self.free_blocks.pop_front().unwrap();
                self.bytes_written += self.bytes_per_block;
                let evicted_hash = self.blocks[block_id as usize].allocate(hash_at(i));
                evicted_hashes.extend(evicted_hash);
                allocated[i] = Some(block_id);
                if let Some(h) = hash_at(i) {
                    newly_allocated.push((h, block_id));
                }
            }
        }
        let allocated: Vec<BlockId> = allocated.into_iter().map(|b| b.unwrap()).collect();

        if self.enable_prefix_caching {
            for hash in evicted_hashes {
                self.prefix_cache.remove(&hash);
                self.cascade_demote(hash);
            }
            if publish_to_hbm {
                for &(hash, block_id) in &newly_allocated {
                    self.prefix_cache.insert(hash, block_id);
                    self.remove_from_spillover(hash);
                }
            } else {
                // Reservation for an in-flight transfer: register the
                // freshly-allocated blocks in `in_flight_cache` so other
                // requests with the same prefix can join.
                for &(hash, block_id) in &newly_allocated {
                    self.in_flight_cache.insert(
                        hash,
                        InFlightEntry {
                            leader_id: request.request_id.clone(),
                            block_id,
                        },
                    );
                    self.remove_from_spillover(hash);
                }
            }
        }

        Some(allocated)
    }

    /// Free blocks from a request (due to preemption or completion). Blocks
    /// are queued for recycling tail first, so the end of a sequence is
    /// evicted before its beginning and what survives is always a prefix.
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) {
        for &block_id in block_ids.iter().rev() {
            if self.blocks[block_id as usize].release() {
                self.free_blocks.push_back(block_id);
            }
        }
    }

    /// KV bytes written into HBM so far (every fresh block allocation).
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// `bytes_written` plus the bytes of free blocks that hits pulled back
    /// into use.
    pub fn bytes_touched(&self) -> u64 {
        self.bytes_touched
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks as usize
    }

    /// Fraction of blocks in use; 0 when the cache has no blocks.
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.free_blocks.len() as f64 / self.total_blocks as f64)
    }

    /// Look up a request's prompt against the prefix cache hierarchy.
    /// Walks the request's incremental block hashes from the start; for each
    /// block, records whether it lives in HBM, is currently in flight on
    /// behalf of another request, or sits in a spillover tier. Stops at the
    /// first block that's nowhere. Does not touch the lookup statistics; call
    /// `record_prefix_lookup` when the lookup is acted on.
    pub fn peek_prefix_cache(&self, request: &Request) -> PrefixCacheLookup {
        let n_tiers = self.tiers.len();
        let mut lookup = PrefixCacheLookup {
            total_cached_tokens: 0,
            hbm_tokens: 0,
            in_flight_tokens: 0,
            promote_tokens_per_tier: vec![0u32; n_tiers],
            promote_bytes_per_tier: vec![0u64; n_tiers],
            join_leader: None,
        };

        if !self.enable_prefix_caching {
            return lookup;
        }

        for (k, &hash) in request.prompt_block_hashes.iter().enumerate() {
            if self.prefix_cache.contains_key(&hash) {
                lookup.hbm_tokens += self.block_size;
                lookup.total_cached_tokens += self.block_size;
                continue;
            }
            if let Some(entry) = self.in_flight_cache.get(&hash) {
                lookup.in_flight_tokens += self.block_size;
                lookup.total_cached_tokens += self.block_size;
                lookup.join_leader = Some(entry.leader_id.clone());
                continue;
            }
            match self.tiers.iter().position(|tier| tier.contains(hash)) {
                Some(idx) => {
                    let start = k as u32 * self.block_size;
                    let end = start + self.block_size;
                    lookup.promote_tokens_per_tier[idx] += self.block_size;
                    lookup.promote_bytes_per_tier[idx] +=
                        (self.kv_bytes_at)(end).saturating_sub((self.kv_bytes_at)(start));
                    lookup.total_cached_tokens += self.block_size;
                }
                None => break,
            }
        }

        lookup
    }

    /// Whether `hash` is held anywhere on this worker: HBM, in flight from a
    /// slower tier, or resident in a spillover tier.
    fn holds_block(&self, hash: u64) -> bool {
        self.prefix_cache.contains_key(&hash)
            || self.in_flight_cache.contains_key(&hash)
            || self.tiers.iter().any(|tier| tier.contains(hash))
    }

    /// Estimated cached prefix length in tokens, for routing. Finds the
    /// longest held prefix of `hashes` by galloping then binary search on
    /// block presence — O(log n) lookups instead of `peek_prefix_cache`'s
    /// walk — assuming presence is prefix-closed (block k held ⇒ blocks
    /// 0..k held), which is how blocks enter the cache and how they leave it
    /// under LRU eviction of leaves. Where eviction has punched a hole the
    /// estimate can overshoot; the scheduler's admission-time lookup is the
    /// truth, and a router working from an approximate view of each
    /// replica's prefix tree is what real KV-aware front ends do too.
    pub fn cached_prefix_tokens_estimate(&self, hashes: &[u64]) -> u32 {
        if !self.enable_prefix_caching || hashes.is_empty() {
            return 0;
        }
        let n = hashes.len();
        // `lo` blocks are known held; probe at lo + step.
        let mut lo = 0usize;
        let mut step = 1usize;
        while lo + step <= n && self.holds_block(hashes[lo + step - 1]) {
            lo += step;
            step *= 2;
        }
        // Held count lies in [lo, min(lo + step, n)); binary search it.
        let mut hi = (lo + step).min(n + 1); // exclusive upper bound on count
        while lo + 1 < hi {
            let mid = (lo + hi) / 2; // candidate count; block mid-1 must be held
            if self.holds_block(hashes[mid - 1]) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo as u32 * self.block_size
    }

    /// Contiguous prompt prefix already resident in HBM, in tokens: the part
    /// of an incoming context a remote KV transfer (disaggregated hand-off)
    /// can skip. Stops at the first block that is not in HBM (in flight or
    /// in a spillover tier counts as not resident).
    pub fn hbm_prefix_tokens(&self, hashes: &[u64]) -> u32 {
        if !self.enable_prefix_caching {
            return 0;
        }
        let held = hashes
            .iter()
            .take_while(|h| self.prefix_cache.contains_key(h))
            .count();
        held as u32 * self.block_size
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

    /// Begin tracking an in-flight transfer for `request_id`. Per-tier byte
    /// counts come from `lookup.promote_bytes_per_tier`; each non-zero tier
    /// gets a submission on its `Link` (which divides bandwidth across all
    /// its in-flight transfers). The `in_flight_cache` entries are populated
    /// at reservation time by `reserve_blocks_for_transfer`; this method only
    /// kicks off the byte-pumping side.
    pub fn start_transfer(
        &mut self,
        request_id: String,
        lookup: &PrefixCacheLookup,
        current_time: f64,
    ) {
        let mut active_tiers = 0u32;
        for (i, &bytes) in lookup.promote_bytes_per_tier.iter().enumerate() {
            if bytes == 0 {
                continue;
            }
            self.tiers[i]
                .link
                .submit(request_id.clone(), bytes, current_time);
            active_tiers += 1;
        }
        if active_tiers > 0 {
            self.leader_active_tiers
                .insert(request_id.clone(), active_tiers);
            self.leader_joiners.insert(request_id, Vec::new());
        }
    }

    /// Register `joiner_id` as piggybacking on the transfer that owns the
    /// hashes covered by `lookup.in_flight_tokens`. The joiner contributes
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

    /// Advance all in-flight transfers to `current_time`, with each tier's
    /// `Link` charging its bandwidth share appropriate to its contention.
    /// Returns the set of request ids whose transfer has completed (leaders
    /// plus their joiners).
    pub fn advance_transfers(&mut self, current_time: f64) -> HashSet<String> {
        let mut completed: HashSet<String> = HashSet::new();
        for tier in &mut self.tiers {
            let done_on_tier = tier.link.advance(current_time);
            for leader in done_on_tier {
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
        // Tiers are modelled as serial in the cost projection: the request
        // must drain on each tier it has bytes on, and we sum those times.
        self.tiers
            .iter()
            .map(|tier| tier.link.estimate_remaining(leader))
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn block_ref_count(&self, block_id: BlockId) -> u32 {
        self.blocks[block_id as usize].ref_count
    }

    #[cfg(test)]
    fn hbm_contains(&self, hash: u64) -> bool {
        self.prefix_cache.contains_key(&hash)
    }

    #[cfg(test)]
    fn in_flight_contains(&self, hash: u64) -> bool {
        self.in_flight_cache.contains_key(&hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request(id: &str, prompt_tokens: u32) -> Request {
        Request::new(id.to_string(), 0, 0.0, prompt_tokens, 50)
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
    fn incremental_allocation_does_not_alias_the_first_block() {
        // Regression: a decode-step allocation must take a fresh block, not
        // re-reference the request's own hash-0 block.
        let mut m = KVCacheManager::new(16 * 100 * 100, 16, |t| 100 * t as u64, 0, true);
        let mut req = create_test_request("a", 64);
        req.prompt_block_hashes = vec![1, 2, 3, 4];
        let first = m.allocate_blocks(&req, 64).unwrap();
        assert_eq!(first.len(), 4);
        req.kv_blocks.extend(first.iter().copied());
        req.num_computed_tokens = 64;
        let next = m.allocate_blocks(&req, 1).unwrap();
        assert_eq!(next.len(), 1);
        assert!(!first.contains(&next[0]));
        assert_eq!(m.num_free_blocks(), 100 - 5);
    }

    #[test]
    fn test_block_allocation() {
        let mut manager = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, false);
        let mut request = create_test_request("req-1", 32);

        let allocated = manager.allocate_blocks(&request, 32);
        assert!(allocated.is_some());

        let blocks = allocated.unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(manager.num_free_blocks(), 8);

        request.kv_blocks.extend(blocks);
        request.num_computed_tokens = 32;

        let more_blocks = manager.allocate_blocks(&request, 16);
        assert!(more_blocks.is_some());
        assert_eq!(more_blocks.unwrap().len(), 1);
        assert_eq!(manager.num_free_blocks(), 7);
    }

    #[test]
    fn test_block_allocation_failure() {
        let mut manager = KVCacheManager::new(1600, 16, |t| 100 * t as u64, 0, false);
        assert_eq!(manager.total_blocks, 1);

        let request = create_test_request("req-1", 32);
        let allocated = manager.allocate_blocks(&request, 32);
        assert!(allocated.is_none());
    }

    #[test]
    fn test_block_free() {
        let mut manager = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, false);
        let request = create_test_request("req-1", 32);

        let blocks = manager.allocate_blocks(&request, 32).unwrap();
        assert_eq!(manager.num_free_blocks(), 8);

        manager.free_blocks(&blocks);
        assert_eq!(manager.num_free_blocks(), 10);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_utilization() {
        let mut manager = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, false);
        assert_eq!(manager.utilization(), 0.0);

        let request = create_test_request("req-1", 32);
        let blocks = manager.allocate_blocks(&request, 32).unwrap();

        let util = manager.utilization();
        assert!((util - 0.2).abs() < 1e-10);

        manager.free_blocks(&blocks);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_prefix_caching() {
        let mut manager = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, true);

        let mut request1 = create_test_request("req-1", 16);
        request1.prompt_block_hashes = vec![12345];

        let lookup = manager.peek_prefix_cache(&request1);
        manager.record_prefix_lookup(&lookup);
        assert_eq!(lookup.total_cached_tokens, 0);
        assert_eq!(manager.stats.misses, 1);

        let blocks1 = manager.allocate_blocks(&request1, 16).unwrap();
        request1.kv_blocks.extend(blocks1);

        let mut request2 = create_test_request("req-2", 16);
        request2.prompt_block_hashes = vec![12345];
        let lookup = manager.peek_prefix_cache(&request2);
        manager.record_prefix_lookup(&lookup);
        assert_eq!(lookup.total_cached_tokens, 16);
        assert_eq!(manager.stats.hits, 1);

        let mut request3 = create_test_request("req-3", 16);
        request3.prompt_block_hashes = vec![67890];
        let lookup = manager.peek_prefix_cache(&request3);
        manager.record_prefix_lookup(&lookup);
        assert_eq!(lookup.total_cached_tokens, 0);
        assert_eq!(manager.stats.misses, 2);
    }

    #[test]
    fn prefix_estimate_matches_peek_on_a_prefix_closed_cache() {
        let mut manager = KVCacheManager::new(64 * 16 * 100, 16, |t| 100 * t as u64, 0, true);
        // Cache a 37-block prompt.
        let mut held = create_test_request("held", 37 * 16);
        held.prompt_block_hashes = (1..=37).collect();
        let blocks = manager.allocate_blocks(&held, 37 * 16).unwrap();
        held.kv_blocks.extend(blocks);

        for (hashes, want) in [
            ((1..=37).collect::<Vec<u64>>(), 37 * 16),      // exact
            ((1..=50).collect(), 37 * 16),                  // longer prompt, same prefix
            ((1..=5).collect(), 5 * 16),                    // shorter prompt
            ((1..=1).collect(), 16),                        // one block
            (vec![99, 100], 0),                             // cold
            (vec![], 0),                                    // empty
            ((1..=20).chain(200..=210).collect(), 20 * 16), // diverges at block 21
        ] {
            let mut probe = create_test_request("probe", hashes.len() as u32 * 16);
            probe.prompt_block_hashes = hashes.clone();
            let peek = manager.peek_prefix_cache(&probe).total_cached_tokens;
            assert_eq!(peek, want, "peek for {} blocks", hashes.len());
            assert_eq!(
                manager.cached_prefix_tokens_estimate(&hashes),
                want,
                "estimate for {} blocks",
                hashes.len()
            );
        }

        let off = KVCacheManager::new(64 * 16 * 100, 16, |t| 100 * t as u64, 0, false);
        assert_eq!(off.cached_prefix_tokens_estimate(&[1, 2, 3]), 0);
    }

    #[test]
    fn freed_prefixes_survive_churn_and_are_recycled_lru_tail_first() {
        // 8 blocks. Session A caches 3, completes; B allocates 2 in between;
        // A's re-entry must still hit all 3 (LRU recycles the oldest free
        // blocks, not the ones A just released).
        let mut m = KVCacheManager::new(8 * 16 * 100, 16, |t| 100 * t as u64, 0, true);
        let mut a = create_test_request("a", 48);
        a.prompt_block_hashes = vec![1, 2, 3];
        let a_blocks = m.allocate_blocks(&a, 48).unwrap();
        m.free_blocks(&a_blocks);
        assert_eq!(m.num_free_blocks(), 8);

        let mut b = create_test_request("b", 32);
        b.prompt_block_hashes = vec![10, 20];
        let b_blocks = m.allocate_blocks(&b, 32).unwrap();
        // B took the untouched blocks, not A's.
        assert!(
            b_blocks.iter().all(|id| !a_blocks.contains(id)),
            "{b_blocks:?}"
        );

        let mut a2 = create_test_request("a2", 64);
        a2.prompt_block_hashes = vec![1, 2, 3, 4];
        assert_eq!(m.peek_prefix_cache(&a2).total_cached_tokens, 48);
        let a2_blocks = m.allocate_blocks(&a2, 64).unwrap();
        // The three hits are A's physical blocks, now referenced and out of
        // the free queue: 8 − 2 (B) − 3 (hits) − 1 (fresh) = 2 free.
        assert_eq!(&a2_blocks[..3], &a_blocks[..]);
        assert_eq!(m.num_free_blocks(), 2);
        for id in &a2_blocks[..3] {
            assert_eq!(m.block_ref_count(*id), 1);
        }

        // Fill the rest: the two remaining fresh blocks, then eviction has
        // to recycle A's *tail* first (block for hash 4 was freed last of
        // A2's, and B's are still held), i.e. what survives is a prefix.
        m.free_blocks(&a2_blocks);
        let mut c = create_test_request("c", 32);
        c.prompt_block_hashes = vec![100, 200];
        m.allocate_blocks(&c, 32).unwrap(); // takes the 2 never-used blocks
        let mut d = create_test_request("d", 16);
        d.prompt_block_hashes = vec![300];
        m.allocate_blocks(&d, 16).unwrap(); // evicts hash 4 (A2's tail)
        let mut probe = create_test_request("p", 64);
        probe.prompt_block_hashes = vec![1, 2, 3, 4];
        assert_eq!(m.peek_prefix_cache(&probe).total_cached_tokens, 48);
        assert_eq!(m.cached_prefix_tokens_estimate(&[1, 2, 3, 4]), 48);
        let mut e = create_test_request("e", 16);
        e.prompt_block_hashes = vec![400];
        m.allocate_blocks(&e, 16).unwrap(); // evicts hash 3
        assert_eq!(m.peek_prefix_cache(&probe).total_cached_tokens, 32);
    }

    #[test]
    fn a_hit_on_a_free_block_cannot_be_recycled_underneath_it() {
        // 2 blocks. A caches [1], frees. B hits hash 1 and needs one fresh
        // block: the fresh pop must not return A's block (now B's).
        let mut m = KVCacheManager::new(2 * 16 * 100, 16, |t| 100 * t as u64, 0, true);
        let mut a = create_test_request("a", 16);
        a.prompt_block_hashes = vec![1];
        let a_blocks = m.allocate_blocks(&a, 16).unwrap();
        m.free_blocks(&a_blocks);
        let mut b = create_test_request("b", 32);
        b.prompt_block_hashes = vec![1, 2];
        let b_blocks = m.allocate_blocks(&b, 32).unwrap();
        assert_eq!(b_blocks[0], a_blocks[0]);
        assert_ne!(b_blocks[0], b_blocks[1]);
        assert_eq!(m.num_free_blocks(), 0);
        // With both blocks held, a third request that hits nothing can't be
        // allocated: the hit did not leave a phantom free block behind.
        let mut c = create_test_request("c", 16);
        c.prompt_block_hashes = vec![9];
        assert!(m.allocate_blocks(&c, 16).is_none());
        // And a request that hits [1] but needs one more block is also
        // refused (the hit's block is not available to its own miss).
        let mut d = create_test_request("d", 32);
        d.prompt_block_hashes = vec![1, 3];
        m.free_blocks(&b_blocks[1..]);
        assert_eq!(m.num_free_blocks(), 1);
        assert!(m.allocate_blocks(&d, 32).is_some());
    }

    #[test]
    fn test_spillover_demotion_on_eviction() {
        // 2 HBM blocks, 1 host-RAM tier of 2 blocks.
        let mut manager = KVCacheManager::new(2 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
            .with_tiers(&[KVTier {
                name: "host_ram".into(),
                capacity_bytes: 2 * 16 * 100,
                bandwidth_to_hbm: 64e9,
            }]);

        // Fill HBM with two requests' hashes.
        let mut req_a = create_test_request("a", 16);
        req_a.prompt_block_hashes = vec![1];
        let blocks_a = manager.allocate_blocks(&req_a, 16).unwrap();
        req_a.kv_blocks.extend(blocks_a.clone());

        let mut req_b = create_test_request("b", 16);
        req_b.prompt_block_hashes = vec![2];
        let blocks_b = manager.allocate_blocks(&req_b, 16).unwrap();
        req_b.kv_blocks.extend(blocks_b.clone());

        // HBM full. Free both so blocks are reusable but hashes still in HBM index.
        manager.free_blocks(&blocks_a);
        manager.free_blocks(&blocks_b);

        // New request with new hashes evicts both from HBM; they demote to host RAM.
        let mut req_c = create_test_request("c", 32);
        req_c.prompt_block_hashes = vec![3, 4];
        manager.allocate_blocks(&req_c, 32).unwrap();

        // Now lookup of hash 1 should hit host_ram, not HBM.
        let mut probe = create_test_request("probe", 16);
        probe.prompt_block_hashes = vec![1];
        let lookup = manager.peek_prefix_cache(&probe);
        assert_eq!(lookup.hbm_tokens, 0);
        assert_eq!(lookup.promote_tokens_per_tier[0], 16);
        assert_eq!(lookup.total_cached_tokens, 16);
        assert!(lookup.needs_promotion());
    }

    #[test]
    fn test_peek_reports_promotion_bytes_from_kv_curve() {
        // Two spillover tiers; a nonlinear KV curve (bytes = 100 * t^2 / 16 so
        // block k costs more than block k-1) checks that promotion bytes are
        // charged from the curve, not from a per-token constant.
        let curve = |t: u32| (100.0 * (t as f64).powi(2) / 16.0) as u64;
        let mut manager = KVCacheManager::new(curve(1024), 16, curve, 0, true).with_tiers(&[
            KVTier {
                name: "host_ram".into(),
                capacity_bytes: 1_000_000,
                bandwidth_to_hbm: 1e10,
            },
            KVTier {
                name: "nvme".into(),
                capacity_bytes: 10_000_000,
                bandwidth_to_hbm: 1e9,
            },
        ]);
        // Hash 1 (tokens 0..16) in tier 0, hash 2 (tokens 16..32) in tier 1.
        manager.tiers[0].insert(1);
        manager.tiers[1].insert(2);
        let mut req = create_test_request("a", 32);
        req.prompt_block_hashes = vec![1, 2];
        let lookup = manager.peek_prefix_cache(&req);
        assert_eq!(lookup.total_cached_tokens, 32);
        assert_eq!(lookup.promote_tokens_per_tier, vec![16, 16]);
        assert_eq!(
            lookup.promote_bytes_per_tier,
            vec![curve(16) - curve(0), curve(32) - curve(16)]
        );
    }

    #[test]
    fn test_concurrent_transfers_share_bandwidth() {
        // 1 GB/s tier; one transfer of 1 GB should take 1.0s alone, but two
        // concurrent equal-size transfers should each take ~2.0s.
        let mut manager =
            KVCacheManager::new(16_000, 16, |t| 100 * t as u64, 0, true).with_tiers(&[KVTier {
                name: "host_ram".into(),
                capacity_bytes: 10_000_000_000,
                bandwidth_to_hbm: 1e9,
            }]);
        let one_gb_in_tokens = 1_000_000_000 / 100; // 10 million tokens at 100 bytes/token
        let lookup = PrefixCacheLookup {
            total_cached_tokens: one_gb_in_tokens,
            hbm_tokens: 0,
            in_flight_tokens: 0,
            promote_tokens_per_tier: vec![one_gb_in_tokens],
            promote_bytes_per_tier: vec![1_000_000_000],
            join_leader: None,
        };

        // First transfer alone: 1 GB / 1 GB/s = 1.0s
        manager.start_transfer("a".into(), &lookup, 0.0);
        let est_a_alone = manager.estimate_remaining_time("a");
        assert!((est_a_alone - 1.0).abs() < 1e-3);

        // Second transfer admitted at the same instant: each gets half
        // bandwidth, so each individually projects 2.0s remaining.
        manager.start_transfer("b".into(), &lookup, 0.0);
        let est_a_shared = manager.estimate_remaining_time("a");
        let est_b_shared = manager.estimate_remaining_time("b");
        assert!((est_a_shared - 2.0).abs() < 1e-3);
        assert!((est_b_shared - 2.0).abs() < 1e-3);

        // Advance halfway. Each transfer has had 0.5 GB of progress at
        // 0.5 GB/s, so 0.5 GB remains in each. With contention still 2x,
        // each projects 1.0s more.
        let completed = manager.advance_transfers(1.0);
        assert!(completed.is_empty());
        let est_a_mid = manager.estimate_remaining_time("a");
        assert!((est_a_mid - 1.0).abs() < 1e-3);

        // Advance to t=2.0; both should now be done.
        let completed = manager.advance_transfers(2.0);
        assert!(completed.contains("a"));
        assert!(completed.contains("b"));
        assert_eq!(manager.leader_active_tiers.len(), 0);
    }

    #[test]
    fn test_hbm_hit_ref_bumps_existing_block() {
        // Two requests with the same prefix. The second should ref-bump
        // the first one's blocks rather than allocating fresh ones.
        let mut manager = KVCacheManager::new(16_000, 16, |t| 100 * t as u64, 0, true);
        let mut a = create_test_request("a", 32);
        a.prompt_block_hashes = vec![1, 2];
        let blocks_a = manager.allocate_blocks(&a, 32).unwrap();
        a.kv_blocks.extend(blocks_a.clone());
        let free_after_a = manager.num_free_blocks();
        assert_eq!(manager.block_ref_count(blocks_a[0]), 1);
        assert_eq!(manager.block_ref_count(blocks_a[1]), 1);

        // B has the same first two hashes; allocate_blocks should ref the
        // same physical blocks, not consume two more from the free list.
        let mut b = create_test_request("b", 32);
        b.prompt_block_hashes = vec![1, 2];
        let blocks_b = manager.allocate_blocks(&b, 32).unwrap();
        assert_eq!(blocks_b, blocks_a);
        assert_eq!(manager.num_free_blocks(), free_after_a);
        assert_eq!(manager.block_ref_count(blocks_a[0]), 2);
        assert_eq!(manager.block_ref_count(blocks_a[1]), 2);
    }

    #[test]
    fn test_shared_blocks_outlive_first_releaser() {
        let mut manager = KVCacheManager::new(16_000, 16, |t| 100 * t as u64, 0, true);
        let mut a = create_test_request("a", 16);
        a.prompt_block_hashes = vec![42];
        let blocks_a = manager.allocate_blocks(&a, 16).unwrap();
        a.kv_blocks.extend(blocks_a.clone());

        let mut b = create_test_request("b", 16);
        b.prompt_block_hashes = vec![42];
        let blocks_b = manager.allocate_blocks(&b, 16).unwrap();
        assert_eq!(blocks_b, blocks_a);
        assert_eq!(manager.block_ref_count(blocks_a[0]), 2);

        let free_before = manager.num_free_blocks();
        // A finishes and frees its blocks; ref_count goes to 1, block stays
        // out of the free list.
        manager.free_blocks(&blocks_a);
        assert_eq!(manager.num_free_blocks(), free_before);
        assert_eq!(manager.block_ref_count(blocks_a[0]), 1);
        // HBM index still holds the hash because the block is still alive.
        assert!(manager.hbm_contains(42));

        // B finishes and frees; now ref_count = 0 and block returns.
        manager.free_blocks(&blocks_b);
        assert_eq!(manager.num_free_blocks(), free_before + 1);
        assert_eq!(manager.block_ref_count(blocks_a[0]), 0);
    }

    #[test]
    fn test_in_flight_join_refs_leader_block() {
        // Set up a host-RAM tier and seed a prefix into it.
        let mut manager = KVCacheManager::new(2 * 16 * 100, 16, |t| 100 * t as u64, 0, true)
            .with_tiers(&[KVTier {
                name: "host_ram".into(),
                capacity_bytes: 8 * 16 * 100,
                bandwidth_to_hbm: 1e9,
            }]);
        let prefix_hash = 0x1234u64;
        // Get prefix_hash into host_ram by allocating then evicting.
        let mut seed = create_test_request("seed", 16);
        seed.prompt_block_hashes = vec![prefix_hash];
        let seed_blocks = manager.allocate_blocks(&seed, 16).unwrap();
        manager.free_blocks(&seed_blocks);
        // Evict by churning HBM, then free the churn blocks so we have
        // room for the leader to reserve.
        let mut churn = create_test_request("churn", 32);
        churn.prompt_block_hashes = vec![0xFFFE, 0xFFFF];
        let churn_blocks = manager.allocate_blocks(&churn, 32).unwrap();
        assert!(!manager.hbm_contains(prefix_hash));
        manager.free_blocks(&churn_blocks);

        // Leader reserves blocks for transfer.
        let mut leader = create_test_request("leader", 16);
        leader.prompt_block_hashes = vec![prefix_hash];
        let lookup_l = manager.peek_prefix_cache(&leader);
        assert!(lookup_l.needs_promotion());
        let leader_blocks = manager.reserve_blocks_for_transfer(&leader, 16).unwrap();
        manager.start_transfer("leader".into(), &lookup_l, 0.0);
        let free_after_leader = manager.num_free_blocks();
        assert_eq!(manager.block_ref_count(leader_blocks[0]), 1);
        assert!(manager.in_flight_contains(prefix_hash));

        // Joiner sees the prefix in flight; its peek should return needs_join.
        let mut joiner = create_test_request("joiner", 16);
        joiner.prompt_block_hashes = vec![prefix_hash];
        let lookup_j = manager.peek_prefix_cache(&joiner);
        assert!(lookup_j.needs_join());
        assert!(!lookup_j.needs_promotion());
        assert_eq!(lookup_j.join_leader.as_deref(), Some("leader"));

        // Reserving for joiner: ref-bumps the leader's block, no new blocks
        // taken from the free list.
        let joiner_blocks = manager.reserve_blocks_for_transfer(&joiner, 16).unwrap();
        assert_eq!(joiner_blocks, leader_blocks);
        assert_eq!(manager.num_free_blocks(), free_after_leader);
        assert_eq!(manager.block_ref_count(leader_blocks[0]), 2);
    }

    #[test]
    fn test_publish_after_transfer_makes_prefix_hbm_resident() {
        let mut manager =
            KVCacheManager::new(16_000, 16, |t| 100 * t as u64, 0, true).with_tiers(&[KVTier {
                name: "host_ram".into(),
                capacity_bytes: 16_000,
                bandwidth_to_hbm: 1e9,
            }]);
        let mut req = create_test_request("a", 16);
        req.prompt_block_hashes = vec![0xC0FFE];
        // Manually plant the hash in spillover to simulate a prior eviction.
        manager.tiers[0].insert(0xC0FFE);

        let lookup = manager.peek_prefix_cache(&req);
        assert!(lookup.needs_promotion());
        let blocks = manager.reserve_blocks_for_transfer(&req, 16).unwrap();
        manager.start_transfer("a".into(), &lookup, 0.0);
        assert!(manager.in_flight_contains(0xC0FFE));
        assert!(!manager.hbm_contains(0xC0FFE));

        let completed = manager.advance_transfers(10.0);
        assert!(completed.contains("a"));
        manager.publish_transferred_blocks(&[0xC0FFE], &blocks);
        assert!(manager.hbm_contains(0xC0FFE));
        assert!(!manager.in_flight_contains(0xC0FFE));
    }

    #[test]
    fn test_lookup_hbm_only_when_no_tiers() {
        let mut manager = KVCacheManager::new(16000, 16, |t| 100 * t as u64, 0, true);
        let mut req = create_test_request("a", 16);
        req.prompt_block_hashes = vec![100];
        let blocks = manager.allocate_blocks(&req, 16).unwrap();
        req.kv_blocks.extend(blocks);

        let mut probe = create_test_request("probe", 16);
        probe.prompt_block_hashes = vec![100];
        let lookup = manager.peek_prefix_cache(&probe);
        assert_eq!(lookup.hbm_tokens, 16);
        assert!(lookup.promote_tokens_per_tier.is_empty());
        assert!(!lookup.needs_promotion());
    }
}
