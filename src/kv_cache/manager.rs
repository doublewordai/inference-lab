use super::block::Block;
use super::link::Link;
use crate::config::KVTier;
use crate::request::{BlockId, Request};
use std::collections::{HashMap, HashSet, VecDeque};

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
    pub num_evictions: u64,
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

    fn bandwidth_to_hbm(&self) -> f64 {
        self.link.bandwidth()
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
    /// Per-spillover-tier tokens that need to be promoted (bytes/bandwidth).
    /// Indexed aligned with `KVCacheManager::tiers`.
    pub promote_tokens_per_tier: Vec<u32>,
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

/// Manages KV cache blocks for all requests
pub struct KVCacheManager {
    /// Block size in tokens
    block_size: u32,

    /// Total number of blocks available
    total_blocks: u32,

    /// All blocks
    blocks: Vec<Block>,

    /// Free blocks (indices into blocks vec)
    free_blocks: Vec<BlockId>,

    /// Enable prefix caching
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

    /// Bytes of KV per token (used to size tiers and compute promotion cost).
    kv_cache_bytes_per_token: u64,

    /// Metrics
    pub num_prefix_cache_hits: u64,
    pub num_prefix_cache_misses: u64,
    pub hit_size_count: u64,
    pub hit_size_sum: u64,
}

impl KVCacheManager {
    /// Create a new KV cache manager with no spillover hierarchy.
    pub fn new(
        kv_cache_capacity: u64,
        block_size: u32,
        kv_cache_bytes_per_token: u64,
        enable_prefix_caching: bool,
    ) -> Self {
        let bytes_per_block = block_size as u64 * kv_cache_bytes_per_token;
        let total_blocks = (kv_cache_capacity / bytes_per_block) as u32;

        let blocks = (0..total_blocks).map(Block::new).collect();
        let free_blocks = (0..total_blocks).collect();

        Self {
            block_size,
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
            kv_cache_bytes_per_token,
            num_prefix_cache_hits: 0,
            num_prefix_cache_misses: 0,
            hit_size_count: 0,
            hit_size_sum: 0,
        }
    }

    /// Attach a spillover hierarchy. Tier ordering is closest-to-HBM first.
    pub fn with_tiers(mut self, tiers: &[KVTier]) -> Self {
        let bytes_per_block = self.block_size as u64 * self.kv_cache_bytes_per_token;
        self.tiers = tiers
            .iter()
            .map(|t| {
                let capacity_blocks = if bytes_per_block == 0 {
                    0
                } else {
                    (t.capacity_bytes / bytes_per_block) as u32
                };
                SpilloverTier::new(capacity_blocks, t.bandwidth_to_hbm)
            })
            .collect();
        self
    }

    pub fn num_tiers(&self) -> usize {
        self.tiers.len()
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

    /// Try to allocate blocks for a request.
    /// Returns Some(Vec<BlockId>) if successful, None if insufficient blocks.
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
        let blocks_needed = self.calculate_blocks_needed(request, num_tokens);
        let hashes = request.get_prompt_block_hashes();

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
            let dec = match hashes.get(i).copied() {
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

        if self.free_blocks.len() < fresh_needed {
            return None;
        }

        // Pass 2: execute. Track newly-allocated (hash, block_id) pairs so
        // we can publish them to HBM or to in_flight_cache afterwards.
        let mut allocated = Vec::with_capacity(blocks_needed);
        let mut evicted_hashes = Vec::new();
        let mut newly_allocated: Vec<(u64, BlockId)> = Vec::new();
        for (i, decision) in decisions.into_iter().enumerate() {
            match decision {
                Decision::Fresh => {
                    let block_id = self.free_blocks.pop().unwrap();
                    let evicted_hash =
                        self.blocks[block_id as usize].allocate(hashes.get(i).cloned());
                    evicted_hashes.extend(evicted_hash);
                    allocated.push(block_id);
                    if let Some(&h) = hashes.get(i) {
                        newly_allocated.push((h, block_id));
                    }
                }
                Decision::RefExisting(block_id) => {
                    self.blocks[block_id as usize].reference();
                    allocated.push(block_id);
                }
            }
        }

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

    /// Calculate how many new blocks are needed for a request
    fn calculate_blocks_needed(&self, request: &Request, num_new_tokens: u32) -> usize {
        let total_tokens = request.num_computed_tokens + num_new_tokens;
        let total_blocks_needed = total_tokens.div_ceil(self.block_size) as usize;
        total_blocks_needed.saturating_sub(request.kv_blocks.len())
    }

    /// Free blocks from a request (due to preemption or completion)
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) {
        for &block_id in block_ids {
            let block = &mut self.blocks[block_id as usize];
            block.release();

            if block.is_free {
                self.free_blocks.push(block_id);
            }
        }
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks as usize
    }

    pub fn utilization(&self) -> f64 {
        1.0 - (self.free_blocks.len() as f64 / self.total_blocks as f64)
    }

    /// Look up a request's prompt against the prefix cache hierarchy.
    /// Walks the request's incremental block hashes from the start; for each
    /// block, records whether it lives in HBM, is currently in flight on
    /// behalf of another request, or sits in a spillover tier. Stops at the
    /// first block that's nowhere.
    pub fn peek_prefix_cache(&mut self, request: &Request) -> PrefixCacheLookup {
        let n_tiers = self.tiers.len();
        let mut lookup = PrefixCacheLookup {
            total_cached_tokens: 0,
            hbm_tokens: 0,
            in_flight_tokens: 0,
            promote_tokens_per_tier: vec![0u32; n_tiers],
            join_leader: None,
        };

        if !self.enable_prefix_caching {
            return lookup;
        }

        let block_hashes = request.get_prompt_block_hashes();
        if block_hashes.is_empty() {
            return lookup;
        }

        for &hash in block_hashes {
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
            let mut found_in_tier: Option<usize> = None;
            for (idx, tier) in self.tiers.iter().enumerate() {
                if tier.contains(hash) {
                    found_in_tier = Some(idx);
                    break;
                }
            }
            match found_in_tier {
                Some(idx) => {
                    lookup.promote_tokens_per_tier[idx] += self.block_size;
                    lookup.total_cached_tokens += self.block_size;
                }
                None => break,
            }
        }

        lookup
    }

    /// Compatibility shim: returns the total cached prefix length in tokens
    /// (HBM + spillover) and updates hit/miss counters.
    pub fn query_prefix_cache(&mut self, request: &Request) -> u32 {
        let lookup = self.peek_prefix_cache(request);
        let tokens = lookup.total_cached_tokens;
        self.hit_size_count += 1;
        self.hit_size_sum += tokens as u64;
        if tokens == 0 {
            self.num_prefix_cache_misses += 1;
        } else {
            self.num_prefix_cache_hits += 1;
        }
        tokens
    }

    /// Time (seconds) the promotion described by `lookup` would take if it
    /// were the *only* transfer in flight. Used as a baseline; the actual
    /// time can be longer when other transfers contend for the same tier
    /// bandwidth (see `advance_transfers`).
    pub fn promotion_time(&self, lookup: &PrefixCacheLookup) -> f64 {
        let mut t = 0.0;
        for (idx, &tokens) in lookup.promote_tokens_per_tier.iter().enumerate() {
            if tokens == 0 {
                continue;
            }
            let bytes = tokens as u64 * self.kv_cache_bytes_per_token;
            let bw = self.tiers[idx].bandwidth_to_hbm();
            if bw > 0.0 {
                t += bytes as f64 / bw;
            }
        }
        t
    }

    /// Begin tracking an in-flight transfer for `request_id`. Per-tier byte
    /// counts come from `lookup.promote_tokens_per_tier`; each non-zero tier
    /// gets a submission on its `Link` (which divides bandwidth across all
    /// its in-flight transfers). The `in_flight_cache` entries are populated
    /// at reservation time by `reserve_blocks_for_transfer`; this method only
    /// kicks off the byte-pumping side.
    pub fn start_transfer(
        &mut self,
        request_id: String,
        _prompt_block_hashes: &[u64],
        lookup: &PrefixCacheLookup,
        current_time: f64,
    ) {
        let mut active_tiers = 0u32;
        for (i, &tokens) in lookup.promote_tokens_per_tier.iter().enumerate() {
            if tokens == 0 {
                continue;
            }
            let bytes = tokens as u64 * self.kv_cache_bytes_per_token;
            self.tiers[i]
                .link
                .submit(request_id.clone(), bytes, current_time);
            active_tiers += 1;
        }
        if active_tiers > 0 {
            self.leader_active_tiers.insert(request_id.clone(), active_tiers);
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
        let mut t = 0.0;
        for tier in &self.tiers {
            t += tier.link.estimate_remaining(leader);
        }
        t
    }

    pub fn num_in_flight_transfers(&self) -> usize {
        self.leader_active_tiers.len()
    }

    /// Inspect a block's current ref count. Primarily for tests.
    pub fn block_ref_count(&self, block_id: BlockId) -> u32 {
        self.blocks[block_id as usize].ref_count
    }

    /// Whether a hash is currently resident in HBM.
    pub fn hbm_contains(&self, hash: u64) -> bool {
        self.prefix_cache.contains_key(&hash)
    }

    /// Whether a hash is currently in flight (being promoted).
    pub fn in_flight_contains(&self, hash: u64) -> bool {
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
        let manager = KVCacheManager::new(16000, 16, 100, false);
        assert_eq!(manager.block_size, 16);
        assert_eq!(manager.total_blocks, 10);
        assert_eq!(manager.num_free_blocks(), 10);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_block_allocation() {
        let mut manager = KVCacheManager::new(16000, 16, 100, false);
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
        let mut manager = KVCacheManager::new(1600, 16, 100, false);
        assert_eq!(manager.total_blocks, 1);

        let request = create_test_request("req-1", 32);
        let allocated = manager.allocate_blocks(&request, 32);
        assert!(allocated.is_none());
    }

    #[test]
    fn test_block_free() {
        let mut manager = KVCacheManager::new(16000, 16, 100, false);
        let request = create_test_request("req-1", 32);

        let blocks = manager.allocate_blocks(&request, 32).unwrap();
        assert_eq!(manager.num_free_blocks(), 8);

        manager.free_blocks(&blocks);
        assert_eq!(manager.num_free_blocks(), 10);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_utilization() {
        let mut manager = KVCacheManager::new(16000, 16, 100, false);
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
        let mut manager = KVCacheManager::new(16000, 16, 100, true);

        let mut request1 = create_test_request("req-1", 16);
        request1.prompt_block_hashes = vec![12345];

        let cached = manager.query_prefix_cache(&request1);
        assert_eq!(cached, 0);
        assert_eq!(manager.num_prefix_cache_misses, 1);

        let blocks1 = manager.allocate_blocks(&request1, 16).unwrap();
        request1.kv_blocks.extend(blocks1);

        let mut request2 = create_test_request("req-2", 16);
        request2.prompt_block_hashes = vec![12345];
        let cached = manager.query_prefix_cache(&request2);
        assert_eq!(cached, 16);
        assert_eq!(manager.num_prefix_cache_hits, 1);

        let mut request3 = create_test_request("req-3", 16);
        request3.prompt_block_hashes = vec![67890];
        let cached = manager.query_prefix_cache(&request3);
        assert_eq!(cached, 0);
        assert_eq!(manager.num_prefix_cache_misses, 2);
    }

    #[test]
    fn test_spillover_demotion_on_eviction() {
        // 2 HBM blocks, 1 host-RAM tier of 2 blocks.
        let mut manager = KVCacheManager::new(2 * 16 * 100, 16, 100, true).with_tiers(&[KVTier {
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
    fn test_promotion_time() {
        let manager = KVCacheManager::new(16000, 16, 100, true).with_tiers(&[
            KVTier {
                name: "host_ram".into(),
                capacity_bytes: 1_000_000,
                bandwidth_to_hbm: 1e10, // 10 GB/s
            },
            KVTier {
                name: "nvme".into(),
                capacity_bytes: 10_000_000,
                bandwidth_to_hbm: 1e9, // 1 GB/s
            },
        ]);

        let lookup = PrefixCacheLookup {
            total_cached_tokens: 256,
            hbm_tokens: 0,
            in_flight_tokens: 0,
            promote_tokens_per_tier: vec![128, 128], // 128 tokens from each tier
            join_leader: None,
        };
        // bytes per token = 100 in this manager
        // tier 0: 128*100 / 1e10 = 1.28e-6
        // tier 1: 128*100 / 1e9  = 1.28e-5
        let t = manager.promotion_time(&lookup);
        let expected = (128.0 * 100.0) / 1e10 + (128.0 * 100.0) / 1e9;
        assert!((t - expected).abs() < 1e-12);
    }

    #[test]
    fn test_concurrent_transfers_share_bandwidth() {
        // 1 GB/s tier; one transfer of 1 GB should take 1.0s alone, but two
        // concurrent equal-size transfers should each take ~2.0s.
        let mut manager = KVCacheManager::new(16_000, 16, 100, true).with_tiers(&[KVTier {
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
            join_leader: None,
        };

        // First transfer alone: 1 GB / 1 GB/s = 1.0s
        manager.start_transfer("a".into(), &[], &lookup, 0.0);
        let est_a_alone = manager.estimate_remaining_time("a");
        assert!((est_a_alone - 1.0).abs() < 1e-3);

        // Second transfer admitted at the same instant: each gets half
        // bandwidth, so each individually projects 2.0s remaining.
        manager.start_transfer("b".into(), &[], &lookup, 0.0);
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
        assert_eq!(manager.num_in_flight_transfers(), 0);
    }

    #[test]
    fn test_hbm_hit_ref_bumps_existing_block() {
        // Two requests with the same prefix. The second should ref-bump
        // the first one's blocks rather than allocating fresh ones.
        let mut manager = KVCacheManager::new(16_000, 16, 100, true);
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
        let mut manager = KVCacheManager::new(16_000, 16, 100, true);
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
        let mut manager = KVCacheManager::new(2 * 16 * 100, 16, 100, true).with_tiers(&[KVTier {
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
        manager.start_transfer("leader".into(), &leader.prompt_block_hashes, &lookup_l, 0.0);
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
        let mut manager = KVCacheManager::new(16_000, 16, 100, true).with_tiers(&[KVTier {
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
        manager.start_transfer("a".into(), &req.prompt_block_hashes, &lookup, 0.0);
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
        let mut manager = KVCacheManager::new(16000, 16, 100, true);
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
