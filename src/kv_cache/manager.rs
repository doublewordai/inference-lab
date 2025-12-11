use super::block::Block;
use crate::request::{BlockId, Request};
use std::collections::HashMap;

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

    /// Prefix cache: maps block hash -> block_id
    /// Its a big flat hash map - the hashes are supposed to be incremental hashes of all of the
    /// tokens up to a certain point. If a sequence with block (1, 2, 3, 4, 5) is in the cache,
    /// then we get the prefix cache entry by doing cat(cache[1], cache[2], ...).
    prefix_cache: HashMap<u64, BlockId>,

    /// Metrics
    pub num_prefix_cache_hits: u64,
    pub num_prefix_cache_misses: u64,
    pub hit_size_count: u64,
    pub hit_size_sum: u64,
}

impl KVCacheManager {
    /// Create a new KV cache manager
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
            num_prefix_cache_hits: 0,
            num_prefix_cache_misses: 0,
            hit_size_count: 0,
            hit_size_sum: 0,
        }
    }

    /// Try to allocate blocks for a request
    /// Returns Some(Vec<BlockId>) if successful, None if insufficient blocks
    pub fn allocate_blocks(&mut self, request: &Request, num_tokens: u32) -> Option<Vec<BlockId>> {
        let blocks_needed = self.calculate_blocks_needed(request, num_tokens);
        let hashes = request.get_prompt_block_hashes();

        if self.free_blocks.len() < blocks_needed {
            return None; // Not enough blocks
        }

        let mut allocated = Vec::new();
        let mut evicted_hashes = Vec::new();
        for i in 0..blocks_needed {
            let block_id = self.free_blocks.pop().unwrap();
            let evicted_hash = self.blocks[block_id as usize].allocate(hashes.get(i).cloned());
            evicted_hashes.extend(evicted_hash);
            allocated.push(block_id);
        }

        // Update prefix cache with newly allocated/deallocated blocks
        if self.enable_prefix_caching {
            // remove all the content hashes that were overwritten
            for hash in evicted_hashes {
                self.prefix_cache.remove(&hash);
            }
            // Store the new block hashes
            for (i, &hash) in hashes.iter().enumerate() {
                if let Some(&block_id) = allocated.get(i) {
                    self.prefix_cache.insert(hash, block_id);
                }
            }
        };

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

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get total number of blocks
    pub fn total_blocks(&self) -> usize {
        self.total_blocks as usize
    }

    /// Get cache utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        1.0 - (self.free_blocks.len() as f64 / self.total_blocks as f64)
    }

    /// Check for prefix cache hits
    /// Returns the number of tokens that can be served from the cache
    pub fn peek_prefix_cache(&mut self, request: &Request) -> u32 {
        if !self.enable_prefix_caching {
            return 0;
        }

        // Get block hashes from the request
        let block_hashes = request.get_prompt_block_hashes();

        if block_hashes.is_empty() {
            // If there are no block hashes, then theres no caching, so don't increment anything
            return 0;
        }

        // Check consecutive blocks from the start until we find a miss
        let mut cached_blocks = 0;
        for &hash in block_hashes {
            if self.prefix_cache.contains_key(&hash) {
                cached_blocks += 1;
            } else {
                // First cache miss = end of cached prefix
                break;
            }
        }

        cached_blocks * self.block_size
    }

    pub fn query_prefix_cache(&mut self, request: &Request) -> u32 {
        let tokens = self.peek_prefix_cache(request);
        self.hit_size_count += 1;
        self.hit_size_sum += tokens as u64;

        if tokens == 0 {
            self.num_prefix_cache_misses += 1;
        } else {
            self.num_prefix_cache_hits += 1;
        }
        tokens
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
        // Create a manager with capacity for 10 blocks, block size = 16 tokens, 100 bytes per token
        let manager = KVCacheManager::new(16000, 16, 100, false);

        assert_eq!(manager.block_size, 16);
        assert_eq!(manager.total_blocks, 10); // 16000 / (16 * 100) = 10
        assert_eq!(manager.num_free_blocks(), 10);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_block_allocation() {
        let mut manager = KVCacheManager::new(16000, 16, 100, false);
        let mut request = create_test_request("req-1", 32);

        // Allocate blocks for 32 tokens (should need 2 blocks of size 16)
        let allocated = manager.allocate_blocks(&request, 32);
        assert!(allocated.is_some());

        let blocks = allocated.unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(manager.num_free_blocks(), 8);

        request.kv_blocks.extend(blocks);
        request.num_computed_tokens = 32; // Update state

        // Try to allocate more tokens for the same request
        let more_blocks = manager.allocate_blocks(&request, 16);
        assert!(more_blocks.is_some());
        assert_eq!(more_blocks.unwrap().len(), 1); // Need 1 more block
        assert_eq!(manager.num_free_blocks(), 7);
    }

    #[test]
    fn test_block_allocation_failure() {
        let mut manager = KVCacheManager::new(1600, 16, 100, false);
        // Only 1 block available
        assert_eq!(manager.total_blocks, 1);

        let request = create_test_request("req-1", 32);

        // Try to allocate 32 tokens (need 2 blocks, but only 1 available)
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

        // 2 out of 10 blocks used
        let util = manager.utilization();
        assert!((util - 0.2).abs() < 1e-10);

        manager.free_blocks(&blocks);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_prefix_caching() {
        let mut manager = KVCacheManager::new(16000, 16, 100, true);

        // Create first request with a block hash
        let mut request1 = create_test_request("req-1", 16);
        request1.prompt_block_hashes = vec![12345]; // Synthetic hash for 1 block

        // First check - should miss (hash not in cache yet)
        let cached = manager.query_prefix_cache(&request1);
        assert_eq!(cached, 0);
        assert_eq!(manager.num_prefix_cache_misses, 1);

        // Now allocate blocks (this adds to cache)
        let blocks1 = manager.allocate_blocks(&request1, 16).unwrap();
        request1.kv_blocks.extend(blocks1);

        // Second request with same block hash - should hit
        let mut request2 = create_test_request("req-2", 16);
        request2.prompt_block_hashes = vec![12345]; // Same hash = shared prefix
        let cached = manager.query_prefix_cache(&request2);
        assert_eq!(cached, 16); // 1 block * 16 tokens per block
        assert_eq!(manager.num_prefix_cache_hits, 1);

        // Third request with different hash - should miss
        let mut request3 = create_test_request("req-3", 16);
        request3.prompt_block_hashes = vec![67890]; // Different hash
        let cached = manager.query_prefix_cache(&request3);
        assert_eq!(cached, 0);
        assert_eq!(manager.num_prefix_cache_misses, 2);
    }
}
