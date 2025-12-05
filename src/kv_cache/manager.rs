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

    /// Prefix cache: hash -> block_id
    prefix_cache: HashMap<u64, BlockId>,

    /// Metrics
    pub num_prefix_cache_hits: u64,
    pub num_prefix_cache_misses: u64,
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
        }
    }

    /// Try to allocate blocks for a request
    /// Returns Some(Vec<BlockId>) if successful, None if insufficient blocks
    pub fn allocate_blocks(&mut self, request: &Request, num_tokens: u32) -> Option<Vec<BlockId>> {
        let blocks_needed = self.calculate_blocks_needed(request, num_tokens);

        if self.free_blocks.len() < blocks_needed {
            return None; // Not enough blocks
        }

        let mut allocated = Vec::new();
        for _ in 0..blocks_needed {
            let block_id = self.free_blocks.pop().unwrap();
            self.blocks[block_id as usize].allocate();
            allocated.push(block_id);
        }

        Some(allocated)
    }

    /// Calculate how many new blocks are needed for a request
    fn calculate_blocks_needed(&self, request: &Request, num_new_tokens: u32) -> usize {
        let total_tokens = request.num_computed_tokens + num_new_tokens;
        let total_blocks_needed =
            ((total_tokens + self.block_size - 1) / self.block_size) as usize;
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

    /// Get cache utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        1.0 - (self.free_blocks.len() as f64 / self.total_blocks as f64)
    }

    /// Check for prefix cache hits (simplified hash-based implementation)
    pub fn check_prefix_cache(&mut self, request: &Request) -> u32 {
        if !self.enable_prefix_caching {
            return 0;
        }

        // Simplified: hash the prompt tokens
        let prompt_hash = self.hash_prompt(&request.request_id, request.num_prompt_tokens);

        if let Some(&_block_id) = self.prefix_cache.get(&prompt_hash) {
            self.num_prefix_cache_hits += 1;
            // Return number of cached tokens
            // For simplicity, assume full blocks are cached up to block_size
            self.block_size.min(request.num_prompt_tokens)
        } else {
            self.num_prefix_cache_misses += 1;
            // Cache the prompt for future requests
            if let Some(&first_block) = request.kv_blocks.first() {
                self.prefix_cache.insert(prompt_hash, first_block);
            }
            0
        }
    }

    /// Hash a prompt for prefix caching
    fn hash_prompt(&self, request_id: &str, num_tokens: u32) -> u64 {
        // Simplified hash - in reality would hash actual token IDs
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        request_id.hash(&mut hasher);
        num_tokens.hash(&mut hasher);
        hasher.finish()
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

        let mut request1 = create_test_request("req-1", 16);
        let blocks1 = manager.allocate_blocks(&request1, 16).unwrap();
        request1.kv_blocks.extend(blocks1);

        // First check - should miss
        let cached = manager.check_prefix_cache(&request1);
        assert_eq!(cached, 0);
        assert_eq!(manager.num_prefix_cache_misses, 1);

        // Second check with same prompt - should hit
        let request2 = create_test_request("req-1", 16);
        let cached = manager.check_prefix_cache(&request2);
        assert_eq!(cached, 16);
        assert_eq!(manager.num_prefix_cache_hits, 1);
    }
}
