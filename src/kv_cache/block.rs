use crate::request::BlockId;

/// Represents a single KV cache block
#[derive(Debug, Clone)]
pub struct Block {
    /// Unique block ID
    pub block_id: BlockId,

    /// Reference count (number of requests using this block)
    pub ref_count: u32,

    /// Whether this block is free
    pub is_free: bool,

    /// For prefix caching: hash of the token sequence in this block
    pub content_hash: Option<u64>,
}

impl Block {
    /// Create a new free block
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            ref_count: 0,
            is_free: true,
            content_hash: None,
        }
    }

    /// Allocate this block (increment ref count, mark as not free)
    pub fn allocate(&mut self) {
        self.ref_count += 1;
        self.is_free = false;
    }

    /// Release this block (decrement ref count, mark as free if ref count reaches 0)
    pub fn release(&mut self) {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }

        if self.ref_count == 0 {
            self.is_free = true;
            self.content_hash = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let block = Block::new(0);
        assert_eq!(block.block_id, 0);
        assert_eq!(block.ref_count, 0);
        assert!(block.is_free);
        assert!(block.content_hash.is_none());
    }

    #[test]
    fn test_block_allocate() {
        let mut block = Block::new(0);

        block.allocate();
        assert_eq!(block.ref_count, 1);
        assert!(!block.is_free);

        block.allocate();
        assert_eq!(block.ref_count, 2);
        assert!(!block.is_free);
    }

    #[test]
    fn test_block_release() {
        let mut block = Block::new(0);
        block.allocate();
        block.allocate();

        block.release();
        assert_eq!(block.ref_count, 1);
        assert!(!block.is_free);

        block.release();
        assert_eq!(block.ref_count, 0);
        assert!(block.is_free);

        // Releasing when already at 0 should be safe
        block.release();
        assert_eq!(block.ref_count, 0);
        assert!(block.is_free);
    }
}
