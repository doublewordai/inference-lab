/// One KV cache block: a reference count (requests sharing it) and, for
/// prefix caching, the content hash of the prompt prefix it holds. A block
/// is free when nothing references it; a freed block keeps its hash so it
/// can be re-hit until it is recycled.
#[derive(Debug, Clone, Default)]
pub struct Block {
    pub ref_count: u32,
    pub content_hash: Option<u64>,
    /// KV bytes this block holds for its content (from the model's KV
    /// curve at the block's position); what a write to a tier moves.
    pub content_bytes: u64,
}

impl Block {
    pub fn is_free(&self) -> bool {
        self.ref_count == 0
    }

    /// Take a fresh block for new content, returning the hash it held before
    /// (which the caller evicts from the prefix cache).
    pub fn allocate(&mut self, content_hash: Option<u64>) -> Option<u64> {
        self.ref_count += 1;
        std::mem::replace(&mut self.content_hash, content_hash)
    }

    /// Take an additional reference on a block that already holds the
    /// content the caller wants (a prefix-cache hit or an in-flight
    /// promotion), sharing the physical copy.
    pub fn reference(&mut self) {
        self.ref_count += 1;
    }

    /// Drop one reference. Returns true if the block became free.
    pub fn release(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_reference_release() {
        let mut block = Block::default();
        assert!(block.is_free());
        assert!(block.content_hash.is_none());

        assert_eq!(block.allocate(Some(7)), None);
        assert_eq!(block.ref_count, 1);
        block.reference();
        assert_eq!(block.ref_count, 2);
        assert!(!block.is_free());

        assert!(!block.release());
        assert!(block.release());
        assert!(block.is_free());
        // The hash survives until the block is recycled...
        assert_eq!(block.content_hash, Some(7));
        // ...and recycling returns it for eviction.
        assert_eq!(block.allocate(Some(9)), Some(7));
        // Releasing past zero is safe.
        block.release();
        block.release();
        assert!(block.is_free());
    }
}
