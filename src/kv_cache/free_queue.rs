//! LRU queue of free KV blocks.
//!
//! A freed block keeps its content hash and stays hittable until it is
//! recycled, so the order blocks are recycled in is the prefix cache's
//! eviction policy. This queue recycles the least recently freed block
//! first (vLLM's `FreeKVCacheBlockQueue`), and lets a cache hit on a free
//! block pull it back out in O(1) — an intrusive doubly linked list over
//! block ids.

use crate::request::BlockId;

#[derive(Debug, Clone)]
pub struct FreeQueue {
    head: Option<BlockId>,
    tail: Option<BlockId>,
    prev: Vec<Option<BlockId>>,
    next: Vec<Option<BlockId>>,
    in_queue: Vec<bool>,
    len: usize,
}

impl FreeQueue {
    /// Every block `0..num_blocks` free, in id order.
    pub fn new(num_blocks: usize) -> Self {
        let mut q = Self {
            head: None,
            tail: None,
            prev: vec![None; num_blocks],
            next: vec![None; num_blocks],
            in_queue: vec![false; num_blocks],
            len: 0,
        };
        for id in 0..num_blocks {
            q.push_back(id as BlockId);
        }
        q
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, id: BlockId) -> bool {
        self.in_queue[id as usize]
    }

    /// Append `id` as the most recently freed block. No-op if it is already
    /// queued.
    pub fn push_back(&mut self, id: BlockId) {
        let i = id as usize;
        if self.in_queue[i] {
            return;
        }
        self.in_queue[i] = true;
        self.prev[i] = self.tail;
        self.next[i] = None;
        match self.tail {
            Some(t) => self.next[t as usize] = Some(id),
            None => self.head = Some(id),
        }
        self.tail = Some(id);
        self.len += 1;
    }

    /// The least recently freed block, if any.
    pub fn pop_front(&mut self) -> Option<BlockId> {
        let id = self.head?;
        self.remove(id);
        Some(id)
    }

    /// Unlink `id` (a cache hit on a free block). Returns whether it was
    /// queued.
    pub fn remove(&mut self, id: BlockId) -> bool {
        let i = id as usize;
        if !self.in_queue[i] {
            return false;
        }
        let (p, n) = (self.prev[i], self.next[i]);
        match p {
            Some(p) => self.next[p as usize] = n,
            None => self.head = n,
        }
        match n {
            Some(n) => self.prev[n as usize] = p,
            None => self.tail = p,
        }
        self.prev[i] = None;
        self.next[i] = None;
        self.in_queue[i] = false;
        self.len -= 1;
        true
    }

    /// The first `k` queued ids, least recently freed first.
    pub fn front(&self, k: usize) -> Vec<BlockId> {
        let mut out = Vec::with_capacity(k.min(self.len));
        let mut cur = self.head;
        while let Some(id) = cur {
            if out.len() == k {
                break;
            }
            out.push(id);
            cur = self.next[id as usize];
        }
        out
    }

    /// Queued ids, least recently freed first.
    #[cfg(test)]
    pub fn to_vec(&self) -> Vec<BlockId> {
        let mut out = Vec::with_capacity(self.len);
        let mut cur = self.head;
        while let Some(id) = cur {
            out.push(id);
            cur = self.next[id as usize];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_with_removal() {
        let mut q = FreeQueue::new(4);
        assert_eq!(q.to_vec(), vec![0, 1, 2, 3]);
        assert_eq!(q.pop_front(), Some(0));
        assert_eq!(q.pop_front(), Some(1));
        assert_eq!(q.len(), 2);
        // Free 0 again: it goes to the back.
        q.push_back(0);
        assert_eq!(q.to_vec(), vec![2, 3, 0]);
        // A hit on 3 pulls it out of the middle.
        assert!(q.remove(3));
        assert!(!q.remove(3));
        assert!(!q.contains(3));
        assert_eq!(q.to_vec(), vec![2, 0]);
        // Removing head and tail keeps the links straight.
        assert!(q.remove(2));
        assert_eq!(q.to_vec(), vec![0]);
        assert!(q.remove(0));
        assert!(q.is_empty());
        assert_eq!(q.pop_front(), None);
        q.push_back(1);
        q.push_back(1); // idempotent
        assert_eq!(q.to_vec(), vec![1]);
        assert_eq!(q.len(), 1);
    }
}

/// Recycling key of a block from its outlook: blocks with no announced
/// re-entry sort first (`0`), then re-entries farthest away first. Times
/// are non-negative, so their bit patterns order like the floats.
pub fn outlook_key(next_arrival: Option<f64>) -> u64 {
    match next_arrival {
        None => 0,
        Some(t) => u64::MAX - t.max(0.0).to_bits(),
    }
}

/// Free blocks ordered by outlook: the block recycled first is the one
/// whose re-entry is farthest away — or that has none — each sequence tail
/// first so what survives is a prefix. Blocks without an outlook keep LRU
/// order among themselves.
#[derive(Debug, Clone)]
pub struct OutlookFree {
    set: std::collections::BTreeSet<(u64, u64, BlockId)>,
    key_of: Vec<Option<(u64, u64)>>,
    seq: u64,
}

impl OutlookFree {
    pub fn new(num_blocks: usize) -> Self {
        let mut f = Self {
            set: std::collections::BTreeSet::new(),
            key_of: vec![None; num_blocks],
            seq: 0,
        };
        for id in 0..num_blocks {
            f.push_back(id as BlockId, None);
        }
        f
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    pub fn contains(&self, id: BlockId) -> bool {
        self.key_of[id as usize].is_some()
    }

    fn tiebreak(&mut self, key: u64, pos: u32) -> u64 {
        if key == 0 {
            self.seq += 1;
            self.seq
        } else {
            u64::from(u32::MAX - pos)
        }
    }

    /// Free `id` with the outlook `mark` (`(next_arrival, position in its
    /// sequence)`, `None` = no re-entry announced). No-op if already free.
    pub fn push_back(&mut self, id: BlockId, mark: Option<(f64, u32)>) {
        if self.contains(id) {
            return;
        }
        let key = outlook_key(mark.map(|m| m.0));
        let tb = self.tiebreak(key, mark.map_or(0, |m| m.1));
        self.key_of[id as usize] = Some((key, tb));
        self.set.insert((key, tb, id));
    }

    /// Change a free block's outlook in place. No-op if it is not free.
    pub fn rekey(&mut self, id: BlockId, mark: Option<(f64, u32)>) {
        if self.remove(id) {
            self.push_back(id, mark);
        }
    }

    pub fn pop_front(&mut self) -> Option<BlockId> {
        let first = *self.set.iter().next()?;
        self.set.remove(&first);
        self.key_of[first.2 as usize] = None;
        Some(first.2)
    }

    pub fn remove(&mut self, id: BlockId) -> bool {
        match self.key_of[id as usize].take() {
            Some((k, tb)) => self.set.remove(&(k, tb, id)),
            None => false,
        }
    }

    pub fn front(&self, k: usize) -> Vec<BlockId> {
        self.set.iter().take(k).map(|e| e.2).collect()
    }
}

/// The free blocks of one worker under its HBM eviction policy.
#[derive(Debug, Clone)]
pub enum FreeSet {
    Lru(FreeQueue),
    Outlook(OutlookFree),
}

impl FreeSet {
    pub fn len(&self) -> usize {
        match self {
            FreeSet::Lru(q) => q.len(),
            FreeSet::Outlook(o) => o.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: BlockId) -> bool {
        match self {
            FreeSet::Lru(q) => q.contains(id),
            FreeSet::Outlook(o) => o.contains(id),
        }
    }

    pub fn push_back(&mut self, id: BlockId, mark: Option<(f64, u32)>) {
        match self {
            FreeSet::Lru(q) => q.push_back(id),
            FreeSet::Outlook(o) => o.push_back(id, mark),
        }
    }

    /// Re-order a free block after its outlook changed (LRU: no-op).
    pub fn rekey(&mut self, id: BlockId, mark: Option<(f64, u32)>) {
        if let FreeSet::Outlook(o) = self {
            o.rekey(id, mark);
        }
    }

    pub fn pop_front(&mut self) -> Option<BlockId> {
        match self {
            FreeSet::Lru(q) => q.pop_front(),
            FreeSet::Outlook(o) => o.pop_front(),
        }
    }

    pub fn remove(&mut self, id: BlockId) -> bool {
        match self {
            FreeSet::Lru(q) => q.remove(id),
            FreeSet::Outlook(o) => o.remove(id),
        }
    }

    pub fn front(&self, k: usize) -> Vec<BlockId> {
        match self {
            FreeSet::Lru(q) => q.front(k),
            FreeSet::Outlook(o) => o.front(k),
        }
    }
}

#[cfg(test)]
mod outlook_tests {
    use super::*;

    #[test]
    fn outlook_order_is_dead_first_then_farthest_then_tail_first() {
        let mut f = OutlookFree::new(0);
        f.key_of.resize(8, None);
        f.push_back(0, Some((10.0, 0))); // near, head
        f.push_back(1, Some((10.0, 1))); // near, tail
        f.push_back(2, Some((100.0, 0))); // far, head
        f.push_back(3, Some((100.0, 1))); // far, tail
        f.push_back(4, None); // dead, freed first
        f.push_back(5, None); // dead, freed later
        let order: Vec<BlockId> = std::iter::from_fn(|| f.pop_front()).collect();
        assert_eq!(order, vec![4, 5, 3, 2, 1, 0]);
    }

    #[test]
    fn rekey_moves_a_free_block_and_hits_pull_it_out() {
        let mut f = OutlookFree::new(3);
        // All dead: LRU order 0,1,2. Announce a near re-entry for 0.
        f.rekey(0, Some((5.0, 0)));
        assert!(f.remove(1));
        assert_eq!(f.pop_front(), Some(2));
        assert_eq!(f.pop_front(), Some(0));
        assert!(f.is_empty());
        f.rekey(1, None); // not free: no-op
        assert!(!f.contains(1));
    }
}
