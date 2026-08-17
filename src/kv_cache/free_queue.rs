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
