//! The topology's KV state as a radix tree over block hashes.
//!
//! Every request's KV is a root → leaf path of the tree; a node is a run of
//! consecutive blocks no request diverges inside. State is kept per node as
//! *ranges* rather than per block: for each worker, the HBM-resident prefix,
//! the live requests pinning it, its free portion as stamped runs, and any
//! landing (in-flight promotion) ranges; for each store, the ranges held,
//! stamped for eviction. Two requests ending at different positions of the
//! same node leave a breakpoint, so a node carries as many runs as distinct
//! ends touched it — a handful — instead of one entry per block.
//!
//! Along one path, free-time stamps are non-increasing with position (a
//! request that ends at `E` frees everything before `E` at once), so the
//! least recently freed run of a node is always its tail and eviction never
//! punches a hole in HBM. Stores may hold non-prefix ranges (write-back
//! writes tails first); a lookup only sees the prefix reachable from
//! position 0.
//!
//! Block hashes are cumulative (a hash identifies a chain position), so two
//! hash sequences agree on a prefix exactly when they agree at its last
//! position: comparisons are per node boundary, divergence points by binary
//! search, never per block.
//!
//! A lookup tier may also be peer HBM. Unlike a store it owns no ranges:
//! lookup reads sibling workers' resident prefixes directly and returns the
//! concrete source worker with each span. Fetch-source references share the
//! HBM pin boundary with live requests, so recycling cannot pass a span a
//! promotion is reading.

use crate::config::{EvictionPolicy, HbmEviction};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

pub type NodeId = u32;
pub type WorkerId = usize;
pub type StoreId = usize;
type TierRangeId = u64;

/// KV bytes of a `t`-token sequence, from the model.
pub type KvBytesFn = std::sync::Arc<dyn Fn(u32) -> u64 + Send + Sync>;

/// The tree shared by a topology's workers and its memory graph.
pub type SharedRadix = std::sync::Arc<std::sync::Mutex<Radix>>;

/// A range of one node's blocks, `[start, end)` in node-relative blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub node: NodeId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn len(&self) -> u32 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// One node of a path and how much of it the path covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seg {
    pub node: NodeId,
    /// Blocks of the node the path covers (the whole node except possibly
    /// the last segment).
    pub len: u32,
}

/// A request's KV as a path through the tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Path {
    pub segs: Vec<Seg>,
    /// Total blocks covered.
    pub blocks: u32,
}

/// A free (unpinned, resident) run of a node's blocks in a worker's HBM,
/// stamped for eviction order.
#[derive(Debug, Clone, Copy)]
struct Run {
    start: u32,
    end: u32,
    seq: u64,
    key: u64,
}

/// Blocks reserved in HBM for a promotion in flight.
#[derive(Debug, Clone)]
struct Landing {
    start: u32,
    end: u32,
    leader: String,
}

/// One worker's view of one node.
#[derive(Debug, Clone, Default)]
struct HbmState {
    /// Blocks `[0, resident)` are in HBM (fresh, hit, or promoted).
    resident: u32,
    /// Live requests ending at each node offset (`end → count`); positions
    /// below the largest end are pinned.
    refs: BTreeMap<u32, u32>,
    /// Promotion readers pinning another worker's HBM as a source. Like
    /// request references these pin a prefix: HBM residency is contiguous,
    /// so a protected middle span necessarily protects everything before it.
    fetch_refs: BTreeMap<u32, u32>,
    /// Free portion `[pinned, resident)` as runs, ascending by position,
    /// stamps non-increasing.
    runs: Vec<Run>,
    /// Ranges beyond `resident` reserved for promotions in flight.
    landing: Vec<Landing>,
    /// HBM hit counts as breakpoints `(end, count)`, ascending by end,
    /// counts non-increasing (a hit covers a prefix).
    hits: Vec<(u32, u32)>,
}

impl HbmState {
    fn pinned(&self) -> u32 {
        self.refs
            .keys()
            .chain(self.fetch_refs.keys())
            .next_back()
            .copied()
            .unwrap_or(0)
    }

    fn is_empty(&self) -> bool {
        self.resident == 0
            && self.refs.is_empty()
            && self.fetch_refs.is_empty()
            && self.landing.is_empty()
    }

    /// The landing range starting exactly at `resident`, if any.
    fn landing_at(&self, at: u32) -> Option<&Landing> {
        self.landing.iter().find(|l| l.start == at)
    }
}

/// A range a store holds of one node.
#[derive(Debug, Clone)]
struct TierRange {
    /// Stable identity in a store's eviction order. Tree compaction moves
    /// ranges between nodes, but does not change their ordering identity.
    id: TierRangeId,
    start: u32,
    end: u32,
    /// Recycling stamp: insertion order (FIFO), refreshed on promotion
    /// (LRU / TTL).
    seq: u64,
    /// Primary recycling key (outlook eviction) — 0 otherwise.
    key: u64,
    /// Last insert / promotion time (TTL).
    touched: f64,
    /// Transfer id while the write is arriving.
    arriving: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct TierState {
    /// Ascending, disjoint.
    ranges: Vec<TierRange>,
    /// Positions `< read_upto` have been promoted from this store at least
    /// once (evicting them is not dead bytes).
    read_upto: u32,
}

#[derive(Debug)]
struct Node {
    parent: Option<NodeId>,
    /// Blocks before this node along its path.
    depth: u32,
    hashes: Vec<u64>,
    /// Children by their first hash (small fan-out: a linear scan beats
    /// hashing).
    children: Vec<(u64, NodeId)>,
    hbm: FxHashMap<WorkerId, HbmState>,
    tiers: FxHashMap<StoreId, TierState>,
    /// Announced re-entry: `(next_arrival, upto)` — positions `< upto`
    /// carry it.
    outlook: Option<(f64, u32)>,
    alive: bool,
}

impl Node {
    fn len(&self) -> u32 {
        self.hashes.len() as u32
    }
    fn first_hash(&self) -> Option<u64> {
        self.hashes.first().copied()
    }
    fn child(&self, first_hash: u64) -> Option<NodeId> {
        self.children
            .iter()
            .find(|&&(h, _)| h == first_hash)
            .map(|&(_, id)| id)
    }
    /// Outlook key for positions in `[start, end)`; a range straddling the
    /// outlook boundary is split by the caller.
    fn outlook_key_at(&self, p: u32) -> (u64, Option<f64>) {
        match self.outlook {
            Some((t, upto)) if p < upto => (super::free_queue::outlook_key(Some(t)), Some(t)),
            _ => (super::free_queue::outlook_key(None), None),
        }
    }
    fn outlook_boundary(&self) -> Option<u32> {
        self.outlook.map(|(_, u)| u)
    }
    fn prunable(&self) -> bool {
        self.children.is_empty()
            && self.hbm.values().all(|h| h.is_empty())
            && self.tiers.values().all(|t| t.ranges.is_empty())
    }
}

/// Order entry: `(key, seq, node)`; the node's evictable tail run.
type HbmOrder = BTreeSet<(u64, u64, NodeId)>;

#[derive(Debug)]
struct WorkerMeta {
    total: u32,
    /// Blocks never used.
    unused: u32,
    /// Sum of free-run lengths.
    free_in_runs: u32,
    order: HbmOrder,
    seq: u64,
    policy: HbmEviction,
    backed_first: bool,
    /// The worker's lookup tiers, closest first (set by the memory graph).
    tiers: Vec<RadixTier>,
    /// Changes whenever resident HBM is recycled, invalidating an unpinned
    /// peer source's optimistic completion check.
    source_version: u64,
    /// Number of node spans currently pinned as peer-fetch sources.
    fetch_pins: u32,
}

/// Order entry: `(key, seq, range id)`. The range's current tree position is
/// deliberately kept out of this tree: radix splits and merges only update
/// [`StoreMeta::locations`].
type TierOrder = BTreeSet<(u64, u64, TierRangeId)>;

#[derive(Debug)]
struct StoreMeta {
    capacity: u32,
    held: u32,
    order: TierOrder,
    locations: FxHashMap<TierRangeId, (NodeId, u32)>,
    seq: u64,
    eviction: EvictionPolicy,
    /// Exact source spans protected by in-flight promotions. A pinned
    /// store range is skipped wholesale by eviction; this can temporarily
    /// leave the store over capacity until the reader drains.
    pins: BTreeMap<(NodeId, u32, u32), u32>,
    /// Changes whenever held content is removed or evicted.
    source_version: u64,
}

/// A tier a worker can consult during a radix lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RadixTier {
    Store(StoreId),
    /// Candidate source workers on this worker's node, in stable order.
    PeerHbm(Vec<WorkerId>),
}

/// The concrete source selected for a cached span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierSource {
    Store(StoreId),
    PeerHbm(WorkerId),
}

/// A contiguous cached span and the source that will serve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierSpan {
    pub tier: usize,
    pub span: Span,
    pub source: TierSource,
}

/// One concrete source covering an absolute range of a request prefix.
/// Absolute positions remain stable when radix nodes split or merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierSourceRange {
    pub source: TierSource,
    pub start: u32,
    pub end: u32,
}

/// What a worker sees of a request's prefix.
#[derive(Debug, Clone, Default)]
pub struct HbmLookup {
    pub path: Path,
    /// Blocks resident in this worker's HBM (prefix).
    pub hbm: u32,
    /// Blocks in flight for another request right after the resident
    /// prefix, and their leader.
    pub landing: u32,
    pub leader: Option<String>,
    /// Beyond HBM + landing: blocks held per tier (index into the worker's
    /// tier list) and their bytes.
    pub tier_blocks: Vec<u32>,
    pub tier_bytes: Vec<u64>,
    /// Blocks reachable in tiers, in order, with their concrete source.
    pub tier_spans: Vec<TierSpan>,
}

impl HbmLookup {
    pub fn cached(&self) -> u32 {
        self.hbm + self.landing + self.tier_blocks.iter().sum::<u32>()
    }
}

/// Allocation-light prefix view used when one routing decision probes many
/// workers. The matched [`Path`] is shared by the whole batch of lookups, and
/// routing only needs the aggregate tier prefix rather than promotion spans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerLookup {
    pub hbm: u32,
    pub landing: u32,
    pub tier: u32,
}

impl WorkerLookup {
    pub fn cached(&self) -> u32 {
        self.hbm + self.landing + self.tier
    }
}

#[derive(Default)]
struct LookupParts {
    hbm: u32,
    landing: u32,
    leader: Option<String>,
    tier: u32,
    tier_blocks: Vec<u32>,
    tier_bytes: Vec<u64>,
    tier_spans: Vec<TierSpan>,
}

/// A region a worker's HBM recycled.
#[derive(Debug, Clone)]
pub struct HbmEvicted {
    pub span: Span,
    pub next_arrival: Option<f64>,
}

/// Result of taking blocks for a request.
#[derive(Debug, Clone, Default)]
pub struct Acquired {
    /// Blocks newly made resident with content (published fresh blocks).
    pub produced: Vec<Span>,
    /// Regions recycled to make room.
    pub evicted: Vec<HbmEvicted>,
    /// Hit breakpoints touched: `(span, hit count after this hit)`.
    pub hits: Vec<(Span, u32)>,
    /// Fresh blocks (content or anonymous) allocated.
    pub fresh_blocks: u32,
    /// Free-but-cached blocks pulled back into use.
    pub hits_on_free: u32,
}

/// A region a store evicted or expired.
#[derive(Debug, Clone)]
pub struct TierEvicted {
    pub span: Span,
    pub write_id: Option<String>,
    /// Never promoted from the store: dead bytes.
    pub dead_bytes: u64,
    pub bytes: u64,
    pub next_arrival: Option<f64>,
}

pub struct Radix {
    nodes: Vec<Node>,
    root: NodeId,
    free_ids: Vec<NodeId>,
    workers: Vec<Option<WorkerMeta>>,
    stores: Vec<StoreMeta>,
    next_tier_range_id: TierRangeId,
    block_size: u32,
    kv_bytes_at: KvBytesFn,
    /// `kv_bytes_at(k × block_size)`, grown on demand.
    boundary_bytes: RefCell<Vec<u64>>,
    /// Capacity/eviction attempts blocked by a pinned promotion source.
    pin_stalls: u64,
    /// Current radix nodes carrying each in-flight destination landing.
    /// This keeps completion rechecks proportional to the landing path while
    /// split/merge maintenance preserves stable identities.
    landing_nodes: FxHashMap<(WorkerId, String), Vec<NodeId>>,
}

impl std::fmt::Debug for Radix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Radix")
            .field("nodes", &self.nodes.len())
            .field("workers", &self.workers.len())
            .field("stores", &self.stores.len())
            .finish()
    }
}

impl Radix {
    pub fn new(block_size: u32, kv_bytes_at: KvBytesFn) -> Self {
        let root = Node {
            parent: None,
            depth: 0,
            hashes: Vec::new(),
            children: Vec::new(),
            hbm: FxHashMap::default(),
            tiers: FxHashMap::default(),
            outlook: None,
            alive: true,
        };
        Self {
            nodes: vec![root],
            root: 0,
            free_ids: Vec::new(),
            workers: Vec::new(),
            stores: Vec::new(),
            next_tier_range_id: 0,
            block_size: block_size.max(1),
            kv_bytes_at,
            boundary_bytes: RefCell::new(vec![0]),
            pin_stalls: 0,
            landing_nodes: FxHashMap::default(),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// KV bytes at block boundary `k` (i.e. of `k × block_size` tokens).
    fn bytes_at_boundary(&self, k: u32) -> u64 {
        let mut b = self.boundary_bytes.borrow_mut();
        while b.len() <= k as usize {
            let n = b.len() as u32;
            b.push((self.kv_bytes_at)(n * self.block_size));
        }
        b[k as usize]
    }

    /// KV bytes of a `tokens`-token sequence.
    pub fn kv_bytes_for_tokens(&self, tokens: u32) -> u64 {
        (self.kv_bytes_at)(tokens)
    }

    /// Bytes of a span (positions absolute along its path).
    pub fn span_bytes(&self, span: Span) -> u64 {
        let d = self.nodes[span.node as usize].depth;
        self.bytes_at_boundary(d + span.end)
            .saturating_sub(self.bytes_at_boundary(d + span.start))
    }

    /// Bytes of block `[k, k+1)` along any path.
    pub fn block_bytes(&self, k: u32) -> u64 {
        self.bytes_at_boundary(k + 1)
            .saturating_sub(self.bytes_at_boundary(k))
    }

    // ------------------------------------------------------------------
    // Registration

    /// Register worker `w` with `total` HBM blocks.
    pub fn register_worker(
        &mut self,
        w: WorkerId,
        total: u32,
        policy: HbmEviction,
        backed_first: bool,
    ) {
        if self.workers.len() <= w {
            self.workers.resize_with(w + 1, || None);
        }
        self.workers[w] = Some(WorkerMeta {
            total,
            unused: total,
            free_in_runs: 0,
            order: BTreeSet::new(),
            seq: 0,
            policy,
            backed_first,
            tiers: Vec::new(),
            source_version: 0,
            fetch_pins: 0,
        });
    }

    pub fn has_worker(&self, w: WorkerId) -> bool {
        self.workers.get(w).is_some_and(|m| m.is_some())
    }

    pub fn set_worker_tiers(&mut self, w: WorkerId, tiers: Vec<StoreId>) {
        self.wm(w).tiers = tiers.into_iter().map(RadixTier::Store).collect();
    }

    pub fn set_worker_lookup_tiers(&mut self, w: WorkerId, tiers: Vec<RadixTier>) {
        self.wm(w).tiers = tiers;
    }

    pub fn worker_tiers(&self, w: WorkerId) -> &[RadixTier] {
        &self.w(w).tiers
    }

    pub fn pin_stalls(&self) -> u64 {
        self.pin_stalls
    }

    /// Generation of a promotion source. An unchanged generation proves no
    /// source block could have disappeared while a transfer was in flight.
    pub fn source_version(&self, source: TierSource) -> u64 {
        match source {
            TierSource::Store(store) => self.stores[store].source_version,
            TierSource::PeerHbm(worker) => self.w(worker).source_version,
        }
    }

    pub fn set_worker_hbm_policy(&mut self, w: WorkerId, policy: HbmEviction, backed_first: bool) {
        let m = self.wm(w);
        m.policy = policy;
        m.backed_first = backed_first;
    }

    /// Register a store with `capacity` blocks; returns its id.
    pub fn add_store(&mut self, capacity: u32, eviction: EvictionPolicy) -> StoreId {
        self.stores.push(StoreMeta {
            capacity,
            held: 0,
            order: BTreeSet::new(),
            locations: FxHashMap::default(),
            seq: 0,
            eviction,
            pins: BTreeMap::new(),
            source_version: 0,
        });
        self.stores.len() - 1
    }

    pub fn set_store_eviction(&mut self, s: StoreId, eviction: EvictionPolicy) {
        self.stores[s].eviction = eviction;
    }

    pub fn store_held_blocks(&self, s: StoreId) -> u32 {
        self.stores[s].held
    }

    pub fn store_capacity_blocks(&self, s: StoreId) -> u32 {
        self.stores[s].capacity
    }

    fn w(&self, w: WorkerId) -> &WorkerMeta {
        self.workers[w].as_ref().expect("registered worker")
    }
    fn wm(&mut self, w: WorkerId) -> &mut WorkerMeta {
        self.workers[w].as_mut().expect("registered worker")
    }

    /// Free blocks of worker `w`: never used plus free runs.
    pub fn free_blocks(&self, w: WorkerId) -> u32 {
        let m = self.w(w);
        m.unused + m.free_in_runs
    }

    pub fn total_blocks(&self, w: WorkerId) -> u32 {
        self.w(w).total
    }

    // ------------------------------------------------------------------
    // Paths

    /// Length of the common prefix of two cumulative-hash slices. Once a
    /// cumulative hash differs, later positions cannot match the same chain,
    /// so one last-position check and a binary search are sufficient.
    fn matching_hashes(a: &[u64], b: &[u64]) -> usize {
        let cmp = a.len().min(b.len());
        if cmp == 0 || a[0] != b[0] {
            return 0;
        }
        if a[cmp - 1] == b[cmp - 1] {
            return cmp;
        }
        let (mut lo, mut hi) = (0usize, cmp - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if a[mid] == b[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    }

    /// Longest matching prefix of `hashes` in the tree, without inserting.
    /// Node segments are compared at their last covered hash (hashes are
    /// cumulative), so a mismatch inside a node is located by binary search.
    pub fn resolve(&self, hashes: &[u64]) -> Path {
        let mut path = Path::default();
        let mut cur = self.root;
        let mut pos = 0usize;
        while pos < hashes.len() {
            let Some(child) = self.nodes[cur as usize].child(hashes[pos]) else {
                break;
            };
            let node = &self.nodes[child as usize];
            let n = node.len() as usize;
            let avail = hashes.len() - pos;
            let cmp = avail.min(n);
            let matched = if hashes[pos + cmp - 1] == node.hashes[cmp - 1] {
                cmp
            } else {
                // Diverges inside: binary search the last agreeing position.
                let (mut lo, mut hi) = (0usize, cmp - 1); // hashes[pos+lo] agrees (child key), hi disagrees
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if hashes[pos + mid] == node.hashes[mid] {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo + 1
            };
            path.segs.push(Seg {
                node: child,
                len: matched as u32,
            });
            path.blocks += matched as u32;
            pos += matched;
            if matched < n {
                break;
            }
            cur = child;
        }
        path
    }

    /// Current radix spans reserved as `leader`'s landing in the absolute
    /// request range `[from, upto)`. Landing tags move with their ranges when
    /// radix nodes split or merge, so this is a stable alternative to keeping
    /// either node ids or a copy of a potentially very long prompt.
    fn landing_spans_between(
        &self,
        w: WorkerId,
        leader: &str,
        from: u32,
        upto: u32,
    ) -> Vec<(u32, Span)> {
        let mut spans = Vec::new();
        let key = (w, leader.to_string());
        for &node_id in self.landing_nodes.get(&key).into_iter().flatten() {
            let node = &self.nodes[node_id as usize];
            let Some(hbm) = node.hbm.get(&w) else {
                continue;
            };
            for landing in hbm.landing.iter().filter(|l| l.leader == leader) {
                let start = landing.start.max(from.saturating_sub(node.depth));
                let end = landing.end.min(upto.saturating_sub(node.depth));
                if start < end {
                    spans.push((
                        node.depth + start,
                        Span {
                            node: node_id,
                            start,
                            end,
                        },
                    ));
                }
            }
        }
        spans.sort_unstable_by_key(|(start, _)| *start);
        spans
    }

    /// Absolute end of the destination blocks actually reserved for a
    /// leader. Lookup can expose the block containing the final usable token
    /// even when an already-held block means no new landing was allocated.
    pub fn landing_end(&self, w: WorkerId, leader: &str) -> Option<u32> {
        let key = (w, leader.to_string());
        self.landing_nodes
            .get(&key)
            .into_iter()
            .flatten()
            .flat_map(|&node_id| {
                let node = &self.nodes[node_id as usize];
                node.hbm.get(&w).into_iter().flat_map(move |hbm| {
                    hbm.landing
                        .iter()
                        .filter(move |landing| landing.leader == leader)
                        .map(move |landing| node.depth + landing.end)
                })
            })
            .max()
    }

    /// Re-resolve source spans through the destination's stable landing tags.
    /// Used to release pins after a tree split without retaining prompt hashes.
    pub fn landing_source_spans(
        &self,
        w: WorkerId,
        leader: &str,
        ranges: &[TierSourceRange],
    ) -> Vec<(TierSource, Span)> {
        let Some((from, upto)) = ranges
            .first()
            .zip(ranges.last())
            .map(|(first, last)| (first.start, last.end))
        else {
            return Vec::new();
        };
        let landings = self.landing_spans_between(w, leader, from, upto);
        let mut landing_index = 0usize;
        let mut out = Vec::new();
        for range in ranges {
            let mut cursor = range.start;
            while cursor < range.end {
                while landing_index < landings.len()
                    && landings[landing_index].0 + landings[landing_index].1.len() <= cursor
                {
                    landing_index += 1;
                }
                let Some(&(absolute, span)) = landings.get(landing_index) else {
                    break;
                };
                if absolute > cursor {
                    break;
                }
                let offset = cursor - absolute;
                let take = (span.len() - offset).min(range.end - cursor);
                out.push((
                    range.source,
                    Span {
                        start: span.start + offset,
                        end: span.start + offset + take,
                        ..span
                    },
                ));
                cursor += take;
            }
        }
        out
    }

    /// Surviving prefix of the ordered sources for one landing. The lookup
    /// stops at the first missing source block even if a later range remains.
    pub fn source_landing_prefix(
        &self,
        w: WorkerId,
        leader: &str,
        ranges: &[TierSourceRange],
    ) -> u32 {
        let Some((from, upto)) = ranges
            .first()
            .zip(ranges.last())
            .map(|(first, last)| (first.start, last.end))
        else {
            return 0;
        };
        let landings = self.landing_spans_between(w, leader, from, upto);
        let mut landing_index = 0usize;
        let mut survived = 0u32;
        for range in ranges {
            let mut cursor = range.start;
            while cursor < range.end {
                while landing_index < landings.len()
                    && landings[landing_index].0 + landings[landing_index].1.len() <= cursor
                {
                    landing_index += 1;
                }
                let Some(&(absolute, span)) = landings.get(landing_index) else {
                    return survived;
                };
                if absolute > cursor {
                    return survived;
                }
                let offset = cursor - absolute;
                let take = (span.len() - offset).min(range.end - cursor);
                let source_span = Span {
                    start: span.start + offset,
                    end: span.start + offset + take,
                    ..span
                };
                let count = self.source_prefix(range.source, source_span);
                survived += count;
                cursor += count;
                if count < take {
                    return survived;
                }
            }
        }
        survived
    }

    /// The path for `hashes`, inserting what is missing (splitting a node
    /// where the hashes diverge inside it).
    pub fn insert(&mut self, hashes: &[u64]) -> Path {
        let mut path = self.resolve(hashes);
        if path.blocks as usize == hashes.len() {
            return path;
        }
        // Where the match ended: either mid-node (split there) or at a node
        // boundary with no child for the next hash.
        let mut parent = self.root;
        if let Some(last) = path.segs.last().copied() {
            let node_len = self.nodes[last.node as usize].len();
            parent = if last.len < node_len {
                self.split(last.node, last.len);
                last.node
            } else {
                last.node
            };
        }
        let rest = &hashes[path.blocks as usize..];
        let leaf = self.new_node(Some(parent), rest.to_vec());
        self.nodes[parent as usize].children.push((rest[0], leaf));
        path.segs.push(Seg {
            node: leaf,
            len: rest.len() as u32,
        });
        path.blocks += rest.len() as u32;
        path
    }

    /// Rebuild `[0, upto)` through parent pointers from a cached request leaf.
    ///
    /// The leaf hint is valid only while its node is alive and the cumulative
    /// hash at `cached_blocks` still identifies the request. A split that moves
    /// that position or a merge that removes the child therefore fails this
    /// O(1) validation; splits above the leaf remain valid and the new parent
    /// chain is picked up here. `out` is cleared in place so live requests
    /// reuse their path allocation.
    pub(crate) fn rebuild_cached_path(
        &self,
        leaf: NodeId,
        cached_blocks: u32,
        cached_hash: u64,
        upto: u32,
        out: &mut Path,
    ) -> bool {
        let Some(node) = self.nodes.get(leaf as usize) else {
            return false;
        };
        let local_end = cached_blocks.saturating_sub(node.depth);
        if !node.alive
            || cached_blocks <= node.depth
            || local_end > node.len()
            || node.hashes[local_end as usize - 1] != cached_hash
            || upto > cached_blocks
        {
            return false;
        }

        out.segs.clear();
        out.blocks = upto;
        if upto == 0 {
            return true;
        }

        let mut cur = leaf;
        while cur != self.root && self.nodes[cur as usize].depth >= upto {
            cur = self.nodes[cur as usize]
                .parent
                .expect("non-root radix node has a parent");
        }
        while cur != self.root {
            let node = &self.nodes[cur as usize];
            debug_assert!(node.alive, "live leaf has a live parent chain");
            debug_assert!(node.depth < upto);
            out.segs.push(Seg {
                node: cur,
                len: (upto - node.depth).min(node.len()),
            });
            cur = node.parent.expect("non-root radix node has a parent");
        }
        out.segs.reverse();
        true
    }

    /// Insert `hashes` by continuing from a validated request leaf instead of
    /// resolving its full prefix from the root. The common decode-growth case
    /// extends a childless leaf in place; branches are followed or split from
    /// the cached position exactly as [`Self::insert`] would do.
    ///
    /// Returns `false` without changing the tree when the hint is stale, so the
    /// caller can fall back to a root-based insert.
    pub(crate) fn insert_from_cached(
        &mut self,
        leaf: NodeId,
        cached_blocks: u32,
        cached_hash: u64,
        hashes: &[u64],
        out: &mut Path,
    ) -> bool {
        if cached_blocks as usize > hashes.len()
            || !self.rebuild_cached_path(leaf, cached_blocks, cached_hash, cached_blocks, out)
        {
            return false;
        }
        if cached_blocks as usize == hashes.len() {
            return true;
        }

        let mut cur = leaf;
        let mut pos = cached_blocks as usize;
        loop {
            let local = pos as u32 - self.nodes[cur as usize].depth;
            let node_len = self.nodes[cur as usize].len();

            // A request may have ended part-way through a node which another
            // request already extended. Match that suffix before branching.
            if local < node_len {
                let available = (node_len - local) as usize;
                let wanted = hashes.len() - pos;
                let cmp = available.min(wanted);
                let node = &self.nodes[cur as usize];
                let node_hashes = &node.hashes[local as usize..local as usize + cmp];
                let request_hashes = &hashes[pos..pos + cmp];
                let matched = Self::matching_hashes(node_hashes, request_hashes);
                out.segs.last_mut().expect("cached path has a leaf").len += matched as u32;
                out.blocks += matched as u32;
                pos += matched;
                if pos == hashes.len() {
                    return true;
                }
                if local + (matched as u32) < node_len {
                    self.split(cur, local + matched as u32);
                    let rest = &hashes[pos..];
                    let new_leaf = self.new_node(Some(cur), rest.to_vec());
                    self.nodes[cur as usize].children.push((rest[0], new_leaf));
                    out.segs.push(Seg {
                        node: new_leaf,
                        len: rest.len() as u32,
                    });
                    out.blocks += rest.len() as u32;
                    return true;
                }
            }

            debug_assert_eq!(pos as u32, self.nodes[cur as usize].depth + node_len);
            let next = hashes[pos];
            let Some(child) = self.nodes[cur as usize].child(next) else {
                // Backed-first HBM recycling deliberately looks ahead by node
                // tail-runs. Keep its node boundaries stable; otherwise a
                // compacted run changes which backed block is selected. With
                // no such state, extending a childless leaf is range-equivalent.
                let append_in_place = {
                    let node = &self.nodes[cur as usize];
                    node.children.is_empty()
                        && node.tiers.is_empty()
                        && node.hbm.keys().all(|&w| !self.w(w).backed_first)
                };
                if append_in_place {
                    let rest = &hashes[pos..];
                    self.nodes[cur as usize].hashes.extend_from_slice(rest);
                    out.segs.last_mut().expect("cached path has a leaf").len += rest.len() as u32;
                    out.blocks += rest.len() as u32;
                    return true;
                }
                let rest = &hashes[pos..];
                let new_leaf = self.new_node(Some(cur), rest.to_vec());
                self.nodes[cur as usize].children.push((rest[0], new_leaf));
                out.segs.push(Seg {
                    node: new_leaf,
                    len: rest.len() as u32,
                });
                out.blocks += rest.len() as u32;
                return true;
            };

            let child_len = self.nodes[child as usize].len() as usize;
            let cmp = child_len.min(hashes.len() - pos);
            let node_hashes = &self.nodes[child as usize].hashes[..cmp];
            let request_hashes = &hashes[pos..pos + cmp];
            let matched = Self::matching_hashes(node_hashes, request_hashes);
            debug_assert!(matched > 0, "child key matched the next hash");
            out.segs.push(Seg {
                node: child,
                len: matched as u32,
            });
            out.blocks += matched as u32;
            pos += matched;
            cur = child;
            if pos == hashes.len() {
                return true;
            }
            if matched < child_len {
                self.split(child, matched as u32);
                let rest = &hashes[pos..];
                let new_leaf = self.new_node(Some(child), rest.to_vec());
                self.nodes[child as usize]
                    .children
                    .push((rest[0], new_leaf));
                out.segs.push(Seg {
                    node: new_leaf,
                    len: rest.len() as u32,
                });
                out.blocks += rest.len() as u32;
                return true;
            }
        }
    }

    fn new_node(&mut self, parent: Option<NodeId>, hashes: Vec<u64>) -> NodeId {
        let depth = parent.map_or(0, |p| {
            let pn = &self.nodes[p as usize];
            pn.depth + pn.len()
        });
        let node = Node {
            parent,
            depth,
            hashes,
            children: Vec::new(),
            hbm: FxHashMap::default(),
            tiers: FxHashMap::default(),
            outlook: None,
            alive: true,
        };
        if let Some(id) = self.free_ids.pop() {
            self.nodes[id as usize] = node;
            id
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as NodeId
        }
    }

    /// Split `node` at `k` (`0 < k < len`): the node keeps `[0, k)`, a new
    /// child takes `[k, len)` with all of the node's children and the tail
    /// of every worker's and store's state.
    fn split(&mut self, node: NodeId, k: u32) -> NodeId {
        let tail_hashes = self.nodes[node as usize].hashes.split_off(k as usize);
        let children = std::mem::take(&mut self.nodes[node as usize].children);
        let tail = self.new_node(Some(node), tail_hashes);
        for (_, c) in &children {
            self.nodes[*c as usize].parent = Some(tail);
        }
        self.nodes[tail as usize].children = children;
        let first = self.nodes[tail as usize].first_hash().unwrap();
        self.nodes[node as usize].children.push((first, tail));

        // Outlook.
        if let Some((t, upto)) = self.nodes[node as usize].outlook {
            self.nodes[node as usize].outlook = Some((t, upto.min(k)));
            if upto > k {
                self.nodes[tail as usize].outlook = Some((t, upto - k));
            }
        }

        // HBM state per worker.
        let workers: Vec<WorkerId> = self.nodes[node as usize].hbm.keys().copied().collect();
        for w in workers {
            // Drop this node's order entry; re-add both after the split.
            self.order_remove(w, node);
            let mut head = self.nodes[node as usize].hbm.remove(&w).unwrap();
            let mut tail_state = HbmState {
                resident: head.resident.saturating_sub(k),
                ..Default::default()
            };
            head.resident = head.resident.min(k);
            // Refs: ends beyond k pin the whole head and continue in the tail.
            let mut head_refs = BTreeMap::new();
            for (&end, &c) in &head.refs {
                if end > k {
                    *head_refs.entry(k).or_insert(0) += c;
                    *tail_state.refs.entry(end - k).or_insert(0) += c;
                } else {
                    *head_refs.entry(end).or_insert(0) += c;
                }
            }
            head.refs = head_refs;
            let mut head_fetch_refs = BTreeMap::new();
            for (&end, &c) in &head.fetch_refs {
                if end > k {
                    *head_fetch_refs.entry(k).or_insert(0) += c;
                    *tail_state.fetch_refs.entry(end - k).or_insert(0) += c;
                } else {
                    *head_fetch_refs.entry(end).or_insert(0) += c;
                }
            }
            head.fetch_refs = head_fetch_refs;
            let mut head_runs = Vec::new();
            for r in head.runs.drain(..) {
                if r.end <= k {
                    head_runs.push(r);
                } else if r.start >= k {
                    tail_state.runs.push(Run {
                        start: r.start - k,
                        end: r.end - k,
                        ..r
                    });
                } else {
                    head_runs.push(Run { end: k, ..r });
                    tail_state.runs.push(Run {
                        start: 0,
                        end: r.end - k,
                        ..r
                    });
                }
            }
            head.runs = head_runs;
            let mut head_landing = Vec::new();
            for l in head.landing.drain(..) {
                if l.end <= k {
                    head_landing.push(l);
                } else if l.start >= k {
                    tail_state.landing.push(Landing {
                        start: l.start - k,
                        end: l.end - k,
                        leader: l.leader,
                    });
                } else {
                    tail_state.landing.push(Landing {
                        start: 0,
                        end: l.end - k,
                        leader: l.leader.clone(),
                    });
                    head_landing.push(Landing { end: k, ..l });
                }
            }
            head.landing = head_landing;
            let mut head_hits = Vec::new();
            for (end, c) in head.hits.drain(..) {
                if end <= k {
                    head_hits.push((end, c));
                } else {
                    // Counts are non-increasing: the head's last breakpoint
                    // takes this count up to k; the tail continues.
                    if head_hits.last().is_none_or(|&(e, _)| e < k) {
                        head_hits.push((k, c));
                    }
                    tail_state.hits.push((end - k, c));
                }
            }
            head.hits = head_hits;
            self.nodes[node as usize].hbm.insert(w, head);
            if !tail_state.is_empty() || !tail_state.hits.is_empty() {
                self.nodes[tail as usize].hbm.insert(w, tail_state);
            }
            self.order_refresh(w, node);
            self.order_refresh(w, tail);
        }

        // Store state.
        let stores: Vec<StoreId> = self.nodes[node as usize].tiers.keys().copied().collect();
        for s in stores {
            let mut head = self.nodes[node as usize].tiers.remove(&s).unwrap();
            let mut tail_state = TierState {
                read_upto: head.read_upto.saturating_sub(k),
                ..Default::default()
            };
            head.read_upto = head.read_upto.min(k);
            let mut head_ranges = Vec::new();
            for r in head.ranges.drain(..) {
                if r.end <= k {
                    head_ranges.push(r);
                } else if r.start >= k {
                    let moved = TierRange {
                        start: r.start - k,
                        end: r.end - k,
                        ..r
                    };
                    self.stores[s]
                        .locations
                        .insert(moved.id, (tail, moved.start));
                    tail_state.ranges.push(moved);
                } else {
                    let new_id = self.new_tier_range_id();
                    // Preserve the old `(node, start)` tie-break for the two
                    // pieces: the lower node keeps the lower, existing id.
                    let (head_id, tail_id) = if node < tail {
                        (r.id, new_id)
                    } else {
                        (new_id, r.id)
                    };
                    let tail_range = TierRange {
                        id: tail_id,
                        start: 0,
                        end: r.end - k,
                        seq: r.seq,
                        key: r.key,
                        touched: r.touched,
                        arriving: r.arriving.clone(),
                    };
                    let head_range = TierRange {
                        id: head_id,
                        end: k,
                        ..r
                    };
                    self.stores[s]
                        .locations
                        .insert(head_range.id, (node, head_range.start));
                    self.stores[s]
                        .locations
                        .insert(tail_range.id, (tail, tail_range.start));
                    self.stores[s]
                        .order
                        .insert((tail_range.key, tail_range.seq, new_id));
                    tail_state.ranges.push(tail_range);
                    head_ranges.push(head_range);
                }
            }
            head.ranges = head_ranges;
            self.nodes[node as usize].tiers.insert(s, head);
            if !tail_state.ranges.is_empty() {
                self.nodes[tail as usize].tiers.insert(s, tail_state);
            }
            let pins: Vec<_> = self.stores[s]
                .pins
                .iter()
                .filter(|&(&(n, _, _), _)| n == node)
                .map(|(&key, &count)| (key, count))
                .collect();
            for ((_, start, end), count) in pins {
                self.stores[s].pins.remove(&(node, start, end));
                if start < k {
                    *self.stores[s]
                        .pins
                        .entry((node, start, end.min(k)))
                        .or_insert(0) += count;
                }
                if end > k {
                    *self.stores[s]
                        .pins
                        .entry((tail, start.saturating_sub(k), end - k))
                        .or_insert(0) += count;
                }
            }
        }
        for ((w, leader), nodes) in &mut self.landing_nodes {
            if !nodes.contains(&node) {
                continue;
            }
            let head_has = self.nodes[node as usize]
                .hbm
                .get(w)
                .is_some_and(|h| h.landing.iter().any(|l| l.leader == *leader));
            let tail_has = self.nodes[tail as usize]
                .hbm
                .get(w)
                .is_some_and(|h| h.landing.iter().any(|l| l.leader == *leader));
            nodes.retain(|&id| id != node);
            if head_has {
                nodes.push(node);
            }
            if tail_has {
                nodes.push(tail);
            }
            nodes.sort_unstable();
            nodes.dedup();
        }
        tail
    }

    /// Drop `node` and its ancestors while they hold no state and no
    /// children. Callers run it after a batch that may have emptied a node
    /// is fully processed (its evicted regions written on or dropped).
    pub fn prune_if_empty(&mut self, mut node: NodeId) {
        while node != self.root
            && self.nodes[node as usize].alive
            && self.nodes[node as usize].prunable()
        {
            let n = &mut self.nodes[node as usize];
            n.alive = false;
            let parent = n.parent.unwrap();
            let first = n.first_hash();
            n.hbm.clear();
            n.tiers.clear();
            n.hashes = Vec::new();
            if let Some(h) = first {
                self.nodes[parent as usize]
                    .children
                    .retain(|&(k, _)| k != h);
            }
            self.free_ids.push(node);
            node = parent;
        }
        // The last survivor may now be a node with a single child: compact.
        if node != self.root && self.nodes[node as usize].alive {
            self.try_merge_down(node);
        }
    }

    /// Merge `node` with its only child when their state is a consistent
    /// prefix (the child's HBM/store ranges continue the node's), so a chain
    /// forked at every step compacts back into one run once the side
    /// branches are gone.
    fn try_merge_down(&mut self, node: NodeId) {
        let n = &self.nodes[node as usize];
        if n.children.len() != 1 {
            return;
        }
        let child = n.children[0].1;
        let k = n.len();
        let c = &self.nodes[child as usize];
        // Outlook compatibility.
        let outlook = match (n.outlook, c.outlook) {
            (None, None) => None,
            (Some((t, u)), None) => Some((t, u)),
            (Some((t, u)), Some((t2, u2))) if u == k && t == t2 => Some((t, k + u2)),
            _ => return,
        };
        // HBM states must chain: the child's residency needs the node fully
        // resident; the node's refs at k cover the child's refs.
        let workers: Vec<WorkerId> = n.hbm.keys().chain(c.hbm.keys()).copied().collect();
        let mut workers_dedup = workers.clone();
        workers_dedup.sort_unstable();
        workers_dedup.dedup();
        for &w in &workers_dedup {
            let hp = n.hbm.get(&w);
            let hc = c.hbm.get(&w);
            let (rp, refs_at_k, fetch_refs_at_k) = hp.map_or((0, 0, 0), |h| {
                (
                    h.resident,
                    h.refs.get(&k).copied().unwrap_or(0),
                    h.fetch_refs.get(&k).copied().unwrap_or(0),
                )
            });
            if let Some(hc) = hc {
                let child_refs: u32 = hc.refs.values().sum();
                let child_fetch_refs: u32 = hc.fetch_refs.values().sum();
                if (hc.resident > 0 || !hc.landing.is_empty()) && rp < k {
                    return;
                }
                if child_refs > refs_at_k || child_fetch_refs > fetch_refs_at_k {
                    return;
                }
                if !hc.landing.is_empty() && hp.is_some_and(|h| !h.landing.is_empty()) {
                    return;
                }
            }
        }
        // Store states: only require the child's ranges to be shiftable
        // (always true).
        // Commit the merge.
        let mut cn = std::mem::replace(
            &mut self.nodes[child as usize],
            Node {
                parent: None,
                depth: 0,
                hashes: Vec::new(),
                children: Vec::new(),
                hbm: FxHashMap::default(),
                tiers: FxHashMap::default(),
                outlook: None,
                alive: false,
            },
        );
        self.free_ids.push(child);
        for &(_, gc) in &cn.children {
            self.nodes[gc as usize].parent = Some(node);
        }
        // Order entries of both nodes go, and come back for the merged one.
        for &w in &workers_dedup {
            self.order_remove(w, node);
            if let Some(hc) = cn.hbm.get(&w) {
                if let Some(tail) = hc.runs.last() {
                    self.wm(w).order.remove(&(tail.key, tail.seq, child));
                }
            }
        }
        let n = &mut self.nodes[node as usize];
        n.hashes.append(&mut cn.hashes);
        n.children = std::mem::take(&mut cn.children);
        n.outlook = outlook;
        for (w, hc) in cn.hbm.drain() {
            let hp = n.hbm.entry(w).or_default();
            if hc.resident > 0 {
                hp.resident = k + hc.resident;
            }
            let child_refs: u32 = hc.refs.values().sum();
            if child_refs > 0 {
                let e = hp.refs.get_mut(&k).unwrap();
                *e -= child_refs;
                if *e == 0 {
                    hp.refs.remove(&k);
                }
            }
            for (end, cnt) in hc.refs {
                *hp.refs.entry(k + end).or_insert(0) += cnt;
            }
            let child_fetch_refs: u32 = hc.fetch_refs.values().sum();
            if child_fetch_refs > 0 {
                let e = hp.fetch_refs.get_mut(&k).unwrap();
                *e -= child_fetch_refs;
                if *e == 0 {
                    hp.fetch_refs.remove(&k);
                }
            }
            for (end, cnt) in hc.fetch_refs {
                *hp.fetch_refs.entry(k + end).or_insert(0) += cnt;
            }
            for r in hc.runs {
                hp.runs.push(Run {
                    start: k + r.start,
                    end: k + r.end,
                    ..r
                });
            }
            for l in hc.landing {
                hp.landing.push(Landing {
                    start: k + l.start,
                    end: k + l.end,
                    leader: l.leader,
                });
            }
            for (end, cnt) in hc.hits {
                if let Some(last) = hp.hits.last_mut() {
                    if last.1 == cnt {
                        last.0 = k + end;
                        continue;
                    }
                }
                hp.hits.push((k + end, cnt));
            }
        }
        for (s, tc) in cn.tiers.drain() {
            let tp = n.tiers.entry(s).or_default();
            if tc.read_upto > 0 {
                tp.read_upto = k + tc.read_upto;
            }
            for r in tc.ranges {
                let moved = TierRange {
                    start: k + r.start,
                    end: k + r.end,
                    ..r
                };
                self.stores[s]
                    .locations
                    .insert(moved.id, (node, moved.start));
                tp.ranges.push(moved);
            }
            let pins: Vec<_> = self.stores[s]
                .pins
                .iter()
                .filter(|&(&(n, _, _), _)| n == child)
                .map(|(&key, &count)| (key, count))
                .collect();
            for ((_, start, end), count) in pins {
                self.stores[s].pins.remove(&(child, start, end));
                *self.stores[s]
                    .pins
                    .entry((node, k + start, k + end))
                    .or_insert(0) += count;
            }
        }
        for &w in &workers_dedup {
            self.order_refresh(w, node);
        }
        for nodes in self.landing_nodes.values_mut() {
            if nodes.contains(&child) {
                nodes.retain(|&id| id != child);
                nodes.push(node);
                nodes.sort_unstable();
                nodes.dedup();
            }
        }
    }

    // ------------------------------------------------------------------
    // Order bookkeeping (HBM)

    fn order_remove(&mut self, w: WorkerId, node: NodeId) {
        let Some(h) = self.nodes[node as usize].hbm.get(&w) else {
            return;
        };
        if let Some(tail) = h.runs.last() {
            let e = (tail.key, tail.seq, node);
            self.wm(w).order.remove(&e);
        }
    }

    /// (Re)insert `node`'s order entry from its tail run.
    fn order_refresh(&mut self, w: WorkerId, node: NodeId) {
        let Some(h) = self.nodes[node as usize].hbm.get(&w) else {
            return;
        };
        if let Some(tail) = h.runs.last() {
            let e = (tail.key, tail.seq, node);
            self.wm(w).order.insert(e);
        }
    }

    // ------------------------------------------------------------------
    // Lookups

    /// What worker `w` holds of `hashes`: the matched path, the resident
    /// prefix, a landing range to join, and tier-held blocks beyond.
    pub fn lookup(&self, w: WorkerId, hashes: &[u64]) -> HbmLookup {
        let path = self.resolve(hashes);
        let lk = self.lookup_resolved(w, &path, true);
        HbmLookup {
            path,
            hbm: lk.hbm,
            landing: lk.landing,
            leader: lk.leader,
            tier_blocks: lk.tier_blocks,
            tier_bytes: lk.tier_bytes,
            tier_spans: lk.tier_spans,
        }
    }

    /// Resolve `hashes` once and read every worker's HBM, landing, and
    /// aggregate tier prefix from that path. Router probes do not need an
    /// owned path or per-tier promotion details for each worker.
    pub fn lookup_workers(&self, workers: &[WorkerId], hashes: &[u64]) -> Vec<WorkerLookup> {
        let path = self.resolve(hashes);
        let mut lookups = Vec::with_capacity(workers.len());
        for &w in workers {
            let lk = self.lookup_resolved(w, &path, false);
            lookups.push(WorkerLookup {
                hbm: lk.hbm,
                landing: lk.landing,
                tier: lk.tier,
            });
        }
        lookups
    }

    fn lookup_resolved(&self, w: WorkerId, path: &Path, detailed: bool) -> LookupParts {
        let tiers = &self.w(w).tiers;
        let mut lk = LookupParts::default();
        if detailed {
            lk.tier_blocks.resize(tiers.len(), 0);
            lk.tier_bytes.resize(tiers.len(), 0);
        }
        // Resident prefix, then landing.
        let mut i = 0usize;
        while i < path.segs.len() {
            let seg = path.segs[i];
            let node = &self.nodes[seg.node as usize];
            let h = node.hbm.get(&w);
            let resident = h.map_or(0, |h| h.resident).min(seg.len);
            lk.hbm += resident;
            if resident < seg.len {
                if let Some(h) = h {
                    if let Some(l) = h.landing_at(resident) {
                        let take = l.end.min(seg.len) - resident;
                        lk.landing += take;
                        if detailed {
                            lk.leader = Some(l.leader.clone());
                        }
                        // A landing range may continue in the next segment
                        // only if it reaches this node's end.
                        if l.end >= seg.len && take + resident == seg.len {
                            i += 1;
                            // Continue landing through following segments
                            // owned by the same leader.
                            while i < path.segs.len() {
                                let s2 = path.segs[i];
                                let n2 = &self.nodes[s2.node as usize];
                                let Some(h2) = n2.hbm.get(&w) else { break };
                                if h2.resident > 0 {
                                    break;
                                }
                                let Some(l2) = h2.landing_at(0) else { break };
                                if l2.leader != l.leader {
                                    break;
                                }
                                let take2 = l2.end.min(s2.len);
                                lk.landing += take2;
                                if take2 < s2.len {
                                    break;
                                }
                                i += 1;
                            }
                            break;
                        }
                    }
                }
                break;
            }
            i += 1;
        }
        // Tier-held blocks beyond HBM + landing, walking positions.
        if tiers.is_empty() {
            return lk;
        }
        let mut pos_total = lk.hbm + lk.landing;
        // Locate the segment/offset of pos_total.
        let mut acc = 0u32;
        let mut si = 0usize;
        while si < path.segs.len() && acc + path.segs[si].len <= pos_total {
            acc += path.segs[si].len;
            si += 1;
        }
        'outer: while si < path.segs.len() {
            let seg = path.segs[si];
            let node = &self.nodes[seg.node as usize];
            let mut off = pos_total - acc;
            while off < seg.len {
                // Nearest tier holding `off`.
                let mut found: Option<(usize, u32, TierSource)> = None;
                for (ti, tier) in tiers.iter().enumerate() {
                    match tier {
                        RadixTier::Store(s) => {
                            if let Some(ts) = node.tiers.get(s) {
                                // Ranges are sorted and disjoint: the
                                // candidate is the last one starting at or
                                // before `off`.
                                let i = ts.ranges.partition_point(|r| r.start <= off);
                                if i > 0 {
                                    let r = &ts.ranges[i - 1];
                                    if off < r.end {
                                        found =
                                            Some((ti, r.end.min(seg.len), TierSource::Store(*s)));
                                    }
                                }
                            }
                        }
                        RadixTier::PeerHbm(peers) => {
                            for &peer in peers {
                                let resident =
                                    node.hbm.get(&peer).map_or(0, |h| h.resident).min(seg.len);
                                if off < resident {
                                    found = Some((ti, resident, TierSource::PeerHbm(peer)));
                                    break;
                                }
                            }
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
                let Some((ti, end, source)) = found else {
                    break 'outer;
                };
                let span = Span {
                    node: seg.node,
                    start: off,
                    end,
                };
                lk.tier += span.len();
                if detailed {
                    lk.tier_blocks[ti] += span.len();
                    lk.tier_bytes[ti] += self.span_bytes(span);
                    lk.tier_spans.push(TierSpan {
                        tier: ti,
                        span,
                        source,
                    });
                }
                off = end;
                pos_total = acc + off;
            }
            acc += seg.len;
            si += 1;
        }
        lk
    }

    /// Blocks of `hashes` resident in worker `w`'s HBM (prefix).
    pub fn resident_prefix(&self, w: WorkerId, hashes: &[u64]) -> u32 {
        let path = self.resolve(hashes);
        self.resident_prefix_resolved(w, &path)
    }

    /// Resolve once and read the HBM-resident prefix for several workers.
    pub fn resident_prefix_workers(&self, workers: &[WorkerId], hashes: &[u64]) -> Vec<u32> {
        let path = self.resolve(hashes);
        workers
            .iter()
            .map(|&w| self.resident_prefix_resolved(w, &path))
            .collect()
    }

    fn resident_prefix_resolved(&self, w: WorkerId, path: &Path) -> u32 {
        let mut n = 0;
        for seg in &path.segs {
            let r = self.nodes[seg.node as usize]
                .hbm
                .get(&w)
                .map_or(0, |h| h.resident)
                .min(seg.len);
            n += r;
            if r < seg.len {
                break;
            }
        }
        n
    }

    /// How much of `span` is still present at `source`, starting at the
    /// span's first block. Used when an unpinned promotion drains: only
    /// this surviving prefix may become resident at the destination.
    pub fn source_prefix(&self, source: TierSource, span: Span) -> u32 {
        match source {
            TierSource::PeerHbm(w) => self.nodes[span.node as usize]
                .hbm
                .get(&w)
                .map_or(0, |h| h.resident.saturating_sub(span.start))
                .min(span.len()),
            TierSource::Store(s) => {
                let Some(ts) = self.nodes[span.node as usize].tiers.get(&s) else {
                    return 0;
                };
                let mut pos = span.start;
                for r in &ts.ranges {
                    if r.end <= pos {
                        continue;
                    }
                    if r.start > pos || pos >= span.end {
                        break;
                    }
                    pos = pos.max(r.end.min(span.end));
                }
                pos.saturating_sub(span.start)
            }
        }
    }

    /// Protect a promotion source until its flow drains.
    pub fn pin_source(&mut self, source: TierSource, span: Span) {
        match source {
            TierSource::PeerHbm(w) => self.pin_hbm_source(w, span),
            TierSource::Store(s) => {
                *self.stores[s]
                    .pins
                    .entry((span.node, span.start, span.end))
                    .or_insert(0) += 1;
            }
        }
    }

    /// Release a source pin. Store pins can have left their store
    /// temporarily over capacity; return the evictions now unblocked.
    pub fn unpin_source(&mut self, source: TierSource, span: Span) -> Vec<TierEvicted> {
        match source {
            TierSource::PeerHbm(w) => {
                self.unpin_hbm_source(w, span);
                Vec::new()
            }
            TierSource::Store(s) => {
                let key = (span.node, span.start, span.end);
                if let Some(n) = self.stores[s].pins.get_mut(&key) {
                    *n -= 1;
                    if *n == 0 {
                        self.stores[s].pins.remove(&key);
                    }
                }
                Vec::new()
            }
        }
    }

    pub fn trim_store_after_unpin(&mut self, store: StoreId) -> Vec<TierEvicted> {
        self.store_trim_capacity(store)
    }

    fn pin_hbm_source(&mut self, w: WorkerId, span: Span) {
        self.order_remove(w, span.node);
        let old_pinned = self.nodes[span.node as usize]
            .hbm
            .get(&w)
            .map_or(0, HbmState::pinned);
        let mut consumed = 0u32;
        {
            let Some(h) = self.nodes[span.node as usize].hbm.get_mut(&w) else {
                return;
            };
            *h.fetch_refs.entry(span.end).or_insert(0) += 1;
            let new_pinned = h.pinned().min(h.resident);
            if new_pinned > old_pinned {
                let mut kept = Vec::with_capacity(h.runs.len());
                for r in h.runs.drain(..) {
                    let s = r.start.max(old_pinned);
                    let e = r.end.min(new_pinned);
                    if s < e {
                        consumed += e - s;
                    }
                    if r.start < s {
                        kept.push(Run { end: s, ..r });
                    }
                    if e < r.end {
                        kept.push(Run { start: e, ..r });
                    }
                }
                kept.sort_by_key(|r| r.start);
                h.runs = kept;
            }
        }
        self.wm(w).fetch_pins += 1;
        self.wm(w).free_in_runs -= consumed;
        self.order_refresh(w, span.node);
    }

    fn unpin_hbm_source(&mut self, w: WorkerId, span: Span) {
        self.order_remove(w, span.node);
        let old_pinned = self.nodes[span.node as usize]
            .hbm
            .get(&w)
            .map_or(0, HbmState::pinned);
        let seq = {
            let m = self.wm(w);
            m.seq += 1;
            m.seq
        };
        let outlook_ordered = matches!(self.w(w).policy, HbmEviction::Outlook {});
        let mut freed = 0u32;
        let mut unpinned = false;
        {
            let node = &mut self.nodes[span.node as usize];
            let boundary = outlook_ordered.then(|| node.outlook_boundary()).flatten();
            let (key_lo, key_hi) = if outlook_ordered {
                (node.outlook_key_at(0).0, node.outlook_key_at(u32::MAX).0)
            } else {
                (0, 0)
            };
            let Some(h) = node.hbm.get_mut(&w) else {
                return;
            };
            if let Some(n) = h.fetch_refs.get_mut(&span.end) {
                unpinned = true;
                *n -= 1;
                if *n == 0 {
                    h.fetch_refs.remove(&span.end);
                }
            }
            let new_pinned = h.pinned();
            let free_end = old_pinned.min(h.resident);
            if new_pinned < free_end {
                let mut pieces = Vec::new();
                match boundary {
                    Some(u) if new_pinned < u && u < free_end => {
                        pieces.push((new_pinned, u, key_lo));
                        pieces.push((u, free_end, key_hi));
                    }
                    Some(u) if free_end <= u => pieces.push((new_pinned, free_end, key_lo)),
                    _ => pieces.push((new_pinned, free_end, key_hi)),
                }
                for (start, end, key) in pieces {
                    freed += end - start;
                    h.runs.push(Run {
                        start,
                        end,
                        seq,
                        key,
                    });
                }
                h.runs.sort_by_key(|r| r.start);
            }
        }
        if unpinned {
            self.wm(w).fetch_pins -= 1;
        }
        self.wm(w).free_in_runs += freed;
        self.order_refresh(w, span.node);
    }

    /// Record an allocation attempt that could have recycled blocks but is
    /// waiting for an in-flight peer-HBM reader to release them.
    pub fn note_pin_stall(&mut self, w: WorkerId, needed: u32) {
        if self.free_blocks(w) >= needed {
            return;
        }
        if self.w(w).fetch_pins > 0 {
            self.pin_stalls += 1;
        }
    }

    // ------------------------------------------------------------------
    // Acquire / release

    /// What taking `[held, upto)` of `path` on worker `w` would cost:
    /// `(fresh blocks, hits on free-but-cached blocks)` — the blocks the
    /// free set must supply. The dry run of [`Self::acquire`]'s first pass.
    pub fn acquire_cost(&self, w: WorkerId, path: &Path, held: u32, upto: u32) -> (u32, u32) {
        let upto = upto.min(path.blocks);
        let mut fresh = 0u32;
        let mut hits_on_free = 0u32;
        let mut acc = 0u32;
        for seg in &path.segs {
            let (a, b) = (
                held.saturating_sub(acc).min(seg.len),
                upto.saturating_sub(acc).min(seg.len),
            );
            if a < b {
                let node = &self.nodes[seg.node as usize];
                let h = node.hbm.get(&w);
                let resident = h.map_or(0, |h| h.resident);
                let landing_end = h
                    .and_then(|h| h.landing_at(resident.max(a)).map(|l| l.end))
                    .unwrap_or(resident);
                let covered = resident.max(landing_end).min(b);
                if b > covered.max(a) {
                    fresh += b - covered.max(a);
                }
                if let Some(h) = h {
                    let hb = b.min(resident);
                    for r in &h.runs {
                        let s = r.start.max(a);
                        let e = r.end.min(hb);
                        if s < e {
                            hits_on_free += e - s;
                        }
                    }
                }
            }
            acc += seg.len;
        }
        (fresh, hits_on_free)
    }

    /// Take blocks `[held, upto)` of `path` for a request on worker `w`
    /// (which already holds `[0, held)`), pinning hits and allocating fresh
    /// or landing blocks. `publish`: fresh blocks are content this request
    /// produces (resident at once); else they are reserved for `leader`'s
    /// promotion. `anon`: unhashed blocks beyond the path to allocate too.
    /// `None` if the worker cannot free enough.
    #[allow(clippy::too_many_arguments)]
    pub fn acquire(
        &mut self,
        w: WorkerId,
        path: &Path,
        held: u32,
        upto: u32,
        anon: u32,
        publish: bool,
        leader: Option<&str>,
        now_seq: bool,
    ) -> Option<Acquired> {
        let _ = now_seq;
        let upto = upto.min(path.blocks);
        // Pass 1: count fresh and hits-on-free over [held, upto).
        let mut fresh = anon;
        let mut hits_on_free = 0u32;
        let mut acc = 0u32;
        for seg in &path.segs {
            let (a, b) = (
                held.saturating_sub(acc).min(seg.len),
                upto.saturating_sub(acc).min(seg.len),
            );
            if a < b {
                let node = &self.nodes[seg.node as usize];
                let h = node.hbm.get(&w);
                let resident = h.map_or(0, |h| h.resident);
                let landing_end = h
                    .and_then(|h| h.landing_at(resident.max(a)).map(|l| l.end))
                    .unwrap_or(resident);
                let covered = resident.max(landing_end).min(b);
                // Fresh: positions in [max(a, covered), b).
                if b > covered.max(a) {
                    fresh += b - covered.max(a);
                }
                // Hits on free: free runs overlapping [a, min(b, resident)).
                if let Some(h) = h {
                    let hb = b.min(resident);
                    for r in &h.runs {
                        let s = r.start.max(a);
                        let e = r.end.min(hb);
                        if s < e {
                            hits_on_free += e - s;
                        }
                    }
                }
            }
            acc += seg.len;
        }
        let needed = fresh + hits_on_free;
        if self.free_blocks(w) < needed {
            self.note_pin_stall(w, needed);
            return None;
        }
        let mut out = Acquired {
            fresh_blocks: fresh,
            hits_on_free,
            ..Default::default()
        };
        let mut landing_nodes = Vec::new();

        // Pass 2: pin refs, consume runs, record hits.
        acc = 0;
        for seg in &path.segs {
            let (a, b) = (
                held.saturating_sub(acc).min(seg.len),
                upto.saturating_sub(acc).min(seg.len),
            );
            if a < b {
                self.order_remove(w, seg.node);
                let node = &mut self.nodes[seg.node as usize];
                let h = node.hbm.entry(w).or_default();
                // Refs: move this request's end from a (if it had one) to b.
                if held > 0 && a > 0 {
                    // The request's previous end in this node was `a` if it
                    // ended here, else its end was the node length (`a`
                    // == seg.len handled by a<b false).
                    if let Some(c) = h.refs.get_mut(&a) {
                        *c -= 1;
                        if *c == 0 {
                            h.refs.remove(&a);
                        }
                    }
                }
                *h.refs.entry(b).or_insert(0) += 1;
                // Consume free runs in [a, min(b, resident)).
                let hb = b.min(h.resident);
                let mut consumed = 0u32;
                let mut kept = Vec::with_capacity(h.runs.len());
                for r in h.runs.drain(..) {
                    if r.end <= a || r.start >= hb {
                        kept.push(r);
                        continue;
                    }
                    let s = r.start.max(a);
                    let e = r.end.min(hb);
                    consumed += e - s;
                    if r.start < s {
                        kept.push(Run { end: s, ..r });
                    }
                    if e < r.end {
                        kept.push(Run { start: e, ..r });
                    }
                }
                kept.sort_by_key(|r| r.start);
                h.runs = kept;
                // Hits: breakpoints over [a, hb) — a hit covers the prefix
                // [0, hb) but positions < a were counted when first held.
                if hb > a {
                    let mut new_hits = Vec::new();
                    let mut cur = 0u32;
                    let mut old = h.hits.clone();
                    if old.last().is_none_or(|&(e, _)| e < hb) {
                        old.push((hb.max(old.last().map_or(0, |&(e, _)| e)), 0));
                    }
                    for (end, c) in old {
                        // segment (cur, end]
                        let seg_start = cur;
                        let seg_end = end;
                        if seg_end <= a || seg_start >= hb {
                            new_hits.push((seg_end, c));
                        } else {
                            // Split at a and hb.
                            if seg_start < a {
                                new_hits.push((a, c));
                            }
                            let s = seg_start.max(a);
                            let e = seg_end.min(hb);
                            new_hits.push((e, c + 1));
                            out.hits.push((
                                Span {
                                    node: seg.node,
                                    start: s,
                                    end: e,
                                },
                                c + 1,
                            ));
                            if seg_end > hb {
                                new_hits.push((seg_end, c));
                            }
                        }
                        cur = end;
                    }
                    // Merge equal adjacent counts.
                    let mut merged: Vec<(u32, u32)> = Vec::new();
                    for (e, c) in new_hits {
                        if let Some(last) = merged.last_mut() {
                            if last.1 == c {
                                last.0 = e;
                                continue;
                            }
                        }
                        merged.push((e, c));
                    }
                    h.hits = merged;
                }
                self.wm(w).free_in_runs -= consumed;
                self.order_refresh(w, seg.node);
            }
            acc += seg.len;
        }

        // Pass 3: allocate fresh blocks (evicting as needed), then extend
        // resident / landing.
        if fresh > 0 {
            let evicted = self.take_free(w, fresh);
            out.evicted = evicted;
        }
        acc = 0;
        for seg in &path.segs {
            let (a, b) = (
                held.saturating_sub(acc).min(seg.len),
                upto.saturating_sub(acc).min(seg.len),
            );
            if a < b {
                let node = &mut self.nodes[seg.node as usize];
                let h = node.hbm.entry(w).or_default();
                let resident = h.resident;
                let landing_end = h
                    .landing_at(resident.max(a))
                    .map(|l| l.end)
                    .unwrap_or(resident);
                let covered = resident.max(landing_end);
                if b > covered.max(a) {
                    let s = covered.max(a);
                    if publish {
                        // Fresh content: everything up to b becomes resident
                        // (a landing range ending at s must have landed).
                        h.resident = b;
                        out.produced.push(Span {
                            node: seg.node,
                            start: s,
                            end: b,
                        });
                    } else {
                        h.landing.push(Landing {
                            start: s,
                            end: b,
                            leader: leader.unwrap_or("").to_string(),
                        });
                        landing_nodes.push(seg.node);
                    }
                }
            }
            acc += seg.len;
        }
        if let Some(leader) = leader.filter(|_| !landing_nodes.is_empty()) {
            let nodes = self
                .landing_nodes
                .entry((w, leader.to_string()))
                .or_default();
            nodes.extend(landing_nodes);
            nodes.sort_unstable();
            nodes.dedup();
        }
        Some(out)
    }

    /// Recycle `n` blocks for worker `w`: never-used first, then the free
    /// order (LRU / outlook), tails first; with `backed_first`, a run a tier
    /// already holds within 16 blocks of the front goes before the front.
    fn take_free(&mut self, w: WorkerId, n: u32) -> Vec<HbmEvicted> {
        let mut out = Vec::new();
        let mut need = n;
        {
            let m = self.wm(w);
            let from_unused = m.unused.min(need);
            m.unused -= from_unused;
            need -= from_unused;
        }
        while need > 0 {
            let backed_first = self.w(w).backed_first;
            let tiers: Vec<StoreId> = self
                .w(w)
                .tiers
                .iter()
                .filter_map(|t| match t {
                    RadixTier::Store(s) => Some(*s),
                    RadixTier::PeerHbm(_) => None,
                })
                .collect();
            // Candidate: front of the order, or a backed run within 16 blocks.
            let mut chosen: Option<(u64, u64, NodeId)> = None;
            if backed_first && !tiers.is_empty() {
                let mut seen = 0u32;
                for &e in self.w(w).order.iter() {
                    let node = &self.nodes[e.2 as usize];
                    let h = &node.hbm[&w];
                    let tail = *h.runs.last().unwrap();
                    let backed = tiers.iter().any(|s| {
                        node.tiers.get(s).is_some_and(|ts| {
                            ts.ranges
                                .iter()
                                .any(|r| r.start < tail.end && tail.end <= r.end)
                        })
                    });
                    if backed {
                        chosen = Some(e);
                        break;
                    }
                    seen += tail.end - tail.start;
                    if seen >= 16 {
                        break;
                    }
                }
            }
            let e = match chosen.or_else(|| self.w(w).order.iter().next().copied()) {
                Some(e) => e,
                None => break, // caller checked capacity; nothing free
            };
            self.wm(w).order.remove(&e);
            let node_id = e.2;
            let node = &mut self.nodes[node_id as usize];
            let h = node.hbm.get_mut(&w).unwrap();
            let tail = h.runs.pop().unwrap();
            let take = (tail.end - tail.start).min(need);
            let evict_start = tail.end - take;
            // Must be the resident tail: runs are non-increasing in stamp
            // and free runs end at `resident`.
            debug_assert_eq!(tail.end, h.resident, "free tail run ends at resident");
            h.resident = evict_start;
            if evict_start > tail.start {
                h.runs.push(Run {
                    end: evict_start,
                    ..tail
                });
            }
            let next_arrival = node.outlook_key_at(evict_start).1;
            out.push(HbmEvicted {
                span: Span {
                    node: node_id,
                    start: evict_start,
                    end: tail.end,
                },
                next_arrival,
            });
            need -= take;
            self.wm(w).free_in_runs -= take;
            self.order_refresh(w, node_id);
            let empty = self.nodes[node_id as usize]
                .hbm
                .get(&w)
                .is_some_and(|h| h.is_empty());
            if empty {
                self.nodes[node_id as usize].hbm.remove(&w);
            }
        }
        if !out.is_empty() {
            let version = self.w(w).source_version.wrapping_add(1);
            self.wm(w).source_version = version;
        }
        out
    }

    /// A request on worker `w` holding `[0, held)` of `path` (plus `anon`
    /// unhashed blocks) is done with them: unpin, and turn what nothing
    /// else pins into free runs stamped now.
    pub fn release(&mut self, w: WorkerId, path: &Path, held: u32, anon: u32) {
        let held = held.min(path.blocks);
        {
            let m = self.wm(w);
            m.unused += anon;
        }
        // Leaf first: the tail of a chain gets the oldest stamp of this
        // release, so eviction takes it before the head.
        let mut starts = Vec::with_capacity(path.segs.len());
        let mut acc = 0u32;
        for seg in &path.segs {
            starts.push(acc);
            acc += seg.len;
        }
        for (seg, acc) in path.segs.iter().zip(starts).rev() {
            if acc >= held {
                continue;
            }
            let end_here = (held - acc).min(seg.len);
            self.order_remove(w, seg.node);
            let seq = {
                let m = self.wm(w);
                m.seq += 1;
                m.seq
            };
            let outlook_ordered = matches!(self.w(w).policy, HbmEviction::Outlook {});
            let node = &mut self.nodes[seg.node as usize];
            let outlook_boundary = if outlook_ordered {
                node.outlook_boundary()
            } else {
                None
            };
            let (key_lo, key_hi) = if outlook_ordered {
                (node.outlook_key_at(0).0, node.outlook_key_at(u32::MAX).0)
            } else {
                (0, 0)
            };
            let h = node.hbm.entry(w).or_default();
            let old_pinned = h.pinned();
            if let Some(c) = h.refs.get_mut(&end_here) {
                *c -= 1;
                if *c == 0 {
                    h.refs.remove(&end_here);
                }
            }
            let new_pinned = h.pinned();
            let freed_end = old_pinned.min(h.resident);
            let mut freed_blocks = 0u32;
            if new_pinned < freed_end {
                // New free region [new_pinned, freed_end): newest stamp, so
                // it goes at the front (positions lowest). Split at the
                // outlook boundary so keys stay per run.
                let mut pieces = Vec::new();
                match outlook_boundary {
                    Some(u) if new_pinned < u && u < freed_end => {
                        pieces.push((new_pinned, u, key_lo));
                        pieces.push((u, freed_end, key_hi));
                    }
                    Some(u) if freed_end <= u => pieces.push((new_pinned, freed_end, key_lo)),
                    _ => pieces.push((new_pinned, freed_end, key_hi)),
                }
                let mut runs = Vec::with_capacity(h.runs.len() + 2);
                for (s, e, key) in pieces {
                    freed_blocks += e - s;
                    runs.push(Run {
                        start: s,
                        end: e,
                        seq,
                        key,
                    });
                }
                runs.append(&mut h.runs);
                runs.sort_by_key(|r| r.start);
                h.runs = runs;
            }
            self.wm(w).free_in_runs += freed_blocks;
            self.order_refresh(w, seg.node);
        }
    }

    /// Landing blocks `[from, upto)` of `path` for `leader` arrived in
    /// worker `w`'s HBM: they are resident now.
    pub fn landed(&mut self, w: WorkerId, path: &Path, upto: u32) {
        let mut acc = 0u32;
        for seg in &path.segs {
            if acc >= upto {
                break;
            }
            let b = (upto - acc).min(seg.len);
            let node = &mut self.nodes[seg.node as usize];
            if let Some(h) = node.hbm.get_mut(&w) {
                let mut new_landing = Vec::new();
                for l in h.landing.drain(..) {
                    if l.start < b && l.start >= h.resident.min(l.start) {
                        // Landing range that starts within [resident, b).
                        let e = l.end.min(b);
                        if l.start <= h.resident.max(l.start) && e > h.resident {
                            h.resident = h.resident.max(e);
                        }
                        if l.end > b {
                            new_landing.push(Landing {
                                start: b,
                                end: l.end,
                                leader: l.leader,
                            });
                        }
                    } else {
                        new_landing.push(l);
                    }
                }
                h.landing = new_landing;
                // Contiguity: resident cannot exceed what is landed;
                // ensure resident >= b if the path was landing up to b.
                if h.resident < b && h.landing.iter().all(|l| l.start >= b) {
                    h.resident = h.resident.max(b.min(h.resident.max(b)));
                }
            }
            acc += seg.len;
        }
    }

    /// Complete a promotion whose reservation covered `reserved` blocks but
    /// whose source only survived through `upto`. The surviving prefix is
    /// landed; the suffix reservation and this request's references to it
    /// are released as blank capacity rather than publishing stale KV.
    pub fn landed_partial(
        &mut self,
        w: WorkerId,
        path: &Path,
        leader: &str,
        reserved: u32,
        upto: u32,
    ) {
        let reserved = reserved.min(path.blocks);
        let upto = upto.min(reserved);
        self.landed(w, path, upto);
        if upto == reserved {
            self.landing_nodes.remove(&(w, leader.to_string()));
            return;
        }
        let mut returned = 0u32;
        let mut acc = 0u32;
        for seg in &path.segs {
            let old_end = reserved.saturating_sub(acc).min(seg.len);
            let new_end = upto.saturating_sub(acc).min(seg.len);
            acc += seg.len;
            if new_end == old_end {
                continue;
            }
            self.order_remove(w, seg.node);
            if let Some(h) = self.nodes[seg.node as usize].hbm.get_mut(&w) {
                if old_end > 0 {
                    if let Some(n) = h.refs.get_mut(&old_end) {
                        *n -= 1;
                        if *n == 0 {
                            h.refs.remove(&old_end);
                        }
                    }
                }
                if new_end > 0 {
                    *h.refs.entry(new_end).or_insert(0) += 1;
                }
                h.landing
                    .retain(|l| !(l.leader == leader && l.start >= new_end && l.start < old_end));
                returned += old_end - new_end;
            }
            self.order_refresh(w, seg.node);
        }
        self.wm(w).unused += returned;
        self.landing_nodes.remove(&(w, leader.to_string()));
    }

    /// Whether worker `w` has a landing range for `leader` on `path`.
    pub fn has_landing(&self, w: WorkerId, path: &Path, leader: &str) -> bool {
        path.segs.iter().any(|seg| {
            self.nodes[seg.node as usize]
                .hbm
                .get(&w)
                .is_some_and(|h| h.landing.iter().any(|l| l.leader == leader))
        })
    }

    // ------------------------------------------------------------------
    // Outlook

    /// Announce that positions `[0, shared)` of `path` re-enter at
    /// `next_arrival` (`None`: the trajectory is over — clear marks). Free
    /// runs and store ranges of those nodes are re-keyed.
    pub fn set_outlook(&mut self, path: &Path, next_arrival: Option<f64>, shared: u32) {
        let mut acc = 0u32;
        for seg in &path.segs {
            let node_id = seg.node;
            let upto = shared.saturating_sub(acc).min(seg.len);
            let mark = next_arrival.filter(|_| upto > 0).map(|t| (t, upto));
            // Only positions this path covers are re-marked; a node's
            // outlook covers its prefix, so a shorter mark shrinks it.
            let node = &mut self.nodes[node_id as usize];
            // A `None` mark clears the node's outlook outright: the
            // trajectory is over (or the step announces nothing).
            node.outlook = mark;
            // Re-key HBM runs on every outlook-ordered worker.
            let workers: Vec<WorkerId> = self.nodes[node_id as usize].hbm.keys().copied().collect();
            for w in workers {
                if !matches!(self.w(w).policy, HbmEviction::Outlook {}) {
                    continue;
                }
                self.order_remove(w, node_id);
                let node = &mut self.nodes[node_id as usize];
                let boundary = node.outlook_boundary();
                let (klo, khi) = (node.outlook_key_at(0).0, node.outlook_key_at(u32::MAX).0);
                let h = node.hbm.get_mut(&w).unwrap();
                let mut runs = Vec::with_capacity(h.runs.len() + 1);
                for r in h.runs.drain(..) {
                    match boundary {
                        Some(u) if r.start < u && u < r.end => {
                            runs.push(Run {
                                end: u,
                                key: klo,
                                ..r
                            });
                            runs.push(Run {
                                start: u,
                                key: khi,
                                ..r
                            });
                        }
                        Some(u) if r.end <= u => runs.push(Run { key: klo, ..r }),
                        _ => runs.push(Run { key: khi, ..r }),
                    }
                }
                h.runs = runs;
                self.order_refresh(w, node_id);
            }
            // Re-key store ranges.
            let stores: Vec<StoreId> = self.nodes[node_id as usize].tiers.keys().copied().collect();
            for s in stores {
                if !matches!(self.stores[s].eviction, EvictionPolicy::Outlook {}) {
                    continue;
                }
                let boundary = self.nodes[node_id as usize].outlook_boundary();
                let split_id = boundary
                    .filter(|&u| {
                        self.nodes[node_id as usize].tiers[&s]
                            .ranges
                            .iter()
                            .any(|r| r.start < u && u < r.end)
                    })
                    .map(|_| self.new_tier_range_id());
                let node = &mut self.nodes[node_id as usize];
                let (klo, khi) = (node.outlook_key_at(0).0, node.outlook_key_at(u32::MAX).0);
                let ts = node.tiers.get_mut(&s).unwrap();
                let mut ranges = Vec::with_capacity(ts.ranges.len() + 1);
                let mut removed = Vec::new();
                let mut added = Vec::new();
                let mut locations = Vec::new();
                for r in ts.ranges.drain(..) {
                    let old = (r.key, r.seq, r.id);
                    match boundary {
                        Some(u) if r.start < u && u < r.end => {
                            let a = TierRange {
                                end: u,
                                key: klo,
                                arriving: r.arriving.clone(),
                                ..r
                            };
                            let b = TierRange {
                                id: split_id.expect("one range straddles an outlook boundary"),
                                start: u,
                                key: khi,
                                ..r
                            };
                            let a_entry = (a.key, a.seq, a.id);
                            if old != a_entry {
                                removed.push(old);
                                added.push(a_entry);
                            }
                            added.push((b.key, b.seq, b.id));
                            locations.push((b.id, (node_id, b.start)));
                            ranges.push(a);
                            ranges.push(b);
                        }
                        Some(u) if r.end <= u => {
                            let a = TierRange { key: klo, ..r };
                            let new = (a.key, a.seq, a.id);
                            if old != new {
                                removed.push(old);
                                added.push(new);
                            }
                            ranges.push(a);
                        }
                        _ => {
                            let a = TierRange { key: khi, ..r };
                            let new = (a.key, a.seq, a.id);
                            if old != new {
                                removed.push(old);
                                added.push(new);
                            }
                            ranges.push(a);
                        }
                    }
                }
                ts.ranges = ranges;
                for e in removed {
                    self.stores[s].order.remove(&e);
                }
                for e in added {
                    self.stores[s].order.insert(e);
                }
                for (id, location) in locations {
                    self.stores[s].locations.insert(id, location);
                }
            }
            acc += seg.len;
        }
    }

    /// The announced re-entry at the first block of `hashes`, if any.
    pub fn outlook_of(&self, hashes: &[u64]) -> Option<f64> {
        let path = self.resolve(hashes);
        let seg = path.segs.first()?;
        let node = &self.nodes[seg.node as usize];
        node.outlook_key_at(0).1
    }

    // ------------------------------------------------------------------
    // Stores

    fn new_tier_range_id(&mut self) -> TierRangeId {
        let id = self.next_tier_range_id;
        self.next_tier_range_id = self
            .next_tier_range_id
            .checked_add(1)
            .expect("tier range id overflow");
        id
    }

    fn store_key(&self, s: StoreId, node: NodeId, p: u32) -> u64 {
        match self.stores[s].eviction {
            EvictionPolicy::Outlook {} => self.nodes[node as usize].outlook_key_at(p).0,
            _ => 0,
        }
    }

    /// Whether store `s` holds any of `span`'s positions.
    pub fn store_holds(&self, s: StoreId, span: Span) -> bool {
        self.nodes[span.node as usize]
            .tiers
            .get(&s)
            .is_some_and(|ts| {
                ts.ranges
                    .iter()
                    .any(|r| r.start < span.end && span.start < r.end)
            })
    }

    /// Sub-ranges of `span` that store `s` does not hold.
    pub fn store_missing(&self, s: StoreId, span: Span) -> Vec<Span> {
        let mut out = Vec::new();
        let mut p = span.start;
        let Some(ts) = self.nodes[span.node as usize].tiers.get(&s) else {
            return vec![span];
        };
        for r in &ts.ranges {
            if r.end <= p {
                continue;
            }
            if r.start >= span.end {
                break;
            }
            if r.start > p {
                out.push(Span {
                    node: span.node,
                    start: p,
                    end: r.start.min(span.end),
                });
            }
            p = p.max(r.end);
        }
        if p < span.end {
            out.push(Span {
                node: span.node,
                start: p,
                end: span.end,
            });
        }
        out
    }

    /// Put `span` into store `s` (`arriving`: the write's transfer id, or
    /// resident at once). Positions the store already holds are skipped.
    /// Evicts by the store's policy to stay within capacity; returns the
    /// evicted regions.
    pub fn store_insert(
        &mut self,
        s: StoreId,
        span: Span,
        arriving: Option<String>,
        now: f64,
    ) -> Vec<TierEvicted> {
        let mut evicted = Vec::new();
        if self.stores[s].capacity == 0 {
            evicted.push(TierEvicted {
                span,
                write_id: arriving,
                dead_bytes: 0,
                bytes: self.span_bytes(span),
                next_arrival: None,
            });
            return evicted;
        }
        let missing = self.store_missing(s, span);
        for m in missing {
            // Split at the outlook boundary so a range has one key.
            let boundary = self.nodes[m.node as usize].outlook_boundary();
            let pieces: Vec<Span> = match boundary {
                Some(u) if m.start < u && u < m.end => {
                    vec![Span { end: u, ..m }, Span { start: u, ..m }]
                }
                _ => vec![m],
            };
            for piece in pieces {
                let seq = {
                    let sm = &mut self.stores[s];
                    sm.seq += 1;
                    sm.seq
                };
                let id = self.new_tier_range_id();
                let key = self.store_key(s, piece.node, piece.start);
                let ts = self.nodes[piece.node as usize].tiers.entry(s).or_default();
                let r = TierRange {
                    id,
                    start: piece.start,
                    end: piece.end,
                    seq,
                    key,
                    touched: now,
                    arriving: arriving.clone(),
                };
                let at = ts.ranges.partition_point(|x| x.start < r.start);
                ts.ranges.insert(at, r);
                self.stores[s].order.insert((key, seq, id));
                self.stores[s]
                    .locations
                    .insert(id, (piece.node, piece.start));
                self.stores[s].held += piece.len();
            }
        }
        evicted.extend(self.store_trim_capacity(s));
        evicted
    }

    fn store_range_pinned(&self, s: StoreId, node: NodeId, start: u32, end: u32) -> bool {
        self.stores[s]
            .pins
            .iter()
            .any(|(&(n, a, b), &count)| count > 0 && n == node && a < end && start < b)
    }

    /// Evict until `s` meets its capacity, skipping every range protected by
    /// an in-flight reader. If all candidates are pinned, the over-capacity
    /// state is retained and completed when the last relevant pin releases.
    fn store_trim_capacity(&mut self, s: StoreId) -> Vec<TierEvicted> {
        let mut evicted = Vec::new();
        while self.stores[s].held > self.stores[s].capacity {
            let chosen = if self.stores[s].pins.is_empty() {
                self.stores[s].order.iter().next().copied()
            } else {
                self.stores[s].order.iter().copied().find(|e| {
                    let Some(&(node, start)) = self.stores[s].locations.get(&e.2) else {
                        return true;
                    };
                    let end = self.nodes[node as usize]
                        .tiers
                        .get(&s)
                        .and_then(|ts| ts.ranges.iter().find(|r| r.id == e.2))
                        .map_or(start, |r| r.end);
                    !self.store_range_pinned(s, node, start, end)
                })
            };
            let Some(e) = chosen else {
                self.pin_stalls += 1;
                break;
            };
            let excess = self.stores[s].held - self.stores[s].capacity;
            if let Some(ev) = self.store_evict_entry(s, e, excess) {
                evicted.push(ev);
            }
        }
        evicted
    }

    /// Evict up to `max` blocks from the tail of the range at order entry
    /// `e` in store `s`.
    fn store_evict_entry(
        &mut self,
        s: StoreId,
        e: (u64, u64, TierRangeId),
        max: u32,
    ) -> Option<TierEvicted> {
        let (_, _, id) = e;
        let Some(&(node_id, start)) = self.stores[s].locations.get(&id) else {
            self.stores[s].order.remove(&e);
            return None;
        };
        let Some(idx) = self.nodes[node_id as usize]
            .tiers
            .get(&s)
            .and_then(|ts| ts.ranges.iter().position(|r| r.id == id))
        else {
            self.stores[s].order.remove(&e);
            self.stores[s].locations.remove(&id);
            return None;
        };
        let node = &mut self.nodes[node_id as usize];
        let ts = node.tiers.get_mut(&s).unwrap();
        let r = ts.ranges[idx].clone();
        debug_assert_eq!(r.start, start);
        let take = (r.end - r.start).min(max.max(1));
        let ev_start = r.end - take;
        let write_id = r.arriving.clone();
        if ev_start > r.start {
            ts.ranges[idx].end = ev_start;
        } else {
            ts.ranges.remove(idx);
            self.stores[s].order.remove(&e);
            self.stores[s].locations.remove(&id);
        }
        let read_upto = ts.read_upto;
        let span = Span {
            node: node_id,
            start: ev_start,
            end: r.end,
        };
        let dead_span = Span {
            node: node_id,
            start: ev_start.max(read_upto),
            end: r.end,
        };
        self.stores[s].held -= take;
        self.stores[s].source_version = self.stores[s].source_version.wrapping_add(1);
        let bytes = self.span_bytes(span);
        let dead_bytes = if dead_span.is_empty() {
            0
        } else {
            self.span_bytes(dead_span)
        };
        let next_arrival = self.nodes[node_id as usize].outlook_key_at(ev_start).1;
        let empty = self.nodes[node_id as usize]
            .tiers
            .get(&s)
            .is_some_and(|t| t.ranges.is_empty());
        if empty {
            self.nodes[node_id as usize].tiers.remove(&s);
        }
        Some(TierEvicted {
            span,
            write_id,
            dead_bytes,
            bytes,
            next_arrival,
        })
    }

    /// The write `write_id` landed: its ranges in store `s` are resident.
    pub fn store_landed(&mut self, s: StoreId, spans: &[Span]) {
        for sp in spans {
            if let Some(ts) = self.nodes[sp.node as usize].tiers.get_mut(&s) {
                for r in &mut ts.ranges {
                    if r.start >= sp.start && r.end <= sp.end {
                        r.arriving = None;
                    }
                }
            }
        }
    }

    /// Whether any of `span` in store `s` is still arriving; returns the
    /// write ids.
    pub fn store_arriving(&self, s: StoreId, span: Span) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(ts) = self.nodes[span.node as usize].tiers.get(&s) {
            for r in &ts.ranges {
                if r.start < span.end && span.start < r.end {
                    if let Some(id) = &r.arriving {
                        out.push(id.clone());
                    }
                }
            }
        }
        out
    }

    /// `[0, upto)` of `path` was promoted from store `s`: mark read, and
    /// refresh recency under LRU / TTL (splitting a range at `upto` so the
    /// untouched tail keeps its stamp).
    pub fn store_promoted(&mut self, s: StoreId, spans: &[Span], now: f64) -> u64 {
        let refresh = !matches!(self.stores[s].eviction, EvictionPolicy::Fifo {});
        let mut read_bytes = 0u64;
        for sp in spans {
            let node_id = sp.node;
            let depth = self.nodes[node_id as usize].depth;
            let Some(ts) = self.nodes[node_id as usize].tiers.get_mut(&s) else {
                continue;
            };
            // Bytes of the held part of `sp`, and whether anything overlaps.
            let mut overlap = false;
            let mut held_ranges: Vec<(u32, u32)> = Vec::new();
            for r in &ts.ranges {
                if r.end <= sp.start || r.start >= sp.end {
                    continue;
                }
                overlap = true;
                held_ranges.push((r.start.max(sp.start), r.end.min(sp.end)));
            }
            if !overlap {
                continue;
            }
            ts.read_upto = ts.read_upto.max(sp.end);
            for (a, b) in held_ranges {
                read_bytes += self
                    .bytes_at_boundary(depth + b)
                    .saturating_sub(self.bytes_at_boundary(depth + a));
            }
            if !refresh {
                continue;
            }
            // Common case: one range, wholly inside the span — re-stamp in
            // place.
            let overlapping: Vec<usize> = self.nodes[node_id as usize].tiers[&s]
                .ranges
                .iter()
                .enumerate()
                .filter(|(_, r)| r.start < sp.end && sp.start < r.end)
                .map(|(i, _)| i)
                .collect();
            if overlapping.len() == 1 {
                let i = overlapping[0];
                let r = &self.nodes[node_id as usize].tiers[&s].ranges[i];
                if r.start >= sp.start && r.end <= sp.end {
                    let old = (r.key, r.seq, r.id);
                    self.stores[s].seq += 1;
                    let seq = self.stores[s].seq;
                    let ts = self.nodes[node_id as usize].tiers.get_mut(&s).unwrap();
                    let r = &mut ts.ranges[i];
                    r.seq = seq;
                    r.touched = now;
                    let new = (r.key, r.seq, r.id);
                    self.stores[s].order.remove(&old);
                    self.stores[s].order.insert(new);
                    continue;
                }
            }
            // Splitting retains the leftmost piece's identity. Allocate ids
            // only for the additional pieces before borrowing the ranges.
            let new_id_count: usize = self.nodes[node_id as usize].tiers[&s]
                .ranges
                .iter()
                .filter(|r| r.start < sp.end && sp.start < r.end)
                .map(|r| usize::from(r.start < sp.start) + usize::from(r.end > sp.end))
                .sum();
            let mut new_ids = (0..new_id_count)
                .map(|_| self.new_tier_range_id())
                .collect::<Vec<_>>()
                .into_iter();
            let ts = self.nodes[node_id as usize].tiers.get_mut(&s).unwrap();
            let mut ranges = Vec::with_capacity(ts.ranges.len() + 1);
            let mut removed = Vec::new();
            let mut added = Vec::new();
            let mut location_add = Vec::new();
            let mut location_remove = Vec::new();
            // One stamp per promotion: the touched ranges coalesce.
            self.stores[s].seq += 1;
            let seq = self.stores[s].seq;
            for r in ts.ranges.drain(..) {
                if r.end <= sp.start || r.start >= sp.end {
                    ranges.push(r);
                    continue;
                }
                let old = (r.key, r.seq, r.id);
                if r.start < sp.start {
                    let head = TierRange {
                        end: sp.start,
                        arriving: r.arriving.clone(),
                        ..r
                    };
                    ranges.push(head);
                    let mid = TierRange {
                        id: new_ids.next().unwrap(),
                        start: sp.start,
                        end: r.end.min(sp.end),
                        seq,
                        touched: now,
                        arriving: r.arriving.clone(),
                        ..r
                    };
                    added.push((mid.key, mid.seq, mid.id));
                    location_add.push((mid.id, (node_id, mid.start)));
                    ranges.push(mid);
                    if r.end > sp.end {
                        let tail = TierRange {
                            id: new_ids.next().unwrap(),
                            start: sp.end,
                            ..r
                        };
                        added.push((tail.key, tail.seq, tail.id));
                        location_add.push((tail.id, (node_id, tail.start)));
                        ranges.push(tail);
                    }
                } else {
                    let mid = TierRange {
                        end: r.end.min(sp.end),
                        seq,
                        touched: now,
                        arriving: r.arriving.clone(),
                        ..r
                    };
                    removed.push(old);
                    added.push((mid.key, mid.seq, mid.id));
                    ranges.push(mid);
                    if r.end > sp.end {
                        let tail = TierRange {
                            id: new_ids.next().unwrap(),
                            start: sp.end,
                            ..r
                        };
                        added.push((tail.key, tail.seq, tail.id));
                        location_add.push((tail.id, (node_id, tail.start)));
                        ranges.push(tail);
                    }
                }
            }
            debug_assert!(new_ids.next().is_none());
            ranges.sort_by_key(|r| r.start);
            // Merge adjacent ranges with identical stamps.
            let mut merged: Vec<TierRange> = Vec::with_capacity(ranges.len());
            for r in ranges {
                if let Some(last) = merged.last_mut() {
                    if last.end == r.start
                        && last.seq == r.seq
                        && last.key == r.key
                        && last.arriving == r.arriving
                    {
                        let redundant = (r.key, r.seq, r.id);
                        if let Some(i) = added.iter().position(|&e| e == redundant) {
                            added.swap_remove(i);
                        } else {
                            removed.push(redundant);
                        }
                        location_add.retain(|&(id, _)| id != r.id);
                        location_remove.push(r.id);
                        last.end = r.end;
                        continue;
                    }
                }
                merged.push(r);
            }
            ts.ranges = merged;
            for e in removed {
                self.stores[s].order.remove(&e);
            }
            for e in added {
                self.stores[s].order.insert(e);
            }
            for id in location_remove {
                self.stores[s].locations.remove(&id);
            }
            for (id, location) in location_add {
                self.stores[s].locations.insert(id, location);
            }
        }
        read_bytes
    }

    /// Remove `span` from store `s` (whatever of it is held). Returns the
    /// write ids of arriving parts.
    pub fn store_remove(&mut self, s: StoreId, span: Span) -> Vec<String> {
        let mut ids = Vec::new();
        let node_id = span.node;
        let Some(ts) = self.nodes[node_id as usize].tiers.get(&s) else {
            return ids;
        };
        let split_count = ts
            .ranges
            .iter()
            .filter(|r| {
                r.start < span.end && span.start < r.end && r.start < span.start && span.end < r.end
            })
            .count();
        let mut split_ids = (0..split_count)
            .map(|_| self.new_tier_range_id())
            .collect::<Vec<_>>()
            .into_iter();
        let ts = self.nodes[node_id as usize].tiers.get_mut(&s).unwrap();
        let mut kept = Vec::new();
        let mut removed_blocks = 0u32;
        let mut order_rm = Vec::new();
        let mut order_add = Vec::new();
        let mut location_update = Vec::new();
        let mut location_remove = Vec::new();
        for r in ts.ranges.drain(..) {
            if r.end <= span.start || r.start >= span.end {
                kept.push(r);
                continue;
            }
            if let Some(id) = &r.arriving {
                ids.push(id.clone());
            }
            let cut_s = r.start.max(span.start);
            let cut_e = r.end.min(span.end);
            removed_blocks += cut_e - cut_s;
            if r.start < cut_s {
                let head = TierRange {
                    end: cut_s,
                    arriving: r.arriving.clone(),
                    ..r
                };
                kept.push(head);
                if cut_e < r.end {
                    let tail = TierRange {
                        id: split_ids.next().unwrap(),
                        start: cut_e,
                        ..r
                    };
                    order_add.push((tail.key, tail.seq, tail.id));
                    location_update.push((tail.id, (node_id, tail.start)));
                    kept.push(tail);
                }
            } else if cut_e < r.end {
                let tail = TierRange { start: cut_e, ..r };
                location_update.push((tail.id, (node_id, tail.start)));
                kept.push(tail);
            } else {
                order_rm.push((r.key, r.seq, r.id));
                location_remove.push(r.id);
            }
        }
        debug_assert!(split_ids.next().is_none());
        ts.ranges = kept;
        let empty = ts.ranges.is_empty();
        self.stores[s].held -= removed_blocks;
        if removed_blocks > 0 {
            self.stores[s].source_version = self.stores[s].source_version.wrapping_add(1);
        }
        for e in order_rm {
            self.stores[s].order.remove(&e);
        }
        for e in order_add {
            self.stores[s].order.insert(e);
        }
        for id in location_remove {
            self.stores[s].locations.remove(&id);
        }
        for (id, location) in location_update {
            self.stores[s].locations.insert(id, location);
        }
        if empty {
            self.nodes[node_id as usize].tiers.remove(&s);
        }
        self.prune_if_empty(node_id);
        ids
    }

    /// Drop ranges of store `s` untouched for more than `seconds`.
    pub fn store_expire(&mut self, s: StoreId, now: f64, seconds: f64) -> Vec<TierEvicted> {
        let mut out = Vec::new();
        while let Some(e) = self.stores[s].order.iter().next().copied() {
            let (_, _, id) = e;
            let touched = self.stores[s].locations.get(&id).and_then(|&(node_id, _)| {
                self.nodes[node_id as usize]
                    .tiers
                    .get(&s)
                    .and_then(|ts| ts.ranges.iter().find(|r| r.id == id))
                    .map(|r| r.touched)
            });
            match touched {
                Some(t) if now - t > seconds => {
                    if let Some(ev) = self.store_evict_entry(s, e, u32::MAX) {
                        out.push(ev);
                    }
                }
                Some(_) => break,
                None => {
                    self.stores[s].order.remove(&e);
                    self.stores[s].locations.remove(&id);
                }
            }
        }
        out
    }

    /// Store `s` blocks held per node (for totals).
    pub fn store_held(&self, s: StoreId) -> u32 {
        self.stores[s].held
    }

    /// Bytes store `s` holds.
    pub fn store_held_bytes(&self, s: StoreId) -> u64 {
        let mut total = 0;
        for n in &self.nodes {
            if !n.alive {
                continue;
            }
            if let Some(ts) = n.tiers.get(&s) {
                for r in &ts.ranges {
                    let d = n.depth;
                    total += self
                        .bytes_at_boundary(d + r.end)
                        .saturating_sub(self.bytes_at_boundary(d + r.start));
                }
            }
        }
        total
    }

    /// Whether the store's eviction policy orders by outlook.
    pub fn store_outlook_ordered(&self, s: StoreId) -> bool {
        matches!(self.stores[s].eviction, EvictionPolicy::Outlook {})
    }

    // ------------------------------------------------------------------
    // Diagnostics

    /// Live nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.alive).count()
    }

    /// Whether the block of `hash` (wherever it sits on a path) is resident
    /// in worker `w`'s HBM. Test helper: scans the tree.
    pub fn hbm_contains(&self, w: WorkerId, hash: u64) -> bool {
        self.nodes.iter().filter(|n| n.alive).any(|n| {
            n.hashes
                .iter()
                .position(|&h| h == hash)
                .is_some_and(|off| n.hbm.get(&w).is_some_and(|h| h.resident > off as u32))
        })
    }

    /// Whether store `s` holds the block of `hash`. Test helper.
    pub fn store_contains_hash(&self, s: StoreId, hash: u64) -> bool {
        self.nodes.iter().filter(|n| n.alive).any(|n| {
            n.hashes.iter().position(|&h| h == hash).is_some_and(|off| {
                n.tiers.get(&s).is_some_and(|ts| {
                    ts.ranges
                        .iter()
                        .any(|r| r.start <= off as u32 && (off as u32) < r.end)
                })
            })
        })
    }

    /// Whether store `s` holds the block of `hash` with its write landed.
    pub fn store_resident_hash(&self, s: StoreId, hash: u64) -> bool {
        self.nodes.iter().filter(|n| n.alive).any(|n| {
            n.hashes.iter().position(|&h| h == hash).is_some_and(|off| {
                n.tiers.get(&s).is_some_and(|ts| {
                    ts.ranges.iter().any(|r| {
                        r.start <= off as u32 && (off as u32) < r.end && r.arriving.is_none()
                    })
                })
            })
        })
    }

    /// The span of the block of `hash`, if the tree has it. Test helper.
    pub fn span_of_hash(&self, hash: u64) -> Option<Span> {
        self.nodes.iter().enumerate().find_map(|(i, n)| {
            if !n.alive {
                return None;
            }
            n.hashes.iter().position(|&h| h == hash).map(|off| Span {
                node: i as NodeId,
                start: off as u32,
                end: off as u32 + 1,
            })
        })
    }

    /// Blocks resident in worker `w`'s HBM (all nodes), for utilisation.
    pub fn resident_blocks(&self, w: WorkerId) -> u32 {
        self.nodes
            .iter()
            .filter(|n| n.alive)
            .filter_map(|n| n.hbm.get(&w))
            .map(|h| h.resident)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radix() -> Radix {
        // 100 B per token, 16-token blocks: 1600 B per block.
        Radix::new(16, std::sync::Arc::new(|t| 100 * t as u64))
    }

    #[test]
    fn insert_resolve_and_split() {
        let mut r = radix();
        let a = r.insert(&[1, 2, 3, 4]);
        assert_eq!(a.blocks, 4);
        assert_eq!(a.segs.len(), 1);
        // Same prefix, longer: extends as a child.
        let b = r.insert(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(b.blocks, 6);
        assert_eq!(b.segs.len(), 2);
        assert_eq!(b.segs[0], a.segs[0]);
        // Diverging inside the first node splits it.
        let c = r.insert(&[1, 2, 9]);
        assert_eq!(c.blocks, 3);
        assert_eq!(c.segs.len(), 2);
        assert_eq!(c.segs[0].len, 2);
        // The original path is still resolvable and unchanged in blocks.
        let a2 = r.resolve(&[1, 2, 3, 4]);
        assert_eq!(a2.blocks, 4);
        assert_eq!(a2.segs.len(), 2);
        // Partial: [1,2,3] ends mid-node.
        let p = r.resolve(&[1, 2, 3]);
        assert_eq!(p.blocks, 3);
        assert_eq!(p.segs.last().unwrap().len, 1);
        // Unknown: nothing.
        assert_eq!(r.resolve(&[7]).blocks, 0);
        assert_eq!(r.resolve(&[1, 2, 3, 4, 5, 6, 7]).blocks, 6);
    }

    #[test]
    fn cached_leaf_validation_tracks_splits_and_merges() {
        let mut r = radix();
        let base = r.insert(&[1, 2, 3, 4]);
        let base_leaf = base.segs.last().unwrap().node;
        let descendant = r.insert(&[1, 2, 3, 4, 5, 6]);
        let descendant_leaf = descendant.segs.last().unwrap().node;

        // Splitting above a descendant keeps that leaf valid: rebuilding via
        // current parents discovers the newly inserted intermediate node.
        let branch = r.insert(&[1, 2, 9]);
        let mut rebuilt = Path::default();
        assert!(r.rebuild_cached_path(descendant_leaf, 6, 6, 6, &mut rebuilt));
        assert_eq!(rebuilt, r.resolve(&[1, 2, 3, 4, 5, 6]));

        // The split moved base's old end out of its cached node.
        assert!(!r.rebuild_cached_path(base_leaf, 4, 4, 4, &mut rebuilt));
        let split_tail = r.resolve(&[1, 2, 3, 4]).segs.last().unwrap().node;

        // Removing the side branch merges that tail away. Its dead id is
        // rejected, while the descendant leaf follows the compacted parent.
        r.prune_if_empty(branch.segs.last().unwrap().node);
        assert!(!r.rebuild_cached_path(split_tail, 4, 4, 4, &mut rebuilt));
        assert!(r.rebuild_cached_path(descendant_leaf, 6, 6, 6, &mut rebuilt));
        assert_eq!(rebuilt, r.resolve(&[1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn cached_insert_extends_a_leaf_and_branches_from_a_partial_node() {
        let mut r = radix();
        let mut path = r.insert(&[1, 2]);
        let leaf = path.segs.last().unwrap().node;
        assert!(r.insert_from_cached(leaf, 2, 2, &[1, 2, 3, 4], &mut path));
        assert_eq!(path.segs.len(), 1);
        assert_eq!(path.segs[0].node, leaf);
        assert_eq!(path, r.resolve(&[1, 2, 3, 4]));

        // A request cached at the middle of that node can diverge there
        // without walking from the root; the existing suffix becomes a peer.
        let mut partial = Path::default();
        assert!(r.rebuild_cached_path(leaf, 2, 2, 2, &mut partial));
        assert!(r.insert_from_cached(leaf, 2, 2, &[1, 2, 9], &mut partial));
        assert_eq!(partial, r.resolve(&[1, 2, 9]));
        assert_eq!(r.resolve(&[1, 2, 3, 4]).blocks, 4);
    }

    #[test]
    fn cached_insert_preserves_backed_first_node_boundaries() {
        let mut r = radix();
        r.register_worker(0, 8, HbmEviction::Lru {}, true);
        let mut path = r.insert(&[1]);
        r.acquire(0, &path, 0, 1, 0, true, None, false).unwrap();
        let leaf = path.segs[0].node;

        assert!(r.insert_from_cached(leaf, 1, 1, &[1, 2], &mut path));
        assert_eq!(path.segs.len(), 2);
        assert_eq!(path.blocks, 2);
        assert_eq!(r.resolve(&[1, 2]), path);
    }

    #[test]
    fn store_range_ids_survive_radix_split_and_merge() {
        let mut r = radix();
        let s = r.add_store(8, EvictionPolicy::Fifo {});
        let path = r.insert(&[1, 2, 3, 4]);
        let original_node = path.segs[0].node;
        assert!(r
            .store_insert(
                s,
                Span {
                    node: original_node,
                    start: 0,
                    end: 4,
                },
                None,
                0.0,
            )
            .is_empty());
        let original_entry = *r.stores[s].order.first().unwrap();

        // A side branch splits the stored range. The existing order entry
        // stays byte-for-byte unchanged; only one entry is added for the new
        // physical piece, and both ids resolve through the side map.
        let branch = r.insert(&[1, 2, 9]);
        assert!(r.stores[s].order.contains(&original_entry));
        assert_eq!(r.stores[s].order.len(), 2);
        for &(key, seq, id) in &r.stores[s].order {
            let &(node, start) = r.stores[s].locations.get(&id).unwrap();
            let range = r.nodes[node as usize].tiers[&s]
                .ranges
                .iter()
                .find(|range| range.id == id)
                .unwrap();
            assert_eq!((range.key, range.seq, range.start), (key, seq, start));
        }
        let split_order = r.stores[s].order.clone();

        // Once the empty side branch dies the chain compacts again. Neither
        // order key changes; the child range's location alone moves.
        r.prune_if_empty(branch.segs.last().unwrap().node);
        let merged = r.resolve(&[1, 2, 3, 4]);
        assert_eq!(merged.segs.len(), 1);
        assert_eq!(r.stores[s].order, split_order);
        for &(_, _, id) in &r.stores[s].order {
            let &(node, start) = r.stores[s].locations.get(&id).unwrap();
            assert_eq!(node, merged.segs[0].node);
            assert!(r.nodes[node as usize].tiers[&s]
                .ranges
                .iter()
                .any(|range| range.id == id && range.start == start));
        }
    }

    #[test]
    fn acquire_release_lru_recycles_tails_first() {
        let mut r = radix();
        r.register_worker(0, 6, HbmEviction::Lru {}, false);
        let a = r.insert(&[1, 2]);
        let b = r.insert(&[3, 4]);
        let c = r.insert(&[5, 6]);
        for p in [&a, &b, &c] {
            let got = r.acquire(0, p, 0, 2, 0, true, None, false).unwrap();
            assert_eq!(got.fresh_blocks, 2);
            assert_eq!(got.produced.len(), 1);
        }
        assert_eq!(r.free_blocks(0), 0);
        // Free a, then b, then c: LRU order a, b, c.
        r.release(0, &a, 2, 0);
        r.release(0, &b, 2, 0);
        r.release(0, &c, 2, 0);
        assert_eq!(r.free_blocks(0), 6);
        assert_eq!(r.resident_prefix(0, &[1, 2]), 2);
        // A hit on b pulls it out of the free set (touched).
        let got = r.acquire(0, &b, 0, 2, 0, true, None, false).unwrap();
        assert_eq!(got.hits_on_free, 2);
        assert_eq!(got.fresh_blocks, 0);
        assert_eq!(r.free_blocks(0), 4);
        // New content needing 3 blocks evicts a (oldest) fully and c's tail.
        let d = r.insert(&[7, 8, 9]);
        let got = r.acquire(0, &d, 0, 3, 0, true, None, false).unwrap();
        assert_eq!(got.fresh_blocks, 3);
        let ev: Vec<(u32, u32)> = got
            .evicted
            .iter()
            .map(|e| (e.span.start, e.span.end))
            .collect();
        assert_eq!(got.evicted[0].span.node, a.segs[0].node);
        assert_eq!(ev[0], (0, 2));
        assert_eq!(got.evicted[1].span.node, c.segs[0].node);
        assert_eq!(ev[1], (1, 2), "c's tail block, prefix survives");
        assert_eq!(r.resident_prefix(0, &[5, 6]), 1);
        assert_eq!(r.resident_prefix(0, &[1, 2]), 0);
        assert_eq!(r.free_blocks(0), 1);
    }

    #[test]
    fn stamps_are_non_increasing_along_a_chain() {
        // Parent [1,2] then child [1,2,3,4]: child frees later; the parent's
        // blocks get the child's newer stamp; the tail (3,4) is the oldest
        // only if freed earlier — here the child frees everything at once.
        let mut r = radix();
        r.register_worker(0, 8, HbmEviction::Lru {}, false);
        let p = r.insert(&[1, 2]);
        r.acquire(0, &p, 0, 2, 0, true, None, false).unwrap();
        r.release(0, &p, 2, 0);
        let c = r.insert(&[1, 2, 3, 4]);
        let got = r.acquire(0, &c, 0, 4, 0, true, None, false).unwrap();
        assert_eq!(got.hits_on_free, 2);
        assert_eq!(got.fresh_blocks, 2);
        r.release(0, &c, 4, 0);
        // 4 blocks never used remain; a 6-block request takes them and
        // evicts 2 more from the chain, tail-first: node (3,4) goes whole.
        let x = r.insert(&[9, 10, 11, 12, 13, 14]);
        let got = r.acquire(0, &x, 0, 6, 0, true, None, false).unwrap();
        let spans: Vec<(NodeId, u32, u32)> = got
            .evicted
            .iter()
            .map(|e| (e.span.node, e.span.start, e.span.end))
            .collect();
        assert_eq!(spans, vec![(c.segs[1].node, 0, 2)]);
        assert_eq!(r.resident_prefix(0, &[1, 2, 3, 4]), 2);
        // Then the head (1,2) — the child's free stamped them newer than
        // the parent's own free, but they are still the only free blocks.
        let y = r.insert(&[20, 21]);
        let got = r.acquire(0, &y, 0, 2, 0, true, None, false).unwrap();
        assert_eq!(got.evicted.len(), 1);
        assert_eq!(got.evicted[0].span.node, c.segs[0].node);
        assert_eq!(r.resident_prefix(0, &[1, 2]), 0);
    }

    #[test]
    fn stores_hold_ranges_and_evict_by_policy() {
        let mut r = radix();
        r.register_worker(0, 4, HbmEviction::Lru {}, false);
        let s = r.add_store(3, EvictionPolicy::Fifo {});
        r.set_worker_tiers(0, vec![s]);
        let a = r.insert(&[1, 2, 3, 4]);
        let node = a.segs[0].node;
        // Write the tail [2,4) first (write-back order), then [0,2).
        let ev = r.store_insert(
            s,
            Span {
                node,
                start: 2,
                end: 4,
            },
            None,
            0.0,
        );
        assert!(ev.is_empty());
        let lk = r.lookup(0, &[1, 2, 3, 4]);
        assert_eq!(lk.tier_blocks[0], 0, "a hole at 0: nothing reachable");
        let ev = r.store_insert(
            s,
            Span {
                node,
                start: 0,
                end: 2,
            },
            None,
            1.0,
        );
        // Capacity 3: the oldest range [2,4) loses its tail block.
        assert_eq!(ev.len(), 1);
        assert_eq!((ev[0].span.start, ev[0].span.end), (3, 4));
        assert_eq!(ev[0].dead_bytes, 1600);
        let lk = r.lookup(0, &[1, 2, 3, 4]);
        assert_eq!(lk.tier_blocks[0], 3);
        assert_eq!(lk.tier_bytes[0], 3 * 1600);
        assert_eq!(r.store_held(s), 3);
        // Promotion marks read; a later eviction of read blocks is not dead.
        r.store_promoted(
            s,
            &[Span {
                node,
                start: 0,
                end: 3,
            }],
            2.0,
        );
        let ev = r.store_insert(
            s,
            Span {
                node,
                start: 3,
                end: 4,
            },
            None,
            3.0,
        );
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].dead_bytes, 0);
    }

    #[test]
    fn outlook_keys_order_dead_first_then_farthest() {
        let mut r = radix();
        r.register_worker(0, 6, HbmEviction::Outlook {}, false);
        let a = r.insert(&[1, 2]);
        let b = r.insert(&[3, 4]);
        let c = r.insert(&[5, 6]);
        for p in [&a, &b, &c] {
            r.acquire(0, p, 0, 2, 0, true, None, false).unwrap();
        }
        r.set_outlook(&a, Some(100.0), 2);
        r.set_outlook(&b, Some(10.0), 2);
        r.release(0, &a, 2, 0);
        r.release(0, &b, 2, 0);
        r.release(0, &c, 2, 0);
        let d = r.insert(&[7, 8, 9, 10, 11]);
        let got = r.acquire(0, &d, 0, 5, 0, true, None, false).unwrap();
        let order: Vec<NodeId> = got.evicted.iter().map(|e| e.span.node).collect();
        assert_eq!(order[0], c.segs[0].node, "no re-entry first");
        assert_eq!(order[1], a.segs[0].node, "farthest re-entry next");
        assert_eq!(order[2], b.segs[0].node);
        assert_eq!(got.evicted[2].span.len(), 1, "only b's tail block");
        assert_eq!(r.resident_prefix(0, &[3, 4]), 1);
    }

    #[test]
    fn a_landing_across_two_nodes_lands_whole_and_joins_are_seen() {
        let mut r = radix();
        r.register_worker(0, 8, HbmEviction::Lru {}, false);
        // Two nodes: [1,2] and child [3,4] (created by inserting a diverging
        // sibling first).
        r.insert(&[1, 2, 9]);
        let p = r.insert(&[1, 2, 3, 4]);
        assert_eq!(p.segs.len(), 2);
        let got = r.acquire(0, &p, 0, 4, 0, false, Some("L"), false).unwrap();
        assert_eq!(got.fresh_blocks, 4);
        let lk = r.lookup(0, &[1, 2, 3, 4]);
        assert_eq!((lk.hbm, lk.landing), (0, 4));
        assert_eq!(lk.leader.as_deref(), Some("L"));
        // A joiner sees the whole landing range even though it spans nodes.
        let lk2 = r.lookup(0, &[1, 2, 3, 4, 5]);
        assert_eq!((lk2.hbm, lk2.landing), (0, 4));
        r.landed(0, &p, 4);
        let lk = r.lookup(0, &[1, 2, 3, 4]);
        assert_eq!((lk.hbm, lk.landing), (4, 0));
        assert_eq!(r.resident_prefix(0, &[1, 2, 3, 4]), 4);
        r.release(0, &p, 4, 0);
        assert_eq!(r.free_blocks(0), 8);
    }

    #[test]
    fn multi_worker_lookup_matches_individual_views() {
        let mut r = radix();
        for w in 0..3 {
            r.register_worker(w, 8, HbmEviction::Lru {}, false);
        }
        let store = r.add_store(8, EvictionPolicy::Fifo {});
        for w in 0..3 {
            r.set_worker_tiers(w, vec![store]);
        }
        let path = r.insert(&[1, 2, 3, 4]);
        r.acquire(0, &path, 0, 4, 0, true, None, false).unwrap();
        r.acquire(1, &path, 0, 2, 0, true, None, false).unwrap();
        r.acquire(2, &path, 0, 3, 0, false, Some("landing"), false)
            .unwrap();
        r.store_insert(
            store,
            Span {
                node: path.segs[0].node,
                start: 0,
                end: 4,
            },
            None,
            0.0,
        );

        let workers = [0, 1, 2];
        let batched = r.lookup_workers(&workers, &[1, 2, 3, 4]);
        let resident = r.resident_prefix_workers(&workers, &[1, 2, 3, 4]);
        for ((&worker, summary), hbm) in workers.iter().zip(&batched).zip(resident) {
            let individual = r.lookup(worker, &[1, 2, 3, 4]);
            assert_eq!(summary.hbm, individual.hbm);
            assert_eq!(summary.landing, individual.landing);
            assert_eq!(
                summary.tier,
                individual.tier_blocks.iter().copied().sum::<u32>()
            );
            assert_eq!(summary.cached(), individual.cached());
            assert_eq!(hbm, individual.hbm);
        }
    }

    #[test]
    fn landing_then_landed_becomes_resident() {
        let mut r = radix();
        r.register_worker(0, 4, HbmEviction::Lru {}, false);
        let a = r.insert(&[1, 2, 3]);
        let got = r.acquire(0, &a, 0, 2, 0, false, Some("L"), false).unwrap();
        assert_eq!(got.fresh_blocks, 2);
        assert!(got.produced.is_empty());
        let lk = r.lookup(0, &[1, 2, 3]);
        assert_eq!((lk.hbm, lk.landing), (0, 2));
        assert_eq!(lk.leader.as_deref(), Some("L"));
        r.landed(0, &a, 2);
        let lk = r.lookup(0, &[1, 2, 3]);
        assert_eq!((lk.hbm, lk.landing), (2, 0));
        assert!(r.free_blocks(0) == 2);
        r.release(0, &a, 2, 0);
        assert_eq!(r.free_blocks(0), 4);
    }
}
