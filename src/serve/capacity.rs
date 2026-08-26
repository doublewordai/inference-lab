//! Saturation admission control and runtime capacity for `serve` mode.
//!
//! Two things live here, both shared between the HTTP handlers and the
//! [`RealtimeEngine`](super::engine::RealtimeEngine) that owns the simulated
//! engine:
//!
//! - **The waiting-queue bound.** Past `max_waiting` queued requests the
//!   server refuses further arrivals with HTTP 529 instead of enqueueing
//!   them, so a client driving this server can actually observe
//!   backpressure. `0` means unbounded — today's behaviour, unchanged.
//! - **The concurrency cap.** `max_num_seqs` retuned on a live server, which
//!   is how a scale up/down is modelled: it changes how fast work drains,
//!   and therefore the load at which the queue backs up and 529s begin.
//!
//! # Why a shared gauge rather than asking the scheduler
//!
//! `Scheduler::num_waiting()` is not reachable from a handler: the scheduler
//! sits inside the engine, which is owned by a single tokio task and spoken
//! to over an mpsc channel. So the engine loop *publishes* its queue depth
//! here and the handlers read it.
//!
//! # Why the depth is not just `aggregate_waiting()`
//!
//! [`Engine::submit`](crate::simulation::Engine::submit) only queues an
//! `Arrival` event; the request does not reach a scheduler's waiting queue
//! until a later `step()` processes it. A burst of arrivals would therefore
//! read as depth 0 and be admitted wholesale. The depth counted here is
//! `observed + pending`: what the schedulers hold, plus what the handlers
//! have admitted that the engine has not stepped yet.
//!
//! Both counters live in ONE atomic word, and every mutation of either is a
//! CAS on the whole word. That is what makes the bound exact:
//!
//! - Check-and-reserve is a single CAS, so a `publish` cannot land between a
//!   handler reading the depth and taking its slot. Held as two atomics, a
//!   handler could decide against a stale `observed` and admit past the bound.
//! - A request moving from `pending` to `observed` is one write, so it is
//!   never briefly counted twice or missed entirely.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use tokio::sync::Notify;

/// Shared admission state for one model's engine.
pub struct Capacity {
    /// The queue depth, as `observed` in the high 32 bits and `pending` in
    /// the low 32 (see [`pack`]).
    ///
    /// `observed` is what the engine loop last published: the sum of every
    /// worker's `num_waiting()`. `pending` is what handlers have admitted
    /// whose `Arrival` the engine has not stepped yet — counted toward the
    /// depth so a burst cannot slip past the bound in the window before the
    /// engine catches up.
    ///
    /// One word rather than two atomics so that check-and-reserve is a
    /// single CAS; see the module docs.
    depth: AtomicU64,

    /// Running requests, published for the control endpoint's benefit.
    running: AtomicUsize,

    /// The waiting-queue bound. `0` = unbounded (never reject).
    max_waiting: AtomicU32,

    /// Desired per-worker concurrency cap. The engine loop applies this to
    /// its schedulers when it changes; see [`Self::changed`].
    max_num_seqs: AtomicU32,

    /// Woken whenever a control knob is written, so the engine loop applies
    /// the change promptly instead of at its next arrival. `notify_one`
    /// (not `notify_waiters`) because there is exactly one consumer and it
    /// stores a permit when that consumer is between waits — otherwise a
    /// knob turned on a fully idle server would not be applied until the
    /// next request happened to arrive.
    changed: Notify,
}

impl Capacity {
    pub fn new(max_waiting: u32, max_num_seqs: u32) -> Self {
        Self {
            depth: AtomicU64::new(0),
            running: AtomicUsize::new(0),
            max_waiting: AtomicU32::new(max_waiting),
            max_num_seqs: AtomicU32::new(max_num_seqs),
            changed: Notify::new(),
        }
    }

    /// Requests waiting: what the schedulers hold plus what has been
    /// admitted but not yet stepped into them.
    pub fn waiting(&self) -> usize {
        let (observed, pending) = unpack(self.depth.load(Ordering::Relaxed));
        observed as usize + pending as usize
    }

    pub fn running(&self) -> usize {
        self.running.load(Ordering::Relaxed)
    }

    /// `0` when the queue is unbounded.
    pub fn max_waiting(&self) -> u32 {
        self.max_waiting.load(Ordering::Relaxed)
    }

    pub fn max_num_seqs(&self) -> u32 {
        self.max_num_seqs.load(Ordering::Relaxed)
    }

    /// Set the waiting bound; `0` disables rejection entirely. Takes effect
    /// on the next arrival — requests already queued are never retroactively
    /// refused, since their status code is long since spent.
    pub fn set_max_waiting(&self, max_waiting: u32) {
        self.max_waiting.store(max_waiting, Ordering::Relaxed);
        self.changed.notify_one();
    }

    /// Set the concurrency cap. The engine loop picks this up and writes it
    /// through to every scheduler; lowering it drains rather than evicts.
    pub fn set_max_num_seqs(&self, max_num_seqs: u32) {
        self.max_num_seqs.store(max_num_seqs, Ordering::Relaxed);
        self.changed.notify_one();
    }

    /// Wait for a control knob to be written.
    pub async fn changed(&self) {
        self.changed.notified().await
    }

    /// Take a queue slot for one request, or refuse.
    ///
    /// `Ok(())` reserves the slot: the caller MUST follow with either an
    /// engine submit (the engine releases the slot when it steps the
    /// arrival) or [`Self::release`] if the submit fails, or the depth
    /// leaks upward and the model eventually refuses everything.
    ///
    /// `Err(depth)` is a refusal at `depth` waiting requests, and the caller
    /// must turn it into a 529 *before* any response begins.
    ///
    /// The bound is exact at the moment of admission: the depth is tested
    /// and the slot taken in one CAS, so a concurrent [`Self::publish`]
    /// cannot slip in between and let a request past. It says nothing about
    /// later growth — a preemption returns a running request to a
    /// scheduler's waiting queue, which can push the depth over the bound
    /// with no admission involved. That is correct: the bound gates
    /// admission, and an admitted request is never retroactively refused.
    pub fn try_admit(&self) -> Result<(), usize> {
        // 0 = unbounded; the slot is still counted so the control endpoint
        // and any later bound change both see a truthful depth.
        let limit = self.max_waiting.load(Ordering::Relaxed) as usize;
        let mut word = self.depth.load(Ordering::Relaxed);
        loop {
            let (observed, pending) = unpack(word);
            let depth = observed as usize + pending as usize;
            if limit != 0 && depth >= limit {
                return Err(depth);
            }
            match self.depth.compare_exchange_weak(
                word,
                pack(observed, pending.saturating_add(1)),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => word = actual,
            }
        }
    }

    /// Give back a slot taken by [`Self::try_admit`] that never reached the
    /// engine.
    pub fn release(&self) {
        self.update_depth(None, 1);
    }

    /// Publish the engine's own view of its queues, and release the slots of
    /// `arrived` requests that have now landed in a scheduler's waiting
    /// queue.
    ///
    /// Setting `observed` and dropping those `pending` slots is one write, so
    /// an arrival in transit is never counted twice nor missed: a racing
    /// handler sees it either as pending or as observed, and the depth does
    /// not flinch as it crosses.
    pub fn publish(&self, waiting: usize, running: usize, arrived: usize) {
        self.running.store(running, Ordering::Relaxed);
        self.update_depth(Some(saturate(waiting)), saturate(arrived));
    }

    /// CAS `depth`: set `observed` if given, and drop `release` pending
    /// slots. Saturating, because a release without a matching reservation
    /// would otherwise wrap and refuse every subsequent request.
    fn update_depth(&self, observed: Option<u32>, release: u32) {
        let mut word = self.depth.load(Ordering::Relaxed);
        loop {
            let (prev_observed, pending) = unpack(word);
            let next = pack(
                observed.unwrap_or(prev_observed),
                pending.saturating_sub(release),
            );
            match self
                .depth
                .compare_exchange_weak(word, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(actual) => word = actual,
            }
        }
    }
}

/// `observed` in the high 32 bits, `pending` in the low 32.
const fn pack(observed: u32, pending: u32) -> u64 {
    ((observed as u64) << 32) | pending as u64
}

const fn unpack(word: u64) -> (u32, u32) {
    ((word >> 32) as u32, word as u32)
}

/// Counters are `usize` at the call sites but half a word here. Clamping
/// rather than truncating keeps a nonsensical value large instead of
/// wrapping it small, which would read as an empty queue.
fn saturate(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_never_refuses() {
        let cap = Capacity::new(0, 128);
        for _ in 0..10_000 {
            assert!(cap.try_admit().is_ok());
        }
        assert_eq!(cap.waiting(), 10_000);
    }

    #[test]
    fn bound_refuses_at_the_limit() {
        let cap = Capacity::new(3, 128);
        assert!(cap.try_admit().is_ok());
        assert!(cap.try_admit().is_ok());
        assert!(cap.try_admit().is_ok());
        assert_eq!(cap.try_admit(), Err(3));
    }

    /// The bound must hold on pending reservations alone — before the engine
    /// has stepped a single arrival, which is exactly the burst case.
    #[test]
    fn pending_alone_saturates_before_the_engine_observes_anything() {
        let cap = Capacity::new(2, 128);
        assert!(cap.try_admit().is_ok());
        assert!(cap.try_admit().is_ok());
        assert_eq!(cap.waiting(), 2);
        assert!(cap.try_admit().is_err());
    }

    /// An arrival moving from `pending` to `observed` must not change the
    /// depth, or the bound would breathe by the size of the in-flight batch.
    #[test]
    fn publish_hands_the_slot_over_without_double_counting() {
        let cap = Capacity::new(2, 128);
        cap.try_admit().unwrap();
        cap.try_admit().unwrap();
        assert_eq!(cap.waiting(), 2);
        cap.publish(1, 0, 1);
        assert_eq!(cap.waiting(), 2);
        cap.publish(2, 0, 1);
        assert_eq!(cap.waiting(), 2);
        assert!(cap.try_admit().is_err());
    }

    #[test]
    fn draining_the_queue_reopens_admission() {
        let cap = Capacity::new(2, 128);
        cap.try_admit().unwrap();
        cap.try_admit().unwrap();
        assert!(cap.try_admit().is_err());
        cap.publish(0, 2, 2);
        assert_eq!(cap.waiting(), 0);
        assert!(cap.try_admit().is_ok());
    }

    #[test]
    fn failed_submit_gives_the_slot_back() {
        let cap = Capacity::new(1, 128);
        cap.try_admit().unwrap();
        assert!(cap.try_admit().is_err());
        cap.release();
        assert!(cap.try_admit().is_ok());
    }

    /// Raising the bound must let queued-past-the-old-limit load through
    /// again, and lowering it to 0 must disable rejection outright.
    #[test]
    fn bound_is_retunable_at_runtime() {
        let cap = Capacity::new(1, 128);
        cap.try_admit().unwrap();
        assert!(cap.try_admit().is_err());
        cap.set_max_waiting(4);
        assert!(cap.try_admit().is_ok());
        cap.set_max_waiting(0);
        for _ in 0..100 {
            assert!(cap.try_admit().is_ok());
        }
    }

    /// Handlers racing at the bound must serialize: the reservation is a CAS,
    /// so exactly `max_waiting` of them win however many are in flight.
    #[test]
    fn concurrent_admissions_stop_exactly_at_the_bound() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let cap = Arc::new(Capacity::new(8, 128));
        let wins = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..32)
            .map(|_| {
                let (cap, wins) = (cap.clone(), wins.clone());
                std::thread::spawn(move || {
                    for _ in 0..64 {
                        if cap.try_admit().is_ok() {
                            wins.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(wins.load(Ordering::Relaxed), 8);
        assert_eq!(cap.waiting(), 8);
    }

    /// Admission must test against the depth the engine last published, not
    /// against pending alone — the queue can be full with nothing pending.
    #[test]
    fn admission_tests_against_the_published_depth() {
        let cap = Capacity::new(4, 128);
        cap.publish(4, 0, 0);
        assert_eq!(cap.try_admit(), Err(4));

        // One slot short of the bound: exactly one more gets in.
        cap.publish(3, 0, 0);
        assert!(cap.try_admit().is_ok());
        assert_eq!(cap.try_admit(), Err(4));
    }

    /// A preemption returns a running request to a waiting queue, so the
    /// depth can exceed the bound with no admission involved. That must not
    /// wedge anything: admission simply stays shut until it drains back.
    #[test]
    fn depth_above_the_bound_is_tolerated_and_recovers() {
        let cap = Capacity::new(4, 128);
        cap.publish(9, 0, 0);
        assert_eq!(cap.waiting(), 9);
        assert_eq!(cap.try_admit(), Err(9));
        cap.publish(2, 0, 0);
        assert!(cap.try_admit().is_ok());
    }

    #[test]
    fn depth_word_round_trips_both_counters() {
        assert_eq!(unpack(pack(7, 3)), (7, 3));
        assert_eq!(unpack(pack(0, u32::MAX)), (0, u32::MAX));
        assert_eq!(unpack(pack(u32::MAX, 0)), (u32::MAX, 0));
        assert_eq!(saturate(usize::MAX), u32::MAX);
    }

    #[test]
    fn release_cannot_underflow_into_refusing_everything() {
        let cap = Capacity::new(4, 128);
        cap.release();
        cap.publish(0, 0, 9);
        assert_eq!(cap.waiting(), 0);
        assert!(cap.try_admit().is_ok());
    }
}
