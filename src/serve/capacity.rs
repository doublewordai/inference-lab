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
//! `pending` is also what makes the bound hold under concurrency. It is
//! incremented by the admitting handler under a CAS, so N handlers racing at
//! the bound serialize against each other instead of all reading the same
//! stale depth and all being let through.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use tokio::sync::Notify;

/// Shared admission state for one model's engine.
pub struct Capacity {
    /// Queue depth as last published by the engine loop: the sum of every
    /// worker's `num_waiting()`.
    observed: AtomicUsize,

    /// Requests admitted by a handler whose `Arrival` the engine has not
    /// stepped yet. Counted toward the depth so a burst cannot slip past
    /// the bound in the window before the engine catches up.
    pending: AtomicUsize,

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
            observed: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            max_waiting: AtomicU32::new(max_waiting),
            max_num_seqs: AtomicU32::new(max_num_seqs),
            changed: Notify::new(),
        }
    }

    /// Requests waiting: what the schedulers hold plus what has been
    /// admitted but not yet stepped into them.
    pub fn waiting(&self) -> usize {
        self.observed.load(Ordering::Relaxed) + self.pending.load(Ordering::Relaxed)
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
    pub fn try_admit(&self) -> Result<(), usize> {
        let limit = self.max_waiting.load(Ordering::Relaxed) as usize;
        if limit == 0 {
            // Unbounded: still counted, so the control endpoint and a later
            // bound change both see a truthful depth.
            self.pending.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let mut pending = self.pending.load(Ordering::Relaxed);
        loop {
            let depth = self.observed.load(Ordering::Relaxed) + pending;
            if depth >= limit {
                return Err(depth);
            }
            match self.pending.compare_exchange_weak(
                pending,
                pending + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => pending = actual,
            }
        }
    }

    /// Give back a slot taken by [`Self::try_admit`] that never reached the
    /// engine.
    pub fn release(&self) {
        self.release_n(1);
    }

    /// Publish the engine's own view of its queues, and release the slots of
    /// `arrived` requests that have now landed in a scheduler's waiting
    /// queue.
    ///
    /// Order is deliberate: `observed` is written first, so a handler racing
    /// this can only ever *over*count the depth by the arrivals in flight,
    /// never undercount it. Overcounting refuses one request that had room;
    /// undercounting admits one past the bound, and the bound is the point.
    pub fn publish(&self, waiting: usize, running: usize, arrived: usize) {
        self.observed.store(waiting, Ordering::Relaxed);
        self.running.store(running, Ordering::Relaxed);
        if arrived > 0 {
            self.release_n(arrived);
        }
    }

    fn release_n(&self, n: usize) {
        // Saturating: a release without a matching reservation would
        // otherwise wrap to usize::MAX and refuse every subsequent request.
        let _ = self
            .pending
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |p| {
                Some(p.saturating_sub(n))
            });
    }
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

    #[test]
    fn release_cannot_underflow_into_refusing_everything() {
        let cap = Capacity::new(4, 128);
        cap.release();
        cap.publish(0, 0, 9);
        assert_eq!(cap.waiting(), 0);
        assert!(cap.try_admit().is_ok());
    }
}
