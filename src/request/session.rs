//! Stateful request trajectories from Batchbench replay manifests.
//!
//! Each trajectory starts at its recorded `start_after_ms`. Later requests
//! arrive after the previous request completes plus its `delay_after_ms`.
//! Prompt blocks preserve stable cross-trajectory prefix identity without
//! retaining prompt text.

use super::manifest::{PromptIdentity, ReplayManifest};
use super::Request;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};

#[derive(Debug, Clone)]
struct ReplayStep {
    input: u32,
    output: u32,
    recorded_arrival: Option<f64>,
    reset: bool,
    gap: f64,
    kind: Option<String>,
    prompt_identity: Option<PromptIdentity>,
}

#[derive(Debug, Clone)]
struct ReplayTrace {
    id: String,
    start_after_ms: u64,
    steps: Vec<ReplayStep>,
}

impl ReplayTrace {
    fn from_manifest(plan: ReplayManifest) -> Self {
        let mut steps = Vec::with_capacity(plan.requests.len());
        let mut warned_prompt_mismatch = false;
        for (index, request) in plan.requests.iter().enumerate() {
            if !warned_prompt_mismatch
                && request
                    .expected_prompt_tokens()
                    .is_some_and(|expected| expected != u64::from(request.prompt_tokens))
            {
                warned_prompt_mismatch = true;
                log::warn!(
                    "trajectory {:?} request {}: prompt_tokens does not equal sum(blocks.tokens) + overhead_tokens; using prompt_tokens for size",
                    plan.trajectory_id,
                    index + 1
                );
            }
            let prompt_identity = PromptIdentity::from_request(request);
            let kind = prompt_identity.as_ref().and_then(PromptIdentity::step_kind);
            let gap = index
                .checked_sub(1)
                .map(|parent| plan.requests[parent].delay_after_ms as f64 / 1000.0)
                .unwrap_or(0.0);
            steps.push(ReplayStep {
                input: request.prompt_tokens,
                output: request.output_tokens,
                recorded_arrival: request
                    .recorded_start_after_ms
                    .map(|milliseconds| milliseconds as f64 / 1000.0),
                reset: index == 0 || request.reset_before,
                gap,
                kind,
                prompt_identity,
            });
        }
        Self {
            id: plan.trajectory_id,
            start_after_ms: plan.start_after_ms.unwrap_or(0),
            steps,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionLifecycle {
    pub started: u32,
    pub completed: u32,
    pub deadline_censored: u32,
    pub active: u32,
}

/// Which trajectory request this is, stamped on the [`Request`] so metrics
/// can attribute it. `session` is the run's trajectory ordinal.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionStep {
    pub session: u32,
    pub step: u32,
    /// Seconds between the parent's completion and this arrival (0 for the
    /// first request).
    pub gap: f64,
    /// Prompt tokens shared with the parent's context.
    pub shared_tokens: u32,
    /// Leading shared-prefix tokens materialized by the parent's prefill.
    pub shared_prefill_tokens: u32,
    /// Shared-prefix tokens materialized by the parent's decoder.
    pub shared_decode_tokens: u32,
    pub kind: Option<String>,
    /// KV bytes written/touched between the parent completion and this
    /// request. Filled by the simulation driver and engine.
    pub parent_bytes_written: Option<u64>,
    pub reuse_distance_bytes: Option<u64>,
    pub parent_bytes_touched: Option<u64>,
    pub reuse_touched_bytes: Option<u64>,
    /// The next request's think time and reusable prefix, when one exists.
    pub next_gap: Option<f64>,
    pub next_shared_tokens: u32,
}

impl SessionStep {
    pub fn outlook_at(&self, completion_time: f64) -> Option<Outlook> {
        self.next_gap.map(|gap| Outlook {
            next_arrival: completion_time + gap.max(0.0),
            shared_tokens: self.next_shared_tokens,
        })
    }
}

/// A trajectory's announced re-entry. Oracle policies read it; reactive ones
/// never see it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outlook {
    pub next_arrival: f64,
    pub shared_tokens: u32,
}

struct ActiveTrace {
    trace_idx: usize,
    next_step: usize,
    hashes: Vec<u64>,
    context_tokens: u32,
    prefill_materialized_blocks: usize,
    decode_materialized_blocks: usize,
}

/// Turns Batchbench trajectories into requests and queues each re-entry at
/// its parent's completion plus the recorded delay.
pub(crate) struct SessionSource {
    traces: Vec<ReplayTrace>,
    block_size: u32,
    starts: VecDeque<(u64, usize)>,
    started: u32,
    completed: u32,
    deadline_censored: u32,
    active: HashMap<u32, ActiveTrace>,
    pending: BinaryHeap<Reverse<(OrderedTime, u32)>>,
    next_hash: u64,
}

/// f64 wrapper ordered by value so it can sit in a heap key. Times are finite
/// by construction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedTime(f64);

impl Eq for OrderedTime {}

impl PartialOrd for OrderedTime {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedTime {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl SessionSource {
    pub(crate) fn new(manifests: Vec<ReplayManifest>, block_size: u32) -> Self {
        let traces: Vec<_> = manifests
            .into_iter()
            .map(ReplayTrace::from_manifest)
            .collect();
        Self::from_traces(traces, block_size)
    }

    fn from_traces(traces: Vec<ReplayTrace>, block_size: u32) -> Self {
        let mut starts: Vec<_> = traces
            .iter()
            .enumerate()
            .map(|(index, trace)| (trace.start_after_ms, index))
            .collect();
        starts.sort_by_key(|&(offset, index)| (offset, index));
        Self {
            traces,
            block_size: block_size.max(1),
            starts: starts.into(),
            started: 0,
            completed: 0,
            deadline_censored: 0,
            active: HashMap::new(),
            pending: BinaryHeap::new(),
            // Stable Batchbench hashes occupy the high-bit namespace.
            next_hash: 1 << 40,
        }
    }

    pub(crate) fn num_started(&self) -> u32 {
        self.started
    }

    pub(crate) fn num_active(&self) -> usize {
        self.active.len()
    }

    pub(crate) fn lifecycle(&self) -> SessionLifecycle {
        SessionLifecycle {
            started: self.started,
            completed: self.completed,
            deadline_censored: self.deadline_censored,
            active: self.active.len() as u32,
        }
    }

    pub(crate) fn peek_start(&self) -> Option<f64> {
        self.starts
            .front()
            .map(|(milliseconds, _)| *milliseconds as f64 / 1000.0)
    }

    pub(crate) fn starts_exhausted(&self) -> bool {
        self.starts.is_empty()
    }

    pub(crate) fn peek_pending(&self) -> Option<f64> {
        self.pending.peek().map(|Reverse((time, _))| time.0)
    }

    pub(crate) fn start_next(&mut self) -> Option<Request> {
        let (milliseconds, trace_idx) = self.starts.pop_front()?;
        let ordinal = self.started;
        self.started += 1;
        self.active.insert(
            ordinal,
            ActiveTrace {
                trace_idx,
                next_step: 0,
                hashes: Vec::new(),
                context_tokens: 0,
                prefill_materialized_blocks: 0,
                decode_materialized_blocks: 0,
            },
        );
        Some(self.issue_step(ordinal, milliseconds as f64 / 1000.0, 0.0))
    }

    pub(crate) fn next_due(&mut self, current_time: f64) -> Option<Request> {
        let Reverse((time, ordinal)) = *self.pending.peek()?;
        if time.0 > current_time {
            return None;
        }
        self.pending.pop();
        let active = &self.active[&ordinal];
        let gap = self.traces[active.trace_idx].steps[active.next_step]
            .gap
            .max(0.0);
        Some(self.issue_step(ordinal, time.0, gap))
    }

    pub(crate) fn on_step_complete_before(
        &mut self,
        step: &SessionStep,
        completion_time: f64,
        deadline: Option<f64>,
    ) -> bool {
        let Some(active) = self.active.get(&step.session) else {
            return false;
        };
        let trace = &self.traces[active.trace_idx];
        if active.next_step >= trace.steps.len() {
            self.active.remove(&step.session);
            self.completed += 1;
            return false;
        }

        let arrival = completion_time + trace.steps[active.next_step].gap.max(0.0);
        if deadline.is_some_and(|limit| arrival > limit) {
            self.active.remove(&step.session);
            self.deadline_censored += 1;
            return false;
        }
        self.pending
            .push(Reverse((OrderedTime(arrival), step.session)));
        true
    }

    fn issue_step(&mut self, ordinal: u32, arrival_time: f64, gap: f64) -> Request {
        let block_size = self.block_size;
        let active = self.active.get_mut(&ordinal).expect("trajectory is active");
        let step_idx = active.next_step;
        let trace = &self.traces[active.trace_idx];
        let step = &trace.steps[step_idx];
        let input = step.input.max(1);
        let output = step.output.max(1);

        let shared_tokens = if step.reset {
            0
        } else {
            input.min(active.context_tokens) / block_size * block_size
        };
        let shared_blocks = (shared_tokens / block_size) as usize;
        let shared_prefill_blocks = shared_blocks.min(active.prefill_materialized_blocks);
        let shared_decode_blocks = shared_blocks
            .saturating_sub(shared_prefill_blocks)
            .min(active.decode_materialized_blocks);

        let total_blocks = (input + output).div_ceil(block_size) as usize;
        let stable_hashes = if shared_blocks == 0 {
            step.prompt_identity
                .as_ref()
                .map(|identity| identity.full_block_hashes(input, block_size))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut hashes = Vec::with_capacity(total_blocks);
        hashes.extend_from_slice(&active.hashes[..shared_blocks.min(active.hashes.len())]);
        while hashes.len() < total_blocks {
            if hashes.len() < stable_hashes.len() {
                hashes.push(stable_hashes[hashes.len()]);
            } else {
                hashes.push(self.next_hash);
                self.next_hash += 1;
            }
        }

        let context = input + output;
        let next = trace.steps.get(step_idx + 1);
        let next_gap = next.map(|request| request.gap.max(0.0));
        let next_shared_tokens = next
            .filter(|request| !request.reset)
            .map(|request| request.input.min(context) / block_size * block_size)
            .unwrap_or(0);

        let mut request = Request::new_with_target(
            format!("{}/{step_idx}", trace.id),
            0,
            arrival_time,
            input,
            output,
            output,
        );
        request.recorded_arrival_time = step.recorded_arrival;
        request.prompt_block_hashes = hashes.clone();
        request.session = Some(Box::new(SessionStep {
            session: ordinal,
            step: step_idx as u32,
            gap,
            shared_tokens,
            shared_prefill_tokens: shared_prefill_blocks as u32 * block_size,
            shared_decode_tokens: shared_decode_blocks as u32 * block_size,
            kind: step.kind.clone(),
            parent_bytes_written: None,
            reuse_distance_bytes: None,
            parent_bytes_touched: None,
            reuse_touched_bytes: None,
            next_gap,
            next_shared_tokens,
        }));

        active.hashes = hashes;
        active.context_tokens = context;
        active.prefill_materialized_blocks =
            (input.div_ceil(block_size) as usize).min(active.hashes.len());
        active.decode_materialized_blocks = active
            .hashes
            .len()
            .saturating_sub(active.prefill_materialized_blocks);
        active.next_step += 1;
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_step() -> Vec<ReplayManifest> {
        ReplayManifest::read_jsonl(
            concat!(
                r#"{"schema_version":2,"trajectory_id":"s","start_after_ms":1000,"requests":["#,
                r#"{"prompt_tokens":100,"output_tokens":20,"overhead_tokens":4,"delay_after_ms":2500,"blocks":["#,
                r#"{"seed":"sys","tokens":32,"role":"system"},{"seed":"user","tokens":64,"role":"user"}]},"#,
                r#"{"prompt_tokens":130,"output_tokens":5,"overhead_tokens":2,"blocks":["#,
                r#"{"seed":"sys","tokens":32,"role":"system"},{"seed":"user","tokens":64,"role":"user"},"#,
                r#"{"seed":"reply","tokens":20,"role":"assistant","live":true},{"seed":"tool","tokens":12,"role":"tool"}]}]}"#,
                "\n"
            )
            .as_bytes(),
        )
        .unwrap()
    }

    #[test]
    fn re_entry_uses_recorded_start_delay_and_parent_prefix() {
        let mut source = SessionSource::new(two_step(), 16);
        assert_eq!(source.peek_start(), Some(1.0));
        let first = source.start_next().unwrap();
        assert_eq!(first.request_id, "s/0");
        assert_eq!(first.arrival_time, 1.0);
        let first_step = first.session.clone().unwrap();
        assert!(source.on_step_complete_before(&first_step, 10.0, None));
        assert_eq!(source.peek_pending(), Some(12.5));
        let second = source.next_due(12.5).unwrap();
        assert_eq!(second.request_id, "s/1");
        assert_eq!(second.session.as_ref().unwrap().shared_tokens, 112);
        assert_eq!(
            &second.prompt_block_hashes[..7],
            &first.prompt_block_hashes[..7]
        );
        assert!(!source.on_step_complete_before(second.session.as_ref().unwrap(), 13.0, None));
        assert_eq!(source.lifecycle().completed, 1);
    }

    #[test]
    fn deadline_censors_a_future_re_entry() {
        let mut source = SessionSource::new(two_step(), 16);
        let first = source.start_next().unwrap();
        assert!(!source.on_step_complete_before(first.session.as_ref().unwrap(), 10.0, Some(12.0)));
        assert_eq!(source.lifecycle().deadline_censored, 1);
        assert_eq!(source.lifecycle().active, 0);
    }
}
