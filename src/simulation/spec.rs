//! Speculative decoding: draft-depth planning, acceptance outcomes and
//! step pricing. The [`Engine`](super::Engine) owns a [`SpecPlanner`] when
//! speculation is enabled and consults it at two points per iteration:
//! when a verify pass runs (how many draft tokens were accepted) and at the
//! end of the step (how deep to draft next). Everything about policies,
//! acceptance models, trace banks and measured cost tables lives here.
//!
//! Draft outcomes are drawn per decode sequence at planning time as a
//! *round*: the per-depth acceptance signal a gate could see, and the
//! number of draft tokens that would commit at full depth. Analytic
//! acceptance models and replayed trace banks both produce rounds, so the
//! policies and the verify step have one path.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::compute::MeasuredCostTable;
use crate::config::{
    AcceptanceModel, GammaPolicy, SpeculativeConfig, SwitchConstraints, TraceBank,
};
use crate::request::Request;

/// One drawn draft round for a decode sequence.
#[derive(Debug, Clone)]
pub struct Round {
    /// Estimated conditional acceptance per depth (the gate's signal).
    pub a_hat: Vec<f64>,
    /// Draft tokens that commit at full depth; a draft of `g` commits
    /// `min(commits, g)`.
    pub commits: u32,
}

/// Where rounds and expected-acceptance curves come from.
enum Acceptance {
    /// A closed-form per-position acceptance model. Rounds carry the model's
    /// own conditional acceptances as the signal (identical for every
    /// sequence) and a commit count sampled from it.
    Analytic {
        model: AcceptanceModel,
        max_depth: u32,
    },
    /// Real rounds replayed i.i.d. from a measured trace bank.
    Trace(Arc<TraceBank>),
}

impl Acceptance {
    fn load(model: &AcceptanceModel, gamma: u32) -> Result<Self, String> {
        match model {
            AcceptanceModel::TraceRounds { path } => {
                Ok(Self::Trace(Arc::new(TraceBank::load(path)?)))
            }
            other => Ok(Self::Analytic {
                model: other.clone(),
                max_depth: gamma,
            }),
        }
    }

    /// Deepest draft the source carries a signal for.
    fn max_depth(&self) -> u32 {
        match self {
            Self::Analytic { max_depth, .. } => *max_depth,
            Self::Trace(bank) => bank.max_depth,
        }
    }

    /// E[accepted draft tokens] under a homogeneous draft depth `g`.
    fn expected_accepted(&self, g: u32) -> f64 {
        match self {
            Self::Analytic { model, .. } => model.expected_accepted(g),
            Self::Trace(bank) => bank.expected_accepted(g),
        }
    }

    fn draw(&self, rng: &mut StdRng) -> Round {
        match self {
            Self::Analytic { model, max_depth } => Round {
                a_hat: (0..*max_depth as usize).map(|d| model.a_d(d)).collect(),
                commits: model.sample_accepted(*max_depth, rng),
            },
            Self::Trace(bank) => {
                let r = &bank.rounds[rng.gen_range(0..bank.rounds.len())];
                Round {
                    a_hat: r.a_hat.clone(),
                    commits: r.commits,
                }
            }
        }
    }
}

/// `GatedAggregate` switching state under engine constraints, per worker.
#[derive(Debug, Clone, Copy)]
struct AggSwitchState {
    /// Width currently in force (persists between re-evaluations).
    g: u32,
    /// Decode rounds elapsed since the last re-evaluation.
    rounds_since: u32,
    /// Switch cost (seconds) accrued by a width change, to be paid on the
    /// wall time of the next round (the first executed at the new width).
    pending_cost: f64,
}

/// A worker identity for per-worker planner state.
pub type WorkerKey = (usize, usize);

/// Draft-depth planner: owns the speculative config, the acceptance
/// source, the RNG, the optional measured cost table and the per-worker
/// switching state.
pub struct SpecPlanner {
    cfg: SpeculativeConfig,
    acceptance: Acceptance,
    rng: StdRng,
    /// Measured step-cost table when `cfg.measured_cost` is set. Both the
    /// policy's cost curve C(g) and the wall-clock time of decode steps read
    /// from it, so plain-decode and speculative steps are priced
    /// commensurately.
    measured_cost: Option<MeasuredCostTable>,
    /// Per-second buckets of draft-depth decisions:
    /// second -> (sum of per-seq drafts, decode seqs, steps).
    depth_buckets: BTreeMap<u64, (u64, u64, u64)>,
    agg_switch: HashMap<WorkerKey, AggSwitchState>,
}

/// Cost inputs for one planning decision.
pub struct PlanCosts<'a> {
    /// Analytic roofline cost of the decode sub-batch at homogeneous verify
    /// width `1 + g`.
    pub roofline: &'a dyn Fn(u32) -> f64,
    /// Measured-table KV-length correction (0 without a table / `ref_seq_len`).
    pub kv_delta: f64,
    /// bf16 FLOP rate the drafter streams at.
    pub peak: f64,
    /// Memory bandwidth the drafter streams at.
    pub bw: f64,
}

/// One decode sequence's plan for its next step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftPlan {
    pub draft_len: u32,
    /// Full-depth commits of the round drawn for the step; the verify
    /// realises `min(commits, draft_len)`.
    pub commits: u32,
}

impl SpecPlanner {
    /// Build a planner, loading the trace bank / measured table the config
    /// names. A bad path is a configuration error and is returned as such.
    pub fn new(cfg: SpeculativeConfig, seed: u64) -> Result<Self, String> {
        let acceptance = Acceptance::load(&cfg.acceptance, cfg.gamma)?;
        let measured_cost = match &cfg.measured_cost {
            Some(mc) => Some(MeasuredCostTable::load(&mc.path)?),
            None => None,
        };
        Ok(Self {
            cfg,
            acceptance,
            rng: StdRng::seed_from_u64(seed),
            measured_cost,
            depth_buckets: BTreeMap::new(),
            agg_switch: HashMap::new(),
        })
    }

    pub fn config(&self) -> &SpeculativeConfig {
        &self.cfg
    }

    pub fn measured_cost(&self) -> Option<&MeasuredCostTable> {
        self.measured_cost.as_ref()
    }

    /// Sequence length the measured table was benchmarked at, if it says.
    pub fn ref_seq_len(&self) -> Option<u32> {
        self.cfg.measured_cost.as_ref().and_then(|m| m.ref_seq_len)
    }

    /// Draft tokens accepted by a verify of `draft` tokens whose round
    /// committed `commits` at full depth. `None` (no round was planned, e.g.
    /// a request that had no draft) accepts nothing.
    pub fn accepted(draft: u32, commits: Option<u32>) -> u32 {
        commits.map_or(0, |c| c.min(draft))
    }

    /// Drafter time to add to a roofline-priced step whose decode sequences
    /// drafted `draft_widths` (one entry per decode sequence).
    pub fn drafter_seconds(&self, draft_widths: &[u32], peak: f64, bw: f64, verify_c: f64) -> f64 {
        self.cfg
            .ragged_drafter_seconds(draft_widths, peak, bw, verify_c)
    }

    /// Per-switch stall accrued for `worker` by the last width change; paid
    /// once, on the first round executed at the new width.
    pub fn take_pending_switch_cost(&mut self, worker: WorkerKey) -> f64 {
        match self.agg_switch.get_mut(&worker) {
            Some(st) => std::mem::take(&mut st.pending_cost),
            None => 0.0,
        }
    }

    /// Per-second draft-depth series:
    /// (second, mean drafts per decode seq, mean decode batch).
    pub fn depth_series(&self) -> Vec<(u64, f64, f64)> {
        self.depth_buckets
            .iter()
            .map(|(&s, &(drafts, seqs, steps))| {
                (
                    s,
                    drafts as f64 / seqs.max(1) as f64,
                    seqs as f64 / steps.max(1) as f64,
                )
            })
            .collect()
    }

    /// Decide each decode sequence's draft for its NEXT step. Called at the
    /// end of a step — the instant the drafter is about to run and the
    /// decode set is known.
    ///
    /// `dec` is the decode set (running, not prefilling, not finished).
    /// Returns one plan per decode sequence, in `dec` order.
    pub fn plan(
        &mut self,
        worker: WorkerKey,
        dec: &[&Request],
        costs: &PlanCosts<'_>,
        end_time: f64,
    ) -> Vec<DraftPlan> {
        let PlanCosts {
            roofline: roofline_cost,
            kv_delta,
            peak,
            bw,
        } = *costs;
        let n = dec.len();
        if n == 0 {
            return Vec::new();
        }
        // Draw each decode sequence's next round NOW: the gate's per-depth
        // signal and the realised outcome come from the same round.
        let rounds: Vec<Round> = (0..n)
            .map(|_| self.acceptance.draw(&mut self.rng))
            .collect();

        let gamma = self.cfg.gamma.min(self.acceptance.max_depth());
        let drafts: Vec<u32> = match self.cfg.policy {
            GammaPolicy::Fixed => vec![self.cfg.gamma; n],
            GammaPolicy::GoodputBudget | GammaPolicy::GatedBudget | GammaPolicy::GatedAggregate => {
                // Homogeneous verify cost curve C(g) on the live decode
                // batch: all budget policies start here. When a measured
                // cost table is present it is the ONLY price source: a
                // draft depth with no measured rows is not a real candidate
                // (pricing it via the optimistic roofline lets a fantasy
                // width win the argmax), so it carries an INFINITY sentinel
                // and is excluded. The table's `ref_seq_len` KV correction is
                // width-independent, but a constant added to C(g) still
                // moves the goodput-ratio argmax, so the policy must see the
                // same prices the wall clock charges.
                let c_curve: Vec<f64> = (0..=gamma)
                    .map(|g| match &self.measured_cost {
                        Some(table) => match table.step_time(n as u32, g) {
                            Some(t) => (t + kv_delta).max(0.25 * t),
                            None => f64::INFINITY,
                        },
                        None => roofline_cost(g),
                    })
                    .collect();
                let ctx = GateContext {
                    c_curve: &c_curve,
                    gamma,
                    cfg: &self.cfg,
                    peak,
                    bw,
                    n,
                };
                match self.cfg.policy {
                    GammaPolicy::GatedBudget => {
                        let surv = survival_chains(&rounds, gamma);
                        if self.cfg.drafter_is_block() {
                            gated_block_verify(&surv, &ctx)
                        } else {
                            gated_ragged_draft(&surv, &ctx)
                        }
                    }
                    GammaPolicy::GatedAggregate => {
                        // Batch-uniform width from the per-sequence signal.
                        let surv = survival_chains(&rounds, gamma);
                        let raw = aggregate_depth(&surv, &ctx);
                        let g = if self.cfg.switch.is_unconstrained() {
                            raw
                        } else {
                            constrained_aggregate_choice(
                                &mut self.agg_switch,
                                worker,
                                raw,
                                &c_curve,
                                &self.cfg.switch,
                            )
                        };
                        vec![g; n]
                    }
                    _ => {
                        // GoodputBudget: batch-uniform width from the mean
                        // acceptance curve (no per-sequence signal).
                        let e_sum: Vec<f64> = (0..=gamma)
                            .map(|g| n as f64 * self.acceptance.expected_accepted(g))
                            .collect();
                        vec![argmax_goodput(&e_sum, &ctx); n]
                    }
                }
            }
        };

        let e = self
            .depth_buckets
            .entry(end_time.max(0.0) as u64)
            .or_insert((0, 0, 0));
        e.0 += drafts.iter().map(|&d| d as u64).sum::<u64>();
        e.1 += n as u64;
        e.2 += 1;

        drafts
            .into_iter()
            .zip(rounds)
            .map(|(draft_len, r)| DraftPlan {
                draft_len,
                commits: r.commits,
            })
            .collect()
    }
}

/// What the gating policies need to price a candidate allocation.
struct GateContext<'a> {
    /// Homogeneous verify cost at draft depth g (index g), INFINITY where
    /// unmeasured.
    c_curve: &'a [f64],
    gamma: u32,
    cfg: &'a SpeculativeConfig,
    peak: f64,
    bw: f64,
    /// Decode sequences.
    n: usize,
}

impl GateContext<'_> {
    /// Verify cost plus drafter cost at homogeneous depth `g`; INFINITY where
    /// unmeasured.
    fn total_cost(&self, g: u32) -> f64 {
        let cv = self.c_curve[g as usize];
        if cv.is_finite() {
            cv + self
                .cfg
                .drafter_seconds(g, self.n as u32, self.peak, self.bw, cv)
        } else {
            cv
        }
    }
}

/// Estimated survival chains: `surv[k][d]` = P(slots 0..=d all accepted) for
/// sequence `k`, from its round's per-depth confidences (flat beyond the
/// round's signal).
fn survival_chains(rounds: &[Round], gamma: u32) -> Vec<Vec<f64>> {
    rounds
        .iter()
        .map(|r| {
            let mut s = 1.0;
            r.a_hat
                .iter()
                .take(gamma as usize)
                .map(|a| {
                    s *= a.clamp(0.0, 1.0);
                    s
                })
                .collect()
        })
        .collect()
}

/// `e_sum[g]` = Σ_i E[accepted_i | g] from survival chains: expected
/// accepted tokens across the batch at homogeneous depth g.
fn expected_accepted_by_depth(surv: &[Vec<f64>], gamma: u32) -> Vec<f64> {
    let mut e_sum = vec![0.0f64; gamma as usize + 1];
    for chain in surv {
        let mut acc = 0.0;
        for d in 0..(gamma as usize).min(chain.len()) {
            acc += chain[d];
            e_sum[d + 1] += acc;
        }
    }
    e_sum
}

/// Batch-uniform depth maximising `(e_sum[g] + n) / (C(g) + drafter(g))`,
/// skipping unmeasured depths.
fn argmax_goodput(e_sum: &[f64], ctx: &GateContext<'_>) -> u32 {
    let mut best_g = 0u32;
    let mut best_gp = f64::MIN;
    for g in 0..=ctx.gamma {
        let cost = ctx.total_cost(g);
        if !cost.is_finite() {
            continue;
        }
        let gp = (e_sum[g as usize] + ctx.n as f64) / cost.max(1e-12);
        if gp > best_gp {
            best_gp = gp;
            best_g = g;
        }
    }
    best_g
}

/// Batch-adapted draft depth: the width the realizable (aggregate) gate would
/// pick from the per-sequence signal. Choosing the depth globally amortises
/// each pass's shared weight read; ragged allocation is then a refinement
/// within it.
fn aggregate_depth(surv: &[Vec<f64>], ctx: &GateContext<'_>) -> u32 {
    argmax_goodput(&expected_accepted_by_depth(surv, ctx.gamma), ctx)
}

/// Interpolate the homogeneous verify-cost curve at the mean verify width that
/// `t` total verify tokens over `n` sequences implies.
fn verify_cost(c_curve: &[f64], gamma: u32, n: usize, t: f64) -> f64 {
    let w_mean = (t / n as f64 - 1.0).clamp(0.0, gamma as f64);
    let seg = (w_mean.floor() as usize).min(gamma as usize - 1);
    let frac = w_mean - seg as f64;
    if frac > 0.0 {
        c_curve[seg] + frac * (c_curve[seg + 1] - c_curve[seg])
    } else {
        c_curve[seg]
    }
}

/// Autoregressive (MTP-style) gating: ragged *draft* depth. Start every
/// sequence at the global depth `G` (so the per-pass weight reads are already
/// amortised), then greedily *remove* the least-confident deepest slots while
/// that improves goodput. Removing the last sequence from a pass closes it and
/// reclaims its read, which the removal correctly credits — the dual of an
/// add-from-empty greedy that could never justify opening a pass.
fn gated_ragged_draft(surv: &[Vec<f64>], ctx: &GateContext<'_>) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = ctx.n;
    let gd = aggregate_depth(surv, ctx);
    if gd == 0 {
        return vec![0u32; n];
    }
    let pass = |m: u32| ctx.cfg.drafter_pass_seconds(m, ctx.peak, ctx.bw);
    // Start every sequence at G (capped by its round's available signal).
    let mut g: Vec<u32> = (0..n)
        .map(|k| (gd as usize).min(surv[k].len()) as u32)
        .collect();
    let mut nk = vec![0u32; gd as usize]; // nk[d] = #sequences drafting pass d+1
    for &gi in &g {
        for slot in nk.iter_mut().take(gi as usize) {
            *slot += 1;
        }
    }
    let mut drafter: f64 = (0..gd as usize).map(|d| pass(nk[d])).sum();
    let mut expected: f64 = n as f64
        + (0..n)
            .map(|k| (0..g[k] as usize).map(|d| surv[k][d]).sum::<f64>())
            .sum::<f64>();
    let mut t: f64 = g.iter().map(|&gi| (gi + 1) as f64).sum();
    // min-heap on the deepest slot's survival: least confident removed first.
    let mut heap: BinaryHeap<(Reverse<u64>, usize)> = (0..n)
        .filter(|&k| g[k] > 0)
        .map(|k| (Reverse(surv[k][g[k] as usize - 1].to_bits()), k))
        .collect();
    while let Some((Reverse(sb), k)) = heap.pop() {
        let d = g[k] as usize;
        if d == 0 {
            continue;
        }
        let s = f64::from_bits(sb);
        let dmarg = pass(nk[d - 1]) - pass(nk[d - 1] - 1);
        let vc = verify_cost(ctx.c_curve, ctx.gamma, n, t);
        let vp = verify_cost(ctx.c_curve, ctx.gamma, n, t - 1.0);
        let cur = expected / (vc + drafter).max(1e-12);
        let rem = (expected - s) / (vp + drafter - dmarg).max(1e-12);
        if rem > cur {
            g[k] -= 1;
            nk[d - 1] -= 1;
            expected -= s;
            t -= 1.0;
            drafter -= dmarg;
            if g[k] > 0 {
                heap.push((Reverse(surv[k][g[k] as usize - 1].to_bits()), k));
            }
        }
    }
    g
}

/// Block (DFlash-style) gating: the head drafts one uniform block, so the
/// block depth `B` is a single global choice (the batch-adapted depth), and
/// only the *verify* is ragged within it. The drafter cost is the fixed
/// `block(B, n)`, so the per-sequence verify greedy never pays a whole-batch
/// deepening for an individual sequence.
fn gated_block_verify(surv_full: &[Vec<f64>], ctx: &GateContext<'_>) -> Vec<u32> {
    use std::collections::BinaryHeap;
    let n = ctx.n;
    let b = aggregate_depth(surv_full, ctx);
    if b == 0 {
        return vec![0u32; n];
    }
    // Drafter is fixed: one block of depth B over the whole batch.
    let draft_const =
        ctx.cfg
            .drafter_seconds(b, n as u32, ctx.peak, ctx.bw, ctx.c_curve[b as usize]);
    let surv: Vec<&[f64]> = surv_full
        .iter()
        .map(|c| &c[..(b as usize).min(c.len())])
        .collect();
    let mut heap: BinaryHeap<(u64, usize)> = (0..n)
        .filter(|&k| !surv[k].is_empty())
        .map(|k| (surv[k][0].to_bits(), k))
        .collect();
    let mut v = vec![0u32; n]; // ragged verify widths, capped at B
    let mut expected = n as f64;
    let mut t = n as f64;
    while let Some((sb, k)) = heap.pop() {
        let s = f64::from_bits(sb);
        let vc = verify_cost(ctx.c_curve, ctx.gamma, n, t) + draft_const;
        let vn = verify_cost(ctx.c_curve, ctx.gamma, n, t + 1.0) + draft_const;
        if !vn.is_finite() {
            continue;
        }
        let cur = expected / vc.max(1e-12);
        let nxt = (expected + s) / vn.max(1e-12);
        if nxt > cur {
            v[k] += 1;
            expected += s;
            t += 1.0;
            if (v[k] as usize) < surv[k].len() {
                heap.push((surv[k][v[k] as usize].to_bits(), k));
            }
        } else {
            break; // drafter fixed: the bar only rises, so we're done
        }
    }
    v
}

/// Apply the engine's switching constraints to the aggregated gate's raw
/// argmax `raw`. The candidate set is the measured widths (finite entries
/// of `c_curve` — the same exclusion the argmax itself applies), sorted
/// ascending. The first decision a worker ever makes is free (there is no
/// previous width to persist); thereafter the width re-evaluates only
/// every `switch.cooldown_rounds` decode rounds, each re-evaluation walks
/// at most `switch.max_step` candidate indices toward `raw`, and a width
/// change accrues `switch.cost_ms` onto the next round's wall time.
fn constrained_aggregate_choice(
    state: &mut HashMap<WorkerKey, AggSwitchState>,
    key: WorkerKey,
    raw: u32,
    c_curve: &[f64],
    switch: &SwitchConstraints,
) -> u32 {
    let st = match state.get_mut(&key) {
        Some(st) => st,
        None => {
            state.insert(
                key,
                AggSwitchState {
                    g: raw,
                    rounds_since: 0,
                    pending_cost: 0.0,
                },
            );
            return raw;
        }
    };
    st.rounds_since += 1;
    if st.rounds_since >= switch.cooldown_rounds.max(1) {
        st.rounds_since = 0;
        let cands: Vec<u32> = (0..c_curve.len() as u32)
            .filter(|&g| c_curve[g as usize].is_finite())
            .collect();
        let new_g = walk_candidates(&cands, st.g, raw, switch.max_step);
        if new_g != st.g {
            st.g = new_g;
            st.pending_cost += 1e-3 * switch.cost_ms;
        }
    }
    st.g
}

/// Move from `cur` toward `target` through the sorted candidate list,
/// at most `max_step` indices (`None` = land on `target`). A `cur` that
/// is no longer a candidate snaps to the nearest candidate index first.
fn walk_candidates(cands: &[u32], cur: u32, target: u32, max_step: Option<u32>) -> u32 {
    if cands.is_empty() {
        return cur;
    }
    let cur_i = match cands.binary_search(&cur) {
        Ok(i) => i,
        Err(i) => i.min(cands.len() - 1),
    };
    let tgt_i = cands.binary_search(&target).unwrap_or(cur_i);
    let new_i = match max_step {
        None => tgt_i,
        Some(d) => {
            let d = d as usize;
            if tgt_i > cur_i {
                (cur_i + d).min(tgt_i)
            } else {
                cur_i.saturating_sub(d).max(tgt_i)
            }
        }
    };
    cands[new_i]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DrafterCost;

    #[test]
    fn walk_candidates_bounded_and_unbounded() {
        let cands = [0u32, 1, 2, 3, 4, 6, 8];
        // Unbounded: jump straight to the target.
        assert_eq!(walk_candidates(&cands, 0, 8, None), 8);
        // Bounded: at most two INDICES through the list, both directions.
        assert_eq!(walk_candidates(&cands, 0, 8, Some(2)), 2);
        assert_eq!(walk_candidates(&cands, 4, 8, Some(2)), 8); // 4 -> 6 -> 8
        assert_eq!(walk_candidates(&cands, 8, 0, Some(2)), 4);
        // At / near the target: clamp, never overshoot.
        assert_eq!(walk_candidates(&cands, 3, 3, Some(2)), 3);
        assert_eq!(walk_candidates(&cands, 3, 4, Some(2)), 4);
    }

    #[test]
    fn constrained_choice_cooldown_walk_and_cost() {
        // Finite cost at g in {0,1,2,3,4,6,8}; INF (unmeasured) at 5 and 7,
        // so the candidate walk skips them as single index moves.
        let mut c = vec![1.0f64; 9];
        c[5] = f64::INFINITY;
        c[7] = f64::INFINITY;
        let sw = SwitchConstraints {
            cooldown_rounds: 4,
            max_step: Some(2),
            cost_ms: 0.5,
        };
        let mut st: HashMap<WorkerKey, AggSwitchState> = HashMap::new();
        let key = (0usize, 0usize);
        // First decision is free (no previous width to persist).
        assert_eq!(constrained_aggregate_choice(&mut st, key, 0, &c, &sw), 0);
        // Rounds 1..3 of the cooldown hold the width even as the argmax moves.
        for _ in 0..3 {
            assert_eq!(constrained_aggregate_choice(&mut st, key, 8, &c, &sw), 0);
        }
        // Round 4 re-evaluates: walk two indices toward 8 -> g = 2; per-switch
        // cost accrued for the next round to pay.
        assert_eq!(constrained_aggregate_choice(&mut st, key, 8, &c, &sw), 2);
        assert!((st[&key].pending_cost - 0.5e-3).abs() < 1e-12);
        // Hold, then 2 -> 4; hold, then 4 -> 8 (6 and 8 are one index each).
        for _ in 0..3 {
            assert_eq!(constrained_aggregate_choice(&mut st, key, 8, &c, &sw), 2);
        }
        assert_eq!(constrained_aggregate_choice(&mut st, key, 8, &c, &sw), 4);
        for _ in 0..3 {
            assert_eq!(constrained_aggregate_choice(&mut st, key, 8, &c, &sw), 4);
        }
        assert_eq!(constrained_aggregate_choice(&mut st, key, 8, &c, &sw), 8);
        // A re-evaluation that does not change the width accrues no cost.
        let before = st[&key].pending_cost;
        for _ in 0..4 {
            constrained_aggregate_choice(&mut st, key, 8, &c, &sw);
        }
        assert_eq!(st[&key].pending_cost, before);
    }

    fn cfg(policy: GammaPolicy, acceptance: AcceptanceModel) -> SpeculativeConfig {
        SpeculativeConfig {
            gamma: 4,
            acceptance,
            policy,
            measured_cost: None,
            switch: Default::default(),
            drafter: Some(DrafterCost::Fraction { frac: 0.0 }),
        }
    }

    fn costs(roofline: &dyn Fn(u32) -> f64) -> PlanCosts<'_> {
        PlanCosts {
            roofline,
            kv_delta: 0.0,
            peak: 1e15,
            bw: 1e12,
        }
    }

    fn decode_reqs(n: usize) -> Vec<Request> {
        (0..n)
            .map(|i| {
                let mut r = Request::new(format!("r{i}"), 0, 0.0, 64, 32);
                r.num_computed_tokens = 64;
                r.num_output_tokens = 1;
                r
            })
            .collect()
    }

    #[test]
    fn fixed_policy_plans_gamma_and_analytic_rounds_realise_min_commits() {
        let mut p = SpecPlanner::new(
            cfg(GammaPolicy::Fixed, AcceptanceModel::Constant { alpha: 1.0 }),
            1,
        )
        .unwrap();
        let reqs = decode_reqs(3);
        let dec: Vec<&Request> = reqs.iter().collect();
        let plans = p.plan((0, 0), &dec, &costs(&|_| 1.0), 0.0);
        assert_eq!(plans.len(), 3);
        // alpha = 1: every draft token commits, at full depth.
        for pl in &plans {
            assert_eq!(pl.draft_len, 4);
            assert_eq!(pl.commits, 4);
        }
        // A scheduler-trimmed verify realises min(commits, draft).
        assert_eq!(SpecPlanner::accepted(2, Some(plans[0].commits)), 2);
        assert_eq!(SpecPlanner::accepted(4, None), 0);
        assert_eq!(p.depth_series(), vec![(0, 4.0, 3.0)]);
    }

    #[test]
    fn goodput_budget_prefers_deep_drafts_when_verify_is_free() {
        // C(g) constant: goodput (E[acc]+1)/C grows with g, so the argmax is
        // gamma. C(g) steeply increasing with a poor acceptance: argmax 0.
        let reqs = decode_reqs(4);
        let dec: Vec<&Request> = reqs.iter().collect();
        let mut p = SpecPlanner::new(
            cfg(
                GammaPolicy::GoodputBudget,
                AcceptanceModel::Constant { alpha: 0.9 },
            ),
            1,
        )
        .unwrap();
        let plans = p.plan((0, 0), &dec, &costs(&|_| 1.0), 0.0);
        assert!(plans.iter().all(|pl| pl.draft_len == 4));
        let mut p = SpecPlanner::new(
            cfg(
                GammaPolicy::GoodputBudget,
                AcceptanceModel::Constant { alpha: 0.1 },
            ),
            1,
        )
        .unwrap();
        let plans = p.plan((0, 0), &dec, &costs(&|g| 1.0 + 10.0 * g as f64), 0.0);
        assert!(plans.iter().all(|pl| pl.draft_len == 0));
    }

    #[test]
    fn gated_policies_with_a_homogeneous_signal_match_goodput_budget() {
        // With an analytic model every sequence carries the same signal, so
        // the gated policies degenerate to the batch-uniform argmax.
        let reqs = decode_reqs(5);
        let dec: Vec<&Request> = reqs.iter().collect();
        let acc = AcceptanceModel::PerPosition {
            a: vec![0.9, 0.7, 0.3, 0.1],
        };
        let cost = |g: u32| 1.0 + 0.4 * g as f64;
        let mut budget = SpecPlanner::new(cfg(GammaPolicy::GoodputBudget, acc.clone()), 1).unwrap();
        let mut aggregate =
            SpecPlanner::new(cfg(GammaPolicy::GatedAggregate, acc.clone()), 1).unwrap();
        let mut gated = SpecPlanner::new(cfg(GammaPolicy::GatedBudget, acc), 1).unwrap();
        let b = budget.plan((0, 0), &dec, &costs(&cost), 0.0);
        let a = aggregate.plan((0, 0), &dec, &costs(&cost), 0.0);
        let g = gated.plan((0, 0), &dec, &costs(&cost), 0.0);
        let d = |v: &[DraftPlan]| v.iter().map(|p| p.draft_len).collect::<Vec<_>>();
        assert_eq!(d(&b), d(&a));
        assert_eq!(d(&b), d(&g));
        assert!(b[0].draft_len > 0 && b[0].draft_len < 4);
    }
}
