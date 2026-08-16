use rand::Rng;
use serde::Deserialize;

/// How the target accepts a draft. The acceptance model is the knob you vary to
/// study the simulated benefit of speculation under different draft qualities.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AcceptanceModel {
    /// i.i.d. per-position acceptance probability `alpha`. The classic Leviathan
    /// model: E[accepted] = (1 - alpha^(gamma+1)) / (1 - alpha) - 1 over the
    /// draft, geometric in depth.
    Constant { alpha: f64 },
    /// Empirical per-position conditional acceptance `a[d]` = P(accept the draft
    /// token at depth d | reached depth d). Index 0 is the first draft token.
    /// This is the shape measured for real MTP drafters; depths beyond `a.len()`
    /// are treated as never accepted.
    PerPosition { a: Vec<f64> },
    /// Replay real draft rounds from a trace bank (CSV exported from measured
    /// MTP traces). Each round carries the per-depth acceptance the gate could
    /// *estimate* at draft time (via the drafter-logprob calibration) and the
    /// number of draft tokens that *actually* committed, so per-sequence gating
    /// policies see the real joint distribution of signal and outcome,
    /// including calibration error. Rounds are drawn i.i.d. per decode step.
    TraceRounds { path: String },
}

/// One real draft round from the trace bank.
#[derive(Debug, Clone)]
pub struct TraceRound {
    /// Estimated conditional acceptance per depth (the gate's signal).
    pub a_hat: Vec<f64>,
    /// Draft tokens that actually committed at full depth (the outcome).
    /// Truncating the draft to G commits `min(commits, G)`.
    pub commits: u32,
}

/// Loaded trace bank plus its aggregate acceptance curve.
#[derive(Debug, Clone)]
pub struct TraceBank {
    pub rounds: Vec<TraceRound>,
    /// `e_curve[g]` = mean over rounds of `min(commits, g)`: the bank's
    /// E[accepted draft tokens] under homogeneous draft depth g.
    pub e_curve: Vec<f64>,
    pub max_depth: u32,
}

impl TraceBank {
    /// Load from CSV with header `commits,category,a0..a{D-1}`.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("trace bank {path}: {e}"))?;
        let mut lines = text.lines();
        let header = lines.next().ok_or("trace bank: empty file")?;
        let cols: Vec<&str> = header.split(',').collect();
        let depth = cols.iter().filter(|c| c.starts_with('a')).count();
        if cols.first() != Some(&"commits") || depth == 0 {
            return Err(format!("trace bank {path}: unexpected header {header}"));
        }
        let a0 = cols.len() - depth;
        let mut rounds = Vec::new();
        for (i, line) in lines.enumerate() {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != cols.len() {
                return Err(format!("trace bank {path}: bad row {}", i + 2));
            }
            let commits: u32 = f[0].parse().map_err(|e| format!("row {}: {e}", i + 2))?;
            let a_hat = f[a0..]
                .iter()
                .map(|s| s.parse::<f64>().map_err(|e| format!("row {}: {e}", i + 2)))
                .collect::<Result<Vec<f64>, String>>()?;
            rounds.push(TraceRound { a_hat, commits });
        }
        if rounds.is_empty() {
            return Err(format!("trace bank {path}: no rounds"));
        }
        let n = rounds.len() as f64;
        let e_curve = (0..=depth as u32)
            .map(|g| rounds.iter().map(|r| r.commits.min(g) as f64).sum::<f64>() / n)
            .collect();
        Ok(Self {
            rounds,
            e_curve,
            max_depth: depth as u32,
        })
    }

    /// E[accepted draft tokens] under homogeneous draft depth `gamma`.
    pub fn expected_accepted(&self, gamma: u32) -> f64 {
        let g = (gamma.min(self.max_depth)) as usize;
        self.e_curve[g]
    }
}

impl AcceptanceModel {
    /// Conditional acceptance at draft depth `d` (0-based).
    fn a_d(&self, d: usize) -> f64 {
        match self {
            AcceptanceModel::Constant { alpha } => *alpha,
            AcceptanceModel::PerPosition { a } => a.get(d).copied().unwrap_or(0.0),
            // TraceRounds outcomes are realised from drawn rounds in the
            // engine, not from this curve; unreachable in normal operation.
            AcceptanceModel::TraceRounds { .. } => 0.0,
        }
    }

    /// Expected number of *accepted draft* tokens (excluding the always-emitted
    /// bonus token) for a draft of length `gamma`:
    /// `E[L] = sum_{d=0}^{gamma-1} prod_{i<=d} a_i`.
    pub fn expected_accepted(&self, gamma: u32) -> f64 {
        let mut prefix = 1.0;
        let mut e = 0.0;
        for d in 0..gamma as usize {
            prefix *= self.a_d(d);
            e += prefix;
        }
        e
    }

    /// Sample the number of accepted draft tokens in `[0, gamma]`: accept depth
    /// `d` with probability `a_d`, stop at the first rejection.
    pub fn sample_accepted(&self, gamma: u32, rng: &mut impl Rng) -> u32 {
        let mut n = 0;
        for d in 0..gamma as usize {
            if rng.gen::<f64>() < self.a_d(d) {
                n += 1;
            } else {
                break;
            }
        }
        n
    }
}

/// Speculative decoding configuration. Top-level (a serving-strategy choice).
/// When present, each decode step verifies `gamma + 1` tokens per sequence (the
/// cost) and advances by `accepted + 1` tokens (the progress), with `accepted`
/// drawn from `acceptance`.
/// How the draft length is chosen each step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GammaPolicy {
    /// Always use `gamma`.
    #[default]
    Fixed,
    /// Load-aware homogeneous draft depth `G`. At the end of a step (the instant
    /// the drafter is about to run, vLLM-faithful), pick one `G in 0..=gamma` for
    /// the whole decode batch that maximises goodput `(E[accepted|G]+1) / C(G)`,
    /// where `C` is the real roofline cost (MoE-coupon- and MLA-aware) of the
    /// decode sub-batch at verify width `1 + G`, and the drafter is charged for
    /// `G` passes. `G` is sized from the current decode batch, so it tracks the
    /// load (small at the expert-tax shoulder, deep in the memory-bound middle,
    /// back down once compute-bound).
    ///
    /// It prices the decode sub-batch only. Pricing prefill into the same step
    /// does not help: the mandatory prefill is proportional to committed output
    /// so it cancels in the argmax, and routing prefill tokens through the cost
    /// model trips the MoE coupon (verify looks free). Measured against the best
    /// fixed gamma with chunked prefill on (the vLLM-V1 target): within +/-0.5%
    /// when ISL ~ OSL, with a bounded over-speculation tail (up to ~6%) only at
    /// large batch when ISL >> OSL.
    GoodputBudget,
    /// Per-sequence confidence-gated draft depth. Requires `TraceRounds`
    /// acceptance (each decode sequence draws a real round whose per-depth
    /// estimated acceptances are the gate's signal). Greedy: price the
    /// homogeneous verify cost curve C(g) on the live decode batch exactly as
    /// `GoodputBudget` does, then allocate draft slots one at a time in
    /// descending estimated-survival order, accepting a slot only while it
    /// improves expected committed tokens per unit step time (marginal cost
    /// interpolated from the curve; drafter charged on the widest draft
    /// present). With a homogeneous acceptance model every sequence has the
    /// same signal and this degenerates to `GoodputBudget`.
    GatedBudget,
    /// Engine-realizable aggregation of the per-sequence gate. Requires
    /// `TraceRounds` acceptance, like `GatedBudget`: each decode sequence's
    /// drawn round supplies its per-depth estimated acceptances. But the
    /// decision is a SINGLE batch-uniform draft depth:
    /// `g* = argmax_g (Σ_i E[accepted_i | g] + n) / C(g)·(1 + c_draft·g)`,
    /// i.e. the per-sequence expected-accept curves are summed and one width
    /// is chosen for everyone — modelling an engine whose verify width must
    /// be homogeneous across the batch (no ragged verify kernels). Sits
    /// between `GoodputBudget` (bank-mean signal) and `GatedBudget` (ragged
    /// per-sequence widths): same signal as the latter, same shape constraint
    /// as the former.
    GatedAggregate,
}

/// Optional measured step-cost table overriding the analytic roofline in the
/// draft-depth policy decision. CSV with header
/// `batch_size,num_draft_tokens,step_seconds`, one row per measured grid
/// point. See [`crate::compute::MeasuredCostTable`].
#[derive(Debug, Clone, Deserialize)]
pub struct MeasuredCostConfig {
    pub path: String,
    /// Mean KV length (tokens of context per sequence) at which the table's
    /// rows were benchmarked. When set, table-priced decode steps get a
    /// bandwidth-roofline KV-read correction `sum_i (kv_bytes(L_i) -
    /// kv_bytes(ref)) / mem_bw` for the live batch's actual KV lengths `L_i`
    /// — a conservative (lower-bound) recontextualisation, since it only
    /// accounts for the bandwidth cost of the KV delta, not kernel-shape
    /// effects. Unset: table values are used as-is.
    #[serde(default)]
    pub ref_seq_len: Option<u32>,
}

/// Engine-switching constraints for `GatedAggregate`. The real engine cannot
/// re-decide its verify width every round: it re-evaluates only on a cooldown,
/// walks the sorted measured candidate widths a bounded number of indices per
/// re-evaluation, and pays a small stall when the width actually changes.
/// Defaults are fully unconstrained (the raw aggregated gate, bit-identical
/// to the policy without this struct).
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct SwitchConstraints {
    /// Re-evaluate the batch width only every this-many decode rounds
    /// (0 or 1 = every round). Between re-evaluations the previous width
    /// persists.
    #[serde(default)]
    pub cooldown_rounds: u32,
    /// Per re-evaluation, move at most this many indices through the sorted
    /// candidate-width list toward the argmax. `None` = jump straight there.
    #[serde(default)]
    pub max_step: Option<u32>,
    /// Per-switch cost in milliseconds, added to the wall time of the first
    /// round executed at the new width.
    #[serde(default)]
    pub cost_ms: f64,
}

impl SwitchConstraints {
    /// Whether this is the no-op (raw policy) configuration.
    pub fn is_unconstrained(&self) -> bool {
        self.cooldown_rounds <= 1 && self.max_step.is_none() && self.cost_ms == 0.0
    }
}

/// Cost of *producing* the speculated tokens, priced as an absolute time per
/// step (seconds) rather than a fixed fraction of the verify. Both roofline
/// variants charge a bf16 weight sweep over the drafter's active params `P`; the
/// only difference is how many times that sweep is paid for a draft of `gamma`.
/// See the drafter-roofline section of the post.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DrafterCost {
    /// Legacy scalar: overhead = `frac · gamma · verify_cost`. Reproduces the old
    /// multiplicative `(1 + frac·gamma)` exactly when added to the verify cost.
    Fraction { frac: f64 },
    /// Autoregressive head (MTP / EAGLE): `gamma` serial single-token passes,
    /// each streaming the drafter's weights, so cost is LINEAR in `gamma` and the
    /// vocabulary projection is paid `gamma` times. The dense part (tied vocab
    /// head + attention + fusion) is read once per pass; the layer's MoE FFN
    /// coupon-collects routed experts over the B-token batch exactly like the
    /// verifier, so its resident read grows with batch while its active FLOPs
    /// stay sparse. Per pass: `max(2·active·B / peak, 2·resident(B) / bw)`.
    Autoregressive {
        /// Batch-independent params read once per pass: tied vocab head plus the
        /// layer's attention and fusion projections.
        dense_params: f64,
        /// Params per routed expert in the layer's MoE FFN.
        expert_params: f64,
        /// Total routed experts `E` (coupon-collector pool).
        num_experts: u32,
        /// Routed experts activated per token `k`.
        experts_per_tok: u32,
        /// Always-resident shared experts.
        shared_experts: u32,
    },
    /// Block-parallel head (DFlash): one diffusion forward over a FIXED block of
    /// `block` positions, chosen at deploy time, not per step. Weights stream
    /// once, so `t = max(2·P·B·block / peak, 2·P / bw)`, independent of how deep
    /// the policy actually verifies. The per-step lever is the verify, not the
    /// draft, so a shallow verify still pays the full block.
    BlockParallel { params: f64, block: u32 },
}

impl DrafterCost {
    /// Drafter time per step (seconds) for a draft of `gamma` over a decode batch
    /// of `batch` sequences. `peak`/`bw` are the bf16 FLOP rate and memory
    /// bandwidth the drafter streams at; `verify_c` is the verify step cost the
    /// `Fraction` variant scales (ignored by the roofline variants).
    pub fn seconds(&self, gamma: u32, batch: u32, peak: f64, bw: f64, verify_c: f64) -> f64 {
        if gamma == 0 {
            return 0.0;
        }
        let (g, b) = (gamma as f64, batch as f64);
        match *self {
            DrafterCost::Fraction { frac } => frac * g * verify_c,
            DrafterCost::Autoregressive {
                dense_params,
                expert_params,
                num_experts,
                experts_per_tok,
                shared_experts,
            } => {
                let e = num_experts as f64;
                let k = experts_per_tok as f64;
                let shared = shared_experts as f64;
                // Coupon-collector: distinct routed experts touched by b·k draws.
                let loaded = if e > 0.0 {
                    e * (1.0 - (1.0 - 1.0 / e).powf(b * k))
                } else {
                    0.0
                };
                let resident = dense_params + (shared + loaded) * expert_params; // bytes
                let active = dense_params + (k + shared) * expert_params; // FLOPs/token
                let per_pass = ((2.0 * active * b) / peak).max((2.0 * resident) / bw);
                g * per_pass
            }
            DrafterCost::BlockParallel { params, block } => {
                // Fixed block: always the full `block`, never the requested `g`.
                let bl = block as f64;
                ((2.0 * params * b * bl) / peak).max((2.0 * params) / bw)
            }
        }
    }

    /// One autoregressive draft pass over `batch` sequences (a single token each,
    /// coupon-collector experts). The building block of the ragged cost; zero for
    /// non-autoregressive heads.
    pub fn pass_seconds(&self, batch: u32, peak: f64, bw: f64) -> f64 {
        match *self {
            DrafterCost::Autoregressive { .. } if batch > 0 => {
                self.seconds(1, batch, peak, bw, 0.0)
            }
            _ => 0.0,
        }
    }

    /// Drafter time for a *ragged* per-sequence allocation: `draft_widths[i]` is
    /// the draft depth of sequence `i`. An autoregressive head runs a shrinking
    /// sub-batch per depth — sequence `i` is only in passes `1..=draft_widths[i]`,
    /// so the cost is `Σ_k pass(n_k)` with `n_k` the number still drafting at
    /// depth `k`. A block head drafts one uniform block sized to the deepest draft
    /// (it cannot ragged-draft), so the ragged widths change only the verify, and
    /// the drafter cost is the block at `max(draft_widths)`. Reduces to
    /// `seconds(g, n)` when all widths equal `g`.
    pub fn ragged_seconds(&self, draft_widths: &[u32], peak: f64, bw: f64, verify_c: f64) -> f64 {
        let n = draft_widths.len();
        let maxw = draft_widths.iter().copied().max().unwrap_or(0);
        if n == 0 || maxw == 0 {
            return 0.0;
        }
        match *self {
            DrafterCost::Fraction { frac } => frac * maxw as f64 * verify_c,
            DrafterCost::Autoregressive { .. } => (1..=maxw)
                .map(|k| {
                    let n_k = draft_widths.iter().filter(|&&w| w >= k).count() as u32;
                    self.pass_seconds(n_k, peak, bw)
                })
                .sum(),
            DrafterCost::BlockParallel { .. } => self.seconds(maxw, n as u32, peak, bw, 0.0),
        }
    }

    /// Whether the head drafts a single block (cannot ragged-draft): the gating
    /// policy must choose one block depth and ragged the *verify*, not the draft.
    pub fn is_block(&self) -> bool {
        matches!(self, DrafterCost::BlockParallel { .. })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft length under `Fixed`; the maximum candidate `G` under `GoodputBudget`.
    /// The verify pass processes `gamma + 1` positions per sequence.
    pub gamma: u32,
    /// Acceptance model (the quantity to vary).
    pub acceptance: AcceptanceModel,
    /// Draft-length policy. Defaults to `Fixed`.
    #[serde(default)]
    pub policy: GammaPolicy,
    /// Drafter overhead as a fraction of the verify step *per draft token*,
    /// charged on each speculated decode step (total overhead = frac × gamma).
    /// ~0.0 for our validation (free MTP head). Defaults to 0.
    #[serde(default)]
    pub draft_cost_frac: f64,
    /// Optional measured step-cost table. When set, the policy decision's
    /// cost curve `C(g)` is looked up from the table (at the current decode
    /// batch size) instead of the analytic roofline, and pure-decode steps'
    /// wall-clock time is priced from the same table (plain decode reads the
    /// `num_draft_tokens = 1` rows), so speculative and no-spec steps are
    /// commensurate. Entries the table lacks fall back to the roofline.
    /// Measured rows embody the full engine step including the drafter, so
    /// `draft_cost_frac` should be 0 with a table (it is not applied to
    /// table-priced steps).
    #[serde(default)]
    pub measured_cost: Option<MeasuredCostConfig>,
    /// `GatedAggregate` switching constraints (cooldown / bounded walk /
    /// per-switch cost). Default: unconstrained — identical to the raw
    /// aggregated gate. Ignored by the other policies.
    #[serde(default)]
    pub switch: SwitchConstraints,
    /// Drafter cost model. When set, it prices the drafter as an absolute time
    /// added to the verify step (roofline-aware, batch-dependent), replacing the
    /// scalar `draft_cost_frac`. When unset, the drafter falls back to the legacy
    /// `Fraction { frac: draft_cost_frac }` so old configs are unchanged.
    #[serde(default)]
    pub drafter: Option<DrafterCost>,
}

impl SpeculativeConfig {
    /// Drafter time (seconds) for this step, dispatching to the configured
    /// `drafter` model or the legacy `draft_cost_frac` fraction. Added to the
    /// verify cost wherever the engine or a budget policy prices a draft.
    pub fn drafter_seconds(
        &self,
        gamma: u32,
        batch: u32,
        peak: f64,
        bw: f64,
        verify_c: f64,
    ) -> f64 {
        self.resolved_drafter()
            .seconds(gamma, batch, peak, bw, verify_c)
    }

    /// Ragged-allocation drafter time for `draft_widths` (one entry per decode
    /// sequence). See [`DrafterCost::ragged_seconds`].
    pub fn ragged_drafter_seconds(
        &self,
        draft_widths: &[u32],
        peak: f64,
        bw: f64,
        verify_c: f64,
    ) -> f64 {
        self.resolved_drafter()
            .ragged_seconds(draft_widths, peak, bw, verify_c)
    }

    /// True if the head drafts a single block (cannot ragged-draft); the gating
    /// policy then chooses one block depth and raggeds the verify instead.
    pub fn drafter_is_block(&self) -> bool {
        self.resolved_drafter().is_block()
    }

    /// One autoregressive draft pass over `batch` sequences; the per-depth
    /// building block the AR gating greedy adds incrementally.
    pub fn drafter_pass_seconds(&self, batch: u32, peak: f64, bw: f64) -> f64 {
        self.resolved_drafter().pass_seconds(batch, peak, bw)
    }

    fn resolved_drafter(&self) -> DrafterCost {
        self.drafter.unwrap_or(DrafterCost::Fraction {
            frac: self.draft_cost_frac,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn constant_matches_geometric_expectation() {
        let m = AcceptanceModel::Constant { alpha: 0.8 };
        // sum_{d=1}^{4} 0.8^d = 0.8+0.64+0.512+0.4096
        let expected = 0.8 + 0.64 + 0.512 + 0.4096;
        assert!((m.expected_accepted(4) - expected).abs() < 1e-9);
    }

    #[test]
    fn per_position_expectation_and_sampling() {
        let m = AcceptanceModel::PerPosition {
            a: vec![0.84, 0.67, 0.54, 0.45],
        };
        // 0.84 + 0.84*0.67 + 0.84*0.67*0.54 + ...
        let e = m.expected_accepted(4);
        assert!(e > 1.8 && e < 1.9, "got {e}");
        // sample mean converges to expectation
        let mut rng = StdRng::seed_from_u64(0);
        let n = 200_000;
        let mean: f64 = (0..n)
            .map(|_| m.sample_accepted(4, &mut rng) as f64)
            .sum::<f64>()
            / n as f64;
        assert!((mean - e).abs() < 0.02, "sample mean {mean} vs {e}");
    }
}
