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
}

impl AcceptanceModel {
    /// Conditional acceptance at draft depth `d` (0-based).
    fn a_d(&self, d: usize) -> f64 {
        match self {
            AcceptanceModel::Constant { alpha } => *alpha,
            AcceptanceModel::PerPosition { a } => a.get(d).copied().unwrap_or(0.0),
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
        let mean: f64 =
            (0..n).map(|_| m.sample_accepted(4, &mut rng) as f64).sum::<f64>() / n as f64;
        assert!((mean - e).abs() < 0.02, "sample mean {mean} vs {e}");
    }
}
