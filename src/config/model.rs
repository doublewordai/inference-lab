//! Architecture cost model.
//!
//! A [`ModelSpec`] describes a model as the two things the roofline needs:
//! its per-token weight traffic ([`WeightStream`]s, one per precision, with
//! optional MoE routing) and its token-mixing layers ([`LayerClass`]es,
//! each a count of identical layers with a KV / state footprint and an
//! attention span). Every architecture the simulator serves — dense GQA,
//! sliding-window hybrids, MLA with or without compressed history, linear
//! attention / SSM hybrids, any of them MoE — is a composition of these, so
//! nothing in the simulator core knows an architecture by name.
//!
//! Compute is described as *precision-homogeneous streams*: the engine
//! prices each stream at its own FLOP rate as `max(flops/rate, bytes/bw)`
//! and sums the streams (kernels are serial). Attention FLOPs and KV reads
//! attach to `attention_precision`.

use crate::config::Precision;
use serde::Deserialize;

fn default_attention_precision() -> Precision {
    Precision::Bf16
}

fn default_activation_bytes() -> u32 {
    2
}

/// A model, as the roofline sees it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub name: String,
    /// Residual-stream width. Sizes TP collectives, and prices MLA
    /// attention whose head shape is not given.
    pub hidden_dim: u32,
    /// Maximum context the architecture supports (`max_position_embeddings`).
    pub max_seq_len: u32,
    /// Precision the attention score / AV matmuls run at (softmax-sensitive
    /// kernels run bf16 even when weights and KV are narrower). KV reads are
    /// charged against this stream's memory budget.
    #[serde(default = "default_attention_precision")]
    pub attention_precision: Precision,
    /// Bytes per activation element on the wire (TP / EP collectives).
    #[serde(default = "default_activation_bytes")]
    pub activation_bytes: u32,
    /// Per-token GEMM weight streams. At least one.
    pub weights: Vec<WeightStream>,
    /// Token-mixing layer classes. At least one.
    pub layers: Vec<LayerClass>,
}

/// Weights read HBM -> SM every forward pass, at one precision.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightStream {
    pub precision: Precision,
    /// Parameters touched per token (matmul FLOPs = 2 × this).
    pub active_params: u64,
    /// Parameters resident in HBM for this stream.
    pub resident_params: u64,
    /// MoE routing on this stream: the resident read per step then follows
    /// coupon-collector growth with the step's tokens instead of being the
    /// constant active footprint.
    #[serde(default)]
    pub routing: Option<Routing>,
}

/// Expert routing for a weight stream.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    /// Routed-expert pool size.
    pub routed_experts: u32,
    /// Routed experts each token is dispatched to.
    pub experts_per_tok: u32,
    /// Layers that perform routing (EP all-to-all count = 2 × this).
    pub moe_layers: u32,
}

/// Indexer of a sparse-attention (DSA) layer: scores every candidate of the
/// history path with a small attention head and keeps its own KV.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub kv_precision: Precision,
}

/// The long-range path of an MLA layer: the whole history at stride
/// `compress_ratio` (1 = every position), all of it or the `index_topk`
/// positions an `indexer` selects.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct History {
    /// Positions per stored history entry; 1 = uncompressed.
    pub compress_ratio: u32,
    /// Cap on attended history entries; 0 = all of them.
    #[serde(default)]
    pub index_topk: u32,
    #[serde(default)]
    pub indexer: Option<Indexer>,
}

/// A class of identical token-mixing layers.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LayerClass {
    /// GQA / MHA softmax attention with a growing KV cache.
    Attention {
        count: u32,
        heads: u32,
        head_dim: u32,
        kv_heads: u32,
        /// K and V share one tensor (Gemma-4 `attention_k_eq_v`): KV per
        /// token is half of K+V.
        #[serde(default)]
        kv_shared: bool,
        /// Sliding window: attend to and store only the last `window`
        /// tokens. 0 = full context.
        #[serde(default)]
        window: u32,
        kv_precision: Precision,
    },
    /// Multi-head latent attention: KV per token is one latent vector plus
    /// an optional RoPE'd key slice. Without a `history` path the layer
    /// attends its `window` recent tokens, or the whole context when
    /// `window` is 0. With one (DeepSeek sparse / compressed attention) it
    /// attends its `window` recent tokens (possibly none) plus the history
    /// path.
    Mla {
        count: u32,
        latent_dim: u32,
        #[serde(default)]
        rope_dim: u32,
        kv_precision: Precision,
        /// Recent tokens attended directly; 0 = none (with a history path)
        /// or the whole context (without one).
        #[serde(default)]
        window: u32,
        #[serde(default)]
        history: Option<History>,
        /// Attention head shape, for the score/AV FLOP count
        /// (`2 × heads × (qk_head_dim + v_head_dim)` per query-key pair).
        /// When absent, attention is priced as `4 × hidden_dim` per pair.
        #[serde(default)]
        heads: Option<u32>,
        #[serde(default)]
        qk_head_dim: Option<u32>,
        #[serde(default)]
        v_head_dim: Option<u32>,
    },
    /// Linear attention / SSM (GatedDeltaNet, Mamba, KDA): a fixed
    /// per-sequence state, no context-scaling work.
    Linear {
        count: u32,
        /// State bytes per layer per sequence (recurrent + conv state at
        /// its stored precision).
        state_bytes: u64,
    },
}

/// Expected number of distinct experts touched when `draws` independent
/// uniform routing draws are made over a pool of `num_experts`
/// (coupon-collector growth). This is what gives MoE its shallow
/// intensity-vs-batch slope, distant knee, and low-batch "expert tax".
pub fn expected_distinct_experts(num_experts: u32, draws: f64) -> f64 {
    let e = num_experts as f64;
    if e <= 0.0 || draws <= 0.0 {
        return 0.0;
    }
    e * (1.0 - (1.0 - 1.0 / e).powf(draws))
}

impl WeightStream {
    fn bytes_per_value(&self) -> f64 {
        self.precision.bytes_per_value()
    }

    /// Params read for a step over `num_tokens` tokens. Without routing (or
    /// with a pool no larger than the per-token fan-out) this is the active
    /// footprint. With routing, uniform routing over
    /// `num_tokens × experts_per_tok` draws makes the expected distinct
    /// routed experts grow from the active footprint toward all-experts
    /// resident; per-routed-expert params `w` and always-resident shared
    /// params are recovered from the active (`k·w + shared`) and resident
    /// (`E·w + shared`) footprints.
    fn params_read_per_step(&self, num_tokens: u32) -> u64 {
        let Some(r) = self.routing else {
            return self.active_params;
        };
        if r.routed_experts <= r.experts_per_tok || num_tokens == 0 {
            return self.active_params;
        }
        let (e, k) = (r.routed_experts as f64, r.experts_per_tok as f64);
        let w = self.resident_params.saturating_sub(self.active_params) as f64 / (e - k);
        let shared = (self.active_params as f64 - k * w).max(0.0);
        let loaded = expected_distinct_experts(r.routed_experts, num_tokens as f64 * k);
        (shared + loaded * w).round() as u64
    }
}

impl LayerClass {
    pub fn count(&self) -> u32 {
        match self {
            LayerClass::Attention { count, .. }
            | LayerClass::Mla { count, .. }
            | LayerClass::Linear { count, .. } => *count,
        }
    }

    /// Positions attended per query at context length `t` (one layer).
    fn attended(&self, t: u32) -> u32 {
        match self {
            LayerClass::Attention { window, .. } => windowed(t, *window),
            LayerClass::Mla {
                window, history, ..
            } => match history {
                None => windowed(t, *window),
                Some(h) => {
                    let entries = t / h.compress_ratio.max(1);
                    let attended_history = if h.index_topk > 0 {
                        entries.min(h.index_topk)
                    } else {
                        entries
                    };
                    t.min(*window) + attended_history
                }
            },
            LayerClass::Linear { .. } => 0,
        }
    }

    /// KV positions stored at context length `t` (one layer).
    fn stored(&self, t: u32) -> u32 {
        match self {
            LayerClass::Attention { window, .. } => windowed(t, *window),
            LayerClass::Mla {
                window, history, ..
            } => match history {
                None => windowed(t, *window),
                Some(h) => t.min(*window) + t / h.compress_ratio.max(1),
            },
            LayerClass::Linear { .. } => 0,
        }
    }

    /// KV bytes per stored position (one layer).
    fn kv_bytes_per_position(&self) -> f64 {
        match self {
            LayerClass::Attention {
                head_dim,
                kv_heads,
                kv_shared,
                kv_precision,
                ..
            } => {
                let tensors = if *kv_shared { 1.0 } else { 2.0 };
                tensors * *kv_heads as f64 * *head_dim as f64 * kv_precision.bytes_per_value()
            }
            LayerClass::Mla {
                latent_dim,
                rope_dim,
                kv_precision,
                ..
            } => (*latent_dim as f64 + *rope_dim as f64) * kv_precision.bytes_per_value(),
            LayerClass::Linear { .. } => 0.0,
        }
    }

    /// Indexer KV bytes for a `t`-token context (one layer): one entry per
    /// history entry, `head_dim` wide.
    fn indexer_kv_bytes(&self, t: u32) -> f64 {
        match self {
            LayerClass::Mla {
                history:
                    Some(History {
                        compress_ratio,
                        indexer: Some(ix),
                        ..
                    }),
                ..
            } => {
                (t / (*compress_ratio).max(1)) as f64
                    * ix.head_dim as f64
                    * ix.kv_precision.bytes_per_value()
            }
            _ => 0.0,
        }
    }

    /// Attention FLOPs for `s` new tokens against a `t`-token context (one
    /// layer): score + AV over the attended positions, plus indexer scoring
    /// over every compressed candidate.
    fn attention_flops(&self, hidden_dim: u32, s: u32, t: u32) -> f64 {
        let (s, attended) = (s as f64, self.attended(t) as f64);
        match self {
            LayerClass::Attention {
                heads, head_dim, ..
            } => 4.0 * *heads as f64 * *head_dim as f64 * s * attended,
            LayerClass::Mla {
                heads,
                qk_head_dim,
                v_head_dim,
                history,
                ..
            } => {
                let per_pair = match (heads, qk_head_dim, v_head_dim) {
                    (Some(h), Some(qk), Some(v)) => 2.0 * *h as f64 * (*qk as f64 + *v as f64),
                    _ => 4.0 * hidden_dim as f64,
                };
                // Indexer scoring over every history entry: 2 × head_dim
                // FLOPs per (query, candidate, head).
                let indexer_flops = match history {
                    Some(History {
                        compress_ratio,
                        indexer: Some(ix),
                        ..
                    }) => {
                        2.0 * ix.heads as f64
                            * ix.head_dim as f64
                            * s
                            * (t / (*compress_ratio).max(1)) as f64
                    }
                    _ => 0.0,
                };
                per_pair * s * attended + indexer_flops
            }
            LayerClass::Linear { .. } => 0.0,
        }
    }

    fn state_bytes(&self) -> u64 {
        match self {
            LayerClass::Linear { state_bytes, .. } => *state_bytes,
            _ => 0,
        }
    }
}

fn windowed(t: u32, window: u32) -> u32 {
    if window > 0 {
        t.min(window)
    } else {
        t
    }
}

impl ModelSpec {
    /// Structural checks a spec must pass before it prices anything.
    pub fn validate(&self) -> Result<(), String> {
        let err = |m: String| Err(format!("model {:?}: {m}", self.name));
        if self.hidden_dim == 0 {
            return err("hidden_dim must be positive".into());
        }
        if self.max_seq_len == 0 {
            return err("max_seq_len must be positive".into());
        }
        if self.weights.is_empty() {
            return err("at least one weight stream is required".into());
        }
        for w in &self.weights {
            if w.resident_params < w.active_params {
                return err(format!(
                    "{:?} stream: resident_params ({}) < active_params ({})",
                    w.precision, w.resident_params, w.active_params
                ));
            }
            if let Some(r) = w.routing {
                if r.experts_per_tok == 0 || r.moe_layers == 0 {
                    return err("routing needs experts_per_tok and moe_layers > 0".into());
                }
                if r.routed_experts != 0 && r.routed_experts < r.experts_per_tok {
                    return err("routing: routed_experts < experts_per_tok".into());
                }
            }
        }
        if self.layers.is_empty() {
            return err("at least one layer class is required".into());
        }
        for l in &self.layers {
            if l.count() == 0 {
                return err("layer class count must be positive".into());
            }
            match l {
                LayerClass::Attention {
                    heads,
                    head_dim,
                    kv_heads,
                    ..
                } => {
                    if *heads == 0 || *head_dim == 0 || *kv_heads == 0 {
                        return err("attention layers need heads, head_dim, kv_heads > 0".into());
                    }
                }
                LayerClass::Mla {
                    latent_dim,
                    rope_dim,
                    history,
                    heads,
                    qk_head_dim,
                    v_head_dim,
                    ..
                } => {
                    if *latent_dim + *rope_dim == 0 {
                        return err("mla layers need latent_dim (+ rope_dim) > 0".into());
                    }
                    if let Some(h) = history {
                        if h.compress_ratio == 0 {
                            return err("mla history: compress_ratio must be >= 1".into());
                        }
                        if let Some(ix) = h.indexer {
                            if ix.heads == 0 || ix.head_dim == 0 {
                                return err("mla indexer needs heads and head_dim > 0".into());
                            }
                        }
                    }
                    let shape = [heads, qk_head_dim, v_head_dim];
                    if shape.iter().any(|x| x.is_some()) && shape.iter().any(|x| x.is_none()) {
                        return err(
                            "mla: give all of heads, qk_head_dim, v_head_dim or none".into()
                        );
                    }
                }
                LayerClass::Linear { state_bytes, .. } => {
                    if *state_bytes == 0 {
                        return err("linear layers need state_bytes > 0".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Token-mixing layers (sizes the TP all-reduce count).
    pub fn num_layers(&self) -> u32 {
        self.layers.iter().map(|l| l.count()).sum()
    }

    /// Per-token matmul FLOPs by precision (one entry per weight stream; the
    /// engine accumulates entries that share a precision).
    pub fn matmul_flops_per_token_by_prec(&self) -> Vec<(Precision, u64)> {
        self.weights
            .iter()
            .map(|w| (w.precision, 2 * w.active_params))
            .collect()
    }

    /// Weight bytes read HBM -> SM in one forward pass over `num_tokens`
    /// tokens, by precision.
    pub fn weight_bytes_per_step_by_prec(&self, num_tokens: u32) -> Vec<(Precision, u64)> {
        self.weights
            .iter()
            .map(|w| {
                (
                    w.precision,
                    (w.params_read_per_step(num_tokens) as f64 * w.bytes_per_value()) as u64,
                )
            })
            .collect()
    }

    /// Bytes of resident weights in HBM (all streams share one HBM).
    pub fn weight_residency_bytes(&self) -> u64 {
        self.weights
            .iter()
            .map(|w| (w.resident_params as f64 * w.bytes_per_value()) as u64)
            .sum()
    }

    /// Self-attention compute (score plus AV) for `new_tokens` against
    /// `attended_tokens` of context, summed across layers.
    pub fn attention_flops(&self, new_tokens: u32, attended_tokens: u32) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.count() as f64 * l.attention_flops(self.hidden_dim, new_tokens, attended_tokens)
            })
            .sum::<f64>() as u64
    }

    /// Bytes of attention KV read per decode step for a `seq_len`-token
    /// sequence (windows, compression and indexers applied). Fixed
    /// per-sequence state is `per_sequence_state_bytes`, charged separately.
    pub fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.count() as f64
                    * (l.kv_bytes_per_position() * l.attended(seq_len) as f64
                        + l.indexer_kv_bytes(seq_len))
            })
            .sum::<f64>() as u64
    }

    /// Bytes of resident attention KV for a `seq_len`-token sequence.
    /// Monotone non-decreasing in `seq_len`; the KV cache manager quantises
    /// it into blocks. Fixed per-sequence state is `per_sequence_state_bytes`.
    pub fn kv_storage_bytes(&self, seq_len: u32) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.count() as f64
                    * (l.kv_bytes_per_position() * l.stored(seq_len) as f64
                        + l.indexer_kv_bytes(seq_len))
            })
            .sum::<f64>() as u64
    }

    /// Fixed per-sequence state bytes (linear / SSM layers), independent of
    /// context length: reserved for the sequence's lifetime and read once
    /// per sequence per step.
    pub fn per_sequence_state_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.count() as u64 * l.state_bytes())
            .sum()
    }

    /// Activation bytes per token in one TP all-reduce.
    pub fn allreduce_bytes_per_token(&self) -> u64 {
        self.hidden_dim as u64 * self.activation_bytes as u64
    }

    /// TP all-reduces per forward pass (post-attention + post-MLP per layer).
    pub fn num_tp_allreduces_per_pass(&self) -> u32 {
        2 * self.num_layers()
    }

    /// Activation bytes per token sent one way in an EP all-to-all: each
    /// token is dispatched to `experts_per_tok` experts, one full
    /// hidden-wide activation each. Zero without routing.
    pub fn alltoall_bytes_per_token(&self) -> u64 {
        self.weights
            .iter()
            .filter_map(|w| w.routing)
            .map(|r| {
                r.experts_per_tok as u64 * self.hidden_dim as u64 * self.activation_bytes as u64
            })
            .sum()
    }

    /// EP all-to-alls per forward pass (dispatch + combine per MoE layer).
    pub fn num_ep_alltoalls_per_pass(&self) -> u32 {
        self.weights
            .iter()
            .filter_map(|w| w.routing)
            .map(|r| 2 * r.moe_layers)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense GQA model: `params` weights at `precision`, `layers` layers of
    /// `heads` × `head_dim` with `kv_heads` KV heads.
    pub(crate) fn dense(
        params: u64,
        precision: Precision,
        layers: u32,
        hidden: u32,
        heads: u32,
        kv_heads: u32,
    ) -> ModelSpec {
        ModelSpec {
            name: "dense".into(),
            hidden_dim: hidden,
            max_seq_len: 8192,
            attention_precision: precision,
            activation_bytes: 2,
            weights: vec![WeightStream {
                precision,
                active_params: params,
                resident_params: params,
                routing: None,
            }],
            layers: vec![LayerClass::Attention {
                count: layers,
                heads,
                head_dim: hidden / heads,
                kv_heads,
                kv_shared: false,
                window: 0,
                kv_precision: precision,
            }],
        }
    }

    // Minimal MoE stream: 256 routed experts, 8 active per token, one shared
    // expert, 1000 params per expert. active = (8+1)*1000, resident = (256+1)*1000.
    fn moe_stream(routed: u32) -> WeightStream {
        WeightStream {
            precision: Precision::Fp4,
            active_params: 9000,
            resident_params: 257_000,
            routing: Some(Routing {
                routed_experts: routed,
                experts_per_tok: 8,
                moe_layers: 4,
            }),
        }
    }

    #[test]
    fn expert_loading_grows_from_active_to_resident_with_batch() {
        let w = moe_stream(256);
        let one = w.params_read_per_step(1);
        let many = w.params_read_per_step(100_000);
        assert!(
            (one as i64 - 9000).abs() < 500,
            "single-token ≈ active, got {one}"
        );
        assert_eq!(many, 257_000, "large batch loads every expert");
        let mut prev = 0;
        for n in [1u32, 8, 32, 128, 512, 2048, 8192] {
            let v = w.params_read_per_step(n);
            assert!(v >= prev, "expert load must not shrink with batch");
            prev = v;
        }
    }

    #[test]
    fn unset_routed_experts_falls_back_to_constant() {
        let w = moe_stream(0);
        assert_eq!(w.params_read_per_step(1), 9000);
        assert_eq!(w.params_read_per_step(100_000), 9000);
    }

    #[test]
    fn dense_kv_and_flops_match_closed_forms() {
        // 32 layers, hidden 4096, 32 heads (head_dim 128), 8 KV heads, bf16:
        // KV/token/layer = 2 * 8 * 128 * 2 = 4096 B; x32 = 131072 B/token.
        let m = dense(7_000_000_000, Precision::Bf16, 32, 4096, 32, 8);
        assert_eq!(m.kv_storage_bytes(100), 131_072 * 100);
        assert_eq!(m.kv_bytes_read_per_decode_step(100), 131_072 * 100);
        // 4 * heads*head_dim * S * T per layer = 4 * 4096 * 1 * 100 * 32.
        assert_eq!(m.attention_flops(1, 100), 4 * 4096 * 100 * 32);
        assert_eq!(m.weight_residency_bytes(), 14_000_000_000);
        assert_eq!(m.per_sequence_state_bytes(), 0);
        assert_eq!(m.num_tp_allreduces_per_pass(), 64);
        assert_eq!(m.num_ep_alltoalls_per_pass(), 0);
    }

    #[test]
    fn sliding_window_layers_cap_kv_and_attention() {
        let mut m = dense(1000, Precision::Bf16, 4, 16, 4, 2);
        // 2 full layers + 2 windowed (window 8); per layer 2*2*4*2 = 32 B/token.
        m.layers = vec![
            LayerClass::Attention {
                count: 2,
                heads: 4,
                head_dim: 4,
                kv_heads: 2,
                kv_shared: false,
                window: 0,
                kv_precision: Precision::Bf16,
            },
            LayerClass::Attention {
                count: 2,
                heads: 4,
                head_dim: 4,
                kv_heads: 2,
                kv_shared: false,
                window: 8,
                kv_precision: Precision::Bf16,
            },
        ];
        // full: 32 * 2 * 10 = 640; windowed: 32 * 2 * min(10, 8) = 512.
        assert_eq!(m.kv_storage_bytes(10), 1_152);
        assert_eq!(m.kv_bytes_read_per_decode_step(10), 1_152);
        // Storage stops growing past the window on the windowed layers.
        assert_eq!(m.kv_storage_bytes(1000) - m.kv_storage_bytes(999), 64);
        // Shared K/V halves the footprint.
        if let LayerClass::Attention { kv_shared, .. } = &mut m.layers[0] {
            *kv_shared = true;
        }
        assert_eq!(m.kv_storage_bytes(10), 320 + 512);
    }

    #[test]
    fn mla_with_compressed_history_and_indexer() {
        // Four near layers: no window, history at ratio 4, no top-k cap,
        // indexer 1 head x 16.
        let near = |count: u32, indexer: Option<Indexer>| LayerClass::Mla {
            count,
            latent_dim: 512,
            rope_dim: 0,
            kv_precision: Precision::Fp8,
            window: 0,
            history: Some(History {
                compress_ratio: 4,
                index_topk: 0,
                indexer,
            }),
            heads: None,
            qk_head_dim: None,
            v_head_dim: None,
        };
        let ix = Indexer {
            heads: 1,
            head_dim: 16,
            kv_precision: Precision::Fp8,
        };
        let mut m = dense(1000, Precision::Bf16, 4, 1024, 8, 8);
        m.layers = vec![near(4, Some(ix))];
        let t = 1024;
        // Retaining only one indexer scales indexer FLOPs and KV by 1/4 while
        // the sparse reads themselves are unchanged.
        let mut cached = m.clone();
        cached.layers = vec![near(1, Some(ix)), near(3, None)];
        let ix_flops = |m: &ModelSpec| {
            m.attention_flops(1, t) - {
                let mut n = m.clone();
                for l in n.layers.iter_mut() {
                    if let LayerClass::Mla {
                        history: Some(h), ..
                    } = l
                    {
                        h.indexer = None;
                    }
                }
                n.attention_flops(1, t)
            }
        };
        assert_eq!(ix_flops(&cached) * 4, ix_flops(&m));
        // The three layers that dropped their indexer no longer read its KV
        // (256 positions x 16 B each); the sparse KV reads are unchanged.
        assert_eq!(
            m.kv_bytes_read_per_decode_step(t) - cached.kv_bytes_read_per_decode_step(t),
            3 * 256 * 16
        );
        // Storage: 4 layers x (window 0 + all 256 compressed positions) x 512 B
        // + indexer 4 x 256 x 16 B.
        assert_eq!(m.kv_storage_bytes(t), 4 * 256 * 512 + 4 * 256 * 16);
        assert_eq!(cached.kv_storage_bytes(t), 4 * 256 * 512 + 256 * 16);
        // Attention priced at 4 * hidden per pair when the head shape is absent.
        assert_eq!(
            m.attention_flops(1, t),
            4 * 4 * 1024 * 256 + 4 * (2 * 16 * 256)
        );
    }

    #[test]
    fn mla_head_shape_prices_true_attention_flops() {
        let mut m = dense(1000, Precision::Bf16, 1, 7168, 8, 8);
        m.layers = vec![LayerClass::Mla {
            count: 1,
            latent_dim: 512,
            rope_dim: 64,
            kv_precision: Precision::Bf16,
            window: 0,
            history: None,
            heads: Some(64),
            qk_head_dim: Some(192),
            v_head_dim: Some(128),
        }];
        // Full context: attends and stores every position.
        assert_eq!(m.attention_flops(1, 100), 2 * 64 * (192 + 128) * 100);
        assert_eq!(m.kv_storage_bytes(100), (512 + 64) * 2 * 100);
        // Window + history (DeepSeek-V4 style): 128 recent + T/128 far entries.
        m.layers = vec![LayerClass::Mla {
            count: 1,
            latent_dim: 512,
            rope_dim: 64,
            kv_precision: Precision::Fp8,
            window: 128,
            history: Some(History {
                compress_ratio: 128,
                index_topk: 0,
                indexer: None,
            }),
            heads: None,
            qk_head_dim: None,
            v_head_dim: None,
        }];
        assert_eq!(m.kv_storage_bytes(4096), (512 + 64) * (128 + 32));
        assert_eq!(m.attention_flops(1, 4096), 4 * 7168 * (128 + 32));
        // History-only top-k selection (GLM-5 DSA): no local window.
        m.layers = vec![LayerClass::Mla {
            count: 1,
            latent_dim: 512,
            rope_dim: 64,
            kv_precision: Precision::Bf16,
            window: 0,
            history: Some(History {
                compress_ratio: 1,
                index_topk: 2048,
                indexer: None,
            }),
            heads: None,
            qk_head_dim: None,
            v_head_dim: None,
        }];
        assert_eq!(m.attention_flops(1, 4096), 4 * 7168 * 2048);
        assert_eq!(m.kv_storage_bytes(4096), (512 + 64) * 2 * 4096);
    }

    #[test]
    fn linear_layers_are_fixed_state_only() {
        let mut m = dense(1000, Precision::Bf16, 1, 1024, 8, 8);
        m.layers = vec![
            LayerClass::Attention {
                count: 10,
                heads: 16,
                head_dim: 256,
                kv_heads: 2,
                kv_shared: false,
                window: 0,
                kv_precision: Precision::Bf16,
            },
            LayerClass::Linear {
                count: 30,
                state_bytes: 1_000_000,
            },
        ];
        assert_eq!(m.per_sequence_state_bytes(), 30_000_000);
        // Linear layers add nothing to the growing KV or the attention FLOPs.
        assert_eq!(m.kv_storage_bytes(64), 10 * 2 * 2 * 256 * 2 * 64);
        assert_eq!(m.attention_flops(1, 64), 10 * 4 * 16 * 256 * 64);
        assert_eq!(m.num_layers(), 40);
    }

    #[test]
    fn moe_streams_split_flops_bytes_and_collectives() {
        let mut m = dense(1000, Precision::Bf16, 4, 1024, 8, 8);
        m.weights = vec![
            moe_stream(256),
            WeightStream {
                precision: Precision::Fp8,
                active_params: 1000,
                resident_params: 1000,
                routing: None,
            },
        ];
        assert_eq!(
            m.matmul_flops_per_token_by_prec(),
            vec![(Precision::Fp4, 18_000), (Precision::Fp8, 2_000)]
        );
        let small: u64 = m
            .weight_bytes_per_step_by_prec(1)
            .iter()
            .map(|(_, b)| b)
            .sum();
        let large: u64 = m
            .weight_bytes_per_step_by_prec(100_000)
            .iter()
            .map(|(_, b)| b)
            .sum();
        assert!(
            large > small,
            "MoE weight traffic must grow with batch tokens"
        );
        assert_eq!(m.weight_residency_bytes(), 257_000 / 2 + 1000);
        assert_eq!(m.alltoall_bytes_per_token(), 8 * 1024 * 2);
        assert_eq!(m.num_ep_alltoalls_per_pass(), 8);
    }

    #[test]
    fn validate_rejects_broken_specs() {
        let ok = dense(1000, Precision::Bf16, 4, 1024, 8, 8);
        assert!(ok.validate().is_ok());
        let mut m = ok.clone();
        m.weights.clear();
        assert!(m.validate().is_err());
        let mut m = ok.clone();
        m.weights[0].resident_params = 10;
        assert!(m.validate().is_err());
        let mut m = ok.clone();
        m.layers = vec![LayerClass::Linear {
            count: 0,
            state_bytes: 1,
        }];
        assert!(m.validate().is_err());
        let mut m = ok.clone();
        m.layers = vec![LayerClass::Mla {
            count: 1,
            latent_dim: 512,
            rope_dim: 0,
            kv_precision: Precision::Fp8,
            window: 0,
            history: Some(History {
                compress_ratio: 0,
                index_topk: 2048,
                indexer: None,
            }),
            heads: None,
            qk_head_dim: None,
            v_head_dim: None,
        }];
        assert!(m.validate().is_err());
    }

    #[test]
    fn parses_toml_and_rejects_unknown_fields() {
        let m: ModelSpec = toml::from_str(
            r#"
            name = "x"
            hidden_dim = 1024
            max_seq_len = 4096
            [[weights]]
            precision = "fp8"
            active_params = 1000
            resident_params = 1000
            [[layers]]
            kind = "attention"
            count = 4
            heads = 8
            head_dim = 128
            kv_heads = 2
            kv_precision = "bf16"
            "#,
        )
        .unwrap();
        assert!(m.validate().is_ok());
        assert_eq!(m.attention_precision, Precision::Bf16);
        let bad = toml::from_str::<ModelSpec>(
            r#"
            name = "x"
            hidden_dim = 1024
            max_seq_len = 4096
            num_layers = 4
            [[weights]]
            precision = "fp8"
            active_params = 1000
            resident_params = 1000
            [[layers]]
            kind = "attention"
            count = 4
            heads = 8
            head_dim = 128
            kv_heads = 2
            kv_precision = "bf16"
            "#,
        );
        assert!(bad.is_err());
    }
}
