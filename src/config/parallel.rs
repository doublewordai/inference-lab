use serde::Deserialize;

fn default_dim() -> u32 {
    1
}

/// Whether routed-expert communication is serial with expert execution or
/// its bandwidth portion can hide behind the routed expert kernel. The
/// collective call latency is always exposed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeOverlap {
    /// Existing behavior: dispatch, expert execution, and combine are serial.
    #[default]
    Serial,
    /// Dispatch/combine wire time overlaps the routed expert kernel; only the
    /// larger is on the critical path, followed by the call latency floors.
    Hidden,
    /// A wave-pipelined fused MoE layer (megakernel): dispatch, grouped GEMM
    /// and combine run as one persistent kernel per layer, so the wire hides
    /// behind the expert weight read (like `Hidden`) but the exposed
    /// per-layer floor is a fill+drain signal pair plus an epilogue, not the
    /// full collective call latency. Requires [`ParallelConfig::megakernel`].
    Megakernel,
}

/// Per-fused-layer parameters of a [`MoeOverlap::Megakernel`] layer. Opt-in:
/// unset unless `moe_overlap = "megakernel"`, and never consulted in the
/// default `Serial` / `Hidden` paths, so default outputs stay byte-identical.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MegakernelParams {
    /// Seconds per fill/drain signal. A fused layer pays two (first wave in,
    /// last wave out) in place of the collective call floors of `Hidden`.
    /// This is the device-side completion signal (a proxy-posted atomic /
    /// counter), not a collective barrier — the point of the megakernel.
    pub signal_latency: f64,
    /// Seconds of per-layer epilogue that stays exposed (final activation +
    /// cast + any residual scatter the wave pipeline cannot hide). Defaults
    /// to 0.
    #[serde(default)]
    pub epilogue: f64,
    /// Fraction of SMs the comm/pack CTAs take from the expert GEMM,
    /// reducing its FLOP rate proportionally (the GEMM runs at
    /// `peak × (1 - comm_sm_fraction)`). At the decode knee the expert kernel
    /// is weight-read bound, so this bites only when the GEMM is the binding
    /// term. Defaults to 0. Must be in `[0, 1)`.
    #[serde(default)]
    pub comm_sm_fraction: f64,
}

impl MegakernelParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.signal_latency.is_finite() && self.signal_latency >= 0.0) {
            return Err(format!(
                "megakernel.signal_latency must be finite and >= 0 (got {})",
                self.signal_latency
            ));
        }
        if !(self.epilogue.is_finite() && self.epilogue >= 0.0) {
            return Err(format!(
                "megakernel.epilogue must be finite and >= 0 (got {})",
                self.epilogue
            ));
        }
        if !(self.comm_sm_fraction.is_finite() && (0.0..1.0).contains(&self.comm_sm_fraction)) {
            return Err(format!(
                "megakernel.comm_sm_fraction must be in [0, 1) (got {})",
                self.comm_sm_fraction
            ));
        }
        Ok(())
    }
}

/// How one replica of the model is laid out across its GPUs.
///
/// * `tp` — the replica's world size. Its GPUs pool FLOP rate, HBM bandwidth
///   and memory; weights are sharded across them (each expert's matrices
///   included, unless `ep` says otherwise). Every layer's sharded output is
///   all-reduced: once after attention, once after the FFN.
/// * `ep` — experts sharded across `ep` of the ranks (must divide `tp`);
///   expert reads and FLOPs are taken as balanced across the ranks. With TP
///   attention every rank holds every token, so the MoE output is still
///   combined by the FFN all-reduce (vLLM `--enable-expert-parallel`); the
///   traffic changes only under `dp_attention`.
/// * `dp_attention` — attention runs data-parallel over the `tp` ranks
///   (sglang `--enable-dp-attention`): each rank holds the full attention
///   projections (replicated: `tp×` resident, `tp×` read per step) and its
///   own sequences' KV, and needs no attention all-reduce; the FFN, still
///   TP-sharded, gathers the ranks' tokens with an all-gather and returns
///   them with a reduce-scatter; with `ep > 1` the MoE layers instead
///   dispatch each rank's tokens to the expert-owning ranks and combine
///   them back with all-to-alls over the `ep` group (DeepEP-style).
///
/// Collectives are priced on the hardware's [`super::FabricConfig`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelConfig {
    /// Replica world size. Defaults to 1.
    #[serde(default = "default_dim")]
    pub tp: u32,
    /// Expert-parallel group size (divides `tp`). Defaults to 1: experts
    /// TP-sharded like every other weight.
    #[serde(default = "default_dim")]
    pub ep: u32,
    /// Data-parallel attention over the `tp` ranks. Defaults to false.
    #[serde(default)]
    pub dp_attention: bool,
    /// Routed-expert communication overlap policy. Defaults to `serial` so
    /// existing configs retain byte-identical outputs.
    #[serde(default)]
    pub moe_overlap: MoeOverlap,
    /// Fill/drain and epilogue parameters for [`MoeOverlap::Megakernel`].
    /// Required when `moe_overlap = "megakernel"`, ignored otherwise. Unset
    /// by default, so no default path reads it.
    #[serde(default)]
    pub megakernel: Option<MegakernelParams>,
}

impl ParallelConfig {
    /// Reject a `megakernel` overlap policy with no parameters (or invalid
    /// ones). Called from config deserialization; the compute engine also
    /// asserts the invariant where it prices the mode.
    pub fn validate(&self) -> Result<(), String> {
        match (self.moe_overlap, &self.megakernel) {
            (MoeOverlap::Megakernel, None) => Err(
                "moe_overlap = \"megakernel\" requires a [parallel.megakernel] \
                 (or per-hardware `megakernel`) block with signal_latency"
                    .into(),
            ),
            (_, Some(params)) => params.validate(),
            _ => Ok(()),
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tp: 1,
            ep: 1,
            dp_attention: false,
            moe_overlap: MoeOverlap::Serial,
            megakernel: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn megakernel_mode_requires_params() {
        let mut p = ParallelConfig {
            moe_overlap: MoeOverlap::Megakernel,
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("requires"));
        p.megakernel = Some(MegakernelParams {
            signal_latency: 1e-6,
            epilogue: 5e-6,
            comm_sm_fraction: 0.25,
        });
        assert!(p.validate().is_ok());
    }

    #[test]
    fn megakernel_params_reject_bad_fraction() {
        let p = ParallelConfig {
            moe_overlap: MoeOverlap::Megakernel,
            megakernel: Some(MegakernelParams {
                signal_latency: 1e-6,
                epilogue: 0.0,
                comm_sm_fraction: 1.0,
            }),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("comm_sm_fraction"));
    }

    #[test]
    fn serial_and_hidden_ignore_absent_megakernel() {
        for mode in [MoeOverlap::Serial, MoeOverlap::Hidden] {
            let p = ParallelConfig {
                moe_overlap: mode,
                ..Default::default()
            };
            assert!(p.validate().is_ok());
        }
    }

    #[test]
    fn megakernel_deserializes_from_parallel_block() {
        let p: ParallelConfig = toml::from_str(
            r#"
            tp = 16
            ep = 16
            dp_attention = true
            moe_overlap = "megakernel"
            megakernel = { signal_latency = 1e-6, epilogue = 5e-6, comm_sm_fraction = 0.25 }
            "#,
        )
        .unwrap();
        assert_eq!(p.moe_overlap, MoeOverlap::Megakernel);
        let m = p.megakernel.unwrap();
        assert_eq!(m.signal_latency, 1e-6);
        assert_eq!(m.comm_sm_fraction, 0.25);
        p.validate().unwrap();
    }
}
