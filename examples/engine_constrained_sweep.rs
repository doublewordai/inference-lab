//! Constrained GatedAggregate sweep: how much of the aggregated gate's
//! headroom over the best fixed k survives the real engine's switching
//! machinery?
//!
//! The engine realization of the gated_aggregate policy cannot re-decide the
//! verify width every round. The implemented policy (a) re-evaluates only
//! every `switch_cooldown_rounds` decode rounds, (b) per re-evaluation moves
//! at most D indices through the sorted measured candidate widths toward the
//! argmax, and (c) pays a small stall on the round where the width actually
//! changes. This example reruns the engine-target scenario (same config,
//! measured cost table, EAGLE3 trace bank, pure-decode prefilled closed loop,
//! ISL=1024/OSL=256) at concurrency {1, 2, 4} over a grid of
//!   N (cooldown rounds)  in {1, 4, 8, 16, 32}
//!   D (max index step)   in {2, unlimited}
//!   S (per-switch cost)  in {0, 0.5} ms
//! and reports each cell's headroom over the best fixed k at that
//! concurrency. The (N=1, D=unlimited, S=0) cell takes the engine's
//! unconstrained fast path and must reproduce `engine_target_sweep`'s
//! gated_aggregate numbers exactly (same seed, same code path).
//!
//! Run: `cargo run --release --example engine_constrained_sweep --no-default-features`

use inference_lab::compute::MeasuredCostTable;
use inference_lab::config::{
    ClusterSpec, Config, GammaPolicy, SpeculativeConfig, SwitchConstraints,
};
use inference_lab::simulation::{simulate_closed_loop, Topology};
use rayon::prelude::*;
use std::collections::BTreeMap;

const CONFIG_PATH: &str = "configs/llama31-8b-eagle3-gh200.toml";
const CONCS: [u32; 3] = [1, 2, 4];
// 2 and 3 added beyond the pre-registered {1,4,8,16,32}: the decay turned
// out to complete before N=4, so the cliff needs the finer rungs.
const NS: [u32; 7] = [1, 2, 3, 4, 8, 16, 32];
const DS: [Option<u32>; 2] = [Some(2), None];
const SS: [f64; 2] = [0.0, 0.5];
const ISL: u32 = 1024;
const OSL: u32 = 256;

fn topology(cfg: &Config) -> Topology {
    let cluster = ClusterSpec {
        hardware: cfg.hardware.clone(),
        parallel: cfg.parallel.clone(),
        comms: None,
        num_workers: 1,
        node: 0,
    };
    Topology::aggregated(cluster, cfg.model.clone(), cfg.scheduler.clone()).expect("topo")
}

/// Pure-decode target run, identical loop semantics (total, warmup, seed,
/// prefilled arrivals) to `engine_target_sweep::run`.
fn run(cfg: &Config, conc: u32, spec: SpeculativeConfig) -> f64 {
    let total = (conc * 2).max(1000);
    let warmup = conc / 2;
    let res = simulate_closed_loop(
        topology(cfg),
        conc,
        ISL,
        OSL,
        total,
        warmup,
        Some(spec),
        7,
        true, // skip_prefill: pure-decode target
    )
    .expect("run");
    res.throughput() * OSL as f64
}

#[derive(Clone, Copy, PartialEq)]
enum Task {
    /// gamma = 0 (no speculation), priced from the table's plain-decode rows.
    NoSpec,
    Fixed(u32),
    Constrained {
        n: u32,
        d: Option<u32>,
        s_ms: f64,
    },
}

fn spec_for(t: Task, base: &SpeculativeConfig) -> SpeculativeConfig {
    let mut s = base.clone();
    match t {
        Task::NoSpec => {
            s.gamma = 0;
            s.policy = GammaPolicy::Fixed;
        }
        Task::Fixed(g) => {
            s.gamma = g;
            s.policy = GammaPolicy::Fixed;
        }
        Task::Constrained { n, d, s_ms } => {
            s.policy = GammaPolicy::GatedAggregate;
            s.switch = SwitchConstraints {
                cooldown_rounds: n,
                max_step: d,
                cost_ms: s_ms,
            };
        }
    }
    s
}

fn d_label(d: Option<u32>) -> String {
    match d {
        Some(d) => format!("{d}"),
        None => "inf".into(),
    }
}

fn main() {
    let cfg = Config::from_file(CONFIG_PATH).expect("config");
    let base = cfg.speculative.as_ref().expect("config has [speculative]");
    let gamma_max = base.gamma;
    let mc = base
        .measured_cost
        .as_ref()
        .expect("config has measured_cost");
    let table = MeasuredCostTable::load(&mc.path).expect("measured cost table");

    // Same candidate exclusion as the engine: measured widths only.
    let fixed: Vec<u32> = (1..=gamma_max).filter(|&g| table.has_draft(g)).collect();
    let cands: Vec<u32> = std::iter::once(0).chain(fixed.iter().copied()).collect();

    println!("Constrained GatedAggregate sweep: {}", cfg.model_name());
    println!(
        "scenario = engine_target_sweep's TARGET SWEEP (pure decode, prefilled \
         arrivals,\nISL={ISL} OSL={OSL}), conc {CONCS:?}, candidate widths g in \
         {cands:?} (measured only)."
    );
    println!(
        "N = re-evaluation cooldown (decode rounds), D = max candidate-INDEX \
         step per\nre-evaluation, S = per-switch stall (ms, paid on the first \
         round at the new width).\n"
    );

    // ------------------------------------------------------------------
    // All runs in one rayon batch: baselines (nospec + each fixed k) and
    // the constrained grid, at each concurrency.
    // ------------------------------------------------------------------
    let mut tasks: Vec<(u32, Task)> = Vec::new();
    for &conc in &CONCS {
        tasks.push((conc, Task::NoSpec));
        for &g in &fixed {
            tasks.push((conc, Task::Fixed(g)));
        }
        for &n in &NS {
            for &d in &DS {
                for &s_ms in &SS {
                    tasks.push((conc, Task::Constrained { n, d, s_ms }));
                }
            }
        }
    }
    let results: Vec<((u32, usize), f64)> = tasks
        .par_iter()
        .enumerate()
        .map(|(i, &(conc, t))| ((conc, i), run(&cfg, conc, spec_for(t, base))))
        .collect();
    let tps: BTreeMap<usize, f64> = results.into_iter().map(|((_, i), t)| (i, t)).collect();

    // Best fixed per concurrency (nospec competes too, as in the target sweep).
    let mut best_fixed: BTreeMap<u32, (String, f64)> = BTreeMap::new();
    let mut grid: BTreeMap<(u32, Option<u32>, u64), BTreeMap<u32, f64>> = BTreeMap::new();
    for (i, &(conc, t)) in tasks.iter().enumerate() {
        let v = tps[&i];
        match t {
            Task::NoSpec => {
                let e = best_fixed.entry(conc).or_insert(("nospec".into(), v));
                if v > e.1 {
                    *e = ("nospec".into(), v);
                }
            }
            Task::Fixed(g) => {
                let e = best_fixed.entry(conc).or_insert((format!("k{g}"), v));
                if v > e.1 {
                    *e = (format!("k{g}"), v);
                }
            }
            Task::Constrained { n, d, s_ms } => {
                grid.entry((n, d, s_ms.to_bits()))
                    .or_default()
                    .insert(conc, v);
            }
        }
    }

    print!("{:>10}", "best-fixed");
    for &conc in &CONCS {
        let (label, v) = &best_fixed[&conc];
        print!("   conc {conc}: {label} {v:.0} tok/s");
    }
    println!("\n");

    println!(
        "{:>4} {:>4} {:>5} |{}",
        "N",
        "D",
        "S ms",
        CONCS
            .iter()
            .map(|c| format!(" {:>8} {:>7}", format!("c{c} tok/s"), "d%"))
            .collect::<String>()
    );
    println!("{}", "-".repeat(16 + 17 * CONCS.len()));
    for &n in &NS {
        for &d in &DS {
            for &s_ms in &SS {
                let row = &grid[&(n, d, s_ms.to_bits())];
                print!("{n:>4} {:>4} {s_ms:>5.1} |", d_label(d));
                for &conc in &CONCS {
                    let v = row[&conc];
                    let best = best_fixed[&conc].1;
                    print!(" {v:>8.0} {:>+6.2}%", 100.0 * (v - best) / best);
                }
                println!();
            }
        }
        println!("{}", "-".repeat(16 + 17 * CONCS.len()));
    }
    println!(
        "\nd% = headroom over the best fixed k at that concurrency. The \
         (N=1, D=inf, S=0)\nrow takes the unconstrained fast path and must \
         match engine_target_sweep's\ngated_aggregate column."
    );
}

trait ModelName {
    fn model_name(&self) -> &str;
}
impl ModelName for Config {
    fn model_name(&self) -> &str {
        use inference_lab::config::ModelCosts;
        self.model.name()
    }
}
