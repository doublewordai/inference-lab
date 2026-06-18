//! Measured step-cost table: an empirical override for the analytic roofline
//! in the draft-depth policy decision.
//!
//! The roofline prices a hypothetical verify width `1 + g` analytically; a
//! real engine's step time has kernel- and shape-specific structure the
//! roofline can't see. When a benchmark of the actual target engine is
//! available (a grid of `(batch_size, num_draft_tokens) -> step_seconds`
//! measurements), the policy can consult it instead and fall back to the
//! roofline off-grid.
//!
//! File convention (matches the sglang benchmark export): the
//! `num_draft_tokens` column is the per-sequence verify width `g + 1`, so a
//! `num_draft_tokens = 1` row is a plain non-speculative decode step. The
//! lookup API speaks draft length `g` (`step_time(batch, 0)` reads the
//! `num_draft_tokens = 1` rows). Lookup is exact in draft length and
//! nearest-neighbor in `batch_size`.

use std::collections::BTreeMap;

/// Measured `(batch_size, num_draft_tokens) -> step_seconds` grid.
#[derive(Debug, Clone)]
pub struct MeasuredCostTable {
    /// Draft length `g` (= file `num_draft_tokens - 1`) ->
    /// `[(batch_size, step_seconds)]`, each row list sorted ascending by
    /// batch size.
    by_draft: BTreeMap<u32, Vec<(u32, f64)>>,
}

impl MeasuredCostTable {
    /// Load from CSV with header `batch_size,num_draft_tokens,step_seconds`,
    /// one row per measured grid point.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("measured cost table {path}: {e}"))?;
        Self::parse(&text).map_err(|e| format!("measured cost table {path}: {e}"))
    }

    fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty file")?;
        let cols: Vec<&str> = header.split(',').map(str::trim).collect();
        if cols != ["batch_size", "num_draft_tokens", "step_seconds"] {
            return Err(format!("unexpected header {header:?}"));
        }
        let mut by_draft: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
        let mut rows = 0usize;
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() != 3 {
                return Err(format!("bad row {}", i + 2));
            }
            let batch: u32 = f[0].parse().map_err(|e| format!("row {}: {e}", i + 2))?;
            let draft: u32 = f[1].parse().map_err(|e| format!("row {}: {e}", i + 2))?;
            let secs: f64 = f[2].parse().map_err(|e| format!("row {}: {e}", i + 2))?;
            if !(secs.is_finite() && secs > 0.0) {
                return Err(format!("row {}: non-positive step_seconds {secs}", i + 2));
            }
            if draft == 0 {
                return Err(format!(
                    "row {}: num_draft_tokens=0; the file column is the verify \
                     width (1 = plain decode)",
                    i + 2
                ));
            }
            by_draft.entry(draft - 1).or_default().push((batch, secs));
            rows += 1;
        }
        if rows == 0 {
            return Err("no rows".into());
        }
        for v in by_draft.values_mut() {
            v.sort_by_key(|&(b, _)| b);
            // Monotonic sanitisation (mirrors the telemetry fitter's
            // --prune-monotonic): step cost at fixed verify width is
            // non-decreasing in batch size, but transient batch shapes
            // (drain/ramp cells like B = 63, 127, 47 next to a steady
            // power-of-two neighbour) sometimes survive the fitter with
            // host-dispatch or prefill-adjacent contamination, measuring
            // 1.4-2.4x the next-larger batch. Nearest-neighbor lookups then
            // poison every query that lands on them (e.g. a closed loop
            // whose decode batch sits at conc - 1 while one request
            // prefills). Drop any row more than 30% above the row at the
            // next-larger measured batch.
            let mut kept: Vec<(u32, f64)> = Vec::with_capacity(v.len());
            for i in 0..v.len() {
                if i + 1 < v.len() && v[i].1 > 1.3 * v[i + 1].1 {
                    continue;
                }
                kept.push(v[i]);
            }
            *v = kept;
        }
        by_draft.retain(|_, v| !v.is_empty());
        if by_draft.is_empty() {
            return Err("no rows after monotonic sanitisation".into());
        }
        Ok(Self { by_draft })
    }

    /// Measured step time at `(batch_size, draft length)`. `num_draft` is the
    /// draft length `g` (`0` = plain decode, reading the file's
    /// `num_draft_tokens = g + 1` rows). Exact match on draft length;
    /// nearest-neighbor in `batch_size` (ties prefer the smaller batch).
    /// `None` when the table has no row at that draft length — callers fall
    /// back to the analytic roofline.
    pub fn step_time(&self, batch_size: u32, num_draft: u32) -> Option<f64> {
        let rows = self.by_draft.get(&num_draft)?;
        rows.iter()
            .min_by_key(|&&(b, _)| (b.abs_diff(batch_size), b))
            .map(|&(_, s)| s)
    }

    /// Whether the table has any measured rows at draft length `num_draft`
    /// (`0` = plain decode). Widths without rows must not be priced from this
    /// table; policy candidate sets should exclude them rather than fall
    /// back to the (optimistic) analytic roofline.
    pub fn has_draft(&self, num_draft: u32) -> bool {
        self.by_draft.contains_key(&num_draft)
    }

    /// Measured step time at a *fractional* draft length (the mean verify
    /// width of a ragged batch, minus one). Interpolates linearly between the
    /// nearest measured draft lengths bracketing `g` — so interior holes in
    /// the grid (e.g. a table with ndt = {5, 7, 9} but not 6 or 8) are
    /// bridged from measured anchors rather than the roofline. `None` when
    /// `g` lies outside the measured draft-length range (no extrapolation in
    /// width; callers fall back to the roofline).
    pub fn step_time_frac(&self, batch_size: u32, g: f64) -> Option<f64> {
        if !g.is_finite() || g < 0.0 {
            return None;
        }
        let lo = self
            .by_draft
            .keys()
            .copied()
            .filter(|&k| (k as f64) <= g)
            .max()?;
        let hi = self
            .by_draft
            .keys()
            .copied()
            .filter(|&k| (k as f64) >= g)
            .min()?;
        let t_lo = self.step_time(batch_size, lo)?;
        if hi == lo {
            return Some(t_lo);
        }
        let t_hi = self.step_time(batch_size, hi)?;
        let frac = (g - lo as f64) / ((hi - lo) as f64);
        Some(t_lo + frac * (t_hi - t_lo))
    }

    /// Measured batch-size range `(min, max)` at draft length `num_draft`
    /// (`0` = plain decode). Lookups outside this range extrapolate by
    /// nearest-neighbor clamping. `None` when the table has no rows at that
    /// draft length.
    pub fn batch_range(&self, num_draft: u32) -> Option<(u32, u32)> {
        let rows = self.by_draft.get(&num_draft)?;
        match (rows.first(), rows.last()) {
            (Some(&(lo, _)), Some(&(hi, _))) => Some((lo, hi)),
            _ => None,
        }
    }

    /// Largest measured batch size at draft length `num_draft` (`0` = plain
    /// decode). Lookups above this extrapolate by nearest-neighbor clamping.
    /// `None` when the table has no rows at that draft length.
    pub fn max_batch(&self, num_draft: u32) -> Option<u32> {
        self.batch_range(num_draft).map(|(_, hi)| hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // File column is the verify width: ndt=1 is plain decode (g=0), ndt=5 is
    // a 4-token draft (g=4).
    const CSV: &str = "batch_size,num_draft_tokens,step_seconds\n\
                       1,1,0.010\n\
                       8,1,0.012\n\
                       64,1,0.020\n\
                       1,5,0.015\n\
                       8,5,0.018\n\
                       64,5,0.045\n";

    #[test]
    fn exact_grid_points_round_trip() {
        let t = MeasuredCostTable::parse(CSV).unwrap();
        assert_eq!(t.step_time(8, 0), Some(0.012));
        assert_eq!(t.step_time(64, 4), Some(0.045));
    }

    #[test]
    fn nearest_neighbor_in_batch_size() {
        let t = MeasuredCostTable::parse(CSV).unwrap();
        assert_eq!(t.step_time(2, 0), Some(0.010)); // 2 is nearer 1 than 8
        assert_eq!(t.step_time(40, 4), Some(0.045)); // 40 is nearer 64 than 8
        assert_eq!(t.step_time(10_000, 0), Some(0.020)); // clamps to largest
    }

    #[test]
    fn missing_draft_length_is_none() {
        let t = MeasuredCostTable::parse(CSV).unwrap();
        assert_eq!(t.step_time(8, 3), None); // no g=3 rows: roofline fallback
    }

    #[test]
    fn bad_header_rejected() {
        assert!(MeasuredCostTable::parse("a,b,c\n1,2,3\n").is_err());
    }

    #[test]
    fn ndt_zero_rejected() {
        // The file convention is verify width (1 = plain decode); a 0 row is
        // a convention error, not a g=0 row.
        let csv = "batch_size,num_draft_tokens,step_seconds\n1,0,0.010\n";
        assert!(MeasuredCostTable::parse(csv).is_err());
    }

    #[test]
    fn transient_batch_artifacts_pruned() {
        // B=63 measured 2x the B=64 cell: a drain-shape artifact. The loader
        // must drop it so nearest-neighbor queries at B=63 read the steady
        // B=64 cell instead.
        let csv = "batch_size,num_draft_tokens,step_seconds\n\
                   32,1,0.010\n63,1,0.030\n64,1,0.015\n";
        let t = MeasuredCostTable::parse(csv).unwrap();
        assert_eq!(t.step_time(63, 0), Some(0.015));
        // Mild non-monotonicity (within 30%) is kept: real measurement noise.
        let csv2 = "batch_size,num_draft_tokens,step_seconds\n\
                    32,1,0.011\n64,1,0.010\n";
        let t2 = MeasuredCostTable::parse(csv2).unwrap();
        assert_eq!(t2.step_time(32, 0), Some(0.011));
    }

    #[test]
    fn has_draft_reports_width_coverage() {
        let t = MeasuredCostTable::parse(CSV).unwrap();
        assert!(t.has_draft(0));
        assert!(t.has_draft(4));
        assert!(!t.has_draft(2));
        assert!(!t.has_draft(9));
    }

    #[test]
    fn fractional_width_interpolates_across_holes() {
        let t = MeasuredCostTable::parse(CSV).unwrap();
        // Exact anchors round-trip.
        assert_eq!(t.step_time_frac(8, 0.0), Some(0.012));
        assert_eq!(t.step_time_frac(8, 4.0), Some(0.018));
        // g=2 has no rows: bridged linearly between g=0 and g=4 anchors.
        let mid = t.step_time_frac(8, 2.0).unwrap();
        assert!((mid - 0.015).abs() < 1e-12, "got {mid}");
        // Outside the measured width range: no extrapolation.
        assert_eq!(t.step_time_frac(8, 4.5), None);
        assert_eq!(t.step_time_frac(8, -1.0), None);
    }

    #[test]
    fn max_batch_reports_coverage() {
        let t = MeasuredCostTable::parse(CSV).unwrap();
        assert_eq!(t.max_batch(0), Some(64));
        assert_eq!(t.max_batch(4), Some(64));
        assert_eq!(t.max_batch(3), None);
        assert_eq!(t.batch_range(0), Some((1, 64)));
        assert_eq!(t.batch_range(3), None);
    }
}
