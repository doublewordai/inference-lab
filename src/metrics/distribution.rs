//! Exact sample statistics.
//!
//! [`Distribution`] keeps every sample (the drivers ship them to charts
//! anyway) and sorts lazily, so quantiles are exact and repeated snapshots
//! cost one merge of the samples added since the last snapshot rather than a
//! full re-sort. [`RunningMean`] is for per-iteration series where only the
//! mean is reported.

/// A growing set of samples with exact min/mean/quantiles.
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Samples in insertion order.
    values: Vec<f64>,
    /// `values[..n_sorted]` sorted ascending. Merged lazily.
    sorted: Vec<f64>,
    n_sorted: usize,
    sum: f64,
    min: f64,
}

impl Default for Distribution {
    fn default() -> Self {
        Self::new()
    }
}

impl Distribution {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            sorted: Vec::new(),
            n_sorted: 0,
            sum: 0.0,
            min: f64::INFINITY,
        }
    }

    pub fn push(&mut self, v: f64) {
        self.values.push(v);
        self.sum += v;
        if v < self.min {
            self.min = v;
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Samples in insertion order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Smallest sample; 0 when empty.
    pub fn min(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.min
        }
    }

    /// Arithmetic mean; 0 when empty.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Exact quantile `q` in `[0, 1]`, linearly interpolated between order
    /// statistics (the numpy default); 0 when empty.
    pub fn quantile(&mut self, q: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.ensure_sorted();
        let n = self.sorted.len();
        let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            self.sorted[lo]
        } else {
            let frac = pos - lo as f64;
            self.sorted[lo] + frac * (self.sorted[hi] - self.sorted[lo])
        }
    }

    /// Bring `sorted` up to date: sort the tail added since the last call and
    /// merge it with the already-sorted prefix. O(n + k log k) for k new.
    fn ensure_sorted(&mut self) {
        let n = self.values.len();
        if self.n_sorted == n {
            return;
        }
        let mut tail: Vec<f64> = self.values[self.n_sorted..].to_vec();
        tail.sort_by(f64::total_cmp);
        if self.n_sorted == 0 {
            self.sorted = tail;
        } else {
            let mut merged = Vec::with_capacity(n);
            let (mut i, mut j) = (0, 0);
            while i < self.sorted.len() && j < tail.len() {
                if self.sorted[i] <= tail[j] {
                    merged.push(self.sorted[i]);
                    i += 1;
                } else {
                    merged.push(tail[j]);
                    j += 1;
                }
            }
            merged.extend_from_slice(&self.sorted[i..]);
            merged.extend_from_slice(&tail[j..]);
            self.sorted = merged;
        }
        self.n_sorted = n;
    }
}

/// Mean of a stream without retaining it.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunningMean {
    sum: f64,
    n: u64,
}

impl RunningMean {
    pub fn add(&mut self, v: f64) {
        self.sum += v;
        self.n += 1;
    }

    /// Mean so far; 0 when nothing has been added.
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }

    pub fn count(&self) -> u64 {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_distribution_reports_zeros() {
        let mut d = Distribution::default();
        assert_eq!(d.min(), 0.0);
        assert_eq!(d.mean(), 0.0);
        assert_eq!(d.quantile(0.5), 0.0);
        assert!(d.is_empty());
    }

    #[test]
    fn quantiles_are_exact_and_interpolated() {
        let mut d = Distribution::new();
        for v in [5.0, 1.0, 3.0, 2.0, 4.0] {
            d.push(v);
        }
        assert_eq!(d.min(), 1.0);
        assert_eq!(d.mean(), 3.0);
        let mut d2 = Distribution::default();
        d2.push(2.0);
        assert_eq!(d2.min(), 2.0);
        assert_eq!(d.quantile(0.0), 1.0);
        assert_eq!(d.quantile(0.5), 3.0);
        assert_eq!(d.quantile(1.0), 5.0);
        // 0.9 * 4 = 3.6 -> between 4.0 and 5.0
        assert!((d.quantile(0.9) - 4.6).abs() < 1e-12);
    }

    #[test]
    fn incremental_snapshots_match_full_sort() {
        let mut d = Distribution::new();
        let mut all = Vec::new();
        let mut x = 12345u64;
        for round in 0..20 {
            for _ in 0..(round * 7 + 1) {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (x >> 11) as f64 / (1u64 << 53) as f64;
                d.push(v);
                all.push(v);
            }
            let mut sorted = all.clone();
            sorted.sort_by(f64::total_cmp);
            let n = sorted.len();
            for q in [0.0, 0.25, 0.5, 0.9, 0.99, 1.0] {
                let pos = q * (n - 1) as f64;
                let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                let expect = sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo]);
                assert!(
                    (d.quantile(q) - expect).abs() < 1e-12,
                    "q={q} round={round}"
                );
            }
        }
    }

    #[test]
    fn running_mean() {
        let mut m = RunningMean::default();
        assert_eq!(m.mean(), 0.0);
        m.add(2.0);
        m.add(4.0);
        assert_eq!(m.mean(), 3.0);
        assert_eq!(m.count(), 2);
    }
}
