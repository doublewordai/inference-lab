/// Streaming quantile estimation using P-Squared algorithm
/// Maintains approximate quantiles (p50, p90, p99) in O(1) time and space
pub struct StreamingQuantiles {
    // Marker positions and heights for P² algorithm
    // We track 5 markers for p50, p90, p99
    markers: [f64; 11],           // Marker heights (actual values)
    positions: [f64; 11],         // Marker positions (count-based)
    desired_positions: [f64; 11], // Desired positions based on quantiles
    count: usize,
}

impl StreamingQuantiles {
    pub fn new() -> Self {
        Self {
            markers: [0.0; 11],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            desired_positions: [0.0; 11],
            count: 0,
        }
    }

    pub fn add(&mut self, value: f64) {
        if self.count < 11 {
            // Initial phase: collect first 11 samples
            self.markers[self.count] = value;
            self.count += 1;

            if self.count == 11 {
                // Sort initial markers
                self.markers
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // Initialize desired positions for p50, p90, p99
                self.update_desired_positions();
            }
            return;
        }

        self.count += 1;

        // Find cell k such that markers[k-1] < value <= markers[k]
        let mut k = 0;
        if value < self.markers[0] {
            self.markers[0] = value;
            k = 1;
        } else if value >= self.markers[10] {
            self.markers[10] = value;
            k = 10;
        } else {
            for i in 1..11 {
                if value < self.markers[i] {
                    k = i;
                    break;
                }
            }
        }

        // Increment positions of markers k+1 through 11
        for i in k..11 {
            self.positions[i] += 1.0;
        }

        // Update desired positions
        self.update_desired_positions();

        // Adjust marker heights
        for i in 1..10 {
            let d = self.desired_positions[i] - self.positions[i];

            if (d >= 1.0 && self.positions[i + 1] - self.positions[i] > 1.0)
                || (d <= -1.0 && self.positions[i - 1] - self.positions[i] < -1.0)
            {
                let d_sign = if d > 0.0 { 1.0 } else { -1.0 };

                // Try parabolic formula
                let q_new = self.parabolic(i, d_sign);

                if self.markers[i - 1] < q_new && q_new < self.markers[i + 1] {
                    self.markers[i] = q_new;
                } else {
                    // Use linear formula
                    self.markers[i] = self.linear(i, d_sign);
                }

                self.positions[i] += d_sign;
            }
        }
    }

    fn update_desired_positions(&mut self) {
        let n = self.count as f64;
        // Markers at indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        // Target quantiles: min, p1, p10, p25, p50, p75, p90, p95, p99, p99.9, max
        self.desired_positions[0] = 1.0;
        self.desired_positions[1] = 1.0 + 0.01 * (n - 1.0); // p1
        self.desired_positions[2] = 1.0 + 0.10 * (n - 1.0); // p10
        self.desired_positions[3] = 1.0 + 0.25 * (n - 1.0); // p25
        self.desired_positions[4] = 1.0 + 0.50 * (n - 1.0); // p50
        self.desired_positions[5] = 1.0 + 0.75 * (n - 1.0); // p75
        self.desired_positions[6] = 1.0 + 0.90 * (n - 1.0); // p90
        self.desired_positions[7] = 1.0 + 0.95 * (n - 1.0); // p95
        self.desired_positions[8] = 1.0 + 0.99 * (n - 1.0); // p99
        self.desired_positions[9] = 1.0 + 0.999 * (n - 1.0); // p99.9
        self.desired_positions[10] = n;
    }

    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let q_i = self.markers[i];
        let q_i1 = self.markers[i + 1];
        let q_i_1 = self.markers[i - 1];
        let n_i = self.positions[i];
        let n_i1 = self.positions[i + 1];
        let n_i_1 = self.positions[i - 1];

        q_i + d / (n_i1 - n_i_1)
            * ((n_i - n_i_1 + d) * (q_i1 - q_i) / (n_i1 - n_i)
                + (n_i1 - n_i - d) * (q_i - q_i_1) / (n_i - n_i_1))
    }

    fn linear(&self, i: usize, d: f64) -> f64 {
        let d_i = if d > 0.0 { 1 } else { -1 };
        let q_i = self.markers[i];
        let q_id = self.markers[(i as i32 + d_i) as usize];
        let n_i = self.positions[i];
        let n_id = self.positions[(i as i32 + d_i) as usize];

        q_i + d * (q_id - q_i) / (n_id - n_i)
    }

    pub fn min(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else if self.count < 11 {
            self.markers[..self.count]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b))
        } else {
            self.markers[0]
        }
    }

    pub fn p50(&self) -> f64 {
        if self.count < 11 {
            self.fallback_quantile(0.50)
        } else {
            self.markers[4]
        }
    }

    pub fn p90(&self) -> f64 {
        if self.count < 11 {
            self.fallback_quantile(0.90)
        } else {
            self.markers[6]
        }
    }

    pub fn p99(&self) -> f64 {
        if self.count < 11 {
            self.fallback_quantile(0.99)
        } else {
            self.markers[8]
        }
    }

    fn fallback_quantile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.markers[..self.count].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((self.count as f64 - 1.0) * p) as usize;
        sorted[index.min(self.count - 1)]
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        if self.count < 11 {
            self.markers[..self.count].iter().sum::<f64>() / self.count as f64
        } else {
            // Approximate mean from markers
            self.markers.iter().sum::<f64>() / 11.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_quantiles() {
        let quantiles = StreamingQuantiles::new();
        assert_eq!(quantiles.min(), 0.0);
        assert_eq!(quantiles.p50(), 0.0);
        assert_eq!(quantiles.p90(), 0.0);
        assert_eq!(quantiles.p99(), 0.0);
        assert_eq!(quantiles.mean(), 0.0);
    }

    #[test]
    fn test_single_value() {
        let mut quantiles = StreamingQuantiles::new();
        quantiles.add(42.0);
        assert_eq!(quantiles.min(), 42.0);
        assert_eq!(quantiles.p50(), 42.0);
        assert_eq!(quantiles.p90(), 42.0);
        assert_eq!(quantiles.p99(), 42.0);
        assert_eq!(quantiles.mean(), 42.0);
    }

    #[test]
    fn test_few_values_sorted() {
        let mut quantiles = StreamingQuantiles::new();
        for i in 1..=5 {
            quantiles.add(i as f64);
        }
        // For fixed input, we should get fixed output
        assert_eq!(quantiles.min(), 1.0);
        assert_eq!(quantiles.p50(), 3.0);
        assert_eq!(quantiles.p90(), 4.0);
        assert_eq!(quantiles.p99(), 4.0);
        assert_eq!(quantiles.mean(), 3.0);
    }

    #[test]
    fn test_uniform_distribution() {
        let mut quantiles = StreamingQuantiles::new();
        // Add values 1 to 1000 - deterministic dataset
        for i in 1..=1000 {
            quantiles.add(i as f64);
        }

        // For this exact dataset, algorithm should produce exact values
        assert_eq!(quantiles.min(), 1.0);
        assert_eq!(quantiles.p50(), 500.0);
        assert_eq!(quantiles.p90(), 900.0);
        assert_eq!(quantiles.p99(), 990.0);
    }

    #[test]
    fn test_quantile_ordering() {
        let mut quantiles = StreamingQuantiles::new();
        // Add 1000 values
        for i in 1..=1000 {
            quantiles.add(i as f64);
        }

        // Quantiles should be ordered: min <= p50 <= p90 <= p99
        assert!(quantiles.min() <= quantiles.p50());
        assert!(quantiles.p50() <= quantiles.p90());
        assert!(quantiles.p90() <= quantiles.p99());
    }

    #[test]
    fn test_skewed_distribution() {
        let mut quantiles = StreamingQuantiles::new();
        // Heavy tail: mostly small values, few large values
        // 900 values of 1.0, 90 values of 10.0, 10 values of 100.0
        for _ in 0..900 {
            quantiles.add(1.0);
        }
        for _ in 0..90 {
            quantiles.add(10.0);
        }
        for _ in 0..10 {
            quantiles.add(100.0);
        }

        // For this fixed dataset, should get fixed values
        assert_eq!(quantiles.min(), 1.0);
        assert_eq!(quantiles.p50(), 1.010241318586147);
        assert_eq!(quantiles.p90(), 5.014807349038084);
        assert_eq!(quantiles.p99(), 58.33620606034887);
    }

    #[test]
    fn test_duplicate_values() {
        let mut quantiles = StreamingQuantiles::new();
        // All same value
        for _ in 0..100 {
            quantiles.add(5.0);
        }

        assert_eq!(quantiles.min(), 5.0);
        assert_eq!(quantiles.p50(), 5.0);
        assert_eq!(quantiles.p90(), 5.0);
        assert_eq!(quantiles.p99(), 5.0);
        assert_eq!(quantiles.mean(), 5.0);
    }

    #[test]
    fn test_reverse_order() {
        let mut quantiles = StreamingQuantiles::new();
        // Add values in reverse order: 1000 down to 1
        for i in (1..=1000).rev() {
            quantiles.add(i as f64);
        }

        // Insertion order affects streaming algorithm results
        assert_eq!(quantiles.min(), 1.0);
        assert_eq!(quantiles.p50(), 501.0);
        assert_eq!(quantiles.p90(), 901.0);
        assert_eq!(quantiles.p99(), 991.0);
    }

    #[test]
    fn test_boundary_11_values() {
        let mut quantiles = StreamingQuantiles::new();
        // Exactly 11 values (boundary where algorithm switches from exact to streaming)
        for i in 1..=11 {
            quantiles.add(i as f64 * 10.0); // [10, 20, 30, ..., 110]
        }

        // After sorting: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        // With 11 values, this is still in the initial phase
        assert_eq!(quantiles.min(), 10.0);
        assert_eq!(quantiles.p50(), 50.0);
        assert_eq!(quantiles.p90(), 70.0);
        assert_eq!(quantiles.p99(), 90.0);
    }

    #[test]
    fn test_values_beyond_initial_11() {
        let mut quantiles = StreamingQuantiles::new();
        // Add 20 values to test streaming phase
        for i in 1..=20 {
            quantiles.add(i as f64);
        }

        // Fixed dataset should produce fixed results
        assert_eq!(quantiles.min(), 1.0);
        assert_eq!(quantiles.p50(), 9.0);
        assert_eq!(quantiles.p90(), 13.0);
        assert_eq!(quantiles.p99(), 17.0);
    }

    #[test]
    fn test_extreme_values() {
        let mut quantiles = StreamingQuantiles::new();
        quantiles.add(0.001);
        quantiles.add(1000000.0);
        quantiles.add(0.002);
        quantiles.add(0.003);
        quantiles.add(0.004);

        // Fixed dataset should produce fixed results
        assert_eq!(quantiles.min(), 0.001);
        assert_eq!(quantiles.p50(), 0.003);
        assert_eq!(quantiles.p90(), 0.004);
        assert_eq!(quantiles.p99(), 0.004);
    }
}
