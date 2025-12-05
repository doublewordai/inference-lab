/// Streaming quantile estimation using P-Squared algorithm
/// Maintains approximate quantiles (p50, p90, p99) in O(1) time and space
pub struct StreamingQuantiles {
    // Marker positions and heights for P² algorithm
    // We track 5 markers for p50, p90, p99
    markers: [f64; 11],  // Marker heights (actual values)
    positions: [f64; 11], // Marker positions (count-based)
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
                self.markers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        self.desired_positions[1] = 1.0 + 0.01 * (n - 1.0);   // p1
        self.desired_positions[2] = 1.0 + 0.10 * (n - 1.0);   // p10
        self.desired_positions[3] = 1.0 + 0.25 * (n - 1.0);   // p25
        self.desired_positions[4] = 1.0 + 0.50 * (n - 1.0);   // p50
        self.desired_positions[5] = 1.0 + 0.75 * (n - 1.0);   // p75
        self.desired_positions[6] = 1.0 + 0.90 * (n - 1.0);   // p90
        self.desired_positions[7] = 1.0 + 0.95 * (n - 1.0);   // p95
        self.desired_positions[8] = 1.0 + 0.99 * (n - 1.0);   // p99
        self.desired_positions[9] = 1.0 + 0.999 * (n - 1.0);  // p99.9
        self.desired_positions[10] = n;
    }

    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let q_i = self.markers[i];
        let q_i1 = self.markers[i + 1];
        let q_i_1 = self.markers[i - 1];
        let n_i = self.positions[i];
        let n_i1 = self.positions[i + 1];
        let n_i_1 = self.positions[i - 1];

        q_i + d / (n_i1 - n_i_1) * (
            (n_i - n_i_1 + d) * (q_i1 - q_i) / (n_i1 - n_i)
            + (n_i1 - n_i - d) * (q_i - q_i_1) / (n_i - n_i_1)
        )
    }

    fn linear(&self, i: usize, d: f64) -> f64 {
        let d_i = if d > 0.0 { 1 } else { -1 };
        let q_i = self.markers[i];
        let q_id = self.markers[(i as i32 + d_i) as usize];
        let n_i = self.positions[i];
        let n_id = self.positions[(i as i32 + d_i) as usize];

        q_i + d * (q_id - q_i) / (n_id - n_i)
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
