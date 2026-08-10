//! Pipeline hyperparameters for cleaning, clustering, and joining.

use crate::error::{Error, Result};

/// Tunable hyperparameters for cleaning, clustering, and place joining.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Drop pings with horizontal accuracy above this many meters.
    pub max_horizontal_accuracy_m: f64,
    /// Drop consecutive pairs whose implied speed exceeds this (m/s).
    pub max_speed_m_s: f64,
    /// Max `path/net` ratio at or below which a window counts as linear (driving).
    ///
    /// Values near `1.0` are straighter; the default `1.15` allows modest wiggle.
    pub linearity_threshold: f64,
    /// Minimum window length in seconds for the linearity driving filter.
    pub linearity_window_s: f64,
    /// Speeds above this (m/s) inside a linear window count as driving.
    pub driving_speed_m_s: f64,
    /// Max distance (m) between consecutive pings in a time-aware cluster.
    pub dist_threshold_m: f64,
    /// Max distance (m) from the last ping before the cluster breaks.
    pub max_dist_threshold_m: f64,
    /// Max seconds between consecutive pings before a density cluster breaks.
    pub max_time_gap_s: f64,
    /// Minimum pings required to form a density cluster.
    pub min_cluster_pings: usize,
    /// Places with area at or above this (m²) use the large-POI pass.
    pub large_poi_area_m2: f64,
    /// Extra buffer (m) around cluster centroids when joining places.
    pub join_buffer_m: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_horizontal_accuracy_m: 100.0,
            max_speed_m_s: 50.0,
            linearity_threshold: 1.15,
            linearity_window_s: 60.0,
            driving_speed_m_s: 8.0,
            dist_threshold_m: 80.0,
            max_dist_threshold_m: 100.0,
            max_time_gap_s: 1_800.0,
            min_cluster_pings: 2,
            large_poi_area_m2: 50_000.0,
            join_buffer_m: 50.0,
        }
    }
}

impl Config {
    /// Start a fluent builder with default values.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// Check that hyperparameters are finite and internally consistent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when a field is non-finite or
    /// non-positive where required.
    pub fn validate(&self) -> Result<()> {
        for (name, value, allow_zero) in [
            (
                "max_horizontal_accuracy_m",
                self.max_horizontal_accuracy_m,
                false,
            ),
            ("max_speed_m_s", self.max_speed_m_s, false),
            ("linearity_threshold", self.linearity_threshold, false),
            ("linearity_window_s", self.linearity_window_s, false),
            ("driving_speed_m_s", self.driving_speed_m_s, false),
            ("dist_threshold_m", self.dist_threshold_m, false),
            ("max_dist_threshold_m", self.max_dist_threshold_m, false),
            ("max_time_gap_s", self.max_time_gap_s, false),
            ("large_poi_area_m2", self.large_poi_area_m2, false),
            ("join_buffer_m", self.join_buffer_m, true),
        ] {
            if !value.is_finite() || (allow_zero && value < 0.0) || (!allow_zero && value <= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "`{name}` must be finite and {}",
                    if allow_zero { ">= 0" } else { "> 0" }
                )));
            }
        }
        if self.max_dist_threshold_m < self.dist_threshold_m {
            return Err(Error::InvalidInput(
                "`max_dist_threshold_m` must be >= `dist_threshold_m`".into(),
            ));
        }
        if self.min_cluster_pings < 1 {
            return Err(Error::InvalidInput(
                "`min_cluster_pings` must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// Fluent builder for [`Config`].
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a builder seeded with [`Config::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum accepted horizontal accuracy in meters.
    pub fn max_horizontal_accuracy_m(mut self, v: f64) -> Self {
        self.config.max_horizontal_accuracy_m = v;
        self
    }

    /// Set the maximum implied speed between consecutive pings (m/s).
    pub fn max_speed_m_s(mut self, v: f64) -> Self {
        self.config.max_speed_m_s = v;
        self
    }

    /// Set max `path/net` for a window to count as linear (`<=` this value).
    pub fn linearity_threshold(mut self, v: f64) -> Self {
        self.config.linearity_threshold = v;
        self
    }

    /// Set the minimum window duration for the linearity filter (seconds).
    pub fn linearity_window_s(mut self, v: f64) -> Self {
        self.config.linearity_window_s = v;
        self
    }

    /// Set the minimum speed (m/s) that marks a linear window as driving.
    pub fn driving_speed_m_s(mut self, v: f64) -> Self {
        self.config.driving_speed_m_s = v;
        self
    }

    /// Set the neighbor distance threshold for density clustering (meters).
    pub fn dist_threshold_m(mut self, v: f64) -> Self {
        self.config.dist_threshold_m = v;
        self
    }

    /// Set the maximum jump from the last ping within a cluster (meters).
    pub fn max_dist_threshold_m(mut self, v: f64) -> Self {
        self.config.max_dist_threshold_m = v;
        self
    }

    /// Set the max seconds between consecutive pings within a density cluster.
    pub fn max_time_gap_s(mut self, v: f64) -> Self {
        self.config.max_time_gap_s = v;
        self
    }

    /// Set the minimum number of pings required to form a cluster.
    pub fn min_cluster_pings(mut self, v: usize) -> Self {
        self.config.min_cluster_pings = v;
        self
    }

    /// Set the area (m²) at which places use the large-POI clustering pass.
    pub fn large_poi_area_m2(mut self, v: f64) -> Self {
        self.config.large_poi_area_m2 = v;
        self
    }

    /// Set the join buffer around clusters when matching places (meters).
    pub fn join_buffer_m(mut self, v: f64) -> Self {
        self.config.join_buffer_m = v;
        self
    }

    /// Validate and return the built configuration.
    ///
    /// # Errors
    ///
    /// Propagates [`Config::validate`] failures.
    pub fn build(self) -> Result<Config> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn rejects_inverted_distance_thresholds() {
        let err = Config::builder()
            .dist_threshold_m(100.0)
            .max_dist_threshold_m(50.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn rejects_non_positive_max_time_gap() {
        let err = Config::builder().max_time_gap_s(0.0).build().unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }
}
