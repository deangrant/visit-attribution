//! Pipeline hyperparameters for cleaning, clustering, and joining.

use crate::error::{Error, Result};

/// Tunable hyperparameters for cleaning, clustering, and place joining.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Max distance (m) from a place polygon for join candidates.
    pub join_radius_m: f64,
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
            join_radius_m: 50.0,
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
            ("join_radius_m", self.join_radius_m, true),
        ] {
            require_finite(name, value, allow_zero)?;
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

fn require_finite(name: &str, value: f64, allow_zero: bool) -> Result<()> {
    if !value.is_finite() || (allow_zero && value < 0.0) || (!allow_zero && value <= 0.0) {
        return Err(Error::InvalidInput(format!(
            "`{name}` must be finite and {}",
            if allow_zero { ">= 0" } else { "> 0" }
        )));
    }
    Ok(())
}

/// Fluent builder for [`Config`].
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a builder seeded with [`Config::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum number of pings required to form a cluster.
    pub const fn min_cluster_pings(mut self, v: usize) -> Self {
        self.config.min_cluster_pings = v;
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

macro_rules! config_f64_setter {
    ($name:ident, $field:ident, $doc:expr) => {
        impl ConfigBuilder {
            #[doc = $doc]
            pub const fn $name(mut self, v: f64) -> Self {
                self.config.$field = v;
                self
            }
        }
    };
}

config_f64_setter!(
    max_horizontal_accuracy_m,
    max_horizontal_accuracy_m,
    "Set the maximum accepted horizontal accuracy in meters."
);
config_f64_setter!(
    max_speed_m_s,
    max_speed_m_s,
    "Set the maximum implied speed between consecutive pings (m/s)."
);
config_f64_setter!(
    linearity_threshold,
    linearity_threshold,
    "Set max `path/net` for a window to count as linear (`<=` this value)."
);
config_f64_setter!(
    linearity_window_s,
    linearity_window_s,
    "Set the minimum window duration for the linearity filter (seconds)."
);
config_f64_setter!(
    driving_speed_m_s,
    driving_speed_m_s,
    "Set the minimum speed (m/s) that marks a linear window as driving."
);
config_f64_setter!(
    dist_threshold_m,
    dist_threshold_m,
    "Set the neighbor distance threshold for density clustering (meters)."
);
config_f64_setter!(
    max_dist_threshold_m,
    max_dist_threshold_m,
    "Set the maximum jump from the last ping within a cluster (meters)."
);
config_f64_setter!(
    max_time_gap_s,
    max_time_gap_s,
    "Set the max seconds between consecutive pings within a density cluster."
);
config_f64_setter!(
    large_poi_area_m2,
    large_poi_area_m2,
    "Set the area (m²) at which places use the large-POI clustering pass."
);
config_f64_setter!(
    join_radius_m,
    join_radius_m,
    "Set the max distance (m) from a place polygon for join candidates."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn default_config_validates() -> Result<()> {
        Config::default().validate()?;
        assert_eq!(ConfigBuilder::new().build()?, Config::default());
        Ok(())
    }

    #[test]
    fn rejects_inverted_distance_thresholds() -> Result<()> {
        let err = crate::test_util::err(
            Config::builder().dist_threshold_m(100.0).max_dist_threshold_m(50.0).build(),
        )?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn rejects_non_positive_max_time_gap() -> Result<()> {
        let err = crate::test_util::err(Config::builder().max_time_gap_s(0.0).build())?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn rejects_negative_join_radius_and_zero_min_pings() -> Result<()> {
        Config::builder().join_radius_m(0.0).build()?;
        let err = crate::test_util::err(Config::builder().join_radius_m(-1.0).build())?;
        assert!(matches!(err, Error::InvalidInput(_)));
        let err = crate::test_util::err(Config::builder().min_cluster_pings(0).build())?;
        assert!(matches!(err, Error::InvalidInput(_)));
        let err = crate::test_util::err(Config::builder().join_radius_m(f64::NAN).build())?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }
}
