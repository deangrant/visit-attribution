//! Clustering GPS pings into potential visits.

use crate::config::Config;
use crate::geo::{haversine_m, point_in_polygon};
use crate::types::{Cluster, GpsPing, Place};

/// Groups cleaned pings into visit candidates without ranking places.
///
/// # Contract
///
/// - Empty `pings` yields an empty result.
/// - Callers should pass time-ordered pings (as produced by cleaning).
/// - Implementations may ignore `places` when geometry is not required.
/// - Noise / undersized runs are discarded rather than returned as clusters.
pub trait Clusterer {
    /// Produce clusters from time-ordered pings and optional place polygons.
    fn cluster(&self, pings: &[GpsPing], places: &[Place]) -> Vec<Cluster>;
}

/// First-pass clustering: consecutive pings inside large-area POI polygons.
#[derive(Debug, Clone)]
pub struct LargePoiClusterer {
    config: Config,
}

impl LargePoiClusterer {
    /// Create a large-POI clusterer from configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Extract large-POI clusters and a mask of consumed ping indices.
    #[must_use]
    pub fn extract(&self, pings: &[GpsPing], places: &[Place]) -> (Vec<Cluster>, Vec<bool>) {
        let mut used = vec![false; pings.len()];
        if pings.is_empty() {
            return (Vec::new(), used);
        }
        let large: Vec<&Place> =
            places.iter().filter(|p| p.area_m2 >= self.config.large_poi_area_m2).collect();
        let mut clusters = Vec::new();
        let mut i = 0usize;
        while i < pings.len() {
            let place_id = large.iter().find_map(|place| {
                point_in_polygon(pings[i].point, &place.polygon).then_some(place.id)
            });
            let Some(pid) = place_id else {
                i += 1;
                continue;
            };
            let start = i;
            i += 1;
            while i < pings.len() {
                let still = large.iter().any(|place| {
                    place.id == pid && point_in_polygon(pings[i].point, &place.polygon)
                });
                if !still {
                    break;
                }
                i += 1;
            }
            if i - start >= self.config.min_cluster_pings {
                for flag in &mut used[start..i] {
                    *flag = true;
                }
                clusters.push(Cluster::from_pings(pings[start..i].to_vec()));
            }
        }
        (clusters, used)
    }
}

impl Clusterer for LargePoiClusterer {
    fn cluster(&self, pings: &[GpsPing], places: &[Place]) -> Vec<Cluster> {
        self.extract(pings, places).0
    }
}

/// Time-aware density clustering on a time-ordered ping run.
#[derive(Debug, Clone)]
pub struct TimeAwareDbscan {
    config: Config,
}

impl TimeAwareDbscan {
    /// Create a density clusterer from configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

impl Clusterer for TimeAwareDbscan {
    fn cluster(&self, pings: &[GpsPing], _places: &[Place]) -> Vec<Cluster> {
        if pings.is_empty() {
            return Vec::new();
        }
        let dist_threshold_m = self.config.dist_threshold_m;
        let max_dist_threshold_m = self.config.max_dist_threshold_m;
        let max_time_gap_s = self.config.max_time_gap_s;
        let min_cluster_pings = self.config.min_cluster_pings;
        let mut clusters = Vec::new();
        let mut i = 0usize;
        while i < pings.len() {
            let mut members = vec![pings[i]];
            let mut last = pings[i];
            i += 1;
            while i < pings.len() {
                let cur = pings[i];
                if cur.time_s - last.time_s > max_time_gap_s {
                    break;
                }
                let to_last = haversine_m(last.point, cur.point);
                if to_last > max_dist_threshold_m {
                    break;
                }
                // Require proximity to a recent member within dist_threshold.
                let near = members
                    .iter()
                    .rev()
                    .take(8)
                    .any(|m| haversine_m(m.point, cur.point) <= dist_threshold_m);
                if !near && to_last > dist_threshold_m {
                    break;
                }
                members.push(cur);
                last = cur;
                i += 1;
            }
            if members.len() >= min_cluster_pings {
                clusters.push(Cluster::from_pings(members));
            }
        }
        clusters
    }
}

/// Two-pass clustering: large POI containment, then time-aware density.
#[derive(Debug, Clone)]
pub struct TwoPassClusterer {
    large: LargePoiClusterer,
    density: TimeAwareDbscan,
}

impl TwoPassClusterer {
    /// Create a two-pass clusterer from configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            large: LargePoiClusterer::new(config.clone()),
            density: TimeAwareDbscan::new(config),
        }
    }
}

impl Clusterer for TwoPassClusterer {
    fn cluster(&self, pings: &[GpsPing], places: &[Place]) -> Vec<Cluster> {
        if pings.is_empty() {
            return Vec::new();
        }
        let (mut clusters, used) = self.large.extract(pings, places);
        // Density-cluster each contiguous unused run so gaps left by the
        // large-POI pass cannot merge non-adjacent stays into one visit.
        let mut run_start: Option<usize> = None;
        for (idx, &is_used) in used.iter().enumerate() {
            if !is_used {
                if run_start.is_none() {
                    run_start = Some(idx);
                }
            } else if let Some(start) = run_start.take() {
                clusters.extend(self.density.cluster(&pings[start..idx], &[]));
            }
        }
        if let Some(start) = run_start {
            clusters.extend(self.density.cluster(&pings[start..], &[]));
        }
        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn ping(lat: f64, lon: f64, t: f64) -> GpsPing {
        GpsPing::new(lat, lon, t, 10.0)
    }

    #[test]
    fn clusters_nearby_stationary_pings() {
        let cfg = Config::default();
        let clusterer = TwoPassClusterer::new(cfg);
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.0, 0.0001, 20.0),
            ping(1.0, 1.0, 10_000.0),
            ping(1.0001, 1.0, 10_010.0),
        ];
        let clusters = clusterer.cluster(&pings, &[]);
        assert!(!clusters.is_empty());
        assert!(clusters.iter().any(|c| c.pings.len() >= 2));
    }

    #[test]
    fn large_poi_pass_consumes_interior_pings() {
        let cfg = Config::builder().large_poi_area_m2(100.0).min_cluster_pings(2).build().unwrap();
        let place = Place::new(
            1,
            vec![
                Point::new(-0.001, -0.001),
                Point::new(-0.001, 0.001),
                Point::new(0.001, 0.001),
                Point::new(0.001, -0.001),
                Point::new(-0.001, -0.001),
            ],
            Point::new(0.0, 0.0),
            None,
            10_000.0,
        );
        let clusterer = TwoPassClusterer::new(cfg);
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.0, 0.0001, 20.0),
        ];
        let clusters = clusterer.cluster(&pings, &[place]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pings.len(), 3);
    }

    #[test]
    fn density_pass_does_not_merge_across_large_poi_gap() {
        let cfg = Config::builder().large_poi_area_m2(100.0).min_cluster_pings(2).build().unwrap();
        let mall = Place::new(
            1,
            vec![
                Point::new(-0.001, -0.001),
                Point::new(-0.001, 0.001),
                Point::new(0.001, 0.001),
                Point::new(0.001, -0.001),
                Point::new(-0.001, -0.001),
            ],
            Point::new(0.0, 0.0),
            None,
            10_000.0,
        );
        let mall_start = 1_000.0;
        let mall_end = 1_010.0;
        let clusterer = TwoPassClusterer::new(cfg);
        // Cafe → mall → same cafe. Without contiguous-run density, the two
        // cafe stays compact into one visit spanning the mall.
        let pings = vec![
            ping(0.002, 0.0, 0.0),
            ping(0.0021, 0.0, 10.0),
            ping(0.0, 0.0, mall_start),
            ping(0.0001, 0.0, mall_end),
            ping(0.002, 0.0, 2_000.0),
            ping(0.0021, 0.0, 2_010.0),
        ];
        let clusters = clusterer.cluster(&pings, &[mall]);
        assert_eq!(clusters.len(), 3);
        assert!(!clusters
            .iter()
            .any(|c| c.start_time_s < mall_start && c.end_time_s > mall_end));
    }

    #[test]
    fn density_only_clusterer_ignores_places() {
        let density = TimeAwareDbscan::new(Config::default());
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.0, 0.0001, 20.0),
        ];
        let clusters = density.cluster(&pings, &[]);
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn density_breaks_on_large_temporal_gap() {
        let density = TimeAwareDbscan::new(Config::default());
        // Same place morning then evening: must not merge across the gap.
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.0, 0.0, 10_000.0),
            ping(0.0001, 0.0, 10_010.0),
        ];
        let clusters = density.cluster(&pings, &[]);
        assert_eq!(clusters.len(), 2);
        assert!(!clusters.iter().any(|c| c.start_time_s < 10.0 && c.end_time_s > 10_000.0));
    }

    #[test]
    fn density_keeps_short_gaps_in_one_cluster() {
        let density = TimeAwareDbscan::new(Config::default());
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 30.0),
            ping(0.0, 0.0001, 60.0),
        ];
        let clusters = density.cluster(&pings, &[]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pings.len(), 3);
    }
}
