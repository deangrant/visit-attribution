//! Time-aware sequential density clustering on a time-ordered ping run.

use crate::config::Config;
use crate::geo::haversine_m;
use crate::types::{Cluster, GpsPing, Place};

use super::Clusterer;

/// Time-aware sequential density clustering on a time-ordered ping run.
///
/// Grows contiguous runs using distance and time-gap thresholds against recent
/// members. This is not DBSCAN (no ε-neighborhood graph or core/border points).
#[derive(Debug, Clone)]
pub struct TimeAwareDensityClusterer {
    config: Config,
}

impl TimeAwareDensityClusterer {
    /// Create a density clusterer from configuration.
    #[must_use]
    pub const fn new(config: Config) -> Self {
        Self { config }
    }
}

impl Clusterer for TimeAwareDensityClusterer {
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
            let seed = pings[i];
            let mut members = vec![seed];
            let mut last = seed;
            i += 1;
            while i < pings.len() {
                let cur = pings[i];
                if !try_extend_density_member(
                    last,
                    cur,
                    &members,
                    dist_threshold_m,
                    max_dist_threshold_m,
                    max_time_gap_s,
                ) {
                    break;
                }
                members.push(cur);
                last = cur;
                i += 1;
            }
            push_if_dense_enough(&mut clusters, members, min_cluster_pings);
        }
        clusters
    }
}

fn try_extend_density_member(
    last: GpsPing,
    cur: GpsPing,
    members: &[GpsPing],
    dist_threshold_m: f64,
    max_dist_threshold_m: f64,
    max_time_gap_s: f64,
) -> bool {
    if cur.time_s - last.time_s > max_time_gap_s {
        return false;
    }
    let to_last = haversine_m(last.point, cur.point);
    if to_last > max_dist_threshold_m {
        return false;
    }
    // Require proximity to a recent member within the neighbor distance.
    let near = members
        .iter()
        .rev()
        .take(8)
        .any(|m| haversine_m(m.point, cur.point) <= dist_threshold_m);
    if !near && to_last > dist_threshold_m {
        return false;
    }
    true
}

pub(super) fn push_if_dense_enough(
    clusters: &mut Vec<Cluster>,
    members: Vec<GpsPing>,
    min_cluster_pings: usize,
) {
    if members.len() >= min_cluster_pings {
        if let Ok(cluster) = Cluster::from_pings(members) {
            clusters.push(cluster);
        }
    }
}
