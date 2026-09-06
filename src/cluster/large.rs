//! First-pass clustering: consecutive pings inside large-area POI polygons.

use crate::config::Config;
use crate::geo::{point_in_polygon, ring_area_m2};
use crate::types::{Cluster, GpsPing, Place, Point};

use super::Clusterer;

/// First-pass clustering: consecutive pings inside large-area POI polygons.
///
/// Footprint area is computed from each place's exterior ring via
/// [`ring_area_m2`]. When a ping lies in multiple large POIs, the stay is
/// attributed to the smallest computed area (lowest [`PlaceId`] on ties), not
/// catalog iteration order.
///
/// [`PlaceId`]: crate::types::PlaceId
#[derive(Debug, Clone)]
pub struct LargePoiClusterer {
    config: Config,
}

impl LargePoiClusterer {
    /// Create a large-POI clusterer from configuration.
    #[must_use]
    pub const fn new(config: Config) -> Self {
        Self { config }
    }

    /// Extract large-POI clusters and a mask of consumed ping indices.
    ///
    /// Places qualify when their ring area is at least `large_poi_area_m2`.
    /// Overlaps resolve to smallest ring area, then lowest place id.
    #[must_use]
    pub fn extract(&self, pings: &[GpsPing], places: &[Place]) -> (Vec<Cluster>, Vec<bool>) {
        let mut used = vec![false; pings.len()];
        if pings.is_empty() {
            return (Vec::new(), used);
        }
        let large: Vec<&Place> = places
            .iter()
            .filter(|p| ring_area_m2(&p.polygon) >= self.config.large_poi_area_m2)
            .collect();
        let mut clusters = Vec::new();
        let mut i = 0usize;
        while i < pings.len() {
            let seed = pings[i];
            let Some(pid) = smallest_containing_large_place(seed.point, &large) else {
                i += 1;
                continue;
            };
            let start = i;
            i = extend_large_run(pings, &large, pid, start);
            push_large_cluster(
                pings,
                start,
                i,
                self.config.min_cluster_pings,
                &mut used,
                &mut clusters,
            );
        }
        (clusters, used)
    }
}

fn smallest_containing_large_place(point: Point, large: &[&Place]) -> Option<u64> {
    large
        .iter()
        .filter(|place| point_in_polygon(point, &place.polygon))
        .min_by(|a, b| {
            ring_area_m2(&a.polygon)
                .partial_cmp(&ring_area_m2(&b.polygon))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        })
        .map(|p| p.id)
}

fn extend_large_run(pings: &[GpsPing], large: &[&Place], pid: u64, start: usize) -> usize {
    let mut i = start + 1;
    while i < pings.len() {
        let cur = pings[i];
        let still = large
            .iter()
            .any(|place| place.id == pid && point_in_polygon(cur.point, &place.polygon));
        if !still {
            break;
        }
        i += 1;
    }
    i
}

const fn large_run_in_bounds(
    start: usize,
    end: usize,
    min_cluster_pings: usize,
    n_pings: usize,
    n_used: usize,
) -> bool {
    end - start >= min_cluster_pings && end <= n_pings && end <= n_used
}

fn push_large_cluster(
    pings: &[GpsPing],
    start: usize,
    end: usize,
    min_cluster_pings: usize,
    used: &mut [bool],
    clusters: &mut Vec<Cluster>,
) {
    if !large_run_in_bounds(start, end, min_cluster_pings, pings.len(), used.len()) {
        return;
    }
    for flag in &mut used[start..end] {
        *flag = true;
    }
    if let Ok(cluster) = Cluster::from_pings(pings[start..end].to_vec()) {
        clusters.push(cluster);
    }
}

impl Clusterer for LargePoiClusterer {
    fn cluster(&self, pings: &[GpsPing], places: &[Place]) -> Vec<Cluster> {
        self.extract(pings, places).0
    }
}
