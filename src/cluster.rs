//! Clustering GPS pings into potential visits.

use crate::config::Config;
use crate::geo::{haversine_m, point_in_polygon, ring_area_m2};
use crate::types::{Cluster, GpsPing, Place, Point};

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
///
/// Footprint area is computed from each place's exterior ring via
/// [`ring_area_m2`]. When a ping lies in multiple large POIs, the stay is
/// attributed to the smallest computed area (lowest [`PlaceId`] on ties), not
/// catalog iteration order.
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
            let Some(seed) = pings.get(i) else {
                break;
            };
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
        let Some(cur) = pings.get(i) else {
            break;
        };
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

fn push_large_cluster(
    pings: &[GpsPing],
    start: usize,
    end: usize,
    min_cluster_pings: usize,
    used: &mut [bool],
    clusters: &mut Vec<Cluster>,
) {
    if end - start < min_cluster_pings {
        return;
    }
    if let Some(flags) = used.get_mut(start..end) {
        for flag in flags {
            *flag = true;
        }
    }
    if let Some(slice) = pings.get(start..end) {
        if let Ok(cluster) = Cluster::from_pings(slice.to_vec()) {
            clusters.push(cluster);
        }
    }
}

impl Clusterer for LargePoiClusterer {
    fn cluster(&self, pings: &[GpsPing], places: &[Place]) -> Vec<Cluster> {
        self.extract(pings, places).0
    }
}

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
            let Some(&seed) = pings.get(i) else {
                break;
            };
            let mut members = vec![seed];
            let mut last = seed;
            i += 1;
            while i < pings.len() {
                let Some(&cur) = pings.get(i) else {
                    break;
                };
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
    // Require proximity to a recent member within dist_threshold.
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

fn push_if_dense_enough(
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

/// Two-pass clustering: large POI containment, then time-aware density.
#[derive(Debug, Clone)]
pub struct TwoPassClusterer {
    large: LargePoiClusterer,
    density: TimeAwareDensityClusterer,
}

impl TwoPassClusterer {
    /// Create a two-pass clusterer from configuration.
    #[must_use]
    pub const fn new(config: Config) -> Self {
        Self {
            large: LargePoiClusterer::new(config),
            density: TimeAwareDensityClusterer::new(config),
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
        density_cluster_unused_runs(&self.density, pings, &used, &mut clusters);
        clusters
    }
}

fn extend_density_run(
    density: &TimeAwareDensityClusterer,
    pings: &[GpsPing],
    start: usize,
    end: usize,
    clusters: &mut Vec<Cluster>,
) {
    if let Some(run) = pings.get(start..end) {
        clusters.extend(density.cluster(run, &[]));
    }
}

fn density_cluster_unused_runs(
    density: &TimeAwareDensityClusterer,
    pings: &[GpsPing],
    used: &[bool],
    clusters: &mut Vec<Cluster>,
) {
    let mut run_start: Option<usize> = None;
    for (idx, &is_used) in used.iter().enumerate() {
        if !is_used {
            if run_start.is_none() {
                run_start = Some(idx);
            }
        } else if let Some(start) = run_start.take() {
            extend_density_run(density, pings, start, idx, clusters);
        }
    }
    if let Some(start) = run_start {
        extend_density_run(density, pings, start, pings.len(), clusters);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::types::{PlaceId, Point};

    fn ping(lat: f64, lon: f64, t: f64) -> GpsPing {
        GpsPing::new(lat, lon, t, 10.0)
    }

    /// Axis-aligned square centered at the origin with half-side `half_deg`.
    fn square_place(id: PlaceId, half_deg: f64) -> Place {
        Place::new(
            id,
            vec![
                Point::new(-half_deg, -half_deg),
                Point::new(-half_deg, half_deg),
                Point::new(half_deg, half_deg),
                Point::new(half_deg, -half_deg),
                Point::new(-half_deg, -half_deg),
            ],
            Point::new(0.0, 0.0),
            None,
        )
    }

    fn density() -> TimeAwareDensityClusterer {
        TimeAwareDensityClusterer::new(Config::default())
    }

    fn large_cfg() -> Result<Config> {
        Config::builder().large_poi_area_m2(100.0).min_cluster_pings(2).build()
    }

    fn assert_split_clusters(clusters: &[Cluster], used: &[bool]) {
        assert_eq!(used, [true, true, true, true]);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].pings.len(), 2);
        assert_eq!(clusters[1].pings.len(), 2);
        assert!((clusters[0].end_time_s - 10.0).abs() < f64::EPSILON);
        assert!((clusters[1].start_time_s - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn clusters_nearby_stationary_pings() {
        let clusterer = TwoPassClusterer::new(Config::default());
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
    fn large_poi_pass_consumes_interior_pings() -> Result<()> {
        let place = square_place(1, 0.001);
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.0, 0.0001, 20.0),
        ];
        let clusters = TwoPassClusterer::new(large_cfg()?).cluster(&pings, &[place]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pings.len(), 3);
        Ok(())
    }

    #[test]
    fn large_poi_overlap_prefers_smallest_area_not_catalog_order() -> Result<()> {
        let mall = square_place(2, 0.002);
        let plaza = square_place(1, 0.0005);
        let pings = vec![
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.001, 0.0, 20.0),
            ping(0.0011, 0.0, 30.0),
        ];
        let clusterer = LargePoiClusterer::new(large_cfg()?);
        let (clusters, used) = clusterer.extract(&pings, &[mall.clone(), plaza.clone()]);
        assert_split_clusters(&clusters, &used);
        let (rev_clusters, rev_used) = clusterer.extract(&pings, &[plaza, mall]);
        assert_eq!(rev_used, used);
        assert_eq!(rev_clusters.len(), clusters.len());
        assert_eq!(rev_clusters[0].pings.len(), clusters[0].pings.len());
        assert_eq!(rev_clusters[1].pings.len(), clusters[1].pings.len());
        Ok(())
    }

    #[test]
    fn large_poi_without_overlap_still_uses_containing_place() -> Result<()> {
        let mall = square_place(1, 0.002);
        let pings = vec![ping(0.001, 0.0, 0.0), ping(0.0011, 0.0, 10.0)];
        let (clusters, used) = LargePoiClusterer::new(large_cfg()?).extract(&pings, &[mall]);
        assert_eq!(used, vec![true, true]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pings.len(), 2);
        Ok(())
    }

    #[test]
    fn density_pass_does_not_merge_across_large_poi_gap() -> Result<()> {
        let mall = square_place(1, 0.001);
        let mall_start = 1_000.0;
        let mall_end = 1_010.0;
        let pings = vec![
            ping(0.002, 0.0, 0.0),
            ping(0.0021, 0.0, 10.0),
            ping(0.0, 0.0, mall_start),
            ping(0.0001, 0.0, mall_end),
            ping(0.002, 0.0, 2_000.0),
            ping(0.0021, 0.0, 2_010.0),
        ];
        let clusters = TwoPassClusterer::new(large_cfg()?).cluster(&pings, &[mall]);
        assert_eq!(clusters.len(), 3);
        assert!(!clusters.iter().any(|c| c.start_time_s < mall_start && c.end_time_s > mall_end));
        Ok(())
    }

    #[test]
    fn density_clustering_gap_and_short_window_behavior() {
        let short = density().cluster(
            &[
                ping(0.0, 0.0, 0.0),
                ping(0.0001, 0.0, 10.0),
                ping(0.0, 0.0001, 20.0),
            ],
            &[],
        );
        assert_eq!(short.len(), 1);

        let gapped = density().cluster(
            &[
                ping(0.0, 0.0, 0.0),
                ping(0.0001, 0.0, 10.0),
                ping(0.0, 0.0, 10_000.0),
                ping(0.0001, 0.0, 10_010.0),
            ],
            &[],
        );
        assert_eq!(gapped.len(), 2);
        assert!(!gapped.iter().any(|c| c.start_time_s < 10.0 && c.end_time_s > 10_000.0));

        let kept = density().cluster(
            &[
                ping(0.0, 0.0, 0.0),
                ping(0.0001, 0.0, 30.0),
                ping(0.0, 0.0001, 60.0),
            ],
            &[],
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].pings.len(), 3);

        // ~222 m hop exceeds max_dist_threshold_m (100 m).
        let jumped = density().cluster(
            &[
                ping(0.0, 0.0, 0.0),
                ping(0.0001, 0.0, 10.0),
                ping(0.002, 0.0, 20.0),
                ping(0.0021, 0.0, 30.0),
            ],
            &[],
        );
        assert!(!jumped.iter().any(|c| c.start_time_s < 10.0 && c.end_time_s > 20.0));
    }
}
