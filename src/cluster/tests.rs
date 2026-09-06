use super::density::push_if_dense_enough;
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
fn density_clustering_keeps_a_short_stationary_window() {
    let short = density().cluster(
        &[
            ping(0.0, 0.0, 0.0),
            ping(0.0001, 0.0, 10.0),
            ping(0.0, 0.0001, 20.0),
        ],
        &[],
    );
    assert_eq!(short.len(), 1);
}

#[test]
fn density_clustering_splits_on_time_gap() {
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
}

#[test]
fn density_clustering_keeps_slow_hops_in_one_cluster() {
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
}

#[test]
fn density_clustering_splits_on_max_distance_jump() {
    // About 222 m exceeds max_dist_threshold_m (100 m).
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

#[test]
fn density_clustering_rejects_mid_range_hops_without_neighbors() {
    // About 89 m: above dist_threshold_m (80) but below max_dist_threshold_m (100).
    let mid = density().cluster(
        &[
            ping(0.0, 0.0, 0.0),
            ping(0.0008, 0.0, 10.0),
            ping(0.0016, 0.0, 20.0),
        ],
        &[],
    );
    assert!(mid.iter().all(|c| c.pings.len() < 3));
}

#[test]
fn empty_inputs_and_large_poi_trait_cluster() -> Result<()> {
    assert!(density().cluster(&[], &[]).is_empty());
    assert!(TwoPassClusterer::new(Config::default()).cluster(&[], &[]).is_empty());
    let (clusters, used) = LargePoiClusterer::new(large_cfg()?).extract(&[], &[]);
    assert!(clusters.is_empty());
    assert!(used.is_empty());
    let place = square_place(1, 0.001);
    let pings = [ping(0.0, 0.0, 0.0), ping(0.0001, 0.0, 10.0)];
    let via_trait = LargePoiClusterer::new(large_cfg()?).cluster(&pings, &[place]);
    assert_eq!(via_trait.len(), 1);
    Ok(())
}

#[test]
fn short_large_poi_run_is_discarded() -> Result<()> {
    let place = square_place(1, 0.001);
    let (clusters, used) =
        LargePoiClusterer::new(large_cfg()?).extract(&[ping(0.0, 0.0, 0.0)], &[place]);
    assert!(clusters.is_empty());
    assert_eq!(used, [false]);
    let mut out = Vec::new();
    push_if_dense_enough(&mut out, Vec::new(), 0);
    assert!(out.is_empty());
    push_if_dense_enough(&mut out, vec![ping(0.0, 0.0, 0.0)], 2);
    assert!(out.is_empty());
    Ok(())
}
