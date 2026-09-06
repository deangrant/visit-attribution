//! Clustering GPS pings into potential visits.

mod density;
mod large;

use crate::config::Config;
use crate::types::{Cluster, GpsPing, Place};

pub use density::TimeAwareDensityClusterer;
pub use large::LargePoiClusterer;

/// Groups cleaned pings into visit candidates without ranking places.
///
/// # Contract
///
/// - Empty `pings` yields an empty result.
/// - Callers should pass time-ordered pings (as produced by cleaning).
/// - Implementations may ignore `places` when geometry is not required.
/// - Noise, undersized runs, and failed [`Cluster::from_pings`] construction
///   are discarded rather than returned as clusters.
pub trait Clusterer {
    /// Produce clusters from time-ordered pings and optional place polygons.
    fn cluster(&self, pings: &[GpsPing], places: &[Place]) -> Vec<Cluster>;
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
        // Density-cluster each contiguous unused run. Gaps left by the large-POI
        // pass must not merge non-adjacent stays into one visit.
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
    if start < end && end <= pings.len() {
        clusters.extend(density.cluster(&pings[start..end], &[]));
    }
}

fn open_or_keep_run(run_start: &mut Option<usize>, idx: usize) {
    if run_start.is_none() {
        *run_start = Some(idx);
    }
}

fn close_unused_run(
    run_start: &mut Option<usize>,
    density: &TimeAwareDensityClusterer,
    pings: &[GpsPing],
    end: usize,
    clusters: &mut Vec<Cluster>,
) {
    if let Some(start) = run_start.take() {
        extend_density_run(density, pings, start, end, clusters);
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
        if is_used {
            close_unused_run(&mut run_start, density, pings, idx, clusters);
        } else {
            open_or_keep_run(&mut run_start, idx);
        }
    }
    close_unused_run(&mut run_start, density, pings, pings.len(), clusters);
}

#[cfg(test)]
#[allow(clippy::cognitive_complexity)]
mod tests;
