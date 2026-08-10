//! Spatial join between clusters and places.

use crate::config::Config;
use crate::geo::{buffer_ring, distance_to_polygon_m, BBox};
use crate::types::{Cluster, Place, PlaceId};

/// Looks up candidate places for a cluster.
///
/// # Contract
///
/// - Empty indexes return an empty candidate list.
/// - Implementations may be approximate (false negatives/positives) but must
///   not panic on empty clusters or empty place catalogs.
/// - Returned places are owned clones suitable for ranking.
pub trait PlaceIndex {
    /// Return places that could explain `cluster`.
    fn candidates(&self, cluster: &Cluster) -> Vec<Place>;
}

/// Builds a [`PlaceIndex`] from a place catalog for one attribution run.
///
/// # Contract
///
/// - `create` must accept an empty `places` slice and yield a usable index.
/// - The factory owns join configuration; the catalog is supplied per call.
pub trait PlaceIndexFactory {
    /// Concrete index type produced by this factory.
    type Index: PlaceIndex;

    /// Build an index over `places`.
    fn create(&self, places: &[Place]) -> Self::Index;
}

/// Linear scan over place polygons with a buffered cluster query.
#[derive(Debug, Clone)]
pub struct BruteForcePlaceIndex {
    places: Vec<Place>,
    join_buffer_m: f64,
}

impl BruteForcePlaceIndex {
    /// Index a place list for joining.
    #[must_use]
    pub fn new(places: Vec<Place>, config: &Config) -> Self {
        Self {
            places,
            join_buffer_m: config.join_buffer_m,
        }
    }

    /// Borrow the indexed places.
    #[must_use]
    pub fn places(&self) -> &[Place] {
        &self.places
    }
}

impl PlaceIndex for BruteForcePlaceIndex {
    fn candidates(&self, cluster: &Cluster) -> Vec<Place> {
        let ha = cluster.pings.iter().map(|p| p.horizontal_accuracy_m).fold(0.0_f64, f64::max);
        let radius = self.join_buffer_m + ha;
        let probe = cluster_probe_ring(cluster, radius);
        let Some(query_bbox) = BBox::from_ring(&probe, 0.0) else {
            return Vec::new();
        };
        self.places
            .iter()
            .filter(|place| {
                let Some(pb) = BBox::from_ring(&place.polygon, radius) else {
                    return false;
                };
                if !pb.intersects(query_bbox) {
                    return false;
                }
                let buffered = buffer_ring(&place.polygon, radius);
                distance_to_polygon_m(cluster.centroid, &buffered) <= 0.0
                    || distance_to_polygon_m(cluster.centroid, &place.polygon) <= radius
                    || cluster
                        .pings
                        .iter()
                        .any(|ping| distance_to_polygon_m(ping.point, &place.polygon) <= radius)
            })
            .cloned()
            .collect()
    }
}

/// Factory for [`BruteForcePlaceIndex`].
#[derive(Debug, Clone)]
pub struct BruteForcePlaceIndexFactory {
    config: Config,
}

impl BruteForcePlaceIndexFactory {
    /// Create a factory from pipeline configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

impl PlaceIndexFactory for BruteForcePlaceIndexFactory {
    type Index = BruteForcePlaceIndex;

    fn create(&self, places: &[Place]) -> Self::Index {
        BruteForcePlaceIndex::new(places.to_vec(), &self.config)
    }
}

fn cluster_probe_ring(cluster: &Cluster, radius_m: f64) -> Vec<crate::types::Point> {
    // Represent the cluster as a small ring around the centroid for bbox queries.
    use crate::geo::meters_to_degrees;
    use crate::types::Point;
    let (dlat, dlon) = meters_to_degrees(cluster.centroid.lat, radius_m.max(1.0));
    let c = cluster.centroid;
    vec![
        Point::new(c.lat - dlat, c.lon - dlon),
        Point::new(c.lat - dlat, c.lon + dlon),
        Point::new(c.lat + dlat, c.lon + dlon),
        Point::new(c.lat + dlat, c.lon - dlon),
        Point::new(c.lat - dlat, c.lon - dlon),
    ]
}

/// Resolve places by id from a slice.
#[must_use]
pub fn places_by_ids(places: &[Place], ids: &[PlaceId]) -> Vec<Place> {
    ids.iter()
        .filter_map(|id| places.iter().find(|p| p.id == *id).cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GpsPing, Point};

    #[test]
    fn finds_nearby_place() {
        let cfg = Config::default();
        let place = Place::new(
            7,
            vec![
                Point::new(0.0, 0.0),
                Point::new(0.0, 0.001),
                Point::new(0.001, 0.001),
                Point::new(0.001, 0.0),
                Point::new(0.0, 0.0),
            ],
            Point::new(0.0005, 0.0005),
            Some(445_110),
            1_000.0,
        );
        let factory = BruteForcePlaceIndexFactory::new(cfg);
        let index = factory.create(&[place]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0005, 0.0005, 0.0, 10.0),
            GpsPing::new(0.0006, 0.0005, 30.0, 10.0),
        ]);
        let cands = index.candidates(&cluster);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, 7);
    }
}
