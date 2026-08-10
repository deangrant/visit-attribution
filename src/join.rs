//! Spatial join between clusters and places.
//!
//! The default index prunes candidates with a bbox quadtree, then confirms
//! matches with [`distance_to_polygon_m`]. Implement [`PlaceIndex`] for
//! specialized catalogs if needed.

use std::collections::HashSet;

use crate::config::Config;
use crate::geo::{distance_to_polygon_m, meters_to_degrees, BBox};
use crate::spatial::BBoxQuadtree;
use crate::types::{Cluster, Place, PlaceId, Point};

/// Looks up candidate places for a cluster.
///
/// # Contract
///
/// - Empty indexes return an empty candidate list.
/// - A place is a candidate when the cluster centroid or any ping is within
///   `join_radius_m + max_horizontal_accuracy` of the polygon (edge or interior).
/// - Implementations must not panic on empty clusters or empty place catalogs.
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

/// Quadtree-accelerated place index with distance-based join matching.
///
/// Place bboxes (padded by `join_radius_m`) are stored in an axis-aligned
/// quadtree. Queries collect overlapping leaves, then confirm with
/// [`distance_to_polygon_m`].
#[derive(Debug, Clone)]
pub struct QuadtreePlaceIndex {
    places: Vec<Place>,
    bboxes: Vec<BBox>,
    tree: BBoxQuadtree,
    join_radius_m: f64,
}

impl QuadtreePlaceIndex {
    /// Index a place list for joining.
    #[must_use]
    pub fn new(places: Vec<Place>, config: &Config) -> Self {
        let join_radius_m = config.join_radius_m;
        let mut bboxes = Vec::with_capacity(places.len());
        for place in &places {
            let bbox = BBox::from_ring(&place.polygon, join_radius_m).unwrap_or(BBox {
                min_lat: place.centroid.lat,
                max_lat: place.centroid.lat,
                min_lon: place.centroid.lon,
                max_lon: place.centroid.lon,
            });
            bboxes.push(bbox);
        }
        let tree = BBoxQuadtree::build(&bboxes);
        Self {
            places,
            bboxes,
            tree,
            join_radius_m,
        }
    }

    /// Borrow the indexed places.
    #[must_use]
    pub fn places(&self) -> &[Place] {
        &self.places
    }
}

impl PlaceIndex for QuadtreePlaceIndex {
    fn candidates(&self, cluster: &Cluster) -> Vec<Place> {
        if self.places.is_empty() {
            return Vec::new();
        }
        let ha = cluster.pings.iter().map(|p| p.horizontal_accuracy_m).fold(0.0_f64, f64::max);
        let radius = self.join_radius_m + ha;
        let probe = cluster_probe_ring(cluster, radius);
        let Some(query_bbox) = BBox::from_ring(&probe, 0.0) else {
            return Vec::new();
        };

        let mut hits = Vec::new();
        self.tree.query(query_bbox, &mut hits);

        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for idx in hits {
            if !seen.insert(idx) {
                continue;
            }
            let place = &self.places[idx];
            let bbox = self.bboxes[idx];
            if !bbox.intersects(query_bbox) {
                continue;
            }
            if within_join_radius(cluster, place, radius) {
                out.push(place.clone());
            }
        }
        out
    }
}

/// Factory for the default bbox-quadtree place index.
#[derive(Debug, Clone)]
pub struct QuadtreePlaceIndexFactory {
    config: Config,
}

impl QuadtreePlaceIndexFactory {
    /// Create a factory from pipeline configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

impl PlaceIndexFactory for QuadtreePlaceIndexFactory {
    type Index = QuadtreePlaceIndex;

    fn create(&self, places: &[Place]) -> Self::Index {
        QuadtreePlaceIndex::new(places.to_vec(), &self.config)
    }
}

fn within_join_radius(cluster: &Cluster, place: &Place, radius: f64) -> bool {
    distance_to_polygon_m(cluster.centroid, &place.polygon) <= radius
        || cluster
            .pings
            .iter()
            .any(|ping| distance_to_polygon_m(ping.point, &place.polygon) <= radius)
}

fn cluster_probe_ring(cluster: &Cluster, radius_m: f64) -> Vec<Point> {
    // Represent the cluster as a small ring around the centroid for bbox queries.
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
    use crate::types::GpsPing;

    fn square(id: PlaceId, lat0: f64, lon0: f64) -> Place {
        Place::new(
            id,
            vec![
                Point::new(lat0, lon0),
                Point::new(lat0, lon0 + 0.001),
                Point::new(lat0 + 0.001, lon0 + 0.001),
                Point::new(lat0 + 0.001, lon0),
                Point::new(lat0, lon0),
            ],
            Point::new(lat0 + 0.0005, lon0 + 0.0005),
            Some(445_110),
        )
    }

    /// L-shaped footprint with a concave notch in the upper-right.
    fn l_shape(id: PlaceId) -> Place {
        Place::new(
            id,
            vec![
                Point::new(0.0, 0.0),
                Point::new(0.002, 0.0),
                Point::new(0.002, 0.001),
                Point::new(0.001, 0.001),
                Point::new(0.001, 0.002),
                Point::new(0.0, 0.002),
                Point::new(0.0, 0.0),
            ],
            Point::new(0.0007, 0.0007),
            None,
        )
    }

    #[test]
    fn finds_nearby_place() {
        let cfg = Config::default();
        let place = square(7, 0.0, 0.0);
        let factory = QuadtreePlaceIndexFactory::new(cfg);
        let index = factory.create(&[place]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0005, 0.0005, 0.0, 10.0),
            GpsPing::new(0.0006, 0.0005, 30.0, 10.0),
        ])
        .unwrap();
        let cands = index.candidates(&cluster);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, 7);
    }

    #[test]
    fn empty_catalog_yields_no_candidates() {
        let index = QuadtreePlaceIndexFactory::new(Config::default()).create(&[]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 0.0, 5.0),
            GpsPing::new(0.0, 0.0, 10.0, 5.0),
        ])
        .unwrap();
        assert!(index.candidates(&cluster).is_empty());
    }

    #[test]
    fn concave_notch_farther_than_radius_is_not_a_candidate() {
        // Small radius so the open notch (~55 m from the inner corner) is outside.
        let cfg = Config::builder().join_radius_m(10.0).build().unwrap();
        let index = QuadtreePlaceIndexFactory::new(cfg).create(&[l_shape(1)]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0015, 0.0015, 0.0, 5.0),
            GpsPing::new(0.00155, 0.0015, 10.0, 5.0),
        ])
        .unwrap();
        assert!(index.candidates(&cluster).is_empty());
    }

    #[test]
    fn point_near_l_arm_within_radius_is_candidate() {
        let cfg = Config::builder().join_radius_m(50.0).build().unwrap();
        let index = QuadtreePlaceIndexFactory::new(cfg).create(&[l_shape(1)]);
        // ~5–6 m east of the horizontal bar (0.00005° ≈ 5.5 m).
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.00205, 0.0005, 0.0, 5.0),
            GpsPing::new(0.00206, 0.0005, 10.0, 5.0),
        ])
        .unwrap();
        let cands = index.candidates(&cluster);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, 1);
    }

    #[test]
    fn finds_nearby_among_many_far_places() {
        let cfg = Config::default();
        let mut places = vec![square(1, 0.0, 0.0)];
        for i in 0u32..200 {
            let offset = 1.0 + f64::from(i) * 0.01;
            places.push(square(u64::from(i) + 2, offset, offset));
        }
        let index = QuadtreePlaceIndexFactory::new(cfg).create(&places);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0005, 0.0005, 0.0, 10.0),
            GpsPing::new(0.00055, 0.0005, 20.0, 10.0),
        ])
        .unwrap();
        let cands = index.candidates(&cluster);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, 1);
    }

    #[test]
    fn dense_local_catalog_still_returns_only_nearby_match() {
        let cfg = Config::builder().join_radius_m(30.0).build().unwrap();
        let mut places = vec![square(1, 0.0, 0.0)];
        // Pack many places in a tight metro-scale neighborhood (~1 km steps).
        for i in 0u32..64 {
            let row = f64::from(i / 8);
            let col = f64::from(i % 8);
            places.push(square(
                u64::from(i) + 2,
                0.01 + row * 0.01,
                0.01 + col * 0.01,
            ));
        }
        let index = QuadtreePlaceIndexFactory::new(cfg).create(&places);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0005, 0.0005, 0.0, 5.0),
            GpsPing::new(0.00055, 0.0005, 10.0, 5.0),
        ])
        .unwrap();
        let cands = index.candidates(&cluster);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].id, 1);
    }
}
