//! Lightweight WGS84 geometry helpers used by cleaning, clustering, and join.

use crate::types::Point;

/// Mean Earth radius in meters (WGS84 spherical approximation).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Great-circle distance in meters between two WGS84 points.
#[must_use]
pub fn haversine_m(a: Point, b: Point) -> f64 {
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * h.sqrt().asin()
}

/// Convert meters to degrees of latitude (constant) and longitude at `lat`.
#[must_use]
pub fn meters_to_degrees(lat: f64, meters: f64) -> (f64, f64) {
    let dlat = meters / 111_320.0;
    let cos_lat = lat.to_radians().cos().abs().max(1e-6);
    let dlon = meters / (111_320.0 * cos_lat);
    (dlat, dlon)
}

fn ensure_closed(ring: &[Point]) -> Vec<Point> {
    if ring.is_empty() {
        return Vec::new();
    }
    let mut out = ring.to_vec();
    let first = out[0];
    let last = out[out.len() - 1];
    if (first.lat - last.lat).abs() > f64::EPSILON || (first.lon - last.lon).abs() > f64::EPSILON {
        out.push(first);
    }
    out
}

/// Point-in-polygon test for an exterior ring (holes unsupported).
///
/// Boundary-inclusive: points on edges or vertices count as inside. The open
/// interior is decided by ray casting after an explicit on-edge check.
#[must_use]
pub fn point_in_polygon(point: Point, ring: &[Point]) -> bool {
    let ring = ensure_closed(ring);
    if ring.len() < 4 {
        return false;
    }
    if point_on_ring_edge(point, &ring) {
        return true;
    }
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() - 1 {
        let pi = ring[i];
        let pj = ring[j];
        if (pi.lat - pj.lat).abs() < f64::EPSILON {
            j = i;
            continue;
        }
        let intersect = ((pi.lat > point.lat) != (pj.lat > point.lat))
            && (point.lon < (pj.lon - pi.lon) * (point.lat - pi.lat) / (pj.lat - pi.lat) + pi.lon);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Absolute epsilon in degrees for on-edge collinearity / bbox padding.
const ON_EDGE_DEG_EPS: f64 = 1e-12;

fn point_on_segment(p: Point, a: Point, b: Point) -> bool {
    let min_lat = a.lat.min(b.lat) - ON_EDGE_DEG_EPS;
    let max_lat = a.lat.max(b.lat) + ON_EDGE_DEG_EPS;
    let min_lon = a.lon.min(b.lon) - ON_EDGE_DEG_EPS;
    let max_lon = a.lon.max(b.lon) + ON_EDGE_DEG_EPS;
    if p.lat < min_lat || p.lat > max_lat || p.lon < min_lon || p.lon > max_lon {
        return false;
    }
    let dlon = b.lon - a.lon;
    let dlat = b.lat - a.lat;
    let cross = dlon * (p.lat - a.lat) - dlat * (p.lon - a.lon);
    cross.abs() <= ON_EDGE_DEG_EPS * (1.0 + dlon.abs() + dlat.abs())
}

fn point_on_ring_edge(point: Point, ring: &[Point]) -> bool {
    ring.windows(2).any(|ab| point_on_segment(point, ab[0], ab[1]))
}

fn dist_point_segment_m(p: Point, a: Point, b: Point) -> f64 {
    // Project in a local equirectangular frame around `p`.
    let (mx, my) = meters_to_degrees(p.lat, 1.0);
    let px = p.lon / my;
    let py = p.lat / mx;
    let ax = a.lon / my;
    let ay = a.lat / mx;
    let bx = b.lon / my;
    let by = b.lat / mx;
    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;
    let ab2 = abx * abx + aby * aby;
    let t = if ab2 < 1e-18 {
        0.0
    } else {
        ((apx * abx + apy * aby) / ab2).clamp(0.0, 1.0)
    };
    let cx = ax + t * abx;
    let cy = ay + t * aby;
    let dx = px - cx;
    let dy = py - cy;
    (dx * dx + dy * dy).sqrt()
}

/// Distance in meters from a point to the nearest location on a polygon.
///
/// Returns `0.0` when the point lies inside or on the boundary of the ring.
#[must_use]
pub fn distance_to_polygon_m(point: Point, ring: &[Point]) -> f64 {
    if point_in_polygon(point, ring) {
        return 0.0;
    }
    let ring = ensure_closed(ring);
    if ring.len() < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for i in 0..ring.len() - 1 {
        best = best.min(dist_point_segment_m(point, ring[i], ring[i + 1]));
    }
    best
}

/// Approximate geodesic area of an exterior ring in square meters.
#[must_use]
pub fn ring_area_m2(ring: &[Point]) -> f64 {
    let ring = ensure_closed(ring);
    if ring.len() < 4 {
        return 0.0;
    }
    // Spherical excess approximation via equirectangular shoe-lace at mean lat.
    let mean_lat = ring.iter().map(|p| p.lat).sum::<f64>() / (ring.len() as f64);
    let (mx, my) = meters_to_degrees(mean_lat, 1.0);
    let mut area = 0.0;
    for i in 0..ring.len() - 1 {
        let x1 = ring[i].lon / my;
        let y1 = ring[i].lat / mx;
        let x2 = ring[i + 1].lon / my;
        let y2 = ring[i + 1].lat / mx;
        area += x1 * y2 - x2 * y1;
    }
    area.abs() * 0.5
}

/// Axis-aligned bounding box in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl BBox {
    /// Bounding box of a ring, optionally expanded by `pad_m` meters.
    #[must_use]
    pub fn from_ring(ring: &[Point], pad_m: f64) -> Option<Self> {
        if ring.is_empty() {
            return None;
        }
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        for p in ring {
            min_lat = min_lat.min(p.lat);
            max_lat = max_lat.max(p.lat);
            min_lon = min_lon.min(p.lon);
            max_lon = max_lon.max(p.lon);
        }
        let mid_lat = (min_lat + max_lat) * 0.5;
        let (dlat, dlon) = meters_to_degrees(mid_lat, pad_m);
        Some(Self {
            min_lat: min_lat - dlat,
            max_lat: max_lat + dlat,
            min_lon: min_lon - dlon,
            max_lon: max_lon + dlon,
        })
    }

    /// Whether two boxes overlap (inclusive).
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        self.min_lat <= other.max_lat
            && self.max_lat >= other.min_lat
            && self.min_lon <= other.max_lon
            && self.max_lon >= other.min_lon
    }

    /// Axis-aligned union of two boxes.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min_lat: self.min_lat.min(other.min_lat),
            max_lat: self.max_lat.max(other.max_lat),
            min_lon: self.min_lon.min(other.min_lon),
            max_lon: self.max_lon.max(other.max_lon),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.001),
            Point::new(0.001, 0.001),
            Point::new(0.001, 0.0),
            Point::new(0.0, 0.0),
        ]
    }

    #[test]
    fn haversine_zero_for_same_point() {
        let p = Point::new(51.5, -0.12);
        assert!(haversine_m(p, p) < 1e-6);
    }

    #[test]
    fn point_inside_square() {
        let ring = unit_square();
        assert!(point_in_polygon(Point::new(0.0005, 0.0005), &ring));
        assert!(!point_in_polygon(Point::new(0.002, 0.002), &ring));
    }

    #[test]
    fn point_on_boundary_counts_as_inside() {
        let ring = unit_square();
        // Mid-edges (Point is lat, lon).
        assert!(point_in_polygon(Point::new(0.0005, 0.0), &ring)); // west
        assert!(point_in_polygon(Point::new(0.0005, 0.001), &ring)); // east
        assert!(point_in_polygon(Point::new(0.0, 0.0005), &ring)); // south
        assert!(point_in_polygon(Point::new(0.001, 0.0005), &ring)); // north
                                                                     // Vertices.
        assert!(point_in_polygon(Point::new(0.0, 0.0), &ring));
        assert!(point_in_polygon(Point::new(0.001, 0.001), &ring));
        assert!((distance_to_polygon_m(Point::new(0.0005, 0.0), &ring)).abs() < 1e-9);
    }

    #[test]
    fn distance_inside_is_zero() {
        let ring = unit_square();
        assert!((distance_to_polygon_m(Point::new(0.0005, 0.0005), &ring)).abs() < 1e-9);
    }

    #[test]
    fn ring_area_positive() {
        assert!(ring_area_m2(&unit_square()) > 0.0);
    }
}
