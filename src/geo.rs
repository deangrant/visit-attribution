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

/// Ray-casting point-in-polygon test for an exterior ring (holes unsupported).
#[must_use]
pub fn point_in_polygon(point: Point, ring: &[Point]) -> bool {
    let ring = ensure_closed(ring);
    if ring.len() < 4 {
        return false;
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
/// Returns `0.0` when the point lies inside the ring.
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

/// Expand an exterior ring outward by approximately `buffer_m` meters.
///
/// Each vertex is pushed away from the ring centroid in a local equirectangular
/// frame. This is a coarse visual/heuristic pad only—not a cadastral buffer and
/// not used by the place join (join uses [`distance_to_polygon_m`] instead).
/// Concave rings can self-intersect or fill notches incorrectly.
#[must_use]
pub fn buffer_ring(ring: &[Point], buffer_m: f64) -> Vec<Point> {
    if buffer_m <= 0.0 || ring.len() < 3 {
        return ensure_closed(ring);
    }
    let closed = ensure_closed(ring);
    let n = closed.len() - 1;
    if n < 3 {
        return closed;
    }
    let mean_lat = closed[..n].iter().map(|p| p.lat).sum::<f64>() / n as f64;
    let mean_lon = closed[..n].iter().map(|p| p.lon).sum::<f64>() / n as f64;
    let (mx, my) = meters_to_degrees(mean_lat, 1.0);
    let mut out = Vec::with_capacity(n + 1);
    for p in closed.iter().take(n) {
        let dx = (p.lon - mean_lon) / my;
        let dy = (p.lat - mean_lat) / mx;
        let len = (dx * dx + dy * dy).sqrt();
        let (ux, uy) = if len < 1e-12 {
            (1.0, 0.0)
        } else {
            (dx / len, dy / len)
        };
        out.push(Point::new(
            p.lat + uy * buffer_m * mx,
            p.lon + ux * buffer_m * my,
        ));
    }
    if let Some(first) = out.first().copied() {
        out.push(first);
    }
    out
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

    /// Whether the point lies inside the box.
    #[must_use]
    pub fn contains_point(self, p: Point) -> bool {
        p.lat >= self.min_lat
            && p.lat <= self.max_lat
            && p.lon >= self.min_lon
            && p.lon <= self.max_lon
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
    fn distance_inside_is_zero() {
        let ring = unit_square();
        assert!((distance_to_polygon_m(Point::new(0.0005, 0.0005), &ring)).abs() < 1e-9);
    }

    #[test]
    fn ring_area_positive() {
        assert!(ring_area_m2(&unit_square()) > 0.0);
    }

    #[test]
    fn buffer_increases_area() {
        let ring = unit_square();
        let buffered = buffer_ring(&ring, 20.0);
        assert!(ring_area_m2(&buffered) > ring_area_m2(&ring));
    }
}
