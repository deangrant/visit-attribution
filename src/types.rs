//! Core GPS, place, cluster, and visit value types.

use crate::error::{Error, Result};

/// Geographic coordinate in WGS84 degrees (latitude, longitude).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Latitude in degrees.
    pub lat: f64,
    /// Longitude in degrees.
    pub lon: f64,
}

impl Point {
    /// Create a point from latitude and longitude in degrees.
    #[must_use]
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
}

impl From<[f64; 2]> for Point {
    fn from(value: [f64; 2]) -> Self {
        let [lat, lon] = value;
        Self::new(lat, lon)
    }
}

impl From<(f64, f64)> for Point {
    fn from(value: (f64, f64)) -> Self {
        Self::new(value.0, value.1)
    }
}

/// A single GPS observation from a device trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsPing {
    /// Observed location.
    pub point: Point,
    /// Timestamp in seconds (Unix or any monotonic epoch).
    pub time_s: f64,
    /// Reported horizontal accuracy in meters.
    pub horizontal_accuracy_m: f64,
}

impl GpsPing {
    /// Create a ping from lat, lon, time (seconds), and horizontal accuracy.
    #[must_use]
    pub fn new(lat: f64, lon: f64, time_s: f64, horizontal_accuracy_m: f64) -> Self {
        Self {
            point: Point::new(lat, lon),
            time_s,
            horizontal_accuracy_m,
        }
    }
}

/// Stable identifier for a place of interest.
pub type PlaceId = u64;

/// Point of interest with an exterior polygon ring.
///
/// Large-POI clustering derives footprint area from [`polygon`](Self::polygon)
/// at cluster time (not a separate catalog area field).
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    /// Caller-defined place identifier.
    pub id: PlaceId,
    /// Closed or open exterior ring in WGS84 degrees.
    pub polygon: Vec<Point>,
    /// Representative centroid (caller-supplied or geometric).
    pub centroid: Point,
    /// Optional NAICS code; first four digits drive time-of-day features.
    pub naics: Option<u32>,
}

impl Place {
    /// Build a place from id, polygon, centroid, and optional NAICS.
    #[must_use]
    pub fn new(id: PlaceId, polygon: Vec<Point>, centroid: Point, naics: Option<u32>) -> Self {
        Self {
            id,
            polygon,
            centroid,
            naics,
        }
    }

    /// Axis-aligned square; SW corner `(lat, lon)`, side `side_deg`.
    ///
    /// Centroid is placed at the half-side offset from the SW corner.
    #[must_use]
    pub fn square(id: PlaceId, lat: f64, lon: f64, side_deg: f64, naics: Option<u32>) -> Self {
        let half = side_deg * 0.5;
        Self::new(
            id,
            vec![
                Point::new(lat, lon),
                Point::new(lat, lon + side_deg),
                Point::new(lat + side_deg, lon + side_deg),
                Point::new(lat + side_deg, lon),
                Point::new(lat, lon),
            ],
            Point::new(lat + half, lon + half),
            naics,
        )
    }

    /// Four-digit NAICS prefix, if a NAICS code is present.
    ///
    /// Takes the most-significant four digits for any positive code length
    /// (6-digit → 4-digit industry group; already-4-digit unchanged).
    #[must_use]
    pub fn naics4(&self) -> Option<u32> {
        self.naics.map(|mut n| {
            while n >= 10_000 {
                n /= 10;
            }
            n
        })
    }
}

/// A density cluster of GPS pings treated as one potential visit.
#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    /// Pings assigned to this cluster (time-ordered).
    pub pings: Vec<GpsPing>,
    /// Mean lat/lon of member pings.
    pub centroid: Point,
    /// Start time in seconds (min ping time).
    pub start_time_s: f64,
    /// End time in seconds (max ping time).
    pub end_time_s: f64,
}

impl Cluster {
    /// Build a cluster from one or more pings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when `pings` is empty.
    pub fn from_pings(pings: Vec<GpsPing>) -> Result<Self> {
        if pings.is_empty() {
            return Err(Error::InvalidInput(
                "cluster requires at least one ping".into(),
            ));
        }
        let n = pings.len() as f64;
        let lat = pings.iter().map(|p| p.point.lat).sum::<f64>() / n;
        let lon = pings.iter().map(|p| p.point.lon).sum::<f64>() / n;
        let start_time_s = pings.iter().map(|p| p.time_s).fold(f64::INFINITY, f64::min);
        let end_time_s = pings.iter().map(|p| p.time_s).fold(f64::NEG_INFINITY, f64::max);
        Ok(Self {
            pings,
            centroid: Point::new(lat, lon),
            start_time_s,
            end_time_s,
        })
    }

    /// Duration of the cluster in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        (self.end_time_s - self.start_time_s).max(0.0)
    }

    /// Hour of day (0–23) from `start_time_s` as Unix-like seconds.
    ///
    /// Uses seconds-of-day via `rem_euclid` (negative times wrap). Non-finite
    /// values (`NaN`, ±∞) return `0`.
    #[must_use]
    pub fn hour_of_day(&self) -> u8 {
        if !self.start_time_s.is_finite() {
            return 0;
        }
        let hour = (self.start_time_s.rem_euclid(86_400.0) / 3600.0).floor();
        // rem_euclid keeps seconds in [0, 86400); hour is in [0, 23].
        hour as u8
    }
}

/// A cluster attributed to a single place.
#[derive(Debug, Clone, PartialEq)]
pub struct Visit {
    /// Source cluster.
    pub cluster: Cluster,
    /// Chosen place id.
    pub place_id: PlaceId,
    /// Tournament wins for the chosen place (among candidates).
    pub wins: u32,
    /// Candidate place ids considered for this cluster.
    pub candidates: Vec<PlaceId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naics4_takes_most_significant_four_digits() {
        let six = Place::new(1, vec![], Point::new(0.0, 0.0), Some(445_110));
        let five = Place::new(2, vec![], Point::new(0.0, 0.0), Some(44_511));
        let four = Place::new(3, vec![], Point::new(0.0, 0.0), Some(4_451));
        assert_eq!(six.naics4(), Some(4_451));
        assert_eq!(five.naics4(), Some(4_451));
        assert_eq!(four.naics4(), Some(4_451));
        assert_eq!(
            Place::new(4, vec![], Point::new(0.0, 0.0), None).naics4(),
            None
        );
    }

    #[test]
    fn from_pings_rejects_empty() {
        let err = Cluster::from_pings(vec![]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn from_pings_uses_single_ping_as_centroid() {
        let ping = GpsPing::new(1.5, -2.5, 10.0, 5.0);
        let cluster = Cluster::from_pings(vec![ping]).unwrap();
        assert_eq!(cluster.centroid, Point::new(1.5, -2.5));
        assert!((cluster.start_time_s - 10.0).abs() < f64::EPSILON);
        assert!((cluster.end_time_s - 10.0).abs() < f64::EPSILON);
    }

    fn cluster_at(time_s: f64) -> Cluster {
        Cluster::from_pings(vec![GpsPing::new(0.0, 0.0, time_s, 5.0)]).unwrap()
    }

    #[test]
    fn hour_of_day_boundaries_and_wrap() {
        assert_eq!(cluster_at(0.0).hour_of_day(), 0);
        assert_eq!(cluster_at(3_600.0).hour_of_day(), 1);
        assert_eq!(cluster_at(86_399.0).hour_of_day(), 23);
        assert_eq!(cluster_at(86_400.0).hour_of_day(), 0);
        assert_eq!(cluster_at(3_600.5).hour_of_day(), 1);
        assert_eq!(cluster_at(-1.0).hour_of_day(), 23);
    }

    #[test]
    fn hour_of_day_non_finite_is_zero() {
        assert_eq!(cluster_at(f64::NAN).hour_of_day(), 0);
        assert_eq!(cluster_at(f64::INFINITY).hour_of_day(), 0);
        assert_eq!(cluster_at(f64::NEG_INFINITY).hour_of_day(), 0);
    }
}
