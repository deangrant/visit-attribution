//! Feature vectors for preference learning over place candidates.

use crate::geo::{distance_to_polygon_m, haversine_m};
use crate::types::{Cluster, Place, PlaceId};

/// Absolute feature layout shared by training and inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSchema {
    /// Sorted unique 4-digit NAICS prefixes used for time-of-day one-hots.
    pub naics4: Vec<u32>,
}

impl FeatureSchema {
    /// Build a schema from the NAICS codes present on places.
    #[must_use]
    pub fn from_places(places: &[Place]) -> Self {
        let mut naics4: Vec<u32> = places.iter().filter_map(Place::naics4).collect();
        naics4.sort_unstable();
        naics4.dedup();
        Self { naics4 }
    }

    /// Explicit schema from caller-supplied NAICS4 prefixes.
    #[must_use]
    pub fn new(mut naics4: Vec<u32>) -> Self {
        naics4.sort_unstable();
        naics4.dedup();
        Self { naics4 }
    }

    /// Number of dense features per (cluster, place) row.
    #[must_use]
    pub fn dim(&self) -> usize {
        // Four distance features plus NAICS4 × hour one-hots.
        4 + self.naics4.len() * 24
    }
}

/// One labeled training row before pairwise expansion.
#[derive(Debug, Clone)]
pub struct LabeledExample {
    /// Cluster of pings.
    pub cluster: Cluster,
    /// Candidate places (must include the true place).
    pub candidates: Vec<Place>,
    /// Ground-truth place id.
    pub true_place_id: PlaceId,
}

/// Build absolute feature rows for each candidate of a cluster.
#[must_use]
pub fn absolute_features(
    schema: &FeatureSchema,
    cluster: &Cluster,
    candidates: &[Place],
) -> Vec<(PlaceId, Vec<f64>)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let centroid_dists: Vec<f64> =
        candidates.iter().map(|p| haversine_m(cluster.centroid, p.centroid)).collect();
    let wkt_dists: Vec<f64> = candidates
        .iter()
        .map(|p| distance_to_polygon_m(cluster.centroid, &p.polygon))
        .collect();
    let centroid_ranks = ranks(&centroid_dists);
    let wkt_ranks = ranks(&wkt_dists);
    let hour = cluster.hour_of_day();
    candidates
        .iter()
        .enumerate()
        .map(|(i, place)| {
            let mut row = Vec::with_capacity(schema.dim());
            row.push(centroid_dists[i]);
            row.push(wkt_dists[i]);
            row.push(centroid_ranks[i]);
            row.push(wkt_ranks[i]);
            append_naics_hour(schema, place.naics4(), hour, &mut row);
            (place.id, row)
        })
        .collect()
}

fn append_naics_hour(schema: &FeatureSchema, naics4: Option<u32>, hour: u8, row: &mut Vec<f64>) {
    let hour = usize::from(hour.min(23));
    for code in &schema.naics4 {
        for h in 0..24 {
            let on = matches!(naics4, Some(n) if n == *code) && h == hour;
            row.push(if on { 1.0 } else { 0.0 });
        }
    }
}

/// Dense ranks (1 = closest). Ties get the minimum rank.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; values.len()];
    let mut rank = 1.0_f64;
    let mut i = 0usize;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && (values[order[j]] - values[order[i]]).abs() < 1e-9 {
            j += 1;
        }
        for k in i..j {
            out[order[k]] = rank;
        }
        rank += (j - i) as f64;
        i = j;
    }
    out
}

/// Preference-learning pairs: difference vectors and ±1 labels.
#[derive(Debug, Clone)]
pub struct PreferencePair {
    /// Feature vector A minus feature vector B.
    pub diff: Vec<f64>,
    /// `1` if A is the true place, `-1` if B is the true place.
    pub label: f64,
    /// Left-hand place id (A).
    pub left_id: PlaceId,
    /// Right-hand place id (B).
    pub right_id: PlaceId,
}

/// Expand absolute rows into preference pairs for a known true place.
#[must_use]
pub fn preference_pairs(
    rows: &[(PlaceId, Vec<f64>)],
    true_place_id: PlaceId,
) -> Vec<PreferencePair> {
    let mut pairs = Vec::new();
    for i in 0..rows.len() {
        for j in 0..rows.len() {
            if i == j {
                continue;
            }
            let a_true = rows[i].0 == true_place_id;
            let b_true = rows[j].0 == true_place_id;
            let label = if a_true && !b_true {
                1.0
            } else if b_true && !a_true {
                -1.0
            } else {
                0.0
            };
            if label == 0.0 {
                continue;
            }
            let diff: Vec<f64> =
                rows[i].1.iter().zip(rows[j].1.iter()).map(|(a, b)| a - b).collect();
            pairs.push(PreferencePair {
                diff,
                label,
                left_id: rows[i].0,
                right_id: rows[j].0,
            });
        }
    }
    pairs
}

/// All unordered-style ordered pairs for inference (A vs B for A != B).
#[must_use]
pub fn inference_pairs(rows: &[(PlaceId, Vec<f64>)]) -> Vec<PreferencePair> {
    let mut pairs = Vec::new();
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let diff: Vec<f64> =
                rows[i].1.iter().zip(rows[j].1.iter()).map(|(a, b)| a - b).collect();
            pairs.push(PreferencePair {
                diff,
                label: 0.0,
                left_id: rows[i].0,
                right_id: rows[j].0,
            });
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GpsPing, Point};

    #[test]
    fn preference_labels_only_involve_true_place() {
        let schema = FeatureSchema::new(vec![4451]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 0.0, 5.0),
            GpsPing::new(0.0, 0.0, 10.0, 5.0),
        ]);
        let a = Place::new(
            1,
            vec![
                Point::new(-0.001, -0.001),
                Point::new(-0.001, 0.001),
                Point::new(0.001, 0.001),
                Point::new(0.001, -0.001),
            ],
            Point::new(0.0, 0.0),
            Some(445_110),
            100.0,
        );
        let b = Place::new(
            2,
            vec![
                Point::new(0.01, 0.01),
                Point::new(0.01, 0.011),
                Point::new(0.011, 0.011),
                Point::new(0.011, 0.01),
            ],
            Point::new(0.0105, 0.0105),
            Some(445_110),
            100.0,
        );
        let rows = absolute_features(&schema, &cluster, &[a, b]);
        let pairs = preference_pairs(&rows, 1);
        assert!(!pairs.is_empty());
        assert!(pairs.iter().all(|p| {
            (p.label - 1.0).abs() < f64::EPSILON || (p.label + 1.0).abs() < f64::EPSILON
        }));
        assert!(pairs.iter().all(|p| p.left_id == 1 || p.right_id == 1));
    }
}
