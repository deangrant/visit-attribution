//! Feature vectors for preference learning over place candidates.

use std::collections::HashMap;

use crate::geo::{distance_to_polygon_m, haversine_m};
use crate::types::{Cluster, Place, PlaceId};

/// Absolute feature layout shared by training and inference.
///
/// The schema is frozen at train time: known NAICS4 prefixes are listed in
/// [`FeatureSchema::naics4`], and `dim` always reserves one extra hour block for
/// unknown/missing NAICS (serve-time codes not in the list map there). Cap
/// growth with `max_naics4` when building via [`FeatureSchema::from_places`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSchema {
    /// Sorted unique known 4-digit NAICS prefixes (UNK bucket is implicit).
    pub naics4: Vec<u32>,
}

impl FeatureSchema {
    /// Build a schema from place NAICS, keeping up to `max_naics4` frequent codes.
    ///
    /// Frequency ties break by ascending code. Codes beyond the cap, and places
    /// with missing NAICS, use the implicit UNK hour block at feature time.
    /// `max_naics4 == 0` yields an UNK-only schema.
    #[must_use]
    pub fn from_places(places: &[Place], max_naics4: usize) -> Self {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for place in places {
            if let Some(code) = place.naics4() {
                *counts.entry(code).or_insert(0) += 1;
            }
        }
        let mut ranked: Vec<(u32, usize)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let naics4: Vec<u32> = ranked.into_iter().take(max_naics4).map(|(code, _)| code).collect();
        // Keep known codes sorted for stable layout / model text.
        let mut naics4 = naics4;
        naics4.sort_unstable();
        Self { naics4 }
    }

    /// Explicit schema from caller-supplied known NAICS4 prefixes.
    ///
    /// The UNK hour block is always implied by [`Self::dim`] and is not listed.
    #[must_use]
    pub fn new(mut naics4: Vec<u32>) -> Self {
        naics4.sort_unstable();
        naics4.dedup();
        Self { naics4 }
    }

    /// Number of dense features per (cluster, place) row.
    #[must_use]
    pub fn dim(&self) -> usize {
        // Four distance features + known NAICS4×hour + one UNK×hour block.
        4 + (self.naics4.len() + 1) * 24
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
    let in_schema = naics4.map(|n| schema.naics4.binary_search(&n).is_ok()).unwrap_or(false);
    for code in &schema.naics4 {
        for h in 0..24 {
            let on = matches!(naics4, Some(n) if n == *code) && h == hour;
            row.push(if on { 1.0 } else { 0.0 });
        }
    }
    for h in 0..24 {
        row.push(if !in_schema && h == hour { 1.0 } else { 0.0 });
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

/// Expand absolute rows into unordered preference pairs for a known true place.
///
/// Emits one row per unordered candidate pair that involves the true place,
/// matching [`inference_pairs`] orientation (`i < j`, `diff = left − right`).
#[must_use]
pub fn preference_pairs(
    rows: &[(PlaceId, Vec<f64>)],
    true_place_id: PlaceId,
) -> Vec<PreferencePair> {
    let mut pairs = Vec::new();
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let a_true = rows[i].0 == true_place_id;
            let b_true = rows[j].0 == true_place_id;
            let label = if a_true && !b_true {
                1.0
            } else if b_true && !a_true {
                -1.0
            } else {
                continue;
            };
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

    fn bare_place(id: PlaceId, naics: Option<u32>) -> Place {
        Place::new(
            id,
            vec![
                Point::new(0.0, 0.0),
                Point::new(0.0, 0.001),
                Point::new(0.001, 0.001),
                Point::new(0.001, 0.0),
            ],
            Point::new(0.0005, 0.0005),
            naics,
        )
    }

    #[test]
    fn preference_labels_only_involve_true_place() {
        let schema = FeatureSchema::new(vec![4451]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 0.0, 5.0),
            GpsPing::new(0.0, 0.0, 10.0, 5.0),
        ])
        .unwrap();
        let a = bare_place(1, Some(445_110));
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
        );
        let rows = absolute_features(&schema, &cluster, &[a, b]);
        let pairs = preference_pairs(&rows, 1);
        assert!(!pairs.is_empty());
        assert!(pairs.iter().all(|p| {
            (p.label - 1.0).abs() < f64::EPSILON || (p.label + 1.0).abs() < f64::EPSILON
        }));
        assert!(pairs.iter().all(|p| p.left_id == 1 || p.right_id == 1));
    }

    #[test]
    fn preference_pairs_emit_one_unordered_pair_per_distractor() {
        let schema = FeatureSchema::new(vec![]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 0.0, 5.0),
            GpsPing::new(0.0, 0.0, 10.0, 5.0),
        ])
        .unwrap();
        let true_place = bare_place(1, None);
        let a = Place::new(
            2,
            vec![
                Point::new(0.01, 0.01),
                Point::new(0.01, 0.011),
                Point::new(0.011, 0.011),
                Point::new(0.011, 0.01),
            ],
            Point::new(0.0105, 0.0105),
            None,
        );
        let b = Place::new(
            3,
            vec![
                Point::new(0.02, 0.02),
                Point::new(0.02, 0.021),
                Point::new(0.021, 0.021),
                Point::new(0.021, 0.02),
            ],
            Point::new(0.0205, 0.0205),
            None,
        );
        let rows = absolute_features(&schema, &cluster, &[true_place, a, b]);
        let pairs = preference_pairs(&rows, 1);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|p| {
            (p.left_id == 1) != (p.right_id == 1)
                && ((p.label - 1.0).abs() < f64::EPSILON || (p.label + 1.0).abs() < f64::EPSILON)
        }));
        let mut distractors: Vec<_> = pairs
            .iter()
            .map(|p| {
                if p.left_id == 1 {
                    p.right_id
                } else {
                    p.left_id
                }
            })
            .collect();
        distractors.sort_unstable();
        assert_eq!(distractors, vec![2, 3]);
    }

    #[test]
    fn dim_includes_unk_hour_block() {
        let schema = FeatureSchema::new(vec![4451]);
        assert_eq!(schema.dim(), 4 + 2 * 24);
        assert_eq!(FeatureSchema::new(vec![]).dim(), 4 + 24);
    }

    #[test]
    fn unseen_or_missing_naics_activates_unk_hour() {
        let schema = FeatureSchema::new(vec![4451]);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 3_600.0, 5.0), // hour 1 if Unix-like epoch day
            GpsPing::new(0.0, 0.0, 3_610.0, 5.0),
        ])
        .unwrap();
        let hour = usize::from(cluster.hour_of_day());
        let unseen = bare_place(1, Some(722_515));
        let missing = bare_place(2, None);
        let known = bare_place(3, Some(445_110));

        let rows = absolute_features(&schema, &cluster, &[unseen, missing, known]);
        let unk_base = 4 + 24; // after known code block
        assert!((rows[0].1[unk_base + hour] - 1.0).abs() < f64::EPSILON);
        assert!((rows[1].1[unk_base + hour] - 1.0).abs() < f64::EPSILON);
        assert!((rows[2].1[unk_base + hour]).abs() < f64::EPSILON);
        assert!((rows[2].1[4 + hour] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn from_places_keeps_top_k_by_frequency() {
        let places = vec![
            bare_place(1, Some(445_110)),
            bare_place(2, Some(445_110)),
            bare_place(3, Some(445_110)),
            bare_place(4, Some(722_515)),
        ];
        let schema = FeatureSchema::from_places(&places, 1);
        assert_eq!(schema.naics4, vec![4451]);

        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 0.0, 5.0),
            GpsPing::new(0.0, 0.0, 10.0, 5.0),
        ])
        .unwrap();
        let hour = usize::from(cluster.hour_of_day());
        let rare = bare_place(9, Some(722_515));
        let row = &absolute_features(&schema, &cluster, &[rare])[0].1;
        let unk_base = 4 + 24;
        assert!((row[unk_base + hour] - 1.0).abs() < f64::EPSILON);
    }
}
