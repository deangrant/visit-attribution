//! Preference ranking and tournament scorecards.

use crate::error::{Error, Result};
use crate::features::{absolute_features, preference_pairs, FeatureSchema, LabeledExample};
use crate::gbdt::{GbdtModel, TrainConfig};
use crate::types::{Cluster, Place, PlaceId, Visit};

/// Chooses a place among candidates for a cluster.
///
/// # Contract
///
/// - Empty `candidates` returns [`Error::InvalidInput`].
/// - A singleton candidate list returns that place with `wins = 0`.
/// - Ties must resolve deterministically; [`GbdtRanker`] prefers the highest
///   win count, then the lowest [`PlaceId`].
pub trait Ranker {
    /// Rank candidates and return the winning place id with win count.
    ///
    /// # Errors
    ///
    /// Returns an error when ranking cannot proceed (e.g. empty candidates).
    fn rank(&self, cluster: &Cluster, candidates: &[Place]) -> Result<(PlaceId, u32)>;
}

/// Learning-to-rank model using pairwise GBDT scores and a win scorecard.
#[derive(Debug, Clone)]
pub struct GbdtRanker {
    model: GbdtModel,
}

impl GbdtRanker {
    /// Wrap a trained model.
    #[must_use]
    pub fn new(model: GbdtModel) -> Self {
        Self { model }
    }

    /// Train from labeled cluster/place examples.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when no preference pairs can be built.
    pub fn train(examples: &[LabeledExample], config: &TrainConfig) -> Result<Self> {
        if examples.is_empty() {
            return Err(Error::InvalidInput(
                "need at least one labeled example".into(),
            ));
        }
        let mut places = Vec::new();
        for ex in examples {
            places.extend(ex.candidates.iter().cloned());
        }
        let schema = FeatureSchema::from_places(&places, config.max_naics4);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for ex in examples {
            let rows = absolute_features(&schema, &ex.cluster, &ex.candidates);
            for pair in preference_pairs(&rows, ex.true_place_id) {
                xs.push(pair.diff);
                ys.push(pair.label);
            }
        }
        if xs.is_empty() {
            return Err(Error::InvalidInput(
                "no preference pairs; each example needs the true place among >=2 candidates"
                    .into(),
            ));
        }
        let model = GbdtModel::train(schema, &xs, &ys, config)?;
        Ok(Self { model })
    }

    /// Load a persisted model from disk.
    ///
    /// # Errors
    ///
    /// Propagates model I/O and parse errors.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            model: GbdtModel::load(path)?,
        })
    }

    /// Persist the underlying model.
    ///
    /// # Errors
    ///
    /// Propagates filesystem errors.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.model.save(path)
    }

    /// Borrow the feature schema.
    #[must_use]
    pub fn schema(&self) -> &FeatureSchema {
        &self.model.schema
    }
}

impl Ranker for GbdtRanker {
    fn rank(&self, cluster: &Cluster, candidates: &[Place]) -> Result<(PlaceId, u32)> {
        if candidates.is_empty() {
            return Err(Error::InvalidInput(
                "cannot rank an empty candidate set".into(),
            ));
        }
        if candidates.len() == 1 {
            return Ok((candidates[0].id, 0));
        }
        let rows = absolute_features(&self.model.schema, cluster, candidates);
        let mut wins = vec![0_u32; rows.len()];
        let mut diff = Vec::with_capacity(self.model.schema.dim());
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                diff.clear();
                diff.extend(
                    rows[i]
                        .1
                        .iter()
                        .zip(rows[j].1.iter())
                        .map(|(a, b)| a - b),
                );
                let score = self.model.predict_raw(&diff)?;
                if score >= 0.0 {
                    wins[i] += 1;
                } else {
                    wins[j] += 1;
                }
            }
        }
        let best = (0..rows.len())
            .max_by(|&a, &b| {
                wins[a]
                    .cmp(&wins[b])
                    .then_with(|| rows[b].0.cmp(&rows[a].0))
            })
            .expect("rows non-empty");
        Ok((rows[best].0, wins[best]))
    }
}

/// Attach a ranked place to a cluster, producing a [`Visit`].
pub fn visit_from_rank(
    cluster: Cluster,
    candidates: &[Place],
    place_id: PlaceId,
    wins: u32,
) -> Visit {
    Visit {
        cluster,
        place_id,
        wins,
        candidates: candidates.iter().map(|p| p.id).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GpsPing, Point};

    fn square(id: PlaceId, lat: f64, lon: f64, naics: u32) -> Place {
        Place::new(
            id,
            vec![
                Point::new(lat, lon),
                Point::new(lat, lon + 0.001),
                Point::new(lat + 0.001, lon + 0.001),
                Point::new(lat + 0.001, lon),
                Point::new(lat, lon),
            ],
            Point::new(lat + 0.0005, lon + 0.0005),
            Some(naics),
            500.0,
        )
    }

    #[test]
    fn trains_and_picks_closer_place() {
        let near = square(1, 0.0, 0.0, 445_110);
        let far = square(2, 0.05, 0.05, 445_110);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
            GpsPing::new(0.0005, 0.0005, 20.0, 5.0),
            GpsPing::new(0.0006, 0.0004, 40.0, 5.0),
        ]);
        let examples = vec![LabeledExample {
            cluster: cluster.clone(),
            candidates: vec![near.clone(), far.clone()],
            true_place_id: 1,
        }];
        let ranker = GbdtRanker::train(&examples, &TrainConfig::default()).unwrap();
        let (id, _) = ranker.rank(&cluster, &[near, far]).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn ranks_many_candidates_with_index_scorecard() {
        let near = square(1, 0.0, 0.0, 445_110);
        let mid = square(3, 0.02, 0.02, 445_110);
        let far = square(5, 0.05, 0.05, 445_110);
        let farther = square(2, 0.08, 0.08, 445_110);
        let farthest = square(4, 0.1, 0.1, 445_110);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
            GpsPing::new(0.0005, 0.0005, 20.0, 5.0),
            GpsPing::new(0.0006, 0.0004, 40.0, 5.0),
        ]);
        let candidates = vec![
            near.clone(),
            mid.clone(),
            far.clone(),
            farther.clone(),
            farthest.clone(),
        ];
        let examples = vec![LabeledExample {
            cluster: cluster.clone(),
            candidates: candidates.clone(),
            true_place_id: 1,
        }];
        let ranker = GbdtRanker::train(&examples, &TrainConfig::default()).unwrap();
        let (id, wins) = ranker.rank(&cluster, &candidates).unwrap();
        assert_eq!(id, 1);
        assert!(wins >= 1);
    }
}
