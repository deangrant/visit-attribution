//! Preference ranking and tournament scorecards.

use crate::error::{Error, Result};
use crate::features::{
    absolute_features, inference_pairs, preference_pairs, FeatureSchema, LabeledExample,
};
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
        let schema = FeatureSchema::from_places(&places);
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
        let mut wins: Vec<(PlaceId, u32)> = rows.iter().map(|(id, _)| (*id, 0_u32)).collect();
        for pair in inference_pairs(&rows) {
            let score = self.model.predict_raw(&pair.diff)?;
            if score >= 0.0 {
                if let Some(entry) = wins.iter_mut().find(|(id, _)| *id == pair.left_id) {
                    entry.1 += 1;
                }
            } else if let Some(entry) = wins.iter_mut().find(|(id, _)| *id == pair.right_id) {
                entry.1 += 1;
            }
        }
        wins.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(wins[0])
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
}
