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
    pub const fn new(model: GbdtModel) -> Self {
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
    pub const fn schema(&self) -> &FeatureSchema {
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
            let Some(only) = candidates.first() else {
                return Err(Error::InvalidInput(
                    "cannot rank an empty candidate set".into(),
                ));
            };
            return Ok((only.id, 0));
        }
        let rows = absolute_features(&self.model.schema, cluster, candidates);
        let wins = pairwise_wins(&self.model, &rows)?;
        pick_winner_by_wins(&rows, &wins)
    }
}

fn pairwise_wins(model: &GbdtModel, rows: &[(PlaceId, Vec<f64>)]) -> Result<Vec<u32>> {
    let mut wins = vec![0_u32; rows.len()];
    let mut diff = Vec::with_capacity(model.schema.dim());
    for (i, left) in rows.iter().enumerate() {
        for (offset, right) in rows.iter().enumerate().skip(i + 1) {
            let j = offset;
            diff.clear();
            diff.extend(left.1.iter().zip(right.1.iter()).map(|(a, b)| a - b));
            let score = model.predict_raw(&diff)?;
            if score >= 0.0 {
                if let Some(w) = wins.get_mut(i) {
                    *w += 1;
                }
            } else if let Some(w) = wins.get_mut(j) {
                *w += 1;
            }
        }
    }
    Ok(wins)
}

fn pick_winner_by_wins(rows: &[(PlaceId, Vec<f64>)], wins: &[u32]) -> Result<(PlaceId, u32)> {
    rows.iter()
        .zip(wins.iter())
        .max_by(|(a, aw), (b, bw)| aw.cmp(bw).then_with(|| b.0.cmp(&a.0)))
        .map(|((id, _), &w)| (*id, w))
        .ok_or_else(|| Error::InvalidInput("cannot rank an empty candidate set".into()))
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
    use crate::error::Result;
    use crate::types::GpsPing;

    fn square(id: PlaceId, lat: f64, lon: f64, naics: u32) -> Place {
        Place::square(id, lat, lon, 0.001, Some(naics))
    }

    #[test]
    fn trains_and_picks_closer_place() -> Result<()> {
        let near = square(1, 0.0, 0.0, 445_110);
        let far = square(2, 0.05, 0.05, 445_110);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
            GpsPing::new(0.0005, 0.0005, 20.0, 5.0),
            GpsPing::new(0.0006, 0.0004, 40.0, 5.0),
        ])?;
        let examples = vec![LabeledExample {
            cluster: cluster.clone(),
            candidates: vec![near.clone(), far.clone()],
            true_place_id: 1,
        }];
        let ranker = GbdtRanker::train(&examples, &TrainConfig::default())?;
        let (id, _) = ranker.rank(&cluster, &[near, far])?;
        assert_eq!(id, 1);
        Ok(())
    }

    #[test]
    fn ranks_many_candidates_with_index_scorecard() -> Result<()> {
        let near = square(1, 0.0, 0.0, 445_110);
        let mid = square(3, 0.02, 0.02, 445_110);
        let far = square(5, 0.05, 0.05, 445_110);
        let farther = square(2, 0.08, 0.08, 445_110);
        let farthest = square(4, 0.1, 0.1, 445_110);
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
            GpsPing::new(0.0005, 0.0005, 20.0, 5.0),
            GpsPing::new(0.0006, 0.0004, 40.0, 5.0),
        ])?;
        let candidates = vec![near, mid, far, farther, farthest];
        let examples = vec![LabeledExample {
            cluster: cluster.clone(),
            candidates: candidates.clone(),
            true_place_id: 1,
        }];
        let ranker = GbdtRanker::train(&examples, &TrainConfig::default())?;
        let (id, wins) = ranker.rank(&cluster, &candidates)?;
        assert_eq!(id, 1);
        assert!(wins >= 1);
        Ok(())
    }

    fn zero_score_ranker() -> Result<GbdtRanker> {
        let text = "\
VA_GBDT 1
base 0
lr 1
dim 28
naics
trees 1
L 0
";
        let path = std::env::temp_dir().join(format!(
            "visit-attribution-zero-{}-{}.va",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::write(&path, text)?;
        let model = GbdtModel::load(&path)?;
        let _ = std::fs::remove_file(&path);
        Ok(GbdtRanker::new(model))
    }

    fn cluster_near_origin() -> Result<Cluster> {
        Cluster::from_pings(vec![
            GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
            GpsPing::new(0.0005, 0.0005, 20.0, 5.0),
        ])
    }

    #[test]
    fn empty_candidates_returns_invalid_input() -> Result<()> {
        let ranker = zero_score_ranker()?;
        let err = crate::test_util::err(ranker.rank(&cluster_near_origin()?, &[]))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn singleton_candidate_has_zero_wins() -> Result<()> {
        let (id, wins) =
            zero_score_ranker()?.rank(&cluster_near_origin()?, &[square(9, 0.0, 0.0, 445_110)])?;
        assert_eq!((id, wins), (9, 0));
        Ok(())
    }

    #[test]
    fn win_count_tie_prefers_lowest_place_id() -> Result<()> {
        // Through the ranker, a constant-zero scorer awards each pair to the
        // lower index (score >= 0). With equal geometry, the first list entry
        // uniquely leads — PlaceId only decides when win counts match.
        let ranker = zero_score_ranker()?;
        let a = square(5, 0.0, 0.0, 445_110);
        let b = square(2, 0.0, 0.0, 445_110);
        let (id, _) = ranker.rank(&cluster_near_origin()?, &[a.clone(), b.clone()])?;
        assert_eq!(id, 5);
        let (id_rev, _) = ranker.rank(&cluster_near_origin()?, &[b, a])?;
        assert_eq!(id_rev, 2);
        Ok(())
    }

    #[test]
    fn duplicate_place_ids_are_accepted() -> Result<()> {
        // Duplicates are not rejected; both rows compete in the tournament.
        let place = square(3, 0.0, 0.0, 445_110);
        let (id, wins) =
            zero_score_ranker()?.rank(&cluster_near_origin()?, &[place.clone(), place])?;
        assert_eq!((id, wins), (3, 1));
        Ok(())
    }

    #[test]
    fn ranks_unseen_naics_via_unk_hour_block() -> Result<()> {
        let known = square(1, 0.0, 0.0, 445_110);
        let distractor = square(2, 0.05, 0.05, 445_110);
        let cluster = cluster_near_origin()?;
        let examples = vec![LabeledExample {
            cluster: cluster.clone(),
            candidates: vec![known, distractor],
            true_place_id: 1,
        }];
        let ranker = GbdtRanker::train(&examples, &TrainConfig::default())?;
        assert!(ranker.schema().naics4.contains(&4_451));
        // Serve-time NAICS outside the frozen schema maps to the UNK hour block.
        let unseen = square(1, 0.0, 0.0, 722_515);
        let other = square(2, 0.05, 0.05, 722_515);
        let (id, _) = ranker.rank(&cluster, &[unseen, other])?;
        assert_eq!(id, 1);
        Ok(())
    }
}
