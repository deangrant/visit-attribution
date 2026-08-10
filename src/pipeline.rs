//! End-to-end visit attribution orchestration.

use crate::clean::{DefaultPingCleaner, PingCleaner};
use crate::cluster::{Clusterer, TwoPassClusterer};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::join::{PlaceIndex, PlaceIndexFactory, QuadtreePlaceIndexFactory};
use crate::rank::{visit_from_rank, GbdtRanker, Ranker};
use crate::types::{Cluster, GpsPing, Place, Visit};

/// Outcome of attributing a trajectory: ranked visits plus join misses.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributionResult {
    /// Clusters successfully ranked to a place.
    pub visits: Vec<Visit>,
    /// Clusters with no join candidates (catalog or join-radius miss).
    pub unmatched_clusters: Vec<Cluster>,
}

/// Full pipeline: clean → cluster → join → rank.
#[derive(Debug, Clone)]
pub struct VisitAttributor<Cl, C, F, R> {
    config: Config,
    cleaner: Cl,
    clusterer: C,
    index_factory: F,
    ranker: R,
}

type DefaultAttributor =
    VisitAttributor<DefaultPingCleaner, TwoPassClusterer, QuadtreePlaceIndexFactory, GbdtRanker>;

impl DefaultAttributor {
    /// Start a builder for the default stage implementations.
    pub fn builder() -> VisitAttributorBuilder {
        VisitAttributorBuilder::default()
    }
}

impl<Cl, C, F, R> VisitAttributor<Cl, C, F, R>
where
    Cl: PingCleaner,
    C: Clusterer,
    F: PlaceIndexFactory,
    R: Ranker,
{
    /// Attribute visits for a device trajectory against a place catalog.
    ///
    /// Clusters that find no join candidates are not ranked; they are returned
    /// in [`AttributionResult::unmatched_clusters`] so callers can detect
    /// catalog gaps or join-radius misses in production.
    ///
    /// # Errors
    ///
    /// Propagates ranking errors for clusters that have candidates.
    pub fn attribute(&self, pings: &[GpsPing], places: &[Place]) -> Result<AttributionResult> {
        let cleaned = self.cleaner.clean(pings);
        let clusters = self.clusterer.cluster(&cleaned, places);
        let index = self.index_factory.create(places);
        let mut visits = Vec::new();
        let mut unmatched_clusters = Vec::new();
        for cluster in clusters {
            let candidates = index.candidates(&cluster);
            if candidates.is_empty() {
                unmatched_clusters.push(cluster);
                continue;
            }
            let (place_id, wins) = self.ranker.rank(&cluster, &candidates)?;
            visits.push(visit_from_rank(cluster, &candidates, place_id, wins));
        }
        Ok(AttributionResult {
            visits,
            unmatched_clusters,
        })
    }

    /// Borrow the pipeline configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// Fluent builder for [`VisitAttributor`] with default stages.
#[derive(Debug, Default)]
#[must_use]
pub struct VisitAttributorBuilder {
    config: Option<Config>,
    ranker: Option<GbdtRanker>,
}

impl VisitAttributorBuilder {
    /// Set validated pipeline configuration.
    pub fn config(mut self, config: Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the trained or loaded ranker.
    pub fn ranker(mut self, ranker: GbdtRanker) -> Self {
        self.ranker = Some(ranker);
        self
    }

    /// Build the attributor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when the ranker is missing or the
    /// configuration is invalid.
    pub fn build(self) -> Result<DefaultAttributor> {
        let config = self.config.unwrap_or_default();
        config.validate()?;
        let ranker = self.ranker.ok_or_else(|| Error::InvalidInput("ranker is required".into()))?;
        Ok(VisitAttributor {
            cleaner: DefaultPingCleaner::new(config.clone()),
            clusterer: TwoPassClusterer::new(config.clone()),
            index_factory: QuadtreePlaceIndexFactory::new(config.clone()),
            ranker,
            config,
        })
    }
}

/// Construct an attributor with explicit stage implementations.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when `config` fails [`Config::validate`].
pub fn with_parts<Cl, C, F, R>(
    config: Config,
    cleaner: Cl,
    clusterer: C,
    index_factory: F,
    ranker: R,
) -> Result<VisitAttributor<Cl, C, F, R>> {
    config.validate()?;
    Ok(VisitAttributor {
        config,
        cleaner,
        clusterer,
        index_factory,
        ranker,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PlaceId;

    struct IdentityCleaner;

    impl PingCleaner for IdentityCleaner {
        fn clean(&self, pings: &[GpsPing]) -> Vec<GpsPing> {
            pings.to_vec()
        }
    }

    struct FixedClusterer(Cluster);

    impl Clusterer for FixedClusterer {
        fn cluster(&self, _pings: &[GpsPing], _places: &[Place]) -> Vec<Cluster> {
            vec![self.0.clone()]
        }
    }

    struct EmptyIndex;

    impl PlaceIndex for EmptyIndex {
        fn candidates(&self, _cluster: &Cluster) -> Vec<Place> {
            Vec::new()
        }
    }

    struct EmptyIndexFactory;

    impl PlaceIndexFactory for EmptyIndexFactory {
        type Index = EmptyIndex;

        fn create(&self, _places: &[Place]) -> Self::Index {
            EmptyIndex
        }
    }

    struct PanicRanker;

    impl Ranker for PanicRanker {
        fn rank(&self, _cluster: &Cluster, _candidates: &[Place]) -> Result<(PlaceId, u32)> {
            panic!("rank should not run for unmatched clusters");
        }
    }

    #[test]
    fn surfaces_clusters_with_no_candidates() {
        let cluster = Cluster::from_pings(vec![
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.0001, 0.0, 10.0, 10.0),
        ])
        .unwrap();
        let attributor = with_parts(
            Config::default(),
            IdentityCleaner,
            FixedClusterer(cluster.clone()),
            EmptyIndexFactory,
            PanicRanker,
        )
        .unwrap();
        let result = attributor.attribute(&cluster.pings, &[]).unwrap();
        assert!(result.visits.is_empty());
        assert_eq!(result.unmatched_clusters, vec![cluster]);
    }

    #[test]
    fn with_parts_rejects_invalid_config() {
        let config = Config {
            max_time_gap_s: 0.0,
            ..Config::default()
        };
        let result = with_parts(
            config,
            IdentityCleaner,
            FixedClusterer(
                Cluster::from_pings(vec![
                    GpsPing::new(0.0, 0.0, 0.0, 10.0),
                    GpsPing::new(0.0, 0.0, 1.0, 10.0),
                ])
                .unwrap(),
            ),
            EmptyIndexFactory,
            PanicRanker,
        );
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }
}
