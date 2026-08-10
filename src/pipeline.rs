//! End-to-end visit attribution orchestration.

use crate::clean::{DefaultPingCleaner, PingCleaner};
use crate::cluster::{Clusterer, TwoPassClusterer};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::join::{BruteForcePlaceIndexFactory, PlaceIndex, PlaceIndexFactory};
use crate::rank::{visit_from_rank, GbdtRanker, Ranker};
use crate::types::{GpsPing, Place, Visit};

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
    VisitAttributor<DefaultPingCleaner, TwoPassClusterer, BruteForcePlaceIndexFactory, GbdtRanker>;

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
    /// # Errors
    ///
    /// Propagates ranking errors for clusters that have candidates.
    pub fn attribute(&self, pings: &[GpsPing], places: &[Place]) -> Result<Vec<Visit>> {
        let cleaned = self.cleaner.clean(pings);
        let clusters = self.clusterer.cluster(&cleaned, places);
        let index = self.index_factory.create(places);
        let mut visits = Vec::new();
        for cluster in clusters {
            let candidates = index.candidates(&cluster);
            if candidates.is_empty() {
                continue;
            }
            let (place_id, wins) = self.ranker.rank(&cluster, &candidates)?;
            visits.push(visit_from_rank(cluster, &candidates, place_id, wins));
        }
        Ok(visits)
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
            index_factory: BruteForcePlaceIndexFactory::new(config.clone()),
            ranker,
            config,
        })
    }
}

/// Construct an attributor with explicit stage implementations.
#[must_use]
pub fn with_parts<Cl, C, F, R>(
    config: Config,
    cleaner: Cl,
    clusterer: C,
    index_factory: F,
    ranker: R,
) -> VisitAttributor<Cl, C, F, R> {
    VisitAttributor {
        config,
        cleaner,
        clusterer,
        index_factory,
        ranker,
    }
}
