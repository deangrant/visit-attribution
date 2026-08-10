//! Visit attribution: map GPS trajectories to point-of-interest visits.
//!
//! The pipeline cleans GPS pings, clusters them into potential visits, joins
//! each cluster to nearby places, and ranks candidates with a
//! preference-learning gradient-boosted model. The crate is **std-first**: no
//! required third-party dependencies.
//!
//! The primary API is train-then-attribute ([`GbdtRanker`], [`VisitAttributor`]).
//! Stage traits and default implementations are exported for custom wiring via
//! [`with_parts`].
//!
//! # Quick start
//!
//! Train a ranker on labeled clusters, then attribute a trajectory:
//!
//! ```
//! use visit_attribution::{
//!     Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Point,
//!     TrainConfig, VisitAttributor,
//! };
//!
//! let near = Place::new(
//!     1,
//!     vec![
//!         Point::new(0.0, 0.0),
//!         Point::new(0.0, 0.001),
//!         Point::new(0.001, 0.001),
//!         Point::new(0.001, 0.0),
//!         Point::new(0.0, 0.0),
//!     ],
//!     Point::new(0.0005, 0.0005),
//!     Some(445_110),
//!     800.0,
//! );
//! let far = Place::new(
//!     2,
//!     vec![
//!         Point::new(0.05, 0.05),
//!         Point::new(0.05, 0.051),
//!         Point::new(0.051, 0.051),
//!         Point::new(0.051, 0.05),
//!         Point::new(0.05, 0.05),
//!     ],
//!     Point::new(0.0505, 0.0505),
//!     Some(445_110),
//!     800.0,
//! );
//! let cluster = Cluster::from_pings(vec![
//!     GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
//!     GpsPing::new(0.0005, 0.0005, 30.0, 5.0),
//!     GpsPing::new(0.00055, 0.00045, 60.0, 5.0),
//! ])
//! .unwrap();
//! let ranker = GbdtRanker::train(
//!     &[LabeledExample {
//!         cluster: cluster.clone(),
//!         candidates: vec![near.clone(), far.clone()],
//!         true_place_id: 1,
//!     }],
//!     &TrainConfig::default(),
//! )
//! .unwrap();
//!
//! let places = vec![near, far];
//! let attributor = VisitAttributor::builder()
//!     .config(Config::default())
//!     .ranker(ranker)
//!     .build()
//!     .unwrap();
//! let visits = attributor
//!     .attribute(&cluster.pings, &places)
//!     .unwrap()
//!     .visits;
//! assert!(!visits.is_empty());
//! assert_eq!(visits[0].place_id, 1);
//! ```

mod clean;
mod cluster;
mod config;
mod error;
mod features;
mod gbdt;
mod geo;
mod join;
mod pipeline;
mod rank;
mod types;

// Domain
#[doc(inline)]
pub use types::{Cluster, GpsPing, Place, PlaceId, Point, Visit};
#[doc(inline)]
pub use config::{Config, ConfigBuilder};
#[doc(inline)]
pub use error::{Error, Result};

// Pipeline
#[doc(inline)]
pub use features::{FeatureSchema, LabeledExample};
#[doc(inline)]
pub use gbdt::{GbdtModel, TrainConfig};
#[doc(inline)]
pub use rank::{GbdtRanker, Ranker};
#[doc(inline)]
pub use pipeline::{with_parts, AttributionResult, VisitAttributor, VisitAttributorBuilder};

// Extension (custom stage wiring via `with_parts`)
#[doc(inline)]
pub use clean::{DefaultPingCleaner, PingCleaner};
#[doc(inline)]
pub use cluster::{Clusterer, TwoPassClusterer};
#[doc(inline)]
pub use join::{BruteForcePlaceIndexFactory, PlaceIndex, PlaceIndexFactory};
