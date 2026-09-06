//! Persist a trained ranker: train → save → load → attribute.
//!
//! Demonstrates reusing a `.va` model file across processes/runs.

use std::io::{self, Write};
use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Result, TrainConfig,
    VisitAttributor,
};

fn main() -> Result<()> {
    let near = Place::square(1, 0.0, 0.0, 0.001, Some(445_110));
    let far = Place::square(2, 0.05, 0.05, 0.001, Some(445_110));
    let pings = vec![
        GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
        GpsPing::new(0.0005, 0.0005, 30.0, 5.0),
        GpsPing::new(0.00055, 0.00045, 60.0, 5.0),
    ];
    let cluster = Cluster::from_pings(pings.clone())?;

    let trained = GbdtRanker::train(
        &[LabeledExample {
            cluster,
            candidates: vec![near.clone(), far.clone()],
            true_place_id: 1,
        }],
        &TrainConfig {
            n_trees: 24,
            max_depth: 3,
            ..TrainConfig::default()
        },
    )?;

    let path = std::env::temp_dir().join("visit-attribution-persist.va");
    trained.save(&path)?;
    writeln!(io::stdout(), "saved model to {}", path.display())?;

    let loaded = GbdtRanker::load(&path)?;
    let _ = std::fs::remove_file(&path);

    let places = vec![near, far];
    let visits = VisitAttributor::builder()
        .config(Config::builder().join_radius_m(80.0).build()?)
        .ranker(loaded)
        .build()?
        .attribute(&pings, &places)?
        .visits;

    let mut out = io::stdout();
    for visit in &visits {
        writeln!(
            out,
            "visit place_id={} wins={} duration_s={:.0} candidates={:?}",
            visit.place_id,
            visit.wins,
            visit.cluster.duration_s(),
            visit.candidates
        )?;
    }
    assert!(
        !visits.is_empty(),
        "loaded ranker should attribute the stay"
    );
    assert_eq!(visits[0].place_id, 1);
    Ok(())
}
