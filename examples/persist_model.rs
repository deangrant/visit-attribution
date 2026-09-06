//! Persist a trained ranker: train → save → load → attribute.
//!
//! Demonstrates reusing a `.va` model file across processes/runs.

use std::io::{self, Write};
use std::path::Path;
use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Result, TrainConfig, Visit,
    VisitAttributor,
};

fn catalog_and_pings() -> (Place, Place, Vec<GpsPing>) {
    (
        Place::square(1, 0.0, 0.0, 0.001, Some(445_110)),
        Place::square(2, 0.05, 0.05, 0.001, Some(445_110)),
        vec![
            GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
            GpsPing::new(0.0005, 0.0005, 30.0, 5.0),
            GpsPing::new(0.00055, 0.00045, 60.0, 5.0),
        ],
    )
}

fn train_ranker(near: &Place, far: &Place, pings: &[GpsPing]) -> Result<GbdtRanker> {
    let cluster = Cluster::from_pings(pings.to_vec())?;
    GbdtRanker::train(
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
    )
}

fn report(visits: &[Visit]) -> Result<()> {
    let mut out = io::stdout();
    for visit in visits {
        writeln!(
            out,
            "visit place_id={} wins={} duration_s={:.0} candidates={:?}",
            visit.place_id,
            visit.wins,
            visit.cluster.duration_s(),
            visit.candidates
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let (near, far, pings) = catalog_and_pings();
    let trained = train_ranker(&near, &far, &pings)?;
    let path = std::env::temp_dir().join("visit-attribution-persist.va");
    trained.save(&path)?;
    writeln!(io::stdout(), "saved model to {}", path.display())?;
    let loaded = GbdtRanker::load(Path::new(&path))?;
    let _ = std::fs::remove_file(&path);
    let places = vec![near, far];
    let visits = VisitAttributor::builder()
        .config(Config::builder().join_radius_m(80.0).build()?)
        .ranker(loaded)
        .build()?
        .attribute(&pings, &places)?
        .visits;
    report(&visits)?;
    assert!(
        !visits.is_empty(),
        "loaded ranker should attribute the stay"
    );
    assert_eq!(visits[0].place_id, 1);
    Ok(())
}
