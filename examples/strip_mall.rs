//! Strip mall: large-POI pass plus preference ranking among adjacent stores.
//!
//! A mall footprint qualifies for the large-POI clusterer. Interior pings still
//! join nearby store polygons; the trained ranker prefers store A over store B.

use std::io::{self, Write};
use visit_attribution::{
    AttributionResult, Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Result,
    TrainConfig, VisitAttributor,
};

fn catalog() -> (Place, Place, Place) {
    (
        Place::square(1, -0.002, -0.002, 0.004, Some(531_120)),
        Place::square(10, -0.0010, -0.0010, 0.0008, Some(445_110)),
        Place::square(20, -0.0010, 0.0002, 0.0008, Some(722_515)),
    )
}

fn dwell_pings() -> Vec<GpsPing> {
    vec![
        GpsPing::new(-0.0007, -0.0007, 0.0, 6.0),
        GpsPing::new(-0.00065, -0.0007, 15.0, 6.0),
        GpsPing::new(-0.0007, -0.00065, 30.0, 6.0),
        GpsPing::new(-0.00068, -0.00068, 45.0, 6.0),
    ]
}

fn train_ranker(
    mall: &Place,
    store_a: &Place,
    store_b: &Place,
    pings: &[GpsPing],
) -> Result<GbdtRanker> {
    let cluster = Cluster::from_pings(pings.to_vec())?;
    let candidates = vec![mall.clone(), store_a.clone(), store_b.clone()];
    let examples: Vec<LabeledExample> = (0..8)
        .map(|_| LabeledExample {
            cluster: cluster.clone(),
            candidates: candidates.clone(),
            true_place_id: 10,
        })
        .collect();
    GbdtRanker::train(
        &examples,
        &TrainConfig {
            n_trees: 48,
            max_depth: 3,
            learning_rate: 0.2,
            ..TrainConfig::default()
        },
    )
}

fn main() -> Result<()> {
    let (mall, store_a, store_b) = catalog();
    let pings = dwell_pings();
    let ranker = train_ranker(&mall, &store_a, &store_b, &pings)?;
    let config = Config::builder()
        .large_poi_area_m2(1_000.0)
        .join_radius_m(80.0)
        .min_cluster_pings(2)
        .build()?;
    let places = vec![mall, store_a, store_b];
    let result = VisitAttributor::builder()
        .config(config)
        .ranker(ranker)
        .build()?
        .attribute(&pings, &places)?;
    report_visits(&result)?;
    assert_store_a_won(&result);
    Ok(())
}

fn report_visits(result: &AttributionResult) -> Result<()> {
    let mut out = io::stdout();
    writeln!(out, "visits={}", result.visits.len())?;
    for visit in &result.visits {
        writeln!(
            out,
            "  place_id={} wins={} duration_s={:.0} candidates={:?}",
            visit.place_id,
            visit.wins,
            visit.cluster.duration_s(),
            visit.candidates
        )?;
    }
    Ok(())
}

fn assert_store_a_won(result: &AttributionResult) {
    assert_eq!(result.visits.len(), 1, "expected a single strip-mall visit");
    assert_eq!(
        result.visits[0].place_id, 10,
        "ranker should prefer store A over mall/store B"
    );
    assert!(
        result.visits[0].candidates.contains(&1),
        "join should include the mall as a competing candidate"
    );
    assert!(
        result.visits[0].candidates.len() >= 2,
        "join should surface competing candidates"
    );
}
