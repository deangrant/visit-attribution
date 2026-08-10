//! End-to-end example: train a ranker and attribute a short trajectory.

use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, TrainConfig, VisitAttributor,
};

fn main() {
    let cafe = Place::square(1, 51.5000, -0.1200, 0.001, Some(722_515));
    let shop = Place::square(2, 51.5000, -0.1190, 0.001, Some(445_110));
    let pings = vec![
        GpsPing::new(51.5004, -0.1196, 1_700_000_000.0, 8.0),
        GpsPing::new(51.5005, -0.1195, 1_700_000_030.0, 8.0),
        GpsPing::new(51.50045, -0.11955, 1_700_000_060.0, 8.0),
    ];
    let cluster = Cluster::from_pings(pings.clone()).unwrap();
    let ranker = GbdtRanker::train(
        &[LabeledExample {
            cluster: cluster.clone(),
            candidates: vec![cafe.clone(), shop.clone()],
            true_place_id: 1,
        }],
        &TrainConfig {
            n_trees: 16,
            ..TrainConfig::default()
        },
    )
    .expect("train");

    let places = vec![cafe, shop];
    let attributor = VisitAttributor::builder()
        .config(Config::builder().join_radius_m(120.0).build().expect("config"))
        .ranker(ranker)
        .build()
        .expect("build");
    let visits = attributor.attribute(&pings, &places).expect("attribute").visits;
    for visit in &visits {
        println!(
            "visit place_id={} wins={} duration_s={:.0} candidates={:?}",
            visit.place_id,
            visit.wins,
            visit.cluster.duration_s(),
            visit.candidates
        );
    }
    assert!(!visits.is_empty());
}
