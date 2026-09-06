//! Integration: synthetic strip mall with two adjacent stores.

use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Result, TrainConfig,
    VisitAttributor,
};

#[test]
fn attributes_visit_to_trained_place() -> Result<()> {
    let left = Place::square(10, 0.0, 0.0, 0.0008, Some(445_110));
    let right = Place::square(20, 0.0, 0.0010, 0.0008, Some(722_515));
    let pings = vec![
        GpsPing::new(0.0003, 0.0003, 0.0, 5.0),
        GpsPing::new(0.0004, 0.00035, 20.0, 5.0),
        GpsPing::new(0.00035, 0.0004, 40.0, 5.0),
        GpsPing::new(0.00045, 0.0003, 60.0, 5.0),
    ];
    let cluster = Cluster::from_pings(pings.clone())?;
    let ranker = GbdtRanker::train(
        &[LabeledExample {
            cluster,
            candidates: vec![left.clone(), right.clone()],
            true_place_id: 10,
        }],
        &TrainConfig {
            n_trees: 24,
            max_depth: 3,
            ..TrainConfig::default()
        },
    )?;

    let config = Config::builder().min_cluster_pings(2).join_radius_m(80.0).build()?;
    let places = vec![left, right];
    let attributor = VisitAttributor::builder().config(config).ranker(ranker).build()?;
    let result = attributor.attribute(&pings, &places)?;
    assert_eq!(result.visits.len(), 1);
    assert_eq!(result.visits[0].place_id, 10);
    assert!(result.unmatched_clusters.is_empty());
    Ok(())
}
