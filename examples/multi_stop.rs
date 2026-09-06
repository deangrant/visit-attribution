//! Multi-stop day: cleaning, multiple visits, and unmatched clusters.
//!
//! Trajectory: dwell at a cafe → driving hop → bad-accuracy spike → dwell at a
//! shop → orphan dwell far from the catalog.

use std::io::{self, Write};
use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Result, TrainConfig,
    VisitAttributor,
};

fn main() -> Result<()> {
    // Catalog: two places far enough apart that each dwell joins only one.
    let cafe = Place::square(1, 0.0, 0.0, 0.001, Some(722_515));
    let shop = Place::square(2, 0.05, 0.05, 0.001, Some(445_110));

    // Trajectory (seconds from an arbitrary epoch).
    let pings = vec![
        // Dwell near the cafe.
        GpsPing::new(0.0004, 0.0004, 0.0, 8.0),
        GpsPing::new(0.0005, 0.00045, 20.0, 8.0),
        GpsPing::new(0.00045, 0.0005, 40.0, 8.0),
        // Driving hop (~1.1 km in 10 s) — cleaner should drop arrivals.
        GpsPing::new(0.01, 0.01, 50.0, 8.0),
        GpsPing::new(0.02, 0.02, 60.0, 8.0),
        GpsPing::new(0.03, 0.03, 70.0, 8.0),
        // Accuracy spike — dropped by max_horizontal_accuracy_m.
        GpsPing::new(0.0504, 0.0504, 80.0, 5_000.0),
        // Dwell near the shop (after a time gap past the drive).
        GpsPing::new(0.0504, 0.0504, 2_000.0, 8.0),
        GpsPing::new(0.0505, 0.05045, 2_020.0, 8.0),
        GpsPing::new(0.05045, 0.0505, 2_040.0, 8.0),
        // Orphan dwell far from every place (slow enough to pass jump filter,
        // far enough to miss join_radius_m) → unmatched_clusters.
        GpsPing::new(-0.05, 0.05, 5_000.0, 8.0),
        GpsPing::new(-0.0501, 0.05, 5_020.0, 8.0),
        GpsPing::new(-0.05, 0.0501, 5_040.0, 8.0),
    ];

    let cafe_cluster = Cluster::from_pings(vec![
        GpsPing::new(0.0004, 0.0004, 0.0, 8.0),
        GpsPing::new(0.0005, 0.00045, 20.0, 8.0),
        GpsPing::new(0.00045, 0.0005, 40.0, 8.0),
    ])?;
    let shop_cluster = Cluster::from_pings(vec![
        GpsPing::new(0.0504, 0.0504, 2_000.0, 8.0),
        GpsPing::new(0.0505, 0.05045, 2_020.0, 8.0),
        GpsPing::new(0.05045, 0.0505, 2_040.0, 8.0),
    ])?;

    let ranker = GbdtRanker::train(
        &[
            LabeledExample {
                cluster: cafe_cluster,
                candidates: vec![cafe.clone(), shop.clone()],
                true_place_id: 1,
            },
            LabeledExample {
                cluster: shop_cluster,
                candidates: vec![cafe.clone(), shop.clone()],
                true_place_id: 2,
            },
        ],
        &TrainConfig {
            n_trees: 24,
            max_depth: 3,
            ..TrainConfig::default()
        },
    )?;

    let config = Config::builder().join_radius_m(80.0).min_cluster_pings(2).build()?;
    let places = vec![cafe, shop];
    let result = VisitAttributor::builder()
        .config(config)
        .ranker(ranker)
        .build()?
        .attribute(&pings, &places)?;

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
    writeln!(
        out,
        "unmatched_clusters={}",
        result.unmatched_clusters.len()
    )?;
    for cluster in &result.unmatched_clusters {
        writeln!(
            out,
            "  start_s={:.0} duration_s={:.0} pings={}",
            cluster.start_time_s,
            cluster.duration_s(),
            cluster.pings.len()
        )?;
    }

    assert!(
        result.visits.len() >= 2,
        "expected cafe and shop visits, got {}",
        result.visits.len()
    );
    assert!(
        !result.unmatched_clusters.is_empty(),
        "expected an orphan dwell far from the catalog"
    );
    Ok(())
}
