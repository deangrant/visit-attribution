# visit-attribution

Rust library that attributes GPS trajectory pings to point-of-interest (POI)
visits. The pipeline cleans noisy pings, clusters them into potential visits,
joins each cluster to nearby places, and ranks candidates with a
preference-learning gradient-boosted model.

**Std-first:** no required third-party crates.

## Pipeline

1. **Clean** — drop poor horizontal accuracy, impossible jumps, and driving.
2. **Cluster** — large-footprint POI pass, then time-aware density clustering.
3. **Join** — match each cluster to nearby place polygons (with join radius).
4. **Rank** — pairwise preference features + GBDT + tournament scorecard.

## Quick start

```rust
use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, Point,
    TrainConfig, VisitAttributor,
};

# fn main() -> visit_attribution::Result<()> {
let place = Place::new(
    1,
    vec![
        Point::new(0.0, 0.0),
        Point::new(0.0, 0.001),
        Point::new(0.001, 0.001),
        Point::new(0.001, 0.0),
        Point::new(0.0, 0.0),
    ],
    Point::new(0.0005, 0.0005),
    Some(445_110),
    800.0,
);
let other = Place::new(
    2,
    vec![
        Point::new(0.05, 0.05),
        Point::new(0.05, 0.051),
        Point::new(0.051, 0.051),
        Point::new(0.051, 0.05),
        Point::new(0.05, 0.05),
    ],
    Point::new(0.0505, 0.0505),
    Some(445_110),
    800.0,
);
let pings = vec![
    GpsPing::new(0.0004, 0.0004, 0.0, 5.0),
    GpsPing::new(0.0005, 0.0005, 30.0, 5.0),
    GpsPing::new(0.00055, 0.00045, 60.0, 5.0),
];
let cluster = Cluster::from_pings(pings.clone());
let ranker = GbdtRanker::train(
    &[LabeledExample {
        cluster,
        candidates: vec![place.clone(), other.clone()],
        true_place_id: 1,
    }],
    &TrainConfig::default(),
)?;
let places = vec![place, other];
let result = VisitAttributor::builder()
    .config(Config::default())
    .ranker(ranker)
    .build()?
    .attribute(&pings, &places)?;
// `result.unmatched_clusters` holds stay clusters with no join candidates.
let visits = result.visits;
# let _ = visits;
# Ok(())
# }
```

Run the example:

```bash
cargo run --example basic
```

## Default parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `max_horizontal_accuracy_m` | 100 | Drop inaccurate pings |
| `max_speed_m_s` | 50 | Drop impossible jumps |
| `dist_threshold_m` | 80 | Cluster neighbor distance |
| `max_dist_threshold_m` | 100 | Max jump within a cluster |
| `max_time_gap_s` | 1_800 | Max gap between consecutive cluster pings |
| `min_cluster_pings` | 2 | Minimum cluster size |
| `large_poi_area_m2` | 50_000 | Large-POI first pass |
| `join_radius_m` | 50 | Max distance from place polygon for join candidates |

## Model persistence

```rust
# use visit_attribution::{GbdtRanker, TrainConfig, LabeledExample};
# fn demo(ranker: GbdtRanker) -> visit_attribution::Result<()> {
ranker.save("model.va")?;
let loaded = GbdtRanker::load("model.va")?;
# let _ = loaded;
# Ok(())
# }
```

## License

MIT
