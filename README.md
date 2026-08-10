# visit-attribution

visit-attribution is a Rust crate. Use it to attribute GPS trajectory **pings**
to point-of-interest (**POI**) **visits**.

The pipeline cleans noisy pings, clusters them into potential visits, joins each
cluster to nearby places, and ranks candidates with a preference-learning
gradient-boosted model.

## Technical overview

visit-attribution works in four stages.

1. **Clean** — Drop poor horizontal accuracy, impossible jumps, and driving.
2. **Cluster** — Run a large-footprint POI pass, then time-aware density
   clustering on the remaining pings.
3. **Join** — Match each cluster to nearby place polygons within a join radius.
4. **Rank** — Score candidates with pairwise preference features, an in-crate
   GBDT, and a tournament scorecard.

Each attributed stay becomes a **visit** (chosen `place_id`, wins, candidates).
Clusters with no join candidates appear in `unmatched_clusters`. That outcome
is not an error.

The crate is **std-first**. It has no required third-party dependencies.

## Requirements

- Use Rust and Cargo.
- Use Rust 1.74 or later (MSRV).
- This crate has no required third-party dependencies.

## Install

Add the crate to your project.

```toml
[dependencies]
visit-attribution = "0.1"
```

## Usage

### Train, then attribute

1. Build places (for example with `Place::square`).
2. Train a `GbdtRanker` on labeled examples. The true place must be among at
   least two candidates so preference pairs exist.
3. Build a `VisitAttributor` with a `Config` and the ranker.
4. Call `attribute` with pings and the place catalog.

The builder requires a ranker. There is no heuristic default ranker.

```rust
use visit_attribution::{
    Cluster, Config, GbdtRanker, GpsPing, LabeledExample, Place, TrainConfig,
    VisitAttributor,
};

fn main() -> visit_attribution::Result<()> {
    let cafe = Place::square(1, 51.5000, -0.1200, 0.001, Some(722_515));
    let shop = Place::square(2, 51.5000, -0.1190, 0.001, Some(445_110));
    let pings = vec![
        GpsPing::new(51.5004, -0.1196, 1_700_000_000.0, 8.0),
        GpsPing::new(51.5005, -0.1195, 1_700_000_030.0, 8.0),
        GpsPing::new(51.50045, -0.11955, 1_700_000_060.0, 8.0),
    ];
    let cluster = Cluster::from_pings(pings.clone())?;
    let ranker = GbdtRanker::train(
        &[LabeledExample {
            cluster,
            candidates: vec![cafe.clone(), shop.clone()],
            true_place_id: 1,
        }],
        &TrainConfig::default(),
    )?;

    let places = vec![cafe, shop];
    let result = VisitAttributor::builder()
        .config(Config::builder().join_radius_m(120.0).build()?)
        .ranker(ranker)
        .build()?
        .attribute(&pings, &places)?;

    for visit in &result.visits {
        println!(
            "place_id={} wins={} candidates={:?}",
            visit.place_id, visit.wins, visit.candidates
        );
    }
    println!("unmatched={}", result.unmatched_clusters.len());
    Ok(())
}
```

### Persist a model

Save and load a trained ranker as text `VA_GBDT 1` (`.va` by convention).

```rust
use visit_attribution::GbdtRanker;

fn demo(ranker: GbdtRanker) -> visit_attribution::Result<()> {
    ranker.save("model.va")?;
    let loaded = GbdtRanker::load("model.va")?;
    let _ = loaded;
    Ok(())
}
```

### Custom stages

Defaults use `DefaultPingCleaner`, `TwoPassClusterer`, and
`QuadtreePlaceIndexFactory`. Use `with_parts` to inject custom stage
implementations (for example a specialized `PlaceIndex` for a large catalog).

## Input and output

### Input

- `GpsPing`: WGS84 latitude and longitude, time in seconds, horizontal accuracy
  in metres.
- `Place`: id, exterior polygon ring, centroid, optional NAICS code.
- Training: `LabeledExample` with a cluster, candidate places, and
  `true_place_id`.

### Output

- `AttributionResult.visits`: ranked visits (`place_id`, `wins`, `candidates`,
  source `cluster`).
- `AttributionResult.unmatched_clusters`: stays with an empty join set.

## Parameters

Set pipeline values on `Config::builder`. Defaults come from `Config::default`.

- `max_horizontal_accuracy_m` (default `100`): Drop inaccurate pings.
- `max_speed_m_s` (default `50`): Drop impossible jumps.
- `linearity_threshold` (default `1.15`): Max path/net ratio for a driving
  window.
- `linearity_window_s` (default `60`): Minimum window length for the driving
  filter.
- `driving_speed_m_s` (default `8`): Speed that marks driving inside a linear
  window.
- `dist_threshold_m` (default `80`): Cluster neighbor distance.
- `max_dist_threshold_m` (default `100`): Max jump within a density cluster.
- `max_time_gap_s` (default `1800`): Max gap between consecutive cluster pings.
- `min_cluster_pings` (default `2`): Minimum density cluster size.
- `large_poi_area_m2` (default `50000`): Area threshold for the large-POI pass.
- `join_radius_m` (default `50`): Max distance from a place polygon for join
  candidates.

Effective join distance is `join_radius_m` plus the maximum ping horizontal
accuracy in the cluster.

Set GBDT values on `TrainConfig` (defaults: `n_trees = 32`, `max_depth = 3`,
`learning_rate = 0.1`, `max_naics4 = 32`, and related leaf/bin settings).

## Errors

- `InvalidInput`: The config, training data, or other caller input is not valid.
- `Model`: A model file or feature schema does not match expectations.
- `Io`: A file operation failed (for example save or load).

## Examples

```bash
cargo run --example basic
cargo run --example multi_stop      # cleaning, multiple visits, unmatched clusters
cargo run --example strip_mall      # large-POI pass + adjacent-store ranking
cargo run --example persist_model   # train → save → load → attribute
```

Run the tests.

```bash
cargo test
```

## Further reading

- [Architecture](.agents/docs/ARCHITECTURE.md) — module map, stage detail, extension points
- [DeepWiki](https://deepwiki.com/deangrant/visit-attribution) — indexed project wiki
- [AGENTS.md](AGENTS.md) — agent and contributor tooling index

## Security

Use [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit) to
check dependencies against the RustSec advisory database.

```bash
cargo install cargo-audit --locked
```

This project ignores `Cargo.lock` in git (library convention). Generate a
lockfile locally when you need it, then run the audit.

```bash
cargo generate-lockfile
cargo audit
```

GitHub Actions runs the same check on dependency and workflow changes, on a
daily schedule, and on manual workflow dispatch. See
[`.github/workflows/audit.yml`](.github/workflows/audit.yml).

## License

This project uses the MIT license. See [LICENSE](LICENSE).
