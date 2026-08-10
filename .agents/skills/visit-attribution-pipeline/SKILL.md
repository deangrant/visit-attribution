---
name: visit-attribution-pipeline
description: >-
  Domain guidance for the visit-attribution GPS→POI pipeline: clean, cluster,
  join, rank, training, tests, and examples. Use when extending stages, fixing
  geo/join behavior, training or persisting GBDT models, or writing
  attribution tests and examples.
---

# visit-attribution pipeline

Std-first crate: clean → cluster → join → rank. No third-party dependencies.

## Stage map

| Stage | Default | Trait / entry |
|-------|---------|----------------|
| Clean | `DefaultPingCleaner` | `PingCleaner` |
| Cluster | `TwoPassClusterer` | `Clusterer` |
| Join | quadtree factory | `PlaceIndex` / `PlaceIndexFactory` |
| Rank | `GbdtRanker` | `Ranker` |
| Orchestrate | `VisitAttributor` | builder or `with_parts` |

Public surface is re-exported from `src/lib.rs`. `geo` and `spatial` stay
crate-internal. Prefer `with_parts` + stage traits for custom catalogs (see
`solid-rust` skill).

## Behavioral notes

- Large-POI clustering runs first and consumes matching pings; density pass is
  time-aware / sequential (not DBSCAN).
- Join candidate if within `join_radius_m + max(HA)` of the polygon. Empty rings
  never match. Lon bboxes do not wrap the antimeridian.
- Ranker is mandatory on the builder. Empty candidate sets become
  `unmatched_clusters`, not errors.
- Training: `LabeledExample` with `true_place_id` among **≥2** candidates so
  preference pairs exist. Schema frozen at train; unknown NAICS → UNK.
- Inference: pairwise tournament; max wins; tie-break lowest `PlaceId`.
- Persist with `GbdtRanker::save` / `load` (text `VA_GBDT 1`, `.va` by convention).

## Test recipe

1. Build places with `Place::square(...)` (or small closed rings).
2. Build a short synthetic trajectory (small lat/lon deltas, finite HA).
3. Train a tiny `GbdtRanker` from `LabeledExample`s.
4. Attribute via `VisitAttributor::builder()`.
5. Assert `visits` and/or `unmatched_clusters`.

## Examples

| Example | Shows |
|---------|--------|
| `basic` | Train + attribute |
| `multi_stop` | Cleaning, multi-visit, unmatched |
| `strip_mall` | Large-POI + adjacent-store rank |
| `persist_model` | Train → save → load → attribute |

## Verify

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```
