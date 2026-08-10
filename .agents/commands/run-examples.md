# Run examples

Build and run every crate example. Summarize stdout briefly per example.

## Steps

```bash
cargo run --example basic
cargo run --example multi_stop
cargo run --example strip_mall
cargo run --example persist_model
```

## What each covers

- `basic` — train + attribute
- `multi_stop` — cleaning, multiple visits, unmatched clusters
- `strip_mall` — large-POI pass + adjacent-store ranking
- `persist_model` — train → save → load → attribute

## Success

Each example exits 0. Note any panic, attribution mismatch, or I/O failure.
