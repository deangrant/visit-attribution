# Agent and contributor guidance

Structured conventions for AI agents and humans working in this repository. For
fuller context, see [README.md](README.md).

## Docs

- [`.agents/docs/ARCHITECTURE.md`](.agents/docs/ARCHITECTURE.md) — high-level system architecture and diagrams
- [DeepWiki](https://deepwiki.com/deangrant/visit-attribution) — indexed project wiki (architecture, API, pipeline)

## Rules

- [`.agents/rules/`](.agents/rules/) (symlinked from [`.cursor/rules`](.cursor/rules))
- [`.agents/rules/visit-attribution-core.mdc`](.agents/rules/visit-attribution-core.mdc) — always-on crate policy (std-first, MSRV, CI, API)
- [`.agents/rules/rust-style-surface.mdc`](.agents/rules/rust-style-surface.mdc) — Rust formatting and docs surface checklist
- [`.agents/rules/pipeline-domain.mdc`](.agents/rules/pipeline-domain.mdc) — clean/cluster/join/rank domain gotchas

## Skills

- [`.agents/skills/`](.agents/skills/)
- [`.agents/skills/visit-attribution-pipeline/`](.agents/skills/visit-attribution-pipeline/) — pipeline stages, training, tests, examples
- [`.agents/skills/style-guide-rust/`](.agents/skills/style-guide-rust/) — Rust style, rustfmt, docs, naming
- [`.agents/skills/solid-rust/`](.agents/skills/solid-rust/) — SOLID design in Rust

## Commands

- [`.agents/commands/`](.agents/commands/) (symlinked from [`.cursor/commands`](.cursor/commands))
- `/ci-check` — fmt, clippy (`all`/`pedantic`/`nursery`, `-D warnings`),
  rustdoc (`-D warnings`), test, and `cargo deny check` when available
- `/run-examples` — run `basic`, `multi_stop`, `strip_mall`, `persist_model`
- `/msrv-check` — `cargo +1.74 test`
- `/api-surface` — checklist before expanding `pub use` in `lib.rs`

## Hooks

- Config: [`.cursor/hooks.json`](.cursor/hooks.json)
- `afterFileEdit` → [`.agents/hooks/rustfmt.sh`](.agents/hooks/rustfmt.sh) formats edited `*.rs` files
