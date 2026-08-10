# Rust Documentation

When writing or editing Rust doc comments and module docs, follow these rules.
They align with the [Microsoft Pragmatic Rust Guidelines —
Documentation](https://microsoft.github.io/rust-guidelines/guidelines/docs/)
(M-FIRST-DOC-SENTENCE, M-MODULE-DOCS, M-CANONICAL-DOCS, M-DOC-INLINE).

## First sentence (summary)

- The **first sentence** of an item’s doc comment is the summary shown in the
  module overview.
- Keep the first sentence on **one line** and to approximately **15 words** so
  docs stay skimmable and avoid widows in rendered output.
- Use outer doc comments (`///`) for items; use inner (`//!`) only for module-
  or crate-level docs.

## Module documentation

- **Public modules** must have `//!` module documentation.
- The first sentence of the module doc must follow the same rule (one line, ~15
  words).
- Module docs should cover, where relevant:
  - What the module contains and when to use it (and when not)
  - Examples and subsystem specs (e.g. formatting language)
  - Observable side effects and guarantees
  - Relevant implementation details (e.g. system APIs used)

## Canonical sections

- The **summary sentence** must always be present.
- Use these **canonical sections** when they apply (after the summary and any
  extended prose):
  - `# Examples` — one or more usage examples
  - `# Errors` — if the item returns `Result`, describe known error conditions
  - `# Panics` — if the item can panic, describe when
  - `# Safety` — if the item is `unsafe` or can cause UB, list all conditions
    the caller must uphold
  - `# Abort` — if the item can abort the process, describe when
- **Do not** use parameter tables. Describe parameters in prose (e.g. “Copies
  from `src` to `dst`”), not as a `# Parameters` list.

## Re-exports

- When publicly re-exporting **crate items** with `pub use foo::Foo` (or `pub
  use foo::*`), add **`#[doc(inline)]`** at the use site so the re-export is
  inlined in the docs.
- **Do not** use `#[doc(inline)]` for `std` or third-party types; leave those as
  opaque re-exports so it is clear they are external.
- Prefer explicit re-exports over globs; `#[doc(inline)]` does not override the
  general preference to avoid glob re-exports.

## Reference

- **Source**: [Documentation — Pragmatic Rust
  Guidelines](https://microsoft.github.io/rust-guidelines/guidelines/docs/)

Apply these documentation rules for all Rust doc comments and `//!` module docs
unless the user explicitly asks to ignore them.
