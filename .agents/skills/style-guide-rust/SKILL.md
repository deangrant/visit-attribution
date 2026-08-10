---
name: style-guide-rust
description: >-
  Enforce Rust code style for formatting, file size, rustfmt alignment, comments,
  documentation, naming, and API conventions. Use when writing or editing Rust
  code, reviewing .rs diffs, applying rustfmt preferences, or checking doc
  comments and module docs against project style.
trigger: >-
  Rust style, rustfmt, line length 100, max file 500 lines, indentation, trailing
  commas, blank lines, doc comments, ///, //!, #[doc(inline)], API guidelines,
  naming conventions, early returns, let-else, #[expect], Debug Display
---

# Rust Code Style

You are an expert Rust developer following a strict, clean, and maintainable
style. When writing or editing Rust code, strictly obey these rules (they
override any conflicting defaults). Align with the referenced style guides and
checklists.

## Core Formatting Rules

- **Maximum line length: 100 characters**
  Never exceed 100 characters per line (including indentation, comments,
  strings). Prefer 80–90 when it improves readability.

- **Maximum file length: 500 lines**
  Keep `.rs` files under 500 lines (excluding blank lines and license header).
  If a file approaches ~450 lines:
  - Extract structs/enums + impl blocks to separate modules
  - Move large helper functions or private sub-modules
  - Group related items; avoid god-files

## rustfmt Alignment

Follow rustfmt defaults with these overrides:

- Spaces only, never tabs; **indentation = 4 spaces**
- `max_width = 100`, `comment_width = 100` (prefer wrapping prose at ~80)
- Trailing commas in multi-line lists/arrays/tuples/matches
- `chain_width = 80` — avoid long method chains on one line
- Block indentation style (not visual/aligned) for function args, structs, etc.
- `imports_granularity = "Module"` (or `"Crate"` consistently)
- `group_imports = "StdExternalCrate"`
- **Single `#[derive(...)]` attribute**: one attribute with multiple traits
  comma-separated (e.g. `#[derive(Debug, Clone)]`), not multiple separate
  `#[derive(...)]` attributes. Order of derived names matters for tooling.

## Blank Lines and Whitespace

- **Blank lines**: Separate items and statements by zero or one blank line
  (one or two newlines). Do not use two or more consecutive blank lines between
  items.
- **Trailing whitespace**: No trailing whitespace on any line (code, comments,
  blank lines, or string literals). Preserve literal value when editing strings.

## Comments

- Prefer line comments (`//`) over block comments (`/* ... */`). Put a single
  space after `//`.
- Prefer a comment on its own line. If it follows code, put a single space
  before it.
- Comments should be complete sentences: start with a capital letter, end with a
  period. Inline block comments may be notes without punctuation.
- Lines that are entirely a comment: limit to 80 characters (including sigils,
  excluding indentation), or the max line width including indentation (100),
  whichever is smaller.

## Doc Comments and Documentation

- Prefer outer doc comments (`///`) and line style; use inner (`//!`) only for
  module- or crate-level docs.
- **Put doc comments before attributes** (e.g. `/// Summary.` then
  `#[derive(Debug)]`).
- **First sentence**: One line, approximately 15 words; it becomes the summary
  in module overview. Keep it skimmable.
- **Module docs**: Public modules must have `//!` documentation; first sentence
  follows the same rule.
- **Canonical sections** when applicable: `# Examples`, `# Errors`, `# Panics`,
  `# Safety`, `# Abort`. Do not use parameter tables; describe parameters in
  prose (e.g. "Copies from `src` to `dst`").
- For `pub use` re-exports of crate items, use `#[doc(inline)]` at the use site
  so docs inline; do not use for `std` or third-party types.
- For full documentation rules (first sentence, module docs, canonical sections,
  re-exports), see [documentation.md](documentation.md).
- Reference: [Rust API Guidelines — Documentation](https://rust-lang.github.io/api-guidelines/documentation.html), [Microsoft Docs guidelines](https://microsoft.github.io/rust-guidelines/guidelines/docs/).

## Attributes

- Put each attribute on its own line, indented to the item level. Prefer outer
  attributes. For attributes with argument lists, format like function calls;
  for `key = value`, use a space before and after `=`.

## Small Items

- Use single-line formatting when a construct is "small" (e.g. `Foo { f1, f2 }`).
  Use multi-line block style when items are longer or more complex. Prefer tool
  heuristics (e.g. character count or simple vs complex sub-expressions).

## Style Preferences

- Prefer small functions (ideally < 40–50 lines).
- Prefer early returns over deep nesting; use `let-else` and `if let` to reduce
  nesting.
- Avoid rightward drift — break long expressions early.
- **Functions and methods**: Use inherent methods for constructors (static, no
  `self`) and when there is a clear receiver; otherwise prefer regular (free)
  functions unless an associated function improves clarity. See
  [API Guidelines — Predictability](https://rust-lang.github.io/api-guidelines/predictability.html)
  and
  [Microsoft M-REGULAR-FN](https://microsoft.github.io/rust-guidelines/guidelines/universal/#M-REGULAR-FN).
- **Naming**: Consistent word order; getters per Rust convention; ad-hoc
  conversions use `as_`, `to_`, `into_`; iterators use `iter`, `iter_mut`,
  `into_iter`; names concise, free of weasel words. Casing per RFC 430.
- **Public types**: Implement `Debug` for all public types; implement `Display`
  for types meant to be read by users.
- **Lint overrides**: Prefer `#[expect(...)]` over `#[allow(...)]` when
  overriding lints.
- **Panic vs error**: Use panics for programming bugs (stop the program); use
  `Result`/errors for recoverable failure. Panic means "stop the program."
- Tests: focused `#[test]` functions with descriptive names; avoid huge test
  blocks.

## References

- **Documentation**: See [documentation.md](documentation.md) for API doc and
  module-doc rules aligned with
  [Microsoft Documentation](https://microsoft.github.io/rust-guidelines/guidelines/docs/).
- **Full style references**: See [references.md](references.md).

## Enforcement

- Never produce lines > 100 chars.
- Warn or suggest refactor when a file nears 500 lines.
- Use rustfmt-compatible style in all snippets.
- If something cannot fit in 100 chars (e.g. long macros/literals), break it
  vertically with good taste.

Apply this style in all Rust-related responses unless the user explicitly asks
to ignore it.
