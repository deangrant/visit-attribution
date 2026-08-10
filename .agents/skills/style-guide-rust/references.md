# Rust style guide — references

Align with these style guides and checklists when applying the style-guide-rust
skill.

## Official Rust style guide

- **URL**: https://doc.rust-lang.org/beta/style-guide/index.html
- Line length 100, 4-space indent, block indent, trailing commas, blank lines,
  trailing whitespace, comments, attributes, small items.

## Rust API guidelines

- **URL**: https://rust-lang.github.io/api-guidelines/
- **Checklist**: https://rust-lang.github.io/api-guidelines/checklist.html
- Naming, interoperability, documentation, predictability, flexibility, type
  safety, dependability, debuggability, future proofing.

## Microsoft Pragmatic Rust Guidelines

- **URL**: https://microsoft.github.io/rust-guidelines/
- **Checklist**:
  https://microsoft.github.io/rust-guidelines/guidelines/checklist/index.html
- **Documentation**:
  https://microsoft.github.io/rust-guidelines/guidelines/docs/ — first sentence
  ~15 words (M-FIRST-DOC-SENTENCE), module docs (M-MODULE-DOCS), canonical
  sections (M-CANONICAL-DOCS), `#[doc(inline)]` for crate re-exports
  (M-DOC-INLINE). Enforced by [documentation.md](documentation.md) in this
  folder.
- Universal, library (interop, UX, resilience, building), apps, FFI, safety,
  performance, documentation, AI. Apply spirit of guidelines; prefer regular
  functions where appropriate (M-REGULAR-FN).

## Comprehensive Rust (Google)

- **Course**: https://google.github.io/comprehensive-rust/
- **Repo**: https://github.com/google/comprehensive-rust
- Course follows official Rust style.

## rustfmt

- **URL**: https://rust-lang.github.io/rustfmt/
- Config and options.
