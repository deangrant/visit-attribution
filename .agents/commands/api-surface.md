# API surface checklist

Use before expanding `pub` items or `pub use` in `src/lib.rs`.

## Checklist

- [ ] Is this type/trait part of the supported public contract, or an internal
  helper (`geo`, `spatial`, preference helpers)?
- [ ] Prefer keeping `geo` / `spatial` crate-private; callers should not need
  haversine/PIP directly.
- [ ] Unexported-but-`pub` cluster/join types stay out of `lib.rs` unless there
  is a deliberate extension story (use `with_parts` instead).
- [ ] New public items have `///` docs (and `# Errors` when fallible).
- [ ] Re-exports use `#[doc(inline)]` to match existing style.
- [ ] Examples and/or unit tests cover the new surface.
- [ ] Ask the human before widening the public API.

## Success

No accidental publicization; docs and tests updated when the API intentionally
grows.
