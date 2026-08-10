# MSRV check

Confirm the crate builds and tests on the declared MSRV (`rust-version = 1.74`
in `Cargo.toml`). CI currently runs stable only, so this catches newer syntax.

## Steps

1. Ensure the toolchain exists: `rustup toolchain install 1.74`
2. `cargo +1.74 test`

## Success

`cargo +1.74 test` exits 0. If it fails on syntax or std APIs, rewrite to stay
within 1.74—do not raise MSRV without an explicit human ask.
