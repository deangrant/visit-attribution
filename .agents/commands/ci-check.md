# CI check

Run the same gates as GitHub Actions lint + test + supply-chain. Report
failures only; do not commit.

## Steps

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
4. `cargo test --all-targets --all-features`
5. `cargo deny check` (if `cargo-deny` is installed)

## Success

All commands exit 0. If any fail, fix the underlying code (not by weakening
lints) and re-run.
