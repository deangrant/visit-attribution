# CI check

Run the same gates as GitHub Actions lint + test. Report failures only; do not
commit.

## Steps

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo test`

## Success

All three commands exit 0. If any fail, fix the underlying code (not by
weakening lints) and re-run.
