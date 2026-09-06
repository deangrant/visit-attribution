//! Helpers so tests can avoid `unwrap` / `expect` / `panic`.

use crate::error::{Error, Result};

/// Pass through a successful result; map unexpected `Ok` after `err`.
pub fn err<T: std::fmt::Debug>(result: Result<T>) -> Result<Error> {
    match result {
        Err(e) => Ok(e),
        Ok(value) => Err(Error::InvalidInput(format!(
            "expected error, got {value:?}"
        ))),
    }
}

/// Require `Some` without `unwrap`.
pub fn some<T>(value: Option<T>) -> Result<T> {
    value.ok_or_else(|| Error::InvalidInput("expected Some".into()))
}
