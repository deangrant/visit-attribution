//! Crate-wide error type and result alias.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Errors returned by visit attribution, training, and helpers.
#[derive(Debug)]
pub enum Error {
    /// Input data or configuration failed validation.
    InvalidInput(String),
    /// Model file or feature schema did not match expectations.
    Model(String),
    /// I/O failure (e.g. saving or loading a model).
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Model(msg) => write!(f, "model error: {msg}"),
            Error::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            Error::InvalidInput(_) | Error::Model(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Error::Io(value)
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    #[test]
    fn io_variant_preserves_error_kind() {
        let err: Error = io::Error::new(ErrorKind::NotFound, "missing model").into();
        match &err {
            Error::Io(inner) => assert_eq!(inner.kind(), ErrorKind::NotFound),
            other => panic!("expected Io, got {other:?}"),
        }
        assert!(err.source().is_some());
        assert!(err.to_string().contains("missing model"));
    }
}
