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
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Model(msg) => write!(f, "model error: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidInput(_) | Self::Model(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    #[test]
    fn io_variant_preserves_error_kind() -> Result<()> {
        let err: Error = io::Error::new(ErrorKind::NotFound, "missing model").into();
        let Error::Io(inner) = &err else {
            return Err(Error::InvalidInput("expected Io".into()));
        };
        assert_eq!(inner.kind(), ErrorKind::NotFound);
        assert!(err.source().is_some());
        assert!(err.to_string().contains("missing model"));
        Ok(())
    }
}
