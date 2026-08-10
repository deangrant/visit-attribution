//! Crate-wide error type and result alias.

use std::error::Error as StdError;
use std::fmt;

/// Errors returned by visit attribution, training, and helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Input data or configuration failed validation.
    InvalidInput(String),
    /// Model file or feature schema did not match expectations.
    Model(String),
    /// I/O failure (e.g. saving or loading a model).
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Model(msg) => write!(f, "model error: {msg}"),
            Error::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl StdError for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value.to_string())
    }
}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;
