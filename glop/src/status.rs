//! Unrecoverable GLOP errors.
//!
//! This is the Rust counterpart of `ortools/glop/status.{h,cc}`. Recoverable
//! simplex outcomes use `ProblemStatus`; this type is only for failures that
//! prevent the algorithm from continuing.

use std::fmt;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ErrorCode {
    #[default]
    GlopOk,
    ErrorLu,
    ErrorBound,
    ErrorNull,
    ErrorInvalidProblem,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::GlopOk => "GLOP_OK",
            Self::ErrorLu => "ERROR_LU",
            Self::ErrorBound => "ERROR_BOUND",
            Self::ErrorNull => "ERROR_NULL",
            Self::ErrorInvalidProblem => "INVALID_PROBLEM",
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Status {
    error_code: ErrorCode,
    error_message: String,
}

impl Status {
    #[must_use]
    pub fn new(error_code: ErrorCode, error_message: impl Into<String>) -> Self {
        Self {
            error_code,
            error_message: if error_code == ErrorCode::GlopOk {
                String::new()
            } else {
                error_message.into()
            },
        }
    }

    #[must_use]
    pub const fn ok() -> Self {
        Self {
            error_code: ErrorCode::GlopOk,
            error_message: String::new(),
        }
    }

    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        self.error_code
    }

    #[must_use]
    pub fn error_message(&self) -> &str {
        &self.error_message
    }

    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self.error_code, ErrorCode::GlopOk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_discards_message_like_upstream() {
        let status = Status::new(ErrorCode::GlopOk, "ignored");
        assert!(status.is_ok());
        assert_eq!(status.error_message(), "");
        assert_eq!(
            ErrorCode::ErrorInvalidProblem.to_string(),
            "INVALID_PROBLEM"
        );
    }
}
