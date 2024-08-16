// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::any::Any;
use std::error::Error;
use std::fmt::Display;
use std::sync::Arc;

use super::status_code::StatusCode;

/// Extension to [`Error`](std::error::Error) in std.
pub trait ErrorExt {
    /// Map this error to [StatusCode].
    fn status_code(&self) -> StatusCode {
        StatusCode::Unknown
    }

    /// Returns the error as [Any](std::any::Any) so that it can be
    /// downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;
}

pub trait StackError: Error {
    fn debug_fmt(&self, layer: usize, buf: &mut Vec<String>);
}

impl<T: ?Sized + StackError> StackError for Arc<T> {
    fn debug_fmt(&self, layer: usize, buf: &mut Vec<String>) {
        self.as_ref().debug_fmt(layer, buf)
    }
}

impl<T: StackError> StackError for Box<T> {
    fn debug_fmt(&self, layer: usize, buf: &mut Vec<String>) {
        self.as_ref().debug_fmt(layer, buf)
    }
}

/// An opaque boxed error based on errors that implement [ErrorExt] trait.
pub struct BoxedError {
    inner: Box<dyn ErrorExt + Send + Sync>,
}

impl BoxedError {
    pub fn new<E: ErrorExt + Send + Sync + Display + 'static>(err: E) -> Self {
        Self {
            inner: Box::new(err),
        }
    }
}

impl std::fmt::Debug for BoxedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Use the pretty debug format of inner error for opaque error.
        let _debug_format = super::format::DebugFormat::new(&*self.inner);
        // debug_format.fmt(f)
        todo!("BoxedError::Debug")
    }
}

impl Display for BoxedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Ok(write!(f, "{:?}", self.inner)?)
        todo!("BoxedError::Debug")
    }
}

impl ErrorExt for BoxedError {
    fn status_code(&self) -> StatusCode {
        self.inner.status_code()
    }

    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }
}

// Implement ErrorCompat for this opaque error so the backtrace is also available
// via `ErrorCompat::backtrace()`.
impl snafu::ErrorCompat for BoxedError {
    fn backtrace(&self) -> Option<&snafu::Backtrace> {
        None
    }
}

impl Error for BoxedError {}

impl StackError for BoxedError {
    fn debug_fmt(&self, layer: usize, buf: &mut Vec<String>) {
        // self.inner.debug_fmt(layer, buf)
        todo!("BoxedError::debug_fmt")
    }
}

/// Error type with plain error message
#[derive(Debug)]
pub struct PlainError {
    msg: String,
    status_code: StatusCode,
}

impl PlainError {
    pub fn new(msg: String, status_code: StatusCode) -> Self {
        Self { msg, status_code }
    }
}

impl Display for PlainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl Error for PlainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl ErrorExt for PlainError {
    fn status_code(&self) -> StatusCode {
        self.status_code
    }

    fn as_any(&self) -> &dyn Any {
        self as _
    }
}

impl StackError for PlainError {
    fn debug_fmt(&self, layer: usize, buf: &mut Vec<String>) {
        buf.push(format!("{}: {}", layer, self.msg))
    }
}