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

/// Pretty debug format for error, also prints source and backtrace.
pub struct DebugFormat<'a, E: ?Sized>(&'a E);

impl<'a, E: ?Sized> DebugFormat<'a, E> {
    /// Create a new format struct from `err`.
    pub fn new(err: &'a E) -> Self {
        Self(err)
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::error::ext::{ErrorExt, StackError};
    use snafu::prelude::*;
    use snafu::Location;

    #[derive(Debug, Snafu)]
    #[snafu(display("This is a leaf error"))]
    struct Leaf;

    impl ErrorExt for Leaf {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl StackError for Leaf {
        fn debug_fmt(&self, _: usize, _: &mut Vec<String>) {}
    }

    #[derive(Debug, Snafu)]
    #[snafu(display("This is a leaf with location"))]
    struct LeafWithLocation {
        location: Location,
    }

    impl ErrorExt for LeafWithLocation {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl StackError for LeafWithLocation {
        fn debug_fmt(&self, _: usize, _: &mut Vec<String>) {}
    }

    #[derive(Debug, Snafu)]
    #[snafu(display("Internal error"))]
    struct Internal {
        #[snafu(source)]
        source: Leaf,
        location: Location,
    }

    impl ErrorExt for Internal {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl StackError for Internal {
        fn debug_fmt(&self, layer: usize, buf: &mut Vec<String>) {
            buf.push(format!("{}: Internal error", layer));
            self.source.debug_fmt(layer + 1, buf);
        }
    }
}
