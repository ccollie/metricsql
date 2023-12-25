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

// -----------------------------------------------------------------------------

// ---------- Names reserved for internal columns and query_engine -------------------

/// Name for reserved column: sequence
pub const SEQUENCE_COLUMN_NAME: &str = "__sequence";

/// Name for reserved column: op_type
pub const OP_TYPE_COLUMN_NAME: &str = "__op_type";

/// Name for reserved column: primary_key
pub const PRIMARY_KEY_COLUMN_NAME: &str = "__primary_key";

/// Internal Column Name
static INTERNAL_COLUMN_VEC: [&str; 3] = [
    SEQUENCE_COLUMN_NAME,
    OP_TYPE_COLUMN_NAME,
    PRIMARY_KEY_COLUMN_NAME,
];

pub fn is_internal_column(name: &str) -> bool {
    INTERNAL_COLUMN_VEC.contains(&name)
}

// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_internal_column() {
        // contain internal column names
        assert!(is_internal_column(SEQUENCE_COLUMN_NAME));
        assert!(is_internal_column(OP_TYPE_COLUMN_NAME));
        assert!(is_internal_column(PRIMARY_KEY_COLUMN_NAME));

        // don't contain internal column names
        assert!(!is_internal_column("my__column"));
        assert!(!is_internal_column("my__sequence"));
        assert!(!is_internal_column("my__op_type"));
        assert!(!is_internal_column("my__primary_key"));
    }
}