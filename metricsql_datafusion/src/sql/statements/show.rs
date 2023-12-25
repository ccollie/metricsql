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

use std::fmt;

use datafusion_expr::Expr;

/// Show kind for SQL expressions like `SHOW DATABASE` or `SHOW TABLE`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShowKind {
    All,
    Like(String),
    Where(Expr),
}

impl fmt::Display for ShowKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ShowKind::All => write!(f, "ALL"),
            ShowKind::Like(ident) => write!(f, "LIKE {ident}"),
            ShowKind::Where(expr) => write!(f, "WHERE {expr}"),
        }
    }
}

/// SQL structure for `SHOW DATABASES`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowDatabases {
    pub kind: ShowKind,
}

impl ShowDatabases {
    /// Creates a statement for `SHOW DATABASES`
    pub fn new(kind: ShowKind) -> Self {
        ShowDatabases { kind }
    }
}

/// SQL structure for `SHOW TABLES`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowTables {
    pub kind: ShowKind,
    pub database: Option<String>,
    pub full: bool,
}

#[cfg(test)]
mod tests {
}
