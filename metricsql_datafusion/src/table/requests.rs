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

//! Table and TableEngine requests

use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use arrow::array::ArrayRef;
use serde::{Deserialize, Serialize};

use crate::common::ReadableSize;
use crate::table::error;
use crate::table::error::ParseTableOptionSnafu;
use crate::table::metadata::TableId;
use crate::table::schema::raw::RawSchema;
use crate::table::TableReference;

pub const DB_TABLE_NAME: &str = "databases";

pub const FILE_TABLE_LOCATION_KEY: &str = "location";
pub const FILE_TABLE_PATTERN_KEY: &str = "pattern";
pub const FILE_TABLE_FORMAT_KEY: &str = "format";

/// Create table request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTableRequest {
    pub id: TableId,
    pub catalog_name: String,
    pub schema_name: String,
    pub table_name: String,
    pub desc: Option<String>,
    pub schema: RawSchema,
    pub primary_key_indices: Vec<usize>,
    pub table_options: TableOptions,
    pub engine: String,
}

impl CreateTableRequest {
    pub fn table_ref(&self) -> TableReference {
        TableReference {
            catalog: &self.catalog_name,
            schema: &self.schema_name,
            table: &self.table_name,
        }
    }
}

#[derive(Debug)]
pub struct InsertRequest {
    pub catalog_name: String,
    pub schema_name: String,
    pub table_name: String,
    pub columns_values: HashMap<String, ArrayRef>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TableOptions {
    pub time_column: Option<String>,
    pub value_column: Option<String>,
    /// Memtable size of memtable.
    pub write_buffer_size: Option<ReadableSize>,
    /// Time-to-live of table. Expired data will be automatically purged.
    #[serde(with = "humantime_serde")]
    pub ttl: Option<Duration>,
    /// Extra options that may not applicable to all table engines.
    pub extra_options: HashMap<String, String>,
}

pub const WRITE_BUFFER_SIZE_KEY: &str = "write_buffer_size";
pub const TTL_KEY: &str = "ttl";
pub const REGIONS_KEY: &str = "regions";

pub const TIME_COLUMN_KEY: &str = "time_column";
pub const VALUE_COLUMN_KEY: &str = "value_column";

impl TryFrom<&HashMap<String, String>> for TableOptions {
    type Error = error::Error;

    fn try_from(value: &HashMap<String, String>) -> Result<Self, Self::Error> {
        let mut options = TableOptions::default();
        if let Some(write_buffer_size) = value.get(WRITE_BUFFER_SIZE_KEY) {
            let size = ReadableSize::from_str(write_buffer_size).map_err(|_| {
                ParseTableOptionSnafu {
                    key: WRITE_BUFFER_SIZE_KEY,
                    value: write_buffer_size,
                }
                .build()
            })?;
            options.write_buffer_size = Some(size)
        }

        if let Some(time_column) = value.get(TIME_COLUMN_KEY) {
            options.time_column = Some(time_column.clone());
        }
        if let Some(value_column) = value.get(VALUE_COLUMN_KEY) {
            options.value_column = Some(value_column.clone());
        }
        if let Some(ttl) = value.get(TTL_KEY) {
            let ttl_value = ttl
                .parse::<humantime::Duration>()
                .map_err(|_| {
                    ParseTableOptionSnafu {
                        key: TTL_KEY,
                        value: ttl,
                    }
                    .build()
                })?
                .into();
            options.ttl = Some(ttl_value);
        }
        options.extra_options = HashMap::from_iter(value.iter().filter_map(|(k, v)| {
            if k != WRITE_BUFFER_SIZE_KEY && k != TTL_KEY {
                Some((k.clone(), v.clone()))
            } else {
                None
            }
        }));
        Ok(options)
    }
}

impl From<&TableOptions> for HashMap<String, String> {
    fn from(opts: &TableOptions) -> Self {
        let mut res = HashMap::with_capacity(2 + opts.extra_options.len());
        if let Some(write_buffer_size) = opts.write_buffer_size {
            let _ = res.insert(
                WRITE_BUFFER_SIZE_KEY.to_string(),
                write_buffer_size.to_string(),
            );
        }
        if let Some(ttl) = opts.ttl {
            let ttl_str = humantime::format_duration(ttl).to_string();
            let _ = res.insert(TTL_KEY.to_string(), ttl_str);
        }
        res.extend(
            opts.extra_options
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );
        res
    }
}

/// Open table request
#[derive(Debug, Clone)]
pub struct OpenTableRequest {
    pub catalog_name: String,
    pub schema_name: String,
    pub table_name: String,
    pub table_id: TableId,
}

/// Drop table request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropTableRequest {
    pub catalog_name: String,
    pub schema_name: String,
    pub table_name: String,
    pub table_id: TableId,
}

impl DropTableRequest {
    pub fn table_ref(&self) -> TableReference {
        TableReference {
            catalog: &self.catalog_name,
            schema: &self.schema_name,
            table: &self.table_name,
        }
    }
}

/// Drop table request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseTableRequest {
    pub catalog_name: String,
    pub schema_name: String,
    pub table_name: String,
    pub table_id: TableId,
}

impl CloseTableRequest {
    pub fn table_ref(&self) -> TableReference {
        TableReference {
            catalog: &self.catalog_name,
            schema: &self.schema_name,
            table: &self.table_name,
        }
    }
}

#[macro_export]
macro_rules! meter_insert_request {
    ($req: expr) => {
        meter_macros::write_meter!(
            $req.catalog_name.to_string(),
            $req.schema_name.to_string(),
            $req.table_name.to_string(),
            $req
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_table_options() {
        let options = TableOptions {
            time_column: None,
            value_column: None,
            write_buffer_size: None,
            ttl: Some(Duration::from_secs(1000)),
            extra_options: HashMap::new(),
        };
        let serialized = serde_json::to_string(&options).unwrap();
        let deserialized: TableOptions = serde_json::from_str(&serialized).unwrap();
        assert_eq!(options, deserialized);
    }

    #[test]
    fn test_convert_hashmap_between_table_options() {
        let options = TableOptions {
            time_column: None,
            value_column: None,
            write_buffer_size: Some(ReadableSize::mb(128)),
            ttl: Some(Duration::from_secs(1000)),
            extra_options: HashMap::new(),
        };
        let serialized_map = HashMap::from(&options);
        let serialized = TableOptions::try_from(&serialized_map).unwrap();
        assert_eq!(options, serialized);

        let options = TableOptions {
            time_column: None,
            value_column: None,
            write_buffer_size: None,
            ttl: None,
            extra_options: HashMap::new(),
        };
        let serialized_map = HashMap::from(&options);
        let serialized = TableOptions::try_from(&serialized_map).unwrap();
        assert_eq!(options, serialized);

        let options = TableOptions {
            time_column: None,
            value_column: None,
            write_buffer_size: Some(ReadableSize::mb(128)),
            ttl: Some(Duration::from_secs(1000)),
            extra_options: HashMap::from([("a".to_string(), "A".to_string())]),
        };
        let serialized_map = HashMap::from(&options);
        let serialized = TableOptions::try_from(&serialized_map).unwrap();
        assert_eq!(options, serialized);
    }
}
