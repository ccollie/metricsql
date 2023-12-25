use std::collections::HashMap;

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
use arrow_schema::{DataType, Field, TimeUnit};
use opendal::services::Fs;
use tempfile::TempDir;

use crate::object_store::ObjectStore;
use crate::test_util::temp_dir::create_temp_dir;

pub fn new_test_object_store(prefix: &str) -> (TempDir, ObjectStore) {
    let dir = create_temp_dir(prefix);
    let store_dir = dir.path().to_string_lossy();
    let mut builder = Fs::default();
    let _ = builder.root(&store_dir);
    (dir, ObjectStore::new(builder).unwrap().finish())
}

pub fn new_test_column_metadata() -> Vec<Field> {
    vec![
        Field::new(
            "ts",
            DataType::timestamp_datatype(TimeUnit::Millisecond),
            false,
        ),
        Field::new("str", DataType::Utf8, false),
        Field::new("num", DataType::Int64, false),
    ]
}

pub fn new_test_options() -> HashMap<String, String> {
    HashMap::from([
        ("format".to_string(), "csv".to_string()),
        ("location".to_string(), "test".to_string()),
        (
            "__private.file_table_meta".to_string(),
            "{\"files\":[\"1.csv\"],\"file_column_schemas\":[]}".to_string(),
        ),
    ])
}
