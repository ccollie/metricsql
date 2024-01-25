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
use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{Field, SchemaRef};
use crate::common::recordbatch::SendableRecordBatchStream;
use datafusion_expr::TableType;
use serde::{Deserialize, Serialize};
use snafu::ResultExt;

use metricsql_common::prelude::BoxedError;

use crate::data_source::DataSource;
use crate::datasource::file_format::Format;
use crate::table::{Table, TableInfo, TableInfoRef};
use crate::table::error::{TableResult, TablesRecordBatchSnafu};
use crate::table::storage::ScanRequest;

use super::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOptions {
    pub files: Vec<String>,
    pub file_column_schemas: Vec<Field>,
}


#[derive(Debug)]
pub struct FileTable {
    pub(crate) file_options: FileOptions,
    pub table_info: TableInfo,
    pub url: String,
    pub format: Format,
    pub(crate) options: HashMap<String, String>,
}

pub type FileTableRef = Arc<FileTable>;

#[derive(Debug, Clone)]
pub struct FileTableCreateRequest {
    pub url: String,
    /// Region query_engine name
    pub engine: String,

    pub format: Format,
    /// Columns in this region.
    pub column_metadatas: Vec<Field>,

    pub file_options: FileOptions,

    /// Options of the created table.
    pub options: HashMap<String, String>,
}

impl FileTable {
    pub async fn create(
        request: FileTableCreateRequest,
    ) -> Result<FileTableRef> {
        let url = request.url;
        let options = request.options.clone();

        Ok(Arc::new(Self {
            url,
            file_options: request.file_options,
            format: request.format,
            options,
            table_info,
        }))
    }


    pub fn schema(&self) -> SchemaRef {
        self.table_info.schema().clone()
    }
}

impl DataSource for FileTable {
    fn get_stream(&self, request: ScanRequest) -> std::result::Result<SendableRecordBatchStream, BoxedError> {
        self.query(request).map_err(BoxedError::new)
    }
}

impl Table for FileTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema()
    }

    fn table_info(&self) -> TableInfoRef {
        Arc::new(self.table_info.clone())
    }

    fn table_type(&self) -> TableType {
        self.table_info.table_type
    }

    async fn scan_to_stream(&self, request: ScanRequest) -> TableResult<SendableRecordBatchStream> {
        self.query(request)
            .context(TablesRecordBatchSnafu)
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use crate::file_engine::test_util::{new_test_column_metadata, new_test_object_store, new_test_options};

    use super::*;
}
