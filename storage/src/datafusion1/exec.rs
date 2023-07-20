// Copyright 2022 Zinc Labs Inc. and Contributors
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

use std::sync::Arc;

use ahash::AHashMap as HashMap;
use datafusion::datasource::object_store::DefaultObjectStoreRegistry;
use datafusion::{
    arrow::{datatypes::Schema, record_batch::RecordBatch},
    datasource::{
        file_format::file_type::{FileType, GetExt},
        listing::{ListingTable, ListingTableConfig},
        object_store::ObjectStoreRegistry,
    },
    error::{DataFusionError, Result},
    execution::{
        context::SessionConfig,
        runtime_env::{RuntimeConfig, RuntimeEnv},
    },
    prelude::SessionContext,
};
use once_cell::sync::Lazy;

use crate::common::meta::{common::FileMeta, search::Session as SearchSession, sql};
use crate::service::search::sql::Sql;

use super::storage::{file_list, StorageType};
use super::transform_udf::get_all_transform;

pub fn create_runtime_env() -> Result<RuntimeEnv> {
    let object_store_registry = DefaultObjectStoreRegistry::new();

    let fsm = super::storage::memory::FS::new();
    let fsm_url = url::Url::parse("fsm:///").unwrap();
    object_store_registry.register_store(&fsm_url, Arc::new(fsm));

    let fsn = super::storage::nocache::FS::new();
    let fsn_url = url::Url::parse("fsn:///").unwrap();
    object_store_registry.register_store(&fsn_url, Arc::new(fsn));

    let rn_config =
        RuntimeConfig::new().with_object_store_registry(Arc::new(object_store_registry));
    RuntimeEnv::new(rn_config)
}

pub fn prepare_datafusion_context() -> Result<SessionContext, DataFusionError> {
    let runtime_env = create_runtime_env()?;
    let session_config = SessionConfig::new()
        .with_batch_size(8192)
        .set_bool("datafusion.execution.parquet.pushdown_filters", true);
    Ok(SessionContext::with_config_rt(
        session_config,
        Arc::new(runtime_env),
    ))
}

async fn register_udf(ctx: &mut SessionContext) {
    ctx.register_udf(super::match_udf::MATCH_UDF.clone());
    ctx.register_udf(super::match_udf::MATCH_IGNORE_CASE_UDF.clone());
    ctx.register_udf(super::regexp_udf::REGEX_MATCH_UDF.clone());
    ctx.register_udf(super::regexp_udf::REGEX_NOT_MATCH_UDF.clone());
    ctx.register_udf(super::time_range_udf::TIME_RANGE_UDF.clone());
    ctx.register_udf(super::date_format_udf::DATE_FORMAT_UDF.clone());
}

pub async fn register_table(
    session: &SearchSession,
    schema: Arc<Schema>,
    table_name: &str,
    files: &[String],
    file_type: FileType,
) -> Result<SessionContext> {
    let ctx = prepare_datafusion_context()?;

    let config = ListingTableConfig::new(prefix)
        .with_listing_options(listing_options)
        .with_schema(schema);
    let table = ListingTable::try_new(config)?;
    ctx.register_table(table_name, Arc::new(table))?;

    Ok(ctx)
}

#[cfg(test)]
mod test {
    use arrow::array::Int32Array;
    use arrow_schema::Field;

    use super::*;

    #[actix_web::test]
    async fn test_register_udf() {
        let mut ctx = SessionContext::new();
        let _ = register_udf(&mut ctx, "nexus").await;
        //assert!(res)
    }
}
