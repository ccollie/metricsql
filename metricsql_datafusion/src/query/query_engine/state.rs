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
use std::sync::Arc;

use datafusion::catalog::MemoryCatalogList;
use datafusion::dataframe::DataFrame;
use datafusion::error::Result as DfResult;
use datafusion::execution::context::{SessionConfig, SessionState};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::SessionContext;

use crate::catalog::CatalogManagerRef;
use crate::table::adapter::DfTableProviderAdapter;
use crate::table::TableRef;

/// Query query_engine global state
// TODO(yingwen): This QueryEngineState still relies on datafusion, maybe we can define a trait
// for it, which allows different implementation use different query_engine state. The state can also
// be an associated type in QueryEngine trait.
#[derive(Clone)]
pub struct QueryEngineState {
    df_context: SessionContext,
    catalog_manager: CatalogManagerRef,
    disallow_cross_schema_query: bool,
}

impl fmt::Debug for QueryEngineState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("QueryEngineState")
            .field("state", &self.df_context.state())
            .finish()
    }
}

impl QueryEngineState {
    pub fn new(
        catalog_list: CatalogManagerRef,
    ) -> Self {
        let runtime_env = Arc::new(RuntimeEnv::default());
        let session_config = SessionConfig::new().with_create_default_catalog_and_schema(false);
        let session_state = SessionState::new_with_config_rt_and_catalog_list(
            session_config,
            runtime_env,
            Arc::new(MemoryCatalogList::default()), // pass a dummy catalog list
        );

        let df_context = SessionContext::new_with_state(session_state);

        Self {
            df_context,
            catalog_manager: catalog_list,
            disallow_cross_schema_query: false,
        }
    }

    #[inline]
    pub fn catalog_manager(&self) -> &CatalogManagerRef {
        &self.catalog_manager
    }

    pub(crate) fn disallow_cross_schema_query(&self) -> bool {
        self.disallow_cross_schema_query
    }

    pub(crate) fn session_state(&self) -> SessionState {
        self.df_context.state()
    }

    /// Create a DataFrame for a table
    pub fn read_table(&self, table: TableRef) -> DfResult<DataFrame> {
        self.df_context
            .read_table(Arc::new(DfTableProviderAdapter::new(table)))
    }
}