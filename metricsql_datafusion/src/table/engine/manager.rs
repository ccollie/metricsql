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

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use snafu::{ensure, OptionExt};
use tracing::error;

use crate::table::engine::TableEngineRef;
use crate::table::error::{EngineExistSnafu, EngineNotFoundSnafu, Result};

#[async_trait::async_trait]
pub trait TableEngineManager: Send + Sync {
    /// returns [Error::EngineNotFound](crate::error::Error::EngineNotFound) if query_engine not found
    fn engine(&self, name: &str) -> Result<TableEngineRef>;

    /// returns [Error::EngineExist](crate::error::Error::EngineExist) if query_engine exists
    fn register_engine(&self, name: &str, engine: TableEngineRef) -> Result<()>;

    /// closes all registered engines
    async fn close(&self) -> Result<()>;
}

pub type TableEngineManagerRef = Arc<dyn TableEngineManager>;

/// Simple in-memory table query_engine manager
pub struct MemoryTableEngineManager {
    pub engines: RwLock<HashMap<String, TableEngineRef>>,
}

impl MemoryTableEngineManager {
    /// Create a new [MemoryTableEngineManager] with single table `query_engine`.
    pub fn new(engine: TableEngineRef) -> Self {
        MemoryTableEngineManager::alias(engine.name().to_string(), engine)
    }

    /// Create a new [MemoryTableEngineManager] with single table `query_engine` and
    /// an alias `name` instead of the query_engine's name.
    pub fn alias(name: String, engine: TableEngineRef) -> Self {
        let engines = HashMap::from([(name, engine)]);
        let engines = RwLock::new(engines);

        MemoryTableEngineManager { engines }
    }

    pub fn with(engines: Vec<TableEngineRef>) -> Self {
        let engines = engines
            .into_iter()
            .map(|engine| (engine.name().to_string(), engine))
            .collect::<HashMap<_, _>>();
        let engines = RwLock::new(engines);
        MemoryTableEngineManager { engines }
    }
}

#[async_trait]
impl TableEngineManager for MemoryTableEngineManager {
    fn engine(&self, name: &str) -> Result<TableEngineRef> {
        let engines = self.engines.read().unwrap();
        engines
            .get(name)
            .cloned()
            .context(EngineNotFoundSnafu { engine: name })
    }

    fn register_engine(&self, name: &str, engine: TableEngineRef) -> Result<()> {
        let mut engines = self.engines.write().unwrap();

        ensure!(
            !engines.contains_key(name),
            EngineExistSnafu { engine: name }
        );

        let _ = engines.insert(name.to_string(), engine);

        Ok(())
    }

    async fn close(&self) -> Result<()> {
        let engines = {
            let engines = self.engines.write().unwrap();
            engines.values().cloned().collect::<Vec<_>>()
        };

        if let Err(err) =
            futures::future::try_join_all(engines.iter().map(|engine| engine.close())).await
        {
            error!("Failed to close query_engine: {:?}", err);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::sync::Arc;

    use datafusion::logical_expr::UserDefinedLogicalNode;

    use crate::table::engine::manager::MemoryTableEngineManager;
    use crate::table::error::Error::EngineNotFound;
    use crate::table::test_util::mock_engine::MockTableEngine;

    use super::*;

    #[test]
    fn test_table_engine_manager() {
        let table_engine = MockTableEngine::new();
        let table_engine_ref = Arc::new(table_engine);
        let table_engine_manager = MemoryTableEngineManager::new(table_engine_ref.clone());

        table_engine_manager
            .register_engine("yet_another", table_engine_ref.clone())
            .unwrap();

        let got = table_engine_manager.engine(table_engine_ref.name());

        assert_eq!(got.unwrap().name(), table_engine_ref.name());

        let got = table_engine_manager.engine("yet_another");

        assert_eq!(got.unwrap().name(), table_engine_ref.name());

        let missing = table_engine_manager.engine("not_exists");

        assert_matches!(missing.err().unwrap(), EngineNotFound { .. });

        assert!(table_engine_manager
            .engine_procedure(table_engine_ref.name())
            .is_ok());
        assert_matches!(
            table_engine_manager
                .engine_procedure("unknown")
                .err()
                .unwrap(),
            EngineNotFound { .. }
        );
    }
}