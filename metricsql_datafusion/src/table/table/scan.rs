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
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow_schema::SchemaRef;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::{Partitioning, PhysicalSortExpr};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use futures::{Stream, StreamExt};
use snafu::OptionExt;

use crate::common::recordbatch::{RecordBatch, RecordBatchStream, SendableRecordBatchStream};
use crate::common::recordbatch::error::RecordBatchResult;
use crate::query::error::{ExecuteRepeatedlySnafu, Result as QueryResult};
use crate::query::physical_plan::{PhysicalPlan, PhysicalPlanRef};
use crate::table::table::metrics::MemoryUsageMetrics;

/// Adapt greptime's [SendableRecordBatchStream] to GreptimeDB's [PhysicalPlan].
pub struct StreamScanAdapter {
    stream: Mutex<Option<SendableRecordBatchStream>>,
    schema: SchemaRef,
    output_ordering: Option<Vec<PhysicalSortExpr>>,
    metric: ExecutionPlanMetricsSet,
}

impl Debug for StreamScanAdapter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamScanAdapter")
            .field("stream", &"<SendableRecordBatchStream>")
            .field("schema", &self.schema.arrow_schema().fields)
            .finish()
    }
}

impl StreamScanAdapter {
    pub fn new(stream: SendableRecordBatchStream) -> Self {
        let schema = stream.schema();

        Self {
            stream: Mutex::new(Some(stream)),
            schema,
            output_ordering: None,
            metric: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn with_output_ordering(mut self, output_ordering: Vec<PhysicalSortExpr>) -> Self {
        self.output_ordering = Some(output_ordering);
        self
    }
}

impl PhysicalPlan for StreamScanAdapter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.output_ordering.as_deref()
    }

    fn children(&self) -> Vec<PhysicalPlanRef> {
        vec![]
    }

    fn with_new_children(&self, _children: Vec<PhysicalPlanRef>) -> QueryResult<PhysicalPlanRef> {
        Ok(Arc::new(Self::new(
            self.stream.lock().unwrap().take().unwrap(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> QueryResult<SendableRecordBatchStream> {
        let mut stream = self.stream.lock().unwrap();
        let stream = stream.take().context(ExecuteRepeatedlySnafu)?;
        let mem_usage_metrics = MemoryUsageMetrics::new(&self.metric, partition);
        Ok(Box::pin(StreamWithMetricWrapper {
            stream,
            metric: mem_usage_metrics,
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metric.clone_inner())
    }
}

pub struct StreamWithMetricWrapper {
    stream: SendableRecordBatchStream,
    metric: MemoryUsageMetrics,
}

impl Stream for StreamWithMetricWrapper {
    type Item = RecordBatchResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll = this.stream.poll_next_unpin(cx);
        if let Poll::Ready(Some(Ok(record_batch))) = &poll {
            let batch_mem_size = record_batch
                .columns()
                .iter()
                .map(|vec_ref| vec_ref.memory_size())
                .sum::<usize>();
            // we don't record elapsed time here
            // since it's calling storage api involving I/O ops
            this.metric.record_mem_usage(batch_mem_size);
            this.metric.record_output(record_batch.num_rows());
        }

        poll
    }
}

impl RecordBatchStream for StreamWithMetricWrapper {
    fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }
}

#[cfg(test)]
mod test {
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Schema};
    use datafusion::prelude::SessionContext;

    use crate::common::recordbatch::{RecordBatches, util};
    use crate::table::schema::column_schema::Field;

    use super::*;

    #[tokio::test]
    async fn test_simple_table_scan() {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));

        let batch1 = RecordBatch::new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_slice([1, 2])) as _],
        )
        .unwrap();
        let batch2 = RecordBatch::new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_slice([3, 4, 5])) as _],
        )
        .unwrap();

        let record_batches =
            RecordBatches::try_new(schema.clone(), vec![batch1.clone(), batch2.clone()]).unwrap();
        let stream = record_batches.as_stream();

        let scan = StreamScanAdapter::new(stream);

        assert_eq!(scan.schema(), schema);

        let stream = scan.execute(0, ctx.task_ctx()).unwrap();
        let recordbatches = util::collect(stream).await.unwrap();
        assert_eq!(recordbatches[0], batch1);
        assert_eq!(recordbatches[1], batch2);

        let result = scan.execute(0, ctx.task_ctx());
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e
                .to_string()
                .contains("Not expected to run ExecutionPlan more than once")),
            _ => unreachable!(),
        }
    }
}
