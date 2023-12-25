use std::fmt::{Debug, Formatter};

pub use ::datafusion::physical_plan::ExecutionPlan;

use crate::common::recordbatch::{RecordBatches, SendableRecordBatchStream};

pub mod error;
pub mod sql;
pub mod physical_plan;
pub mod physical_planner;
pub mod query_engine;
pub mod datafusion;
pub(crate) mod metrics;
pub mod executor;
pub(crate) mod planner;
pub(crate) mod logical_optimizer;
pub(crate) mod physical_optimizer;
mod helper;

// sql output
pub enum Output {
    AffectedRows(usize),
    RecordBatches(RecordBatches),
    Stream(SendableRecordBatchStream),
}

impl Debug for Output {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return match self {
            Output::AffectedRows(rows) => write!(f, "Output::AffectedRows({rows})"),
            Output::RecordBatches(batches) => {
                write!(f, "Output::RecordBatches({batches:?})")
            }
            Output::Stream(_) => write!(f, "Output::Stream(<stream>)"),
        }
    }
}

