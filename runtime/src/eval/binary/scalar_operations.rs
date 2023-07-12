use metricsql::common::Operator;

use crate::{RuntimeError, RuntimeResult};

/// Supported operation between two float type values.
pub(crate) fn scalar_binary_operations(op: Operator, lhs: f64, rhs: f64, return_bool: bool) -> RuntimeResult<f64> {
    metricsql::binaryop::scalar_binary_operation(lhs, rhs, op, return_bool).map_err(|_| {
        RuntimeError::NotImplemented(format!(
            "scalar_binary_operation failed: op = {:?}, lhs = {:?}, rhs = {:?}",
            op, lhs, rhs
        ))
    })
}
