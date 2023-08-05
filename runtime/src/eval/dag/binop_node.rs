use serde::{Deserialize, Serialize};

use metricsql::common::{BinModifier, Operator};

use crate::eval::binary::{
    eval_scalar_vector_binop, eval_string_string_binop, eval_vector_scalar_binop,
    scalar_binary_operation,
};
use crate::eval::dag::utils::{exec_vector_vector, resolve_value};
use crate::eval::dag::ExecutableNode;
use crate::eval::eval_number;
use crate::{Context, EvalConfig, QueryValue, RuntimeError, RuntimeResult};

/// A node that represents a binary operation between two runtime QueryValue nodes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BinopNode {
    pub op: Operator,
    pub modifier: Option<BinModifier>,
    pub left_idx: usize,
    pub right_idx: usize,
    #[serde(skip)]
    pub left: QueryValue,
    #[serde(skip)]
    pub right: QueryValue,
}

impl BinopNode {
    pub fn new(
        left_idx: usize,
        right_idx: usize,
        op: Operator,
        modifier: Option<BinModifier>,
    ) -> Self {
        Self {
            op,
            left_idx,
            right_idx,
            modifier,
            ..Default::default()
        }
    }
}

impl ExecutableNode for BinopNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        resolve_value(self.left_idx, &mut self.left, dependencies);
        resolve_value(self.right_idx, &mut self.right, dependencies);
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        use QueryValue::*;

        let is_tracing = ctx.trace_enabled();
        let (bool_modifier, keep_metric_names) = if let Some(modifier) = &self.modifier {
            let is_bool = modifier.return_bool;
            // todo: check op
            (is_bool, modifier.keep_metric_names)
        } else {
            (false, false)
        };

        match (&mut self.left, &mut self.right) {
            (Scalar(left), Scalar(right)) => {
                let value = scalar_binary_operation(self.op, *left, *right, bool_modifier)?;
                Ok(Scalar(value))
            }
            (InstantVector(ref mut left_vec), InstantVector(ref mut right_vec)) => {
                let left = std::mem::take(left_vec);
                let right = std::mem::take(right_vec);
                exec_vector_vector(ctx, left, right, self.op, &self.modifier)
            }
            (InstantVector(ref mut vector), Scalar(scalar)) => {
                if self.op.is_logical_op() {
                    // expand scalar to vector. Note we handle unless specially, since it's
                    // possible to calculate without expanding the scalar (and allocating memory).
                    if self.op != Operator::Unless {
                        let left = std::mem::take(vector);
                        let right = eval_number(ec, *scalar)?;
                        return exec_vector_vector(ctx, left, right, self.op, &self.modifier);
                    }
                }
                eval_vector_scalar_binop(
                    std::mem::take(vector),
                    self.op,
                    *scalar,
                    bool_modifier,
                    keep_metric_names,
                    is_tracing,
                )
            }
            (Scalar(scalar), InstantVector(ref mut vector)) => {
                if self.op.is_logical_op() {
                    let left = eval_number(ec, *scalar)?;
                    let right = std::mem::take(vector);
                    return exec_vector_vector(ctx, left, right, self.op, &self.modifier);
                };
                eval_scalar_vector_binop(
                    *scalar,
                    self.op,
                    std::mem::take(vector),
                    bool_modifier,
                    keep_metric_names,
                    is_tracing,
                )
            }
            (String(left), String(right)) => {
                eval_string_string_binop(self.op, &left, &right, bool_modifier)
            }
            _ => {
                return Err(RuntimeError::NotImplemented(format!(
                    "invalid binary operation: {} {} {}",
                    self.left.data_type_name(),
                    self.op,
                    self.right.data_type_name()
                )));
            }
        }
    }
}
