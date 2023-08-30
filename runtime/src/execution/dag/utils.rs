use std::sync::Arc;

use tracing::{field, trace_span, Span};

use metricsql::common::{BinModifier, Operator};
use metricsql::functions::RollupFunction;

use crate::execution::binary::{exec_binop, BinaryOpFuncArg};
use crate::execution::utils::series_len;
use crate::execution::{Context, EvalConfig};
use crate::functions::rollup::{get_rollup_function_handler, RollupHandler};
use crate::{InstantVector, QueryValue, RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn resolve_value(index: usize, value: &mut QueryValue, computed: &mut [QueryValue]) {
    // Note: we return values in this particular way because of an optimization in the evaluator.
    // Consider a call like
    //
    //     sort_desc(2 * (label_set(time(), "foo", "bar", "__name__", "q1")
    //          or label_set(10, "foo", "qwerty", "__name__", "q2")
    //      ) keep_metric_names)
    //
    //  Note that we have many constant dependencies (args) here. Rather than create a node for
    //  the each constant, we just store it directly in the computed array.
    //  We won't ever have non primitive (string/scalar) constants, so we can just swap the value
    //  of InstantVector/RangeVectors, but we need to preserve the value of constants.
    let dependency = &mut computed[index];
    match dependency {
        QueryValue::InstantVector(_) | QueryValue::RangeVector(_) => {
            std::mem::swap(value, dependency)
        }
        QueryValue::Scalar(v) => *value = QueryValue::Scalar(*v),
        QueryValue::String(s) => *value = QueryValue::from(s.as_str()),
    }
}

pub(crate) fn resolve_args(
    indices: &[usize],
    args: &mut Vec<QueryValue>,
    computed: &mut [QueryValue],
) {
    if args.len() != indices.len() {
        args.resize(indices.len(), QueryValue::default());
    }
    for (index, arg) in indices.iter().zip(args.iter_mut()) {
        resolve_value(*index, arg, computed);
    }
}

pub(crate) fn resolve_vector(
    index: usize,
    computed: &mut [QueryValue],
) -> RuntimeResult<InstantVector> {
    let dependency = &mut computed[index];
    // use std::mem::take to avoid clone()
    if let QueryValue::InstantVector(vector) = dependency {
        Ok(std::mem::take(vector))
    } else {
        // todo: argument error
        Err(RuntimeError::TypeCastError(format!(
            "expected vector, got {}",
            dependency.data_type_name()
        )))
    }
}

pub(super) fn resolve_rollup_handler(
    func: RollupFunction,
    arg_indexes: &[usize],
    dependencies: &mut [QueryValue],
) -> RuntimeResult<RollupHandler> {
    if arg_indexes.is_empty() {
        let empty = vec![];
        return get_rollup_function_handler(func, &empty);
    }
    let mut args = Vec::with_capacity(arg_indexes.len());
    resolve_args(arg_indexes, &mut args, dependencies);
    get_rollup_function_handler(func, &args)
}

pub(super) fn resolve_at_value(
    arg_index: Option<usize>,
    computed: &mut [QueryValue],
) -> RuntimeResult<Option<i64>> {
    if let Some(index) = arg_index {
        let value = get_at_value(&computed[index])?;
        return Ok(Some(value));
    }
    Ok(None)
}

pub(super) fn get_at_value(value: &QueryValue) -> RuntimeResult<i64> {
    match value {
        QueryValue::Scalar(v) => Ok((v * 1000_f64) as i64),
        QueryValue::InstantVector(v) => {
            if v.len() != 1 {
                let msg = format!(
                    "`@` modifier must return a single series; it returns {} series instead",
                    v.len()
                );
                return Err(RuntimeError::from(msg));
            }
            let ts = &v[0];
            if ts.values.is_empty() {
                let msg = "`@` modifier expression returned an empty value";
                // todo: different error type?
                return Err(RuntimeError::from(msg));
            }
            Ok((ts.values[0] * 1000_f64) as i64)
        }
        _ => {
            let msg =
                format!("cannot evaluate '{value}' as a timestamp in `@` modifier expression");
            // todo: different error type?
            Err(RuntimeError::from(msg))
        }
    }
}

pub(crate) fn exec_vector_vector(
    ctx: &Context,
    left: InstantVector,
    right: InstantVector,
    op: Operator,
    modifier: &Option<BinModifier>,
) -> RuntimeResult<QueryValue> {
    let is_tracing = ctx.trace_enabled();

    let span = if is_tracing {
        trace_span!("binary op", "op" = op.as_str(), series = field::Empty)
    } else {
        Span::none()
    }
    .entered();

    let mut bfa = BinaryOpFuncArg::new(left, op, right, modifier);

    let result = exec_binop(&mut bfa).map(QueryValue::InstantVector)?;

    if is_tracing {
        let series_count = series_len(&result);
        span.record("series", series_count);
    }

    Ok(result)
}

pub(super) fn expand_single_value(tss: &mut [Timeseries], ec: &EvalConfig) -> RuntimeResult<()> {
    // expand single-point tss to the original time range.
    let timestamps = ec.get_timestamps()?;
    for ts in tss.iter_mut() {
        ts.timestamps = Arc::clone(&timestamps);
        ts.values = vec![ts.values[0]; timestamps.len()];
    }
    Ok(())
}

pub(super) fn adjust_series_by_offset(rvs: &mut [Timeseries], offset: i64) {
    if offset != 0 && !rvs.is_empty() {
        // Make a copy of timestamps, since they may be used in other values.
        let src_timestamps = &rvs[0].timestamps;
        let dst_timestamps = src_timestamps.iter().map(|x| x + offset).collect();
        let shared = Arc::new(dst_timestamps);
        for ts in rvs.iter_mut() {
            ts.timestamps = Arc::clone(&shared);
        }
    }
}
