use crate::functions::arg_parse::get_float_arg;
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerFloatArg};
use crate::{QueryValue, RuntimeError, RuntimeResult};

#[inline]
fn less_or_equal(x: f64, y: f64) -> bool {
    x.le(&y)
}

#[inline]
fn greater(x: f64, y: f64) -> bool {
    x > y
}

#[inline]
fn equal(x: f64, y: f64) -> bool {
    x == y
}

#[inline]
fn not_equal(x: f64, y: f64) -> bool {
    x != y
}

fn count_filtered(values: &[f64], limit: f64, pred: fn(f64, f64) -> bool) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let res = values.iter().filter(|v| pred(**v, limit)).count() as f64;
    res
}

fn share_filtered(values: &[f64], limit: f64, pred: fn(f64, f64) -> bool) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let len = values.len();
    let n = count_filtered(values, limit, pred);
    n / len as f64
}

fn get_limit(args: &[QueryValue], func_name: &str, param_name: &str) -> RuntimeResult<f64> {
    get_float_arg(args, 0, None).map_err(|_| {
        RuntimeError::ArgumentError(format!(
            "expecting scalar as {param_name} arg to {func_name}()"
        ))
    })
}

macro_rules! make_count_fn {
    ( $name: ident, $func_name: tt, $param_name: tt, $predicate_fn: expr ) => {
        pub(super) fn $name(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            let limit = get_limit(args, $func_name, $param_name)?;
            let handler =
                RollupHandlerFloatArg::new(limit, |rfa: &RollupFuncArg, limit: &f64| -> f64 {
                    count_filtered(rfa.values, *limit, $predicate_fn)
                });
            Ok(RollupHandler::FloatArg(handler))
        }
    };
}

make_count_fn!(
    new_rollup_count_le,
    "count_le_over_time",
    "le",
    less_or_equal
);
make_count_fn!(new_rollup_count_gt, "count_gt_over_time", "gt", greater);
make_count_fn!(new_rollup_count_eq, "count_eq_over_time", "eq", equal);
make_count_fn!(new_rollup_count_ne, "count_ne_over_time", "ne", not_equal);

macro_rules! make_share_fn {
    ( $name: ident, $func_name: tt, $param_name: tt, $predicate_fn: expr ) => {
        pub(super) fn $name(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
            let limit = get_limit(args, $func_name, $param_name)?;
            let handler =
                RollupHandlerFloatArg::new(limit, |rfa: &RollupFuncArg, limit: &f64| -> f64 {
                    share_filtered(rfa.values, *limit, $predicate_fn)
                });
            Ok(RollupHandler::FloatArg(handler))
        }
    };
}

make_share_fn!(
    new_rollup_share_le,
    "share_le_over_time",
    "le",
    less_or_equal
);
make_share_fn!(new_rollup_share_gt, "share_gt_over_time", "gt", greater);
make_share_fn!(new_rollup_share_eq, "share_eq_over_time", "eq", equal);
