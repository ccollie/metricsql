use crate::functions::arg_parse::get_float_arg;
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerFloatArg};
use crate::{QueryValue, RuntimeError, RuntimeResult};

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

fn new_rollup_share_filter<F>(args: &[QueryValue], base_factory: F) -> RuntimeResult<RollupHandler>
where
    F: Fn(&[QueryValue]) -> RuntimeResult<RollupHandler> + 'static,
{
    let rf = base_factory(args)?;
    let f = move |rfa: &RollupFuncArg| -> f64 {
        let n = rf.eval(rfa);
        n / rfa.values.len() as f64
    };
    Ok(RollupHandler::General(Box::new(f)))
}

fn get_limit(args: &[QueryValue], func_name: &str, param_name: &str) -> RuntimeResult<f64> {
    get_float_arg(args, 0, None).map_err(|_| {
        RuntimeError::ArgumentError(format!(
            "expecting scalar as {param_name} arg to {func_name}()"
        ))
    })
}

fn count_filtered(values: &[f64], limit: f64, pred: fn(f64, f64) -> bool) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().filter(|v| pred(**v, limit)).count() as f64
}

fn less_or_equal(x: f64, y: f64) -> bool {
    x.le(&y)
}

fn greater(x: f64, y: f64) -> bool {
    x > y
}

fn equal(x: f64, y: f64) -> bool {
    x == y
}

fn not_equal(x: f64, y: f64) -> bool {
    x != y
}

pub(super) fn new_rollup_share_le(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    // todo: map_err so we can get the function name
    new_rollup_share_filter(args, new_rollup_count_le)
}

pub(super) fn new_rollup_share_gt(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    // todo: map_err so we can get the function name
    new_rollup_share_filter(args, new_rollup_count_gt)
}

pub(super) fn new_rollup_share_eq(args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    // todo: map_err so we can get the function name
    new_rollup_share_filter(args, new_rollup_count_eq)
}
