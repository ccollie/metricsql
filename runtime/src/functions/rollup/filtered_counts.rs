use crate::functions::arg_parse::get_float_arg;
use crate::functions::rollup::{RollupFuncArg, RollupHandler, RollupHandlerEnum};
use crate::{EvalConfig, QueryValue, RuntimeError, RuntimeResult};

macro_rules! make_count_fn {
    ( $name: ident, $func_name: tt, $param_name: tt, $count_fn: expr ) => {
        pub(super) fn $name(
            args: &Vec<QueryValue>,
            _ec: &EvalConfig,
        ) -> RuntimeResult<RollupHandlerEnum> {
            let limit = get_limit(args, $func_name, $param_name)?;

            println!("parsed limit: {}", limit);

            let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
                println!("limit: {}", limit);
                $count_fn(&rfa.values, limit)
            });

            Ok(RollupHandlerEnum::General(f))
        }
    };
}

make_count_fn!(new_rollup_count_le, "count_le_over_time", "le", count_le);
make_count_fn!(new_rollup_count_gt, "count_gt_over_time", "gt", count_gt);
make_count_fn!(new_rollup_count_ge, "count_ge_over_time", "ge", count_ge);
make_count_fn!(new_rollup_count_eq, "count_eq_over_time", "eq", count_eq);
make_count_fn!(new_rollup_count_ne, "count_ne_over_time", "ne", count_ne);

fn new_rollup_filtered_count(
    args: &Vec<QueryValue>,
    func_name: &str,
    pred: fn(f64, f64) -> bool,
) -> RuntimeResult<RollupHandlerEnum> {
    let limit = get_limit(args, func_name, "limit")?;
    let pred = pred;

    let f = move |rfa: &mut RollupFuncArg| -> f64 { count_filtered(&rfa.values, limit, pred) };

    Ok(RollupHandlerEnum::General(Box::new(f)))
}

fn get_limit(args: &Vec<QueryValue>, func_name: &str, param_name: &str) -> RuntimeResult<f64> {
    get_float_arg(args, 1, None).map_err(|_| {
        RuntimeError::ArgumentError(format!(
            "expecting scalar as {param_name} arg to {func_name}()"
        ))
    })
}

fn count_filtered(values: &Vec<f64>, limit: f64, pred: fn(f64, f64) -> bool) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mut n = 0;
    for v in values.iter() {
        if pred(*v, limit) {
            n += 1;
        }
    }

    n as f64
}

fn count_le(values: &Vec<f64>, limit: f64) -> f64 {
    count_filtered(values, limit, less_or_equal)
}

fn count_ge(values: &Vec<f64>, limit: f64) -> f64 {
    count_filtered(values, limit, greater_or_equal)
}

fn count_gt(values: &Vec<f64>, limit: f64) -> f64 {
    count_filtered(values, limit, greater)
}

fn count_eq(values: &Vec<f64>, limit: f64) -> f64 {
    count_filtered(values, limit, equal)
}

fn count_ne(values: &Vec<f64>, limit: f64) -> f64 {
    count_filtered(values, limit, not_equal)
}

fn less(x: f64, y: f64) -> bool {
    x < y
}

fn less_or_equal(x: f64, y: f64) -> bool {
    x.le(&y)
}

fn greater(x: f64, y: f64) -> bool {
    x > y
}

fn greater_or_equal(x: f64, y: f64) -> bool {
    x >= y
}

fn equal(x: f64, y: f64) -> bool {
    x == y
}

fn not_equal(x: f64, y: f64) -> bool {
    x != y
}

fn new_rollup_share_filter<F>(
    args: &Vec<QueryValue>,
    ec: &EvalConfig,
    base_factory: F,
) -> RuntimeResult<RollupHandlerEnum>
where
    F: Fn(&Vec<QueryValue>, &EvalConfig) -> RuntimeResult<RollupHandlerEnum> + 'static,
{
    let rf = base_factory(args, ec)?;
    let f = move |rfa: &mut RollupFuncArg| -> f64 {
        let n = rf.eval(rfa);
        return n / rfa.values.len() as f64;
    };

    Ok(RollupHandlerEnum::General(Box::new(f)))
}

pub(super) fn new_rollup_share_le(
    args: &Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
    // todo: map_err so we can get the function name
    return new_rollup_share_filter(args, ec, new_rollup_count_le);
}

pub(super) fn new_rollup_share_gt(
    args: &Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
    // todo: map_err so we can get the function name
    return new_rollup_share_filter(args, ec, new_rollup_count_gt);
}
