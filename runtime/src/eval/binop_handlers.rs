use lib::BuildNoHashHasher;
use std::collections::HashMap;
use xxhash_rust::xxh3::Xxh3;

use crate::eval::utils::remove_empty_series;
use metricsql::ast::BinaryExpr;
use metricsql::binaryop::{get_scalar_binop_handler, get_scalar_comparison_handler, BinopFunc};
use metricsql::common::{GroupModifier, JoinModifier, JoinModifierOp, Operator};

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::Timeseries;

pub(crate) struct BinaryOpFuncArg<'a> {
    be: &'a BinaryExpr,
    left: Vec<Timeseries>,
    right: Vec<Timeseries>,
}

impl<'a> BinaryOpFuncArg<'a> {
    pub fn new(left: Vec<Timeseries>, be: &'a BinaryExpr, right: Vec<Timeseries>) -> Self {
        Self { left, be, right }
    }
}

pub type BinaryOpFuncResult = RuntimeResult<Vec<Timeseries>>;

pub(crate) trait BinaryOpFn:
    Fn(&mut BinaryOpFuncArg) -> BinaryOpFuncResult + Send + Sync
{
}

impl<T> BinaryOpFn for T where T: Fn(&mut BinaryOpFuncArg) -> BinaryOpFuncResult + Send + Sync {}

type TimeseriesHashMap = HashMap<u64, Vec<Timeseries>, BuildNoHashHasher<u64>>;

macro_rules! make_binary_func {
    ($name: ident, $op: expr) => {
        fn $name(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            const FUNC: BinopFunc = get_scalar_binop_handler($op, false);
            binary_op_func_impl(FUNC, bfa)
        }
    };
}

make_binary_func!(binary_op_add, Operator::Add);
make_binary_func!(binary_op_atan2, Operator::Atan2);
make_binary_func!(binary_op_sub, Operator::Sub);
make_binary_func!(binary_op_mul, Operator::Mul);
make_binary_func!(binary_op_div, Operator::Div);
make_binary_func!(binary_op_mod, Operator::Mod);
make_binary_func!(binary_op_pow, Operator::Pow);

fn binary_op_comparison(bfa: &mut BinaryOpFuncArg) -> BinaryOpFuncResult {
    let bf = get_scalar_comparison_handler(bfa.be.op, bfa.be.bool_modifier);
    binary_op_func_impl(bf, bfa)
}

pub(super) fn exec_binop(bfa: &mut BinaryOpFuncArg) -> BinaryOpFuncResult {
    use Operator::*;
    match bfa.be.op {
        Add => binary_op_add(bfa),
        Atan2 => binary_op_atan2(bfa),
        Sub => binary_op_sub(bfa),
        Mul => binary_op_mul(bfa),
        Div => binary_op_div(bfa),
        Mod => binary_op_mod(bfa),
        Pow => binary_op_pow(bfa),
        // comparison operators
        Eql | NotEq | Gt | Gte | Lt | Lte => binary_op_comparison(bfa),
        And => binary_op_and(bfa),
        Or => binary_op_or(bfa),
        Unless => binary_op_unless(bfa),
        If => binary_op_if(bfa),
        IfNot => binary_op_if_not(bfa),
        Default => binary_op_default(bfa),
    }
}

fn binary_op_func_impl(bf: BinopFunc, bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if bfa.left.len() == 0 || bfa.right.len() == 0 {
        return Ok(vec![]);
    }

    // todo: should this also be applied to scalar/vector and vector/scalar?
    if bfa.be.op.is_comparison() {
        // Do not remove empty series for comparison operations,
        // since this may lead to missing result.
    } else {
        remove_empty_series(&mut bfa.left);
        remove_empty_series(&mut bfa.right);
    }

    let (left, right, mut dst) = adjust_binary_op_tags(bfa)?;
    if left.len() != right.len() || left.len() != dst.len() {
        return Err(RuntimeError::InvalidState(format!(
            "BUG: left.len() must match right.len() and dst.len(); got {} vs {} vs {}",
            left.len(),
            right.len(),
            dst.len()
        )));
    }

    for ((left_ts, right_ts), curr_dest) in left.iter().zip(right.iter()).zip(dst.iter_mut()) {
        let dst_len = curr_dest.values.len();
        if left_ts.len() != right_ts.len() || left_ts.len() != dst_len {
            let msg = format!("BUG: left_values.len() must match right_values.len() and dst_values.len(); got {} vs {} vs {}",
                              left_ts.len(), right_ts.len(), dst_len);
            return Err(RuntimeError::InvalidState(msg));
        }

        for ((left, right), dest) in left_ts
            .values
            .iter()
            .zip(right_ts.values.iter())
            .zip(curr_dest.values.iter_mut())
        {
            *dest = bf(*left, *right);
        }
    }

    // do not remove time series containing only NaNs, since then the `(foo op bar) default N`
    // won't work as expected if `(foo op bar)` results to NaN series.
    Ok(dst)
}

fn adjust_binary_op_tags(
    bfa: &mut BinaryOpFuncArg,
) -> RuntimeResult<(
    Vec<Timeseries>, // left
    Vec<Timeseries>, // right
    Vec<Timeseries>, // dest
)> {
    // `vector op vector` or `a op {on|ignoring} {group_left|group_right} b`
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);

    let mut rvs_left: Vec<Timeseries> = Vec::with_capacity(1);
    let mut rvs_right: Vec<Timeseries> = Vec::with_capacity(1);

    for (k, tss_left) in m_left.iter_mut() {
        let mut tss_right = m_right.remove(k).unwrap_or(vec![]);
        if tss_right.len() == 0 {
            continue;
        }

        match &bfa.be.join_modifier {
            Some(modifier) => match modifier.op {
                JoinModifierOp::GroupLeft => group_join(
                    "right",
                    &bfa.be,
                    &mut rvs_left,
                    &mut rvs_right,
                    tss_left,
                    &mut tss_right,
                )?,
                JoinModifierOp::GroupRight => group_join(
                    "left",
                    &bfa.be,
                    &mut rvs_right,
                    &mut rvs_left,
                    &mut tss_right,
                    tss_left,
                )?,
            },
            None => {
                ensure_single_timeseries("left", &bfa.be, tss_left)?;
                ensure_single_timeseries("right", &bfa.be, &mut tss_right)?;
                let mut ts_left = &mut tss_left[0];
                reset_metric_group_if_required(&bfa.be, &mut ts_left);

                if let Some(modifier) = &bfa.be.group_modifier {
                    ts_left.metric_name.update_tags_by_group_modifier(modifier);
                }

                rvs_left.push(std::mem::take(&mut ts_left));
                rvs_right.push(tss_right.remove(0))
            }
        }
    }

    // todo: how to avoid this clone?
    let dst = if is_group_right(&bfa.be) {
        rvs_right.clone()
    } else {
        rvs_left.clone()
    };

    return Ok((rvs_left, rvs_right, dst));
}

fn ensure_single_timeseries(
    side: &str,
    be: &BinaryExpr,
    tss: &mut Vec<Timeseries>,
) -> RuntimeResult<()> {
    if tss.len() == 0 {
        return Err(RuntimeError::General(
            "BUG: tss must contain at least one value".to_string(),
        ));
    }

    let mut acc = tss.pop().unwrap();

    for ts in tss.iter() {
        if !merge_non_overlapping_timeseries(&mut acc, &ts) {
            let msg = format!(
                "duplicate time series on the {} side of {} {}: {} and {}",
                side,
                be.op,
                group_modifier_to_string(&be.group_modifier),
                acc.metric_name,
                ts.metric_name
            );

            return Err(RuntimeError::from(msg));
        }
    }

    Ok(())
}

fn group_join(
    single_timeseries_side: &str,
    be: &BinaryExpr,
    rvs_left: &mut Vec<Timeseries>,
    rvs_right: &mut Vec<Timeseries>,
    tss_left: &mut Vec<Timeseries>,
    tss_right: &mut Vec<Timeseries>,
) -> RuntimeResult<()> {
    let empty_labels: Vec<String> = vec![];

    let join_tags = if let Some(be_) = &be.join_modifier {
        &be_.labels
    } else {
        &empty_labels
    };

    struct TsPair {
        left: Timeseries,
        right: Timeseries,
    }

    let mut hasher = Xxh3::new();

    let mut map: HashMap<u64, TsPair, BuildNoHashHasher<u64>> =
        HashMap::with_capacity_and_hasher(tss_left.len(), BuildNoHashHasher::default());

    let should_reset_name = should_reset_metric_group(&be.op, be.bool_modifier);

    for ts_left in tss_left.into_iter() {
        if should_reset_name {
            ts_left.metric_name.reset_metric_group();
        }

        if tss_right.len() == 1 {
            let mut right = tss_right.remove(0);
            // Easy case - right part contains only a single matching time series.
            ts_left
                .metric_name
                .set_tags(join_tags, &mut right.metric_name);
            rvs_left.push(std::mem::take(ts_left));
            rvs_right.push(right);
            continue;
        }

        // Hard case - right part contains multiple matching time series.
        // Verify it doesn't result in duplicate MetricName values after adding missing tags.
        map.clear();

        for ts_right in tss_right.into_iter() {
            // todo: Question - do we need to copy? I dont think tss_left and tss_right
            // are used after we exit this function
            let mut ts_copy = Timeseries::copy(ts_left);
            ts_copy
                .metric_name
                .set_tags(join_tags, &mut ts_right.metric_name);

            let key = ts_copy
                .metric_name
                .get_hash_by_group_modifier(&mut hasher, &be.group_modifier);

            // todo: check specifically for error
            // todo: Entry API is not ideal here, we don't need to copy the key
            match map.get_mut(&key) {
                None => {
                    map.insert(
                        key,
                        TsPair {
                            left: ts_copy,
                            right: std::mem::take(ts_right),
                        },
                    );
                    continue;
                }
                Some(pair) => {
                    // todo(perf): may not need to copy
                    // Try merging pair.right with ts_right if they don't overlap.
                    if !merge_non_overlapping_timeseries(&mut pair.right, ts_right) {
                        let err = format!(
                            "duplicate time series on the {} side of `{} {} {}`: {} and {}",
                            single_timeseries_side,
                            be.op,
                            group_modifier_to_string(&be.group_modifier),
                            join_modifier_to_string(&be.join_modifier),
                            pair.right.metric_name,
                            ts_right.metric_name
                        );

                        return Err(RuntimeError::from(err));
                    }
                }
            }
        }

        for (_, pair) in map.iter_mut() {
            rvs_left.push(std::mem::take(&mut pair.left));
            rvs_right.push(std::mem::take(&mut pair.right));
        }
    }

    Ok(())
}

fn group_modifier_to_string(modifier: &Option<GroupModifier>) -> String {
    match &modifier {
        Some(value) => {
            format!("{}", value)
        }
        None => "None".to_string(),
    }
}

fn join_modifier_to_string(modifier: &Option<JoinModifier>) -> String {
    match &modifier {
        Some(value) => {
            format!("{}", value)
        }
        None => "None".to_string(),
    }
}

pub fn merge_non_overlapping_timeseries(dst: &mut Timeseries, src: &Timeseries) -> bool {
    // Verify whether the time series can be merged.
    let mut overlaps = 0;

    for (src_val, dst_val) in src.values.iter().zip(dst.values.iter()) {
        if src_val.is_nan() {
            continue;
        }
        if !dst_val.is_nan() {
            overlaps += 1;
            // Allow up to two overlapping data points, which can appear due to staleness algorithm,
            // which can add a few data points in the end of time series.
            if overlaps > 2 {
                return false;
            }
        }
    }

    // do not merge time series with too small number of data points.
    // This can be the case during evaluation of instant queries (alerting or recording rules).
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1141
    if src.values.len() <= 2 && dst.values.len() <= 2 {
        return false;
    }
    // Time series can be merged. Merge them.
    for (src_val, dst_val) in src.values.iter().zip(dst.values.iter_mut()) {
        if !src_val.is_nan() {
            *dst_val = *src_val
        }
    }
    true
}

fn binary_op_if(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());

    for (k, tss_left) in m_left.iter_mut() {
        if let Some(tss_right) = series_by_key(&m_right, k) {
            add_right_nans_to_left(tss_left, tss_right);
            rvs.extend(tss_left.drain(..))
        }
    }

    Ok(rvs)
}

fn binary_op_and(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if bfa.left.is_empty() || bfa.right.is_empty() {
        return Ok(vec![]); // Short-circuit: AND with nothing is nothing.
    }

    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(std::cmp::min(m_left.len(), m_right.len()));

    for (k, tss_right) in m_right.into_iter() {
        if let Some(tss_left) = m_left.get_mut(&k) {
            // Add gaps to tss_left if there are gaps at tss_right.
            add_right_nans_to_left(tss_left, &tss_right);
            rvs.extend(tss_left.drain(..));
        }
    }

    Ok(rvs)
}

fn add_right_nans_to_left(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for ts_left in tss_left.iter_mut() {
        for i in 0..ts_left.values.len() {
            let mut has_value = false;
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    has_value = true;
                    break;
                }
            }
            if !has_value {
                ts_left.values[i] = f64::NAN
            }
        }
    }

    remove_empty_series(tss_left)
}

fn binary_op_default(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);

    if m_left.len() == 0 {
        // todo: see if we can make this more efficient
        let items = m_right.into_values().flatten().collect();
        return Ok(items);
    }

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());
    for (k, mut tss_left) in m_left.iter_mut() {
        if let Some(tss_right) = series_by_key(&m_right, &k) {
            fill_left_nans_with_right_values(tss_left, tss_right);
        }
        rvs.append(&mut tss_left);
    }

    Ok(rvs)
}

pub(super) fn should_reset_metric_group(op: &Operator, is_bool: bool) -> bool {
    if op.is_comparison() && !is_bool {
        // do not reset MetricGroup for non-boolean `compare` binary ops like Prometheus does.
        return false;
    }
    match op {
        Operator::Default | Operator::If | Operator::IfNot => false,
        _ => true,
    }
}

pub(super) fn reset_metric_group_if_required(be: &BinaryExpr, ts: &mut Timeseries) {
    if should_reset_metric_group(&be.op, be.bool_modifier) {
        ts.metric_name.reset_metric_group()
    }
}

fn binary_op_or(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if bfa.left.is_empty() {
        // Short-circuit.
        return Ok(std::mem::take(&mut bfa.right));
    }

    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut right: Vec<Timeseries> = Vec::with_capacity(m_right.len());

    for (k, tss_right) in m_right.iter_mut() {
        match m_left.get_mut(k) {
            None => {
                right.extend_from_slice(&tss_right);
            }
            Some(tss_left) => {
                fill_left_nans_with_right_values(tss_left, tss_right);
            }
        }
    }

    let mut rvs = m_left.into_values().flatten().collect::<Vec<_>>();
    rvs.append(&mut right);

    Ok(rvs)
}

#[inline]
/// Fill gaps in tss_left with values from tss_right as Prometheus does.
/// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/552
fn fill_left_nans_with_right_values(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for ts_left in tss_left.iter_mut() {
        for (i, left_value) in ts_left.values.iter_mut().enumerate() {
            if !left_value.is_nan() {
                continue;
            }
            for ts_right in tss_right.iter() {
                let v_right = ts_right.values[i];
                if !v_right.is_nan() {
                    *left_value = v_right;
                    break;
                }
            }
        }
    }
}

fn binary_op_unless(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    // Short-circuit: empty rhs means we will return everything in lhs;
    // empty lhs means we will return empty - don't need to build a map.
    if bfa.right.is_empty() {
        return Ok(std::mem::take(&mut bfa.left));
    } else if bfa.left.is_empty() {
        return Ok(vec![]);
    }

    let (m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        match m_right.get(&k) {
            None => rvs.append(&mut tss_left),
            Some(tss_right) => {
                // Add gaps to tss_left if the are no gaps at tss_right.
                add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
                rvs.append(&mut tss_left);
            }
        }
    }

    Ok(rvs)
}

fn binary_op_if_not(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        match series_by_key(&m_right, &k) {
            None => {
                rvs.extend(tss_left.into_iter());
            }
            Some(tss_right) => {
                add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
                rvs.extend(tss_left.into_iter());
            }
        }
    }

    Ok(rvs)
}

fn add_left_nans_if_no_right_nans(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    // todo: zip

    for ts_left in tss_left.iter_mut() {
        for (i, left_value) in ts_left.values.iter_mut().enumerate() {
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    *left_value = f64::NAN;
                    break;
                }
            }
        }
    }

    remove_empty_series(tss_left);
}

fn get_tags_map_with_fn<'a>(
    hasher: &mut Xxh3,
    arg: &mut Vec<Timeseries>,
    modifier: &Option<GroupModifier>,
) -> TimeseriesHashMap {
    let mut m: TimeseriesHashMap =
        HashMap::with_capacity_and_hasher(arg.len(), BuildNoHashHasher::default());

    for ts in arg.into_iter() {
        let key = ts.metric_name.get_hash_by_group_modifier(hasher, modifier);
        m.entry(key).or_insert(vec![]).push(std::mem::take(ts));
    }

    return m;
}

fn create_timeseries_map_by_tag_set(
    bfa: &mut BinaryOpFuncArg,
) -> (TimeseriesHashMap, TimeseriesHashMap) {
    let mut hasher = Xxh3::new();

    let modifier = &bfa.be.group_modifier;
    let m_left = get_tags_map_with_fn(&mut hasher, &mut bfa.left, modifier);
    let m_right = get_tags_map_with_fn(&mut hasher, &mut bfa.right, modifier);
    return (m_left, m_right);
}

pub(super) fn is_scalar(arg: &[Timeseries]) -> bool {
    if arg.len() != 1 {
        return false;
    }
    let mn = &arg[0].metric_name;
    mn.tags.is_empty() && mn.metric_group.is_empty()
}

fn series_by_key<'a>(m: &'a TimeseriesHashMap, key: &u64) -> Option<&'a Vec<Timeseries>> {
    match m.get(key) {
        Some(v) => Some(v),
        None => {
            if m.len() != 1 {
                return None;
            }
            for tss in m.values() {
                if is_scalar(tss) {
                    return Some(&tss);
                }
                return None;
            }
            return None;
        }
    }
}

fn is_group_right(bfa: &BinaryExpr) -> bool {
    match &bfa.join_modifier {
        Some(modifier) => modifier.op == JoinModifierOp::GroupRight,
        None => false,
    }
}
