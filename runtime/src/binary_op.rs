use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::ops::DerefMut;

use lib::get_pooled_buffer;
use metricsql::ast::{
    BinaryOp,
    BinaryOpExpr,
    Expression,
    GroupModifier,
    GroupModifierOp,
    JoinModifier,
    JoinModifierOp
};
use metricsql::parser::visit_all;

use crate::exec::remove_empty_series;
use crate::get_pooled_metric_name;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::timeseries::Timeseries;

pub(crate) struct BinaryOpFuncArg {
    pub be: BinaryOpExpr,
    pub left: Vec<Timeseries>,
    pub right: Vec<Timeseries>,
}

pub(self) trait BinaryOpFn: FnMut(&mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

impl<T> BinaryOpFn for T where T: Fn(&mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

type TimeseriesHashMap = HashMap<String, Vec<Timeseries>>;

macro_rules! make_comp_func {
    ( $name:ident, $af: expr ) => {
        static $name: Box<dyn BinaryOpFn> = Box::new(new_binary_op_cmp_func($af));
    };
}

macro_rules! make_arith_func {
    ( $name:ident, $af: expr ) => {
        static $name: Box<dyn BinaryOpFn> = Box::new(new_binary_op_arith_func($af));
    };
}

make_arith_func!(BINARY_OP_ADD, metricsql::binaryop::plus);
make_arith_func!(BINARY_OP_SUB, metricsql::binaryop::minus);
make_arith_func!(BINARY_OP_MUL, metricsql::binaryop::mul);
make_arith_func!(BINARY_OP_DIV, metricsql::binaryop::div);
make_arith_func!(BINARY_OP_MOD, metricsql::binaryop::mod_);
make_arith_func!(BINARY_OP_POW, metricsql::binaryop::pow);
make_arith_func!(BINARY_OP_ATAN2, metricsql::binaryop::atan2);

make_comp_func!(BINARY_OP_EQ, metricsql::binaryop::eq);
make_comp_func!(BINARY_OP_NEQ, metricsql::binaryop::neq);
make_comp_func!(BINARY_OP_GT, metricsql::binaryop::gt);
make_comp_func!(BINARY_OP_GTE, metricsql::binaryop::gte);
make_comp_func!(BINARY_OP_LT, metricsql::binaryop::lt);
make_comp_func!(BINARY_OP_LTE, metricsql::binaryop::lte);


pub(crate) fn exec_binop(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    match bfa.be.op {
        BinaryOp::Add => BINARY_OP_ADD(bfa),
        BinaryOp::Sub => BINARY_OP_SUB(bfa),
        BinaryOp::Mul => BINARY_OP_MUL(bfa),
        BinaryOp::Div => BINARY_OP_DIV(bfa),
        BinaryOp::Mod => BINARY_OP_MOD(bfa),
        BinaryOp::Pow => BINARY_OP_POW(bfa),

        // See https://github.com/prometheus/prometheus/pull/9248
        BinaryOp::Atan2 => BINARY_OP_ATAN2(bfa),

        // cmp ops
        BinaryOp::Eql => BINARY_OP_EQ(bfa),
        BinaryOp::Neq => BINARY_OP_NEQ(bfa),
        BinaryOp::Gt => BINARY_OP_GT(bfa),
        BinaryOp::Gte => BINARY_OP_GTE(bfa),
        BinaryOp::Lt => BINARY_OP_LT(bfa),
        BinaryOp::Lte => BINARY_OP_LTE(bfa),

        // logical set ops
        BinaryOp::And => binary_op_and(bfa),
        BinaryOp::Or => binary_op_or(bfa),
        BinaryOp::Unless => binary_op_unless(bfa),

        // New ops
        BinaryOp::If => binary_op_if(bfa),
        BinaryOp::IfNot => binary_op_if_not(bfa),
        BinaryOp::Default => binary_op_default(bfa)
    }
}

const fn new_binary_op_cmp_func(cf: fn(left: f64, right: f64) -> bool) -> impl BinaryOpFn {
    let cfe = move |left: f64, right: f64, is_bool: bool| -> f64 {
        if !is_bool {
            if cf(left, right) {
                return left
            }
            return f64::NAN
        }
        if left.is_nan() {
            return f64::NAN
        }
        if cf(left, right) {
            return 1.0
        }
        return 0.0
    };

    new_binary_op_func(cfe)
}

const fn new_binary_op_arith_func(af: fn(left: f64, right: f64) -> f64) -> impl BinaryOpFn {

    let afe = move |left: f64, right: f64, is_bool: bool| -> f64 {
        return af(left, right)
    };

    new_binary_op_func(afe)
}

trait BinopClosureFn: Fn(f64, f64, bool) -> f64 + Send + Sync {}
impl<T> BinopClosureFn for T where T: Fn(f64, f64, bool) -> f64 + Send + Sync {}

// Possibly inline this or make it a macro
const fn new_binary_op_func(bf: impl BinopClosureFn) -> impl BinaryOpFn {
    move |bfa: &mut BinaryOpFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let op = bfa.be.op;

        if op.is_comparison() {
            // Do not remove empty series for comparison operations,
            // since this may lead to missing result.
        } else {
            remove_empty_series(&mut bfa.left);
            remove_empty_series(&mut bfa.right);
        }

        if bfa.left.len() == 0 || bfa.right.len() == 0 {
            return Ok(vec![]);
        }

        let (left, right, mut dst) = adjust_binary_op_tags(bfa)?;
        if left.len() != right.len() || left.len() != dst.len() {
            panic!("BUG: left.len() must match right.len() and dst.len(); got {} vs {} vs {}", left.len(), right.len(), dst.len());
        }

        let is_bool = bfa.be.bool_modifier;
        for (i, tsLeft) in bfa.left.iter_mut().enumerate() {
            let left_values = &tsLeft.values;
            let right_values = &right[i].values;
            let dst_len = dst[i].values.len();

            if left_values.len() != right_values.len() || left_values.len() != dst_len {
                panic!("BUG: left_values.len() must match right_values.len() and len(dst_values); got {} vs {} vs {}",
                                  left_values.len(), right_values.len(), dst_len);
            }
            for (j, a) in left_values.iter().enumerate() {
                let b = right_values[j];
                dst[i].values[j] = bf(*a, b, is_bool)
            }
        }
        // do not remove time series containing only NaNs, since then the `(foo op bar) default N`
        // won't work as expected if `(foo op bar)` results to NaN series.
        Ok(dst)
    }
}

fn adjust_binary_op_tags(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<(Vec<Timeseries>, Vec<Timeseries>, Vec<Timeseries>)> {

    if bfa.be.group_modifier.is_none() && bfa.be.join_modifier.is_none() {
        if is_scalar(&bfa.left) {
            // Fast path: `scalar op vector`
            let mut rvs_left: Vec<Timeseries> = Vec::with_capacity(bfa.right.len());
            let ts_left = bfa.left.remove(0).borrow_mut();

            for mut ts_right in bfa.right.into_iter() {
                reset_metric_group_if_required(&bfa.be, &mut ts_right);
                rvs_left.push(std::mem::take(ts_left)); // should be clone ????
            }
            return Ok((rvs_left, bfa.right, bfa.right))
        }

        if is_scalar(&bfa.right) {
            // Fast path: `vector op scalar`
            let mut rvs_right: Vec<Timeseries> = Vec::with_capacity(bfa.left.len());
            let mut ts_right = bfa.right.remove(0).borrow_mut();
            for tsLeft in bfa.left.iter_mut() {
                reset_metric_group_if_required(&bfa.be, tsLeft);
                rvs_right.push(std::mem::take(ts_right));
            }
            return Ok((bfa.left, rvs_right, bfa.left))
        }
    }

    // Slow path: `vector op vector` or `a op {on|ignoring} {group_left|group_right} b`
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);

    let (group_op, group_tags) = get_modifier_or_default(&mut bfa.be);

    let mut rvs_left: Vec<Timeseries> = Vec::with_capacity(1);
    let mut rvs_right: Vec<Timeseries> = Vec::with_capacity(1);

    for (k, mut tss_left) in m_left.iter_mut() {
        let mut tss_right = m_right.remove(k);
        if tss_right.is_none() {
            continue;
        }

        let mut tss_right = tss_right.unwrap();
        if tss_right.len() == 0 {
            continue
        }

        match &bfa.be.join_modifier {
            Some(modifier) => {
               match modifier.op {
                   JoinModifierOp::GroupLeft => {
                       group_join(
                           "right",
                           &bfa.be,
                           &mut rvs_left,
                           &mut rvs_right,
                           tss_left,
                           &mut tss_right)?
                   },
                   JoinModifierOp::GroupRight => {
                       group_join("left",
                                  &bfa.be,
                                  &mut rvs_right,
                                  &mut rvs_left,
                                  &mut tss_right,
                                  tss_left)?
                   }
               }
            },
            None => {
                ensure_single_timeseries("left", &bfa.be, tss_left)?;
                ensure_single_timeseries("right", &bfa.be, &mut tss_right)?;
                let mut ts_left = &mut tss_left[0];
                reset_metric_group_if_required(&bfa.be, &mut ts_left);
                match group_op {
                    GroupModifierOp::On => {
                        ts_left.metric_name.remove_tags_on(group_tags)
                    },
                    GroupModifierOp::Ignoring => {
                        ts_left.metric_name.remove_tags_ignoring(group_tags)
                    }
                }

                rvs_left.push(std::mem::take(&mut ts_left));
                rvs_right.push(tss_right.remove(0))
            }
        }
    }

    let mut dst = if is_group_right(&bfa.be) { rvs_right } else { rvs_left };

    return Ok((rvs_left, rvs_right, dst))
}

fn ensure_single_timeseries(side: &str, be: &BinaryOpExpr, tss: &mut Vec<Timeseries>) -> RuntimeResult<()> {
    if tss.len() == 0 {
        return Err(RuntimeError::General("BUG: tss must contain at least one value".to_string()));
    }
    while tss.len() > 1 {
        let last = tss.remove(tss.len() - 1);
        if !merge_non_overlapping_timeseries(&mut tss[0], &last) {
            let msg = format!("duplicate time series on the {} side of {} {}: {} and {}",
                              side,
                              be.op,
                              group_modifier_to_string(&be.group_modifier),
                              tss[0].metric_name,
                              last.metric_name);

            return Err(RuntimeError::from(msg));
        }
    }

    Ok(())
}

fn group_join<'a>(
    single_timeseries_side: &str,
    be: &BinaryOpExpr,
    rvs_left: &'a mut Vec<Timeseries>,
    rvs_right: &'a mut Vec<Timeseries>,
    tss_left: &'a mut Vec<Timeseries>,
    tss_right: &'a mut Vec<Timeseries>) -> RuntimeResult<()> {

    let join_tags = if let Some(be_) = &be.join_modifier {
        &be_.labels
    } else {
        &vec![]
    };

    struct TsPair<'a> {
        left: &'a Timeseries,
        right: &'a Timeseries
    }

    let mut m: HashMap<String, TsPair> = HashMap::with_capacity(rvs_left.len());
    let mut bb = get_pooled_buffer(512);

    for tsLeft in tss_left.into_iter() {
        reset_metric_group_if_required(be, tsLeft);
        if tss_right.len() == 1 {
            // Easy case - right part contains only a single matching time series.
            tsLeft.metric_name.set_tags(join_tags, &tss_right[0].metric_name);
            rvs_left.push(std::mem::take(tsLeft));
            rvs_right.push(tss_right.remove(0));
            continue
        }

        // Hard case - right part contains multiple matching time series.
        // Verify it doesn't result in duplicate MetricName values after adding missing tags.
        m.clear();

        for tsRight in tss_right {
            // todo: Question - do we need to copy? I dont think tss_left and tss_right
            // are used anymore when we exit this function
            let mut ts_copy = Timeseries::copy(tsLeft);
            ts_copy.metric_name.set_tags(join_tags, &tsRight.metric_name);
            let key = ts_copy.metric_name.marshal_to_string(bb.deref_mut()).to_string();

            // todo: check specifically for error
            match m.get_mut(&key) {
                None => {
                    m.insert(key, TsPair {
                        left:  &ts_copy,
                        right: tsRight,
                    });
                    continue
                },
                Some(mut pair) => {
                    // Todo: may not need to copy
                    // Try merging pair.right with tsRight if they don't overlap.
                    let mut tmp: Timeseries = Timeseries::copy(pair.right);
                    if !merge_non_overlapping_timeseries(&mut tmp, tsRight) {
                        let err = format!("duplicate time series on the {} side of `{} {} {}`: {} and {}",
                                          single_timeseries_side, be.op,
                                          group_modifier_to_string(&be.group_modifier),
                                          join_modifier_to_string(&be.join_modifier),
                                          tmp.metric_name,
                                          tsRight.metric_name);

                        return Err(RuntimeError::from(err));
                    }

                    pair.right = &tmp
                }
            }

            bb.clear();
        }

        for pair in m.into_values() {
            rvs_left.push(std::mem::take(&mut pair.left));
            rvs_right.push(*pair.right);
        }
    }

    Ok(())
}

fn group_modifier_to_string(modifier: &Option<GroupModifier>) -> String {
    match &modifier {
        Some(value) => {
            format!("{}", value)
        },
        None => "None".to_string()
    }
}

fn join_modifier_to_string(modifier: &Option<JoinModifier>) -> String {
    match &modifier {
        Some(value) => {
            format!("{}", value)
        },
        None => "None".to_string()
    }
}

pub fn merge_non_overlapping_timeseries(dst: &mut Timeseries, src: &Timeseries) -> bool {
    // Verify whether the time series can be merged.
    let mut overlaps = 0;

    for (i, v) in src.values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if !dst.values[i].is_nan() {
            overlaps += 1;
        }
    }

    // Allow up to two overlapping datapoints, which can appear due to staleness algorithm,
    // which can add a few datapoints in the end of time series.
    if overlaps > 2 {
        return false
    }
    // do not merge time series with too small number of datapoints.
    // This can be the case during evaluation of instant queries (alerting or recording rules).
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1141
    if src.values.len() <= 2 && dst.values.len() <= 2 {
        return false
    }
    // Time series can be merged. Merge them.
    for (i, v) in src.values.iter().enumerate() {
        if v.is_nan() {
            continue
        }
        dst.values[i] = *v;
    }
    true
}

fn binary_op_if(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len() );

    for (k, tssLeft) in m_left.into_iter() {
        match series_by_key(m_right, k) {
            None => continue,
            Some(tss_right) => {
                add_right_nans_to_left(tssLeft, tss_right);
                rvs.extend(tssLeft.drain(..))
            }
        }
    }

    Ok(rvs)
}

fn binary_op_and(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( std::cmp::min(m_left.len(), m_right.len() ) );

    for (k, tss_right) in m_right.into_iter() {
        match m_left.get_mut(k) {
            None => continue,
            Some(tss_left) => {
                // Add gaps to tss_left if there are gaps at tssRight.
                add_right_nans_to_left(tss_left, &tss_right);
                rvs.extend(tss_left.drain(0..));
            }
        }
    }

    Ok(rvs)
}

fn add_right_nans_to_left(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for ts_left in tss_left {
        for i in 0 .. ts_left.values.len() {
            let mut has_value = false;
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    has_value = true;
                    break
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
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);

    if m_left.len() == 0 {
        // see if we can make this more efficient
        let items = m_right.into_values().flatten().collect();
        return Ok(items)
    }
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len() );
    for (k, mut tss_left) in m_left.iter_mut() {
        match series_by_key(m_right, &k) {
            None => {},
            Some(tss_right) => {
               fill_left_nans_with_right_values(tss_left, tss_right);
            }
        }
        rvs.append(&mut tss_left);
    }

    Ok(rvs)
}


fn reset_metric_group_if_required(be: &BinaryOpExpr, ts: &mut Timeseries) {
    if be.op.is_comparison() && !be.bool_modifier {
        // do not reset MetricGroup for non-boolean `compare` binary ops like Prometheus does.
        return
    }

    if be.op == BinaryOp::Default || be.op == BinaryOp::If || be.op == BinaryOp::IfNot {
        // do not reset MetricGroup for these ops.
        return
    }

    ts.metric_name.reset_metric_group()
}

fn binary_op_or(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( std::cmp::max(m_left.len(), m_right.len() ) );

    rvs.append(&mut m_left.into_values().flatten().collect::<Vec<_>>());

    for (k, tss_right) in m_right.iter_mut() {
        match m_left.get(k) {
            None => {
                rvs.extend(tss_right.drain(0..));
            },
            Some(mut tss_left) => {
                fill_left_nans_with_right_values(&mut tss_left, tss_right);
            }
        }
    }

    Ok(rvs)
}

#[inline]
fn fill_left_nans_with_right_values(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    // Fill gaps in tss_left with values from tss_right as Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/552
    for tsLeft in tss_left.iter_mut() {
        for (i, v) in tsLeft.values.iter().enumerate() {
            if !v.is_nan() {
                continue
            }
            for ts_right in tss_right {
                let v_right = ts_right.values[i];
                if !v_right.is_nan() {
                    tsLeft.values[i] = v_right;
                    break
                }
            }
        }
    }
}

fn binary_op_unless(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        match m_right.get(k) {
            None => rvs.append(&mut tss_left),
            Some(tss_right) => {
                // Add gaps to tssLeft if the are no gaps at tss_right.
                add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
                rvs.append(&mut tss_left);
            }
        }
    }

    Ok(rvs)
}

fn binary_op_if_not(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        match series_by_key(m_right, k) {
            None => {
                rvs.extend( tss_left.drain(0..) );
            },
            Some(tss_right) => {
                add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
                rvs.extend(tss_left.drain(0..).collect::<Vec<_>>());
            }
        }
    }

    Ok(rvs)
}

fn add_left_nans_if_no_right_nans(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for tsLeft in tss_left {
        for i in 0 .. tsLeft.values.len() {
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    tsLeft.values[i] = f64::NAN;
                    break;
                }
            }
        }
    }

    remove_empty_series(tss_left);
}

fn create_timeseries_map_by_tag_set(bfa: &mut BinaryOpFuncArg) -> (&mut TimeseriesHashMap, &TimeseriesHashMap) {

    let (group_op, group_tags) = get_modifier_or_default(&mut bfa.be);

    let get_tags_map = |arg: &mut Vec<Timeseries>| -> TimeseriesHashMap {
        // todo:: tinyvec
        let mut bb = get_pooled_buffer(512);

        let mut m = TimeseriesHashMap::with_capacity(arg.len());
        let mut mn = get_pooled_metric_name();
        for ts in arg.into_iter() {
            mn.copy_from(&ts.metric_name);
            mn.reset_metric_group();
            match group_op {
                GroupModifierOp::On => {
                    mn.remove_tags_on(group_tags);
                },
                GroupModifierOp::Ignoring => {
                    mn.remove_tags_ignoring(group_tags);
                },
            }
            {
                let key = mn.marshal_to_string(bb.deref_mut());
                m.entry(key.to_string())
                    .or_insert(vec![])
                    .push(std::mem::take(ts));

                bb.clear();
            }
        }

        return m
    };
    let mut m_left = get_tags_map(&mut bfa.left);
    let m_right = get_tags_map(&mut bfa.right);
    return (&mut m_left, &m_right)
}

fn is_scalar(arg: &[Timeseries]) -> bool {
    if arg.len() != 1 {
        return false
    }
    let mn = &arg[0].metric_name;
    if mn.metric_group.len() > 0 {
        return false
    }
    mn.get_tag_count() == 0
}

fn series_by_key<'a>(m: &'a TimeseriesHashMap, key: &'a str) -> Option<&'a Vec<Timeseries>> {
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

#[inline]
fn get_modifier_or_default(be: &mut BinaryOpExpr) -> (GroupModifierOp, &Vec<String>) {
    match &be.group_modifier {
        None => {
            (GroupModifierOp::Ignoring, &vec![])
        },
        Some(modifier) => {
            (modifier.op, &modifier.labels)
        }
    }
}

fn is_group_right(bfa: &BinaryOpExpr) -> bool {
    match &bfa.join_modifier {
        Some(modifier) => modifier.op == JoinModifierOp::GroupRight,
        None => false
    }
}

// todo: put in optimize phase
pub(crate) fn adjust_cmp_ops(e: &mut Expression) {
    visit_all(e, |mut expr: &Expression|
        {
            match expr {
                Expression::BinaryOperator(mut be) => {
                    let _ = be.adjust_comparison_op();
                },
                _ => {}
            }
        });
}