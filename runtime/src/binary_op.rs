use std::borrow::{Cow};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;

use metricsql::ast::{
    BinaryOp,
    BinaryOpExpr,
    GroupModifier,
    GroupModifierOp,
    JoinModifier,
    JoinModifierOp
};


use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::timeseries::Timeseries;

pub(crate) struct BinaryOpFuncArg<'a> {
    be: &'a BinaryOpExpr,
    left: Vec<Timeseries>,
    right: Vec<Timeseries>,
}

impl<'a> BinaryOpFuncArg<'a> {
    pub fn new(left: Vec<Timeseries>, be: &'a BinaryOpExpr, right: Vec<Timeseries>) -> Self {
        Self {
            left,
            be,
            right
        }
    }
}

pub(crate) trait BinaryOpFn: Fn(&mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

impl<T> BinaryOpFn for T where T: Fn(&mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

pub(crate) type BinaryOpFnImplementation = Arc<dyn BinaryOpFn<Output=RuntimeResult<Vec<Timeseries>>>>;

type TimeseriesHashMap = HashMap<String, Vec<Timeseries>>;

macro_rules! boxed {
    ( $af: expr ) => {
        Arc::new($af)
    };
}

macro_rules! arith_func {
    ($af: expr) => {
        Arc::new(
                new_binary_op_func(|left: f64, right: f64, _is_bool: bool| $af(left, right))
            )
    }
}

macro_rules! comp_func {
    ( $af: expr ) => {
        Arc::new(new_binary_op_cmp_func($af))
    };
}


static HANDLER_MAP: Lazy<RwLock<HashMap<BinaryOp, BinaryOpFnImplementation>>> = Lazy::new(|| {
    use BinaryOp::*;

    let mut m: HashMap<BinaryOp, BinaryOpFnImplementation> = HashMap::with_capacity(14);

    m.insert(Add,   arith_func!(metricsql::binaryop::plus));
    m.insert(Sub,   arith_func!(metricsql::binaryop::minus));
    m.insert(Mul,   arith_func!(metricsql::binaryop::mul));
    m.insert(Div,   arith_func!(metricsql::binaryop::div));
    m.insert(Mod,   arith_func!(metricsql::binaryop::mod_));
    m.insert(Pow,   arith_func!(metricsql::binaryop::pow));

    // See https://github.com/prometheus/prometheus/pull/9248
    m.insert(Atan2, arith_func!(metricsql::binaryop::atan2));

    // cmp ops
    m.insert(Eql,   comp_func!(metricsql::binaryop::eq));
    m.insert(Neq,   comp_func!(metricsql::binaryop::neq));
    m.insert(Gt,    comp_func!(metricsql::binaryop::gt));
    m.insert(Gte,   comp_func!(metricsql::binaryop::gte));
    m.insert(Lt,    comp_func!(metricsql::binaryop::lt));
    m.insert(Lte,   comp_func!(metricsql::binaryop::lte));

    // logical set ops
    m.insert(And,   boxed!(binary_op_and));
    m.insert(Or,    boxed!(binary_op_or));
    m.insert(Unless, boxed!(binary_op_unless));

    // New ops
    m.insert(If,        boxed!(binary_op_if));
    m.insert(IfNot,     boxed!(binary_op_if_not));
    m.insert(Default,   boxed!(binary_op_default));

    RwLock::new(m)
});


pub(crate) fn get_binary_op_handler(op: BinaryOp) -> BinaryOpFnImplementation {
    let map = HANDLER_MAP.read().unwrap();
    map.get(&op).unwrap().clone()
}


fn new_binary_op_cmp_func(cf: fn(left: f64, right: f64) -> bool) -> impl BinaryOpFn {
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

        let is_bool = bfa.be.bool_modifier;

        let (left, right, mut dst) = adjust_binary_op_tags(bfa)?;
        if left.len() != right.len() || left.len() != dst.len() {
            return Err(RuntimeError::InvalidState(
                format!("BUG: left.len() must match right.len() and dst.len(); got {} vs {} vs {}", left.len(), right.len(), dst.len())
            ));
        }

        for (i, ts_left) in left.iter().enumerate() {
            let left_values = &ts_left.values;
            let right_values = &right[i].values;
            let dst_len = dst[i].values.len();

            if left_values.len() != right_values.len() || left_values.len() != dst_len {
                let msg = format!("BUG: left_values.len() must match right_values.len() and len(dst_values); got {} vs {} vs {}",
                                  left_values.len(), right_values.len(), dst_len);
                return Err(RuntimeError::InvalidState(msg));
            }

            for (j, a) in left_values.iter().enumerate() {
                let b = right_values[j];
                dst[i].values[j] = bf(*a, b, is_bool)
            }
        }

        // do not remove time series containing only NaNs, since then the `(foo op bar) default N`
        // won't work as expected if `(foo op bar)` results to NaN series.
        let res: Vec<Timeseries> = dst
            .drain(0..)
            .collect::<Vec<_>>();

        Ok(res)
    }
}

#[inline]
pub(super) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| {
        !ts.values.iter().all(|v| v.is_nan())
    });
}

fn adjust_binary_op_tags<'a>(bfa: &'a mut BinaryOpFuncArg)
                         -> RuntimeResult<(Cow<'a, Vec<Timeseries>>, Cow<'a, Vec<Timeseries>>, Vec<Timeseries>)> {

    if bfa.be.group_modifier.is_none() && bfa.be.join_modifier.is_none() {
        if is_scalar(&bfa.left) {
            // Fast path: `scalar op vector`
            let mut rvs_left: Vec<Timeseries> = Vec::with_capacity(bfa.right.len());
            let ts_left = &bfa.left[0];

            // todo(perf): optimize and avoid clone of ts_left in case bfa.right.len() == 1
            for mut ts_right in bfa.right.iter_mut() {
                reset_metric_group_if_required(bfa.be, &mut ts_right);
                rvs_left.push(ts_left.clone());
            }

            let dst = bfa.right.clone();
            let right = Cow::Borrowed(&bfa.right);
            let left = Cow::Owned(rvs_left);
            return Ok((left, right, dst))
        }

        if is_scalar(&bfa.right) {
            // Fast path: `vector op scalar`
            let mut rvs_right: Vec<Timeseries> = Vec::with_capacity(bfa.left.len());
            let ts_right = &bfa.right[0];
            for ts_left in bfa.left.iter_mut() {
                reset_metric_group_if_required(bfa.be, ts_left);
                rvs_right.push(ts_right.clone());
            }

            let left = Cow::Borrowed(&bfa.left);
            let right = Cow::Owned(rvs_right);
            let dst = bfa.left.clone();

            return Ok((left, right, dst))
        }
    }

    // Slow path: `vector op vector` or `a op {on|ignoring} {group_left|group_right} b`
    let (mut m_left, mut m_right) = create_timeseries_map_by_tag_set(bfa);

    let mut rvs_left: Vec<Timeseries> = Vec::with_capacity(1);
    let mut rvs_right: Vec<Timeseries> = Vec::with_capacity(1);

    for (k, tss_left) in m_left.iter_mut() {
        let mut tss_right = m_right.remove(k).unwrap_or(vec![]);
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

                if let Some(modifier) = &bfa.be.group_modifier {
                    ts_left.metric_name.update_tags_by_group_modifier(modifier);
                }

                rvs_left.push(std::mem::take(&mut ts_left));
                rvs_right.push(tss_right.remove(0))
            }
        }
    }

    let dst = if is_group_right(&bfa.be) {
        rvs_right.clone()
    } else {
        rvs_left.clone()
    };

    let right = Cow::Owned(rvs_right);
    let left = Cow::Owned(rvs_left);

    return Ok((left, right, dst))
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

    let empty_labels: Vec<String> = vec![];

    let join_tags = if let Some(be_) = &be.join_modifier {
        &be_.labels
    } else {
        &empty_labels
    };

    struct TsPair {
        left: Timeseries,
        right: Timeseries
    }

    let mut m: HashMap<String, TsPair> = HashMap::with_capacity(rvs_left.len());

    for ts_left in tss_left.into_iter() {
        reset_metric_group_if_required(be, ts_left);

        if tss_right.len() == 1 {
            let mut right = tss_right.remove(0);
            // Easy case - right part contains only a single matching time series.
            ts_left.metric_name.set_tags(join_tags, &mut right.metric_name);
            rvs_left.push(std::mem::take(ts_left));
            rvs_right.push(std::mem::take(&mut right));
            continue
        }

        // Hard case - right part contains multiple matching time series.
        // Verify it doesn't result in duplicate MetricName values after adding missing tags.
        m.clear();

        for ts_right in tss_right.into_iter() {
            // todo: Question - do we need to copy? I dont think tss_left and tss_right
            // are used anymore when we exit this function
            let mut ts_copy = Timeseries::copy(ts_left);
            ts_copy.metric_name.set_tags(join_tags, &mut ts_right.metric_name);

            let key = ts_copy.metric_name.to_string();

            // todo: check specifically for error
            match m.get_mut(&key) {
                None => {
                    m.insert(key, TsPair {
                        left:  ts_copy,
                        right: std::mem::take(ts_right),
                    });
                    continue
                },
                Some(pair) => {
                    // todo(perf): may not need to copy
                    // Try merging pair.right with ts_right if they don't overlap.
                    if !merge_non_overlapping_timeseries(&mut pair.right, ts_right) {
                        let err = format!("duplicate time series on the {} side of `{} {} {}`: {} and {}",
                                          single_timeseries_side, be.op,
                                          group_modifier_to_string(&be.group_modifier),
                                          join_modifier_to_string(&be.join_modifier),
                                          pair.right.metric_name,
                                          ts_right.metric_name);

                        return Err(RuntimeError::from(err));
                    }
                }
            }

        }

        for (_, pair) in m.iter_mut() {
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
            // Allow up to two overlapping data points, which can appear due to staleness algorithm,
            // which can add a few data points in the end of time series.
            if overlaps > 2 {
                return false
            }
        }
    }

    // do not merge time series with too small number of data points.
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

    for (k, tss_left) in m_left.iter_mut() {
        match series_by_key(&m_right, &k) {
            None => continue,
            Some(tss_right) => {
                add_right_nans_to_left(tss_left, tss_right);
                rvs.extend(tss_left.drain(..))
            }
        }
    }

    Ok(rvs)
}

fn binary_op_and(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( std::cmp::min(m_left.len(), m_right.len() ) );

    for (k, tss_right) in m_right.into_iter() {
        match m_left.get_mut(&k) {
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
    for ts_left in tss_left.iter_mut() {
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
    let (mut m_left, m_right) = create_timeseries_map_by_tag_set(bfa);

    if m_left.len() == 0 {
        // see if we can make this more efficient
        let items = m_right.into_values().flatten().collect();
        return Ok(items)
    }

    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len() );
    for (k, mut tss_left) in m_left.iter_mut() {
        match series_by_key(&m_right, &k) {
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
    let mut right: Vec<Timeseries> = Vec::with_capacity( m_right.len()  );

    for (k, tss_right) in m_right.iter_mut() {
        match m_left.get_mut(k) {
            None => {
                right.extend_from_slice(&tss_right);
            },
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
fn fill_left_nans_with_right_values(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    // Fill gaps in tss_left with values from tss_right as Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/552
    for ts_left in tss_left.iter_mut() {
        for i in 0 .. ts_left.values.len() {
            let v = ts_left.values[i];
            if !v.is_nan() {
                continue
            }
            for ts_right in tss_right {
                let v_right = ts_right.values[i];
                if !v_right.is_nan() {
                    ts_left.values[i] = v_right;
                    break
                }
            }
        }
    }
}

fn binary_op_unless(bfa: &mut BinaryOpFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        match m_right.get(&k) {
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
    let (m_left, m_right) = create_timeseries_map_by_tag_set(bfa);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity( m_left.len());

    for (k, mut tss_left) in m_left.into_iter() {
        match series_by_key(&m_right, &k) {
            None => {
                rvs.extend( tss_left.into_iter());
            },
            Some(tss_right) => {
                add_left_nans_if_no_right_nans(&mut tss_left, tss_right);
                rvs.extend(tss_left.into_iter());
            }
        }
    }

    Ok(rvs)
}

fn add_left_nans_if_no_right_nans(tss_left: &mut Vec<Timeseries>, tss_right: &Vec<Timeseries>) {
    for ts_left in tss_left.iter_mut() {
        for i in 0 .. ts_left.values.len() {
            for ts_right in tss_right {
                if !ts_right.values[i].is_nan() {
                    ts_left.values[i] = f64::NAN;
                    break;
                }
            }
        }
    }

    remove_empty_series(tss_left);
}

fn create_timeseries_map_by_tag_set(bfa: &mut BinaryOpFuncArg) -> (TimeseriesHashMap, TimeseriesHashMap) {

    let get_tags_map = |arg: &mut Vec<Timeseries>| -> TimeseriesHashMap {
        let mut m = TimeseriesHashMap::with_capacity(arg.len());
        for ts in arg.into_iter() {
            let mut mn = ts.metric_name.clone();
            mn.reset_metric_group();
            if let Some(modifier) = &bfa.be.group_modifier {
                mn.update_tags_by_group_modifier(modifier);
            }
            let key = mn.to_string(); // mn.to_canonical_string()

            m.entry(key)
                .or_insert(vec![])
                .push(std::mem::take(ts));
        }

        return m
    };

    let m_left = get_tags_map(&mut bfa.left);
    let m_right = get_tags_map(&mut bfa.right);
    return (m_left, m_right)
}

fn is_scalar(arg: &[Timeseries]) -> bool {
    if arg.len() != 1 {
        return false
    }
    let mn = &arg[0].metric_name;
    if mn.metric_group.len() > 0 {
        return false
    }
    mn.tag_count() == 0
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
fn get_modifier_or_default(be: &BinaryOpExpr) -> (GroupModifierOp, Cow<Vec<String>>) {
    match &be.group_modifier {
        None => {
            (GroupModifierOp::Ignoring, Cow::Owned::<Vec<String>>(vec![]))
        },
        Some(modifier) => {
            (modifier.op, Cow::Borrowed(&modifier.labels))
        }
    }
}

fn is_group_right(bfa: &BinaryOpExpr) -> bool {
    match &bfa.join_modifier {
        Some(modifier) => modifier.op == JoinModifierOp::GroupRight,
        None => false
    }
}
