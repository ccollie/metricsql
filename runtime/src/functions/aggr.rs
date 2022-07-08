use std::cmp::Ordering;
use std::collections::{HashMap};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::ops::{Deref, DerefMut};
use lockfree_object_pool::LinearReusable;
use phf::phf_map;
use tinyvec::*;

use lib::{get_float64s, get_pooled_buffer};
use metricsql::ast::{AggregateModifier, AggregateModifierOp, AggrFuncExpr};

use crate::{MetricName, Timeseries};
use crate::eval::{eval_number, EvalConfig};
use crate::exec::remove_empty_series;
use crate::functions::rollup::get_string;
use crate::functions::transform::vmrange_buckets_to_le;
use crate::histogram::{get_pooled_histogram, Histogram};
use crate::runtime_error::{RuntimeError, RuntimeResult};

#[derive(Clone)]
pub(crate) struct AggrFuncArg {
    args: Vec<Vec<Timeseries>>,
    ae: AggrFuncExpr,
    ec: EvalConfig,
}

impl AggrFuncArg {
    pub fn new(ae: AggrFuncExpr, args: Vec<Vec<Timeseries>>, ec: EvalConfig) -> Self {
        Self {
            args,
            ae,
            ec
        }
    }
}

pub type AggrFunc = fn(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>>;

trait AggrFn: FnMut(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {}
impl<T> AggrFn for T where T: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {}

fn new_aggr_func(afe: fn(tss: &mut Vec<Timeseries>)) -> AggrFunc {
    |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let mut tss = get_aggr_timeseries(&afa.args)?;
        aggr_func_ext(|tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
            afe(tss);
            tss.shrink_to(1);
            tss
        }.to_owned(), // todo: avoid cloning
                      &mut tss, &afa.ae.modifier, afa.ae.limit, false)
    }
}

fn get_aggr_timeseries(args: &Vec<Vec<Timeseries>>) -> RuntimeResult<&Vec<Timeseries>> {
    if args.len() == 0 {
        return Err(RuntimeError::ArgumentError("expecting at least one arg".to_string()));
    }
    let mut tss = &args[0];
    for arg in args[1..].iter_mut() {
        tss.append(arg)
    }
    Ok(tss)
}

pub(crate) fn remove_group_tags(metric_name: &mut MetricName, modifier: &Option<AggregateModifier>) {
    let mut group_op = AggregateModifierOp::By;
    let mut labels = &vec![]; // zero alloc

    if let Some(m) = modifier.deref() {
        group_op = m.op;
        labels = &m.args
    };

    match group_op {
        AggregateModifierOp::By => {
            metric_name.remove_tags_on(labels);
        }
        AggregateModifierOp::Without => {
            metric_name.remove_tags_ignoring(labels);
            // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
            metric_name.reset_metric_group();
        },
        _ => {
            panic!("BUG: unknown group modifier: {}", group_op)
        }
    }
}

fn aggr_func_ext(
    afe: fn(tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>) -> Vec<Timeseries>,
    arg_orig: &mut Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    max_series: usize,
    keep_original: bool) -> RuntimeResult<Vec<Timeseries>> {
    let mut arg = copy_timeseries_metric_names(arg_orig, keep_original);

    // Perform grouping.
    let mut m: HashMap<&str, Vec<Timeseries>> = HashMap::new();

    let mut bb = get_pooled_buffer(256);

    let mut i = 0;
    for mut ts in arg.into_iter() {
        remove_group_tags(&mut ts.metric_name, modifier);
        let key = ts.metric_name.marshal_to_string(bb.deref_mut())?;

        if keep_original {
            ts = arg_orig[i].into();
        }
        let tss = m.get_mut(&key);
        bb.clear();

        if tss.is_none() && max_series > 0 && m.len() >= max_series {
            // We already reached time series limit after grouping. Skip other time series.
            continue;
        }
        let mut tss = tss.unwrap();
        tss.push(ts);
        m.insert(&key, tss.unwrap());
        bb.clear();
        i += 1;
    }

    let mut src_tss_count = 0;
    let mut dst_tss_count = 0;
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, tss) in m.iter_mut() {
        let mut rv = afe(tss, modifier);
        rvs.append(&mut rv);
        src_tss_count += tss.len();
        dst_tss_count += rv.len();
        if dst_tss_count > 2000 && dst_tss_count > 16 * src_tss_count {
            // This looks like count_values explosion.
            let msg = format!(
                "too many timeseries after aggregation; \n
                got {}; want less than {}", dst_tss_count, 16 * src_tss_count);
            return Err(RuntimeError::from(msg));
        }
    }

    Ok(rvs)
}

fn aggr_func_any(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
    let afe = |tss, modifier| -> Vec<Timeseries> {
        return tss[0];
    };
    let mut limit = afa.ae.limit;
    if limit > 1 {
        // Only a single time series per group must be returned
        limit = 1
    }
    aggr_func_ext(afe, &mut tss, &afa.ae.modifier, limit, true)
}

fn aggr_func_group(tss: &mut Vec<Timeseries>) {
    for dv in tss[0].values.iter_mut() {
        let mut v = f64::NAN;
        for (i, ts) in tss.iter().enumerate() {
            if ts.values[i].is_nan() {
                continue;
            }
            v = 1.0;
        }
        *dv = v;
    }
}

fn aggr_func_sum(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        return;
    }

    let mut i: usize = 0;

    for dv in tss[0].values.iter_mut() {
        let mut sum: f64 = 0.0;
        let mut count: usize = 0;
        for ts in tss {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            sum += v;
            count += 1;
        }
        *dv = sum;
        i += i;
    }
}

fn aggr_func_sum2(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        return
    }

    let mut i: usize = 0;

    for v in tss[0].values.iter_mut() {
        let mut sum2: f64 = 0.0;
        let mut count: usize = 0;
        for ts in tss {
            let x = ts.values[i];
            if x.is_nan() {
                continue;
            }
            sum2 += x * x;
            count += 1;
        }
        if count == 0 {
            sum2 = f64::NAN;
        }
        *v = sum2;
        i += i;
    }
}

fn aggr_func_geomean(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to geomean.
        return
    }
    for i in 0..tss[0].values.len() {
        let mut p = 1.0;
        let mut count = 0;
        for ts in tss {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            p *= v;
            count += 1;
        }
        if count == 0 {
            p = f64::NAN
        }
        tss[0].values[i] = p.powf((1 / count) as f64);
    }
}

fn aggr_func_histogram(tss: &mut Vec<Timeseries>) -> Vec<Timeseries> {
    let mut h: LinearReusable<Histogram> = get_pooled_histogram();
    let mut m: HashMap<String, Timeseries> = HashMap::new();
    for i in 0 .. tss[0].values.len() {
        h.reset();
        for ts in tss {
            let v = ts.values[i];
            h.update(v)
        }

        for bucket in h.non_zero_buckets() {
            let mut ts = match m.entry(bucket.vm_range.to_string()) {
                Vacant(entry) => {
                    let mut ts = Timeseries::copy_from_shallow_timestamps(&tss[0]);
                    ts.metric_name.remove_tag("vmrange");
                    ts.metric_name.add_tag("vmrange", bucket.vm_range);

                    // todo: should be a more efficient way to do this
                    for k in 0 .. ts.values.len() {
                        ts.values[k] = 0.0;
                    }
                    entry.insert(ts);
                },
                Occupied(entry) => entry.into_mut(),
            };
            ts.values[i] = bucket.count as f64;
        }
    }

    let mut res: Vec<Timeseries> = Vec::from(m.values());
    return vmrange_buckets_to_le(&mut res);
}

fn aggr_func_min(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to min.
        return
    }

    let mut i = 0;

    for min in tss[0].values.iter_mut() {
        for ts in tss {
            let v = ts.values[i];
            if min.is_nan() || v < *min {
                *min = v;
            }
        }
        i += 1;
    }

}

fn aggr_func_max(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to max.
        return
    }
    let mut i = 0;
    for max in tss[0].values.iter_mut() {
        for ts in tss.iter() {
            let v = ts.values[i];
            if max.is_nan() || v > *max { 
                *max = v;
            }
        }
        i += 1;
    }
}

fn aggr_func_avg(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to avg.
        return;
    }

    let mut i = 0;
    for dst_value in tss[0].values.iter_mut() {
        // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
        // since it is slower and has no obvious benefits in increased precision.
        let mut sum: f64 = 0.0;
        let mut count: usize = 0;
        for ts in tss {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            count += 1;
            sum += v;
        }
        let mut avg = f64::NAN;
        if count > 0 {
            avg = sum / count as f64;
        }
        *dst_value = avg;

        i += 1;
    }
}

fn aggr_func_stddev(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - stddev over a single time series is zero
        for v in tss[0].values.iter_mut() {
            if !v.is_nan() {
                *v = 0.0;
            }
        }
        return;
    }
    aggr_func_stdvar(tss);

    for v in tss[0].values.iter_mut() {
        *v = v.sqrt();
    }
}

fn aggr_func_stdvar(tss: &mut Vec<Timeseries>)  {
    if tss.len() == 1 {
        // Fast path - stdvar over a single time series is zero
        for v in tss[0].values.iter_mut() {
            if !v.is_nan() {
                *v = 0.0;
            }
        }
        return
    }

    let mut i = 0;
    for dst_value in tss[0].values.iter_mut() {
        // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
        let mut avg: f64 = 0.0;
        let mut count: f64 = 0.0;
        let mut q = 0.0;

        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            count += 1.0;
            let avg_new = avg + (v - avg) / count;
            q += (v - avg) * (v - avg_new);
            avg = avg_new
        }
        if count == 0.0 {
            *dst_value = f64::NAN
        } else {
            *dst_value = q / count;
        }
        i += 1;
    }
}

fn aggr_func_count(tss: &mut Vec<Timeseries>) {
    for dst_val in tss[0].values.iter_mut() {
        let mut count = 0;
        for (i, ts) in tss.iter().enumerate() {
            if ts.values[i].is_nan() {
                continue;
            }
            count += 1;
        }
        let mut v: f64 = count as f64;
        if count == 0 {
            v = f64::NAN
        }
        *dst_val = v;
    }
}

fn aggr_func_distinct(tss: &mut Vec<Timeseries>) {
    let mut values: Vec<f64> = Vec::with_capacity(tss.len());

    let mut i = 0;
    for dst_value in tss[0].values.iter_mut() {
        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            values.push(v);
        }
        values.sort_by(|a, b| a.total_cmp(&b));
        values.dedup();
        let n = values.len();
        *dst_value = if n == 0 { f64::NAN } else { n as f64 };

        values.clear();
    }
}

fn aggr_func_mode(tss: &mut Vec<Timeseries>) {
    let mut a: Vec<f64> = Vec::with_capacity(tss.len());
    let mut i = 0;
    for dst_value in tss[0].values.iter_mut() {
        for ts in tss {
            let v = ts.values[i];
            if !v.is_nan() {
                a.push(v);
            }
        }
        *dst_value = mode_no_nans(f64::NAN, &mut a);
        a.clear();
        i += 1;
    }
}

fn aggr_func_zscore(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
    let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        for i in 0..tss[0].values.len() {
            // Calculate avg and stddev for tss points at position i.
            // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
            let mut avg: f64 = 0.0;
            let mut count: u32 = 0;
            let mut q: f64 = 0.0;

            for ts in tss {
                let v = ts.values[i];
                if v.is_nan() {
                    continue;
                }
                count += 1;
                let avg_new = avg + (v - avg) / f64::from(count);
                q += (v - avg) * (v - avg_new);
                avg = avg_new
            }
            if count == 0 {
                // Cannot calculate z-score for NaN points.
                continue;
            }

            // Calculate z-score for tss points at position i.
            // See https://en.wikipedia.org/wiki/Standard_score
            let stddev = (q / count as f64).sqrt();
            for ts in tss.iter_mut() {
                let v = ts.values[i];
                if v.is_nan() {
                    continue;
                }
                ts.values[i] = (v - avg) / stddev
            }
        }

        // Remove MetricGroup from all the tss.
        for ts in tss.iter_mut() {
            ts.metric_name.reset_metric_group()
        }

        std::mem::take(tss)
    };

    aggr_func_ext(afe, &mut tss, &afa.ae.modifier, afa.ae.limit, true)
}

/// modeNoNaNs returns mode for a.
///
/// It is expected that a doesn't contain NaNs.
///
/// The function modifies contents for a, so the caller must prepare it accordingly.
///
/// See https://en.wikipedia.org/wiki/Mode_(statistics)
pub(crate) fn mode_no_nans(mut prev_value: f64, a: &mut Vec<f64>) -> f64 {
    if a.len() == 0 {
        return prev_value;
    }
    a.sort_by(|a, b| a.total_cmp(b));
    let mut j = -1;
    let mut d_max = 0;
    let mut mode = prev_value;
    for (i, v) in a.iter().enumerate() {
        if prev_value == *v {
            continue;
        }
        let d = i - j;
        if d > d_max || mode.is_nan() {
            d_max = d;
            mode = prev_value;
        }
        j = i;
        prev_value = *v;
    }
    let d = a.len() - j;
    if d > d_max || mode.is_nan() {
        mode = prev_value
    }
    return mode;
}

fn aggr_func_count_values(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&afa.args, 2)?;
    let dst_label = get_string(&afa.args[0], 0)?;

    // Remove dst_label from grouping like Prometheus does.
    if let Some(mut modifier) = &afa.ae.modifier {
        match modifier.op {
            AggregateModifierOp::Without => {
                modifier.args.push( dst_label);
            },
            AggregateModifierOp::By => {
                modifier.args.retain(|x| x != &dst_label);
            }
        }
    }

    let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| -> Vec<Timeseries> {
        let mut values: Vec<f64> = Vec::with_capacity(16); // todo: calculate initial capacity
        for ts in tss.iter() {
            for v in ts.values.iter() {
                if v.is_nan() {
                    continue;
                }
                values.push(*v);
            }
        }

        values.sort_by(|a, b| a.total_cmp(b));
        values.dedup();

        let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
        for v in values {
            let mut dst: Timeseries = Timeseries::copy_from_shallow_timestamps(&tss[0]);
            dst.metric_name.remove_tag(&dst_label);
            dst.metric_name.add_tag(&dst_label, format!("{}", v).as_str());

            let mut i = 0;
            for dst_value in dst.values.iter_mut() {
                let mut count = 0;
                for ts in tss {
                    if ts.values[i] == v {
                        count += 1;
                    }
                }
                *dst_value = if count == 0 { f64::NAN } else { count as f64 };
                i += 1;
            }

            rvs.push(dst);
        }
        return rvs
    };

    return aggr_func_ext(afe, &mut afa.args[1], &afa.ae.modifier, afa.ae.limit, false)
}

fn new_aggr_func_topk(is_reverse: bool) -> AggrFunc {
    |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let args = &afa.args;
        expect_transform_args_num(args, 2)?;
        let ks = get_scalar(&args[0], 0)?;

        let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| -> Vec<Timeseries> {
            for n in 0..tss[0].values.len() {
                tss.sort_by(|first, second| {
                    let a = first.values[n];
                    let b = second.values[n];
                    if is_reverse {
                        b.total_cmp(&a)
                    } else {
                        a.total_cmp(&b)
                    }
                });
                fill_nans_at_idx(n, ks[n], tss)
            }
            remove_empty_series(tss);
            tss.reverse();
            return tss;
        };

        return aggr_func_ext(afe, &mut args[1], &afa.ae.modifier, afa.ae.limit, true);
    }
}

/////////////////////////////////////////////////

fn new_aggr_func_range_topk(f: fn(values: &[f64]) -> f64, is_reverse: bool) -> AggrFunc {
    return |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let args = &afa.args;
        if afa.args.len() < 2 {
            let msg = format!("unexpected number of args; got {}; want at least {}", args.len(), 2);
            return Err(RuntimeError::ArgumentError(msg));
        }
        if afa.args.len() > 3 {
            let msg = format!("unexpected number of args; got {}; want no more than {}", args.len(), 3);
            return Err(RuntimeError::ArgumentError(msg));
        }
        let ks = get_scalar(&args[0], 0)?;
        let mut remaining_sum_tag_name = "";
        if args.len() == 3 {
            remaining_sum_tag_name = &*get_string(&args[2], 2)?
        }
        let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
            return get_range_topk_timeseries(tss,
                                             modifier,
                                             &ks,
                                             remaining_sum_tag_name,
                                             f,
                                             is_reverse);
        };
        return aggr_func_ext(afe, &mut args[1], &afa.ae.modifier, afa.ae.limit, true);
    };
}

fn get_range_topk_timeseries(tss: &mut Vec<Timeseries>,
                             modifier: &Option<AggregateModifier>,
                             ks: &Vec<f64>,
                             remaining_sum_tag_name: &str,
                             f: fn(values: &[f64]) -> f64,
                             is_reverse: bool) -> Vec<Timeseries> {
    struct TsWithValue<'a> {
        ts: &'a Timeseries,
        value: f64,
    }
    let mut maxs: Vec<TsWithValue> = Vec::with_capacity(tss.len());
    for ts in tss.into_iter() {
        let value = f(&ts.values);
        maxs.push(TsWithValue {
            ts,
            value,
        });
    }
    maxs.sort_by(|first, second| {
        let mut a = first.value;
        let mut b = second.value;
        if is_reverse {
            b.total_cmp(&a)
        } else {
            a.total_cmp(&b)
        }
    });

    let mut series = maxs.into_iter()
        .map(|x| *x.ts);

    tss.extend(series);

    let remaining_sum_ts = get_remaining_sum_timeseries(tss, modifier, ks, remaining_sum_tag_name);
    for (i, k) in ks.iter().enumerate() {
        fill_nans_at_idx(i, *k, tss)
    }

    if let Some(remaining) = remaining_sum_ts {
        tss.push(remaining);
    }

    remove_empty_series(tss);
    tss.reverse();
    return tss;
}

fn get_remaining_sum_timeseries(
    tss: &Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    ks: &[f64],
    remaining_sum_tag_name: &str) -> Option<Timeseries> {
    if remaining_sum_tag_name.len() == 0 || tss.len() == 0 {
        return None;
    }
    let mut dst = Timeseries::copy_from_shallow_timestamps(&tss[0]);
    remove_group_tags(&mut dst.metric_name, modifier);

    let mut tag_value = remaining_sum_tag_name;
    let mut remaining = remaining_sum_tag_name;

    match remaining_sum_tag_name.rsplit_once('=') {
        Some((tag, remains)) => {
            tag_value = tag;
            remaining = remains;
        }
        _ => {}
    }

    dst.metric_name.remove_tag(remaining);
    dst.metric_name.add_tag(remaining, tag_value);
    for (i, k) in ks.iter().enumerate() {
        let kn = get_int_k(*k, tss.len());
        let mut sum: f64 = 0.0;
        let mut count = 0;

        for j in 0 .. tss.len() - kn {
            let mut ts = &tss[j];
            let mut v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            sum += v;
            count = count + 1;
        }
        if count == 0 {
            sum = f64::NAN;
        }
        dst.values[i] = sum;
    }
    return Some(dst);
}

fn fill_nans_at_idx(idx: usize, k: f64, tss: &Vec<Timeseries>) {
    let kn = get_int_k(k, tss.len());
    for ts in &tss[0..tss.len() - kn] {
        ts.values[idx] = f64::NAN
    }
}

fn get_int_k(k: f64, k_max: usize) -> usize {
    if k.is_nan() {
        return 0;
    }
    let kn = k as i64;
    if kn < 0 {
        return 0;
    }
    if kn > k_max as i64 {
        return k_max;
    }
    return kn as usize;
}

fn min_value(values: &[f64]) -> f64 {
    let mut min = f64::NAN;
    let mut i: usize = 0;
    while i < values.len() && values[i].is_nan() {
        i += 1;
    }
    for k in i .. values.len() {
        let v = values[k];
        if !v.is_nan() && v < min {
            min = v
        }
    }
    return min;
}

fn max_value(values: &[f64]) -> f64 {
    let mut max = f64::NAN;
    let mut values = &values;
    let mut i = 0;
    while i < values.len() && values[i].is_nan() {
        i += 1;
    }
    for k in i .. values.len() {
        let v = values[k];
        if !v.is_nan() && v > max {
            max = v
        }
    }
    return max;
}

fn avg_value(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut count = 0;
    for v in values {
        if v.is_nan() {
            continue;
        }
        count += 1;
        sum += v
    }
    if count == 0 {
        return f64::NAN;
    }

    sum / count as f64
}

fn median_value(values: &[f64]) -> f64 {
    let mut h = get_pooled_histogram();
    for v in values {
        if !v.is_nan() {
            h.update(*v)
        }
    }
    h.quantile(0.5)
}

fn aggr_func_outliers_k(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args = &afa.args;
    expect_transform_args_num(args, 2)?;
    let ks = get_scalar(&args[0], 0)?;
    let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        // Calculate medians for each point across tss.
        let medians: Vec<f64> = Vec::new();
        let mut h = get_pooled_histogram();
        // todo: set upper limit on ks ?
        for n in 0..ks.len() {
            h.reset();
            for ts in tss.iter() {
                let v = ts.values[n];
                if !v.is_nan() {
                    h.update(v);
                }
            }
            medians[n] = h.quantile(0.5)
        }

        // Return topK time series with the highest variance from median.
        let f = |values: &[f64]| -> f64 {
            let mut sum2: f64 = 0.0;
            for (n, v) in values.iter().enumerate() {
                let d = v - medians[n];
                sum2 += d * d;
            }
            return sum2;
        };

        return get_range_topk_timeseries(tss, &afa.ae.modifier, &ks, "", f, false);
    };

    return aggr_func_ext(afe, &mut args[1], &afa.ae.modifier, afa.ae.limit, true);
}

fn aggr_func_limitk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut args = &afa.args;
    expect_transform_args_num(&args, 2)?;
    let ks = get_scalar(&args[0], 0)?;
    let mut max_k = ks.iter().max().unwrap().floor() as usize;

    let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        if tss.len() > max_k {
            // todo: set_len
            tss.truncate(max_k);
        }
        for (i, kf) in ks.iter().enumerate() {
            let mut k = *kf as usize;
            if k < 0 {
                k = 0
            }
            let mut j = k;
            while j < tss.len() {
                tss[j].values[i] = f64::NAN;
                j += 1;
            }
        }
        return tss
    };
    return aggr_func_ext(afe, &mut args[1], &afa.ae.modifier, afa.ae.limit, true)
}

fn aggr_func_quantiles(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut args = &afa.args;
    expect_at_least_n_args(args, 3)?;

    let mut dst_label: String;

    match get_string(&args[0], 0) {
        Ok(label) => dst_label = label,
        Err(err) => {
            return Err(RuntimeError::ArgumentError(format!("cannot obtain dst_label: {:?}", err)))
        }
    }

    let phi_args = &args[1 .. args.len()-1];
    
    // todo: smallvec
    let mut phis: Vec<f64> = Vec::with_capacity(phi_args.len());
    for (i, phiArg) in phi_args.iter().enumerate() {
        let phis_local = get_scalar(phiArg, i+1)?;
        if phis.len() == 0 {
            panic!("BUG: expecting at least a single sample")
        }
        phis.push(phis_local[0]);
    }
    let mut arg_orig = &args[args.len() - 1];
    let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        // todo: smallvec
        let tss_dst: Vec<Timeseries> = Vec::with_capacity(phi_args.len());
        for j in 0 .. tss_dst.len() {
            let mut ts = Timeseries::copy_from_shallow_timestamps(&tss[0]);
            ts.metric_name.remove_tag(&dst_label);
            // TODO
            ts.metric_name.add_tag(&dst_label, &format!("{}", phis[j]));
            tss_dst[j] = ts;
        }

        let mut _vals = tiny_vec!([f64; 10]);
        let mut qs = get_float64s(phi_args.len());

        let mut values = get_float64s(phi_args.len());
        for n in tss[0].values {
            values.clear();
            let idx = n as usize; // todo: handle panic
            for ts in tss {
                values.push(ts.values[idx]);
            }
            quantiles(qs.deref_mut(), &phis, values.deref());

            for j in 0 .. tss_dst.len() {
                tss_dst[j].values[idx] = qs[j];
            }
        }
        return tss_dst
    };

    return aggr_func_ext(afe, &mut arg_orig, &afa.ae.modifier, afa.ae.limit, false)
}

fn aggr_func_quantile(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&afa.args, 2)?;
    let phis = get_scalar(&afa.args[0], 0)?;
    let afe = new_aggr_quantile_func(phis);
    return aggr_func_ext(afe, &mut afa.args[1], &afa.ae.modifier, afa.ae.limit, false);
}

fn aggr_func_median(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
    let phis = &eval_number(&mut afa.ec, 0.5)[0].values;
    let afe = new_aggr_quantile_func(phis);
    return aggr_func_ext(afe, &mut tss, &afa.ae.modifier, afa.ae.limit, false);
}

fn new_aggr_quantile_func(phis: &Vec<f64>) -> fn(tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>) -> Vec<Timeseries> {
    |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| -> Vec<Timeseries> {
        let mut h = get_pooled_histogram();

        for n in 0 .. tss[0].values.len() {
            h.reset();
            for ts in tss.iter() {
                let v = ts.values[n];
                if !v.is_nan() {
                    h.update(v);
                }
            }
            let phi = phis[n];
            tss[0].values[n] = h.quantile(phi);
        }
        return tss
    }
}

fn less_with_nans(a: f64, b: f64) -> bool {
    if a.is_nan() {
        return !b.is_nan();
    }
    return a < b;
}

// quantiles calculates the given phis from originValues without modifying originValues, appends them to qs and returns the result.
pub fn quantiles(qs: &mut [f64], phis: &[f64], origin_values: &[f64]) {
    if origin_values.len() <= 64 {
        let mut vec = tiny_vec!([f64; 64]);
        prepare_tv_for_quantile_float64(&mut vec, origin_values);
        return quantiles_sorted(qs, phis, &vec)
    }

    let mut block = get_float64s(phis.len());
    let a = block.deref_mut();
    prepare_for_quantile_float64(a, origin_values);
    quantiles_sorted(qs, phis, a)
}

// quantile calculates the given phi from originValues without modifying originValues
pub fn quantile(phi: f64, origin_values: &[f64]) -> f64 {
    // todo: smallvec
    let mut block = get_float64s(origin_values.len());
    let a = block.deref_mut();
    prepare_for_quantile_float64(a, origin_values);
    quantile_sorted(phi, a)
}

/// prepare_for_quantile_float64 copies items from src to dst but removes NaNs and sorts the dst
fn prepare_for_quantile_float64(dst: &mut Vec<f64>, src: &[f64]) {
    for v in src {
        if v.is_nan() {
            continue
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// copies items from src to dst but removes NaNs and sorts the dst
fn prepare_tv_for_quantile_float64(dst: &mut TinyVec<[f64; 64]>, src: &[f64]) {
    for v in src {
        if v.is_nan() {
            continue
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// calculates the given phis over a sorted list of values, appends them to qs and returns the result.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub fn quantiles_sorted(qs: &mut [f64], phis: &[f64], values: &[f64]) {
    for (i, phi) in phis.iter().enumerate() {
        qs[i] = quantile_sorted(*phi, values);
    }
}

/// quantile_sorted calculates the given quantile over a sorted list of values.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub fn quantile_sorted(phi: f64, values: &[f64]) -> f64 {
    if values.len() == 0 || phi.is_nan() {
        return f64::NAN
    }
    if phi < 0.0 {
        return f64::NEG_INFINITY;
    }
    if phi > 1.0 {
        return f64::INFINITY;
    }
    let n = values.len() as f64;
    let rank = phi * (n - 1.0);

    let lower_index = std::cmp::max(0, rank.floor() as usize);
    let upper_index = std::cmp::min(n-1.0, (lower_index + 1) as f64) as usize;

    let weight = rank - rank.floor();
    return values[lower_index]*(1.0-weight) + values[upper_index]*weight
}


static AGGR_FUNCS: phf::Map<&'static str, AggrFunc> = phf_map! {
// See https => //prometheus.io/docs/prometheus/latest/querying/operators/#aggregation-operators
    "sum" =>           new_aggr_func(aggr_func_sum),
    "min" =>           new_aggr_func(aggr_func_min),
    "max" =>           new_aggr_func(aggr_func_max),
    "avg" =>           new_aggr_func(aggr_func_avg),
    "stddev" =>        new_aggr_func(aggr_func_stddev),
    "stdvar" =>        new_aggr_func(aggr_func_stdvar),
    "count" =>         new_aggr_func(aggr_func_count),
    "count_values" =>  aggr_func_count_values,
    "bottomk" =>       new_aggr_func_topk(true),
    "topk" =>          new_aggr_func_topk(false),
    "quantile" =>      aggr_func_quantile,
    "quantiles" =>     aggr_func_quantiles,
    "group" =>         new_aggr_func(aggr_func_group),

    // PromQL extension funcs
    "median" =>          aggr_func_median,
    "limitk" =>          aggr_func_limitk,
    "distinct" =>        new_aggr_func(aggr_func_distinct),
    "sum2" =>            new_aggr_func(aggr_func_sum2),
    "geomean" =>         new_aggr_func(aggr_func_geomean),
    "histogram" =>       new_aggr_func(aggr_func_histogram),
    "topk_min" =>        new_aggr_func_range_topk(min_value, false),
    "topk_max" =>        new_aggr_func_range_topk(max_value, false),
    "topk_avg" =>        new_aggr_func_range_topk(avg_value, false),
    "topk_median" =>     new_aggr_func_range_topk(median_value, false),
    "bottomk_min" =>     new_aggr_func_range_topk(min_value, true),
    "bottomk_max" =>     new_aggr_func_range_topk(max_value, true),
    "bottomk_avg" =>     new_aggr_func_range_topk(avg_value, true),
    "bottomk_median" =>  new_aggr_func_range_topk(median_value, true),
    "any" =>             aggr_func_any,
    "outliersk" =>       aggr_func_outliers_k,
    "mode" =>            new_aggr_func(aggr_func_mode),
    "zscore" =>          aggr_func_zscore,
};


pub fn get_aggr_func(name: &str) -> Option<&'static AggrFunc> {
    let lower = name.to_lowercase().as_str();
    return AGGR_FUNCS.get(lower);
}


fn expect_transform_args_num(args: &Vec<Vec<Timeseries>>, expected_num: usize) -> RuntimeResult<()> {
    if args.len() == expected_num {
        return Ok(());
    }
    return Err(RuntimeError::ArgumentError(format!("unexpected number of args; got {}; want {}", args.len(), expected_num)));
}

fn expect_at_least_n_args(tfa: &Vec<Vec<Timeseries>>, n: usize) -> RuntimeResult<()> {
    let len = tfa.len();
    if len < n {
        let err = format!("not enough args; got {}; want at least {}", len, n);
        return Err(RuntimeError::ArgumentError(err));
    }
    Ok(())
}


/// returns a copy of tss with real copy of metric_names,
/// but with shallow copy of Timestamps and Values if make_copy is set.
///
/// Otherwise tss is returned.
/// todo: COW
fn copy_timeseries_metric_names(tss: &Vec<Timeseries>, make_copy: bool) -> Vec<Timeseries> {
    if !make_copy {
        return tss
    }
    tss.iter().map(Timeseries::copy_from_metric_name).collect()
}

fn get_scalar(arg: &[Timeseries], arg_num: usize) -> RuntimeResult<&Vec<f64>> {
    if arg.len() != 1 {
        let msg = format!("arg # {} must contain a single timeseries; got {} timeseries", arg_num + 1, arg.len());
        Err(RuntimeError::ArgumentError(msg))
    }
    Ok(&arg[0].values)
}
