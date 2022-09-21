use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::ops::{Deref, DerefMut};

use lockfree_object_pool::LinearReusable;
use phf::phf_map;
use tinyvec::*;
use xxhash_rust::xxh3::Xxh3;

use lib::{get_float64s, get_pooled_buffer};
use metricsql::ast::{AggregateModifier, AggregateModifierOp};
use metricsql::functions::AggregateFunction;

use crate::{EvalConfig, Timeseries};
use crate::eval::eval_number;
use crate::exec::remove_empty_series;
use crate::functions::{mode_no_nans, quantile, quantiles, skip_trailing_nans};
use crate::functions::transform::vmrange_buckets_to_le;
use crate::functions::types::ParameterValue;
use crate::histogram::{get_pooled_histogram, Histogram};
use crate::runtime_error::{RuntimeError, RuntimeResult};

// todo: add lifetime so we dont need to copy modifier
pub struct AggrFuncArg<'a> {
    pub(crate) args: Vec<ParameterValue>,
    pub(crate) ec: &'a EvalConfig,
    pub(crate) modifier: Option<AggregateModifier>,
    pub(crate) limit: usize
}

impl<'a> AggrFuncArg<'a> {
    pub fn new(ec: &'a EvalConfig,
               args: Vec<ParameterValue>, 
               modifier: &Option<AggregateModifier>,
               limit: usize) -> Self {
        Self {
            args,
            ec,
            modifier: modifier.clone(),
            limit
        }
    }
}

pub type AggrFunctionResult = RuntimeResult<Vec<Timeseries>>;

pub trait AggrFn: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}
impl<T> AggrFn for T where T: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}


fn new_aggr_func(afe: fn(tss: &mut Vec<Timeseries>)) -> impl AggrFn {
    move |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let mut tss = get_aggr_timeseries(&afa.args)?;
        aggr_func_ext(move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
            afe(tss);
            tss.shrink_to(1);
            std::mem::take(tss)
        }, // todo: avoid cloning
                      &mut tss, &afa.modifier, afa.limit, false)
    }
}

fn get_aggr_timeseries(args: &Vec<ParameterValue>) -> RuntimeResult<&mut Vec<Timeseries>> {
    let mut tss = args[0].get_series_mut();
    for i in 1 .. args.len() {
        let other = args[i].get_series_mut();
        tss.append(other);
    }
    Ok(tss)
}


trait AggrFnExt: FnMut(&mut Vec<Timeseries>, &Option<AggregateModifier>) -> Vec<Timeseries> {}
impl<T> AggrFnExt for T where T: FnMut(&mut Vec<Timeseries>, &Option<AggregateModifier>) -> Vec<Timeseries> {}

fn aggr_func_ext(
    afe: impl AggrFnExt,
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
        ts.metric_name.remove_group_tags(modifier);
        let key = ts.metric_name.marshal_to_string(bb.deref_mut());

        if keep_original {
            std::mem::swap(&mut ts, &mut arg_orig[i]);
        }

        let k = key.as_ref();

        match m.entry(k) {
            Vacant(ve) => {
                if max_series > 0 && m.len() >= max_series {
                    // We already reached time series limit after grouping. Skip other time series.
                    continue;
                }
                let value = vec![ts];
                ve.insert(value);
            },
            Occupied(oe) => {
                oe.get().push(ts);
            }
        }

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
    let afe = move |tss: &mut Vec<Timeseries>, modifier| {
        tss.drain(0..).collect()
    };
    let mut limit = afa.limit;
    if limit > 1 {
        // Only a single time series per group must be returned
        limit = 1
    }
    aggr_func_ext(afe, &mut tss, &afa.modifier, limit, true)
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

fn aggr_func_histogram(tss: &mut Vec<Timeseries>) {
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
                    entry.insert(ts)
                },
                Occupied(mut entry) => entry.get_mut(),
            };
            ts.values[i] = bucket.count as f64;
        }
    }

    let mut res: Vec<Timeseries> = m.into_values().collect();
    let mut series = vmrange_buckets_to_le(&mut res);
    std::mem::swap(tss, &mut series);
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

    aggr_func_ext(afe, &mut tss, &afa.modifier, afa.limit, true)
}


fn aggr_func_count_values(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = afa.args[0].get_str()?;

    // Remove dst_label from grouping like Prometheus does.
    if let Some(mut modifier) = &afa.modifier {
        match modifier.op {
            AggregateModifierOp::Without => {
                modifier.args.push( dst_label.to_string());
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

    let mut series = get_series(afa, 1)?;
    aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, false)
}

fn new_aggr_func_topk(is_reverse: bool) -> impl AggrFn {
    move |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let ks = afa.args[0].get_vector()?;
        let (left, right) = afa.args.split_at_mut(1);

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

            std::mem::take(tss)
        };

        let mut series = get_series_mut(afa, 1)?;
        aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
    }
}


fn new_aggr_func_range_topk(f: fn(values: &[f64]) -> f64, is_reverse: bool) -> impl AggrFn {
    return move |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let args_len = afa.args.len();
        let ks: &Vec<f64> = afa.args[0].get_vector()?;

        let remaining_sum_tag_name = if args_len == 3 {
            afa.args[2].get_str()?
        } else {
            ""
        };

        let afe = move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
            return get_range_topk_timeseries(tss,
                                             modifier,
                                             &ks,
                                             remaining_sum_tag_name,
                                             f,
                                             is_reverse);
        };

        let mut series = get_series_mut(afa, 1)?;
        aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
    };
}


fn get_range_topk_timeseries<F>(tss: &mut Vec<Timeseries>,
                             modifier: &Option<AggregateModifier>,
                             ks: &Vec<f64>,
                             remaining_sum_tag_name: &str,
                             f: F,
                             is_reverse: bool) -> Vec<Timeseries>
where F: Fn(&[f64]) -> f64,
{
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
    maxs.sort_by(move |first, second| {
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
    return tss.to_vec();
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
    dst.metric_name.remove_group_tags(modifier);

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
    quantile(0.5, values)
}

fn last_value(values: &[f64]) -> f64 {
    let vals = skip_trailing_nans(values);
    if vals.len() == 0 {
        return f64::NAN;
    }
    values[values.len() - 1]
}

fn aggr_func_outliersk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let ks = afa.args[0].get_vector()?;
    let afe = move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        // Calculate medians for each point across tss.
        let medians = get_per_point_medians(tss);

        // Return topK time series with the highest variance from median.
        let f = move |values: &[f64]| -> f64 {
            let mut sum2 = 0_f64;
            for (n, v) in values.iter().enumerate() {
                let d = v - medians[n];
                sum2 += d * d;
            }
            sum2
        };

        return get_range_topk_timeseries(tss, &afa.modifier, &ks, "", f, false);
    };

    let mut series = get_series(afa, 1)?;
    aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
}

fn aggr_func_limitk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut limit = afa.args[0].get_int()?;
    if limit < 0 {
        limit = 0
    }

    let limit = limit as usize;

    let afe =  move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        let mut hasher: Xxh3 = Xxh3::new();
        let mut buf = get_pooled_buffer(256);

        let mut map: BTreeMap<u64, Timeseries> = BTreeMap::new();

        // Sort series by metricName hash in order to get consistent set of output series
        // across multiple calls to limitk() function.
        // Sort series by hash in order to guarantee uniform selection across series.
        for ts in tss.into_iter() {
            ts.metric_name.marshal(&mut buf);
            hasher.update(buf.as_slice());
            let digest = hasher.digest();
            map.insert(digest, std::mem::take(ts));
            buf.clear()
        }

        let res = map.into_values().take(limit).collect::<Vec<_>>();
        return res
    };

    let mut series = get_series(afa, 1)?;
    aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
}


fn aggr_func_quantiles(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut dst_label = afa.args[0].get_string()?;

    // todo: smallvec
    let mut phis: Vec<f64> = Vec::with_capacity(afa.args.len() - 2);
    for i in 1..afa.args.len() - 1 {
        phis.push(afa.args[i].get_float()?);
    }

    let afe = |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        // todo: smallvec
        let tss_dst: Vec<Timeseries> = Vec::with_capacity(phis.len());
        for j in 0..tss_dst.len() {
            let mut ts = Timeseries::copy_from_shallow_timestamps(&tss[0]);
            ts.metric_name.remove_tag(&dst_label);
            // TODO
            ts.metric_name.add_tag(&dst_label, &format!("{}", phis[j]));
            tss_dst[j] = ts;
        }

        let mut _vals = tiny_vec!([f64; 10]);
        let mut qs = get_float64s(phis.len());

        let mut values = get_float64s(phis.len());
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

    let mut last = afa.args.remove(afa.args.len() - 1).get_series();
    aggr_func_ext(afe, &mut last, &afa.modifier, afa.limit, false)
}

fn aggr_func_quantile(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (left, orig) = afa.args.split_at_mut(1);
    let phis = left[0].get_vector()?;
    let afe = new_aggr_quantile_func(phis);
    let mut orig = orig[0].get_series();
    return aggr_func_ext(afe, &mut orig, &afa.modifier, afa.limit, false);
}

fn aggr_func_median(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
    let phis = &eval_number(&afa.ec, 0.5)[0].values; // todo: use more efficient method
    let afe = new_aggr_quantile_func(phis);
    return aggr_func_ext(afe, &mut tss, &afa.modifier, afa.limit, false);
}

fn new_aggr_quantile_func(phis: &Vec<f64>) -> impl AggrFnExt + '_ {
    move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| -> Vec<Timeseries> {

        let count = tss[0].values.len();
        let mut values = get_float64s(count);

        for n in 0 .. count {
            for ts in tss.iter() {
                values.push(ts.values[n]);
            }

            tss[0].values[n] = quantile(phis[n], &values);
            values.clear();
        }

        return std::mem::take(tss)
    }
}

fn aggr_func_MAD(tss: &mut Vec<Timeseries>) {
    // Calculate medians for each point across tss.
    let medians = get_per_point_medians(tss);
    // Calculate MAD values multiplied by tolerance for each point across tss.
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    let mads = get_per_point_mads(tss, &medians);

    tss[0].values.extend_from_slice(&mads);
}

fn aggr_func_outliers_MAD(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tolerances = afa.args[0].get_vector().unwrap();

    let afe = move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        // Calculate medians for each point across tss.
        let medians = get_per_point_medians(tss);
        // Calculate MAD values multiplied by tolerance for each point across tss.
        // See https://en.wikipedia.org/wiki/Median_absolute_deviation
        let mut mads = get_per_point_mads(tss, &medians);
        let mut n = 0;
        for mad in mads.iter_mut() {
            *mad *= tolerances[n];
            n += 1;
        }

        // Leave only time series with at least a single peak above the MAD multiplied by tolerance.
        tss.retain(|ts| {
            for (n, v) in ts.values.iter().enumerate() {
                let ad = (v - medians[n]).abs();
                let mad = mads[n];
                if ad > mad {
                    return true;
                }
            }
            false
        });

        std::mem::take(tss)
    };

    let mut series = get_series(afa, 1)?;
    aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
}

fn get_per_point_medians(tss: &mut Vec<Timeseries>) -> Vec<f64> {
    if tss.len() == 0 {
        // logger.Panicf("BUG: expecting non-empty tss")
    }
    let medians = Vec::with_capacity(tss[0].values.len());
    let mut values = get_float64s(medians.len());
    for n in 0 .. medians.len() {
        values.clear();
        for j in 0 .. tss.len() {
            let v = tss[j].values[n];
            if !v.is_nan() {
                values.push(v);
            }
        }
        medians[n] = quantile(0.5, &values)
    }

    medians
}

fn get_per_point_mads(tss: &Vec<Timeseries>, medians: &[f64]) -> Vec<f64> {
    let mut mads: Vec<f64> = Vec::with_capacity(medians.len());
    // todo: tinyvec
    let mut values = get_float64s(mads.len());
    for (n, median) in medians.iter().enumerate() {
        values.clear();
        for j in 0 .. tss.len() {
            let v = tss[j].values[n];
            if !v.is_nan() {
                let ad = (v - median).abs();
                values.push(ad);
            }
        }
        mads[n] = quantile(0.5, &values)
    }
    mads
}

macro_rules! create_aggr_fn {
    ($f:expr) => {
        Box::new(new_aggr_func($f))
    }
}

macro_rules! create_range_fn {
    ($f:expr, $top:expr) => {
        Box::new(new_aggr_func_range_topk($f, $top))
    }
}

static AGGR_FUNCS: phf::Map<&'static str, Box<dyn AggrFn>> = phf_map! {
// See https://prometheus.io/docs/prometheus/latest/querying/operators/#aggregation-operators
    "sum" =>           create_aggr_fn!(aggr_func_sum),
    "min" =>           create_aggr_fn!(aggr_func_min),
    "max" =>           create_aggr_fn!(aggr_func_max),
    "avg" =>           create_aggr_fn!(aggr_func_avg),
    "stddev" =>        create_aggr_fn!(aggr_func_stddev),
    "stdvar" =>        create_aggr_fn!(aggr_func_stdvar),
    "count" =>         create_aggr_fn!(aggr_func_count),
    "count_values" =>  Box::new(aggr_func_count_values),
    "bottomk" =>       Box::new(new_aggr_func_topk(true)),
    "topk" =>          Box::new(new_aggr_func_topk(false)),
    "quantile" =>      Box::new(aggr_func_quantile),
    "quantiles" =>     Box::new(aggr_func_quantiles),
    "group" =>         create_aggr_fn!(aggr_func_group),

    // PromQL extension funcs
    "median" =>          Box::new(aggr_func_median),
    "mad" =>             create_aggr_fn!(aggr_func_MAD),
    "limitk" =>          Box::new(aggr_func_limitk),
    "distinct" =>        create_aggr_fn!(aggr_func_distinct),
    "sum2" =>            create_aggr_fn!(aggr_func_sum2),
    "geomean" =>         create_aggr_fn!(aggr_func_geomean),
    "histogram" =>       Box::new(new_aggr_func(aggr_func_histogram)),
    "topk_min" =>        create_range_fn!(min_value, false),
    "topk_max" =>        create_range_fn!(max_value, false),
    "topk_avg" =>        create_range_fn!(avg_value, false),
    "topk_last" =>       create_range_fn!(last_value, false),
    "topk_median" =>     create_range_fn!(median_value, false),
    "bottomk_min" =>     create_range_fn!(min_value, true),
    "bottomk_max" =>     create_range_fn!(max_value, true),
    "bottomk_avg" =>     create_range_fn!(avg_value, true),
    "bottomk_last" =>    create_range_fn!(last_value, true),
    "bottomk_median" =>  create_range_fn!(median_value, true),
    "any" =>             Box::new(aggr_func_any),
    "outliersk" =>       Box::new(aggr_func_outliersk),
    "outliers_mad" =>    Box::new(aggr_func_outliers_MAD),
    "mode" =>            create_aggr_fn!(aggr_func_mode),
    "zscore" =>          Box::new(aggr_func_zscore),
};

pub fn get_aggr_func(op: &AggregateFunction) -> &'static Box<dyn AggrFn> {
    get_aggr_func_by_name(format!("{}", op).as_str()).unwrap()
}

pub fn get_aggr_func_by_name(name: &str) -> RuntimeResult<&Box<dyn AggrFn>> {
    let lower = name.to_lowercase().as_str();
    match AGGR_FUNCS.get(lower) {
        Some(handler) => Ok(handler),
        None => Err( RuntimeError::UnknownFunction(name.to_string())) 
    }
}

/// returns a copy of tss with real copy of metric_names,
/// but with shallow copy of Timestamps and Values if make_copy is set.
///
/// Otherwise tss is returned.
/// todo: COW
fn copy_timeseries_metric_names<'a>(tss: &Vec<Timeseries>, make_copy: bool) -> Cow<'a, Vec<Timeseries>> {
    if !make_copy {
        return Cow::Borrowed(tss)
    }
    Cow::Owned(tss.iter().map(Timeseries::copy_from_metric_name).collect())
}

fn get_scalar(arg: &[Timeseries], arg_num: usize) -> RuntimeResult<&Vec<f64>> {
    if arg.len() != 1 {
        let msg = format!("arg # {} must contain a single timeseries; got {} timeseries", arg_num + 1, arg.len());
        return Err(RuntimeError::ArgumentError(msg))
    }
    Ok(&arg[0].values)
}

fn get_series<'a>(tfa: &'a AggrFuncArg, arg_num: usize) -> RuntimeResult<&'a Vec<Timeseries>> {
    let tss = tfa.args[arg_num].get_series();
    Ok(tss)
}

fn get_series_mut<'a>(tfa: &'a AggrFuncArg, arg_num: usize) -> RuntimeResult<&'a mut Vec<Timeseries>> {
    let tss = tfa.args[arg_num].get_series_mut();
    Ok(tss)
}