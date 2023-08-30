use std::borrow::BorrowMut;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};

use lockfree_object_pool::LinearReusable;

use lib::{get_pooled_vec_f64, get_pooled_vec_f64_filled};
use metricsql::common::AggregateModifier;
use metricsql::functions::AggregateFunction;

use crate::common::math::{mode_no_nans, quantile, quantiles};
use crate::execution::{eval_number, remove_empty_series, EvalConfig};
use crate::functions::arg_parse::{
    get_float_arg, get_int_arg, get_scalar_arg_as_vec, get_series_arg, get_string_arg,
};
use crate::functions::skip_trailing_nans;
use crate::functions::transform::vmrange_buckets_to_le;
use crate::functions::utils::float_to_int_bounded;
use crate::histogram::{get_pooled_histogram, Histogram};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::signature::Signature;
use crate::{QueryValue, Timeseries};

const MAX_SERIES_PER_AGGR_FUNC: usize = 100000;

// todo: add lifetime so we dont need to copy modifier
pub struct AggrFuncArg<'a> {
    pub args: Vec<QueryValue>,
    pub ec: &'a EvalConfig,
    pub modifier: &'a Option<AggregateModifier>,
    pub limit: usize,
}

pub trait AggrFn: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

impl<T> AggrFn for T where T: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

macro_rules! make_aggr_fn {
    ($name:ident, $f:expr) => {
        fn $name(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            aggr_func_impl($f, afa)
        }
    };
}

macro_rules! make_range_fn {
    ($name: ident, $f:expr, $top:expr) => {
        fn $name(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            range_topk_impl(afa, $f, $top)
        }
    };
}

make_aggr_fn!(aggregate_avg, aggr_func_avg);
make_aggr_fn!(aggregate_count, aggr_func_count);
make_aggr_fn!(aggregate_distinct, aggr_func_distinct);
make_aggr_fn!(aggregate_geo_mean, aggr_func_geomean);
make_aggr_fn!(aggregate_group, aggr_func_group);
make_aggr_fn!(aggregate_histogram, aggr_func_histogram);
make_aggr_fn!(aggregate_min, aggr_func_min);
make_aggr_fn!(aggregate_mad, aggr_func_mad);
make_aggr_fn!(aggregate_max, aggr_func_max);
make_aggr_fn!(aggregate_mode, aggr_func_mode);
make_aggr_fn!(aggregate_stddev, aggr_func_stddev);
make_aggr_fn!(aggregate_stdvar, aggr_func_stdvar);
make_aggr_fn!(aggregate_sum, aggr_func_sum);
make_aggr_fn!(aggregate_sum2, aggr_func_sum2);

make_range_fn!(aggregate_bottomk_avg, avg_value, true);
make_range_fn!(aggregate_bottomk_last, last_value, true);
make_range_fn!(aggregate_bottomk_median, median_value, true);
make_range_fn!(aggregate_bottomk_min, min_value, true);
make_range_fn!(aggregate_bottomk_max, max_value, true);

make_range_fn!(aggregate_topk_avg, avg_value, false);
make_range_fn!(aggregate_topk_last, last_value, false);
make_range_fn!(aggregate_topk_min, min_value, false);
make_range_fn!(aggregate_topk_max, max_value, false);
make_range_fn!(aggregate_topk_median, median_value, false);

fn aggregate_bottomk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    func_topk_impl(afa, true)
}

fn aggregate_topk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    func_topk_impl(afa, false)
}

pub(crate) fn exec_aggregate_fn(
    function: AggregateFunction,
    afa: &mut AggrFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    use AggregateFunction::*;

    match function {
        Any => aggr_func_any(afa),
        Avg => aggregate_avg(afa),
        Bottomk => aggregate_bottomk(afa),
        BottomkAvg => aggregate_bottomk_avg(afa),
        BottomkLast => aggregate_bottomk_last(afa),
        BottomkMax => aggregate_bottomk_max(afa),
        BottomkMedian => aggregate_bottomk_median(afa),
        BottomkMin => aggregate_bottomk_min(afa),
        Count => aggregate_count(afa),
        CountValues => aggr_func_count_values(afa),
        Distinct => aggregate_distinct(afa),
        GeoMean => aggregate_geo_mean(afa),
        Group => aggregate_group(afa),
        Histogram => aggregate_histogram(afa),
        Limitk => aggr_func_limitk(afa),
        MAD => aggregate_mad(afa),
        Median => aggr_func_median(afa),
        Mode => aggregate_mode(afa),
        Share => aggr_func_share(afa),
        Sum => aggregate_sum(afa),
        Sum2 => aggregate_sum2(afa),
        Min => aggregate_min(afa),
        Max => aggregate_max(afa),
        Outliersk => aggr_func_outliersk(afa),
        OutliersMAD => aggr_func_outliers_mad(afa),
        Quantile => aggr_func_quantile(afa),
        Quantiles => aggr_func_quantiles(afa),
        StdDev => aggregate_stddev(afa),
        StdVar => aggregate_stdvar(afa),
        Topk => aggregate_topk(afa),
        TopkAvg => aggregate_topk_avg(afa),
        TopkLast => aggregate_topk_last(afa),
        TopkMax => aggregate_topk_max(afa),
        TopkMedian => aggregate_topk_median(afa),
        TopkMin => aggregate_topk_min(afa),
        ZScore => aggr_func_zscore(afa),
    }
}

fn aggr_func_impl(
    afe: fn(tss: &mut Vec<Timeseries>),
    arg: &mut AggrFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(arg)?;
    aggr_func_ext(
        move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| {
            afe(tss);
            std::mem::take(tss)
        }, // todo: avoid cloning
        &mut tss,
        arg.modifier,
        arg.limit,
        false,
    )
}

fn get_aggr_timeseries(tfa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.is_empty() {
        return Err(RuntimeError::ArgumentError(
            "aggregate function requires at least one argument".to_string(),
        ));
    }
    let mut tss = tfa.args[0].get_instant_vector(tfa.ec)?;
    for arg in tfa.args.iter_mut().skip(1) {
        let mut other = arg.get_instant_vector(tfa.ec)?;
        tss.append(&mut other);
    }
    Ok(tss)
}

trait AggrFnExt: FnMut(&mut Vec<Timeseries>, &Option<AggregateModifier>) -> Vec<Timeseries> {}

impl<T> AggrFnExt for T where
    T: FnMut(&mut Vec<Timeseries>, &Option<AggregateModifier>) -> Vec<Timeseries>
{
}

fn aggr_func_ext(
    mut afe: impl AggrFnExt,
    arg_orig: &mut Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    max_series: usize,
    keep_original: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    // Perform grouping.
    let mut series_by_name = aggr_prepare_series(arg_orig, modifier, max_series, keep_original);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(series_by_name.len());
    for tss in series_by_name.values_mut() {
        let mut rv = afe(tss, modifier);
        rvs.append(&mut rv);
    }

    Ok(rvs)
}

fn aggr_prepare_series(
    arg_orig: &mut Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    max_series: usize,
    keep_original: bool,
) -> HashMap<Signature, Vec<Timeseries>> {
    // Remove empty time series, e.g. series with all NaN samples,
    // since such series are ignored by aggregate functions.
    remove_empty_series(arg_orig);

    let capacity = arg_orig.len();

    // Perform grouping.
    let mut m = HashMap::with_capacity(capacity);
    for mut ts in arg_orig.drain(0..) {
        let (series, k) = if keep_original {
            let mut mn = ts.metric_name.clone();
            mn.remove_group_tags(modifier);
            (ts, mn.signature())
        } else {
            ts.metric_name.remove_group_tags(modifier);
            let sig = ts.metric_name.signature();
            (ts, sig)
        };

        let len = m.len();
        match m.entry(k) {
            Entry::Vacant(entry) => {
                if max_series > 0 && len >= max_series {
                    // We already reached time series limit after grouping. Skip other time series.
                    continue;
                }
                let mut tss: Vec<Timeseries> = Vec::with_capacity(4);
                tss.push(series);
                entry.insert(tss);
            }
            Entry::Occupied(mut entry) => {
                let tss = entry.get_mut();
                tss.push(series);
            }
        }
    }
    m
}

fn aggr_func_any(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(afa)?;
    let afe = move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| {
        if !tss.is_empty() {
            let ts = std::mem::take(&mut tss[0]);
            return vec![ts];
        }
        vec![]
    };
    // Only a single time series per group must be returned
    let mut limit = afa.limit;
    if limit > 1 {
        limit = 1;
    }
    aggr_func_ext(afe, &mut tss, afa.modifier, limit, true)
}

fn aggr_func_group(tss: &mut Vec<Timeseries>) {
    for i in 0..tss[0].values.len() {
        let mut v = f64::NAN;
        for ts in tss.iter() {
            if !ts.values[i].is_nan() {
                v = 1f64;
                // todo: break ?
            }
        }
        tss[0].values[i] = v;
    }

    tss.truncate(1);
}

fn aggr_func_sum(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        return;
    }

    for i in 0..tss[0].values.len() {
        let mut sum: f64 = 0.0;
        let mut count: usize = 0;
        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            sum += v;
            count += 1;
        }

        if count == 0 {
            sum = f64::NAN
        }

        tss[0].values[i] = sum;
    }

    tss.truncate(1);
}

fn aggr_func_sum2(tss: &mut Vec<Timeseries>) {
    for i in 0..tss[0].values.len() {
        let mut sum2: f64 = 0.0;
        let mut count: usize = 0;
        for ts in tss.iter() {
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

        tss[0].values[i] = sum2;
    }

    tss.truncate(1);
}

fn aggr_func_geomean(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to geomean.
        return;
    }
    for i in 0..tss[0].values.len() {
        let mut p = 1.0;
        let mut count = 0;
        for ts in tss.iter() {
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

    tss.truncate(1);
}

fn aggr_func_histogram(tss: &mut Vec<Timeseries>) {
    let mut h: LinearReusable<Histogram> = get_pooled_histogram();
    let mut m: HashMap<String, Timeseries> = HashMap::new();
    for i in 0..tss[0].values.len() {
        h.reset();
        for ts in tss.iter() {
            let v = ts.values[i];
            h.update(v)
        }

        for bucket in h.non_zero_buckets() {
            let ts = m.entry(bucket.vm_range.to_string()).or_insert_with(|| {
                let mut ts = tss[0].clone();
                ts.metric_name.remove_tag("vmrange");
                ts.metric_name.set_tag("vmrange", bucket.vm_range);

                // todo(perf): should be a more efficient way to do this
                for k in 0..ts.values.len() {
                    ts.values[k] = 0.0;
                }
                ts
            });
            ts.values[i] = bucket.count as f64;
        }
    }

    let res: Vec<Timeseries> = m.into_values().collect();
    let mut series = vmrange_buckets_to_le(res);
    std::mem::swap(tss, &mut series);
}

fn aggr_func_min(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to min.
        return;
    }

    for i in 0..tss[0].values.len() {
        let min = tss[0].values[i];
        for j in 0..tss.len() {
            let v = tss[j].values[i];
            if min.is_nan() || v < min {
                tss[0].values[i] = v;
            }
        }
    }

    tss.truncate(1);
}

fn aggr_func_max(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to max.
        return;
    }

    for i in 0..tss[0].values.len() {
        let max = tss[0].values[i];

        for j in 0..tss.len() {
            let v = tss[j].values[i];
            if max.is_nan() || v > max {
                tss[0].values[i] = v;
            }
        }
    }

    tss.truncate(1);
}

fn aggr_func_avg(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - nothing to avg.
        return;
    }

    for j in 0..tss[0].values.len() {
        // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
        // since it is slower and has no obvious benefits in increased precision.
        let mut sum: f64 = 0.0;
        let mut count: usize = 0;
        for ts in tss.iter() {
            let v = ts.values[j];
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
        tss[0].values[j] = avg;
    }

    tss.truncate(1);
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

fn aggr_func_stdvar(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        // Fast path - stdvar over a single time series is zero
        for v in tss[0].values.iter_mut() {
            if !v.is_nan() {
                *v = 0.0;
            }
        }
        return;
    }

    for k in 0..tss[0].values.len() {
        // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
        let mut avg: f64 = 0.0;
        let mut count: f64 = 0.0;
        let mut q = 0.0;

        for ts in tss.iter() {
            let v = ts.values[k];
            if v.is_nan() {
                continue;
            }
            count += 1.0;
            let avg_new = avg + (v - avg) / count;
            q += (v - avg) * (v - avg_new);
            avg = avg_new
        }

        tss[0].values[k] = if count == 0.0 { f64::NAN } else { q / count };
    }

    tss.truncate(1);
}

fn aggr_func_count(tss: &mut Vec<Timeseries>) {
    for i in 0..tss[0].values.len() {
        let mut count = 0;
        for (j, ts) in tss.iter().enumerate() {
            if ts.values[j].is_nan() {
                continue;
            }
            count += 1;
        }
        let mut v: f64 = count as f64;
        if count == 0 {
            v = f64::NAN
        }

        tss[0].values[i] = v;
    }

    tss.truncate(1);
}

fn aggr_func_distinct(tss: &mut Vec<Timeseries>) {
    let mut values: Vec<f64> = Vec::with_capacity(tss.len());

    // todo: split_mut()]
    for i in 0..tss[0].values.len() {
        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            values.push(v);
        }
        values.sort_by(|a, b| a.total_cmp(b));
        values.dedup();
        let n = values.len();
        tss[0].values[i] = if n == 0 { f64::NAN } else { n as f64 };

        values.clear();
    }

    tss.truncate(1);
}

fn aggr_func_mode(tss: &mut Vec<Timeseries>) {
    let mut a: Vec<f64> = Vec::with_capacity(tss.len());
    for i in 0..tss[0].values.len() {
        for ts in tss.iter() {
            let v = ts.values[i];
            if !v.is_nan() {
                a.push(v);
            }
        }
        tss[0].values[i] = mode_no_nans(f64::NAN, &mut a);
        a.clear();
    }

    tss.truncate(1);
}

fn aggr_func_share(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(afa)?;

    let afe = |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| -> Vec<Timeseries> {
        for i in 0..tss[0].values.len() {
            // Calculate sum for non-negative points at position i.
            let mut sum: f64 = 0.0;
            for ts in tss.iter() {
                // todo(perf): how to eliminate bounds check ?
                let v = ts.values[i];
                if v.is_nan() || v < 0.0 {
                    continue;
                }
                sum += v;
            }
            // Divide every non-negative value at position i by sum in order to get its' share.
            for ts in tss.iter_mut() {
                let v = ts.values[i];
                ts.values[i] = if v.is_nan() || v < 0.0 {
                    f64::NAN
                } else {
                    v / sum
                }
            }
        }

        std::mem::take(tss)
    };

    aggr_func_ext(afe, &mut tss, afa.modifier, afa.limit, true)
}

fn aggr_func_zscore(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(afa)?;
    let afe = |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| {
        for i in 0..tss[0].values.len() {
            // Calculate avg and stddev for tss points at position i.
            // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
            let mut avg: f64 = 0.0;
            let mut count: u32 = 0;
            let mut q: f64 = 0.0;

            for ts in tss.iter() {
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

        std::mem::take(tss)
    };

    aggr_func_ext(afe, &mut tss, afa.modifier, afa.limit, true)
}

fn aggr_func_count_values(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = get_string_arg(&afa.args, 0)?.to_string();

    // Remove dst_label from grouping like Prometheus does.
    let modifier = if let Some(modifier) = &afa.modifier {
        let mut new_modifier = modifier.clone();
        match new_modifier.borrow_mut() {
            AggregateModifier::Without(args) => {
                args.push(dst_label.clone());
            }
            AggregateModifier::By(args) => {
                args.retain(|x| x != &dst_label);
            }
        }
        Some(new_modifier)
    } else {
        None
    };

    let afe = move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| -> Vec<Timeseries> {
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
            let mut dst: Timeseries = tss[0].clone();
            dst.metric_name.remove_tag(&dst_label);
            dst.metric_name
                .set_tag(&dst_label, format!("{}", v).as_str());

            for (i, dst_value) in dst.values.iter_mut().enumerate() {
                let mut count = 0;
                for ts in tss.iter() {
                    if ts.values[i] == v {
                        count += 1;
                    }
                }
                *dst_value = if count == 0 { f64::NAN } else { count as f64 };
            }

            rvs.push(dst);
        }
        rvs
    };

    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    let mut series_by_labels = aggr_prepare_series(&mut series, afa.modifier, afa.limit, false);

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(series_by_labels.len());
    for tss in series_by_labels.values_mut() {
        let mut rv = afe(tss, &modifier);
        rvs.append(&mut rv);

        // todo: how to config this limit?
        if rvs.len() >= MAX_SERIES_PER_AGGR_FUNC {
            let msg = format!(
                "more than -provider.MAX_SERIES_PER_AGGR_FUNC={} are generated by count_values()",
                MAX_SERIES_PER_AGGR_FUNC
            );
            return Err(RuntimeError::ArgumentError(msg));
        }
    }

    Ok(rvs)
}

fn func_topk_impl(afa: &mut AggrFuncArg, is_reverse: bool) -> RuntimeResult<Vec<Timeseries>> {
    let k = get_float_arg(&afa.args, 0, None)?;

    let afe =
        |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| -> Vec<Timeseries> {
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
                fill_nans_at_idx(n, k, tss)
            }
            remove_empty_series(tss);
            tss.reverse();

            std::mem::take(tss)
        };

    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    aggr_func_ext(afe, &mut series, afa.modifier, afa.limit, true)
}

fn range_topk_impl(
    afa: &mut AggrFuncArg,
    f: fn(values: &[f64]) -> f64,
    is_reverse: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let args_len = afa.args.len();
    let ks: Vec<f64> = get_scalar_arg_as_vec(&afa.args, 0, afa.ec)?;

    let remaining_sum_tag_name = if args_len == 3 {
        get_string_arg(&afa.args, 2)?.to_string()
    } else {
        "".to_string()
    };

    let afe = move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
        get_range_topk_timeseries(tss, modifier, &ks, &remaining_sum_tag_name, f, is_reverse)
    };

    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    aggr_func_ext(afe, &mut series, afa.modifier, afa.limit, true)
}

fn get_range_topk_timeseries<F>(
    tss: &mut Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    ks: &[f64],
    remaining_sum_tag_name: &str,
    f: F,
    is_reverse: bool,
) -> Vec<Timeseries>
where
    F: Fn(&[f64]) -> f64,
{
    struct TsWithValue {
        ts: Timeseries,
        value: f64,
    }

    let mut maxes: Vec<TsWithValue> = Vec::with_capacity(tss.len());
    for ts in tss.drain(0..) {
        let value = f(&ts.values);
        maxes.push(TsWithValue { ts, value });
    }

    maxes.sort_by(move |first, second| {
        let a = first.value;
        let b = second.value;
        if is_reverse {
            b.total_cmp(&a)
        } else {
            a.total_cmp(&b)
        }
    });

    let series = maxes.into_iter().map(|x| x.ts);

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
    tss.to_vec()
}

fn get_remaining_sum_timeseries(
    tss: &Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    ks: &[f64],
    remaining_sum_tag_name: &str,
) -> Option<Timeseries> {
    if remaining_sum_tag_name.is_empty() || tss.is_empty() {
        return None;
    }
    let mut dst = tss[0].clone();
    dst.metric_name.remove_group_tags(modifier);

    let mut tag_value = remaining_sum_tag_name;
    let mut remaining = remaining_sum_tag_name;

    if let Some((tag, remains)) = remaining_sum_tag_name.rsplit_once('=') {
        tag_value = tag;
        remaining = remains;
    }

    dst.metric_name.remove_tag(remaining);
    dst.metric_name.set_tag(remaining, tag_value);
    for (i, k) in ks.iter().enumerate() {
        let kn = get_int_k(*k, tss.len());
        let mut sum: f64 = 0.0;
        let mut count = 0;

        for ts in &tss[0..tss.len() - kn] {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            sum += v;
            count += 1;
        }

        if count == 0 {
            sum = f64::NAN;
        }
        dst.values[i] = sum;
    }
    Some(dst)
}

fn fill_nans_at_idx(idx: usize, k: f64, tss: &mut Vec<Timeseries>) {
    let kn = get_int_k(k, tss.len());

    let len = tss.len() - kn;
    for ts in tss[0..len].iter_mut() {
        ts.values[idx] = f64::NAN;
    }
}

fn get_int_k(k: f64, k_max: usize) -> usize {
    if k.is_nan() {
        return 0;
    }
    let kn = float_to_int_bounded(k).clamp(0, k_max as i64);
    kn as usize
}

fn min_value(values: &[f64]) -> f64 {
    let mut min = f64::NAN;
    for v in values.iter() {
        if !v.is_nan() && *v < min {
            min = *v
        }
    }
    min
}

fn max_value(values: &[f64]) -> f64 {
    let mut max = f64::NAN;
    let mut ofs = 0;
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        max = *v;
        ofs = i;
        break;
    }
    for v in values[ofs..].iter() {
        if !v.is_nan() && *v > max {
            max = *v
        }
    }
    max
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
    if vals.is_empty() {
        return f64::NAN;
    }
    values[values.len() - 1]
}

fn aggr_func_outliersk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let ks = get_scalar_arg_as_vec(&afa.args, 0, afa.ec)?;
    let afe = |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| {
        // Calculate medians for each point across tss.
        let medians = get_per_point_medians(tss);

        // Return topK time series with the highest variance from median.
        let f = |values: &[f64]| -> f64 {
            let mut sum2 = 0_f64;
            for (v, median) in values.iter().zip(medians.iter()) {
                let d = v - median;
                sum2 += d * d;
            }
            sum2
        };

        get_range_topk_timeseries(tss, afa.modifier, &ks, "", f, false)
    };

    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    aggr_func_ext(afe, &mut series, afa.modifier, afa.limit, true)
}

fn aggr_func_limitk(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_arg_count(afa, 2)?;

    let mut limit = get_int_arg(&afa.args, 0)?;
    if limit < 0 {
        limit = 0
    }

    let limit = limit as usize;

    let afe = |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| {
        let mut map: BTreeMap<Signature, Timeseries> = BTreeMap::new();

        // Sort series by metricName hash in order to get consistent set of output series
        // across multiple calls to limitk() function.
        // Sort series by hash in order to guarantee uniform selection across series.
        for ts in tss.iter_mut() {
            let digest = ts.metric_name.signature();
            map.insert(digest, std::mem::take(ts));
        }

        map.into_values().take(limit).collect::<Vec<_>>()
    };

    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    aggr_func_ext(afe, &mut series, afa.modifier, afa.limit, true)
}

fn aggr_func_quantiles(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = get_string_arg(&afa.args, 0)?.to_string();

    // todo: I'm sure this should have been checked in the parser
    let phi_count = afa.args.len() - 2;
    if phi_count == 0 {
        return Err(RuntimeError::ArgumentError(
            "quantiles() must have at least one phi argument".to_string(),
        ));
    }

    // tinyvec ??
    let mut phis: Vec<f64> = Vec::with_capacity(phi_count);

    for arg in afa.args[1..afa.args.len() - 1].iter() {
        phis.push(arg.get_scalar()?);
    }

    let afe = |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| {
        // todo: tinyvec ?
        let mut tss_dst: Vec<Timeseries> = Vec::with_capacity(phi_count);
        for phi in phis.iter() {
            let mut ts = tss[0].clone();
            ts.metric_name.set_tag(&dst_label, &format!("{}", phi));
            tss_dst.push(ts);
        }

        //let _vals = tiny_vec!([f64; 10]);
        let mut qs = get_pooled_vec_f64_filled(phis.len(), 0f64);

        let mut values = get_pooled_vec_f64_filled(phis.len(), f64::NAN);
        for n in 0..tss[0].values.len() {
            values.clear();
            for ts in tss.iter() {
                values.push(ts.values[n]);
            }
            quantiles(qs.deref_mut(), &phis, values.deref());

            for j in 0..tss_dst.len() {
                tss_dst[j].values[n] = qs[j];
            }
        }
        tss_dst
    };

    let mut last = afa
        .args
        .remove(afa.args.len() - 1)
        .get_instant_vector(afa.ec)?;

    aggr_func_ext(afe, &mut last, afa.modifier, afa.limit, false)
}

fn aggr_func_quantile(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let phis = get_scalar_arg_as_vec(&afa.args, 0, afa.ec)?;
    let afe = new_aggr_quantile_func(&phis);
    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    aggr_func_ext(afe, &mut series, afa.modifier, afa.limit, false)
}

fn aggr_func_median(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(afa)?;
    let phis = &eval_number(afa.ec, 0.5)?[0].values; // todo: use more efficient method
    let afe = new_aggr_quantile_func(phis);
    aggr_func_ext(afe, &mut tss, afa.modifier, afa.limit, false)
}

fn new_aggr_quantile_func(phis: &[f64]) -> impl AggrFnExt + '_ {
    move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| -> Vec<Timeseries> {
        let count = tss[0].values.len();
        let mut values = get_pooled_vec_f64(count);

        for (n, phi) in phis.iter().enumerate() {
            for ts in tss.iter() {
                values.push(ts.values[n]);
            }

            tss[0].values[n] = quantile(*phi, &values);
            values.clear();
        }

        tss.truncate(1);
        std::mem::take(tss)
    }
}

fn aggr_func_mad(tss: &mut Vec<Timeseries>) {
    // Calculate medians for each point across tss.
    let medians = get_per_point_medians(tss);

    // Calculate MAD values multiplied by tolerance for each point across tss.
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    tss[0].values = get_per_point_mads(tss, &medians);
    tss.truncate(1);
}

fn aggr_func_outliers_mad(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tolerances = get_scalar_arg_as_vec(&afa.args, 0, afa.ec)?;

    let afe = move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| {
        // Calculate medians for each point across tss.
        let medians = get_per_point_medians(tss);
        // Calculate MAD values multiplied by tolerance for each point across tss.
        // See https://en.wikipedia.org/wiki/Median_absolute_deviation
        let mut mads = get_per_point_mads(tss, &medians);

        for (mad, tolerance) in mads.iter_mut().zip(tolerances.iter()) {
            *mad *= tolerance;
        }

        // Leave only time series with at least a single peak above the MAD multiplied by tolerance.
        tss.retain(|ts| {
            for ((v, median), mad) in ts.values.iter().zip(medians.iter()).zip(mads.iter()) {
                let ad = (v - median).abs();
                if ad > *mad {
                    return true;
                }
            }
            false
        });

        std::mem::take(tss)
    };

    let mut series = get_series_arg(&afa.args, 1, afa.ec)?;
    aggr_func_ext(afe, &mut series, afa.modifier, afa.limit, true)
}

fn get_per_point_medians(tss: &mut Vec<Timeseries>) -> Vec<f64> {
    if tss.is_empty() {
        // todo: handle this case
        // panic!("BUG: expecting non-empty tss")
    }
    let count = tss[0].values.len();
    let mut medians = Vec::with_capacity(count);
    let mut values = get_pooled_vec_f64(count);
    for n in 0..count {
        values.clear();
        for ts in tss.iter() {
            let v = ts.values[n];
            if !v.is_nan() {
                values.push(v);
            }
        }
        medians.push(median_value(&values))
    }

    medians
}

fn get_per_point_mads(tss: &[Timeseries], medians: &[f64]) -> Vec<f64> {
    let mut mads: Vec<f64> = Vec::with_capacity(medians.len());
    let mut values = get_pooled_vec_f64(medians.len());
    for (n, median) in medians.iter().enumerate() {
        values.clear();
        for ts in tss.iter() {
            let v = ts.values[n];
            if !v.is_nan() {
                let ad = (v - median).abs();
                values.push(ad);
            }
        }
        mads.push(median_value(&values))
    }
    mads
}

fn expect_arg_count(tfa: &AggrFuncArg, expected: usize) -> RuntimeResult<()> {
    let arg_count = tfa.args.len();
    if arg_count == expected {
        return Ok(());
    }
    Err(RuntimeError::ArgumentError(format!(
        "unexpected number of args; got {arg_count}; want {expected}",
    )))
}
