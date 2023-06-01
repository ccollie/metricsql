use std::borrow::BorrowMut;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock};

use lockfree_object_pool::LinearReusable;
use once_cell::sync::Lazy;
use tinyvec::*;

use crate::eval::eval_number;
use crate::exec::remove_empty_series;
use crate::functions::transform::vmrange_buckets_to_le;
use crate::functions::utils::float_to_int_bounded;
use crate::functions::{mode_no_nans, quantile, quantiles, skip_trailing_nans};
use crate::histogram::{get_pooled_histogram, Histogram};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, QueryValue, Timeseries};
use lib::get_float64s;
use metricsql::common::{AggregateModifier, AggregateModifierOp};
use metricsql::functions::AggregateFunction;

// todo: add lifetime so we dont need to copy modifier
pub struct AggrFuncArg<'a> {
    pub args: Vec<QueryValue>,
    pub ec: &'a EvalConfig,
    pub modifier: Option<AggregateModifier>,
    pub limit: usize,
}

impl<'a> AggrFuncArg<'a> {
    pub fn new(
        ec: &'a EvalConfig,
        args: Vec<QueryValue>,
        modifier: &Option<AggregateModifier>,
        limit: usize,
    ) -> Self {
        Self {
            args,
            ec,
            modifier: modifier.clone(),
            limit,
        }
    }
}

pub type AggrFunctionResult = RuntimeResult<Vec<Timeseries>>;

pub trait AggrFn: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

impl<T> AggrFn for T where T: Fn(&mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> + Send + Sync {}

macro_rules! create_aggr_fn {
    ($f:expr) => {
        Arc::new(new_aggr_func($f))
    };
}

macro_rules! create_range_fn {
    ($f:expr, $top:expr) => {
        Arc::new(new_aggr_func_range_topk($f, $top))
    };
}

macro_rules! create_simple_fn {
    ($f:expr) => {
        Arc::new($f)
    };
}

static HANDLER_MAP: Lazy<
    RwLock<HashMap<AggregateFunction, Arc<dyn AggrFn<Output = AggrFunctionResult>>>>,
> = Lazy::new(|| {
    use AggregateFunction::*;

    let mut m: HashMap<AggregateFunction, Arc<dyn AggrFn<Output = AggrFunctionResult>>> =
        HashMap::with_capacity(33);
    m.insert(Sum, create_aggr_func(Sum));
    m.insert(Min, create_aggr_func(Min));
    m.insert(Max, create_aggr_func(Max));
    m.insert(Avg, create_aggr_func(Avg));
    m.insert(StdDev, create_aggr_func(StdDev));
    m.insert(StdVar, create_aggr_func(StdVar));
    m.insert(Count, create_aggr_func(Count));
    m.insert(CountValues, create_aggr_func(CountValues));
    m.insert(Bottomk, create_aggr_func(Bottomk));
    m.insert(Topk, create_aggr_func(Topk));
    m.insert(Quantile, create_aggr_func(Quantile));
    m.insert(Quantiles, create_aggr_func(Quantiles));
    m.insert(Group, create_aggr_func(Group));

    // PromQL extension funcs
    m.insert(Median, create_aggr_func(Median));
    m.insert(MAD, create_aggr_func(MAD));
    m.insert(Limitk, create_aggr_func(Limitk));
    m.insert(Distinct, create_aggr_func(Distinct));
    m.insert(Sum2, create_aggr_func(Sum2));
    m.insert(GeoMean, create_aggr_func(GeoMean));
    m.insert(Histogram, create_aggr_func(Histogram));
    m.insert(TopkMin, create_aggr_func(TopkMin));
    m.insert(TopkMax, create_aggr_func(TopkMax));
    m.insert(TopkAvg, create_aggr_func(TopkAvg));
    m.insert(TopkLast, create_aggr_func(TopkLast));
    m.insert(TopkMedian, create_aggr_func(TopkMedian));
    m.insert(BottomkMin, create_aggr_func(BottomkMin));
    m.insert(BottomkMax, create_aggr_func(BottomkMax));
    m.insert(BottomkAvg, create_aggr_func(BottomkAvg));
    m.insert(BottomkLast, create_aggr_func(BottomkLast));
    m.insert(BottomkMedian, create_aggr_func(BottomkMedian));
    m.insert(Any, create_aggr_func(Any));
    m.insert(Outliersk, create_aggr_func(Outliersk));
    m.insert(OutliersMAD, create_aggr_func(OutliersMAD));
    m.insert(Mode, create_aggr_func(Mode));
    m.insert(ZScore, create_aggr_func(ZScore));
    RwLock::new(m)
});

pub fn create_aggr_func(_fn: AggregateFunction) -> Arc<dyn AggrFn<Output = AggrFunctionResult>> {
    use AggregateFunction::*;
    match _fn {
        Sum => create_aggr_fn!(aggr_func_sum),
        Min => create_aggr_fn!(aggr_func_min),
        Max => create_aggr_fn!(aggr_func_max),
        Avg => create_aggr_fn!(aggr_func_avg),
        StdDev => create_aggr_fn!(aggr_func_stddev),
        StdVar => create_aggr_fn!(aggr_func_stdvar),
        Count => create_aggr_fn!(aggr_func_count),
        CountValues => create_simple_fn!(aggr_func_count_values),
        Bottomk => create_simple_fn!(new_aggr_func_topk(true)),
        Topk => create_simple_fn!(new_aggr_func_topk(false)),
        Quantile => create_simple_fn!(aggr_func_quantile),
        Quantiles => create_simple_fn!(aggr_func_quantiles),
        Group => create_aggr_fn!(aggr_func_group),

        // PromQL extension funcs
        Median => create_simple_fn!(aggr_func_median),
        MAD => create_aggr_fn!(aggr_func_mad),
        Limitk => create_simple_fn!(aggr_func_limitk),
        Distinct => create_aggr_fn!(aggr_func_distinct),
        Sum2 => create_aggr_fn!(aggr_func_sum2),
        GeoMean => create_aggr_fn!(aggr_func_geomean),
        Histogram => create_simple_fn!(new_aggr_func(aggr_func_histogram)),
        TopkMin => create_range_fn!(min_value, false),
        TopkMax => create_range_fn!(max_value, false),
        TopkAvg => create_range_fn!(avg_value, false),
        TopkLast => create_range_fn!(last_value, false),
        TopkMedian => create_range_fn!(median_value, false),
        BottomkMin => create_range_fn!(min_value, true),
        BottomkMax => create_range_fn!(max_value, true),
        BottomkAvg => create_range_fn!(avg_value, true),
        BottomkLast => create_range_fn!(last_value, true),
        BottomkMedian => create_range_fn!(median_value, true),
        Any => create_simple_fn!(aggr_func_any),
        Outliersk => create_simple_fn!(aggr_func_outliersk),
        OutliersMAD => create_simple_fn!(aggr_func_outliers_mad),
        Mode => create_aggr_fn!(aggr_func_mode),
        Share => create_simple_fn!(aggr_func_share),
        ZScore => create_simple_fn!(aggr_func_zscore),
    }
}

pub fn get_aggr_func(
    op: &AggregateFunction,
) -> RuntimeResult<Arc<dyn AggrFn<Output = AggrFunctionResult>>> {
    let map = HANDLER_MAP.read().unwrap();
    let res = map.get(&op);
    match res {
        Some(aggr_func) => Ok(Arc::clone(&aggr_func)),
        None => Err(
            // should not happen !! Maybe panic ?
            RuntimeError::UnknownFunction(op.to_string()),
        ),
    }
}

fn new_aggr_func(afe: fn(tss: &mut Vec<Timeseries>)) -> impl AggrFn {
    move |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let mut tss = get_aggr_timeseries(&afa.args)?;
        aggr_func_ext(
            move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| {
                afe(tss);
                tss.shrink_to(1);
                std::mem::take(tss)
            }, // todo: avoid cloning
            &mut tss,
            &afa.modifier,
            afa.limit,
            false,
        )
    }
}

fn get_aggr_timeseries(args: &Vec<QueryValue>) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = args[0].get_instant_vector()?;
    for i in 1..args.len() {
        let mut other = args[i].get_instant_vector()?;
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
    remove_empty_series(arg_orig);

    // Perform grouping.
    let mut series_by_name: HashMap<String, Vec<Timeseries>> = HashMap::new();

    for mut ts in arg_orig.drain(0..) {
        ts.metric_name.remove_group_tags(modifier);

        let key = ts.metric_name.to_string(); // to canonical_string

        let series_len = series_by_name.len();

        let group = series_by_name.entry(key).or_default();

        if group.len() == 0 && max_series > 0 && series_len >= max_series {
            // We already reached time series limit after grouping. Skip other time series.
            continue;
        }

        if keep_original {
            // TODO: is it ok to do ts.into()? Does it allocate ?
            group.push(ts);
        } else {
            // Todo(perf) - does this need to be copied ? Although arg_orig is passed
            // mutably, this fn is the only consumer
            let copy = Timeseries::copy_from_metric_name(&ts);
            group.push(copy);
        };
    }

    let mut src_tss_count = 0;
    let mut dst_tss_count = 0;
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(series_by_name.len());
    for (_, tss) in series_by_name.iter_mut() {
        let mut rv = afe(tss, modifier);
        rvs.append(&mut rv);
        src_tss_count += tss.len();
        dst_tss_count += rv.len();
        if dst_tss_count > 2000 && dst_tss_count > 16 * src_tss_count {
            // This looks like count_values explosion.
            let msg = format!(
                "too many timeseries after aggregation; \n
                got {}; want less than {}",
                dst_tss_count,
                16 * src_tss_count
            );
            return Err(RuntimeError::from(msg));
        }
    }

    Ok(rvs)
}

fn aggr_func_any(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
    let afe = move |tss: &mut Vec<Timeseries>, _modifier_: &Option<AggregateModifier>| {
        tss.drain(0..).collect()
    };
    // Only a single time series per group must be returned
    let limit = afa.limit.max(1);
    aggr_func_ext(afe, &mut tss, &afa.modifier, limit, true)
}

fn aggr_func_group(tss: &mut Vec<Timeseries>) {
    for i in 0..tss[0].values.len() {
        let mut v = f64::NAN;
        for (i, ts) in tss.iter().enumerate() {
            if ts.values[i].is_nan() {
                continue;
            }
            v = 1.0;
        }
        tss[0].values[i] = v;
    }
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
}

fn aggr_func_sum2(tss: &mut Vec<Timeseries>) {
    if tss.len() == 1 {
        return;
    }

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
                let mut ts = Timeseries::copy_from_shallow_timestamps(&tss[0]);
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
        values.sort_by(|a, b| a.total_cmp(&b));
        values.dedup();
        let n = values.len();
        tss[0].values[i] = if n == 0 { f64::NAN } else { n as f64 };

        values.clear();
    }
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
}

fn aggr_func_share(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;

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

    return aggr_func_ext(afe, &mut tss, &afa.modifier, afa.limit, true);
}

fn aggr_func_zscore(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
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

    aggr_func_ext(afe, &mut tss, &afa.modifier, afa.limit, true)
}

fn aggr_func_count_values(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = afa.args[0].get_string()?;

    // Remove dst_label from grouping like Prometheus does.
    match afa.modifier.borrow_mut() {
        None => {}
        Some(modifier) => match modifier.op {
            AggregateModifierOp::Without => {
                modifier.args.push(dst_label.clone());
            }
            AggregateModifierOp::By => {
                modifier.args.retain(|x| x != &dst_label);
            }
        },
    }

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
            let mut dst: Timeseries = Timeseries::copy_from_shallow_timestamps(&tss[0]);
            dst.metric_name.remove_tag(&dst_label);
            dst.metric_name
                .set_tag(&dst_label, format!("{}", v).as_str());

            let mut i = 0;
            for dst_value in dst.values.iter_mut() {
                let mut count = 0;
                for ts in tss.iter() {
                    if ts.values[i] == v {
                        count += 1;
                    }
                }
                *dst_value = if count == 0 { f64::NAN } else { count as f64 };
                i += 1;
            }

            rvs.push(dst);
        }
        return rvs;
    };

    let mut series = get_series(afa, 1)?;
    aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, false)
}

fn new_aggr_func_topk(is_reverse: bool) -> impl AggrFn {
    move |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let ks = get_scalar(&afa, 0)?;

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
                    fill_nans_at_idx(n, ks[n], tss)
                }
                remove_empty_series(tss);
                tss.reverse();

                std::mem::take(tss)
            };

        let mut series = get_series(afa, 1)?;
        aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
    }
}

fn new_aggr_func_range_topk(f: fn(values: &[f64]) -> f64, is_reverse: bool) -> impl AggrFn {
    return move |afa: &mut AggrFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let args_len = afa.args.len();
        let ks: Vec<f64> = get_scalar(&afa, 0)?;

        let remaining_sum_tag_name = if args_len == 3 {
            afa.args[2].get_string()?
        } else {
            "".to_string()
        };

        let afe = move |tss: &mut Vec<Timeseries>, modifier: &Option<AggregateModifier>| {
            return get_range_topk_timeseries(
                tss,
                modifier,
                &ks,
                &remaining_sum_tag_name,
                f,
                is_reverse,
            );
        };

        let mut series = get_series(afa, 1)?;
        aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
    };
}

fn get_range_topk_timeseries<F>(
    tss: &mut Vec<Timeseries>,
    modifier: &Option<AggregateModifier>,
    ks: &Vec<f64>,
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

    let mut maxs: Vec<TsWithValue> = Vec::with_capacity(tss.len());
    for ts in tss.drain(0..) {
        let value = f(&ts.values);
        maxs.push(TsWithValue { ts, value });
    }

    maxs.sort_by(move |first, second| {
        let a = first.value;
        let b = second.value;
        if is_reverse {
            b.total_cmp(&a)
        } else {
            a.total_cmp(&b)
        }
    });

    let series = maxs.into_iter().map(|x| x.ts);

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
    remaining_sum_tag_name: &str,
) -> Option<Timeseries> {
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
    dst.metric_name.set_tag(remaining, tag_value);
    for (i, k) in ks.iter().enumerate() {
        let kn = get_int_k(*k, tss.len());
        let mut sum: f64 = 0.0;
        let mut count = 0;

        for j in 0..tss.len() - kn {
            let ts = &tss[j];
            let v = ts.values[i];
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

fn fill_nans_at_idx(idx: usize, k: f64, tss: &mut Vec<Timeseries>) {
    let kn = get_int_k(k, tss.len());

    let mut i = 0;
    while i < tss.len() - kn {
        tss[i].values[idx] = f64::NAN;
        i += 1;
    }
}

fn get_int_k(k: f64, k_max: usize) -> usize {
    if k.is_nan() {
        return 0;
    }
    let kn = float_to_int_bounded(k);
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
    for v in values.iter() {
        if !v.is_nan() && *v < min {
            min = *v
        }
    }
    return min;
}

fn max_value(values: &[f64]) -> f64 {
    let mut max = f64::NAN;
    for v in values {
        if !v.is_nan() && *v > max {
            max = *v
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
    let ks = get_scalar(&afa, 0)?;
    let afe = |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| {
        // Calculate medians for each point across tss.
        let medians = get_per_point_medians(tss);

        // Return topK time series with the highest variance from median.
        let f = |values: &[f64]| -> f64 {
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
    expect_arg_count(afa, 2)?;

    let mut limit = get_int_number(afa, 0)?;
    if limit < 0 {
        limit = 0
    }

    let limit = limit as usize;

    let afe = |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| {
        let mut map: BTreeMap<u64, Timeseries> = BTreeMap::new();

        // Sort series by metricName hash in order to get consistent set of output series
        // across multiple calls to limitk() function.
        // Sort series by hash in order to guarantee uniform selection across series.
        for ts in tss.into_iter() {
            let digest = ts.metric_name.fast_hash();
            map.insert(digest, std::mem::take(ts));
        }

        let res = map.into_values().take(limit).collect::<Vec<_>>();
        return res;
    };

    let mut series = get_series(afa, 1)?;
    aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, true)
}

fn aggr_func_quantiles(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = afa.args[0].get_string()?;

    let mut phis: Vec<f64> = Vec::with_capacity(afa.args.len() - 2);
    for i in 1..afa.args.len() - 1 {
        phis.push(afa.args[i].get_scalar()?);
    }

    let afe = |tss: &mut Vec<Timeseries>, _modifier: &Option<AggregateModifier>| {
        // todo: smallvec
        let mut tss_dst: Vec<Timeseries> = Vec::with_capacity(phis.len());
        for j in 0..phis.len() {
            let mut ts = Timeseries::copy_from_shallow_timestamps(&tss[0]);
            ts.metric_name.remove_tag(&dst_label);
            // TODO
            ts.metric_name.set_tag(&dst_label, &format!("{}", phis[j]));
            tss_dst.push(ts);
        }

        let mut _vals = tiny_vec!([f64; 10]);
        let mut qs = get_float64s(phis.len());

        let mut values = get_float64s(phis.len());
        for n in tss[0].values.iter() {
            values.clear();
            let idx = *n as usize; // todo: handle panic
            for ts in tss.iter() {
                values.push(ts.values[idx]);
            }
            quantiles(qs.deref_mut(), &phis, values.deref());

            for j in 0..tss_dst.len() {
                tss_dst[j].values[idx] = qs[j];
            }
        }
        return tss_dst;
    };

    let mut last = afa.args.remove(afa.args.len() - 1).get_instant_vector()?;
    aggr_func_ext(afe, &mut last, &afa.modifier, afa.limit, false)
}

fn aggr_func_quantile(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let phis = get_scalar(&afa, 0)?;
    let afe = new_aggr_quantile_func(&phis);
    let mut series = get_series(afa, 1)?;
    return aggr_func_ext(afe, &mut series, &afa.modifier, afa.limit, false);
}

fn aggr_func_median(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_aggr_timeseries(&afa.args)?;
    let phis = &eval_number(&afa.ec, 0.5)[0].values; // todo: use more efficient method
    let afe = new_aggr_quantile_func(phis);
    return aggr_func_ext(afe, &mut tss, &afa.modifier, afa.limit, false);
}

fn new_aggr_quantile_func(phis: &Vec<f64>) -> impl AggrFnExt + '_ {
    move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| -> Vec<Timeseries> {
        let count = tss[0].values.len();
        let mut values = get_float64s(count);

        for n in 0..count {
            for ts in tss.iter() {
                values.push(ts.values[n]);
            }

            tss[0].values[n] = quantile(phis[n], &values);
            values.clear();
        }

        return std::mem::take(tss);
    }
}

fn aggr_func_mad(tss: &mut Vec<Timeseries>) {
    // Calculate medians for each point across tss.
    let medians = get_per_point_medians(tss);
    // Calculate MAD values multiplied by tolerance for each point across tss.
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    let mads = get_per_point_mads(tss, &medians);

    tss[0].values.extend_from_slice(&mads);
}

fn aggr_func_outliers_mad(afa: &mut AggrFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tolerances = get_scalar(&afa, 0)?;

    let afe = move |tss: &mut Vec<Timeseries>, _: &Option<AggregateModifier>| {
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
        // todo: handle this case
        // logger.Panicf("BUG: expecting non-empty tss")
    }
    let mut medians = Vec::with_capacity(tss[0].values.len());
    let mut values = get_float64s(medians.len());
    for n in 0..medians.len() {
        values.clear();
        for ts in tss.iter() {
            let v = ts.values[n];
            if !v.is_nan() {
                values.push(v);
            }
        }
        medians.push(quantile(0.5, &values))
    }

    medians
}

fn get_per_point_mads(tss: &Vec<Timeseries>, medians: &[f64]) -> Vec<f64> {
    let mut mads: Vec<f64> = Vec::with_capacity(medians.len());
    // todo: tinyvec
    let mut values = get_float64s(mads.len());
    for (n, median) in medians.iter().enumerate() {
        values.clear();
        for ts in tss.iter() {
            let v = ts.values[n];
            if !v.is_nan() {
                let ad = (v - median).abs();
                values.push(ad);
            }
        }
        mads.push(quantile(0.5, &values))
    }
    mads
}

fn get_series(tfa: &AggrFuncArg, arg_num: usize) -> RuntimeResult<Vec<Timeseries>> {
    Ok(tfa.args[arg_num].get_instant_vector()?)
}

// todo(perf) Cow
fn get_scalar(tfa: &AggrFuncArg, arg_num: usize) -> RuntimeResult<Vec<f64>> {
    let arg = &tfa.args[arg_num];
    return match arg {
        QueryValue::RangeVector(val) => {
            let values = val
                .iter()
                .map(|ts| {
                    if ts.values.len() > 0 {
                        ts.values[0]
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            Ok(values)
        }
        QueryValue::InstantVector(val) => {
            if val.len() != 1 {
                let msg = format!(
                    "arg #{} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    val.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            let mut res: Vec<f64> = Vec::with_capacity(val.len());
            for ts in val.iter() {
                let v = if ts.values.len() > 0 {
                    ts.values[0]
                } else {
                    f64::NAN
                };
                res.push(v);
            }
            Ok(res)
        } // ????
        QueryValue::Scalar(n) => {
            let len = tfa.ec.timestamps().len();
            // todo: tinyvec
            let values = vec![*n; len];
            Ok(values)
        }
        _ => Err(RuntimeError::InvalidNumber(
            "vector parameter expected ".to_string(),
        )),
    };
}

pub(crate) fn get_int_number(tfa: &AggrFuncArg, arg_num: usize) -> RuntimeResult<i64> {
    let v = get_scalar(tfa, arg_num)?;
    let mut n = 0;
    if v.len() > 0 {
        n = float_to_int_bounded(v[0]);
    }
    return Ok(n);
}

fn expect_arg_count(tfa: &AggrFuncArg, expected: usize) -> RuntimeResult<()> {
    let arg_count = tfa.args.len();
    if arg_count == expected {
        return Ok(());
    }
    return Err(RuntimeError::ArgumentError(format!(
        "unexpected number of args; got {}; want {}",
        arg_count, expected
    )));
}
