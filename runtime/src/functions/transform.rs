use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ops::DerefMut;

use chrono::{Datelike, DateTime, Timelike, TimeZone, Utc, Weekday};
use chrono_tz::Tz;
use phf::phf_map;
use rand::{Rng, rngs::StdRng, SeedableRng, thread_rng};
use rand_distr::{Exp1, StandardNormal};
use rand_distr::num_traits::FloatConst;
use rand_distr::num_traits::real::Real;
use regex::Regex;
use tinyvec::TinyVec;

use lib::{copysign, fmod, from_float, get_float64s, get_pooled_buffer, isinf, modf};
use metricsql::ast::{Expression, FuncExpr};
use metricsql::parser::compile_regexp;

use crate::{MetricName, NAME_LABEL, Timeseries};
use crate::binary_op::merge_non_overlapping_timeseries;
use crate::eval::{eval_number, eval_time, EvalConfig};
use crate::functions::aggr::quantile_sorted;
use crate::runtime_error::{RuntimeError, RuntimeResult};

const inf: f64 = f64::INFINITY;

pub(crate) struct TransformFuncArg<'a> {
    pub ec: &'a EvalConfig,
    pub fe: &'a FuncExpr,
    pub args: &'a Vec<Vec<Timeseries>>,
}

impl<'a> TransformFuncArg<'a> {
    pub fn new(ec: &'a EvalConfig, fe: &'a FuncExpr, args: &'a Vec<Vec<Timeseries>>) -> Self {
        Self { ec, fe, args }
    }
}

pub(crate) type TransformFunc = fn(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>>;


// https://stackoverflow.com/questions/57937436/how-to-alias-an-impl-trait
// https://www.worthe-it.co.za/blog/2017-01-15-aliasing-traits-in-rust.html

// This trait is local to this crate,
// so we can implement it on any type we want.
trait TransformFn: Fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {}

/// implement `Transform` on any type that implements `Fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>>>`.
impl<T> TransformFn for T where T: Fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {}

const nan: f64 = f64::NAN;

static TRANSFORM_FUNCS: phf::Map<&'static str, TransformFunc> = phf_map! {
""=>  transform_union, // empty func is a synonym to union
"abs"=>  new_transform_func_one_arg(transform_abs),
"absent"=>  transform_absent,
"acos"=>  new_transform_func_one_arg(transform_acos),
"acosh"=>  new_transform_func_one_arg(transform_acosh),
"asin"=>  new_transform_func_one_arg(transform_asin),
"asinh"=>  new_transform_func_one_arg(transform_asinh),
"atan"=>  new_transform_func_one_arg(transform_atan),
"atanh"=>  new_transform_func_one_arg(transform_atanh),
"bitmap_and"=>  new_transform_bitmap(bitmap_and),
"bitmap_or"=>  new_transform_bitmap(bitmap_or),
"bitmap_xor"=>  new_transform_bitmap(bitmap_xor),
"buckets_limit"=>  transform_buckets_limit,
"ceil" => new_transform_func_one_arg(transform_ceil),
"clamp" => transform_clamp,
"clamp_max" => transform_clamp_max,
"clamp_min" => transform_clamp_min,
"cos" => new_transform_func_one_arg(transform_cos),
"cosh" => new_transform_func_one_arg(transform_cosh),
"day_of_month"=> new_transform_func_datetime(transform_day_of_month),
"day_of_week"=> new_transform_func_datetime(transform_day_of_week),
"days_in_month"=> new_transform_func_datetime(transform_days_in_month),
"deg"=> new_transform_func_one_arg(transform_deg),
"drop_common_labels"=> transform_drop_common_labels,
"end" => new_transform_func_zero_args(transform_end),
"exp" => new_transform_func_one_arg(transform_exp),
"floor" => new_transform_func_one_arg(transform_floor),
"histogram_avg" =>  transform_histogram_avg,
"histogram_quantile" => transform_histogram_quantile,
"histogram_quantiles" => transform_histogram_quantiles,
"histogram_share" => transform_histogram_share,
"histogram_stddev" => transform_histogram_stddev,
"histogram_stdvar" => transform_histogram_stdvar,
"hour" => new_transform_func_datetime(transform_hour),
"interpolate" => transform_interpolate,
"keep_last_value" => transform_keep_last_value,
"keep_next_value" => transform_keep_next_value,
"label_copy" =>  transform_label_copy,
"label_del" =>  transform_label_del,
"label_graphite_group" =>  transform_label_graphite_group,
"label_join" =>  transform_label_join,
"label_keep" =>  transform_label_keep,
"label_lowercase" =>  transform_label_lowercase,
"label_map" =>  transform_label_map,
"label_match" =>  transform_label_match,
"label_mismatch" =>  transform_label_mismatch,
"label_move" =>  transform_label_move,
"label_replace" =>  transform_label_replace,
"label_set" =>  transform_label_set,
"label_transform" =>  transform_label_transform,
"label_uppercase" =>  transform_label_uppercase,
"label_value" =>  transform_label_value,
"limit_offset" =>  transform_limit_offset,
"ln" =>  new_transform_func_one_arg(transform_ln),
"log2" =>  new_transform_func_one_arg(transform_log2),
"log10" =>  new_transform_func_one_arg(transform_log10),
"minute" =>  new_transform_func_datetime(transform_minute),
"month" =>  new_transform_func_datetime(transform_month),
"now" =>  transform_now,
"pi" => transform_pi,
"prometheus_buckets" => transform_prometheus_buckets,
"rad" =>  new_transform_func_one_arg(transform_rad),
"random" =>  new_transform_rand(new_rand_float64),
"rand_exponential" => new_transform_rand(new_rand_exp_float64),
"rand_normal" => new_transform_rand(new_rand_norm_float64),
"range_avg" => new_transform_func_range(running_avg),
"range_first" =>  transform_range_first,
"range_last" =>  transform_range_last,
"range_max" =>  new_transform_func_range(running_max),
"range_min" =>  new_transform_func_range(running_min),
"range_quantile" =>  transform_range_quantile,
"range_sum" =>  new_transform_func_range(running_sum),
"remove_resets" =>  transform_remove_resets,
"round" =>  transform_round,
"running_avg" => new_transform_func_running(running_avg),
"running_max" => new_transform_func_running(running_max),
"running_min" => new_transform_func_running(running_min),
"running_sum" => new_transform_func_running(running_sum),
"scalar" => transform_scalar,
"sgn" => transform_sgn,
"sin" => new_transform_func_one_arg(transform_sin),
"sinh" => new_transform_func_one_arg(transform_sinh),
"smooth_exponential" => transform_smooth_exponential,
"sort" =>  new_transform_func_sort(false),
"sort_by_label" => new_transform_func_sort_by_label(false),
"sort_by_label_desc" => new_transform_func_sort_by_label(true),
"sort_by_label_numeric" => new_transform_func_alpha_numeric_sort(false),
"sort_by_label_numeric_desc" => new_transform_func_alpha_numeric_sort(true),
"sort_desc" => new_transform_func_sort(true),
"sqrt" => new_transform_func_one_arg(transform_sqrt),
"start" => new_transform_func_zero_args(transform_start),
"step" => new_transform_func_zero_args(transform_step),
"tan" => new_transform_func_one_arg(transform_tan),
"tanh" => new_transform_func_one_arg(transform_tanh),
"time" => transform_time,
// "timestamp" has been moved to rollup funcs. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415
"timezone_offset" => transform_timezone_offset,
"union" =>  transform_union,
"vector" =>  transform_vector,
"year" =>  new_transform_func_datetime(transform_year),
};

/// These functions don't change physical meaning of input time series,
/// so they don't drop metric name
pub(crate) fn transform_fn_keep_metric_name(name: &str) -> bool {
    match name.to_lowercase().as_str() {
        "ceil" | "clamp" | "clamp_min" | "clamp_max" | "floor" | "interpolate"
        | "keep_last_value" | "keep_next_value" | "range_avg" | "range_first" | "range_last"
        | "range_max" | "range_min" | "range_quantile" | "round" | "running_avg"
        | "running_max" | "running_min" | "smooth_exponential" => true,
        _ => false,
    }
}

pub fn get_transform_func(s: &str) -> Option<&TransformFunc> {
    TRANSFORM_FUNCS.get(s.to_lowercase().as_str())
}

trait TransformValuesFn: FnMut(&mut [f64]) -> () {}
impl<T> TransformValuesFn for T where T: FnMut(&mut [f64]) -> () {}

fn new_transform_func_one_arg(tf: fn(v: f64) -> f64) -> TransformFunc {
    let tfe = move |values: &mut [f64]| {
        for value in values.iter_mut() {
            *value = tf(*value)
        }
    };

    move |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        expect_transform_args_num(tfa, 1)?;
        do_transform_values(&mut tfa.args[0], tfe, &tfa.fe)
    }
}



#[inline]
fn do_transform_values(
    arg: &mut Vec<Timeseries>,
    tf: impl TransformValuesFn,
    fe: &FuncExpr,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut keep_metric_names = fe.keep_metric_names || transform_fn_keep_metric_name(&fe.name);
    for ts in arg {
        if !keep_metric_names {
            ts.metric_name.reset_metric_group();
        }
        tf(&mut ts.values);
    }
    Ok(*arg)
}

macro_rules! return_arg {
    ($tfa:expr, $arg_num:expr) => {{
        if $tfa.args.len() == 1 || $arg_num == $tfa.args.len() - 1 {
            Ok($tfa.args.remove($arg_num))
        } else {
            let res = $tfa.args.swap_remove($arg_num);
            Ok(res)
        }
    }};
}

#[inline]
fn transform_abs(v: f64) -> f64 {
    v.acos()
}

fn transform_absent(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args = &tfa.args;

    expect_transform_args_num(tfa, 1)?;
    let tss = &args[0];
    let mut rvs = get_absent_timeseries(&mut tfa.ec, &tfa.fe.args[0]);
    if tfa.args[0].len() == 0 {
        return Ok(rvs);
    }
    for i in 0..tss[0].values.len() {
        let mut is_absent = true;
        for ts in tfa.args[0].iter() {
            if !ts.values[i].is_nan() {
                is_absent = false;
                break;
            }
        }
        if !is_absent {
            rvs[0].values[i] = nan
        }
    }
    return Ok(rvs);
}

pub(crate) fn get_absent_timeseries(ec: &mut EvalConfig, arg: &Expression) -> Vec<Timeseries> {
    // Copy tags from arg
    let mut rvs = eval_number(ec, 1.0);
    match arg {
        Expression::MetricExpression(me) => {
            for tf in me.label_filters.iter() {
                if tf.label.len() == 0 {
                    continue;
                }
                if tf.is_regexp() || tf.is_negative() {
                    continue;
                }
                rvs[0].metric_name.add_tag(&tf.label, &tf.value)
            }
        }
        _ => {}
    }
    return rvs;
}

#[inline]
fn transform_ceil(v: f64) -> f64 {
    v.ceil()
}

fn transform_clamp(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 3)?;
    let mins = get_scalar(&tfa, 1)?;
    let maxs = get_scalar(&tfa, 2)?;
    // are these guaranteed to be of equal length ?
    let tf = |values: &mut [f64]| {
        let mut i = 0;
        for v in values {
            *v = v.clamp(mins[i], maxs[i]);
            i += 1;
        }
    };
    do_transform_values(&mut tfa.args[0], tf, tfa.fe)
}

fn transform_clamp_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let maxs = get_scalar(&tfa, 1)?;
    let tf = |values: &mut [f64]| {
        let mut i = 0;
        for v in values {
            if v > &mut maxs[i] {
                *v = maxs[i]
            }
            i += 1;
        }
    };
    do_transform_values(&mut tfa.args[0], tf, tfa.fe)
}

fn transform_clamp_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let mins = get_scalar(&tfa, 1)?;
    let tf = |values: &mut [f64]| {
        let mut i = 0;
        for v in values {
            if *v < mins[i] {
                *v = mins[i]
            }
            i += 1;
        }
    };

    do_transform_values(&mut tfa.args[0], tf, &tfa.fe)
}

fn new_transform_func_datetime(f: fn(t: DateTime<Utc>) -> f64) -> TransformFunc {
    |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let args = &tfa.args;
        if args.len() > 1 {
            let msg = format!("too many args; got {}; want up to {}", args.len(), 1);
            return Err(RuntimeError::ArgumentError(msg));
        }

        let mut arg: Vec<Timeseries> = if args.len() == 0 {
            eval_time(&mut tfa.ec)
        } else {
            args[0]
        };

        let tf = |values: &mut [f64]| {
            for (i, v) in values.iter().enumerate() {
                if v.is_nan() {
                    continue;
                }
                let t = Utc.timestamp(*v as i64, 0);
                values[i] = f(t) as f64;
            }
        };
        return do_transform_values(&mut arg, tf, tfa.fe);
    }
}

#[inline]
fn transform_day_of_month(t: DateTime<Utc>) -> f64 {
    t.day() as f64
}

#[inline]
fn transform_day_of_week(t: DateTime<Utc>) -> f64 {
    return match t.weekday() {
        Weekday::Sun => 0.0,
        Weekday::Mon => 1.0,
        Weekday::Tue => 2.0,
        Weekday::Wed => 3.0,
        Weekday::Thu => 4.0,
        Weekday::Fri => 5.0,
        Weekday::Sat => 6.0
    };
}

fn transform_days_in_month(t: DateTime<Utc>) -> f64 {
    let m = t.month();
    if m == 2 && is_leap_year(t.year() as u32) {
        return 29 as f64;
    }
    DAYS_IN_MONTH[m as usize] as f64
}

#[inline]
fn transform_exp(v: f64) -> f64 {
    v.exp()
}

#[inline]
fn transform_floor(v: f64) -> f64 {
    v.floor()
}

fn transform_buckets_limit(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let limits = get_scalar(&tfa, 1)?;
    let mut limit: usize = 0;
    if limits.len() > 0 {
        limit = limits[0] as usize // trunc ???
    }
    if limit <= 0 {
        return Ok(vec![]);
    }
    if limit < 3 {
        // Preserve the first and the last bucket for better accuracy for min and max values.
        limit = 3
    }
    let mut tss = vmrange_buckets_to_le(&mut tfa.args[1]);
    if tss.len() == 0 {
        return Ok(vec![]);
    }

    // Group timeseries by all MetricGroup+tags excluding `le` tag.
    struct Bucket {
        le: f64,
        hits: f64,
        ts: Timeseries,
    }

    let mut m: HashMap<String, Vec<Bucket>> = HashMap::new();

    // todo: make sure to cap data returned from marshal below
    let mut b = TinyVec::<[u8; 1024]>::with_capacity(512);

    let mut mn: MetricName = MetricName::default();

    for ts in tss {
        let le_str = ts
            .metric_name
            .get_tag_value("le")
            .unwrap_or(&String::from(""));
        if le_str.len() == 0 {
            // Skip time series without `le` tag.
            continue;
        }

        match le_str.parse::<f64>() {
            Ok(le) => {
                mn.copy_from(&ts.metric_name);
                mn.remove_tag("le");

                let key = mn.marshal_to_string(b.deref_mut())?;
                m.entry(key)
                    .or_insert(vec![])
                    .push(Bucket { le, hits: 0.0, ts });
            }
            _ => {
                // Skip time series with invalid `le` tag.
                continue;
            }
        }
    }

    // Remove buckets with the smallest counters.
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for (_, mut leGroup) in m.iter_mut() {
        if leGroup.len() <= limit {
            // Fast path - the number of buckets doesn't exceed the given limit.
            // Keep all the buckets as is.
            rvs.extend(&mut leGroup.into_iter());
            continue;
        }
        // Slow path - remove buckets with the smallest number of hits until their count reaches the limit.

        // Calculate per-bucket hits.
        leGroup.sort_by(|a, b| a.le.total_cmp(&b.le));
        for n in 0..limits.len() {
            let mut prev_value: f64 = 0.0;
            for i in 0..leGroup.len() {
                let xx = &leGroup[i];
                let value = xx.ts.values[n];
                leGroup[i].hits += value - prev_value;
                prev_value = value
            }
        }
        while leGroup.len() > limit {
            // Preserve the first and the last bucket for better accuracy for min and max values
            let mut xx_min_idx = 1;
            let mut min_merge_hits = leGroup[1].hits + leGroup[2].hits;
            for i in 0..leGroup[1..leGroup.len() - 2].len() {
                let merge_hits = leGroup[i + 1].hits + leGroup[i + 2].hits;
                if merge_hits < min_merge_hits {
                    xx_min_idx = i + 1;
                    min_merge_hits = merge_hits
                }
            }
            leGroup[xx_min_idx + 1].hits += leGroup[xx_min_idx].hits;
            // remove item at xx_min_idx ?
            // leGroup = append(leGroup[: xx_min_idx], leGroup[xx_min_idx + 1: ]...)
            leGroup.remove(xx_min_idx);
        }

        rvs.extend(leGroup.drain(..));
    }
    return Ok(rvs);
}

fn transform_prometheus_buckets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    let mut rvs = vmrange_buckets_to_le(&mut tfa.args[0]);
    return Ok(rvs);
}

static ELLIPSIS: &str = "...";

pub(crate) fn vmrange_buckets_to_le(tss: &mut Vec<Timeseries>) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());

    // Group timeseries by MetricGroup+tags excluding `vmrange` tag.
    #[derive(Clone, Default)]
    struct Bucket<'a> {
        start_str: String,
        end_str: String,
        start: f64,
        end: f64,
        ts: &'a Timeseries,
    }

    let mut m: HashMap<String, Vec<Bucket>> = HashMap::new();

    let mut bb = get_pooled_buffer(512);

    for ts in tss.into_iter() {
        let mut vm_range = match ts.metric_name.get_tag_value("vmrange") {
            Some(value) => value,
            None => &"".to_string(),
        };

        if vm_range.len() == 0 {
            match ts.metric_name.get_tag_value("le") {
                Some(le) => {
                    if le.len() > 0 {
                        // Keep Prometheus-compatible buckets.
                        rvs.push(ts.into());
                    }
                }
                None => {}
            }
            continue;
        }

        let n = match vm_range.find(ELLIPSIS) {
            Some(pos) => pos,
            None => continue,
        };

        let start_str = &vm_range[0..n];
        let start = match start_str.parse::<f64>() {
            Err(_) => continue,
            Ok(n) => n,
        };

        let end_str = &vm_range[(n + ELLIPSIS.len())..vm_range.len()];
        let end = match end_str.parse::<f64>() {
            Err(_) => continue,
            Ok(n) => n,
        };

        ts.metric_name.remove_tag("le");
        ts.metric_name.remove_tag("vmrange");
        let key = ts.metric_name.marshal_to_string(bb.deref_mut())?;

        m.entry(key).or_default().push(Bucket {
            start_str: start_str.to_string(),
            end_str: end_str.to_string(),
            start,
            end,
            ts: &ts,
        });

        bb.clear()
    }

    // Convert `vmrange` label in each group of time series to `le` label.
    let copy_ts = |src: &Timeseries, le_str: &str| -> Timeseries {
        let mut ts: Timeseries = Timeseries::default();
        ts.copy_from_shallow_timestamps(src);
        ts.values.resize(ts.values.len(), 0.0);
        ts.metric_name.remove_tag("le");
        ts.metric_name.add_tag("le", le_str);
        return ts;
    };

    let is_zero_ts = |ts: &Timeseries| -> bool { ts.values.iter().all(|x| *x <= 0.0) };

    for (_, mut xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.end.total_cmp(&b.end));
        let mut xss_new: Vec<Bucket> = Vec::with_capacity(xss.len() + 1);
        let mut xs_prev: &Bucket;

        let mut uniq_ts: HashMap<&String, &Timeseries> = HashMap::with_capacity(xss.len());
        for xs in xss.iter_mut() {
            let mut ts = xs.ts;
            if is_zero_ts(ts) {
                // Skip time series with zeros. They are substituted by xss_new below.
                xs_prev = xs;
                continue;
            }
            if xs.start != xs_prev.end && !uniq_ts.contains_key(&xs.start_str) {
                uniq_ts.insert(&xs.start_str, &xs.ts);
                xss_new.push(Bucket {
                    start_str: "".to_string(),
                    end_str: xs.start_str.clone(),
                    start: 0.0,
                    end: xs.start,
                    ts: &copy_ts(ts, &xs.start_str),
                })
            }
            xs.ts.metric_name.add_tag("le", &xs.end_str);
            match uniq_ts.get(&xs.end_str) {
                Some(mut prev_ts) => {
                    // the end of the current bucket is not unique, need to merge it with the existing bucket.
                    merge_non_overlapping_timeseries(&mut prev_ts, xs.ts);
                }
                None => {
                    xss_new.push(*xs);
                    uniq_ts.insert(&xs.end_str, xs.ts);
                }
            }
            xs_prev = xs
        }
        if !isinf(xs_prev.end, 1) {
            xss_new.push(Bucket {
                start_str: "".to_string(),
                end_str: "+Inf".to_string(),
                start: 0.0,
                end: f64::INFINITY,
                ts: &copy_ts(&xs_prev.ts, "+Inf"),
            })
        }
        let xss = &xss_new;
        for i in 0..xss[0].ts.values.len() {
            let mut count: f64 = 0.0;
            for xs in xss {
                let mut ts = xs.ts;
                let v = ts.values[i];
                if !v.is_nan() && v > 0.0 {
                    count += v
                }
                ts.values[i] = count
            }
        }
        for mut xs in xss.into_iter() {
            rvs.push(std::mem::take(&mut xs.ts))
        }
    }
    return rvs;
}

fn transform_histogram_share(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args = &tfa.args;
    if args.len() < 2 || args.len() > 3 {
        let msg = format!("unexpected number of args; got {}; want 2...3", args.len());
        return Err(RuntimeError::ArgumentError(msg));
    }

    let les: &Vec<f64>;
    match get_scalar(&tfa, 0) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "Cannot parse le: {:?}",
                e
            )));
        }
        Ok(v) => les = &v,
    }

    // Convert buckets with `vmrange` labels to buckets with `le` labels.
    let mut tss = vmrange_buckets_to_le(&mut tfa.args[1]);

    // Parse bounds_label. See https://github.com/prometheus/prometheus/issues/5706 for details.
    let bounds_label: String;

    if args.len() > 2 {
        match get_string(&tfa, 2) {
            Err(_) => {
                return Err(RuntimeError::ArgumentError(format!(
                    "cannot parse bounds_label (arg #3)"
                )));
            }
            Ok(v) => bounds_label = v,
        }
    } else {
        bounds_label = "".to_string();
    }

    // Group metrics by all tags excluding "le"
    let mut m = group_le_timeseries(&mut tss);

    // Calculate share for les
    let share = |i: usize, les: &[f64], xss: &mut Vec<LeTimeseries>| -> (f64, f64, f64) {
        let le_req = les[i];
        if le_req.is_nan() || xss.len() == 0 {
            return (nan, nan, nan);
        }
        fix_broken_buckets(i, xss);
        if le_req < 0.0 {
            return (0.0, 0.0, 0.0);
        }
        if isinf(le_req, 1) {
            return (1.0, 1.0, 1.0);
        }
        let mut v_prev: f64 = 0.0;
        let mut le_prev: f64 = 0.0;

        for xs in xss {
            let v = xs.ts.values[i];
            let le = xs.le;
            if le_req >= le {
                v_prev = v;
                le_prev = le;
                continue;
            }
            // precondition: le_prev <= le_req < le
            let v_last = xss[xss.len() - 1].ts.values[i];
            let lower = v_prev / v_last;
            if isinf(le, 1) {
                return (lower, lower, 1.0);
            }
            if le_prev == le_req {
                return (lower, lower, lower);
            }
            let upper = v / v_last;
            let q = lower + (v - v_prev) / v_last * (le_req - le_prev) / (le - le_prev);
            return (q, lower, upper);
        }
        // precondition: le_req > leLast
        return (1.0, 1.0, 1.0);
    };

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        let dst = xss[0].ts;
        let mut ts_lower: Timeseries;
        let mut ts_upper: Timeseries;

        if bounds_label.len() > 0 {
            ts_lower = Timeseries::copy_from_shallow_timestamps(dst);
            ts_lower.metric_name.remove_tag(&bounds_label);
            ts_lower.metric_name.add_tag(&bounds_label, "lower");

            ts_upper = Timeseries::copy_from_shallow_timestamps(dst);
            ts_upper.metric_name.remove_tag(&bounds_label);
            ts_upper.metric_name.add_tag(&bounds_label, "upper")
        } else {
            ts_lower = Timeseries::default();
            ts_upper = Timeseries::default();
        }

        for i in 0..dst.values.len() {
            let (q, lower, upper) = share(i, les, xss);
            xss[0].ts.values[i] = q;
            if bounds_label.len() > 0 {
                ts_lower.values[i] = lower;
                ts_upper.values[i] = upper
            }
        }
        rvs.push(dst);
        if bounds_label.len() > 0 {
            rvs.push(ts_lower);
            rvs.push(ts_upper);
        }
    }
    return Ok(rvs);
}

fn transform_histogram_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;

    let mut tss = vmrange_buckets_to_le(&mut tfa.args[0]);
    let mut m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());

    for (_, xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            xss[0].ts.values[i] = avg_for_le_timeseries(i, xss)
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    return Ok(rvs);
}

fn transform_histogram_stddev(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    let mut tss = vmrange_buckets_to_le(&mut tfa.args[0]);
    let mut m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());

    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            let v = stdvar_for_le_timeseries(i, &xss);
            xss[0].ts.values[i] = v.sqrt();
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    return Ok(rvs);
}

fn transform_histogram_stdvar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    let mut tss = vmrange_buckets_to_le(&mut tfa.args[0]);
    let mut m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        let dst = xss[0].ts;
        for i in 0..dst.values.len() {
            dst.values[i] = stdvar_for_le_timeseries(i, &xss)
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    return Ok(rvs);
}

fn avg_for_le_timeseries(i: usize, xss: &[LeTimeseries]) -> f64 {
    let mut le_prev: f64 = 0.0;
    let mut v_prev: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut weight_total: f64 = 0.0;
    for xs in xss {
        if isinf(xs.le, 0) {
            continue;
        }
        let le = xs.le;
        let n = f64::from(le + le_prev) / 2_f64;
        let v = xs.ts.values[i];
        let weight = v - v_prev;
        sum += n * weight;
        weight_total += weight;
        le_prev = le;
        v_prev = v;
    }
    if weight_total == 0.0 {
        return nan;
    }
    return sum / weight_total;
}

fn stdvar_for_le_timeseries(i: usize, xss: &[LeTimeseries]) -> f64 {
    let mut le_prev: f64 = 0.0;
    let mut v_prev: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    let mut weight_total: f64 = 0.0;
    for xs in xss {
        if isinf(xs.le, 0) {
            continue;
        }
        let le = xs.le;
        let n = (le + le_prev) / 2.0;
        let v = xs.ts.values[i];
        let weight = v - v_prev;
        sum += n * weight;
        sum2 += n * n * weight;
        weight_total += weight;
        le_prev = le;
        v_prev = v
    }
    if weight_total == 0.0 {
        return nan;
    }
    let avg = sum / weight_total;
    let avg2 = sum2 / weight_total;
    let mut stdvar = avg2 - avg * avg;
    if stdvar < 0.0 {
        // Correct possible calculation error.
        stdvar = 0.0
    }
    return stdvar;
}

fn transform_histogram_quantiles(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(tfa, 3)?;

    let dst_label: &str;
    match get_string(&tfa, 0) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "Cannot obtain dstLabel: {:?}",
                e
            )));
        }
        Ok(v) => dst_label = &v,
    }

    let len = tfa.args.len();
    let mut phi_args = tfa.args[1..len - 1].as_mut();
    let mut tss_orig = tfa.args[len - 1..].as_mut();
    // Calculate quantile individually per each phi.
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tfa.args.len());
    for (i, phiArg) in phi_args.iter().enumerate() {
        let phis = get_scalar(phiArg, i);
        let arg = phis[0];
        if arg < 0.0 || arg > 1.0 {
            let msg = "got unexpected phi arg. it should contain only numbers in the range [0..1]";
            return Err(RuntimeError::ArgumentError(msg.to_string()));
        }
        let phi_str = phis[0].to_string();
        let tss = copy_timeseries(tss_orig);

        let args: Vec<Vec<Timeseries>> = vec![*phiArg, tss];
        let mut tfa_tmp = TransformFuncArg {
            ec: tfa.ec,
            fe: tfa.fe,
            args: &args,
        };

        match transform_histogram_quantile(&mut tfa_tmp) {
            Err(e) => {
                let msg = format!("cannot calculate quantile {}: {:?}", phi_str, e);
                return Err(RuntimeError::General(msg));
            }
            Ok(mut tss_tmp) => {
                for ts in tss_tmp.iter_mut() {
                    ts.metric_name.remove_tag(dst_label);
                    ts.metric_name.add_tag(dst_label, &phi_str);
                }
                rvs.extend(tss_tmp)
            }
        }
    }
    return Ok(rvs);
}

fn transform_histogram_quantile(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.len() < 2 || tfa.args.len() > 3 {
        let msg = format!(
            "unexpected number of args; got {}; want 2...3",
            tfa.args.len()
        );
        return Err(RuntimeError::ArgumentError(msg));
    }

    let phis: &Vec<f64>;
    match get_scalar(&tfa, 0) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "Cannot parse phi: {:?}",
                e
            )));
        }
        Ok(v) => phis = v,
    }

    // Convert buckets with `vmrange` labels to buckets with `le` labels.
    let mut tss = vmrange_buckets_to_le(&mut tfa.args[1]);

    // Parse bounds_label. See https://github.com/prometheus/prometheus/issues/5706 for details.
    let bounds_label: String;
    if tfa.args.len() > 2 {
        match get_string(&tfa, 2) {
            Err(err) => {
                return Err(RuntimeError::ArgumentError(format!(
                    "cannot parse bounds_label (arg #3): {:?}",
                    err
                )));
            }
            Ok(s) => bounds_label = s,
        }
    } else {
        bounds_label = "".to_string();
    }

    // Group metrics by all tags excluding "le"
    let mut m = group_le_timeseries(&mut tss);

    // Calculate quantile for each group in m
    let last_non_inf = |i: usize, xss: &[LeTimeseries]| -> f64 {
        let mut cur = xss;
        while cur.len() > 0 {
            let xs_last = &cur[cur.len() - 1];
            if !isinf(xs_last.le, 0) {
                return xs_last.le;
            }
            cur = &cur[0..cur.len() - 1]
        }
        return nan;
    };

    let quantile = |i: usize, phis: &[f64], xss: &mut Vec<LeTimeseries>| -> (f64, f64, f64) {
        let phi = phis[i];
        if phi.is_nan() {
            return (nan, nan, nan);
        }
        fix_broken_buckets(i, xss);
        let mut v_last: f64 = 0.0;
        if xss.len() > 0 {
            v_last = xss[xss.len() - 1].ts.values[i]
        }
        if v_last == 0.0 {
            return (nan, nan, nan);
        }
        if phi < 0.0 {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY, xss[0].ts.values[i]);
        }
        if phi > 1.0 {
            return (inf, v_last, inf);
        }
        let v_req = v_last * phi;
        let mut v_prev: f64 = 0.0;
        let mut le_prev: f64 = 0.0;
        for xs in xss {
            let v = xs.ts.values[i];
            let le = xs.le;
            if v <= 0.0 {
                // Skip zero buckets.
                le_prev = le;
                continue;
            }
            if v < v_req {
                v_prev = v;
                le_prev = le;
                continue;
            }
            if isinf(le, 0) {
                break;
            }
            if v == v_prev {
                return (le_prev, le_prev, v);
            }
            let vv = le_prev + (le - le_prev) * (v_req - v_prev) / (v - v_prev);
            return (vv, le_prev, le);
        }
        let vv = last_non_inf(i, xss);
        return (vv, vv, inf);
    };

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        let dst = xss[0].ts;
        let mut ts_lower: Timeseries;
        let mut ts_upper: Timeseries;

        if bounds_label.len() > 0 {
            ts_lower = Timeseries::copy_from_shallow_timestamps(dst);
            ts_lower.metric_name.remove_tag(&bounds_label);
            ts_lower.metric_name.add_tag(&bounds_label, "lower");

            ts_upper = Timeseries::copy_from_shallow_timestamps(dst);
            ts_upper.metric_name.remove_tag(&bounds_label);
            ts_upper.metric_name.add_tag(&bounds_label, "upper");
        } else {
            ts_lower = Timeseries::default();
            ts_upper = Timeseries::default();
        }
        for i in 0..xss[0].ts.values.len() {
            let (v, lower, upper) = quantile(i, phis, xss);
            dst.values[i] = v;
            if bounds_label.len() > 0 {
                ts_lower.values[i] = lower;
                ts_upper.values[i] = upper;
            }
        }
        rvs.push(dst.into());
        if bounds_label.len() > 0 {
            rvs.push(ts_lower);
            rvs.push(ts_upper);
        }
    }
    return Ok(rvs);
}

struct LeTimeseries<'a> {
    le: f64,
    ts: &'a Timeseries,
}

fn group_le_timeseries(tss: &mut Vec<Timeseries>) -> HashMap<String, Vec<LeTimeseries>> {
    let mut m: HashMap<String, Vec<LeTimeseries>> = HashMap::new();

    // todo: tinyvec ???
    let mut bb = get_pooled_buffer(512);

    for ts in tss.iter_mut() {
        match ts.metric_name.get_tag_value("le") {
            None => {}
            Some(tag_value) => {
                if tag_value.len() == 0 {
                    continue;
                }
                match tag_value.parse::<f64>() {
                    Err(..) => continue,
                    Ok(le) => {
                        ts.metric_name.reset_metric_group();
                        ts.metric_name.remove_tag("le");
                        let key = ts.metric_name.marshal_to_string(bb.deref_mut())?;
                        m.entry(key).or_default().push(LeTimeseries { le, ts });
                        bb.clear()
                    }
                }
            }
        }
    }
    return m;
}

fn fix_broken_buckets(i: usize, xss: &mut Vec<LeTimeseries>) {
    // Buckets are already sorted by le, so their values must be in ascending order,
    // since the next bucket includes all the previous buckets.
    // If the next bucket has lower value than the current bucket,
    // then the current bucket must be substituted with the next bucket value.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2819
    if xss.len() < 2 {
        return;
    }
    let mut j = xss.len() - 1;
    while j >= 0 {
        let v = xss[j].ts.values[i];
        if !v.is_nan() {
            j += 1;
            while j < xss.len() {
                xss[j].ts.values[i] = v;
                j += 1;
            }
            break;
        }
        j -= 1;
    }

    let mut v_next = xss[xss.len() - 1].ts.values[i];

    let mut j = xss.len() - 1;
    while j >= 0 {
        let v = xss[j].ts.values[i];
        if v.is_nan() || v > v_next {
            xss[j].ts.values[i] = v_next
        } else {
            v_next = v;
        }
        j -= 1;
    }
}

#[inline]
fn transform_hour(t: DateTime<Utc>) -> f64 {
    t.hour() as f64
}

#[inline]
fn running_sum(a: f64, b: f64, idx: usize) -> f64 {
    return a + b;
}

#[inline]
fn running_max(a: f64, b: f64, idx: usize) -> f64 {
    if a > b {
        return a;
    }
    return b;
}

#[inline]
fn running_min(a: f64, b: f64, idx: usize) -> f64 {
    if a < b {
        return a;
    }
    return b;
}

#[inline]
fn running_avg(a: f64, b: f64, idx: usize) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
    a + (b - a) / (idx + 1) as f64
}

fn skip_leading_nans(values: &[f64]) -> &[f64] {
    let mut i = 0;
    while i < values.len() && values[i].is_nan() {
        i += 1;
    }
    return &values[i..];
}

fn skip_trailing_nans(values: &[f64]) -> &[f64] {
    let mut i = values.len() - 1;
    while i >= 0 && values[i].is_nan() {
        i -= 1;
    }
    return &values[0..i + 1];
}

fn transform_keep_last_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    for ts in tfa.args[0].iter_mut() {
        if ts.values.len() == 0 {
            continue;
        }
        let mut last_value = ts.values[0];
        for v in ts.values.iter_mut() {
            if !v.is_nan() {
                last_value = *v;
                continue;
            }
            *v = last_value
        }
    }
    return_arg!(tfa, 0)
}

fn transform_keep_next_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    for ts in tfa.args[0].iter_mut() {
        let mut values = &ts.values;
        if values.len() == 0 {
            continue;
        }
        let mut next_value = *ts.values.last().unwrap();
        let mut i = ts.values.len() - 1;
        while i >= 0 {
            let v = ts.values[i];
            if !v.is_nan() {
                next_value = v;
                continue;
            }
            ts.values[i] = next_value;
            i -= 1;
        }
    }
    return_arg!(tfa, 0)
}

fn transform_interpolate(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    for ts in tfa.args[0].iter_mut() {
        if ts.values.len() == 0 {
            continue;
        }
        let mut prev_value = f64::NAN;
        let mut next_value: f64;
        let mut i = 0;
        while i < ts.values.len() {
            if !ts.values[i].is_nan() {
                continue;
            }
            if i > 0 {
                prev_value = ts.values[i - 1]
            }
            let mut j = i + 1;
            while j < ts.values.len() {
                if !ts.values[j].is_nan() {
                    break;
                }
                j += 1;
            }
            if j >= ts.values.len() {
                next_value = prev_value
            } else {
                next_value = ts.values[j]
            }
            if prev_value.is_nan() {
                prev_value = next_value
            }
            let delta = (next_value - prev_value) / (j - i + 1) as f64;
            while i < j {
                prev_value += delta;
                ts.values[i] = prev_value;
                i += 1;
            }
        }
    }
    return_arg!(tfa, 0)
}

fn new_transform_func_running(rf: fn(a: f64, b: f64, idx: usize) -> f64) -> TransformFunc {
    |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        expect_transform_args_num(tfa, 1)?;
        let mut rvs = &tfa.args[0];
        for ts in rvs.iter_mut() {
            ts.metric_name.reset_metric_group();
            let mut values = skip_leading_nans(&ts.values);
            if values.len() == 0 {
                continue;
            }
            let mut prev_value = values[0];
            values = &values[1..];
            for (i, v) in values.iter_mut().enumerate() {
                if !v.is_nan() {
                    prev_value = rf(prev_value, *v, i + 1)
                }
                *v = prev_value
            }
        }
        return_arg!(tfa, 0)
    }
}

fn new_transform_func_range(rf: fn(a: f64, b: f64, idx: usize) -> f64) -> TransformFunc {
    let tfr = new_transform_func_running(rf);
    move |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let mut rvs = tfr(tfa)?;
        set_last_values(&mut rvs);
        return Ok(rvs);
    }
}

fn transform_range_quantile(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let phis = get_scalar(tfa, 0)?;
    let mut phi = 0.0;
    if phis.len() > 0 {
        phi = phis[0]
    }

    let mut values = get_float64s(tfa.args[1].len()).to_vec();

    for ts in tfa.args[1].iter() {
        let mut last_idx = 0;
        let mut origin_values = &ts.values;
        values.clear();
        for (i, v) in origin_values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            values.push(*v);
            last_idx = i;
        }
        if last_idx >= 0 {
            values.sort_by(|a, b| a.total_cmp(&b));
            origin_values[last_idx] = quantile_sorted(phi, &values)
        }
    }
    set_last_values(&mut tfa.args[1]);
    return_arg!(tfa, 1)
}

fn transform_range_first(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    for ts in tfa.args[0].iter_mut() {
        let mut values = skip_leading_nans(&ts.values);
        if values.len() == 0 {
            continue;
        }
        let v_first = values[0];
        for v in values.iter_mut() {
            if v.is_nan() {
                continue;
            }
            *v = v_first;
        }
    }
    return_arg!(tfa, 0)
}

fn transform_range_last(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    set_last_values(&mut tfa.args[0]);
    return Ok(tfa.args.remove(tfa.args.len() - 1));
}

fn set_last_values(tss: &mut Vec<Timeseries>) {
    for ts in tss {
        let mut values = skip_trailing_nans(&ts.values);
        if values.len() == 0 {
            continue;
        }
        let v_last = values[values.len() - 1];
        for v in values.iter_mut() {
            if v.is_nan() {
                continue;
            }
            *v = v_last;
        }
    }
}

fn transform_smooth_exponential(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let sfs = get_scalar(&tfa, 1)?;

    for ts in tfa.args[0] {
        let mut values = skip_leading_nans(&ts.values);
        for (i, v) in values.iter().enumerate() {
            if !isinf(*v, 0) {
                values = &values[1..];
                break;
            }
        }
        if values.len() == 0 {
            continue;
        }
        let mut avg = values[0];
        values = &values[1..];
        let sfs_x = &sfs[ts.values.len() - values.len()..];
        for (i, v) in values.iter_mut().enumerate() {
            if v.is_nan() {
                continue;
            }
            if isinf(*v, 0) {
                *v = avg;
                continue;
            }
            let mut sf = sfs_x[i];
            if sf.is_nan() {
                sf = 1.0;
            }
            sf = sf.clamp(0.0, 1.0);

            avg = avg * (1.0 - sf) + (*v) * sf;
            *v = avg;
        }
    }

    return_arg!(tfa, 0)
}

fn transform_remove_resets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    for ts in tfa.args[0].iter_mut() {
        remove_counter_resets_maybe_nans(&mut ts.values);
    }
    return_arg!(tfa, 0)
}

fn transform_union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args = &tfa.args;
    if args.len() < 1 {
        return Ok(eval_number(&mut tfa.ec, nan));
    }

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(args[0].len());
    let mut m: HashSet<String> = HashSet::with_capacity(args[0].len());
    // todo:: tinyvec
    let mut bb = get_pooled_buffer(512).to_vec();
    for arg in tfa.args.iter_mut() {
        for ts in arg.iter_mut() {
            let key = ts.metric_name.marshal_to_string(&mut bb)?;
            if m.insert(key) {
                rvs.push(ts);
            }
        }
    }
    return Ok(rvs);
}

fn transform_label_keep(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(tfa, 1)?;

    let mut keep_labels: Vec<String> = Vec::with_capacity(tfa.args.len());
    for i in 1..tfa.args.len() {
        let keep_label = get_string(&tfa, i)?;
        keep_labels.push(keep_label);
    }

    for mut ts in tfa.args[0].iter_mut() {
        ts.metric_name.remove_tags_on(&keep_labels)
    }

    return_arg!(tfa, 0)
}

fn transform_label_del(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(tfa, 1)?;
    let mut del_labels: Vec<String> = Vec::with_capacity(tfa.args.len());
    for i in 1..tfa.args.len() {
        let del_label = get_string(&tfa, i)?;
        del_labels.push(del_label);
    }

    for ts in tfa.args[0].iter_mut() {
        ts.metric_name.remove_tags_ignoring(&del_labels[0..])
    }

    return_arg!(tfa, 0)
}

///////////////////////

fn transform_label_set(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(tfa, 1)?;

    let (dst_labels, dst_values) = get_string_pairs(tfa, 1)?;
    for ts in tfa.args[0].iter_mut() {
        let mut mn = &ts.metric_name;
        let mut i = 0;
        for dstLabel in dst_labels.into_iter() {
            let mut dst_value = get_dst_value(&mut ts.metric_name, &dstLabel);
            if dst_values[i].len() == 0 {
                mn.remove_tag(&dstLabel);
            }
            dst_value.push_str(&dst_values[i]);
            i += 1;
        }
    }

    return_arg!(tfa, 0)
}

fn transform_label_uppercase(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    return transform_label_value_func(tfa, |x| x.to_uppercase());
}

fn transform_label_lowercase(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    return transform_label_value_func(tfa, |x| x.to_lowercase());
}

fn transform_label_value_func(
    tfa: &mut TransformFuncArg,
    f: fn(arg: &str) -> String,
) -> RuntimeResult<Vec<Timeseries>> {
    let args = tfa.args;
    expect_at_least_n_args(tfa, 2)?;

    // todo: Smallvec/Arrayyvec
    let mut labels = Vec::with_capacity(tfa.args.len() - 1);
    for i in 1..tfa.args.len() {
        let label = get_string(&tfa, i)?;
        labels.push(label);
    }

    for ts in tfa.args[0].iter_mut() {
        for label in labels.iter() {
            let mut dst_value = get_dst_value(&mut ts.metric_name, label);
            dst_value.push_str(&*f(dst_value));
            if dst_value.len() == 0 {
                ts.metric_name.remove_tag(label);
            }
        }
    }

    return_arg!(tfa, 0)
}

fn transform_label_map(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(&tfa, 2)?;

    let label = get_label(tfa, "", 1)?;

    let (src_values, dst_values) = get_string_pairs(tfa, 2)?;
    let mut m: HashMap<&str, &str> = HashMap::with_capacity(src_values.len());
    for (i, src_value) in src_values.iter().enumerate() {
        m.insert(src_value, &dst_values[i]);
    }
    for ts in tfa.args[0].iter_mut() {
        let mut dst_value = get_dst_value(&mut ts.metric_name, &label);
        if let Some(value) = m.get(dst_value) {
            dst_value.push_str(value);
        }
        if dst_value.len() == 0 {
            ts.metric_name.remove_tag(&label)
        }
    }

    return_arg!(tfa, 0)
}

fn transform_drop_common_labels(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(tfa, 1)?;

    for tss in tfa.args[1..].iter_mut() {
        &tfa[0].append(tss)
    }
    let mut m: HashMap<&str, HashMap<&str, usize>> = HashMap::new();

    let count_label = |name, value: &str| {
        let x = m.entry(name).or_insert_with(|| HashMap::new());
        x.entry(name).and_modify(|count| *count += 1).or_insert(0);
    };

    for ts in tfa.args[0].iter_mut() {
        count_label(NAME_LABEL, &ts.metric_name.metric_group);
        for tag in ts.metric_name.get_tags() {
            count_label(&tag.key, &tag.value);
        }
    }

    for (label_name, x) in m.iter() {
        for (_, count) in x {
            if *count != tfa.args[0].len() {
                continue;
            }
            for ts in tfa.args[0].iter_mut() {
                ts.metric_name.remove_tag(label_name);
            }
        }
    }

    return_arg!(tfa, 0)
}

fn transform_label_copy(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_copy_ext(tfa, false)
}

fn transform_label_move(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_copy_ext(tfa, true)
}

fn transform_label_copy_ext(
    tfa: &mut TransformFuncArg,
    remove_src_labels: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(&tfa, 1)?;

    let (src_labels, dst_labels) = get_string_pairs(tfa, 1)?;

    for ts in tfa.args[0].iter_mut() {
        for (i, srcLabel) in src_labels.iter().enumerate() {
            let dst_label = &dst_labels[i];
            match ts.metric_name.get_tag_value(srcLabel) {
                Some(value) => {
                    if value.len() == 0 {
                        // do not remove destination label if the source label doesn't exist.
                        continue;
                    }
                    let mut dst_value = get_dst_value(&mut ts.metric_name, dst_label);
                    dst_value.push_str(&value);
                    if remove_src_labels && srcLabel != dst_label {
                        ts.metric_name.remove_tag(srcLabel)
                    }
                }
                None => {}
            }
        }
    }

    return_arg!(tfa, 0)
}

fn get_string_pairs(
    tfa: &mut TransformFuncArg,
    start: usize,
) -> RuntimeResult<(Vec<String>, Vec<String>)> {
    let args = tfa.args;
    if args.len() % 2 != 0 {
        return Err(RuntimeError::ArgumentError(format!(
            "the number of string args must be even; got {}",
            args.len()
        )));
    }
    let result_len = args.len() / 2;
    let mut ks: Vec<String> = Vec::with_capacity(result_len);
    let mut vs: Vec<String> = Vec::with_capacity(result_len);
    let mut i = start;
    while i < args.len() {
        let k = get_string(&tfa, i)?;
        ks.push(k);
        let v = get_string(&tfa, i + 1)?;
        vs.push(v);
        i += 2;
    }
    return Ok((ks, vs));
}

fn transform_label_join(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_at_least_n_args(tfa, 3)?;

    let dst_label = get_string(&tfa, 1)?;
    let separator = get_string(&tfa, 2)?;

    // todo: user something like SmallVec/StaticVec/ArrayVec
    let mut src_labels: Vec<String> = Vec::with_capacity(tfa.args.len() - 3);
    for i in 3..tfa.args.len() {
        let src_label = get_string(&tfa, i)?;
        src_labels.push(src_label);
    }

    for ts in tfa.args[0].iter_mut() {
        let mut dst_value = get_dst_value(&mut ts.metric_name, &dst_label);
        // use some manner of string buffer

        dst_value.clear(); //??? test this

        for (j, srcLabel) in src_labels.iter().enumerate() {
            match ts.metric_name.get_tag_value(srcLabel) {
                Some(src_value) => {
                    dst_value.push_str(&src_value);
                    if j + 1 < src_labels.len() {
                        dst_value.push_str(&separator)
                    }
                }
                None => {}
            }
        }

        if dst_value.len() == 0 {
            ts.metric_name.remove_tag(&dst_label);
        }
    }

    return_arg!(tfa, 0)
}

fn transform_label_transform(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 4)?;
    let label = get_string(&tfa, 1)?;
    let regex = get_string(&tfa, 2)?;
    let replacement = get_string(&tfa, 3)?;

    let r = compile_regexp(&regex);
    match r {
        Ok(..) => {}
        Err(err) => {
            return Err(RuntimeError::from(format!(
                "cannot compile regex {}: {}",
                &regex,
                err,
            )));
        }
    }
    return label_replace(&mut tfa.args[0], &label, &r.unwrap(), &label, &replacement);
}

fn transform_label_replace(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 5)?;
    let regex = get_string(&tfa, 4)?;

    process_anchored_regex(tfa, &regex, |tfa, r| {
        let dst_label = get_string(&tfa, 1)?;
        let replacement = get_string(&tfa, 2)?;
        let src_label = get_string(&tfa, 3)?;

        label_replace(&mut tfa.args[0], &src_label, &r, &dst_label, &replacement)
    })
}

fn label_replace(
    tss: &mut Vec<Timeseries>,
    src_label: &str,
    r: &Regex,
    dst_label: &str,
    replacement: &str,
) -> RuntimeResult<Vec<Timeseries>> {
    for ts in tss.iter_mut() {
        match ts.metric_name.get_tag_value(src_label) {
            Some(src_value) => {
                if !r.is_match(&src_value) {
                    continue;
                }
                let b = r.replace_all(&src_value, replacement);
                if b.len() == 0 {
                    ts.metric_name.remove_tag(dst_label)
                } else {
                    ts.metric_name.add_tag(dst_label, &b);
                }
            }
            _ => {
                continue;
            }
        }
    }
    return tss;
}

fn transform_label_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 2)?;

    let label_name = get_label(tfa, "", 1)?;
    let mut x: f64;
    for ts in tfa.args[0].iter_mut() {
        ts.metric_name.reset_metric_group();
        match ts.metric_name.get_tag_value(&label_name) {
            Some(label_value) => {
                x = match label_value.parse::<f64>() {
                    Ok(v) => v,
                    Err(..) => nan,
                };

                for val in ts.values.iter_mut() {
                    *val = x;
                }
            }
            None => {}
        }
    }

    // do not remove timeseries with only NaN values, so `default` could be applied to them:
    // label_value(q, "label") default 123
    return_arg!(tfa, 0)
}

// todo: make this a macro
#[inline]
fn process_regex(
    tfa: &mut TransformFuncArg,
    re: &str,
    handler: fn(tfa: &mut TransformFuncArg, r: &Regex) -> RuntimeResult<Vec<Timeseries>>,
) -> RuntimeResult<Vec<Timeseries>> {
    match Regex::new(re) {
        Err(e) => Err(RuntimeError::from(format!(
            "cannot compile regexp {} : {}",
            &re, e
        ))),
        Ok(ref r) => handler(tfa, r),
    }
}

#[inline]
fn process_anchored_regex(
    tfa: &mut TransformFuncArg,
    re: &str,
    handler: fn(tfa: &mut TransformFuncArg, r: &Regex) -> RuntimeResult<Vec<Timeseries>>,
) -> RuntimeResult<Vec<Timeseries>> {
    let anchored = format!("^(?:{})$", re);
    match Regex::new(&anchored) {
        Err(e) => Err(RuntimeError::from(format!(
            "cannot compile regexp {} : {}",
            &re, e
        ))),
        Ok(ref r) => handler(tfa, r),
    }
}

fn transform_label_match(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 3)?;
    let label_name = get_label(tfa, "", 1)?;
    let label_re = get_label(tfa, "regexp", 1)?;

    process_anchored_regex(tfa, &label_re, |tfa, r| {
        &tfa.args[0].retain(|ts| {
            if let Some(label_value) = ts.metric_name.get_tag_value(&label_name) {
                r.is_match(label_value)
            } else {
                false
            }
        });
        return_arg!(tfa, 0)
    })
}

fn transform_label_mismatch(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 3)?;
    let label_re = get_label(tfa, "regexp", 2)?;

    process_anchored_regex(tfa, &label_re, |tfa, r| {
        let label_name = get_label(tfa, "", 1)?;
        tfa.args[0].retain(|ts| {
            if let Some(label_value) = ts.metric_name.get_tag_value(&label_name) {
                !r.is_match(label_value)
            } else {
                false
            }
        });
        return_arg!(tfa, 0)
    })
}

fn transform_label_graphite_group(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.len() < 2 {
        return Err(RuntimeError::ArgumentError(format!(
            "unexpected number of args: {}; want at least 2 args",
            tfa.args.len()
        )));
    }

    let mut group_ids: Vec<i64> = Vec::with_capacity(tfa.args.len() - 1);
    for i in 1..tfa.args.len() {
        match get_int_number(&tfa, i + 1) {
            Ok(gid) => group_ids.push(gid),
            Err(e) => {
                let msg = format!("cannot get group name from arg #{}: {:?}", i + 1, e);
                return Err(RuntimeError::ArgumentError(msg));
            }
        }
    }
    for ts in tfa.args[0].iter_mut() {
        let groups: Vec<&str> = ts.metric_name.metric_group.split(DOT_SEPARATOR).collect();
        let mut group_name: String =
            String::with_capacity(ts.metric_name.metric_group.len() + groups.len());
        for (j, groupID) in group_ids.iter().enumerate() {
            if *groupID >= 0_i64 && *groupID < (groups.len() as i64) {
                let idx = *groupID as usize;
                group_name.push_str(groups[idx]);
            }
            if j < group_ids.len() - 1 {
                group_name.push('.')
            }
        }
        ts.metric_name.metric_group = group_name
    }

    return_arg!(tfa, 0)
}

const DOT_SEPARATOR: &str = ".";

fn transform_limit_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 3)?;
    let mut limit: usize = 0;
    let mut offset: usize = 0;

    match get_int_number(&tfa, 0) {
        Err(err) => {
            return Err(RuntimeError::from("cannot obtain limit arg: {}"));
        }
        Ok(l) => {
            limit = l as usize;
        }
    }

    match get_int_number(&tfa, 1) {
        Err(err) => {
            return Err(RuntimeError::from("cannot obtain offset arg"));
        }
        Ok(v) => {
            offset = v as usize;
        }
    }

    let mut rvs = &mut tfa.args[2];
    if rvs.len() >= offset {
        rvs.drain(0..offset);
    }
    if rvs.len() > limit {
        rvs.resize(limit, Timeseries::default());
    }

    Ok(rvs)
}

#[inline]
fn transform_ln(v: f64) -> f64 {
    v.ln()
}

#[inline]
fn transform_log2(v: f64) -> f64 {
    v.log2()
}

#[inline]
fn transform_log10(v: f64) -> f64 {
    v.log10()
}

#[inline]
fn transform_minute(t: DateTime<Utc>) -> f64 {
    t.minute() as f64
}

#[inline]
fn transform_month(t: DateTime<Utc>) -> f64 {
    t.month() as f64
}

fn transform_round(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args = &tfa.args;
    if args.len() != 1 && args.len() != 2 {
        return Err(RuntimeError::ArgumentError(format!(
            "unexpected number of args: {}; want 1 or 2",
            args.len()
        )));
    }
    let mut nearest_arg = if args.len() == 1 {
        eval_number(&mut tfa.ec, 1.0)
    } else {
        args[1].drain(0..).collect() // ?? use into ???
    };

    let nearest = get_scalar(&nearest_arg, 1)?;
    let tf = |values: &mut &[f64]| {
        let mut n_prev: f64;
        let mut p10: f64 = 0.0;
        for (i, v) in values.iter_mut().enumerate() {
            let mut n = nearest[i];
            if n != n_prev {
                n_prev = n as f64;
                let (_, e) = from_float(n);
                p10 = -e.powi(10) as f64;
            }
            *v += 0.5 * copysign(n, *v);
            *v -= fmod(*v, n);
            let (x, _) = modf(*v * p10);
            *v = x / p10;
        }
    };
    return do_transform_values(&mut args[0], tf, &tfa.fe);
}

fn transform_sgn(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        for v in values {
            let mut sign = 0.0;
            if *v < 0.0 {
                sign = -1.0;
            } else if *v > 0.0 {
                sign = 1.0;
            }
            *v = sign;
        }
    };

    expect_transform_args_num(&tfa, 1)?;
    do_transform_values(&mut tfa.args[0], tf, tfa.fe)
}

fn transform_scalar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 1)?;

    // Verify whether the arg is a string.
    // Then try converting the string to number.
    match &tfa.fe.args[0] {
        Expression::String(se) => {
            let n = match se.s.parse::<f64>() {
                Ok(n) => n,
                Err(e) => f64::NAN,
            };
            Ok(eval_number(&mut tfa.ec, n))
        }
        _ => {
            // The arg isn't a string. Extract scalar from it.
            if tfa.args.len() != 1 {
                Ok(eval_number(&mut tfa.ec, nan))
            } else {
                let mut arg = tfa.args[0].pop().unwrap();
                arg.metric_name.reset();
                Ok(vec![arg])
            }
        }
    }
}

fn new_transform_func_sort_by_label(is_desc: bool) -> TransformFunc {
    |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        expect_at_least_n_args(&tfa, 2)?;
        let mut labels: Vec<String> = Vec::with_capacity(1);

        for i in 1..tfa.args.len() {
            let label = get_string(tfa, i)?;
            labels.push(label);
        }

        tfa.args[0].sort_by(|first, second| {
            let mut i = 0;
            while i < labels.len() {
                let label = &labels[i];
                i += 1;
                let a = first.metric_name.get_tag_value(&label);
                let b = second.metric_name.get_tag_value(&label);
                match (a, b) {
                    (Some(a1), Some(b1)) => {
                        if a1 == b1 {
                            continue;
                        } else if is_desc {
                            return b1.cmp(&a1);
                        } else {
                            return a1.cmp(&b1);
                        }
                    }
                    (Some(_), None) => return Ordering::Greater,
                    (None, Some(_)) => return Ordering::Less,
                    (None, None) => return Ordering::Equal,
                }
            }
            return Ordering::Equal;
        });

        return_arg!(tfa, 0)
    }
}

fn new_transform_func_alpha_numeric_sort(is_desc: bool) -> TransformFunc {
    move |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        let args = tfa.args;
        if args.len() < 2 {
            return Err(RuntimeError::ArgumentError(format!(
                "expecting at least 2 args; got {} args",
                args.len()
            )));
        }
        let mut labels: Vec<String>;
        for i in 1..tfa.args.len() {
            match get_string(tfa, 1) {
                Err(err) => {
                    return Err(RuntimeError::ArgumentError(format!(
                        "cannot parse label {} for sorting: {:?}",
                        i + 1,
                        err
                    )));
                }
                Ok(label) => labels.push(label),
            }
        }

        tfa.args[0].sort_by(|first, second| {
            for label in labels {
                match (
                    first.metric_name.get_tag_value(&label),
                    second.metric_name.get_tag_value(&label),
                ) {
                    (None, None) => continue,
                    (Some(a), Some(b)) => {
                        return if is_desc {
                            compare_str_alphanumeric(b, a)
                        } else {
                            compare_str_alphanumeric(a, b)
                        };
                    }
                    (None, Some(_)) => {
                        return if is_desc {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        };
                    }
                    (Some(_), None) => {
                        return if is_desc {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        };
                    }
                }
            }
            Ordering::Equal
        });

        return_arg!(tfa, 0)
    }
}

/// Compare two strings.
/// Source: https://crates.io/crates/alphanumeric-sort
pub fn compare_str_alphanumeric<A: AsRef<str>, B: AsRef<str>>(a: A, b: B) -> Ordering {
    let mut c1 = a.as_ref().chars();
    let mut c2 = b.as_ref().chars();

    // this flag is to handle something like "1點" < "1-1點"
    let mut last_is_number = false;

    let mut v1: Option<char> = None;
    let mut v2: Option<char> = None;

    loop {
        let ca = {
            match v1.take() {
                Some(c) => c,
                None => match c1.next() {
                    Some(c) => c,
                    None => {
                        return if v2.take().is_some() || c2.next().is_some() {
                            Ordering::Less
                        } else {
                            Ordering::Equal
                        };
                    }
                },
            }
        };

        let cb = {
            match v2.take() {
                Some(c) => c,
                None => match c2.next() {
                    Some(c) => c,
                    None => {
                        return Ordering::Greater;
                    }
                },
            }
        };

        if ('0'..='9').contains(&ca) && ('0'..='9').contains(&cb) {
            let mut da = f64::from(ca as u32) - f64::from(b'0');
            let mut db = f64::from(cb as u32) - f64::from(b'0');

            // this counter is to handle something like "001" > "01"
            let mut dc = 0isize;

            for ca in c1.by_ref() {
                if ('0'..='9').contains(&ca) {
                    da = da * 10.0 + (f64::from(ca as u32) - f64::from(b'0'));
                    dc += 1;
                } else {
                    v1 = Some(ca);
                    break;
                }
            }

            for cb in c2.by_ref() {
                if ('0'..='9').contains(&cb) {
                    db = db * 10.0 + (f64::from(cb as u32) - f64::from(b'0'));
                    dc -= 1;
                } else {
                    v2 = Some(cb);
                    break;
                }
            }

            last_is_number = true;

            let ordering = da.total_cmp(&db);
            if ordering != Ordering::Equal {
                return ordering;
            } else {
                match dc.cmp(&0) {
                    Ordering::Equal => (),
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Less => return Ordering::Less,
                }
            }
        } else {
            match ca.cmp(&cb) {
                Ordering::Equal => last_is_number = false,
                Ordering::Greater => {
                    return if last_is_number && (ca > (255 as char)) ^ (cb > (255 as char)) {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                }
                Ordering::Less => {
                    return if last_is_number && (ca > (255 as char)) ^ (cb > (255 as char)) {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
            }
        }
    }
}

fn new_transform_func_sort(is_desc: bool) -> TransformFunc {
    return |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        expect_transform_args_num(tfa, 1)?;

        tfa.args[0].sort_by(|first, second| {
            let a = &first.values;
            let b = &second.values;
            let mut n = a.len() - 1;
            while n >= 0 {
                if !a[n].is_nan() && !b[n].is_nan() && a[n] != b[n] {
                    break;
                }
                n -= 1;
            }
            if n < 0 {
                return Ordering::Greater;
            }
            if is_desc {
                return b[n].total_cmp(&a[n]);
            }
            return a[n].total_cmp(&b[n]);
        });

        return_arg!(tfa, 0)
    };
}

#[inline]
fn transform_sqrt(v: f64) -> f64 {
    v.sqrt()
}

#[inline]
fn transform_sin(v: f64) -> f64 {
    v.sin()
}

#[inline]
fn transform_sinh(v: f64) -> f64 {
    v.sinh()
}

#[inline]
fn transform_cos(v: f64) -> f64 {
    v.cos()
}

#[inline]
fn transform_cosh(v: f64) -> f64 {
    v.cosh()
}

#[inline]
fn transform_asin(v: f64) -> f64 {
    v.asin()
}

#[inline]
fn transform_asinh(v: f64) -> f64 {
    v.asinh()
}

#[inline]
fn transform_acos(v: f64) -> f64 {
    v.acos()
}

#[inline]
fn transform_acosh(v: f64) -> f64 {
    v.acosh()
}

#[inline]
fn transform_tan(v: f64) -> f64 {
    v.tan()
}

#[inline]
fn transform_tanh(v: f64) -> f64 {
    v.tanh()
}

#[inline]
fn transform_atan(v: f64) -> f64 {
    v.atan()
}

#[inline]
fn transform_atanh(v: f64) -> f64 {
    v.atanh()
}

#[inline]
fn transform_deg(v: f64) -> f64 {
    v * 180.0 / f64::PI()
}

#[inline]
fn transform_rad(v: f64) -> f64 {
    v * f64::PI() / 180.0
}

fn new_transform_rand(new_rand_func: fn(r: &mut StdRng) -> fn() -> f64) -> TransformFunc {
    |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        if tfa.args.len() > 1 {
            return Err(RuntimeError::ArgumentError(format!(
                "unexpected number of args; got {}; want 0 or 1",
                tfa.args.len()
            )));
        }
        // todo: default to thread_rng if seed is not specified ?
        let mut rng = if tfa.args.len() == 1 {
            let tmp = get_scalar(&tfa, 0)?;
            if tmp.len() > 0 {
                let seed: u64 = tmp[0].floor() as u64;
                StdRng::seed_from_u64(seed)
            } else {
                thread_rng()
            }
        } else {
            thread_rng()
        };
        let rand_func = new_rand_func(&mut rng);
        let tss = eval_number(&mut tfa.ec, 0.0);
        for value in tss[0].values.iter_mut() {
            *value = rand_func();
        }
        Ok(tss)
    }
}

#[inline]
fn new_rand_float64(r: &mut StdRng) -> fn() -> f64 {
    || r.gen()
}

#[inline]
fn new_rand_norm_float64(r: &mut StdRng) -> fn() -> f64 {
    || StandardNormal.sample(r)
}

#[inline]
fn new_rand_exp_float64(r: &mut StdRng) -> fn() -> f64 {
    || Exp1.sample(r)
}

fn transform_pi(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 0)?;
    Ok(eval_number(&mut tfa.ec, f64::PI()))
}

fn transform_now(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 0)?;
    let now: f64 = Utc::now().timestamp() as f64 / 1e9_f64;
    Ok(eval_number(&mut tfa.ec, now))
}

#[inline]
fn bitmap_and(a: u64, b: u64) -> u64 {
    a & b
}

#[inline]
fn bitmap_or(a: u64, b: u64) -> u64 {
    a | b
}

#[inline]
fn bitmap_xor(a: u64, b: u64) -> u64 {
    a ^ b
}

fn new_transform_bitmap(bitmap_func: fn(a: u64, b: u64) -> u64) -> TransformFunc {
    move |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        expect_transform_args_num(tfa, 2)?;
        let ns = get_scalar(&tfa, 1)?;

        let tf = |values: &mut [f64]| {
            let mut i = 0;
            for v in values {
                let b = ns[i];
                *v = bitmap_func(*v as u64, b as u64) as f64;
                i += 1;
            }
        };

        do_transform_values(&mut tfa.args[0], tf, tfa.fe)
    }
}

fn transform_timezone_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;

    let tz_string: String;

    let tz: Tz;

    match get_string(tfa, 0) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot get timezone name from arg: {:?}",
                e
            )));
        }
        Ok(tz_name) => {
            tz_string = tz_name;
            match tz_name.parse() {
                Err(e) => {
                    return Err(RuntimeError::ArgumentError(format!(
                        "cannot load timezone {}: {:?}",
                        tz_string, e
                    )));
                }
                Ok(zone) => {
                    tz = zone;
                }
            }
        }
    }

    let timestamps = tfa.ec.get_shared_timestamps();
    let mut values: Vec<f64> = Vec::with_capacity(timestamps.len());
    for v in timestamps.iter() {
        let dt = Utc.timestamp(v / 1000, 0);
        let in_tz = dt.with_timezone(&tz);
        let ofs = in_tz.offset().local_minus_utc(); // this is in secs
        values.push(ofs as f64);
    }
    let ts = Timeseries {
        metric_name: MetricName::default(),
        values,
        timestamps: timestamps.clone(),
        deny_reuse: false,
    };
    Ok(vec![ts])
}

fn transform_time(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 0)?;
    Ok(eval_time(&mut tfa.ec))
}

fn transform_vector(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(&tfa, 1)?;
    return_arg!(tfa, 0)
}

#[inline]
fn transform_year(t: DateTime<Utc>) -> f64 {
    t.year() as f64
}

fn new_transform_func_zero_args(f: fn(tfa: &mut TransformFuncArg) -> f64) -> TransformFunc {
    |tfa: &mut TransformFuncArg| -> RuntimeResult<Vec<Timeseries>> {
        expect_transform_args_num(&tfa, 0)?;
        let v = f(tfa);
        Ok(eval_number(&mut tfa.ec, v))
    }
}

#[inline]
fn transform_step(tfa: &mut TransformFuncArg) -> f64 {
    return tfa.ec.step as f64 / 1e3_f64;
}

#[inline]
fn transform_start(tfa: &mut TransformFuncArg) -> f64 {
    tfa.ec.start as f64 / 1e3_f64
}

#[inline]
fn transform_end(tfa: &mut TransformFuncArg) -> f64 {
    tfa.ec.end as f64 / 1e3_f64
}

/// copy_timeseries returns a copy of tss.
fn copy_timeseries(tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for src in tss {
        let dst = Timeseries::copy_from_shallow_timestamps(src);
        rvs.push(dst);
    }
    return rvs;
}

/// copy_timeseries_metric_names returns a copy of tss with real copy of MetricNames,
/// but with shallow copy of Timestamps and Values if make_copy is set.
///
/// Otherwise tss is returned.
fn copy_timeseries_metric_names(tss: &Vec<Timeseries>, make_copy: bool) -> Vec<Timeseries> {
    if !make_copy {
        return tss;
    }
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for (i, src) in tss.iter().enumerate() {
        let dst: Timeseries = Timeseries::copy_from_metric_name(src);
        rvs.push(dst);
    }
    return rvs;
}

fn get_dst_value<'a>(mn: &'a mut MetricName, dst_label: &str) -> &'a String {
    match mn.get_tag_value_mut(dst_label) {
        Some(val) => val,
        None => {
            mn.add_tag(dst_label, "");
            mn.get_tag_value_mut(dst_label)?
        }
    }
}

fn is_leap_year(y: u32) -> bool {
    if y % 4 != 0 {
        return false;
    }
    if y % 100 != 0 {
        return true;
    }
    return y % 400 == 0;
}

const DAYS_IN_MONTH: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

fn expect_transform_args_num(tfa: &TransformFuncArg, expected_num: usize) -> RuntimeResult<()> {
    if tfa.args.len() == expected_num {
        return Ok(());
    }
    let msg = format!(
        "unexpected number of args; got {}; want {}",
        tfa.args.len(),
        expected_num
    );
    return Err(RuntimeError::ArgumentError(msg));
}

fn expect_at_least_n_args(tfa: &TransformFuncArg, n: usize) -> RuntimeResult<()> {
    let len = tfa.args.len();
    if len < n {
        let msg = format!("not enough args; got {}; want at least {}", len, n);
        return Err(RuntimeError::ArgumentError(msg));
    }
    Ok(())
}

fn remove_counter_resets_maybe_nans(mut values: &[f64]) {
    values = skip_leading_nans(values);
    if values.len() == 0 {
        return;
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];
    for v in values.iter_mut() {
        if v.is_nan() {
            continue;
        }
        let mut d = *v - prev_value;
        if d < 0.0 {
            if (-d * 8.0) < prev_value {
                // This is likely jitter from `Prometheus HA pairs`.
                // Just substitute v with prev_value.
                *v = prev_value;
            } else {
                correction += prev_value
            }
        }
        prev_value = *v;
        *v = *v + correction;
    }
}

fn get_string(tfa: &TransformFuncArg, arg_num: usize) -> RuntimeResult<String> {
    let tss = &tfa.args[arg_num];
    if tss.len() != 1 {
        let msg = format!(
            "arg # {} must contain a single timeseries; got {} timeseries",
            arg_num + 1,
            tss.len()
        );
        return Err(RuntimeError::ArgumentError(msg));
    }
    let ts = &tss[0];
    for v in ts.values {
        if !v.is_nan() {
            let msg = format!("arg # {} contains non - string timeseries", arg_num + 1);
            return Err(RuntimeError::ArgumentError(msg));
        }
    }
    // todo: return reference
    return Ok(ts.metric_name.metric_group.clone());
}

fn get_label(tfa: &TransformFuncArg, name: &str, arg_num: usize) -> RuntimeResult<String> {
    match get_string(&tfa, arg_num) {
        Ok(lbl) => Ok(lbl),
        Err(err) => {
            let msg = format!("cannot read {} label name", name);
            return Err(RuntimeError::ArgumentError(msg));
        }
    }
}

pub fn get_scalar<'a>(tfa: &'a TransformFuncArg, arg_num: usize) -> RuntimeResult<&'a Vec<f64>> {
    let tss = &tfa.args[arg_num];
    if tss.len() != 1 {
        let msg = format!(
            "arg # {} must contain a single timeseries; got {} timeseries",
            arg_num + 1,
            tss.len()
        );
        Err(RuntimeError::ArgumentError(msg))
    }
    Ok(&tss[0].values)
}

fn get_int_number(tfa: &TransformFuncArg, arg_num: usize) -> RuntimeResult<i64> {
    let v = get_scalar(tfa, arg_num)?;
    let mut n = 0;
    if v.len() > 0 {
        n = v[0] as i64;
    }
    return Ok(n);
}
