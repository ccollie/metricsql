use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use metricsql::ast::Expr;
use metricsql::functions::{can_adjust_window, RollupFunction, TransformFunction};

use crate::common::math::quantile;
use crate::eval::validate_max_points_per_timeseries;
use crate::functions::rollup::rollup_fns::{
    delta_values, deriv_values, remove_counter_resets, rollup_avg, rollup_close, rollup_fake,
    rollup_high, rollup_low, rollup_max, rollup_min, rollup_open,
};
use crate::functions::rollup::{
    get_rollup_fn, get_rollup_func_by_name, RollupFuncArg, RollupHandler, RollupHandlerEnum,
    TimeseriesMap,
};
use crate::types::get_timeseries;
use crate::{get_timestamps, RuntimeError, RuntimeResult, Timestamp};

/// The maximum interval without previous rows.
pub const MAX_SILENCE_INTERVAL: i64 = 5 * 60 * 1000;

// Pre-allocated handlers for closure to save allocations at runtime
macro_rules! wrap_rollup_fn {
    ( $name: ident, $rf: expr ) => {
        pub(crate) const $name: RollupHandlerEnum = RollupHandlerEnum::Wrapped($rf);
    };
}

pub(crate) type PreFunction = fn(&mut [f64], &[i64]) -> ();

#[inline]
pub(crate) fn eval_prefuncs(fns: &Vec<PreFunction>, values: &mut [f64], timestamps: &[i64]) {
    for f in fns {
        f(values, timestamps)
    }
}

#[inline]
fn remove_counter_resets_pre_func(values: &mut [f64], _: &[i64]) {
    remove_counter_resets(values);
}

#[inline]
fn delta_values_pre_func(values: &mut [f64], _: &[i64]) -> () {
    delta_values(values);
}

/// Calculate intervals in seconds between samples.
fn calc_sample_intervals_pre_fn(values: &mut [f64], timestamps: &[i64]) {
    // Calculate intervals in seconds between samples.
    let mut ts_secs_prev = f64::NAN;
    for (value, ts) in values.iter_mut().zip(timestamps.iter()) {
        let ts_secs = (ts / 1000) as f64;
        *value = ts_secs - ts_secs_prev;
        ts_secs_prev = ts_secs;
    }

    if values.len() > 1 {
        // Overwrite the first NaN interval with the second interval,
        // So min, max and avg rollup could be calculated properly,
        // since they don't expect to receive NaNs.
        values[0] = values[1]
    }
}

wrap_rollup_fn!(FN_OPEN, rollup_open);
wrap_rollup_fn!(FN_CLOSE, rollup_close);
wrap_rollup_fn!(FN_MIN, rollup_min);
wrap_rollup_fn!(FN_MAX, rollup_max);
wrap_rollup_fn!(FN_AVG, rollup_avg);
wrap_rollup_fn!(FN_LOW, rollup_low);
wrap_rollup_fn!(FN_HIGH, rollup_high);
wrap_rollup_fn!(FN_FAKE, rollup_fake);

// todo: use tinyvec for return values
pub(crate) fn get_rollup_configs<'a>(
    func: &RollupFunction,
    rf: &'a RollupHandlerEnum,
    expr: &Expr,
    start: Timestamp,
    end: Timestamp,
    step: i64,
    window: i64,
    max_points_per_series: usize,
    min_staleness_interval: usize,
    lookback_delta: i64,
    shared_timestamps: &Arc<Vec<i64>>,
) -> RuntimeResult<(Vec<RollupConfig>, Vec<PreFunction>)> {
    // todo: use tinyvec
    let mut pre_funcs: Vec<PreFunction> = Vec::with_capacity(3);

    if func.should_remove_counter_resets() {
        pre_funcs.push(remove_counter_resets_pre_func);
    }

    let may_adjust_window = can_adjust_window(func);
    let is_default_rollup = *func == RollupFunction::DefaultRollup;
    let samples_scanned_per_call = rollup_samples_scanned_per_call(func);

    let template = RollupConfig {
        tag_value: String::from(""),
        handler: FN_FAKE.clone(),
        start,
        end,
        step,
        window,
        may_adjust_window,
        lookback_delta,
        timestamps: Arc::clone(&shared_timestamps),
        is_default_rollup,
        max_points_per_series,
        min_staleness_interval,
        samples_scanned_per_call,
    };

    let new_rollup_config = |rf: &RollupHandlerEnum, tag_value: &str| -> RollupConfig {
        template.clone_with_fn(rf, tag_value)
    };

    let append_rollup_configs = |dst: &mut Vec<RollupConfig>, expr: &Expr| -> RuntimeResult<()> {
        let tag = get_rollup_tag(expr)?;
        if let Some(tag) = tag {
            match tag.as_str() {
                "min" => dst.push(new_rollup_config(&FN_MIN, "min")),
                "max" => dst.push(new_rollup_config(&FN_MAX, "max")),
                "avg" => dst.push(new_rollup_config(&FN_AVG, "avg")),
                _ => {
                    let msg = format!(
                        "unexpected rollup tag value {}; wanted min, max or avg",
                        tag
                    );
                    return Err(RuntimeError::ArgumentError(msg));
                }
            }
        } else {
            dst.push(new_rollup_config(&FN_MIN, "min"));
            dst.push(new_rollup_config(&FN_MAX, "max"));
            dst.push(new_rollup_config(&FN_AVG, "avg"));
        }
        Ok(())
    };

    // todo: tinyvec
    let mut rcs: Vec<RollupConfig> = Vec::with_capacity(4);
    match func {
        RollupFunction::Rollup => {
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::RollupRate | RollupFunction::RollupDeriv => {
            pre_funcs.push(deriv_values);
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::RollupIncrease | RollupFunction::RollupDelta => {
            pre_funcs.push(delta_values_pre_func);
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::RollupCandlestick => {
            let tag = get_rollup_tag(expr)?;
            if let Some(tag) = tag {
                match tag.as_str() {
                    "open" => rcs.push(new_rollup_config(&FN_OPEN, "open")),
                    "close" => rcs.push(new_rollup_config(&FN_CLOSE, "close")),
                    "low" => rcs.push(new_rollup_config(&FN_LOW, "low")),
                    "high" => rcs.push(new_rollup_config(&FN_HIGH, "high")),
                    _ => {
                        let msg = format!(
                            "unexpected rollup tag value {}; wanted open, close, low or high",
                            tag
                        );
                        return Err(RuntimeError::ArgumentError(msg));
                    }
                }
            } else {
                rcs.push(new_rollup_config(&FN_OPEN, "open"));
                rcs.push(new_rollup_config(&FN_CLOSE, "close"));
                rcs.push(new_rollup_config(&FN_LOW, "low"));
                rcs.push(new_rollup_config(&FN_HIGH, "high"));
            }
        }
        RollupFunction::RollupScrapeInterval => {
            pre_funcs.push(calc_sample_intervals_pre_fn);
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::AggrOverTime => {
            let funcs = get_rollup_aggr_funcs(expr)?;
            for rf in funcs {
                if rf.should_remove_counter_resets() {
                    // There is no need to save the previous pre_func, since it is either empty or the same.
                    pre_funcs.clear();
                    pre_funcs.push(remove_counter_resets_pre_func);
                }
                let rollup_fn = get_rollup_fn(&rf)?;
                let handler = RollupHandlerEnum::wrap(rollup_fn);
                let clone = template.clone_with_fn(&handler, &rf.name());
                rcs.push(clone);
            }
        }
        _ => {
            rcs.push(new_rollup_config(rf, ""));
        }
    }

    Ok((rcs, pre_funcs))
}

#[derive(Clone)]
pub(crate) struct RollupConfig {
    /// This tag value must be added to "rollup" tag if non-empty.
    pub tag_value: String,
    pub handler: RollupHandlerEnum,
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub window: i64,

    /// Whether window may be adjusted to 2 x interval between data points.
    /// This is needed for functions which have dt in the denominator
    /// such as rate, deriv, etc.
    /// Without the adjustment their value would jump in unexpected directions
    /// when using window smaller than 2 x scrape_interval.
    pub may_adjust_window: bool,

    pub timestamps: Arc<Vec<i64>>,

    /// lookback_delta is the analog to `-query.lookback-delta` from Prometheus world.
    pub lookback_delta: i64,

    /// Whether default_rollup is used.
    pub is_default_rollup: bool,

    /// The maximum number of points which can be generated per each series.
    pub max_points_per_series: usize,

    /// The minimum interval for staleness calculations. This could be useful for removing gaps on
    /// graphs generated from time series with irregular intervals between samples.
    pub min_staleness_interval: usize,

    /// The estimated number of samples scanned per Func call.
    ///
    /// If zero, then it is considered that Func scans all the samples passed to it.
    pub samples_scanned_per_call: usize,
}

impl Default for RollupConfig {
    fn default() -> Self {
        Self {
            tag_value: "".to_string(),
            handler: RollupHandlerEnum::Fake("uninitialized"),
            start: 0,
            end: 0,
            step: 0,
            window: 0,
            may_adjust_window: false,
            timestamps: Arc::new(vec![]),
            lookback_delta: 0,
            is_default_rollup: false,
            max_points_per_series: 0,
            min_staleness_interval: 0,
            samples_scanned_per_call: 0,
        }
    }
}

impl RollupConfig {
    fn clone_with_fn(&self, rollup_fn: &RollupHandlerEnum, tag_value: &str) -> Self {
        return RollupConfig {
            tag_value: tag_value.to_string(), // should this be Arc ??
            handler: rollup_fn.clone(),
            start: self.start,
            end: self.end,
            step: self.step,
            window: self.window,
            may_adjust_window: self.may_adjust_window,
            lookback_delta: self.lookback_delta,
            timestamps: Arc::clone(&self.timestamps),
            is_default_rollup: self.is_default_rollup,
            max_points_per_series: self.max_points_per_series,
            min_staleness_interval: self.min_staleness_interval,
            samples_scanned_per_call: self.samples_scanned_per_call,
        };
    }

    // mostly for testing
    pub(crate) fn get_timestamps(&mut self) -> RuntimeResult<Arc<Vec<i64>>> {
        self.ensure_timestamps()?;
        Ok(Arc::clone(&self.timestamps))
    }

    pub(crate) fn ensure_timestamps(&mut self) -> RuntimeResult<()> {
        if self.timestamps.len() == 0 {
            let ts = get_timestamps(self.start, self.end, self.step, self.max_points_per_series)?;
            self.timestamps = Arc::new(ts);
        }
        Ok(())
    }

    /// calculates rollup for the given timestamps and values, appends
    /// them to dst_values and returns results.
    ///
    /// rc.timestamps are used as timestamps for dst_values.
    ///
    /// timestamps must cover time range [rc.start - rc.window - MAX_SILENCE_INTERVAL ... rc.end].
    pub(crate) fn exec(
        &self,
        dst_values: &mut Vec<f64>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        self.do_internal(dst_values, None, values, timestamps)
    }

    /// calculates rollup for the given timestamps and values and puts them to tsm.
    /// returns the number of samples scanned
    pub(crate) fn do_timeseries_map(
        &self,
        tsm: &Rc<RefCell<TimeseriesMap>>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        let mut ts = get_timeseries();
        self.do_internal(&mut ts.values, Some(tsm), values, timestamps)
    }

    fn do_internal(
        &self,
        dst_values: &mut Vec<f64>,
        tsm: Option<&Rc<RefCell<TimeseriesMap>>>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        // Sanity checks.
        self.validate()?;

        // Extend dst_values in order to remove allocations below.
        dst_values.reserve(self.timestamps.len());

        let scrape_interval = get_scrape_interval(&timestamps);
        let mut max_prev_interval = get_max_prev_interval(scrape_interval);
        if self.lookback_delta > 0 && max_prev_interval > self.lookback_delta {
            max_prev_interval = self.lookback_delta
        }
        if self.min_staleness_interval > 0 {
            let msi = self.min_staleness_interval as i64;
            if msi > 0 && max_prev_interval < msi {
                max_prev_interval = msi
            }
        }
        let mut window = self.window;
        if window <= 0 {
            window = self.step;
            if self.may_adjust_window && window < max_prev_interval {
                // Adjust lookbehind window only if it isn't set explicitly, e.g. rate(foo).
                // In the case of missing lookbehind window it should be adjusted in order to return non-empty graph
                // when the window doesn't cover at least two raw samples (this is what most users expect).
                //
                // If the user explicitly sets the lookbehind window to some fixed value, e.g. rate(foo[1s]),
                // then it is expected he knows what he is doing. Do not adjust the lookbehind window then.
                //
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3483
                window = max_prev_interval
            }
            if self.is_default_rollup && self.lookback_delta > 0 && window > self.lookback_delta {
                // Implicit window exceeds -search.maxStalenessInterval, so limit it to
                // -search.maxStalenessInterval
                // according to https://github.com/VictoriaMetrics/VictoriaMetrics/issues/784
                window = self.lookback_delta
            }
        }

        let mut rfa = RollupFuncArg::default();
        rfa.idx = 0;
        rfa.window = window;
        rfa.tsm = if let Some(t) = tsm {
            Some(Rc::clone(t))
        } else {
            None
        };

        let mut i = 0;
        let mut j = 0;
        let mut ni = 0;
        let mut nj = 0;

        let mut samples_scanned = values.len() as u64;
        let samples_scanned_per_call = self.samples_scanned_per_call as u64;

        for t_end in self.timestamps.iter() {
            let t_start = *t_end - window;
            ni = seek_first_timestamp_idx_after(&timestamps[i..], t_start, ni);
            i += ni;
            if j < i {
                j = i;
            }

            nj = seek_first_timestamp_idx_after(&timestamps[j..], *t_end, nj);
            j += nj;

            rfa.prev_value = f64::NAN;
            rfa.prev_timestamp = t_start - max_prev_interval;
            if i > 0 && i < timestamps.len() {
                let prev_ts = timestamps[i - 1];
                if prev_ts > rfa.prev_timestamp {
                    rfa.prev_value = values[i - 1];
                    rfa.prev_timestamp = prev_ts;
                }
            }

            rfa.values.clear();
            rfa.timestamps.clear();
            rfa.values.extend_from_slice(&values[i..j]);
            rfa.timestamps.extend_from_slice(&timestamps[i..j]);

            rfa.real_prev_value = if i > 0 { values[i - 1] } else { f64::NAN };
            rfa.real_next_value = if j < values.len() {
                values[j]
            } else {
                f64::NAN
            };

            rfa.curr_timestamp = *t_end;
            let value = (self.handler).eval(&mut rfa);
            rfa.idx += 1;

            if samples_scanned_per_call > 0 {
                samples_scanned += samples_scanned_per_call
            } else {
                samples_scanned += rfa.values.len() as u64;
            }

            dst_values.push(value);
        }

        Ok(samples_scanned)
    }

    fn validate(&self) -> RuntimeResult<()> {
        // Sanity checks.
        if self.step <= 0 {
            let msg = format!("BUG: step must be bigger than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        if self.window < 0 {
            let msg = format!("BUG: window must be non-negative; got {}", self.window);
            return Err(RuntimeError::from(msg));
        }
        match validate_max_points_per_timeseries(
            self.start,
            self.end,
            self.step,
            self.max_points_per_series,
        ) {
            Err(err) => {
                let msg = format!(
                    "BUG: {:?}; this must be validated before the call to rollupConfig.exec",
                    err
                );
                return Err(RuntimeError::from(msg));
            }
            _ => Ok(()),
        }
    }
}

/// rollup_samples_scanned_per_call contains functions which scan lower number of samples
/// than is passed to the rollup func.
///
/// It is expected that the remaining rollupFuncs scan all the samples passed to them.
const fn rollup_samples_scanned_per_call(rf: &RollupFunction) -> usize {
    use RollupFunction::*;

    return match rf {
        AbsentOverTime => 1,
        CountOverTime => 1,
        DefaultRollup => 1,
        Delta => 2,
        DeltaPrometheus => 2,
        DerivFast => 2,
        FirstOverTime => 1,
        IDelta => 2,
        IDeriv => 2,
        Increase => 2,
        IncreasePrometheus => 2,
        IncreasePure => 2,
        IRate => 2,
        Lag => 1,
        LastOverTime => 1,
        Lifetime => 2,
        PresentOverTime => 1,
        Rate => 2,
        ScrapeInterval => 2,
        TFirstOverTime => 1,
        Timestamp => 1,
        TimestampWithName => 1,
        TLastOverTime => 1,
        _ => 0, // == num rows
    };
}

fn seek_first_timestamp_idx_after(
    timestamps: &[Timestamp],
    seek_timestamp: Timestamp,
    n_hint: usize,
) -> usize {
    let mut timestamps = timestamps;
    let count = timestamps.len();

    if count == 0 || timestamps[0] > seek_timestamp {
        return 0;
    }
    let mut start_idx = if n_hint >= 2 { n_hint - 2 } else { 0 };
    if start_idx >= count {
        start_idx = count - 1
    }

    let mut end_idx = n_hint + 2;
    if end_idx > count {
        end_idx = count
    }
    if start_idx > 0 && timestamps[start_idx] <= seek_timestamp {
        timestamps = &timestamps[start_idx..];
        end_idx -= start_idx
    } else {
        start_idx = 0
    }
    if end_idx < timestamps.len() && timestamps[end_idx] > seek_timestamp {
        timestamps = &timestamps[0..end_idx];
    }
    if timestamps.len() < 16 {
        // Fast path: the number of timestamps to search is small, so scan them all.
        for (i, timestamp) in timestamps.iter().enumerate() {
            if *timestamp > seek_timestamp {
                return start_idx + i;
            }
        }
        return start_idx + timestamps.len();
    }
    // Slow path: too big timestamps.len(), so use binary search.
    let requested = seek_timestamp + 1;
    match timestamps.binary_search(&requested) {
        Ok(pos) => start_idx + pos,
        Err(suggested) => start_idx + suggested,
    }
}

fn get_scrape_interval(timestamps: &[Timestamp]) -> i64 {
    if timestamps.len() < 2 {
        return MAX_SILENCE_INTERVAL;
    }

    // Estimate scrape interval as 0.6 quantile for the first 20 intervals.
    let mut ts_prev = timestamps[0];
    let timestamps = &timestamps[1..];
    let len = timestamps.len().clamp(0, 20);

    let mut intervals: [f64; 20] = [0_f64; 20];
    for (interval, ts) in intervals.iter_mut().zip(timestamps.iter()) {
        *interval = (ts - ts_prev) as f64;
        ts_prev = *ts
    }
    let scrape_interval = quantile(0.6, &intervals[0..len]) as i64;
    if scrape_interval <= 0 {
        return MAX_SILENCE_INTERVAL;
    }
    return scrape_interval;
}

fn get_max_prev_interval(scrape_interval: i64) -> i64 {
    // Increase scrape_interval more for smaller scrape intervals in order to hide possible gaps
    // when high jitter is present.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/139 .
    if scrape_interval <= 2_000i64 {
        return scrape_interval + 4 * scrape_interval;
    }
    if scrape_interval <= 4_000i64 {
        return scrape_interval + 2 * scrape_interval;
    }
    if scrape_interval <= 8_000i64 {
        return scrape_interval + scrape_interval;
    }
    if scrape_interval <= 16_000i64 {
        return scrape_interval + scrape_interval / 2;
    }
    if scrape_interval <= 32_000i64 {
        return scrape_interval + scrape_interval / 4;
    }
    return scrape_interval + scrape_interval / 8;
}

fn get_rollup_tag(expr: &Expr) -> RuntimeResult<Option<&String>> {
    return if let Expr::Function(fe) = expr {
        if fe.args.len() < 2 {
            return Ok(None);
        }
        if fe.args.len() != 2 {
            let msg = format!(
                "unexpected number of args for rollup function {}; got {:?}; want 2",
                fe.name, fe.args
            );
            return Err(RuntimeError::ArgumentError(msg));
        }
        let arg = &fe.args[1];
        if let Expr::StringLiteral(se) = arg {
            if se.is_empty() {
                return Err(RuntimeError::ArgumentError(
                    "unexpected empty rollup tag value".to_string(),
                ));
            }
            Ok(Some(se))
        } else {
            Err(RuntimeError::ArgumentError(format!(
                "unexpected rollup tag value {arg}; wanted min, max or avg",
            )))
        }
    } else {
        let msg = format!("BUG: unexpected expression; want FunctionExpr; got {expr};");
        Err(RuntimeError::ArgumentError(msg))
    };
}

// todo: use in optimize so its cached in the ast
fn get_rollup_aggr_funcs(expr: &Expr) -> RuntimeResult<Vec<RollupFunction>> {
    fn get_func_by_name(name: &str) -> RuntimeResult<RollupFunction> {
        return if let Ok(func) = get_rollup_func_by_name(name) {
            if !func.is_aggregate_function() {
                let msg = format!(
                    "{name} cannot be used in `aggr_over_time` function; expecting aggregate function name",
                );
                return Err(RuntimeError::General(msg));
            }
            Ok(func)
        } else {
            let msg =
                format!("Unknown aggregate function {name} used in `aggr_over_time` function;",);
            Err(RuntimeError::ArgumentError(msg))
        };
    }

    fn get_funcs(args: &[Expr]) -> RuntimeResult<Vec<RollupFunction>> {
        if args.is_empty() {
            return Err(RuntimeError::ArgumentError(
                "aggr_over_time() must contain at least a single aggregate function name"
                    .to_string(),
            ));
        }
        let mut funcs = Vec::with_capacity(args.len());
        for arg in args.iter() {
            if let Expr::StringLiteral(name) = arg {
                let func = get_func_by_name(&name)?;
                funcs.push(func)
            } else {
                let msg = format!(
                    "{arg} cannot be passed here; expecting quoted aggregate function name",
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
        }
        Ok(funcs)
    }

    let expr = match expr {
        Expr::Aggregation(afe) => {
            // This is for incremental aggregate function case:
            //
            //     sum(aggr_over_time(...))
            // See aggr_incremental.rs for details.
            &afe.args[0]
        }
        _ => expr,
    };
    match expr {
        Expr::Function(fe) => {
            let is_aggr_over_time = fe.is_rollup_function(RollupFunction::AggrOverTime);
            if !is_aggr_over_time {
                return Err(RuntimeError::ArgumentError(format!(
                    "BUG: unexpected function name: {}; want `aggr_over_time`",
                    fe.name
                )));
            }
            if fe.args.len() < 2 {
                let msg = format!(
                    "unexpected number of args to aggr_over_time(); got {}; want at least 2",
                    fe.args.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }

            let args = &fe.args[0];
            return match args {
                Expr::StringLiteral(name) => {
                    let func = get_func_by_name(&name)?;
                    Ok(vec![func])
                }
                Expr::Parens(pe) => {
                    if pe.expressions.is_empty() {
                        return Err(RuntimeError::ArgumentError(
                            "unexpected empty parens; want at least one expression inside parens"
                                .to_string(),
                        ));
                    }
                    get_funcs(&pe.expressions)
                }
                Expr::Function(fe) if fe.is_transform_function(TransformFunction::Union) => {
                    get_funcs(&fe.args)
                }
                _ => Err(RuntimeError::General(format!(
                    "{args} cannot be passed here; expecting quoted aggregate function name"
                ))),
            };
        }
        _ => {
            let msg = format!(
                "BUG: unexpected expression; want FunctionExpr; got {}; value: {expr}",
                expr.variant_name()
            );
            Err(RuntimeError::ArgumentError(msg))
        }
    }
}
