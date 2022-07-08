use chrono::Duration;
use metricsql::ast::*;
use once_cell::unsync::OnceCell;
use std::sync::Arc;
use crate::context::Context;

use super::rollup::RollupEvaluator;
use super::traits::{NullEvaluator, Evaluator};
use crate::eval::aggregate::{create_aggr_evaluator, AggregateEvaluator};
use crate::eval::binaryop::BinaryOpEvaluator;
use crate::eval::duration::DurationEvaluator;
use crate::eval::function::{create_function_evaluator, TransformEvaluator};
use crate::eval::number::NumberEvaluator;
use crate::eval::string::StringEvaluator;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::Deadline;
use crate::timeseries::Timeseries;
use crate::traits::{Timestamp};

pub(crate) enum ExprEvaluator {
    Null(NullEvaluator),
    Aggregate(AggregateEvaluator),
    BinaryOp(BinaryOpEvaluator),
    Duration(DurationEvaluator),
    Function(TransformEvaluator),
    Number(NumberEvaluator),
    Rollup(RollupEvaluator),
    String(StringEvaluator),
}

impl Evaluator for ExprEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            ExprEvaluator::Null(e) => e.eval(ctx, ec),
            ExprEvaluator::Aggregate(ae) => ae.eval(ctx, ec),
            ExprEvaluator::BinaryOp(bo) => bo.eval(ctx, ec),
            ExprEvaluator::Duration(de) => de.eval(ctx, ec),
            ExprEvaluator::Function(fe) => fe.eval(ctx, ec),
            ExprEvaluator::Number(n) => n.eval(ctx, ec),
            ExprEvaluator::Rollup(ref mut re) => re.eval(ctx, ec),
            ExprEvaluator::String(se) => se.eval(ctx, ec),
        }
    }
}

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
pub static MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: usize = 50;

/// validate_max_points_per_timeseries checks the maximum number of points that
/// may be returned per each time series.
///
/// The number mustn't exceed -search.maxPointsPerTimeseries.
pub(crate) fn validate_max_points_per_timeseries(
    start: i64,
    end: i64,
    step: i64,
    max_points_per_timeseries: usize,
) -> RuntimeResult<()> {
    let points = (end - start) / step + 1;
    if (max_points_per_timeseries > 0) && points > max_points_per_timeseries as i64 {
        let msg = format!("too many points for the given step={}, start={} and end={}: {}; cannot exceed -search.maxPointsPerTimeseries={}",
                          step, start, end, points, max_points_per_timeseries);
        Err(RuntimeError::from(msg))
    } else {
        Ok(())
    }
}

pub fn adjust_start_end(start: i64, end: i64, step: i64) -> (i64, i64) {
    // if disableCache {
    //     // do not adjust start and end values when cache is disabled.
    //     // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/563
    //     return (start, end);
    // }
    let points = (end - start) / step + 1;
    if points < MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING as i64 {
        // Too small number of points for rounding.
        return (start, end);
    }

    // Round start and end to values divisible by step in order
    // to enable response caching (see EvalConfig.mayCache).
    let (start, end) = align_start_end(start, end, step);

    // Make sure that the new number of points is the same as the initial number of points.
    let mut new_points = (end - start) / step + 1;
    let mut _end = end;
    while new_points > points {
        _end = end - step;
        new_points -= 1;
    }

    return (start, _end);
}

pub fn align_start_end(start: i64, end: i64, step: i64) -> (i64, i64) {
    // Round start to the nearest smaller value divisible by step.
    let new_start = start - start % step;
    // Round end to the nearest bigger value divisible by step.
    let adjust = end % step;
    let mut new_end = end;
    if adjust > 0 {
        new_end += step - adjust
    }
    return (new_start, new_end);
}

#[derive(Copy, Clone, Debug)]
pub struct EvalOptions {
    /// Whether to disable response caching. This may be useful during data backfilling
    pub disable_cache: bool,

    /// The maximum points per a single timeseries returned from /api/v1/query_range.
    /// This option doesn't limit the number of scanned raw samples in the database. The main
    /// purpose of this option is to limit the number of per-series points returned to graphing UI
    /// such as Grafana. There is no sense in setting this limit to values bigger than the horizontal
    /// resolution of the graph
    pub max_points_per_timeseries: usize,

    /// Set this flag to true if the database doesn't contain Prometheus stale markers, so there is
    /// no need in spending additional CPU time on its handling. Staleness markers may exist only in
    /// data obtained from Prometheus scrape targets
    pub no_stale_markers: bool,

    /// The maximum interval for staleness calculations. By default it is automatically calculated from
    /// the median interval between samples. This could be useful for tuning Prometheus data model
    /// closer to Influx-style data model.
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness for details.
    /// See also '-search.setLookbackToStep' flag
    pub max_staleness_interval: Duration,

    /// The minimum interval for staleness calculations. This could be useful for removing gaps on
    /// graphs generated from time series with irregular intervals between samples.
    pub min_staleness_interval: Duration,
}

impl EvalOptions {
    pub fn new() -> Self {
        EvalOptions::default()
    }
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            disable_cache: false,
            max_points_per_timeseries: 30e3 as usize,
            no_stale_markers: true,
            max_staleness_interval: Duration::milliseconds(0),
            min_staleness_interval: Duration::milliseconds(0),
        }
    }
}

#[derive(Clone)]
pub struct EvalConfig {
    pub start: Timestamp,
    pub end: Timestamp,
    pub step: Timestamp,

    /// max_series is the maximum number of time series, which can be scanned by the query.
    /// Zero means 'no limit'
    pub max_series: usize,

    /// quoted remote address.
    pub quoted_remote_addr: Option<String>,

    pub deadline: Deadline,

    /// Whether the response can be cached.
    _may_cache: bool,

    /// lookback_delta is analog to `-query.lookback-delta` from Prometheus.
    pub lookback_delta: i64,

    /// How many decimal digits after the point to leave in response.
    pub round_digits: i16,

    /// enforced_tag_filterss may contain additional label filters to use in the query.
    pub enforced_tag_filterss: Vec<Vec<LabelFilter>>,

    pub options: EvalOptions,

    /// The limit on the number of points which can be generated per each returned time series.
    pub max_points_per_series: usize,

    /// Whether to disable response caching. This may be useful during data backfilling
    pub disable_cache: bool,

    ts_cell: OnceCell<Arc<Vec<i64>>>,
}

impl EvalConfig {
    pub fn new() -> Self {
        EvalConfig {
            start: 0,
            end: 0,
            step: 0,
            max_series: 0,
            quoted_remote_addr: None,
            deadline: Deadline::default(),
            _may_cache: false,
            lookback_delta: 0,
            round_digits: 0,
            enforced_tag_filterss: vec![],
            ts_cell: OnceCell::new(),
            options: EvalOptions::default(),
            max_points_per_series: 0,
            disable_cache: false
        }
    }

    pub fn copy_no_timestamps(self) -> EvalConfig {
        let ec = EvalConfig {
            start: self.start,
            end: self.end,
            step: self.step,
            deadline: self.deadline,
            max_series: self.max_series,
            quoted_remote_addr: self.quoted_remote_addr.clone(),
            _may_cache: self._may_cache,
            lookback_delta: self.lookback_delta,
            round_digits: self.round_digits,
            enforced_tag_filterss: vec![],
            // do not copy src.timestamps - they must be generated again.
            options: EvalOptions::default(),
            max_points_per_series: self.max_points_per_series,
            disable_cache: self.disable_cache,
            ts_cell: Default::default(),
        };
        return ec;
    }

    pub fn validate(&self) -> RuntimeResult<()> {
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        if self.step <= 0 {
            let msg = format!("BUG: step must be greater than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        Ok(())
    }

    pub fn may_cache(&self) -> bool {
        if self.options.disable_cache {
            return false;
        }
        if self._may_cache {
            return true;
        }
        if self.start % self.step != 0 {
            return false;
        }
        if self.end % self.step != 0 {
            return false;
        }
        return true;
    }

    pub fn get_shared_timestamps(&mut self) -> &Arc<Vec<i64>> {
        self.ts_cell
            .get_or_init(|| Arc::new(get_timestamps(
                self.start,
                self.end,
                self.step,
                self.max_points_per_series as usize
            )?))
    }
}

pub fn get_timestamps(start: i64, end: i64, step: i64, max_timestamps_per_timeseries: usize) -> RuntimeResult<Vec<i64>> {
    // Sanity checks.
    if step <= 0 {
        let msg = format!("BUG: Step must be bigger than 0; got {}", step);
        return Err(RuntimeError::from(msg));
    }
    if start > end {
        let msg = format!("BUG: Start cannot exceed End; got {} vs {}", start, end);
        return Err(RuntimeError::from(msg));
    }
    match validate_max_points_per_timeseries(start, end, step, max_timestamps_per_timeseries) {
        Err(err) => {
            let msg = format!(
                "BUG: {:?}; this must be validated before the call to get_timestamps",
                err
            );
            return Err(RuntimeError::from(msg));
        }
        _ => (),
    }

    // Prepare timestamps.
    let n: usize = (1 + (end - start) / step) as usize;
    let mut timestamps: Vec<i64> = Vec::with_capacity(n);
    let mut cursor = start;
    for i in 0..n {
        timestamps.push(cursor);
        cursor += step;
    }
    return Ok(timestamps);
}

// copy_eval_config returns src copy.
pub(crate) fn copy_eval_config(src: &EvalConfig) -> EvalConfig {
    src.copy_no_timestamps()
}

pub fn create_evaluator(expr: &Expression) -> RuntimeResult<ExprEvaluator> {
    match expr {
        Expression::Aggregation(ae) => create_aggr_evaluator(ae),
        Expression::MetricExpression(me) => {
            Ok(ExprEvaluator::Rollup(RollupEvaluator::from_metric_expression(*me)?))
        }
        Expression::Rollup(re) => Ok(ExprEvaluator::Rollup(RollupEvaluator::new(re)?)),
        Expression::Function(fe) => create_function_evaluator(fe),
        Expression::BinaryOperator(be) => Ok(
            ExprEvaluator::BinaryOp(BinaryOpEvaluator::new(be)?)
        ),
        Expression::Number(ne) => Ok(ExprEvaluator::Number(NumberEvaluator::new(&ne))),
        Expression::String(se) => Ok(ExprEvaluator::String(StringEvaluator::new(&se))),
        Expression::Duration(de) => Ok(ExprEvaluator::Duration(DurationEvaluator::new(de))),
        Expression::With(_) => {
            panic!("unexpected WITH expression - {}: Should have been expanded during parsing", expr);
        }
        _ => {
            panic!("unexpected expression {}: ", expr);
        }
    }
}

pub(crate) fn create_evaluators(vec: &Vec<BExpression>) -> RuntimeResult<Vec<ExprEvaluator>> {
    let mut res: Vec<ExprEvaluator> = Vec::with_capacity(vec.len());
    for arg in vec {
        match create_evaluator(arg) {
            Err(e) => return Err(e),
            Ok(eval) => { res.push(eval) },
        }
    }
    Ok(res)
}

pub(crate) fn eval_args(
    ctx: &mut Context,
    ec: &mut EvalConfig,
    args: &Vec<ExprEvaluator>,
) -> RuntimeResult<Vec<Vec<Timeseries>>> {
    let mut res: Vec<Vec<Timeseries>> = Vec::with_capacity(args.len());
    for evaluator in args {
        match evaluator.eval(ctx, ec) {
            Err(e) => return Err(e),
            Ok(val) => res.push(val)
        }
    }
    Ok(res)
}

pub(crate) fn eval_number(ec: &mut EvalConfig, n: f64) -> Vec<Timeseries> {
    let timestamps = ec.get_shared_timestamps();
    let values = vec![n; timestamps.len()];
    let ts = Timeseries::with_shared_timestamps(timestamps, &values);
    vec![ts]
}

pub(crate) fn eval_string(ec: &mut EvalConfig, s: &str) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, f64::NAN);
    rv[0].metric_name.metric_group = s.to_string();
    rv
}

pub(crate) fn eval_time(ec: &mut EvalConfig) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, f64::NAN);
    let timestamps = &rv[0].timestamps;
    for (i, ts) in timestamps.iter().enumerate() {
        rv[0].values[i] = *ts as f64 / 1e3_f64;
    }
    rv
}
