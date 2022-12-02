use std::sync::Arc;
use metricsql::ast::*;
use metricsql::binaryop::{eval_binary_op, string_compare};
use metricsql::functions::{DataType, Signature, Volatility};

use crate::context::Context;
use crate::eval::aggregate::{AggregateEvaluator, create_aggr_evaluator};
use crate::eval::binary::BinaryEvaluator;
use crate::eval::duration::DurationEvaluator;
use crate::eval::function::{create_function_evaluator, TransformEvaluator};
use crate::eval::instant_vector::InstantVectorEvaluator;
use crate::eval::scalar::ScalarEvaluator;
use crate::eval::string::StringEvaluator;
use crate::functions::types::AnyValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::Deadline;
use crate::timeseries::Timeseries;
use crate::traits::Timestamp;

use super::rollup::RollupEvaluator;
use super::traits::{Evaluator, NullEvaluator};

pub enum ExprEvaluator {
    Null(NullEvaluator),
    Aggregate(AggregateEvaluator),
    BinaryOp(BinaryEvaluator),
    Duration(DurationEvaluator),
    Function(TransformEvaluator),
    Number(ScalarEvaluator),
    Rollup(RollupEvaluator),
    String(StringEvaluator),
    InstantVector(InstantVectorEvaluator)
}

impl ExprEvaluator {
    /// returns true if the evaluator returns a const value i.e. calling it is
    /// essentially idempotent
    pub fn is_const(&self) -> bool {
        match self {
            ExprEvaluator::Number(_) |
            ExprEvaluator::String(_) |
            ExprEvaluator::Null(_) => true,
            ExprEvaluator::Duration(d) => d.is_const(),
            _ => false
        }
    }
}
impl Evaluator for ExprEvaluator {
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        match self {
            ExprEvaluator::Null(e) => e.eval(ctx, ec),
            ExprEvaluator::Aggregate(ae) => ae.eval(ctx, ec),
            ExprEvaluator::BinaryOp(bo) => bo.eval(ctx, ec),
            ExprEvaluator::Duration(de) => de.eval(ctx, ec),
            ExprEvaluator::Function(fe) => fe.eval(ctx, ec),
            ExprEvaluator::Number(n) => n.eval(ctx, ec),
            ExprEvaluator::Rollup(ref re) => re.eval(ctx, ec),
            ExprEvaluator::String(se) => se.eval(ctx, ec),
            ExprEvaluator::InstantVector(iv) => iv.eval(ctx, ec)
        }
    }

    fn return_type(&self) -> DataType {
        match self {
            ExprEvaluator::Null(e) => e.return_type(),
            ExprEvaluator::Aggregate(ae) => ae.return_type(),
            ExprEvaluator::BinaryOp(bo) => bo.return_type(),
            ExprEvaluator::Duration(de) => de.return_type(),
            ExprEvaluator::Function(fe) => fe.return_type(),
            ExprEvaluator::Number(n) => n.return_type(),
            ExprEvaluator::Rollup(ref re) => re.return_type(),
            ExprEvaluator::String(se) => se.return_type(),
            ExprEvaluator::InstantVector(iv) => iv.return_type()
        }
    }
}

impl Default for ExprEvaluator {
    fn default() -> Self {
        ExprEvaluator::Null(NullEvaluator{})
    }
}

impl From<i64> for ExprEvaluator {
    fn from(val: i64) -> Self {
        Self::Number(ScalarEvaluator::from(val as f64))
    }
}

impl From<f64> for ExprEvaluator {
    fn from(val: f64) -> Self {
        Self::Number(ScalarEvaluator::from(val))
    }
}

impl From<String> for ExprEvaluator {
    fn from(val: String) -> Self {
        Self::String(StringEvaluator::new(&val))
    }
}

impl From<&str> for ExprEvaluator {
    fn from(val: &str) -> Self {
        Self::String(StringEvaluator::new(val))
    }
}

pub fn create_evaluator(expr: &Expression) -> RuntimeResult<ExprEvaluator> {
    match expr {
        Expression::Aggregation(ae) => create_aggr_evaluator(ae),
        Expression::MetricExpression(me) => {
            Ok(ExprEvaluator::Rollup(RollupEvaluator::from_metric_expression(me.clone())?))
        }
        Expression::Rollup(re) => Ok(ExprEvaluator::Rollup(RollupEvaluator::new(re)?)),
        Expression::Function(fe) => create_function_evaluator(fe),
        Expression::BinaryOperator(be) => create_evaluator_from_binop(be),
        Expression::Number(ne) => Ok(ExprEvaluator::from(ne.value)),
        Expression::String(se) => Ok(ExprEvaluator::from(se.value())),
        Expression::Duration(de) => {
            if de.requires_step {
                Ok(ExprEvaluator::Duration(DurationEvaluator::new(de)))
            } else {
                Ok(ExprEvaluator::from(de.value))
            }
        },
        Expression::Parens(pe) => create_parens_evaluator(pe),
        Expression::With(_) => {
            panic!("unexpected WITH expression - {}: Should have been expanded during parsing", expr);
        }
    }
}

fn create_parens_evaluator(expr: &ParensExpr) -> RuntimeResult<ExprEvaluator> {
    if expr.len() == 1 {
        let mut exp = expr;
        let res = exp.expressions[0].clone(); // todo: can we take ??
        return create_evaluator(&res);
    }
    // Treat parensExpr as a function with empty name, i.e. union()
    let fe: FuncExpr;
    match FuncExpr::new("union", expr.expressions.clone(), expr.span) {
        Err(_) => return Err(RuntimeError::UnknownFunction("union".to_string())),
        Ok(f) => fe = f
    }
    create_function_evaluator(&fe)
}

fn create_evaluator_from_binop(be: &BinaryOpExpr) -> RuntimeResult<ExprEvaluator> {
    match (be.left.as_ref(), be.right.as_ref()) {
        (Expression::Number(ln), Expression::Number(rn)) => {
            let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
            Ok(ExprEvaluator::from(n))
        }
        (Expression::String(lhs), Expression::String(rhs)) => {
            if be.op == BinaryOp::Add {
                let val = format!("{}{}", lhs.value, rhs.value);
                return Ok(ExprEvaluator::from(val))
            }
            let n = match string_compare(&lhs.value, &rhs.value, be.op) {
                Ok(v) => {
                    if v {
                        1.0
                    } else {
                        if be.bool_modifier { 0.0 } else { f64::NAN }
                    }
                }
                Err(_) => {
                    // todo: should be unreachable
                    f64::NAN
                }
            };
            Ok(ExprEvaluator::from(n))
        }
        _ => {
            Ok(ExprEvaluator::BinaryOp(BinaryEvaluator::new(be)?))
        },
    }
}

pub(crate) fn create_evaluators(vec: &[BExpression]) -> RuntimeResult<Vec<ExprEvaluator>> {
    let mut res: Vec<ExprEvaluator> = Vec::with_capacity(vec.len());
    for arg in vec {
        match create_evaluator(arg) {
            Err(e) => return Err(e),
            Ok(eval) => { res.push(eval) },
        }
    }
    Ok(res)
}

/// validate_max_points_per_timeseries checks the maximum number of points that
/// may be returned per each time series.
///
/// The number mustn't exceed -search.maxPointsPerTimeseries.
pub(crate) fn validate_max_points_per_timeseries(
    start: Timestamp,
    end: Timestamp,
    step: i64,
    max_points_per_timeseries: usize,
) -> RuntimeResult<()> {
    let points = (end - start) / step + 1;
    if (max_points_per_timeseries > 0) && points > max_points_per_timeseries as i64 {
        let msg = format!("too many points for the given step={}, start={} and end={}: {}; cannot exceed {}",
                          step, start, end, points, max_points_per_timeseries);
        Err(RuntimeError::from(msg))
    } else {
        Ok(())
    }
}

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i64 = 50;

pub fn adjust_start_end(start: Timestamp, end: Timestamp, step: i64) -> (Timestamp, Timestamp) {
    // if disableCache {
    //     // do not adjust start and end values when cache is disabled.
    //     // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/563
    //     return (start, end);
    // }
    let points = (end - start) / step + 1;
    if points < MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING {
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

pub fn align_start_end(start: Timestamp, end: Timestamp, step: i64) -> (Timestamp, Timestamp) {
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


#[derive(Clone)]
pub struct EvalConfig {
    pub start: Timestamp,
    pub end: Timestamp,
    pub step: i64, // todo: Duration

    /// max_series is the maximum number of time series which can be scanned by the query.
    /// Zero means 'no limit'
    pub max_series: usize,

    /// quoted remote address.
    pub quoted_remote_addr: Option<String>,

    pub deadline: Deadline,

    /// Whether the response can be cached.
    _may_cache: bool,

    /// lookback_delta is analog to `-query.lookback-delta` from Prometheus.
    /// todo: change type to Duration
    pub lookback_delta: i64,

    /// How many decimal digits after the point to leave in response.
    pub round_digits: i16,

    /// enforced_tag_filterss may contain additional label filters to use in the query.
    pub enforced_tag_filterss: Vec<Vec<LabelFilter>>,

    /// Set this flag to true if the data doesn't contain Prometheus stale markers, so there is
    /// no need in spending additional CPU time on its handling. Staleness markers may exist only in
    /// data obtained from Prometheus scrape targets
    pub no_stale_markers: bool,

    /// The limit on the number of points which can be generated per each returned time series.
    pub max_points_per_series: usize,

    /// Whether to disable response caching. This may be useful during data back-filling
    pub disable_cache: bool,

    _timestamps: Arc<Vec<Timestamp>>,
}

impl EvalConfig {
    pub fn new(start: Timestamp, end: Timestamp, step: i64) -> Self {
        let mut result = EvalConfig::default();
        result.start = start;
        result.end = end;
        result.step = step;
        result
    }

    pub fn copy_no_timestamps(&self) -> EvalConfig {
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
            enforced_tag_filterss: self.enforced_tag_filterss.clone(),
            // do not copy src.timestamps - they must be generated again.
            _timestamps: Arc::new(vec![]),
            no_stale_markers: self.no_stale_markers,
            max_points_per_series: self.max_points_per_series,
            disable_cache: self.disable_cache,
        };
        return ec;
    }

    pub fn adjust_by_offset(&mut self, offset: i64) {
        self.start -= offset;
        self.end -= offset;
        self._timestamps = Arc::new(vec![]);
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
        if self.disable_cache {
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

        true
    }

    pub fn no_cache(&mut self) {
        self._may_cache = false
    }

    pub fn set_caching(&mut self, may_cache: bool) {
        self._may_cache = may_cache;
    }

    pub fn update_from_context(&mut self, ctx: &Context) {
        let state_config = &ctx.config;
        self.disable_cache = state_config.disable_cache;
        self.max_points_per_series = state_config.max_points_subquery_per_timeseries as usize;
        self.no_stale_markers = state_config.no_stale_markers;
        self.lookback_delta = state_config.max_lookback.num_milliseconds();
        self.max_series = state_config.max_unique_timeseries;
    }

    pub fn timestamps(&self) -> Arc<Vec<i64>> {
        Arc::clone(&self._timestamps)
    }

    pub fn get_timestamps(&mut self) -> Arc<Vec<Timestamp>> {
        self.ensure_timestamps().unwrap(); //???
        Arc::clone(&self._timestamps)
    }

    pub(crate) fn ensure_timestamps(&mut self) -> RuntimeResult<()> {
        if self._timestamps.len() == 0 {
            let ts = get_timestamps(
                self.start,
                self.end,
                self.step,
                self.max_points_per_series as usize
            )?;
            self._timestamps = Arc::new(ts);
        }
        Ok(())
    }

    pub fn get_shared_timestamps(&mut self) -> Arc<Vec<i64>> {
        self.get_timestamps()
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            step: 0,
            max_series: 0,
            quoted_remote_addr: None,
            deadline: Deadline::default(),
            _may_cache: false,
            lookback_delta: 0,
            round_digits: 100,
            enforced_tag_filterss: vec![],
            no_stale_markers: true,
            max_points_per_series: 0,
            disable_cache: false,
            _timestamps: Arc::new(vec![])
        }
    }
}

impl From<&Context> for EvalConfig {
    fn from(ctx: &Context) -> Self {
        let mut config = EvalConfig::default();
        config.update_from_context(ctx);
        config
    }
}

pub fn get_timestamps(start: Timestamp, end: Timestamp, step: i64, max_timestamps_per_timeseries: usize) -> RuntimeResult<Vec<i64>> {
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
    while cursor < end {
        timestamps.push(cursor);
        cursor += step;
    }

    return Ok(timestamps);
}

pub(crate) fn eval_number(ec: &EvalConfig, n: f64) -> Vec<Timeseries> {
    let timestamps = ec.timestamps();
    let values = vec![n; timestamps.len()];
    let ts = Timeseries::with_shared_timestamps(&timestamps, &values);
    vec![ts]
}

pub(crate) fn eval_time(ec: &EvalConfig) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, f64::NAN);
    for i in 0 .. rv[0].timestamps.len() {
        let ts = rv[0].timestamps[i];
        rv[0].values[i] = ts as f64 / 1e3_f64;
    }
    rv
}


pub(super) fn eval_volatility(sig: &Signature, args: &Vec<ExprEvaluator>) -> Volatility {
    if sig.volatility != Volatility::Immutable {
        return sig.volatility
    }

    let mut has_volatile = false;
    let mut mutable = false;

    for arg in args.iter() {
        let vol = arg.volatility();
        if vol != Volatility::Immutable {
            mutable = true;
            has_volatile = vol == Volatility::Volatile;
        }
    }

    if mutable {
        return if has_volatile {
            Volatility::Volatile
        } else {
            Volatility::Stable
        }
    } else {
        Volatility::Immutable
    }
}
