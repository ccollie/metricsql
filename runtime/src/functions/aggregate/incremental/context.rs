use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use metricsql::ast::AggregationExpr;
use metricsql::functions::AggregateFunction;

use crate::functions::aggregate::IncrementalAggregationHandler;
use crate::{RuntimeError, RuntimeResult, Timeseries};

pub enum IncrementalAggrFuncKind {
    Any,
    Avg,
    Count,
    Geomean,
    Min,
    Max,
    Sum,
    Sum2,
    Group,
}

impl TryFrom<&str> for IncrementalAggrFuncKind {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "count" => Ok(IncrementalAggrFuncKind::Count),
            "geomean" => Ok(IncrementalAggrFuncKind::Geomean),
            "min" => Ok(IncrementalAggrFuncKind::Min),
            "max" => Ok(IncrementalAggrFuncKind::Max),
            "avg" => Ok(IncrementalAggrFuncKind::Avg),
            "sum" => Ok(IncrementalAggrFuncKind::Sum),
            "sum2" => Ok(IncrementalAggrFuncKind::Sum2),
            "any" => Ok(IncrementalAggrFuncKind::Any),
            "group" => Ok(IncrementalAggrFuncKind::Group),
            _ => Err(format!("unknown incremental aggregate function: {}", value)),
        }
    }
}

impl TryFrom<AggregateFunction> for IncrementalAggrFuncKind {
    type Error = String;

    fn try_from(value: AggregateFunction) -> Result<Self, Self::Error> {
        match value {
            AggregateFunction::Count => Ok(IncrementalAggrFuncKind::Count),
            AggregateFunction::GeoMean => Ok(IncrementalAggrFuncKind::Geomean),
            AggregateFunction::Min => Ok(IncrementalAggrFuncKind::Min),
            AggregateFunction::Max => Ok(IncrementalAggrFuncKind::Max),
            AggregateFunction::Avg => Ok(IncrementalAggrFuncKind::Avg),
            AggregateFunction::Sum => Ok(IncrementalAggrFuncKind::Sum),
            AggregateFunction::Sum2 => Ok(IncrementalAggrFuncKind::Sum2),
            AggregateFunction::Any => Ok(IncrementalAggrFuncKind::Any),
            AggregateFunction::Group => Ok(IncrementalAggrFuncKind::Group),
            _ => Err(format!(
                "unknown incremental aggregate function: {:?}",
                value
            )),
        }
    }
}

#[derive(Default)]
pub struct IncrementalAggrContext {
    pub ts: Timeseries,
    pub values: Vec<f64>,
}

pub trait IncrementalAggrHandler {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]);
    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext);
    fn finalize(&self, iac: &mut IncrementalAggrContext) {
        let counts = &iac.values;
        for (count, v) in counts.iter().zip(iac.ts.values.iter_mut()) {
            if *count == 0.0 {
                *v = f64::NAN
            }
        }
    }
    /// Whether to keep the original MetricName for every time series during aggregation
    fn keep_original(&self) -> bool;
}

type ContextHash = HashMap<u64, HashMap<String, IncrementalAggrContext>>;

pub struct IncrementalAggrFuncContext<'a> {
    ae: &'a AggregationExpr,
    // todo: use Rc/Arc based on cfg
    context_map: RwLock<ContextHash>,
    handler: IncrementalAggregationHandler,
}

impl<'a> IncrementalAggrFuncContext<'a> {
    pub(crate) fn new(ae: &'a AggregationExpr) -> RuntimeResult<Self> {
        let m: HashMap<u64, HashMap<String, IncrementalAggrContext>> = HashMap::new();
        let handler = IncrementalAggregationHandler::try_from(ae.function).map_err(|e| {
            RuntimeError::General(format!(
                "cannot create incremental aggregation handler: {}",
                e
            ))
        })?;
        Ok(Self {
            ae,
            context_map: RwLock::new(m),
            handler,
        })
    }

    pub fn update_timeseries(&self, ts_orig: &mut Timeseries, worker_id: u64) -> RuntimeResult<()> {
        let mut im = self.context_map.write().unwrap();
        let m = im.entry(worker_id).or_default();

        if self.ae.limit > 0 && m.len() >= self.ae.limit {
            // Skip this time series, since the limit on the number of output time series has been already reached.
            return Ok(());
        }

        // avoid temporary value dropped while borrowed
        let mut ts: &mut Timeseries = ts_orig;

        let keep_original = self.handler.keep_original();
        if !keep_original {
            ts.metric_name.remove_group_tags(&self.ae.modifier);
        }

        let key = ts.metric_name.to_string(); // todo: use hash() ?

        let value_len = ts.values.len();

        match m.entry(key) {
            Vacant(entry) => {
                if keep_original {
                    ts = ts_orig
                }
                let ts_aggr = Timeseries {
                    metric_name: ts.metric_name.clone(),
                    values: vec![0_f64; value_len],
                    timestamps: Arc::clone(&ts.timestamps),
                };

                let mut iac = IncrementalAggrContext {
                    ts: ts_aggr,
                    values: vec![0.0; value_len],
                };

                self.handler.update(&mut iac, &ts.values);
                entry.insert(iac);
            }
            Occupied(mut entry) => {
                let iac = entry.get_mut();
                iac.values.resize(value_len, 0.0); // ?? NaN
                self.handler.update(iac, &ts.values);
            }
        };

        Ok(())
    }

    fn update_timeseries_internal(
        &self,
        ts_orig: &mut Timeseries,
        worker_id: u64,
    ) -> RuntimeResult<()> {
        let mut im = self.context_map.write().unwrap();
        let m = im.entry(worker_id).or_default();

        if self.ae.limit > 0 && m.len() >= self.ae.limit {
            // Skip this time series, since the limit on the number of output time series has been already reached.
            return Ok(());
        }

        // avoid temporary value dropped while borrowed
        let mut ts: &mut Timeseries = ts_orig;

        let keep_original = self.handler.keep_original();
        if !keep_original {
            ts.metric_name.remove_group_tags(&self.ae.modifier);
        }

        let key = ts.metric_name.to_string(); // todo: use hash() ?

        let value_len = ts.values.len();

        match m.entry(key) {
            Vacant(entry) => {
                if keep_original {
                    ts = ts_orig
                }
                let ts_aggr = Timeseries {
                    metric_name: ts.metric_name.clone(),
                    values: vec![0_f64; value_len],
                    timestamps: Arc::clone(&ts.timestamps),
                };

                let mut iac = IncrementalAggrContext {
                    ts: ts_aggr,
                    values: vec![0.0; value_len],
                };

                self.handler.update(&mut iac, &ts.values);
                entry.insert(iac);
            }
            Occupied(mut entry) => {
                let iac = entry.get_mut();
                iac.values.resize(value_len, 0.0); // ?? NaN
                self.handler.update(iac, &ts.values);
            }
        };

        Ok(())
    }

    pub fn finalize(&self) -> Vec<Timeseries> {
        let mut m_global: HashMap<&String, IncrementalAggrContext> = HashMap::new();
        let mut hash = self.context_map.write().unwrap();
        for (_, m) in hash.iter_mut() {
            for (k, iac) in m.iter_mut() {
                match m_global.get_mut(k) {
                    Some(iac_global) => {
                        self.handler.merge(iac_global, iac);
                    }
                    None => {
                        if self.ae.limit > 0 && m_global.len() >= self.ae.limit {
                            // Skip this time series, since the limit on the number of output time series
                            // has been already reached.
                            continue;
                        }
                        m_global.insert(k, std::mem::take(iac));
                    }
                }
            }
        }
        let mut tss: Vec<Timeseries> = Vec::with_capacity(m_global.len());
        for mut iac in m_global.into_values() {
            self.handler.finalize(&mut iac);
            tss.push(std::mem::take(&mut iac.ts));
        }
        tss
    }
}
