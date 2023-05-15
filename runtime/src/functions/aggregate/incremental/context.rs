use crate::functions::aggregate::Handler;
use crate::{RuntimeResult, Timeseries};
use metricsql::ast::AggregationExpr;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
    handler: &'a Handler,
}

impl<'a> IncrementalAggrFuncContext<'a> {
    pub(crate) fn new(ae: &'a AggregationExpr, handler: &'a Handler) -> Self {
        let m: HashMap<u64, HashMap<String, IncrementalAggrContext>> = HashMap::new();

        Self {
            ae,
            context_map: RwLock::new(m),
            handler,
        }
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
                let mut iac = entry.get_mut();
                iac.values.resize(value_len, 0.0); // ?? NaN
                self.handler.update(&mut iac, &ts.values);
            }
        };

        Ok(())
    }

    pub fn finalize(&mut self) -> Vec<Timeseries> {
        // There is no need in iafc.mLock.lock here, since finalize_timeseries must be called
        // without concurrent threads touching iafc.
        let mut m_global: HashMap<&String, IncrementalAggrContext> = HashMap::new();
        let mut hash = self.context_map.write().unwrap();
        for (_, m) in hash.iter_mut() {
            for (k, iac) in m.into_iter() {
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
        return tss;
    }
}
