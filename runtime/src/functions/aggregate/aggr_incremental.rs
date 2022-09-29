use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::{Arc, RwLock};

use phf::phf_map;

use lib::get_pooled_buffer;
use metricsql::ast::{AggregateModifier, AggrFuncExpr};

use crate::runtime_error::RuntimeResult;
use crate::timeseries::Timeseries;

// todo: make it a trait ??
pub(crate) struct IncrementalAggrFuncCallbacks {
    update_aggr_func: fn(iac: &mut IncrementalAggrContext, values: &[f64]),
    merge_aggr_func: fn(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext),
    finalize_aggr_func: fn(iac: &mut IncrementalAggrContext),
    // Whether to keep the original MetricName for every time series during aggregation
    keep_original: bool,
}

/// callbacks for optimized incremental calculations for aggregate functions
/// over rollup over metricsql.MetricExpr.
///
/// These calculations save RAM for aggregates over big number of time series.
static INCREMENTAL_AGGR_FUNC_CALLBACKS_MAP: phf::Map<&'static str, IncrementalAggrFuncCallbacks> = phf_map! {
"sum" => IncrementalAggrFuncCallbacks {
update_aggr_func:   update_aggr_sum,
merge_aggr_func:    merge_aggr_sum,
finalize_aggr_func: finalize_aggr_common,
        keep_original: false
},
"min" => IncrementalAggrFuncCallbacks{
update_aggr_func:   update_aggr_min,
merge_aggr_func:    merge_aggr_min,
finalize_aggr_func: finalize_aggr_common,
        keep_original: false
},
"max" => IncrementalAggrFuncCallbacks {
update_aggr_func:   update_aggr_max,
merge_aggr_func:    merge_aggr_max,
finalize_aggr_func: finalize_aggr_common,
        keep_original: false
},
"avg" => IncrementalAggrFuncCallbacks{
update_aggr_func:   update_aggr_avg,
merge_aggr_func:    merge_aggr_avg,
finalize_aggr_func: finalize_aggr_avg,
        keep_original: false
},
"count" => IncrementalAggrFuncCallbacks{
update_aggr_func:   update_aggr_count,
merge_aggr_func:    merge_aggr_count,
finalize_aggr_func: finalize_aggr_count,
        keep_original: false
},
"sum2" => IncrementalAggrFuncCallbacks{
update_aggr_func:   update_aggr_sum2,
merge_aggr_func:    merge_aggr_sum2,
finalize_aggr_func: finalize_aggr_common,
        keep_original: false
},
"geomean" => IncrementalAggrFuncCallbacks{
update_aggr_func:   update_aggr_geomean,
merge_aggr_func:    merge_aggr_geomean,
finalize_aggr_func: finalize_aggr_geomean,
        keep_original: false
},
"any" => IncrementalAggrFuncCallbacks{
    update_aggr_func:   update_aggr_any,
    merge_aggr_func:    merge_aggr_any,
    finalize_aggr_func: finalize_aggr_common,
    keep_original: true,
},
"group" => IncrementalAggrFuncCallbacks{
    update_aggr_func:   update_aggr_count,
    merge_aggr_func:    merge_aggr_count,
    finalize_aggr_func: finalize_aggr_group,
        keep_original: false
},
};

#[derive(Default)]
pub(crate) struct IncrementalAggrContext {
    ts: Timeseries,
    values: Vec<f64>,
}

type ContextHash = HashMap<u64, HashMap<String, IncrementalAggrContext>>;

pub(crate) struct IncrementalAggrFuncContext {
    pub(crate) modifier: Option<AggregateModifier>,
    pub(crate) limit: usize,
    // todo: use Rc/Arc based on cfg
    m: Arc<RwLock<ContextHash>>,
    callbacks: &'static IncrementalAggrFuncCallbacks,
}

impl IncrementalAggrFuncContext {
    pub(crate) fn new(ae: &AggrFuncExpr, callbacks: &IncrementalAggrFuncCallbacks) -> Self {
        let m: HashMap<u64, HashMap<String, IncrementalAggrContext>> = HashMap::new();

        IncrementalAggrFuncContext {
            modifier: ae.modifier.clone(),
            limit: ae.limit,
            m: Arc::new(RwLock::new(m)),
            callbacks,
        }
    }

    pub fn update_timeseries(&mut self, ts_orig: &mut Timeseries, worker_id: u64) -> RuntimeResult<()> {
        let mut im = self.m.write().unwrap();
        let m = im.entry(worker_id).or_default();

        if self.limit > 0 && m.len() >= self.limit {
            // Skip this time series, since the limit on the number of output time series has been already reached.
            return Ok(());
        }

        let mut ts = ts_orig;
        let keep_original = self.callbacks.keep_original;
        if keep_original {
            ts = &mut ts_orig.clone();
        }
        ts.metric_name.remove_group_tags(&self.modifier);

        let mut bb = get_pooled_buffer(512);
        let key = ts.metric_name.marshal_to_string(bb.deref_mut()).to_string();

        let mut iac = m.entry(key).or_insert_with(|| {
            let values: Vec<f64> = Vec::with_capacity(ts.values.len());
            let mut ts_aggr = Timeseries::with_shared_timestamps(&ts.timestamps, &values);
            if keep_original {
                ts = ts_orig
            }
            ts_aggr.metric_name = ts.metric_name.clone();
            IncrementalAggrContext {
                ts: ts_aggr,
                values: Vec::with_capacity(ts.values.len()),
            }
        });

        Ok((self.callbacks.update_aggr_func)(&mut iac, &ts.values))
    }

    pub fn finalize_timeseries(&mut self) -> Vec<Timeseries> {
        // There is no need in iafc.mLock.lock here, since finalize_timeseries must be called
        // without concurrent threads touching iafc.
        let mut m_global: HashMap<&String, IncrementalAggrContext> = HashMap::new();
        let merge_aggr_func = self.callbacks.merge_aggr_func;
        let hash = self.m.read().as_mut().unwrap();
        for (_, m) in hash.iter_mut() {
            for (k, iac) in m.into_iter() {
                match m_global.get_mut(k) {
                    Some(iac_global) => {
                        merge_aggr_func(iac_global, iac);
                    },
                    None => {
                        if self.limit > 0 && m_global.len() >= self.limit {
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
        let finalize_aggrfn = self.callbacks.finalize_aggr_func;
        for mut iac in m_global.into_values() {
            finalize_aggrfn(&mut iac);
            tss.push(std::mem::take(&mut iac.ts));
        }
        return tss;
    }
}

pub(crate) fn get_incremental_aggr_func_callbacks(
    name: &str,
) -> Option<&IncrementalAggrFuncCallbacks> {
    return INCREMENTAL_AGGR_FUNC_CALLBACKS_MAP.get(&name.to_lowercase());
}

fn finalize_aggr_common(iac: &mut IncrementalAggrContext) {
    let counts = &iac.values;
    for (i, v) in counts.iter().enumerate() {
        if *v == 0.0 {
            iac.ts.values[i] = f64::NAN
        }
    }
}

fn update_aggr_sum(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] += v;
    }
}

fn merge_aggr_sum(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = 1.0;
            continue;
        }

        dst.ts.values[i] += v;
    }
}

fn update_aggr_min(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        if v < &iac.ts.values[i] {
            iac.ts.values[i] = *v;
        }
    }
}

fn merge_aggr_min(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        let dst_count = dst.values[i];
        let dst_value = dst.ts.values[i];
        if dst_count == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = 1.0;
            continue;
        }
        if *v < dst_value {
            dst.values[i] = *v
        }
    }
}

fn update_aggr_max(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        let dst_value = iac.ts.values[i];

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }
        if v > &dst_value {
            iac.ts.values[i] = *v;
        }
    }
}

fn merge_aggr_max(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    for i in 0 .. src.ts.values.len() {
        let v = src.ts.values[i];
        if src.values[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = v;
            dst.values[i] = 1.0;
            continue;
        }

        if v > dst.ts.values[i] {
            dst.ts.values[i] = v;
        }
    }
}

fn update_aggr_avg(iac: &mut IncrementalAggrContext, values: &[f64]) {
    // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
    // since it is slower and has no obvious benefits in increased precision.
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] += v;
        iac.values[i] += 1.0;
    }
}

// TODO: check original !!!
fn merge_aggr_avg(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.ts.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = src_counts[i];
            continue;
        }

        dst.ts.values[i] += v;
        dst.values[i] += src_counts[i];
    }
}

fn finalize_aggr_avg(iac: &mut IncrementalAggrContext) {
    let counts = &iac.values;

    for (i, v) in counts.iter().enumerate() {
        let dst_value = iac.ts.values[i];
        if *v == 0.0 {
            iac.ts.values[i] = f64::NAN;
            continue;
        }
        iac.ts.values[i]  = dst_value / v;
    }
}

fn update_aggr_count(iac: &mut IncrementalAggrContext, values: &[f64]) {

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        iac.ts.values[i] += 1.0;
    }
}

fn merge_aggr_count(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    for (i, v) in src.ts.values.iter().enumerate() {
        dst.ts.values[i] += v;
    }
}

fn finalize_aggr_count(iac: &mut IncrementalAggrContext) {
    for v in iac.ts.values.iter_mut() {
        if *v == 0.0 {
            *v = f64::NAN
        }
    }
}

fn finalize_aggr_group(iac: &mut IncrementalAggrContext) {
    for v in iac.ts.values.iter_mut() {
        if *v == 0.0 {
            *v = f64::NAN;
        } else {
            *v = 1.0;
        }
    }
}

fn update_aggr_sum2(iac: &mut IncrementalAggrContext, values: &[f64]) {
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        let v_squared = *v * *v;
        if iac.values[i] == 0.0 {
            iac.ts.values[i] = v_squared;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] += v_squared;
    }
}

fn merge_aggr_sum2(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = 1.0;
            continue;
        }

        dst.ts.values[i] += v;
    }
}

fn update_aggr_geomean(iac: &mut IncrementalAggrContext, values: &[f64]) {

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }

        if iac.values[i] == 0.0 {
            iac.ts.values[i] = *v;
            iac.values[i] = 1.0;
            continue;
        }

        iac.ts.values[i] *= v;
        iac.values[i] += 1.0;
    }
}

fn merge_aggr_geomean(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0.0 {
            continue;
        }

        if dst.values[i] == 0.0 {
            dst.ts.values[i] = *v;
            dst.values[i] = src_counts[i];
            continue;
        }

        dst.ts.values[i] *= v;
        dst.values[i] += src_counts[i]
    }
}

fn finalize_aggr_geomean(iac: &mut IncrementalAggrContext) {
    let counts = &iac.values;

    for (i, v) in counts.iter().enumerate() {
        if *v == 0.0 {
            iac.ts.values[i] = f64::NAN;
            continue;
        }
        iac.ts.values[i] = iac.ts.values[i].powf(1.0 / v)
    }
}

fn update_aggr_any(iac: &mut IncrementalAggrContext, values: &[f64]) {
    if iac.values[0] > 0 as f64 {
        return;
    }
    for i in 0..values.len() {
        iac.values[i] = 1.0;
    }
    // ??
    iac.ts.values = Vec::from(values);
}

fn merge_aggr_any(dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = &src.values;
    if dst.values[0] > 0.0 {
        return;
    }
    dst.values[0] = src_counts[0];
    dst.ts.values = src_values.clone();
}
