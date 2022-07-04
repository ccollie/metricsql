use std::collections::HashMap;
use phf::phf_map;
use metricsql::types::AggrFuncExpr;
use crate::timeseries::Timeseries;

// callbacks for optimized incremental calculations for aggregate functions
// over rollups over metricsql.MetricExpr.
//
// These calculations save RAM for aggregates over big number of time series.
static IncrementalAggrFuncCallbacksMap: phf::Map<&'static str, IncrementalAggrFuncCallbacks> = phf_map! {
"sum": IncrementalAggrFuncCallbacks {
updateAggrFunc:   updateAggrSum,
mergeAggrFunc:    mergeAggrSum,
finalizeAggrFunc: finalizeAggrCommon,
},
"min": IncrementalAggrFuncCallbacks{
updateAggrFunc:   updateAggrMin,
mergeAggrFunc:    mergeAggrMin,
finalizeAggrFunc: finalizeAggrCommon,
},
"max": IncrementalAggrFuncCallbacks {
updateAggrFunc:   updateAggrMax,
mergeAggrFunc:    mergeAggrMax,
finalizeAggrFunc: finalizeAggrCommon,
},
"avg": IncrementalAggrFuncCallbacks{
updateAggrFunc:   updateAggrAvg,
mergeAggrFunc:    mergeAggrAvg,
finalizeAggrFunc: finalizeAggrAvg,
},
"count": IncrementalAggrFuncCallbacks{
updateAggrFunc:   updateAggrCount,
mergeAggrFunc:    mergeAggrCount,
finalizeAggrFunc: finalizeAggrCount,
},
"sum2": IncrementalAggrFuncCallbacks{
updateAggrFunc:   updateAggrSum2,
mergeAggrFunc:    mergeAggrSum2,
finalizeAggrFunc: finalizeAggrCommon,
},
"geomean": IncrementalAggrFuncCallbacks{
updateAggrFunc:   updateAggrGeomean,
mergeAggrFunc:    mergeAggrGeomean,
finalizeAggrFunc: finalizeAggrGeomean,
},
"any": IncrementalAggrFuncCallbacks{
    updateAggrFunc:   updateAggrAny,    
    mergeAggrFunc:    mergeAggrAny,
    finalizeAggrFunc: finalizeAggrCommon,
    keepOriginal: true,
},
"group": {
    updateAggrFunc:   updateAggrCount,
    mergeAggrFunc:    mergeAggrCount,
    finalizeAggrFunc: finalizeAggrGroup,
},
};


pub(crate) struct IncrementalAggrContext {
    ts: &Timeseries,
    values: &[f64],
}

impl IncrementalAggrContext {}

pub(crate) struct IncrementalAggrFuncContext {
    pub(crate) ae: AggrFuncExpr,
    mLock: sync.Mutex,
    m: HashMap<usize, HashMap<String, IncrementalAggrContext>>,

    callbacks: IncrementalAggrFuncCallbacks,
}

impl IncrementalAggrFuncContext {
    pub(crate) fn new(ae: &AggrFuncExpr, callbacks: IncrementalAggrFuncCallbacks) -> Self {
        let m: HashMap<usize, HashMap<String, IncrementalAggrContext>> = HashMap::new();
        IncrementalAggrFuncContext {
            ae, // todo: box
            mLock: (),
            m,
            callbacks
        }
    }
}

impl IncrementalAggrFuncContext {
    fn update_timeseries(&mut self, ts_orig: &Timeseries, worker_id: u64) {
        self.mLock.Lock();
        let mut m = iafc.m.get(worker_id);
        if m.is_none() {
            let h: HashMap<String, IncrementalAggrContext> = HashMap::new();
            self.m.set(worker_id, h);
            m = Some(h);
        }
        self.mLock.Unlock();

        let ts = ts_orig;
        let keep_original = iafc.callbacks.keepOriginal;
        if keep_original {
            let dst: Timeseries = ts_orig.copy();
            ts = &dst
        }
        removeGroupTags(&ts.metricName, &iafc.ae.modifier);
        let mut bb = bbPool.Get();
        bb.B = marshalMetricNameSorted(bb.B[: 0], &ts.MetricName)
        let mut iac = m[string(bb.B)];
        if iac == nil {
            if iafc.ae.Limit > 0 && len(m) >= iafc.ae.Limit {
                // Skip this time series, since the limit on the number of output time series has been already reached.
                return;
            }
            let tsAggr = Timeseries::with_shared_timestamps(ts.timestamps, Vec::with_capacity(ts.values.len()));
            if keep_original {
                ts = ts_orig
            }
            tsAggr.metric_name = String::from(&ts.metric_name);
            iac = &IncrementalAggrContext {
                ts: tsAggr,
                values: Vec::with_capacity(ts.values.len()),
            };
            m[string(bb.B)] = iac
        }
        bbPool.Put(bb);
        iafc.callbacks.updateAggrFunc(iac, ts.Values)
    }

    fn finalize_timeseries(&mut self) -> Vec<Timeseries> {
        // There is no need in iafc.mLock.Lock here, since finalize_timeseries must be called
        // without concurrent goroutines touching iafc.
        let mut m_global: HashMap<String, IncrementalAggrFuncContext> = HashMap::new();
        let merge_aggr_func = iafc.callbacks.mergeAggrFunc;
        for m in iafc.m {
            for (k, iac) in m {
                let iac_global = m_global[k];
                if iac_global.is_none() {
                    if iafc.ae.Limit > 0 && m_global.len() >= iafc.ae.Limit {
                        // Skip this time series, since the limit on the number of output time series 
                        // has been already reached.
                        continue;
                    }
                    m_global[k] = iac;
                    continue;
                }
                merge_aggr_func(iac_global, iac)
            }
        }
        let mut tss: Vec<Timeseries> = Vec::with_capacity(m_global.len());
        let finalize_aggrfn = iafc.callbacks.finalizeAggrFunc;
        for iac in m_global {
            finalize_aggrfn(iac);
            tss.push(inc.ts);
        }
        return tss;
    }
}

fn new_incremental_aggr_func_context(
    ae: &AggrFuncExpr,
    callbacks: &IncrementalAggrFuncCallbacks) -> IncrementalAggrFuncContext {
    return IncrementalAggrFuncContext::new(ae, callbacks);
}

// todo: make it a trait ??
pub(crate) struct IncrementalAggrFuncCallbacks {
    updateAggrfn: fn(mut iac: &IncrementalAggrContext, values: &Vec<f64>),
    mergeAggrfn: fn(mut dst: &IncrementalAggrFuncContext, src: &IncrementalAggrContext),
    finalizeAggrfn: fn(mut iac: &IncrementalAggrContext),
    // Whether to keep the original MetricName for every time series during aggregation
    keepOriginal: bool,
}

fn get_incremental_aggr_func_callbacks(name: &str) -> Option<IncrementalAggrFuncCallbacks> {
    let lower = name.to_lowercase().as_str();
    return IncrementalAggrFuncCallbacksMap.get(lower);
}


fn finalize_aggr_common(mut iac: &IncrementalAggrContext) {
    let counts = iac.values;
    let mut dst_values = iac.ts.Values;
    for (i, v) in counts.iter().enumerate() {
        if v == 0 {
            dst_values[i] = f64::NAN
        }
    }
}

fn update_aggr_sum(mut iac: &IncrementalAggrContext, values: &[f64]) {
    let mut dst_values = &iac.ts.values;
    let mut dst_counts = iac.values;

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = *v;
            dst_counts[i] = 1;
            continue;
        }
        dst_values[i] += v
    }
}

fn merge_aggr_sum(mut dst: &IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = src.ts.values;
    let src_counts = src.values;
    let mut dst_values = dst.ts.values;
    let mut dst_counts = dst.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0 {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        dst_values[i] += v
    }
}

fn update_aggr_min(mut iac: &IncrementalAggrContext, values: &[f64]) {
    let dst_values = iac.ts.values;
    let dst_counts = iac.values;

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        if v < dst_values[i] {
            dst_values[i] = v
        }
    }
}

fn merge_aggr_min(mut dst: &IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = src.ts.values;
    let src_counts = src.values;
    let dst_values = &dst.ts.values;
    let dst_counts = &dst.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0 {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        if v < dst_values[i] {
            dst_values[i] = v
        }
    }
}

fn update_aggr_max(mut iac: &IncrementalAggrContext, values: &[f64]) {
    let mut dst_values = iac.ts.values;
    let mut dst_counts = iac.values;

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        if v > dst_values[i] {
            dst_values[i] = v
        }
    }
}

fn merge_aggr_max(mut dst: &IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = src.ts.values;
    let src_counts = src.values;
    let mut dst_values = dst.ts.values;
    let mut dst_counts = dst.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0 {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        if v > dst_values[i] {
            dst_values[i] = v
        }
    }
}

fn update_aggr_avg(mut iac: &IncrementalAggrContext, values: &[f64]) {
// Do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
// since it is slower and has no obvious benefits in increased precision.
    let dst_values = iac.ts.values;
    let dst_counts = iac.values;

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        dst_values[i] += v;
        dst_counts[i] = dst_counts[i] + 1
    }
}

fn merge_aggr_avg(mut dst: &IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = src.ts.values;
    let src_counts = src.values;
    let mut dst_values = dst.ts.values;
    let mut dst_counts = dst.values;

    for (i, v) in src_values.iter().enumerate() {
        if src_counts[i] == 0 {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = src_counts[i];
            continue;
        }
        dst_values[i] += v;
        dst_counts[i] += src_counts[i]
    }
}

fn finalize_aggr_avg(mut iac: &IncrementalAggrContext) {
    let mut dst_values = iac.ts.values;
    let counts = iac.values;

    for (i, v) in counts.iter().enumerate() {
        if v == 0 {
            dst_values[i] = f64::NAN;
            continue;
        }
        dst_values[i] /= v
    }
}

fn update_aggr_count(mut iac: &IncrementalAggrContext, values: &[f64]) {
    let mut dst_values = iac.ts.values;

    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        dst_values[i] = dst_values[i] + 1;
    }
}

fn merge_aggr_count(mut dst: &IncrementalAggrContext, src: IncrementalAggrContext) {
    let mut dst_values = dst.ts.values;
    for (i, v) in src.ts.values {
        dst_values[i] += v;
    }
}

fn finalize_aggr_count(iac: IncrementalAggrContext) {
    let mut dst_values = iac.ts.values;
    for (i, v) in dst_values {
        if v == 0 {
            dst_values[i] = f64::NAN
        }
    }
}

fn finalize_aggr_group(mut iac: &IncrementalAggrContext) {
    let mut dst_values = iac.ts.values;
    for v in dst_values.iter_mut() {
        if v == 0 {
            dst_values[i] = f64::NAN;
        } else {
            dst_values[i] = 1;
        }
    }
}

fn update_aggr_sum2(mut iac: &IncrementalAggrContext, values: &[f64]) {
    let mut dst_values = iac.ts.values;
    let mut dst_counts = iac.values;
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v * v;
            dst_counts[i] = 1;
            continue;
        }
        dst_values[i] += v * v
    }
}

fn merge_aggr_sum2(mut dst: &IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = src.ts.values;
    let src_counts = src.values;
    let mut dst_values = dst.ts.values;
    let mut dst_counts = dst.values;

    for (i, v) in src_values {
        if src_counts[i] == 0 {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        dst_values[i] += v
    }
}

fn updateAggrGeomean(mut iac: &IncrementalAggrContext, values: &[f64]) {
    let mut dst_values = iac.ts.Values;
    let mut dst_counts = iac.values;

    for (i, v) in values {
        if v.is_nan() {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = 1;
            continue;
        }
        dst_values[i] *= v;
        dst_counts[i] = dst_counts[i] + 1;
    }
}

fn mergeAggrGeomean(mut dst: &IncrementalAggrContext, src: IncrementalAggrContext) {
    let src_values = src.ts.values;
    let src_counts = src.values;
    let mut dst_values = dst.ts.values;
    let mut dst_counts = dst.values;

    for (i, v) in src_values {
        if src_counts[i] == 0 {
            continue;
        }
        if dst_counts[i] == 0 {
            dst_values[i] = v;
            dst_counts[i] = src_counts[i];
            continue;
        }
        dst_values[i] *= v;
        dst_counts[i] += src_counts[i]
    }
}

fn finalizeAggrGeomean(iac: IncrementalAggrContext) {
    let dst_values = iac.ts.values;
    let counts = iac.values;

    for (i, v) in counts {
        if v == 0 {
            dst_values[i] = f64::NAN;
            continue;
        }
        dst_values[i] = math.Pow(dst_values[i], 1 / v)
    }
}

fn update_aggr_any(iac: &mut IncrementalAggrContext, values: &[f64]) {
    let mut dst_counts = iac.values;
    if dst_counts[0] > 0 as f64 {
        return;
    }
    for i in 0..values.len() {
        dst_counts[i] = 1;
    }
    iac.ts.values = values;
}

fn merge_aggr_any(mut dst: &IncrementalAggrContext, src: &IncrementalAggrContext) {
    let src_values = &src.ts.values;
    let src_counts = src.values;
    let mut dst_counts = dst.values;
    if dst_counts[0] > 0 {
        return;
    }
    dst_counts[0] = src_counts[0];
    dst.ts.values = src_values;
}