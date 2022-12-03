
// Calculates output=stddev, e.g. the average value over input samples.
pub(crate) struct StdDevAggregate {
    count:   i64,
    avg:     f64,
    q:       f64,
}

struct IncrementalStdDev{}

// todo
impl IncrementalAggr for IncrementalStdDev {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        values.iter().for_each(|v| push_sample(&mut iac.ts.values[0], *v));
    }
    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
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
    fn keep_original() -> bool {
        false
    }
}


fn push_sample(sv: &mut StdDevAggregate, value: f64) {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
    sv.count += 1;
    let avg = sv.avg + (value - sv.avg) / sv.count;
    sv.q += (value - sv.avg) * (value - avg);
    sv.avg = avg
}

fn appendSeriesForFlush(ctx: flushCtx) {
    m.Range(func(k, v interface{}) bool {
        let stddev = (sv.q / sv.count).sqrt();
        return true
    })
}