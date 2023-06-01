
// Calculates output=stddev, e.g. the average value over input samples.
pub(crate) struct IncrementalStdDev{
    count:   i64,
    avg:     f64,
    q:       f64,
}

impl IncrementalStdDev {
    fn new() -> Self {
        Self {
            count: 0,
            avg: 0.0,
            q: 0.0,
        }
    }
}

impl Default for IncrementalStdDev {
    fn default() -> Self {
        Self::new()
    }
}

// todo
impl IncrementalAggr for IncrementalStdDev {
    fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        values.iter().for_each(|v| push_sample(&mut iac.ts.values[0], *v));
    }

    fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        let src_values = &src.ts.values;
        let src_counts = &src.values;

        let iter = izip!(
            src.values.iter(),
            dst.values.iter_mut(),
            dst.ts.values.iter_mut()
        );
        for (counts, dst, ts) in iter {
            if *counts == 0.0 {
                continue;
            }
            if *dst == 0.0 {
                *dst = *ts;
                continue;
            }
            if *ts > *dst {
                *dst = *ts;
            }
        }

        // todo: use zip
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

    fn push_sample(&mut self, value: f64) {
        // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
        self.count += 1;
        let avg = sv.avg + (value - self.avg) / sv.count;
        self.q += (value - self.avg) * (value - avg);
        self.avg = avg
    }

    fn keep_original() -> bool {
        false
    }
}


fn push_sample(sv: &mut IncrementalStdDev, value: f64) {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
    sv.count += 1;
    let avg = sv.avg + (value - sv.avg) / sv.count;
    sv.q += (value - sv.avg) * (value - avg);
    sv.avg = avg
}
