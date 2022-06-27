
pub(crate) struct TimeseriesMap {
    origin: &Timeseries,
    h: Histogram,
    m: HashMap<&str, &Timeseries>
}

impl TimeseriesMap {
    pub(crate) fn new(
        func_name: &str,
        keep_metric_names: bool,
        shared_timestamps: &[i64],
        mn_src: MetricName) -> Option<TimeseriesMap> {

        let name = func_name.to_lowercase().as_str();
        if name != "histogram_over_time" && name != "histogram_over_time" {
            return None;
        }

        let values: ArrayVec<f64> = ArrayVec::with_capacity(sharedTimestamps.len());
        for i in 0 .. shared_timestamps.len() {
            values.push(nan);
        }

        let origin: timeseries
        origin.metric_name.copy_from(mn_src);
        if !keepMetricNames && !rollupFuncsKeepMetricName[funcName] {
            origin.metric_name.reset_metric_group()
        }
        origin.timestamps = sharedTimestamps;
        origin.values = values;
        let m: HashMap<&str, Timeseries> = HashMap::new();

        TimeseriesMap {
            origin,
            m
        }
    }

    pub(crate) fn getOrCreateTimeseries(&mut self, label_name: &str, label_value: &str) -> &Timeseries {
        let ts = self.get(label_value);
        if ts.is_some() {
            return *ts
        }
        ts = &timeseries{}
        ts.CopyFromShallowTimestamps(self.origin);
        ts.metric_name.remove_tag(label_name);
        ts.metric_name.AddTag(label_name, label_value);
        tsm.m.insert(label_value, ts);
        return ts
    }
}

fn AppendTimeseriesTo(dst: &[Timeseries]) []*timeseries {
for _, ts := range tsm.m {
dst = append(dst, ts)
}
return dst
}

