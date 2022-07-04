use lib::error::Error;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Timeseries {
    pub metric_name: MetricName,
    pub values: Vec<f64>,
    pub timestamps: RC<Vec<i64>>,
    deny_reuse: bool
}

impl Timeseries {
    pub fn new(timestamps: Vec<i64>, values: Vec<f64>) -> Self {
        Timeseries {
            metric_name: MetricName::new(),
            values,
            timestamps: RC::new(timestamps),
            deny_reuse: true
        }
    }

    pub fn copy_shallow(src: &Timeseries) -> Self {
        let mut ts = Timeseries {
            timestamps: RC::clone(src.timestamps),
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
            deny_reuse: true
        };
        ts
    }
    
    pub fn with_shared_timestamps(timestamps: RC<Vec<i64>>, values: Vec<f64>) -> Self {
        Timeseries {
            metric_name: MetricName::new(),
            values,
            timestamps: RC::clone(timestamps),
            deny_reuse: true
        }
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.timestamps.clear();
        self.metric_name.reset();
    }

    pub fn copy_from_shallow_timestamps(mut self, src: &Timeseries) {
        self.reset();
        self.metric_name.copy_from(&src.metric_name);
        self.values = ts.values.clone();
        self.timestamps = RC::clone(&src.timestamps);
        self.deny_reuse = true;
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

pub(super) fn assert_identical_timestamps(tss: &Vec<Timeseries>, step: i64) -> Result<(), Error> {
    if tss.len() == 0 {
        return Ok(())
    }
    let ts_golden = tss[0];
    if ts_golden.values.len() != ts_golden.timestamps.len() {
        let err = format!("BUG: len(ts_golden.Values) must match len(ts_golden.Timestamps); got {} vs {}", 
                      ts_golden.values.len(), ts_golden.timestamps.len());
        return Err(Error::new(msg));
    }
    if  ts_golden.timestamps.len() > 0 {
        let mut prev_timestamp = ts_golden.timestamps[0];
        for timestamp in ts_golden.timestamps[1..].iter() {
            if timestamp- prev_timestamp != step {
                let msg = format!("BUG: invalid step between timestamps; got {}; want {}; ts_golden.timestamps={}",
                                  timestamp - prev_timestamp, step, ts_golden.timestamps);
                return Err(Error::new(msg));
            }
            prev_timestamp = timestamp
        }
    }
    for ts in tss {
        if ts.values.len() != ts_golden.values.len() {
            let msg = format!("BUG: unexpected len(ts.Values); got {}; want {}; ts.values={}",
                          ts.values.len(),
                          ts_golden.values.len(),
                          ts.values.len());
            return Err(Error::new(msg));
        }
        if  ts.timestamps.len() != ts_golden.timestamps.len() {
            let msg = format!("BUG: unexpected len(ts.Timestamps); got {}; want {}; ts.timestamps={}", 
                          ts.timestamps.len(), ts_golden.timestamps.len(), ts.timestamps);
            return Err(Error::new(msg));
        }
        if ts.timestamps.len() == 0 {
            continue
        }
        if &ts.timestamps[0] == &ts_golden.timestamps[0] {
            // Fast path - shared timestamps.
            continue
        }
        for i in  0 .. ts.timestamps.len() {
            if ts.timestamps[i] != ts_golden.timestamps[i] {
                let msg = format!("BUG: timestamps mismatch at position {}; got {}; want {}; ts.timestamps={}, ts_golden.timestamps={}",
                              i, ts.timestamps[i], ts_golden.timestamps[i], ts.timestamps, ts_golden.timestamps);
                return Err(Error::new(msg));
            }
        }
    }
    Ok(())
}