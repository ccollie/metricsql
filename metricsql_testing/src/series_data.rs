#[derive(Debug, Default, Clone, Copy)]
pub struct Sample {
    /// Time in microseconds
    pub timestamp: i64,
    pub value: f64,
}

impl Sample {
    pub fn new(timestamp: i64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

// Time series measurements.
pub struct SeriesData {
    pub timestamps: Vec<i64>,
    pub values: Vec<f64>,
}

impl SeriesData {
    pub fn new(n: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(n),
            values: Vec::with_capacity(n),
        }
    }
    pub fn new_with_data(timestamps: Vec<i64>, values: Vec<f64>) -> Self {
        Self { timestamps, values }
    }
    pub fn push(&mut self, ts: i64, value: f64) -> usize {
        match self.timestamps.binary_search(&ts) {
            Ok(pos) => {
                self.values[pos] = value;
                0
            }
            Err(idx) => {
                self.timestamps.insert(idx, ts);
                self.values.insert(idx, value);
                1
            }
        }
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
    pub fn first_timestamp(&self) -> i64 {
        self.timestamps[0]
    }
    pub fn last_timestamp(&self) -> i64 {
        self.timestamps[self.timestamps.len() - 1]
    }
    pub fn iter(&self) -> SeriesDataIter {
        SeriesDataIter::new(self)
    }
}

pub struct SeriesDataIter<'a> {
    series: &'a SeriesData,
    idx: usize,
}

impl<'a> SeriesDataIter<'a> {
    pub fn new(series: &'a SeriesData) -> Self {
        Self { series, idx: 0 }
    }
}

impl Iterator for SeriesDataIter<'_> {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.series.timestamps.len() {
            return None;
        }
        let res = Some(Sample {
            timestamp: self.series.timestamps[self.idx],
            value: self.series.values[self.idx],
        });
        self.idx += 1;
        res
    }
}
