// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


use std::collections::btree_set::BTreeSet;
use std::fmt::{Display, Formatter};

pub(crate) trait Value {
    fn value_type(&self) -> DataType;
}


/// String represents a string value.
pub(crate) struct StringValue {
    pub t: i64,
    pub v: String
}

impl Value for StringValue {
    fn value_type(&self) -> DataType {
        DataType::String
    }
}

impl Display for StringValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "string: {} @[{}]", s.v, self.t);
        Ok(())
    }
}


/// Scalar is a data point that's explicitly not associated with a metric.
pub(crate) struct Scalar {
    pub t: i64,
    pub v: f64
}

impl Value for Scalar {
    fn value_type(&self) -> DataType {
        DataType::Scalar
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "scalar: {} @[{}]", self.v, self.t);
        Ok(())
    }
}


/// Series is a stream of data points belonging to a metric.
pub(crate) struct Series {
    metric: MetricName,
    points: Vec<Point>
}

impl Value for Series {
    fn value_type(&self) -> DataType {
        DataType::InstantVector
    }
}

impl Display for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut vals = Vec::from(self.points.len());
        for v in self.points.iter() {
            vals.push(v.to_string());
        }
        write!(f, format!("{} =>\n{}", self.metric, vals.join("\n")));
        Ok(())
    }
}

/// Point represents a single data point for a given timestamp.
pub(crate) struct Point {
    pub t: i64,
    pub v: f64
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} @[{}]", self.v, self.t);
        Ok(())
    }
}


/// Sample is a single sample belonging to a metric.
pub(crate) struct Sample {
    pub point: Point,
    pub metric: MetricName
}

impl Display for Sample {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} => {}", self.metric, self.t);
        Ok(())
    }
}


/// Vector is basically only an alias for model.Samples, but the
/// contract is that in a Vector, all Samples have the same timestamp.
pub(crate) struct  Vector(Vec<Sample>);

impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut entries = Vec::with_capacity(self.len());
        for s in self.iter() {
            entries.push(s.to_string())
        }
        write!(f, "{}", entries.join("\n"));
        Ok(())
    }
}


// contains_same_label_set checks if a vector has samples with the same labelset
// Such a behavior is semantically undefined
// https://github.com/prometheus/prometheus/issues/4562
impl Vector {
    fn contains_same_label_set(&self) -> bool {
        match self.len() {
            0 | 1 => false,
            2 => return self.0[0].metric.hash() == self.0[1].metric.hash(),
            _ => {
                let l: BTreeSet<u64> = BTreeSet::new();
                for ss in self.iter() {
                    let hash = ss.metric.hash();
                    if l.contains(hash) {
                        return true;
                    }
                    l.insert(hash);
                }
                return false
            }
        }
    }
}


/// Matrix is a slice of Series that implements sort.Interface and
/// has a String method.
pub(crate) struct Matrix(Vec<Series>);

impl Matrix {
    pub fn to_string(&self) -> String {
        // TODO: sort, or can we rely on order from the querier?
        self.0.iter().map(|x| x.to_string()).collect().join("\n")
    }

    /// TotalSamples returns the total number of samples in the series within a matrix.
    pub fn total_samples(&self) -> u64 {
        let mut num_samples = 0;
        for series in self.0.iter() {
            num_samples += series.points.len()
        }
        return num_samples
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    // contains_same_labelset checks if a matrix has samples with the same labelset.
    // Such a behavior is semantically undefined.
    // https://github.com/prometheus/prometheus/issues/4562
    pub fn contains_same_labelset(&self) -> bool {
        let m = self.0;
        match m.len() {
            0 | 1 => false,
            2 => return m[0].metric.hash() == m[1].metric.hash(),
            _ => {
                let l = BTreeSet::new();
                for ss in m.iter() {
                    let hash = ss.metric.hash();
                    if l.contains(hash) {
                        return true;
                    }
                    l.insert(hash);
                }
                return false
            }
        }
    }
}

fn (m Matrix) Less(i, j int) bool { return labels.Compare(m[i].Metric, m[j].Metric) < 0 }


// Result holds the resulting value of an execution or an error
// if any occurred.
pub(crate) struct QueryResult {
    err: Option<RuntimeError>,
    value: Value,
    warnings: storage.Warnings
}

impl QueryResult {
    /// Vector returns a Vector if the result value is one. An error is returned if
    /// the result was an error or the result value is not a Vector.
    pub fn vector(&self) -> RuntimeResult<Vector> {
        if let Some(err) = self.err {
            return Err(err.clone())
        }
        match self.value {
            Vector(v) => Ok(v),
            _ => {
                Err(RuntimeError::from("query result is not a Vector"))
            }
        }
    }

    /// scalar returns a Scalar if the result value is one. An error is returned if
    /// the result was an error or the result value is not a Scalar.
    pub fn scalar(&self) -> RuntimeResult<Scalar> {
        if let Some(err) = self.err {
            return Err(err.clone())
        }
        match self.value {
            Scalar(v) => Ok(v),
            _ => {
                Err(RuntimeError::from("query result is not a Scalar"))
            }
        }
    }

    /// scalar returns a Matrix if the result value is one. An error is returned if
    /// the result was an error or the result value is not a Matrix.
    pub fn matrix(&self) -> RuntimeResult<Matrix> {
        if let Some(err) = self.err {
            return Err(err.clone())
        }
        match self.value {
            Matrix(v) => Ok(v),
            _ => {
                Err(RuntimeError::from("query result is not a Matrix"))
            }
        }
    }
}



fn (r *Result) String() string {
if r.Value == nil {
return ""
}
return r.Value.String()
}

// StorageSeries simulates promql.Series as storage.Series.
pub(crate) struct StorageSeries {
    series: Series
}

impl StorageSeries {
    pub fn new(series: Series) -> Self {
        Self { series }
    }

    pub fn labels(&self) -> Labels {
        self.series.metric
    }
}


struct StorageSeriesIterator {
    points: Vec<Point>,
    curr: usize
}

impl StorageSeriesIterator {
    pub fn new(series: Series) -> Self {
        Self {
            points: series.points.clone(),
            curr: 0 // was - 1
        }
    }

    pub fn next(self) -> Option<Point> {
        if self.curr < self.points.len() {
            let p = self.points[self.curr];
            self.curr += 1;
            return Some(p.clone())
        }
        None
    }

    pub fn seek(&mut self, t: i64) -> bool {
        let mut i = self.curr;
        if i < 0 {
            i = 0
        }
        while i < self.points.len() {
            if self.points[i].t >= t {
                self.curr = i;
                return true
            }
            i += 1
        }
        self.curr = self.points.len() - 1;
        return false
    }

    pub fn at(self) -> Point {
        let p = self.points[self.curr];
        p.clone()
    }
}
