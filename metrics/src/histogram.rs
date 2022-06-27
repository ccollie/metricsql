use std::rand;
use std::rand::Rng;
use once_cell::sync::Lazy;

// Rust port of https://github.com/valyala/histogram
// Package histogram provides building blocks for fast histograms.
const infNeg: f64 = f64::NEG_INFINITY;
const infPos: f64 = f64::INFINITY;
const nan: f64 = f64::NAN;

const maxSamples: usize = 1000;

// Fast is a fast histogram.
//
// It cannot be used from concurrently running goroutines without
// external synchronization.
pub struct Fast {
    max: f64,
    min: f64,
    count: u64,
    a: Vec<f64>,
    tmp: Vec<f64>,
    rng: fastrand::Rng
}

impl Fast {
    // returns new fast histogram.
    pub fn new() -> Self {
        let rng = fastrand::Rng::new();
        rng.seed(1);

        Fast {
            max: f64::NEG_INFINITY,
            min: f64::INFINITY,
            count: 0,
            a: vec![],
            tmp: vec![],
            rng
        }
        // reset
    }

    pub fn with_capacity(n: usize) -> Self {
        let rng = fastrand::Rng::new();
        rng.seed(1);

        Fast {
            max: f64::NEG_INFINITY,
            min: f64::INFINITY,
            count: 0,
            a: Vec::with_capacity(n),
            tmp: Vec::with_capacity(n),
            rng
        }
        // reset
    }

    // Reset resets the histogram.
    pub fn reset(&mut self) {
        self.max = infNeg;
        self.min = infPos;
        self.count = 0;
        if self.a.len() > 0 {
            self.a.clear();
            self.tmp.clear();
        }
        // Reset rng state in order to get repeatable results
        // for the same sequence of values passed to Fast.Update.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1612
        self.rng.seed(1)
    }

    // Update updates the f with v.
    pub fn update(&mut self, v: f64) {
        if v > self.max {
            self.max = v;
        }
        if v < self.min {
            self.min = v
        }

        self.count = self.count + 1;
        if self.count < maxSamples {
            self.a.push(v);
            return;
        }

        let n = self.rng.usize(self.count);
        if n < self.a.len() {
            self.a[n] = v;
        }
    }

    // Quantile returns the quantile value for the given phi.
    pub fn quantile(mut self, phi: f64) -> f64 {
        self.tmp = self.a.iter().sort();
        vec_copy(&self.a, &mut self.tmp);
        self.tmp.sort();
        self.quantile_internal(phi)
    }

    // Quantiles appends quantile values to dst for the given phis.
    pub fn quantiles(mut self, mut dst: &[f64], phis: &[f64]) {
        self.tmp = self.a.iter().sort();
        vec_copy(&self.a, &mut self.tmp);
        self.tmp.sort();
        for phi in phis.iter() {
            dst.push( self.quantile_internal(*phi));
        }
    }

    fn quantile_internal(&self, phi: f64) -> f64 {
        let len = self.tmp.len();
        if len == 0 || phi.is_nan() {
            return nan
        }
        if phi <= 0 as f64 {
            return f.min
        }
        if phi >= 1 as f64 {
            return f.max
        }
        let mut idx = (phi * (len-1) + 0.5);
        if idx >= len {
            idx = len - 1;
        }
        return self.tmp[idx]
    }
}

#[inline]
fn vec_copy<T>(src: &Vec<T>, dst: &mut Vec<T>) {
    dst.resize(src.len(), 0);
    dst.copy_from_slice(src.as_slice());
}

// GetFast returns a histogram from a pool.
func GetFast() *Fast {
v := fastPool.Get()
if v == nil {
return NewFast()
}
return v.(*Fast)
}

// PutFast puts hf to the pool.
//
// hf cannot be used after this call.
func PutFast(f *Fast) {
f.Reset()
fastPool.Put(f)
}

var fastPool sync.Pool