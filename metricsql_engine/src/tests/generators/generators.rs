use std::ops::Range;

use rand::distributions::Uniform;
use rand::prelude::StdRng;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;

use crate::tests::generators::create_rng;
use crate::tests::generators::mackey_glass::mackey_glass;

pub struct RandomGenerator {
    rng: StdRng,
    range: Range<f64>,
}

impl RandomGenerator {
    pub fn new(seed: Option<u64>, range: &Range<f64>) -> Result<Self, String> {
        let rng = create_rng(seed)?;
        Ok(Self {
            rng,
            range: range.clone(),
        })
    }
}

impl Iterator for RandomGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.rng.gen_range(self.range.start..self.range.end))
    }
}

fn get_value_in_range(rng: &mut StdRng, r: &Range<f64>) -> f64 {
    r.start + (r.end - r.start) * rng.gen::<f64>()
}

pub struct StdNormalGenerator {
    rng: StdRng,
    range: Range<f64>,
    last_value: f64,
}

impl StdNormalGenerator {
    pub fn new(seed: Option<u64>, range: &Range<f64>) -> Result<Self, String> {
        let rng = create_rng(seed)?;
        Ok(Self {
            rng,
            range: range.clone(),
            last_value: 0.0,
        })
    }
}

impl Iterator for StdNormalGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let m = self.rng.sample::<f64, _>(StandardNormal);
        let range = self.range.end - self.range.start;
        self.last_value = self.range.start + (range * m);
        Some(self.last_value)
    }
}

pub struct UniformGenerator {
    rng: StdRng,
    uniform: Uniform<f64>,
}

impl UniformGenerator {
    pub fn new(seed: Option<u64>, range: &Range<f64>) -> Result<Self, String> {
        let rng = create_rng(seed)?;
        let uniform = Uniform::new(range.start, range.end);
        Ok(Self { rng, uniform })
    }
}

impl Iterator for UniformGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.uniform.sample(&mut self.rng))
    }
}

pub struct DerivativeGenerator {
    p: f64,
    n: f64,
    rng: StdRng,
    range: Range<f64>,
}

impl DerivativeGenerator {
    pub fn new(seed: Option<u64>, range: &Range<f64>) -> Result<Self, String> {
        let mut rng = create_rng(seed)?;
        let c = get_value_in_range(&mut rng, range);
        let p = c;
        let n = c + get_value_in_range(&mut rng, range);
        Ok(Self {
            p,
            n,
            rng,
            range: range.clone(),
        })
    }
}

impl Iterator for DerivativeGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let v = (self.n - self.p) / 2.0;
        self.p = self.n;
        self.n += v;
        Some(v)
    }
}

pub struct MackeyGlassGenerator {
    buf: Vec<f64>,
    pub tau: usize,
    seed: Option<u64>,
    index: usize,
    pub range: Range<f64>,
}

impl MackeyGlassGenerator {
    pub fn new(tau: usize, seed: Option<u64>, range: &Range<f64>) -> Self {
        Self {
            buf: vec![],
            tau,
            seed,
            index: 0,
            range: range.clone(),
        }
    }
}

impl Default for MackeyGlassGenerator {
    fn default() -> Self {
        let range = Range {
            start: 0.0,
            end: 1.0,
        };
        Self::new(17, None, &range)
    }
}

impl Iterator for MackeyGlassGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buf.len() {
            self.index = 0;
            self.buf = mackey_glass(20, self.tau, self.seed);
        }
        let v = self.buf[self.index];
        self.index += 1;
        return Some(self.range.start + (self.range.end - self.range.start) * v);
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    fn validate_range(iter: impl Iterator<Item = f64>, r: &Range<f64>) {
        let values = iter.take(1000).collect::<Vec<f64>>();
        for v in values {
            assert!(
                v >= r.start && v < r.end,
                "value {} not in range {:?}",
                v,
                r
            );
        }
    }

    #[test]
    fn test_random_generator() {
        let r = Range {
            start: 0.0,
            end: 1.0,
        };
        let mut iter = super::RandomGenerator::new(None, &r).unwrap();
        validate_range(iter, &r);
    }

    #[test]
    fn test_std_normal_generator() {
        let r = Range {
            start: 1.0,
            end: 99.0,
        };
        let iter = super::StdNormalGenerator::new(None, &r).unwrap();
        validate_range(iter, &r);
    }

    #[test]
    fn test_mackey_glass_generator() {
        let r = Range {
            start: 1.0,
            end: 99.0,
        };
        let iter = super::MackeyGlassGenerator::new(17, None, &r);
        validate_range(iter, &r);
    }
}
