use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use ahash::AHashMap;
use rayon::prelude::*;
use xxhash_rust::xxh3::Xxh3;

use metricsql_parser::prelude::VectorMatchModifier;

use crate::{MetricName, Tag, Timeseries};

/// The minimum threshold of timeseries tags to process in parallel when computing signatures.
pub(crate) const SIGNATURE_PARALLELIZATION_THRESHOLD: usize = 4;

pub type TimeseriesHashMap = AHashMap<Signature, Vec<Timeseries>>;

#[derive(Debug, Default, Clone, PartialEq, Eq, Copy, Ord, PartialOrd)]
pub struct Signature(u64);

/// implement hash which returns the value of the inner u64
impl Hash for Signature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Signature {
    pub fn new(labels: &MetricName) -> Self {
        let iter = labels.tags.iter();
        Self::with_name_and_labels(&labels.metric_group, iter)
    }

    pub fn with_labels(labels: &MetricName, names: &[String]) -> Signature {
        let iter = labels.with_labels_iter(names);
        Self::with_name_and_labels(&labels.metric_group, iter)
    }

    /// `signature_without_labels` is just as [`signature`], but only for labels not matching `names`.
    pub fn without_labels(labels: &MetricName, exclude_names: &[String]) -> Signature {
        let iter = labels.without_labels_iter(exclude_names);
        Self::with_name_and_labels(&labels.metric_group, iter)
    }

    fn update_from_iter<'a>(hasher: &mut Xxh3, iter: impl Iterator<Item = &'a Tag>) {
        for tag in iter {
            tag.update_hash(hasher);
        }
    }

    pub fn with_name_and_labels<'a>(name: &String, iter: impl Iterator<Item = &'a Tag>) -> Self {
        let mut hasher = Xxh3::new();
        hasher.write(name.as_bytes());
        Self::update_from_iter(&mut hasher, iter);
        let sig = hasher.digest();
        Signature(sig)
    }
}

impl From<Signature> for u64 {
    fn from(sig: Signature) -> Self {
        sig.0
    }
}

pub fn group_series_by_match_modifier(
    series: &mut Vec<Timeseries>,
    modifier: &Option<VectorMatchModifier>,
) -> TimeseriesHashMap {
    let mut m: TimeseriesHashMap = AHashMap::with_capacity(series.len());

    if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        let sigs: Vec<Signature> = series
            .par_iter()
            .map_with(modifier, |modifier, timeseries| {
                timeseries.metric_name.signature_by_match_modifier(modifier)
            })
            .collect();

        for (ts, sig) in series.iter_mut().zip(sigs.iter()) {
            m.entry(*sig).or_default().push(std::mem::take(ts));
        }
    } else {
        for ts in series.iter_mut() {
            ts.metric_name.sort_tags();
            let key = ts.metric_name.signature_by_match_modifier(modifier);
            m.entry(key).or_default().push(std::mem::take(ts));
        }
    };

    m
}

pub fn group_series_indexes_by_match_modifier(
    series: &Vec<Timeseries>,
    modifier: &Option<VectorMatchModifier>,
) -> AHashMap<Signature, Vec<usize>> {
    let mut m: AHashMap<Signature, Vec<usize>> = AHashMap::with_capacity(series.len());

    if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        let sigs: Vec<(Signature, usize)> = series
            .par_iter()
            .enumerate()
            .map_with(modifier, |modifier, (index, ts)| {
                let sig = ts.metric_name.signature_by_match_modifier(modifier);
                (sig, index)
            })
            .collect();

        for (sig, index) in sigs {
            m.entry(sig).or_default().push(index);
        }
    } else {
        for (index, ts) in series.iter().enumerate() {
            let key = ts.metric_name.signature_by_match_modifier(modifier);
            m.entry(key).or_default().push(index);
        }
    };

    m
}

// todo(perf): return AHashset or use nohash_hasher
pub fn get_signatures_set_by_match_modifier(
    series: &[Timeseries],
    modifier: &Option<VectorMatchModifier>,
) -> HashSet<Signature> {
    let res: HashSet<Signature> = if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        series
            .par_iter()
            .map_with(modifier, |modifier, ts| {
                ts.metric_name.signature_by_match_modifier(modifier)
            })
            .collect::<HashSet<_>>()
    } else {
        series
            .iter()
            .map(|ts| ts.metric_name.signature_by_match_modifier(modifier))
            .collect::<HashSet<_>>()
    };
    res
}
