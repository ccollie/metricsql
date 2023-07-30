use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use rayon::prelude::*;
use xxhash_rust::xxh3::Xxh3;

use metricsql::common::GroupModifier;

use crate::{MetricName, Tag, Timeseries};

/// The minimum threshold of timeseries tags to process in parallel when computing signatures.
pub(crate) const SIGNATURE_PARALLELIZATION_THRESHOLD: usize = 2;

pub type TimeseriesHashMap = HashMap<Signature, Vec<Timeseries>>;

#[derive(Debug, Default, Clone, PartialEq, Eq, Copy, Ord, PartialOrd)]
pub struct Signature(u64);

/// implement hash which returns the value of the inner u64
impl Hash for Signature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
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

    pub(crate) fn from_tag_iter<'a>(iter: impl Iterator<Item = &'a Tag>) -> Self {
        let mut hasher = Xxh3::new();
        Self::update_from_iter(&mut hasher, iter);
        let sig = hasher.digest();
        Signature(sig)
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

pub fn group_series_by_modifier(
    series: &mut Vec<Timeseries>,
    modifier: &Option<GroupModifier>,
) -> TimeseriesHashMap {
    let mut m: TimeseriesHashMap = HashMap::with_capacity(series.len());

    if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        let sigs: Vec<Signature> = series
            .par_iter()
            .map_with(modifier, |modifier, timeseries| {
                timeseries.metric_name.signature_by_group_modifier(modifier)
            })
            .collect();

        for (ts, sig) in series.into_iter().zip(sigs.iter()) {
            m.entry(*sig).or_default().push(std::mem::take(ts));
        }
    } else {
        for ts in series.iter_mut() {
            ts.metric_name.sort_tags();
            let key = ts.metric_name.signature_by_group_modifier(modifier);
            m.entry(key).or_insert(vec![]).push(std::mem::take(ts));
        }
    };

    return m;
}

pub fn group_series_indexes_by_modifier(
    series: &Vec<Timeseries>,
    modifier: &Option<GroupModifier>,
) -> HashMap<Signature, Vec<usize>> {
    let mut m: HashMap<Signature, Vec<usize>> = HashMap::with_capacity(series.len());

    if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        let sigs: Vec<(Signature, usize)> = series
            .par_iter()
            .enumerate()
            .map_with(modifier, |modifier, (index, timeseries)| {
                let sig = timeseries.metric_name.signature_by_group_modifier(modifier);
                (sig, index)
            })
            .collect();

        for (sig, index) in sigs {
            m.entry(sig).or_default().push(index);
        }
    } else {
        for (index, ts) in series.iter().enumerate() {
            let key = ts.metric_name.signature_by_group_modifier(modifier);
            m.entry(key).or_default().push(index);
        }
    };

    return m;
}

pub fn get_signatures_set_by_modifier(
    series: &[Timeseries],
    modifier: &Option<GroupModifier>,
) -> HashSet<Signature> {
    let res: HashSet<Signature> = if series.len() >= SIGNATURE_PARALLELIZATION_THRESHOLD {
        series
            .par_iter()
            .map_with(modifier, |modifier, timeseries| {
                timeseries.metric_name.signature_by_group_modifier(modifier)
            })
            .collect()
    } else {
        series
            .iter()
            .map(|timeseries| timeseries.metric_name.signature_by_group_modifier(modifier))
            .collect()
    };
    res
}
