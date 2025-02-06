use super::index_key::{
    format_key_for_label_value, 
    format_key_for_metric_name, 
    get_key_for_label_prefix, 
    IndexKey
};
use super::provider_error::{ProviderError, ProviderResult};
use crate::types::{MetricName, METRIC_NAME_LABEL};
use metricsql_common::hash::{FastHashMap, FastHashSet, HashSetExt};
use metricsql_parser::label::{Label, MatchOp, Matcher, Matchers};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::hash_map::Entry;
use std::ops::ControlFlow;
use ahash::HashMapExt;

pub type SeriesRef = u64;
pub(crate) use croaring::Bitmap64 as IdBitmap;
use enquote::enquote;
use get_size::GetSize;
use super::posting_stats::{PostingStat, PostingsStats, StatsMaxHeap};

// label
// label=value
pub type ARTBitmap = blart::TreeMap<IndexKey, IdBitmap>;


#[derive(Clone, Default, Debug)]
pub struct MemoryPostings {
    all_postings: IdBitmap,
    /// Map from label name and (label name,  label value) to set of timeseries ids.
    label_index: ARTBitmap,
    label_count: usize,
}

impl MemoryPostings {
    pub fn new() -> Self {
        MemoryPostings {
            all_postings: Default::default(),
            label_index: Default::default(),
            label_count: 0
        }
    }

    pub fn clear(&mut self) {
        self.all_postings.clear();
        self.label_index.clear();
        self.label_count = 0;
    }

    pub fn label_index(&mut self) -> &ARTBitmap {
        &self.label_index
    }

    pub fn label_count(&self) -> usize {
        self.label_count
    }

    pub fn add_posting(&mut self, id: SeriesRef, metric_name: &MetricName) {
        debug_assert!(id != 0);
        
        if !metric_name.measurement.is_empty() {
            self.add_posting_for_label_value(id, METRIC_NAME_LABEL, &metric_name.measurement);
        }

        for Label { name, value } in metric_name.labels.iter() {
            self.add_posting_for_label_value(id, name, value);
        }
        
        self.all_postings.add(id)
    }

    pub fn reindex_posting(&mut self, id: SeriesRef, metric_name: &MetricName) {
        self.remove_posting_by_id_and_labels(id, &metric_name.measurement, &metric_name.labels);
        self.add_posting(id, &metric_name);
    }
    
    pub fn has_posting(&self, id: SeriesRef) -> bool {
        self.all_postings.contains(id)
    }

    pub fn remove_posting(&mut self, id: SeriesRef, metric_name: &MetricName) {
        self.remove_posting_by_id_and_labels(id, &metric_name.measurement, &metric_name.labels)
    }

    fn remove_posting_by_id_and_labels(
        &mut self,
        id: SeriesRef,
        metric_name: &str,
        labels: &[Label],
    ) {
        self.all_postings.remove(id);
        // should never happen, but just in case
        if metric_name.is_empty() && labels.is_empty() {
            return
        }

        if !metric_name.is_empty() {
            self.remove_posting_for_label_value(METRIC_NAME_LABEL, metric_name, id);
        }

        for Label { name, value } in labels.iter() {
            self.remove_posting_for_label_value(name, value, id);
        }
    }

    pub fn has_label(&self, label: &str) -> bool {
        let prefix = get_key_for_label_prefix(label);
        self.label_index.prefix(prefix.as_bytes()).next().is_some()
    }

    fn add_posting_for_label_value(
        &mut self,
        ts_id: SeriesRef,
        label: &str,
        value: &str,
    ) -> bool {
        let key = IndexKey::for_label_value(label, value);
        let result = if let Some(bmp) = self.label_index.get_mut(&key) {
            bmp.add(ts_id);
            false
        } else {
            let mut bmp = IdBitmap::new();
            bmp.add(ts_id);
            // TODO: possibly return Result, though if this fails, it's a bug
            match self
                .label_index
                .try_insert(key, bmp)
                .expect("BUG in posting insert. Key is prefix of another key")
            {
                None => {
                    self.label_count += 1;
                    true
                }
                _ => false,
            }
        };
        result
    }

    fn index_posting_by_label(&mut self, ts_id: SeriesRef, label: &str, value: &str) -> ProviderResult<()> {
        if !self.all_postings.contains(ts_id) {
            return Err(ProviderError::MissingPostingInIndex)
        }
        self.add_posting_for_label_value(ts_id, label, value);
        Ok(())
    }

    fn remove_posting_for_label_value(&mut self, label: &str, value: &str, ts_id: SeriesRef) {
        let key = IndexKey::for_label_value(label, value);
        if let Some(bmp) = self.label_index.get_mut(&key) {
            bmp.remove(ts_id);
            if bmp.is_empty() {
                self.label_index.remove(&key);
                if !self.has_label(label) {
                    self.label_count -= 1;
                }
            }
        }
    }

    /// Returns a list of all series matching `matchers`
    pub fn postings_for_matchers(&self, matchers: &Matchers) -> ProviderResult<Cow<IdBitmap>> {
        if matchers.is_empty() {
            // ??
            return Ok(Cow::Borrowed(&self.all_postings));
        }
        if !matchers.matchers.is_empty() {
            return self.postings_for_matchers_slice(&matchers.matchers);
        }

        if !matchers.or_matchers.is_empty() {
            let parallelize = should_parallelize_matchers(matchers);
            if parallelize {
                run_or_matchers_parallel(self, &matchers.or_matchers)
            } else {
                let mut acc = IdBitmap::new();
                for filter in matchers.or_matchers.iter() {
                    let postings = self.postings_for_matchers_slice(filter)?;
                    acc.or_inplace(&postings);
                }
                Ok(Cow::Owned(acc))
            }
        } else {
            Ok(Cow::Owned(IdBitmap::new()))
        }
    }

    /// `postings_for_matchers` assembles a single postings iterator against the index
    /// based on the given matchers. The resulting postings are not ordered by series. 
    fn postings_for_matchers_slice(&self, ms: &[Matcher]) -> ProviderResult<Cow<IdBitmap>> {
        if ms.len() == 1 {
            let m = &ms[0];
            if m.label.is_empty() && m.label.is_empty() {
                return Ok(Cow::Borrowed(&self.all_postings));
            }
        }

        let mut sorted_matchers: SmallVec<(&Matcher, bool, bool), 4> = SmallVec::new();
        let mut not_its = IdBitmap::new();

        let mut has_subtracting_matchers = false;
        let mut has_intersecting_matchers = false;

        // See which label must be non-empty.
        // Optimization for case like {l=~".", l!="1"}.
        let mut label_must_be_set: FastHashSet<String> = FastHashSet::with_capacity(ms.len());
        for m in ms {
            let matches_empty = m.matches("");
            if !matches_empty {
                label_must_be_set.insert(m.label.clone());
            }
            let is_subtracting = is_subtracting_matcher(m, &label_must_be_set);

            has_subtracting_matchers |= is_subtracting;
            has_intersecting_matchers |= !is_subtracting;

            sorted_matchers.push((m, matches_empty, is_subtracting))
        }

        let mut its = if has_subtracting_matchers && !has_intersecting_matchers {
            // If there's nothing to subtract from, add in everything and remove the not_its later.
            // We prefer to get all_postings so that the base of subtraction (i.e. all_postings)
            // doesn't include series that may be added to the index reader during this function call.
            self.all_postings.clone()
        } else {
            IdBitmap::new()
        };

        // Sort matchers to have the intersecting matchers first.
        // This way the base for subtraction is smaller and there is no chance that the set we subtract
        // from contains postings of series that didn't exist when we constructed the set we subtract by.
        sorted_matchers.sort_by(|i, j| -> Ordering {
            let is_i_subtracting = i.2;
            let is_j_subtracting = j.2;
            if !is_i_subtracting && is_j_subtracting {
                return Ordering::Less;
            }
            // sort by match cost
            let cost_i = i.0.cost();
            let cost_j = j.0.cost();
            cost_i.cmp(&cost_j)
        });

        for (m, matches_empty, _is_subtracting) in sorted_matchers {
            let value = &m.value;
            let name = &m.label;
            let typ = m.op;

            if name.is_empty() && value.is_empty() {
                // If the matchers for a label name selects an empty value, it selects all
                // the series which don't have the label name set too. See:
                //
                return Err(ProviderError::MissingMatcher);
                // todo: better error
            }

            if typ == MatchOp::RegexEqual && value == ".*" {
                // .* regexp matches any string: do nothing.
                continue;
            }

            if typ == MatchOp::RegexNotEqual && value == ".*" {
                return Ok(Cow::Owned(IdBitmap::default()));
            }

            if typ == MatchOp::RegexEqual && value == ".+" {
                // .+ regexp matches any non-empty string: get postings for all label values.
                let it = self.postings_for_all_label_values(&m.label);
                if it.is_empty() {
                    return Ok(Cow::Owned(it));
                }
                its.or_inplace(&it);
            } else if typ == MatchOp::RegexNotEqual && value == ".+" {
                // .+ regexp matches any non-empty string: get postings for all label values and remove them.
                not_its |= self.postings_for_all_label_values(name);
                // its = append(not_its, it)
            } else if label_must_be_set.contains(name) {
                // If this matcher must be non-empty, we can be smarter.
                let is_not = typ == MatchOp::NotEqual || m.op == MatchOp::RegexNotEqual;

                if is_not {
                    // a failure here should probably panic
                    let inverse = m.inverse().map_err(|_| {
                        ProviderError::InvalidMatcher(m.to_string())
                    })?;

                    // If the label can't be empty and is a Not, then subtract it out at the end.
                    if matches_empty {
                        // l!="foo"
                        // If the label can't be empty and is a Not and the inner matcher
                        // doesn't match empty, then subtract it out at the end.
                        let it = self.postings_for_matcher(&inverse);
                        not_its.or_inplace(&it);
                    } else {
                        // l!=""
                        // If the label can't be empty and is a Not, but the inner matcher can
                        // be empty we need to use inverse_postings_for_matcher.
                        let it = inverse_postings_for_matcher(self, &inverse);
                        if it.is_empty() {
                            return Ok(it);
                        }
                        intersect(&mut its, &it);
                    }
                } else {
                    // l="a", l=~"a|b", l=~"a.b", etc.
                    // Non-Not matcher, use normal `postings_for_matcher`.
                    let it = self.postings_for_matcher(m);
                    if it.is_empty() {
                        return Ok(it);
                    }
                    intersect(&mut its, &it);
                }
            } else {
                // l=""
                // If the matchers for a label name selects an empty value, it selects all
                // the series which don't have the label name set too. See:
                // https://github.com/prometheus/prometheus/issues/3575 and
                // https://github.com/prometheus/prometheus/pull/3578#issuecomment-351653555
                let it = inverse_postings_for_matcher(self, m);
                not_its.or_inplace(&it);
            }
        }

        its -= &not_its;
        Ok(Cow::Owned(its))
    }

    pub fn postings_for_all_label_values(&self, label_name: &str) -> IdBitmap {
        let prefix = get_key_for_label_prefix(label_name);
        let mut result = IdBitmap::new();
        for (_, map) in self.label_index.prefix(prefix.as_bytes()) {
            result.or_inplace(map);
        }
        result
    }

    pub fn all_postings(&self) -> &IdBitmap {
        &self.all_postings
    }
    
    
    /// `postings` returns the postings list iterator for the label pairs.
    /// The postings here contain the ids to the series inside the index.
    pub fn postings(&self, name: &str, values: &[String]) -> IdBitmap {
        let mut result = IdBitmap::new();
        for value in values {
            let key = IndexKey::for_label_value(name, value);
            if let Some(bmp) = self.label_index.get(&key) {
                result.or_inplace(bmp);
            }
        }
        result
    }

    pub fn postings_for_label_value<'a>(&'a self, name: &str, value: &str) -> Cow<'a, IdBitmap> {
        let key = IndexKey::for_label_value(name, value);
        if let Some(bmp) = self.label_index.get(&key) {
            Cow::Borrowed(bmp)
        } else {
            Cow::Owned(IdBitmap::default())
        }
    }

    /// `postings_for_label_matching` returns postings having a label with the given name and a value
    /// for which match returns true. If no postings are found having at least one matching label,
    /// an empty bitmap is returned.
    pub fn postings_for_label_matching(&self, name: &str, match_fn: fn(&str) -> bool) -> IdBitmap {
        let prefix = get_key_for_label_prefix(name);
        let start_pos = prefix.len();
        let mut result = IdBitmap::new();
        for (key, map) in self.label_index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if match_fn(value) {
                result.or_inplace(map);
            }
        }
        result
    }

    fn postings_for_matcher_internal(&self, matcher: &Matcher, inverse: bool) -> IdBitmap {
        let mut result = IdBitmap::new();
        let prefix = get_key_for_label_prefix(&matcher.label);
        let start_pos = prefix.len();
        for (key, map) in self.label_index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            let mut matched = matcher.matches(value);
            if inverse {
                matched = !matched;
            }
            if matched {
                result.or_inplace(map);
            }
        }
        result
    }

    pub fn postings_for_matcher(&self, m: &Matcher) -> Cow<IdBitmap> {
        if m.label.is_empty() && m.value.is_empty() {
            return Cow::Borrowed(&self.all_postings);
        }
        if m.op == MatchOp::Equal {
            return self.postings_for_label_value(&m.label, &m.value);
        }
        if m.op == MatchOp::RegexEqual {
            let set_matches = m.set_matches();
            if let Some(matches) = set_matches {
                if matches.len() == 1 {
                    return self.postings_for_label_value(&m.label, &matches[0]);
                }
                return Cow::Owned(self.postings(&m.label, &matches));
            } else if let Some(prefix) = m.prefix() {
                // todo: refactor into a method
                // todo: possible optimization - if there's only one entry, we can return a reference
                let mut result = IdBitmap::new();
                let key_prefix = IndexKey::for_label_value(&m.label, prefix);
                let start_pos = key_prefix.len();
                for (key, map) in self.label_index.prefix(&key_prefix) {
                    let value = key.sub_string(start_pos);
                    if m.matches(value) {
                        result.or_inplace(map);
                    }
                }
                return Cow::Owned(result);
            }
        }

        Cow::Owned(self.postings_for_matcher_internal(m, false))
    }

    pub fn label_values_for_matcher_slice(
        &self,
        name: &str,
        matchers: &[Matcher],
    ) -> ProviderResult<Vec<String>> {
        let mut all_values = self.label_values(name);

        if all_values.is_empty() {
            return Ok(all_values);
        }

        // If we have a matcher for the label name, we can filter out values that don't match
        // before we fetch postings. This is especially useful for labels with many values.
        // e.g. __name__ with a selector like {__name__="xyz"}
        let has_matchers_for_other_labels = matchers.iter().any(|m| m.label != name);
        all_values.retain(|v| matchers.iter().all(|m| m.label != name || m.matches(v)));

        if all_values.is_empty() {
            return Ok(all_values);
        }

        // If we don't have any matchers for other labels, then we're done.
        if !has_matchers_for_other_labels {
            return Ok(all_values);
        }

        let p = self.postings_for_matchers_slice(matchers)?;

        all_values.retain(|v| {
            let postings = self.postings_for_label_value(name, v);
            postings.intersect(&p)
        });

        Ok(all_values)
    }

    /// label_names returns all the unique label names.
    pub fn label_names(&self) -> Vec<String> {
        let mut set: BTreeSet<String> = BTreeSet::new();
        for key in self.label_index.keys() {
            if let Some((label, _)) = key.split() {
                if !label.is_empty() {
                    set.insert(label.to_string());
                }
            }
        }
        set.into_iter().collect()
    }


    pub fn label_values(&self, name: &str) -> Vec<String> {
        let mut values = Vec::new();
        self.process_label_values(
            name,
            &mut values,
            |_| true,
            |values, value, _| {
                values.push(value.to_string());
                ControlFlow::<Option<()>>::Continue(())
            },
        );
        values.sort();

        values
    }

    pub fn label_values_for_matchers(
        &self,
        name: &str,
        matchers: &Matchers,
    ) -> ProviderResult<Vec<String>> {
        if !matchers.matchers.is_empty() {
            return self.label_values_for_matcher_slice(name, &matchers.matchers);
        }

        if !matchers.or_matchers.is_empty() {
            let mut set = BTreeSet::new();
            for filter in matchers.or_matchers.iter() {
                let result = self.label_values_for_matcher_slice(name, filter)?;
                set.extend(result);
            }
            let result = set.into_iter().collect();
            Ok(result)
        } else {
            Ok(vec![])
        }
    }

    /// This exists primarily to ensure that we disallow duplicate metric names
    pub fn posting_by_name_and_labels(
        &self,
        metric: &str,
        labels: &[Label],
    ) -> ProviderResult<Option<SeriesRef>> {
        let mut key: String = String::new();
        format_key_for_metric_name(&mut key, metric);
        if let Some(measurement_bmp) = self.label_index.get(key.as_bytes()) {
            let mut first = true;
            let mut acc = IdBitmap::new();
            for label in labels.iter() {
                format_key_for_label_value(&mut key, &label.name, &label.value);
                if let Some(bmp) = self.label_index.get(key.as_bytes()) {
                    if bmp.is_empty() {
                        break;
                    }
                    if first {
                        acc = measurement_bmp.and(bmp);
                        first = false;
                    } else {
                        acc.and_inplace(bmp);
                    }
                }
            }
            match acc.cardinality() {
                0 => Ok(None),
                1 => Ok(acc.iter().next()),
                _ => {
                    let metric_name = format_prometheus_metric_name(metric, labels);
                    Err(ProviderError::DuplicatePostingInIndex(metric_name))
                }
            }
        } else {
            Ok(None)
        }
    }

    pub fn posting_for_metric(&self, metric: &MetricName) -> ProviderResult<Option<SeriesRef>> {
        self.posting_by_name_and_labels(&metric.measurement, &metric.labels)
    }

    pub fn process_label_values<T, CONTEXT, F, PRED>(
        &self,
        label: &str,
        ctx: &mut CONTEXT,
        predicate: PRED,
        f: F,
    ) -> Option<T>
    where
        F: Fn(&mut CONTEXT, &str, &IdBitmap) -> ControlFlow<Option<T>>,
        PRED: Fn(&str) -> bool,
    {
        let prefix = get_key_for_label_prefix(label);
        let start_pos = prefix.len();
        for (key, map) in self.label_index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if predicate(value) {
                match f(ctx, value, map) {
                    ControlFlow::Break(v) => {
                        return v;
                    }
                    ControlFlow::Continue(_) => continue,
                }
            }
        }
        None
    }

    pub fn stats(&self, label: &str, limit: usize) -> PostingsStats {
        
        #[derive(Clone, Copy)]
        struct SizeAccumulator {
            size: usize,
            count: u64,
        }

        let mut count_map: FastHashMap<&str, SizeAccumulator> = FastHashMap::with_capacity(self.label_count);
        let mut metrics = StatsMaxHeap::new(limit);
        let mut labels = StatsMaxHeap::new(limit);
        let mut label_value_length = StatsMaxHeap::new(limit);
        let mut label_value_pairs = StatsMaxHeap::new(limit);
        let mut num_label_pairs = 0;
        
        for (key, bitmap) in self.label_index.iter() {
            let count = bitmap.cardinality();
            if let Some((name, value)) = key.split() {
                let size = key.get_size() + get_bitmap_size(bitmap);
                match count_map.entry(name) {
                    Entry::Occupied(mut entry) => {
                        let acc = entry.get_mut();
                        acc.count += count;
                        acc.size += size;
                    }
                    Entry::Vacant(entry) => {
                        let acc = SizeAccumulator {
                            size,
                            count,
                        };
                        entry.insert(acc);
                    }
                }
                
                label_value_pairs.push(PostingStat { name: format!("{}={}", name, value), count });
                num_label_pairs += 1;
                
                if label == name {
                    metrics.push(PostingStat { name: name.to_string(), count });                    
                }
            }
        }
        
        for (name, v) in count_map {
            labels.push(PostingStat { name: name.to_string(), count: v.count });
            label_value_length.push(PostingStat { name: name.to_string(), count: v.size as u64 });
        }

        PostingsStats {
            cardinality_metrics_stats: metrics.into_vec(),
            cardinality_label_stats: labels.into_vec(),
            label_value_stats: label_value_length.into_vec(),
            label_value_pairs_stats: label_value_pairs.into_vec(),
            num_label_pairs,
        }
    }
}

#[inline]
fn intersect(dest: &mut IdBitmap, other: &IdBitmap) {
    if dest.is_empty() {
        dest.or_inplace(other);
    } else {
        dest.and_inplace(other);
    }
}

fn is_subtracting_matcher(m: &Matcher, label_must_be_set: &FastHashSet<String>) -> bool {
    if !label_must_be_set.contains(&m.label) {
        return true;
    }
    matches!(m.op, MatchOp::NotEqual | MatchOp::RegexNotEqual if m.matches(""))
}

fn inverse_postings_for_matcher<'a>(postings: &'a MemoryPostings, m: &Matcher) -> Cow<'a, IdBitmap> {
    // Fast-path for RegexNotEqual matching.
    // Inverse of a RegexNotEqual is RegexpEqual (double negation).
    // Fast-path for set matching.
    if m.op == MatchOp::RegexNotEqual {
        if let Some(matches) = m.set_matches() {
            return Cow::Owned(postings.postings(&m.label, &matches));
        }
    }

    // Fast-path for NotEqual matching.
    // Inverse of a NotEqual is Equal (double negation).
    if m.op == MatchOp::NotEqual {
        return postings.postings_for_label_value(&m.label, &m.value);
    }

    // If the matcher being inverted is =~"" or ="", we just want all the values.
    if m.value.is_empty() && (m.op == MatchOp::RegexEqual || m.op == MatchOp::Equal) {
        return Cow::Owned(postings.postings_for_all_label_values(&m.label));
    }

    Cow::Owned(postings.postings_for_matcher_internal(m, true))
}

// Placeholder for more reasonable heuristics
// e.g. if we have a filter that matches all postings, we should not parallelize
// and instead rely on set operations to optimize the query at each iteration
fn should_parallelize_matchers(matchers: &Matchers) -> bool {
    if !matchers.matchers.is_empty() {
        return matchers.matchers.len() > 1;
    }
    if !matchers.or_matchers.is_empty() {
        return matchers.or_matchers.iter().any(|m| m.len() > 3);
    }
    false
}

fn run_or_matchers_parallel<'a>(
    label_index: &'a MemoryPostings,
    matchers: &[Vec<Matcher>],
) -> ProviderResult<Cow<'a, IdBitmap>> {
    let mut scope = chili::Scope::global();

    match matchers {
        [] => Ok(Cow::Owned(IdBitmap::new())),
        [matchers] => label_index.postings_for_matchers_slice(matchers),
        [m1, m2] => {
            let (r1, r2) = scope.join(
                |_| label_index.postings_for_matchers_slice(m1),
                |_| label_index.postings_for_matchers_slice(m2),
            );
            let mut r1 = r1?.into_owned();
            let r2 = r2?;
            r1.or_inplace(&r2);
            Ok(Cow::Owned(r1))
        }
        [m1, m2, m3] => {
            let (x, (y, z)) = scope.join(
                |_| label_index.postings_for_matchers_slice(m1),
                |s2| {
                    s2.join(
                        |_| label_index.postings_for_matchers_slice(m2),
                        |_| label_index.postings_for_matchers_slice(m3),
                    )
                },
            );
            let mut x = x?.into_owned();
            let y = y?;
            let z = z?;
            x.or_inplace(&y);
            x.or_inplace(&z);
            Ok(Cow::Owned(x))
        }
        [m1, m2, m3, m4] => {
            let ((w, x), (y, z)) = scope.join(
                |s1| s1.join(
                    |_| label_index.postings_for_matchers_slice(m1),
                    |_| label_index.postings_for_matchers_slice(m2),
                ),
                |s2| { s2.join(
                    |_| label_index.postings_for_matchers_slice(m3),
                    |_| label_index.postings_for_matchers_slice(m4),
                    )
                },
            );
            let mut w = w?.into_owned();
            let x = x?;
            let y = y?;
            let z = z?;
            w.or_inplace(&x);
            w.or_inplace(&y);
            w.or_inplace(&z);
            Ok(Cow::Owned(w))
        }
        _ => {
            let mid = matchers.len() / 2;
            let (left, right) = matchers.split_at(mid);
            let (left_results, right_results) = scope.join(
                |_| run_or_matchers_parallel(label_index, left),
                |_| run_or_matchers_parallel(label_index, right),
            );
            let right_results = right_results?;
            let mut left_results = left_results?.into_owned();
            left_results.or_inplace(&right_results);
            Ok(Cow::Owned(left_results))
        }
    }
}

// Note - assumes that labels is sorted
fn format_prometheus_metric_name(name: &str, labels: &[Label]) -> String {
    let size_hint = name.len()
        + labels
        .iter()
        .map(|l| l.name.len() + l.value.len() + 3)
        .sum::<usize>();
    let mut full_name: String = String::with_capacity(size_hint);
    format_prometheus_metric_name_into(&mut full_name, name, labels);
    full_name
}

fn format_prometheus_metric_name_into(full_name: &mut String, name: &str, labels: &[Label]) {
    full_name.push_str(name);
    if !labels.is_empty() {
        full_name.push('{');
        for (i, label) in labels.iter().enumerate() {
            full_name.push_str(&label.name);
            full_name.push_str("=\"");
            // avoid allocation if possible
            if label.value.contains('"') {
                let quoted_value = enquote('\"', &label.value);
                full_name.push_str(&quoted_value);
            } else {
                full_name.push_str(&label.value);
            }
            full_name.push('"');
            if i < labels.len() - 1 {
                full_name.push(',');
            }
        }
        full_name.push('}');
    }
}

fn get_bitmap_size(bmp: &IdBitmap) -> usize {
    bmp.cardinality() as usize * size_of::<SeriesRef>()
}