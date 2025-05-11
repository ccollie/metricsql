use super::error::{ProviderError, ProviderResult};
use super::index_key::{
    format_key_for_label_value, format_key_for_metric_name, get_key_for_label_prefix,
    get_key_for_label_value, IndexKey,
};
use crate::types::{MetricName, METRIC_NAME_LABEL};
use ahash::HashMapExt;
use blart::map::Entry as ARTEntry;
use blart::TreeMap;
use croaring::bitmap64::Bitmap64Iterator;
use metricsql_common::hash::FastHashMap;
use metricsql_parser::label::{Label, Matcher, Matchers, NAME_LABEL};
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::BTreeSet;
use std::io;
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::sync::LazyLock;
use async_trait::async_trait;

pub type SeriesRef = u64;
use super::posting_stats::{PostingStat, PostingsStats, StatsMaxHeap};
use crate::provider::index_reader::IndexReader;
use crate::provider::postings::PostingsIterator;
use crate::provider::querier::postings_for_matchers;
use croaring::{Bitmap64, Portable};
use enquote::enquote;
use get_size::GetSize;
use integer_encoding::{VarIntReader, VarIntWriter};

const ALL_POSTINGS_KEY: &str = "$_@LL_P0STINGS_";
static EMPTY_BITMAP: LazyLock<PostingsBitmap> = LazyLock::new(|| PostingsBitmap::new());

pub type PostingsBitmap = Bitmap64;
// label
// label=value
pub type PostingsIndex = TreeMap<IndexKey, PostingsBitmap>;

pub struct BitmapPostings<'a> {
    cursor: Bitmap64Iterator<'a>,
    len: usize,
}

impl<'a> BitmapPostings<'a> {
    pub fn new(bitmap: PostingsBitmap) -> Self {
        let cursor = bitmap.iter();
        let len = bitmap.cardinality() as usize;
        BitmapPostings { cursor, len }
    }
}
impl<'a> Iterator for BitmapPostings<'a> {
    type Item = SeriesRef;

    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a> PostingsIterator for BitmapPostings<'a> {
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Clone, Debug)]
pub struct MemoryPostings {
    all_postings_key: IndexKey,
    /// Map from label name and (label name, label value) to a set of timeseries ids.
    label_index: PostingsIndex,
}

impl Default for MemoryPostings {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPostings {
    pub fn new() -> Self {
        let mut index: PostingsIndex = Default::default();
        let all_postings_key = IndexKey::from(ALL_POSTINGS_KEY);

        index.insert(all_postings_key.clone(), PostingsBitmap::default());
        MemoryPostings {
            all_postings_key,
            label_index: index,
        }
    }

    pub fn clear(&mut self) {
        self.label_index.clear();
    }

    pub fn label_index(&mut self) -> &PostingsIndex {
        &self.label_index
    }

    pub fn add_posting(&mut self, id: SeriesRef, metric_name: &MetricName) {
        debug_assert!(id != 0);

        if !metric_name.measurement.is_empty() {
            self.add_posting_for_label_value(id, METRIC_NAME_LABEL, &metric_name.measurement);
        }

        for Label { name, value } in metric_name.labels.iter() {
            self.add_posting_for_label_value(id, name, value);
        }

        self.add_id_to_all_postings(id);
    }

    pub fn reindex_posting(&mut self, id: SeriesRef, metric_name: &MetricName) {
        self.remove_posting_by_id_and_labels(id, &metric_name.measurement, &metric_name.labels);
        self.add_posting(id, &metric_name);
    }

    pub fn has_posting(&self, id: SeriesRef) -> bool {
        self.all_postings().contains(id)
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
        self.remove_id_from_all_postings(id);

        // should never happen, but just in case
        if metric_name.is_empty() && labels.is_empty() {
            return;
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

    fn add_posting_for_label_value(&mut self, ts_id: SeriesRef, label: &str, value: &str) -> bool {
        let key = IndexKey::for_label_value(label, value);
        match self.label_index.entry(key) {
            ARTEntry::Occupied(mut entry) => {
                entry.get_mut().add(ts_id);
                false
            }
            ARTEntry::Vacant(entry) => {
                let mut bmp = PostingsBitmap::new();
                bmp.add(ts_id);
                entry.insert(bmp);
                true
            }
        }
    }

    fn remove_posting_for_label_value(&mut self, label: &str, value: &str, ts_id: SeriesRef) {
        let key = IndexKey::for_label_value(label, value);
        if let Some(bmp) = self.label_index.get_mut(&key) {
            bmp.remove(ts_id);
            if bmp.is_empty() {
                self.label_index.remove(&key);
            }
        }
    }

    pub fn postings_for_all_label_values(&self, label_name: &str) -> PostingsBitmap {
        let prefix = get_key_for_label_prefix(label_name);
        let mut result = PostingsBitmap::new();
        for (_, map) in self.label_index.prefix(prefix.as_bytes()) {
            result |= map;
        }
        result
    }

    pub fn all_postings(&self) -> &PostingsBitmap {
        self.label_index
            .get(&self.all_postings_key)
            .unwrap_or(&*EMPTY_BITMAP)
    }

    pub fn max_id(&self) -> SeriesRef {
        self.all_postings().maximum().unwrap_or_default()
    }

    fn add_id_to_all_postings(&mut self, id: SeriesRef) {
        if let Some(bitmap) = self.label_index.get_mut(&self.all_postings_key) {
            bitmap.add(id);
        } else {
            let mut bmp = PostingsBitmap::new();
            bmp.add(id);
            self.label_index.insert(self.all_postings_key.clone(), bmp);
        }
    }

    fn remove_id_from_all_postings(&mut self, id: SeriesRef) {
        if let Some(bmp) = self.label_index.get_mut(&self.all_postings_key) {
            bmp.remove(id);
        }
    }

    /// `postings` returns the postings list iterator for the label pairs.
    /// The postings here contain the ids to the series inside the index.
    pub fn postings(&self, name: &str, values: &[String]) -> PostingsBitmap {
        let mut result = PostingsBitmap::new();
        for value in values {
            let key = IndexKey::for_label_value(name, value);
            if let Some(bmp) = self.label_index.get(&key) {
                result |= bmp;
            }
        }
        result
    }

    pub fn postings_for_label_value<'a>(
        &'a self,
        name: &str,
        value: &str,
    ) -> Cow<'a, PostingsBitmap> {
        let key = IndexKey::for_label_value(name, value);
        if let Some(bmp) = self.label_index.get(&key) {
            Cow::Borrowed(bmp)
        } else {
            Cow::Owned(PostingsBitmap::default())
        }
    }

    /// `postings_for_label_matching` returns postings having a label with the given name and a value
    /// for which match returns true. If no postings are found having at least one matching label,
    /// an empty bitmap is returned.
    pub fn postings_for_label_matching(
        &self,
        name: &str,
        match_fn: fn(&str) -> bool,
    ) -> PostingsBitmap {
        let prefix = get_key_for_label_prefix(name);
        let start_pos = prefix.len();
        let mut result = PostingsBitmap::new();
        for (key, map) in self.label_index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if match_fn(value) {
                result |= map;
            }
        }
        result
    }

    fn label_values_for_matcher_slice(
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
        // E.g., __name__ with a selector like {__name__="xyz"}
        let has_matchers_for_other_labels = matchers.iter().any(|m| m.label != name);
        all_values.retain(|v| matchers.iter().all(|m| m.label != name || m.matches(v)));

        if all_values.is_empty() {
            return Ok(all_values);
        }

        // If we don't have any matchers for other labels, then we're done.
        if !has_matchers_for_other_labels {
            return Ok(all_values);
        }

        let p = self.postings_for_matchers(matchers)?;

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
        let _ = self.process_label_values(
            name,
            &mut values,
            |_, _| true,
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
            let mut acc = PostingsBitmap::new();
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
                        acc &= bmp;
                    }
                }
            }
            match acc.cardinality() {
                0 => Ok(None),
                1 => Ok(acc.iter().next()),
                _ => {
                    let metric_name = format_metric_name(metric, labels);
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
        F: Fn(&mut CONTEXT, &str, &PostingsBitmap) -> ControlFlow<Option<T>>,
        PRED: Fn(&str, &PostingsBitmap) -> bool,
    {
        let prefix = get_key_for_label_prefix(label);
        let start_pos = prefix.len();
        for (key, map) in self.label_index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if predicate(value, map) {
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

        let mut count_map: FastHashMap<&str, SizeAccumulator> = FastHashMap::new();
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
                        let acc = SizeAccumulator { size, count };
                        entry.insert(acc);
                    }
                }

                label_value_pairs.push(PostingStat {
                    name: format!("{}={}", name, value),
                    count,
                });
                num_label_pairs += 1;

                if label == name {
                    metrics.push(PostingStat {
                        name: name.to_string(),
                        count,
                    });
                }
            }
        }

        let mut num_labels: usize = 0;

        for (name, v) in count_map {
            labels.push(PostingStat {
                name: name.to_string(),
                count: v.count,
            });
            label_value_length.push(PostingStat {
                name: name.to_string(),
                count: v.size as u64,
            });
            if name != NAME_LABEL && name != ALL_POSTINGS_KEY {
                num_labels += 1;
            }
        }

        PostingsStats {
            cardinality_metrics_stats: metrics.into_vec(),
            cardinality_label_stats: labels.into_vec(),
            label_value_stats: label_value_length.into_vec(),
            label_value_pairs_stats: label_value_pairs.into_vec(),
            num_label_pairs,
            num_labels,
        }
    }

    pub fn series_count_by_metric_name(
        &self,
        limit: usize,
        prefix: Option<&str>,
    ) -> Vec<(String, usize)> {
        let mut metrics = StatsMaxHeap::new(limit);
        let prefix = get_key_for_label_value(METRIC_NAME_LABEL, prefix.unwrap_or(""));
        for (key, bmp) in self.label_index.prefix(prefix.as_bytes()) {
            // keys and values are expected to be utf-8. If we panic, we have bigger issues
            if let Some((_name, value)) = key.split() {
                metrics.push(PostingStat {
                    name: value.to_string(),
                    count: bmp.cardinality(),
                });
            }
        }
        let items = metrics.into_vec();
        let mut result = Vec::new();
        for item in items {
            result.push((item.name, item.count as usize));
        }
        result
    }

    pub fn serialize_into<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // todo: version
        writer.write_varint(self.label_index.len() as u64)?;
        let mut buffer = Vec::new();
        for (key, bitmap) in self.label_index.iter() {
            write_key(key, &mut *writer)?;
            write_bitmap(writer, &mut buffer, bitmap)?;
        }
        Ok(())
    }

    pub fn deserialize_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let len = reader.read_varint::<u64>()?;

        let mut result = MemoryPostings::new();
        let mut buffer = Vec::new();
        for _ in 0..len {
            let key = read_key(&mut *reader)?;
            let bitmap = read_bitmap(reader, &mut buffer)?;
            result.label_index.insert(key, bitmap);
        }
        Ok(result)
    }
}

#[async_trait]
impl IndexReader for MemoryPostings {
    type Postings<'a> = BitmapPostings<'a>;

    async fn all_postings<'a>(&'a self) -> ProviderResult<Self::Postings<'a>> {
        let bmp = self.all_postings();
        let result = BitmapPostings::new(bmp);
        Ok(result)
    }

    async fn sorted_label_values(
        &self,
        name: &str,
        matchers: Option<&Matchers>,
    ) -> ProviderResult<Vec<String>> {
        let mut values = <MemoryPostings as IndexReader>::label_values(self, name.to_string(), matchers).await?;
        values.sort();
        Ok(values)
    }

    async fn label_values(
        &self,
        name: String,
        matchers: Option<&Matchers>,
    ) -> ProviderResult<Vec<String>> {
        let mut values: Vec<String> = Vec::new();
        if let Some(matchers) = matchers {
            let matched_postings = postings_for_matchers(self, matchers).await?;
            let postings = Bitmap64::from_iter(matched_postings);
            if !postings.is_empty() {
                self.process_label_values(
                    &name,
                    &mut values,
                    |_, ids| ids.intersect(&postings),
                    |state, value, _| {
                        state.push(value.to_string());
                        ControlFlow::<Option<()>>::Continue(())
                    },
                );
            }
        } else {
            self.process_label_values(
                &name,
                &mut values,
                |_, _| true,
                |state, value, _| {
                    state.push(value.to_string());
                    ControlFlow::<Option<()>>::Continue(())
                },
            );
        }
        Ok(values)
    }

    async fn postings<'a>(&'a self, name: String, values: Vec<String>) -> ProviderResult<Self::Postings<'a>> {
        let bmp = self.postings(&name, &values);
        let result = BitmapPostings::new(bmp);
        Ok(result)
    }

    async fn postings_for_label_matching<'a>(
        &'a self,
        name: String,
        match_fn: impl Fn(&'a str) -> bool + Send,
    ) -> ProviderResult<Self::Postings<'a>> {
        let mut res = PostingsBitmap::new();
        let prefix = get_key_for_label_prefix(&name);
        let start_pos = prefix.len();
        for (key, map) in self.label_index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if match_fn(value) {
                res |= map;
            }
        }
        let result = BitmapPostings::new(res);
        Ok(result)
    }

    async fn postings_for_all_label_values<'a>(&'a self, name: String) -> ProviderResult<Self::Postings<'a>> {
        let res = self.postings_for_all_label_values(&name);
        let result = BitmapPostings::new(res);
        Ok(result)
    }

    async fn sorted_postings<'a>(
        &'a self,
        _postings: impl Iterator<Item = SeriesRef> + Send,
    ) -> ProviderResult<Self::Postings<'a>> {
        
        unimplemented!("sorted_postings")
    }

    async fn label_names(&self, matchers: Option<&Matchers>) -> ProviderResult<Vec<String>> {
        let mut set: BTreeSet<String> = BTreeSet::new();
        if let Some(matchers) = matchers {
            let postings = postings_for_matchers(self, matchers).await?;
            let matched_bitmap = Bitmap64::from_iter(postings);
            if !matched_bitmap.is_empty() {
                for (k, postings) in self.label_index.iter() {
                    if let Some((key, _)) = k.split() {
                        if key != ALL_POSTINGS_KEY
                            && !set.contains(key)
                            && postings.intersect(&matched_bitmap)
                        {
                            set.insert(key.to_string());
                        }
                    }
                }
            }
        } else {
            for k in self.label_index.keys() {
                if let Some((key, _)) = k.split() {
                    if !set.contains(key) && key != ALL_POSTINGS_KEY {
                        set.insert(key.to_string());
                    }
                }
            }
        }

        let res = set.into_iter().collect::<Vec<_>>();
        Ok(res)
    }

    async fn label_value_for(&self, id: SeriesRef, label: String) -> ProviderResult<String> {
        let mut state = ();
        let value = self.process_label_values(
            &label,
            &mut state,
            |_, postings| postings.contains(id),
            |_, value, _| ControlFlow::Break(Some(value.to_string())),
        );
        if let Some(value) = value {
            Ok(value)
        } else {
            Err(ProviderError::NotFound) // todo: better error
        }
    }

    async fn label_names_for(
        &self,
        postings: impl Iterator<Item = SeriesRef>,
    ) -> ProviderResult<Vec<String>> {
        let bitmap: Bitmap64 = Bitmap64::from_iter(postings);
        // Slow
        let mut set: BTreeSet<String> = BTreeSet::new();
        if !bitmap.is_empty() {
            for (k, postings) in self.label_index.iter() {
                if let Some((key, _)) = k.split() {
                    if key != ALL_POSTINGS_KEY && !set.contains(key) && postings.intersect(&bitmap)
                    {
                        set.insert(key.to_string());
                    }
                }
            }
        }
        Ok(set.into_iter().collect::<Vec<_>>())
    }
}

fn write_bitmap<W: Write>(
    writer: &mut W,
    buf: &mut Vec<u8>,
    bitmap: &PostingsBitmap,
) -> io::Result<()> {
    buf.clear();

    // Docs for Portable state that it is Endian dependent, whereas the docs below imply that the data is
    // serialized as Little Endian
    // https://github.com/RoaringBitmap/RoaringFormatSpec?tab=readme-ov-file#extension-for-64-bit-implementations
    let serialized = bitmap.serialize_into_vec::<Portable>(buf);
    // Because Bitmap has no method to serialize using a writer,
    // we end up duplicating the serialized length so we can properly deserialize below
    writer.write_varint(serialized.len())?;
    writer.write_all(serialized)
}

fn read_bitmap<R: Read>(reader: &mut R, buf: &mut Vec<u8>) -> io::Result<PostingsBitmap> {
    let len = reader.read_varint::<usize>()?;
    buf.resize(len, 0);

    reader.read_exact(buf)?;
    // Not sure how I feel about the possible silent failure
    Ok(PostingsBitmap::deserialize::<Portable>(&buf))
}

fn write_key<W: Write>(key: &IndexKey, mut writer: W) -> io::Result<()> {
    let val = key.as_str();
    writer.write_varint(val.len())?;
    writer.write_all(val.as_bytes())
}

fn read_key<R: Read>(reader: &mut R) -> io::Result<IndexKey> {
    let len = reader.read_varint::<usize>()?;

    let mut data = vec![0u8; len];
    reader.read_exact(&mut data)?;
    let key = IndexKey::from(data);
    Ok(key)
}

// Note - assumes that labels is sorted
fn format_metric_name(name: &str, labels: &[Label]) -> String {
    let size_hint = name.len()
        + labels
            .iter()
            .map(|l| l.name.len() + l.value.len() + 3)
            .sum::<usize>();
    let mut full_name: String = String::with_capacity(size_hint);
    format_metric_name_into(&mut full_name, name, labels);
    full_name
}

fn format_metric_name_into(full_name: &mut String, name: &str, labels: &[Label]) {
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

fn get_bitmap_size(bmp: &PostingsBitmap) -> usize {
    bmp.cardinality() as usize * size_of::<SeriesRef>()
}
