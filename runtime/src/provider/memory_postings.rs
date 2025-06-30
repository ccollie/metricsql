use super::error::{ProviderError, ProviderResult};
use super::index_key::{
    format_key_for_label_value, format_key_for_metric_name, get_key_for_label_prefix,
    get_key_for_label_value, IndexKey,
};
use crate::types::{MetricName, METRIC_NAME_LABEL};
use async_trait::async_trait;
use blart::map::Entry as ARTEntry;
use blart::TreeMap;
use croaring::bitmap64::Bitmap64Iterator;
use metricsql_common::hash::FastHashMap;
use metricsql_parser::label::{Label, Matchers, NAME_LABEL};
use std::collections::hash_map::Entry;
use std::collections::BTreeSet;
use std::io;
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::sync::{LazyLock, RwLock};

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
static EMPTY_BITMAP: LazyLock<PostingsBitmap> = LazyLock::new(PostingsBitmap::new);

pub type PostingsBitmap = Bitmap64;
// label
// label=value
pub type PostingsIndex = TreeMap<IndexKey, PostingsBitmap>;

pub struct BitmapPostings {
    bitmap: PostingsBitmap,
    cursor: Option<Bitmap64Iterator<'static>>, // We'll use transmute to make this work
    len: usize,
}

// Explicitly implement Send for BitmapPostings
// This is safe because the cursor is only used internally and never exposed
unsafe impl Send for BitmapPostings {}

impl BitmapPostings {
    pub fn new(bitmap: PostingsBitmap) -> Self {
        let len = bitmap.cardinality() as usize;
        // Create a static lifetime iterator that actually points to our owned bitmap
        // SAFETY: This is safe because the iterator is bound to the lifetime of bitmap,
        // which we own and keep alive for the entire lifetime of this struct
        let cursor = unsafe {
            let iter = bitmap.iter();
            std::mem::transmute::<Bitmap64Iterator<'_>, Bitmap64Iterator<'static>>(iter)
        };

        BitmapPostings {
            bitmap,
            cursor: Some(cursor),
            len,
        }
    }
}

impl Iterator for BitmapPostings {
    type Item = SeriesRef;

    fn next(&mut self) -> Option<Self::Item> {
        // Unwrap is safe here as we never set cursor to None except after consuming it
        self.cursor.as_mut().and_then(|cursor| cursor.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl PostingsIterator for BitmapPostings {
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug)]
pub struct MemoryPostings {
    all_postings_key: IndexKey,
    // Map from label name and (label name, label value) to a set of timeseries ids.
    label_index: RwLock<PostingsIndex>,
}

impl Default for MemoryPostings {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MemoryPostings {
    fn clone(&self) -> Self {
        let copy = self
            .label_index
            .read()
            .expect("Failed to read label index from RWLock")
            .clone();
        MemoryPostings {
            all_postings_key: self.all_postings_key.clone(),
            label_index: RwLock::new(copy),
        }
    }
}

impl MemoryPostings {
    pub fn new() -> Self {
        let mut index: PostingsIndex = Default::default();
        let all_postings_key = IndexKey::from(ALL_POSTINGS_KEY);

        index.insert(all_postings_key.clone(), PostingsBitmap::default());
        MemoryPostings {
            all_postings_key,
            label_index: RwLock::new(index),
        }
    }

    #[inline]
    fn read_index(&self) -> std::sync::RwLockReadGuard<'_, PostingsIndex> {
        self.label_index
            .read()
            .expect("Failed to read label index from RWLock")
    }

    fn write_index(&self) -> std::sync::RwLockWriteGuard<'_, PostingsIndex> {
        self.label_index
            .write()
            .expect("Failed to write label index from RWLock")
    }

    pub fn clear(&self) {
        let mut index = self.write_index();
        index.clear();
    }

    pub fn label_index(&self) -> std::sync::RwLockReadGuard<'_, PostingsIndex> {
        self.label_index.read().unwrap()
    }

    pub fn add_posting(&self, id: SeriesRef, metric_name: &MetricName) {
        debug_assert!(id != 0);

        let mut index = self.write_index();

        if !metric_name.measurement.is_empty() {
            Self::add_posting_for_label_value_internal(
                &mut index,
                id,
                METRIC_NAME_LABEL,
                &metric_name.measurement,
            );
        }

        for Label { name, value } in metric_name.labels.iter() {
            Self::add_posting_for_label_value_internal(&mut index, id, name, value);
        }

        Self::add_id_to_all_postings_internal(&mut index, id, &self.all_postings_key);
    }

    pub fn reindex_posting(&self, id: SeriesRef, metric_name: &MetricName) {
        let mut index = self.write_index();
        Self::remove_posting_by_id_and_labels_internal(
            &mut index,
            id,
            &metric_name.measurement,
            &metric_name.labels,
            &self.all_postings_key,
        );

        if !metric_name.measurement.is_empty() {
            Self::add_posting_for_label_value_internal(
                &mut index,
                id,
                METRIC_NAME_LABEL,
                &metric_name.measurement,
            );
        }

        for Label { name, value } in metric_name.labels.iter() {
            Self::add_posting_for_label_value_internal(&mut index, id, name, value);
        }

        Self::add_id_to_all_postings_internal(&mut index, id, &self.all_postings_key);
    }

    pub fn has_posting(&self, id: SeriesRef) -> bool {
        let index = self.read_index();
        index
            .get(&self.all_postings_key)
            .is_some_and(|bitmap| bitmap.contains(id))
    }

    pub fn remove_posting(&self, id: SeriesRef, metric_name: &MetricName) {
        let mut index = self.write_index();
        Self::remove_posting_by_id_and_labels_internal(
            &mut index,
            id,
            &metric_name.measurement,
            &metric_name.labels,
            &self.all_postings_key,
        );
    }

    fn remove_posting_by_id_and_labels_internal(
        index: &mut PostingsIndex,
        id: SeriesRef,
        metric_name: &str,
        labels: &[Label],
        all_postings_key: &IndexKey,
    ) {
        Self::remove_id_from_all_postings_internal(index, id, all_postings_key);

        // should never happen, but just in case
        if metric_name.is_empty() && labels.is_empty() {
            return;
        }

        if !metric_name.is_empty() {
            Self::remove_posting_for_label_value_internal(
                index,
                METRIC_NAME_LABEL,
                metric_name,
                id,
            );
        }

        for Label { name, value } in labels.iter() {
            Self::remove_posting_for_label_value_internal(index, name, value, id);
        }
    }

    pub fn has_label(&self, label: &str) -> bool {
        let index = self.read_index();
        let prefix = get_key_for_label_prefix(label);
        index.prefix(prefix.as_bytes()).next().is_some()
    }

    fn add_posting_for_label_value_internal(
        index: &mut PostingsIndex,
        ts_id: SeriesRef,
        label: &str,
        value: &str,
    ) -> bool {
        let key = IndexKey::for_label_value(label, value);
        match index.entry(key) {
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

    fn remove_posting_for_label_value_internal(
        index: &mut PostingsIndex,
        label: &str,
        value: &str,
        ts_id: SeriesRef,
    ) {
        let key = IndexKey::for_label_value(label, value);
        if let Some(bmp) = index.get_mut(&key) {
            bmp.remove(ts_id);
            if bmp.is_empty() {
                index.remove(&key);
            }
        }
    }

    pub fn postings_for_all_label_values(&self, label_name: &str) -> PostingsBitmap {
        let index = self.read_index();
        let prefix = get_key_for_label_prefix(label_name);
        let mut result = PostingsBitmap::new();
        for (_, map) in index.prefix(prefix.as_bytes()) {
            result |= map;
        }
        result
    }

    pub fn all_postings(&self) -> PostingsBitmap {
        let index = self.read_index();
        index
            .get(&self.all_postings_key)
            .map_or_else(|| EMPTY_BITMAP.clone(), |bitmap| bitmap.clone())
    }

    pub fn max_id(&self) -> SeriesRef {
        let index = self.read_index();
        index
            .get(&self.all_postings_key)
            .and_then(|bitmap| bitmap.maximum())
            .unwrap_or_default()
    }

    fn add_id_to_all_postings_internal(
        index: &mut PostingsIndex,
        id: SeriesRef,
        all_postings_key: &IndexKey,
    ) {
        if let Some(bitmap) = index.get_mut(all_postings_key) {
            bitmap.add(id);
        } else {
            let mut bmp = PostingsBitmap::new();
            bmp.add(id);
            index.insert(all_postings_key.clone(), bmp);
        }
    }

    fn remove_id_from_all_postings_internal(
        index: &mut PostingsIndex,
        id: SeriesRef,
        all_postings_key: &IndexKey,
    ) {
        if let Some(bmp) = index.get_mut(all_postings_key) {
            bmp.remove(id);
        }
    }

    /// `postings` returns the postings list iterator for the label pairs.
    /// The postings here contain the ids to the series inside the index.
    pub fn postings(&self, name: &str, values: &[String]) -> PostingsBitmap {
        let index = self.read_index();
        let mut result = PostingsBitmap::new();
        for value in values {
            let key = IndexKey::for_label_value(name, value);
            if let Some(bmp) = index.get(&key) {
                result |= bmp;
            }
        }
        result
    }

    pub fn postings_for_label_value(&self, name: &str, value: &str) -> PostingsBitmap {
        let index = self.read_index();
        let key = IndexKey::for_label_value(name, value);
        index
            .get(&key)
            .map_or_else(PostingsBitmap::default, |bmp| bmp.clone())
    }

    /// `postings_for_label_matching` returns postings having a label with the given name and a value
    /// for which the match returns true. If no postings are found having at least one matching label,
    /// an empty bitmap is returned.
    pub fn postings_for_label_matching(
        &self,
        name: &str,
        match_fn: fn(&str) -> bool,
    ) -> PostingsBitmap {
        let index = self.read_index();
        let prefix = get_key_for_label_prefix(name);
        let start_pos = prefix.len();
        let mut result = PostingsBitmap::new();
        for (key, map) in index.prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if match_fn(value) {
                result |= map;
            }
        }
        result
    }

    /// label_names returns all the unique label names.
    pub fn label_names(&self) -> Vec<String> {
        let index = self.read_index();
        let mut set: BTreeSet<String> = BTreeSet::new();
        for key in index.keys() {
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

    /// This exists primarily to ensure that we disallow duplicate metric names
    pub fn posting_by_name_and_labels(
        &self,
        metric: &str,
        labels: &[Label],
    ) -> ProviderResult<Option<SeriesRef>> {
        let index = self.read_index();
        let mut key: String = String::new();
        format_key_for_metric_name(&mut key, metric);
        if let Some(measurement_bmp) = index.get(key.as_bytes()) {
            let mut first = true;
            let mut acc = PostingsBitmap::new();
            for label in labels.iter() {
                format_key_for_label_value(&mut key, &label.name, &label.value);
                if let Some(bmp) = index.get(key.as_bytes()) {
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
        let index = self.read_index();
        let prefix = get_key_for_label_prefix(label);
        let start_pos = prefix.len();
        for (key, map) in index.prefix(prefix.as_bytes()) {
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
        let index = self.read_index();
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

        for (key, bitmap) in index.iter() {
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
                    name: format!("{name}={value}"),
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
        let index = self.read_index();
        let mut metrics = StatsMaxHeap::new(limit);
        let prefix = get_key_for_label_value(METRIC_NAME_LABEL, prefix.unwrap_or(""));
        for (key, bmp) in index.prefix(prefix.as_bytes()) {
            // Keys and values are expected to be utf-8. If we panic, we have bigger issues
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
        let index = self.read_index();
        // todo: version
        writer.write_varint(index.len() as u64)?;
        let mut buffer = Vec::new();
        for (key, bitmap) in index.iter() {
            write_key(key, &mut *writer)?;
            write_bitmap(writer, &mut buffer, bitmap)?;
        }
        Ok(())
    }

    pub fn deserialize_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let len = reader.read_varint::<u64>()?;

        let mut index: PostingsIndex = Default::default();
        let all_postings_key = IndexKey::from(ALL_POSTINGS_KEY);

        let mut buffer = Vec::new();
        for _ in 0..len {
            let key = read_key(&mut *reader)?;
            let bitmap = read_bitmap(reader, &mut buffer)?;
            index.insert(key, bitmap);
        }

        Ok(MemoryPostings {
            all_postings_key,
            label_index: RwLock::new(index),
        })
    }
}

#[async_trait]
impl IndexReader for MemoryPostings {
    type Postings = BitmapPostings;

    async fn all_postings(&self) -> ProviderResult<Self::Postings> {
        let bmp = self.all_postings();
        let result = BitmapPostings::new(bmp);
        Ok(result)
    }

    async fn sorted_label_values(
        &self,
        name: &str,
        matchers: Option<&Matchers>,
    ) -> ProviderResult<Vec<String>> {
        let mut values =
            <MemoryPostings as IndexReader>::label_values(self, name.to_string(), matchers).await?;
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

    async fn postings(&self, name: String, values: Vec<String>) -> ProviderResult<Self::Postings> {
        let bmp = self.postings(&name, &values);
        let result = BitmapPostings::new(bmp);
        Ok(result)
    }

    async fn postings_for_label_matching<'a, F>(
        &'a self,
        name: String,
        match_fn: F,
    ) -> ProviderResult<Self::Postings>
    where
        F: Fn(&str) -> bool + Send,
    {
        let mut res = PostingsBitmap::new();
        let prefix = get_key_for_label_prefix(&name);
        let start_pos = prefix.len();
        for (key, map) in self.label_index.read().unwrap().prefix(prefix.as_bytes()) {
            let value = key.sub_string(start_pos);
            if match_fn(value) {
                res |= map;
            }
        }
        let result = BitmapPostings::new(res);
        Ok(result)
    }

    async fn postings_for_all_label_values(&self, name: String) -> ProviderResult<Self::Postings> {
        let res = self.postings_for_all_label_values(&name);
        let result = BitmapPostings::new(res);
        Ok(result)
    }

    async fn sorted_postings(
        &self,
        _postings: impl Iterator<Item = SeriesRef> + Send,
    ) -> ProviderResult<Self::Postings> {
        unimplemented!("sorted_postings")
    }

    async fn label_names(&self, matchers: Option<&Matchers>) -> ProviderResult<Vec<String>> {
        let mut set: BTreeSet<String> = BTreeSet::new();
        if let Some(matchers) = matchers {
            let postings = postings_for_matchers(self, matchers).await?;
            if !postings.is_empty() {
                let matched = Bitmap64::from_iter(postings);
                for (k, postings) in self.label_index.read().unwrap().iter() {
                    if let Some((key, _)) = k.split() {
                        if key != ALL_POSTINGS_KEY
                            && !set.contains(key)
                            && postings.intersect(&matched)
                        {
                            set.insert(key.to_string());
                        }
                    }
                }
            }
        } else {
            for k in self.label_index.read().unwrap().keys() {
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
        postings: impl Iterator<Item = SeriesRef> + Send,
    ) -> ProviderResult<Vec<String>> {
        let bitmap: Bitmap64 = Bitmap64::from_iter(postings);
        // Slow
        let mut set: BTreeSet<String> = BTreeSet::new();
        if !bitmap.is_empty() {
            for (k, postings) in self.label_index.read().unwrap().iter() {
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
    Ok(PostingsBitmap::deserialize::<Portable>(buf))
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

// Note - assumes that labels are sorted
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
