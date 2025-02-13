use crate::provider::error::ProviderResult;
use crate::provider::postings::PostingsList;
use crate::SeriesRef;
use futures::future::BoxFuture;
use metricsql_parser::label::{Matcher, Matchers};


/// IndexReader provides reading access of serialized index data.
pub trait IndexReader {
    type Iter: PostingsList;
    
    fn all_postings<'a>(&'a self) -> BoxFuture<'a, ProviderResult<Box<dyn PostingsList>>>;
    
    /// `sorted_label_values` returns sorted possible label values.
    async fn sorted_label_values(&self, name: &str, matchers: &[Matchers]) -> ProviderResult<Vec<String>>;

    /// `label_values` returns possible label values which may not be sorted.
    async fn label_values(&self, name: &str, matchers: &[Matcher]) -> ProviderResult<Vec<String>>;

    /// returns the postings list iterator for the label pairs.
    /// The Postings here contain the offsets to the series inside the index.
    /// Found IDs are not strictly required to point to a valid Series, e.g. during background garbage collections.
    fn postings(&self, name: &str, values: &[String]) -> BoxFuture<'_, ProviderResult<Box<dyn PostingsList>>>;

    /// `postings_for_label_matching` returns a sorted iterator over postings having a label with the given name and a value for which match returns true.
    /// If no postings are found having at least one matching label, an empty iterator is returned. 
    fn postings_for_label_matching(&self, name: &str, match_fn: impl Fn(&str) -> bool) -> BoxFuture<'_, ProviderResult<Box<dyn PostingsList>>>;

    /// `postings_for_all_label_values` returns a sorted iterator over all postings having a label with the given name.
    /// If no postings are found with the label in question, an empty iterator is returned.
    fn postings_for_all_label_values(&self, name: &str) -> BoxFuture<'_, ProviderResult<Box<dyn PostingsList>>>;

    /// `sorted_postings` returns a postings list that is reordered to be sorted by the label set of the underlying series.
    fn sorted_postings(&self, postings: impl Iterator<Item=SeriesRef>) -> BoxFuture<'_, ProviderResult<Box<dyn PostingsList>>>;

    // async fn series(&self, ref_id: SeriesRef, builder: &mut LabelsBuilder, chunks: &mut Vec<ChunkMeta>) -> Result<()>;

    /// `label_names` returns all the unique label names present in the index in sorted order.
    async fn label_names(&self, matchers: &[Matcher]) -> ProviderResult<Vec<String>>;

    /// `label_value_for` returns the label value for the given label name in the series referred to by ID.
    /// If the series couldn't be found or the series doesn't have the requested label a
    /// storage.ErrNotFound is returned as error.
    async fn label_value_for(&self, id: SeriesRef, label: &str) -> ProviderResult<String>;

    /// `label_names_for` returns all the label names for the series referred to by the postings.
    /// The names returned are sorted.
    async fn label_names_for(&self, postings: impl Iterator<Item=SeriesRef>) -> ProviderResult<Vec<String>>;
}