use async_trait::async_trait;
use crate::provider::error::ProviderResult;
use crate::provider::postings::PostingsIterator;
use crate::SeriesRef;
use metricsql_parser::label::Matchers;

/// IndexReader provides read access to serialized index data.
#[async_trait]
pub trait IndexReader {
    type Postings<'a>: PostingsIterator where Self: 'a;

    async fn all_postings<'a>(&'a self) -> ProviderResult<Self::Postings<'a>>;

    /// `sorted_label_values` returns sorted possible label values.
    async fn sorted_label_values(
        &self,
        name: &str,
        matchers: Option<&Matchers>,
    ) -> ProviderResult<Vec<String>>;

    /// `label_values` returns possible label values which may not be sorted.
    async fn label_values(
        &self,
        name: String,
        matchers: Option<&Matchers>,
    ) -> ProviderResult<Vec<String>>;

    /// Returns the postings list iterator for the label pairs.
    /// Found IDs are not strictly required to point to a valid Series, e.g. during background garbage collections.
    async fn postings<'a>(&'a self, name: String, values: Vec<String>)
        -> ProviderResult<Self::Postings<'a>>;

    /// `postings_for_label_matching` returns a sorted iterator over postings having a label with the given name and a value for which match returns true.
    /// If no postings are found having at least one matching label, an empty iterator is returned.
    async fn postings_for_label_matching<'a>(
        &'a self,
        name: String,
        match_fn: impl Fn(&'a str) -> bool + Send,
    ) -> ProviderResult<Self::Postings<'a>>;

    /// `postings_for_all_label_values` returns a sorted iterator over all postings having a label with the given name.
    /// If no postings are found with the label in question, an empty iterator is returned.
    async fn postings_for_all_label_values<'a>(
        &'a self,
        name: String,
    ) -> ProviderResult<Self::Postings<'a>>;

    /// `sorted_postings` returns a posting list reordered to be sorted by the label set of the underlying series.
    async fn sorted_postings<'a>(
        &'a self,
        postings: impl Iterator<Item = SeriesRef>,
    ) -> ProviderResult<Self::Postings<'a>>;

    // async fn series(&self, ref_id: SeriesRef, builder: &mut LabelsBuilder, chunks: &mut Vec<ChunkMeta>) -> Result<()>;

    /// `label_names` returns all the unique label names present in the index in sorted order.
    async fn label_names(&self, matchers: Option<&Matchers>) -> ProviderResult<Vec<String>>;

    /// `label_value_for` returns the label value for the given label name in the series referred to by ID.
    /// If the series couldn't be found or the series doesn't have the requested label a
    /// storage.ErrNotFound is returned as an error.
    async fn label_value_for(&self, id: SeriesRef, label: String) -> ProviderResult<String>;

    /// `label_names_for` returns all the label names for the series referred to by the postings.
    /// The names returned are sorted.
    async fn label_names_for(
        &self,
        postings: impl Iterator<Item = SeriesRef>,
    ) -> ProviderResult<Vec<String>>;
}
