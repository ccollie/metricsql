use crate::provider::error::ProviderResult;
use crate::SeriesRef;
use metricsql_common::hash::{FastHashSet};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

pub type PostingsListVec = SmallVec<SeriesRef, 16>;

pub trait PostingsIterator: Iterator<Item = SeriesRef> {
    fn is_empty(&self) -> bool {
        self.size_hint().0 == 0
    }
}

pub trait PostingsList {
    fn is_empty(&self) -> bool;
}

pub enum PostingsEnum<T> {
    Empty,
    List(ListPostings),
    Wrapped(T),
}

impl<T: PostingsIterator> PostingsEnum<T> {
    pub(super) fn empty() -> Self {
        PostingsEnum::Empty
    }

    pub(super) fn wrap(inner: T) -> Self {
        PostingsEnum::Wrapped(inner)
    }

    pub fn list(inner: impl Iterator<Item = SeriesRef>) -> Self {
        let inner = ListPostings::new(&inner.collect::<Vec<_>>());
        PostingsEnum::List(inner)
    }

    fn is_empty(&self) -> bool {
        match self {
            PostingsEnum::Empty => true,
            PostingsEnum::List(l) => l.list.is_empty(),
            PostingsEnum::Wrapped(w) => w.is_empty(),
        }
    }
}

impl<T: PostingsIterator> Iterator for PostingsEnum<T> {
    type Item = SeriesRef;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PostingsEnum::Empty => None,
            PostingsEnum::List(list) => list.next(),
            PostingsEnum::Wrapped(wrapped) => wrapped.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            PostingsEnum::Empty => (0, Some(0)),
            PostingsEnum::List(list) => list.size_hint(),
            PostingsEnum::Wrapped(wrapped) => wrapped.size_hint(),
        }
    }
}

impl<T: PostingsIterator> PostingsIterator for PostingsEnum<T> {
    fn is_empty(&self) -> bool {
        match self {
            PostingsEnum::Empty => true,
            PostingsEnum::List(l) => l.list.is_empty(),
            PostingsEnum::Wrapped(w) => w.is_empty(),
        }
    }
}

pub struct EmptyPostings {}
impl PostingsList for EmptyPostings {
    fn is_empty(&self) -> bool {
        true
    }
}

impl EmptyPostings {
    pub fn new() -> Self {
        EmptyPostings {}
    }
}

impl Iterator for EmptyPostings {
    type Item = SeriesRef;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl PostingsIterator for EmptyPostings {
    fn is_empty(&self) -> bool {
        true
    }
}

#[derive(Clone)]
pub(super) struct ListPostings {
    list: PostingsListVec,
    idx: usize,
}

impl ListPostings {
    fn new(values: &[SeriesRef]) -> Self {
        let mut list: PostingsListVec = PostingsListVec::new();
        list.extend_from_slice(values);
        list.sort();

        ListPostings { list, idx: 0 }
    }
}

impl Iterator for ListPostings {
    type Item = SeriesRef;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.list.len() {
            // SAFETY: we know that the index is within bounds because of the above check
            unsafe {
                let res = self.list.get_unchecked(self.idx);
                self.idx += 1;
                Some(*res)
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.list.len() - self.idx;
        (len, Some(len))
    }
}

// Define a struct to wrap Postings and implement Ord, PartialOrd, Eq, and PartialEq for BinaryHeap
struct PostingsWrapper<T: PostingsList> {
    postings: T,
    cur: SeriesRef,
}

impl<T: PostingsList> Ord for PostingsWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cur.cmp(&self.cur)
    }
}

impl<T: PostingsList> PartialOrd for PostingsWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PostingsList> Eq for PostingsWrapper<T> {}

impl<T: PostingsList> PartialEq for PostingsWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cur == other.cur
    }
}

pub(super) fn find_intersecting_postings<T: PostingsIterator>(
    p: T,
    candidates: Vec<PostingsEnum<T>>,
) -> ProviderResult<FastHashSet<PostingsWithIndex>> {
    let mut set: FastHashSet<PostingsWithIndex> = FastHashSet::with_capacity(candidates.len() * 4);
    if p.is_empty() {
        return Ok(set);
    }

    fn add_iter(
        dest: &mut FastHashSet<PostingsWithIndex>,
        index: usize,
        iter: impl Iterator<Item = SeriesRef>,
    ) {
        let items = iter.map(|at| PostingsWithIndex { index, at });
        dest.extend(items);
    }

    add_iter(&mut set, usize::MAX, p);

    for (index, it) in candidates.into_iter().enumerate() {
        if !it.is_empty() {
            add_iter(&mut set, index, it)
        }
    }

    set.retain(|x| x.index != usize::MAX);

    Ok(set)
}

/// PostingsWithIndex is used as postingsWithIndexHeap elements by FindIntersectingPostings,
/// keeping track of the original index of each posting while they move inside the heap.
#[derive(Eq, PartialEq)]
pub(super) struct PostingsWithIndex {
    pub(super) index: usize,
    pub(super) at: SeriesRef,
}

impl Ord for PostingsWithIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        other.at.cmp(&self.at)
    }
}

impl PartialOrd for PostingsWithIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for PostingsWithIndex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.at.hash(state);
    }
}

#[cfg(test)]
mod tests {}
