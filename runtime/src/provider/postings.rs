use crate::provider::index_reader::IndexReaderResult;
use crate::SeriesRef;
use metricsql_common::hash::{FastHashSet, HashSetExt};
use smallvec::{smallvec, SmallVec};
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};
use std::hash::{Hash, Hasher};
use std::iter::Peekable;

pub type PostingsListVec = SmallVec<SeriesRef, 16>;


pub enum PostingsEnum {
    Empty(EmptyPostings),
    List(ListPostings),
    Intersection(IntersectPostings),
    Removed(Box<RemovedPostings>),
    Wrapped(Box<dyn PostingsList>),
    Iterator(BoxedIterator)
}

impl PostingsEnum {
    pub(super) fn empty() -> Self {
        PostingsEnum::Empty(EmptyPostings {})
    }

    pub(super) fn remove(full: PostingsEnum, removed: Box<dyn PostingsList>) -> Self {
        PostingsEnum::Removed(Box::new(RemovedPostings::new(full, removed)))
    }

    pub(super) fn intersection(its: Vec<BoxedPostings>) -> Self {
        PostingsEnum::Intersection(IntersectPostings::new(its))
    }

    pub(super) fn wrap(inner: BoxedPostings) -> Self {
        PostingsEnum::Wrapped(inner)
    }
}

impl Iterator for PostingsEnum {
    type Item = SeriesRef;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PostingsEnum::Empty(_e) => None,
            PostingsEnum::List(l) => l.next(),
            PostingsEnum::Intersection(i) => i.next(),
            PostingsEnum::Removed(r) => r.next(),
            PostingsEnum::Wrapped(w) => w.next(),
            PostingsEnum::Iterator(i) => {
                i.next()
            }
        }
    }
}

impl PostingsList for PostingsEnum {

    fn is_empty(&self) -> bool {
        match self {
            PostingsEnum::Empty(_) => true,
            PostingsEnum::List(l) => l.is_empty(),
            PostingsEnum::Intersection(i) => i.is_empty(),
            PostingsEnum::Removed(r) => r.is_empty(),
            PostingsEnum::Wrapped(w) => w.is_empty(),
            PostingsEnum::Iterator(_i) => { false }
        }
    }
}
pub trait PostingsList: Iterator<Item=SeriesRef> {
    fn is_empty(&self) -> bool;
}

pub type BoxedPostings = Box<dyn PostingsList>;

impl PostingsList for Box<dyn PostingsList> {
    fn is_empty(&self) -> bool {
        (**self).is_empty()
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
        EmptyPostings { }
    }
}

impl Iterator for EmptyPostings {
    type Item = SeriesRef;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
pub fn empty_postings() -> impl PostingsList {
    PostingsEnum::empty()
}

pub type BoxedIterator = Box<dyn Iterator<Item=SeriesRef>>;
type PeekableIterator = Box<Peekable<dyn Iterator<Item=SeriesRef>>>;

#[derive(Clone)]
pub(super) struct ListPostings {
    list: PostingsListVec,
    idx: usize,
    cur: SeriesRef,
}

impl ListPostings {
    fn new(values: &[SeriesRef]) -> Self {
        let mut list: PostingsListVec = PostingsListVec::new();
        list.extend_from_slice(values);
        // todo: ensure sorted
        ListPostings { 
            list,
            idx: 0,
            cur: 0 
        }
    }
}

impl PostingsList for ListPostings {
    fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

impl Iterator for ListPostings {
    type Item = SeriesRef;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.list.len() {
            let res = self.list.get(self.idx).map(|item| *item);
            self.idx += 1;
            res
        } else {
            None
        }
    }
}

pub(super) struct IntersectPostings {
    arr: Vec<Box<dyn PostingsList>>,
    values: SmallVec<SeriesRef, 32>,
    idx: usize,
    is_init: bool,
}

impl IntersectPostings {
    fn new(its: Vec<BoxedPostings>) -> Self {
        IntersectPostings {
            arr: its,
            idx: 0,
            values: smallvec![],
            is_init: false
        }
    }
}

impl PostingsList for IntersectPostings {
    fn is_empty(&self) -> bool {
        self.arr.iter().any(|p| p.is_empty())
    }
}


impl Iterator for IntersectPostings {
    type Item = SeriesRef;
    fn next(&mut self) -> Option<Self::Item> {
        if !self.is_init {
            self.is_init = true;
            let mut set: BTreeSet<SeriesRef> = BTreeSet::new();
            for iter in self.arr.iter_mut() {
                set.extend(iter);
            }
            self.values.extend(set);
        }
        if self.idx < self.values.len() {
            let res = self.values.get(self.idx).map(|item| *item);
            self.idx += 1;
            return res
        }
        None
    }
}

/// `intersect` returns a new postings list over the intersection of the input postings.
pub(super) fn intersect(its: Vec<Box<dyn PostingsList>>) -> PostingsEnum {
    if its.is_empty() {
        return PostingsEnum::empty()
    }

    if its.len() == 1 {
        let mut postings = its;
        let first = postings.pop().expect("BUG: index out of bounds in intersect");
        if first.is_empty() {
            return PostingsEnum::empty()
        }
        return PostingsEnum::wrap(first);
    }

    for posting in its.iter() {
        if posting.is_empty() {
            return PostingsEnum::empty()
        }
    }

    PostingsEnum::intersection(its)
}

pub(super) struct RemovedPostings {
    full: Peekable<PostingsEnum>,
    remove: Peekable<Box<dyn PostingsList>>,
}

impl RemovedPostings {
    fn new(full: PostingsEnum, remove: Box<dyn PostingsList>) -> Self {
        RemovedPostings {
            full: full.peekable(),
            remove: remove.peekable(),
         }
    }
}

impl PostingsList for RemovedPostings {
    fn is_empty(&self) -> bool {
        false
    }
}

impl Iterator for RemovedPostings {
    type Item = SeriesRef;

    fn next(&mut self) -> Option<SeriesRef> {
        loop {
            let full = self.full.peek().map(|x| *x);
            let remove = self.remove.peek().map(|x| *x);
            match (full, remove) {
                (Some(fcur), Some(rcur)) => {
                    match fcur.cmp(&rcur) {
                        Ordering::Less => {
                            self.full.next();
                            return Some(fcur);
                        }
                        Ordering::Greater => {
                            self.remove.next();
                        }
                        Ordering::Equal => {
                            self.full.next();
                        }
                    }
                }
                (Some(fcur), None) => {
                    return Some(fcur)
                }
                _=> { return None; }
            }
        }
    }
}


// Define a struct to wrap Postings and implement Ord, PartialOrd, Eq, and PartialEq for BinaryHeap
struct PostingsWrapper {
    postings: Box<dyn PostingsList>,
    cur: SeriesRef,
}

impl Ord for PostingsWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cur.cmp(&self.cur)
    }
}

impl PartialOrd for PostingsWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PostingsWrapper {}

impl PartialEq for PostingsWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.cur == other.cur
    }
}

// Define a struct for mergedPostings
pub(super) struct MergedPostings {
    h: BinaryHeap<PostingsWrapper>,
    cur: SeriesRef,
}

impl MergedPostings {
    fn new(p: Vec<Box<dyn PostingsList>>) -> Self {
        let mut h = BinaryHeap::new();
        for postings in p.into_iter() {
            h.push(PostingsWrapper { postings, cur: u64::MAX });
        }
        Self { h, cur: 0 }
    }
}

impl PostingsList for MergedPostings {
    fn is_empty(&self) -> bool {
        self.h.is_empty()
    }
}

impl Iterator for MergedPostings {
    type Item = SeriesRef;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut pw) = self.h.pop() {
            if let Some(new_item) = pw.postings.next() {
                if new_item <= self.cur {
                    pw.cur = new_item;
                    self.cur = new_item;
                    self.h.push(pw);
                    return Some(self.cur);
                }
                self.h.push(pw);
            }
        }
        None
    }
}

/// `without` returns a new postings list that contains all elements from the full list that
/// are not in the drop list.
pub(super) fn without(full: PostingsEnum, drop: Box<dyn PostingsList>) -> PostingsEnum {
    if full.is_empty() {
        return PostingsEnum::empty()
    }

    if drop.is_empty() {
        return full
    }

    PostingsEnum::remove(full, drop)
}


pub(super) fn find_intersecting_postings(p: impl PostingsList, candidates: Vec<BoxedPostings>) -> IndexReaderResult<FastHashSet<PostingsWithIndex>> {
    let mut set: FastHashSet<PostingsWithIndex> = FastHashSet::with_capacity(candidates.len() * 4);
    if p.is_empty() {
        return Ok(set);
    }

    fn add_iter(dest: &mut FastHashSet<PostingsWithIndex>, index: usize, iter: impl Iterator<Item=SeriesRef>) {
        let items = iter.map(|x| PostingsWithIndex { index, at: x });
        dest.extend(items);
    }

    add_iter(&mut set, usize::MAX, p.into_iter());

    for (index, it) in candidates.into_iter().enumerate() {
        if !it.is_empty() {
            add_iter(&mut set, index, it)
        }
    }

    set.retain(|x| x.index != usize::MAX);

    Ok(set)
}

/// postingsWithIndex is used as postingsWithIndexHeap elements by FindIntersectingPostings,
/// keeping track of the original index of each postings while they move inside the heap.
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
mod tests {
    use super::*;

    fn expand_postings(p: &impl PostingsList) -> Vec<u64> {
        p.iter().collect()
    }

    #[test]
    fn test_intersect() {
        let a = ListPostings::new(&[1, 2, 3]);
        let b = ListPostings::new(&[2, 3, 4]);

        fn empty() -> EmptyPostings { 
            EmptyPostings::new()
        }
        let cases: Vec<(Vec<BoxedIterator>, Vec<SeriesRef>)> = vec![
            (vec![], vec![]),
            (vec![Box::new(a.clone()), Box::new(b.clone()), Box::new(empty())], vec![]),
            (vec![Box::new(b.clone()), Box::new(a.clone()), Box::new(empty())], vec![]),
            (vec![Box::new(empty()), Box::new(b.clone()), Box::new(a.clone())], vec![]),
            (vec![Box::new(empty()), Box::new(a.clone()), Box::new(b.clone())], vec![]),
            (vec![Box::new(a.clone()), Box::new(empty()), Box::new(b.clone())], vec![]),
            (vec![Box::new(b.clone()), Box::new(empty()), Box::new(a.clone())], vec![]),
            (vec![Box::new(b.clone()), Box::new(empty()), Box::new(a.clone()), Box::new(a.clone()), Box::new(b.clone()), Box::new(a.clone()), Box::new(a.clone()), Box::new(a.clone())], vec![]),
            (vec![Box::new(ListPostings::new(&[1, 2, 3, 4, 5])), Box::new(ListPostings::new(&[6, 7, 8, 9, 10]))], vec![]),
            (vec![Box::new(ListPostings::new(&[1, 2, 3, 4, 5])), Box::new(ListPostings::new(&[4, 5, 6, 7, 8]))], vec![4, 5]),
            (vec![Box::new(ListPostings::new(&[1, 2, 3, 4, 9, 10])), Box::new(ListPostings::new(&[1, 4, 5, 6, 7, 8, 10, 11]))], vec![1, 4, 10]),
            (vec![Box::new(ListPostings::new(&[1])), Box::new(ListPostings::new(&[0, 1]))], vec![1]),
            (vec![Box::new(ListPostings::new(&[1]))], vec![1]),
            (vec![Box::new(ListPostings::new(&[1])), Box::new(ListPostings::new(&[]))], vec![]),
            (vec![Box::new(ListPostings::new(&[])), Box::new(ListPostings::new(&[]))], vec![]),
        ];

        for (inputs, expected) in cases {
            let intersect = IntersectPostings::new(inputs.into_iter().collect()).collect();
            let res = expand_postings(&intersect);
            assert_eq!(res, expected);
        }
    }
}