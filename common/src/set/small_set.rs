use core::borrow::Borrow;
use core::fmt::{self, Debug};
use core::hash::{BuildHasher, Hash};
use core::iter::{FromIterator, IntoIterator};
use core::ops::{BitAnd, BitOr, BitXor, Sub};
use small_map::SmallMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasherDefault;
use ahash::AHasher;

/// A set implementation optimized for small collections, based on `SmallMap` from the small-map crate.
///
/// It uses inline storage for small collections and falls back to a standard `HashSet` when needed.
#[derive(Clone)]
pub struct SmallSet<const N: usize, T, S = RandomState>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    inner: SmallMap<N, T, (), S>,
}

impl<const N: usize, T, S> SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    /// Creates an empty `SmallSet` with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self
    where
        S: Default,
    {
        SmallSet {
            inner: SmallMap::with_capacity(capacity),
        }
    }

    /// Creates an empty `SmallSet` with the specified capacity and hash builder.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        SmallSet {
            inner: SmallMap::with_capacity_and_hasher(capacity, hash_builder),
        }
    }

    /// Creates an empty `SmallSet` with the default hasher.
    pub fn new() -> Self
    where
        S: Default,
    {
        SmallSet {
            inner: SmallMap::new(),
        }
    }

    /// Creates an empty `SmallSet` with the specified hash builder.
    pub fn with_hasher(hash_builder: S) -> Self {
        SmallSet {
            inner: SmallMap::with_hasher(hash_builder),
        }
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears the set, removing all elements.
    pub fn clear(&mut self)
    where
        S: Default,
    {
        *self = Self::new();
    }

    /// Returns `true` if the set contains the specified value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get(value).is_some()
    }

    /// Adds a value to the set.
    ///
    /// Returns `true` if the set did not already contain this value.
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value, ()).is_none()
    }

    /// Removes a value from the set.
    ///
    /// Returns `true` if the set contained the value.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    /// Returns an iterator over the set's elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter().map(|(k, _)| k)
    }

    /// Returns the number of elements the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint<'a, S2>(&'a self, other: &'a SmallSet<N, T, S2>) -> bool
    where
        T: Eq + Hash,
        S2: BuildHasher,
    {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if `other` contains all values in `self`.
    pub fn is_subset<'a, S2>(&'a self, other: &'a SmallSet<N, T, S2>) -> bool
    where
        T: Eq + Hash,
        S2: BuildHasher,
    {
        if self.len() > other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if `self` contains all values in `other`.
    pub fn is_superset<'a, S2>(&'a self, other: &'a SmallSet<N, T, S2>) -> bool
    where
        T: Eq + Hash,
        S2: BuildHasher,
    {
        other.is_subset(self)
    }

    pub fn intersection<S2>(&self, other: &SmallSet<N, T, S2>) -> SmallSet<N, T, S2>
    where
        T: Eq + Hash + Clone,
        S2: BuildHasher + Default,
    {
        let mut result = SmallSet::with_capacity(self.len().min(other.len()));
        for value in self.iter() {
            if other.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }
}

impl<const N: usize, T, S> Default for SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, T, S> Debug for SmallSet<N, T, S>
where
    T: Eq + Hash + Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<const N: usize, T, S> PartialEq for SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|key| other.contains(key))
    }
}

impl<const N: usize, T, S> Eq for SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
}

impl<const N: usize, T, S> FromIterator<T> for SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut set = SmallSet::with_capacity(lower);
        for item in iter {
            set.insert(item);
        }
        set
    }
}

impl<const N: usize, T, S> Extend<T> for SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<'a, const N: usize, T, S> Extend<&'a T> for SmallSet<N, T, S>
where
    T: 'a + Eq + Hash + Copy,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

pub struct IntoIter<const N: usize, K, S> {
    iter: <SmallMap<N, K, (), S> as IntoIterator>::IntoIter,
}

impl<const N: usize, T, S> Iterator for IntoIter<N, T, S> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|(k, _)| k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<const N: usize, K, V> ExactSizeIterator for IntoIter<N, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<const N: usize, T, S> IntoIterator for SmallSet<N, T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    type Item = T;
    type IntoIter = IntoIter<N, T, S>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.inner.into_iter(),
        }
    }
}

// Set operations implementations
impl<const N: usize, T, S, S2> BitOr<&SmallSet<N, T, S2>> for &SmallSet<N, T, S>
where
    T: Eq + Hash + Clone,
    S: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = SmallSet<N, T, S>;

    fn bitor(self, rhs: &SmallSet<N, T, S2>) -> Self::Output {
        let mut result = SmallSet::with_capacity(self.len() + rhs.len());
        result.extend(self.iter().cloned());
        result.extend(rhs.iter().cloned());
        result
    }
}


impl<const N: usize, T, S1, S2> BitAnd<&SmallSet<N, T, S2>> for &SmallSet<N, T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = SmallSet<N, T, S1>;

    fn bitand(self, rhs: &SmallSet<N, T, S2>) -> Self::Output {
        let mut result = SmallSet::with_capacity(self.len().min(rhs.len()));
        for value in self.iter() {
            if rhs.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }
}

impl<const N: usize, T, S1, S2> BitXor<&SmallSet<N, T, S2>> for &SmallSet<N, T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = SmallSet<N, T, S1>;

    fn bitxor(self, rhs: &SmallSet<N, T, S2>) -> Self::Output {
        let mut result = SmallSet::with_capacity(self.len() + rhs.len());
        for value in self.iter() {
            if !rhs.contains(value) {
                result.insert(value.clone());
            }
        }
        for value in rhs.iter() {
            if !self.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }
}

impl<const N: usize, T, S1, S2> Sub<&SmallSet<N, T, S2>> for &SmallSet<N, T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = SmallSet<N, T, S1>;

    fn sub(self, rhs: &SmallSet<N, T, S2>) -> Self::Output {
        let mut result = SmallSet::with_capacity(self.len());
        for value in self.iter() {
            if !rhs.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }
}

pub type ASmallSet<const N: usize, K> = SmallSet<N, K, BuildHasherDefault<AHasher>>;

// Convenience tests to verify implementation
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut set = SmallSet::<8, _>::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());

        assert!(set.insert(1));
        assert!(!set.insert(1));
        assert!(set.insert(2));

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());

        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&3));

        assert!(set.remove(&1));
        assert!(!set.remove(&1));
        assert_eq!(set.len(), 1);

        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_from_iterator() {
        let set: SmallSet<16, i32> = [1, 2, 3, 3, 4].iter().cloned().collect();
        assert_eq!(set.len(), 4);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(set.contains(&4));
    }

    #[test]
    fn test_set_operations() {
        let set1: SmallSet<16, i32> = [1, 2, 3].iter().cloned().collect();
        let set2: SmallSet<16, i32> = [3, 4, 5].iter().cloned().collect();

        let union = &set1 | &set2;
        let intersection = &set1 & &set2;
        let difference = &set1 - &set2;
        let symmetric_difference = &set1 ^ &set2;

        assert_eq!(union.len(), 5);
        assert_eq!(intersection.len(), 1);
        assert_eq!(difference.len(), 2);
        assert_eq!(symmetric_difference.len(), 4);

        assert!(intersection.contains(&3));
        assert!(!set1.is_disjoint(&set2));

        let set3: SmallSet<16, i32> = [6, 7].iter().cloned().collect();
        assert!(set1.is_disjoint(&set3));

        assert!(set1.is_subset(&union));
        assert!(set2.is_subset(&union));
        assert!(union.is_superset(&set1));
        assert!(union.is_superset(&set2));
    }
}