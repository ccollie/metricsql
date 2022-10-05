use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use crate::functions::types::{FunctionImplementation, FunctionRegistry};

pub(crate) struct HashMapFunctionRegistry<K, P, R>
    where
        K: Eq + Hash,
        P: ?Sized + Send + Sync,
{
    hash: HashMap<K, Box<dyn FunctionImplementation<P, R, Output=R>>>
}

impl <K, P: ?Sized + Send + Sync, R>HashMapFunctionRegistry<K, P, R>
    where K: Eq + Hash {
    pub fn new() -> Self {
        Self {
            hash: HashMap::with_capacity(24)
        }
    }
}

impl <K, P: ?Sized + Send + Sync, R> FunctionRegistry<K, P, R> for HashMapFunctionRegistry<K, P, R>
    where
        K: Eq + Hash
{
    fn into_vec(self) -> Vec<(K, Box<dyn FunctionImplementation<P, R, Output=R>>)> {
        todo!()
    }

    fn remove<Q: ?Sized>(&mut self, key: &Q) where K: Borrow<Q>, Q: Eq + Hash {
        self.hash.remove(key);
    }

    fn insert(&mut self, key: K, item: Box<dyn FunctionImplementation<P, R, Output=R>>) {
        self.hash.insert(key, item);
    }

    fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool where K: Borrow<Q>, Q: Eq + Hash {
        self.hash.contains_key(key)
    }

    fn get<Q: ?Sized>(&self, key: &Q) -> Option<&Box<dyn FunctionImplementation<P, R, Output=R>>>
        where K: Borrow<Q>, Q: Eq + Hash
    {
        self.hash.get(key)
    }

    fn len(&self) -> usize {
        self.hash.len()
    }
}