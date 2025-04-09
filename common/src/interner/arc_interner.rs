use papaya::{HashMap, HashSet};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, LazyLock};

/// A pointer to a reference-counted interned object.
///
/// The interned object will be held in memory only until its
/// reference count reaches zero.
///
/// # Example
/// ```rust
/// use metricsql_common::interner::ArcIntern;
///
/// let x = ArcIntern::new("hello");
/// let y = ArcIntern::new("world");
/// assert_ne!(x, y);
/// assert_eq!(x, ArcIntern::new("hello"));
/// assert_eq!(*x, "hello"); // dereference an ArcIntern like a pointer
/// ```
#[derive(Debug)]
pub struct ArcIntern<T: Eq + Hash + Send + Sync + 'static + ?Sized> {
    arc: Arc<T>,
}

type Container<T> = HashMap<Arc<T>, ()>;

static CONTAINER: LazyLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>> =
    LazyLock::new(|| HashMap::new());

impl<T: Eq + Hash + Send + Sync + 'static + ?Sized> ArcIntern<T> {
    fn from_arc(val: Arc<T>) -> ArcIntern<T> {
        let type_map = &CONTAINER;

        let map = type_map.pin();
        let boxed = map.get_or_insert_with(TypeId::of::<T>(), || Box::new(Container::<T>::new()));

        let m = boxed
            .downcast_ref::<Container<T>>()
            .expect("BUG: downcast of Container<T>")
            .pin();

        // m.get_or_insert(val, ());
        if let Some((key, _)) = m.get_key_value(&val) {
            ArcIntern { arc: key.clone() }
        } else {
            m.insert(val.clone(), ());
            ArcIntern { arc: val }
        }
    }

    /// See how many objects have been interned.  This may be helpful
    /// in analyzing memory use.
    pub fn num_objects_interned() -> usize {
        if let Some(m) = CONTAINER.pin().get(&TypeId::of::<T>()) {
            return m.downcast_ref::<Container<T>>().unwrap().len();
        }
        0
    }
    /// Return the number of references for this value.
    pub fn refcount(&self) -> usize {
        // One reference is held by the hashset; we return the number of
        // references held by actual clients.
        Arc::strong_count(&self.arc) - 1
    }
}

impl<T: Eq + Hash + Send + Sync + 'static> ArcIntern<T> {
    /// Intern a value.  If this value has not previously been
    /// interned, then `new` will allocate a spot for the value on the
    /// heap.  Otherwise, it will return a pointer to the object
    /// previously allocated.
    ///
    /// Note that `ArcIntern::new` is a bit slow, since it needs to check
    /// a `HashMap` which contains its own mutexes.
    pub fn new(val: T) -> ArcIntern<T> {
        Self::from_arc(Arc::new(val))
    }
}

impl<T: Eq + Hash + Send + Sync + 'static + ?Sized> Clone for ArcIntern<T> {
    fn clone(&self) -> Self {
        ArcIntern {
            arc: self.arc.clone(),
        }
    }
}

impl<T: Eq + Hash + Send + Sync + ?Sized> Drop for ArcIntern<T> {
    fn drop(&mut self) {
        if let Some(m) = CONTAINER.pin().get(&TypeId::of::<T>()) {
            let m = m
                .downcast_ref::<Container<T>>()
                .expect("BUG: downcast of Container<T>")
                .pin();

            if let Some((value, _)) = m.get_key_value(&self.arc) {
                // If the reference count is 2, then the only two remaining references
                // to this value are held by `self` and the hashmap and we can safely
                // deallocate the value.
                if Arc::strong_count(value) == 2 {
                    m.remove(&self.arc);
                }
            }
        }
    }
}

impl<T: Send + Sync + Hash + Eq + ?Sized> AsRef<T> for ArcIntern<T> {
    fn as_ref(&self) -> &T {
        self.arc.as_ref()
    }
}
impl<T: Eq + Hash + Send + Sync + ?Sized> Borrow<T> for ArcIntern<T> {
    fn borrow(&self) -> &T {
        self.as_ref()
    }
}
impl<T: Eq + Hash + Send + Sync + ?Sized> Deref for ArcIntern<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.as_ref()
    }
}

impl<T: Eq + Hash + Send + Sync + Display + ?Sized> Display for ArcIntern<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.deref().fmt(f)
    }
}

impl<T: Eq + Hash + Send + Sync + 'static + ?Sized> From<Box<T>> for ArcIntern<T> {
    fn from(b: Box<T>) -> Self {
        Self::from_arc(Arc::from(b))
    }
}

impl<'a, T> From<&'a T> for ArcIntern<T>
where
    T: Eq + Hash + Send + Sync + 'static + ?Sized,
    Arc<T>: From<&'a T>,
{
    fn from(t: &'a T) -> Self {
        Self::from_arc(Arc::from(t))
    }
}

impl<T: Eq + Hash + Send + Sync + 'static> From<T> for ArcIntern<T> {
    fn from(t: T) -> Self {
        ArcIntern::new(t)
    }
}
impl<T: Eq + Hash + Send + Sync + Default + 'static + ?Sized> Default for ArcIntern<T> {
    fn default() -> ArcIntern<T> {
        ArcIntern::new(Default::default())
    }
}

impl<T: Eq + Hash + Send + Sync + ?Sized> Hash for ArcIntern<T> {
    // `Hash` implementation must be equivalent for owned and borrowed values.
    fn hash<H: Hasher>(&self, state: &mut H) {
        let borrow: &T = self.borrow();
        borrow.hash(state);
    }
}

/// Efficiently compares two interned values by comparing their pointers.
impl<T: Eq + Hash + Send + Sync + ?Sized> PartialEq for ArcIntern<T> {
    fn eq(&self, other: &ArcIntern<T>) -> bool {
        Arc::ptr_eq(&self.arc, &other.arc)
    }
}
impl<T: Eq + Hash + Send + Sync + ?Sized> Eq for ArcIntern<T> {}

impl<T: Eq + Hash + Send + Sync + PartialOrd + ?Sized> PartialOrd for ArcIntern<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.as_ref().partial_cmp(other)
    }
    fn lt(&self, other: &Self) -> bool {
        self.as_ref().lt(other)
    }
    fn le(&self, other: &Self) -> bool {
        self.as_ref().le(other)
    }
    fn gt(&self, other: &Self) -> bool {
        self.as_ref().gt(other)
    }
    fn ge(&self, other: &Self) -> bool {
        self.as_ref().ge(other)
    }
}

impl<T: Eq + Hash + Send + Sync + Ord + ?Sized> Ord for ArcIntern<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_ref().cmp(other)
    }
}

impl<T: Eq + Hash + Send + Sync + Serialize + ?Sized> Serialize for ArcIntern<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(serializer)
    }
}

impl<'de, T: Eq + Hash + Send + Sync + 'static + ?Sized + Deserialize<'de>> Deserialize<'de>
    for ArcIntern<T>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::ArcIntern;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;

    // Test basic functionality.
    #[test]
    fn basic() {
        assert_eq!(ArcIntern::new("foo"), ArcIntern::new("foo"));
        assert_ne!(ArcIntern::new("foo"), ArcIntern::new("bar"));
        // The above refs should be deallocated by now.
        assert_eq!(ArcIntern::<&str>::num_objects_interned(), 0);

        let _interned1 = ArcIntern::new("foo".to_string());
        {
            let interned2 = ArcIntern::new("foo".to_string());
            let interned3 = ArcIntern::new("bar".to_string());

            assert_eq!(interned2.refcount(), 2);
            assert_eq!(interned3.refcount(), 1);
            // We now have two unique interned strings: "foo" and "bar".
            assert_eq!(ArcIntern::<String>::num_objects_interned(), 2);
        }

        // "bar" is now gone.
        assert_eq!(ArcIntern::<String>::num_objects_interned(), 1);
    }

    // Ordering should be based on values, not pointers.
    // Also tests `Display` implementation.
    #[test]
    fn sorting() {
        let mut interned_vals = vec![
            ArcIntern::new(4),
            ArcIntern::new(2),
            ArcIntern::new(5),
            ArcIntern::new(0),
            ArcIntern::new(1),
            ArcIntern::new(3),
        ];
        interned_vals.sort();
        let sorted: Vec<String> = interned_vals.iter().map(|v| format!("{}", v)).collect();
        assert_eq!(&sorted.join(","), "0,1,2,3,4,5");
    }

    #[derive(Eq, PartialEq, Hash)]
    pub struct TestStruct2(String, u64);

    #[test]
    fn sequential() {
        for _i in 0..10_000 {
            let mut interned = Vec::with_capacity(100);
            for j in 0..100 {
                interned.push(ArcIntern::new(TestStruct2("foo".to_string(), j)));
            }
        }

        assert_eq!(ArcIntern::<TestStruct2>::num_objects_interned(), 0);
    }

    #[derive(Eq, PartialEq, Hash)]
    pub struct TestStruct(String, u64, Arc<bool>);

    // Quickly create and destroy a small number of interned objects from
    // multiple threads.
    #[test]
    fn multithreading1() {
        let mut thandles = vec![];
        let drop_check = Arc::new(true);
        for _i in 0..10 {
            let t = thread::spawn({
                let drop_check = drop_check.clone();
                move || {
                    for _i in 0..100_000 {
                        let interned1 =
                            ArcIntern::new(TestStruct("foo".to_string(), 5, drop_check.clone()));
                        let _interned2 =
                            ArcIntern::new(TestStruct("bar".to_string(), 10, drop_check.clone()));
                        let mut m = HashMap::new();
                        // force some hashing
                        m.insert(interned1, ());
                    }
                }
            });
            thandles.push(t);
        }
        for h in thandles.into_iter() {
            h.join().unwrap()
        }
        assert_eq!(Arc::strong_count(&drop_check), 1);
        assert_eq!(ArcIntern::<TestStruct>::num_objects_interned(), 0);
    }

    #[test]
    fn test_unsized() {
        assert_eq!(
            ArcIntern::<[usize]>::from(&[1, 2, 3][..]),
            ArcIntern::from(&[1, 2, 3][..])
        );
        assert_ne!(
            ArcIntern::<[usize]>::from(&[1, 2][..]),
            ArcIntern::from(&[1, 2, 3][..])
        );
        // The above refs should be deallocated by now.
        assert_eq!(ArcIntern::<[usize]>::num_objects_interned(), 0);
    }
}
