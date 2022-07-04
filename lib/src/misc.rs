// see https://stackoverflow.com/questions/58758812/is-there-a-way-to-run-a-closure-on-rwlock-drop
use std::ops::{Deref, DerefMut};

pub struct CleanOnDrop<D, F>
    where
        F: FnOnce(&mut D),
{
    data: D,
    cleanup: Option<F>,
}

impl<D, F> CleanOnDrop<D, F>
    where
        F: FnOnce(&mut D),
{
    pub fn new(data: D, cleanup: F) -> Self {
        Self { data, cleanup: Some(cleanup) }
    }
}

impl<D, F> Drop for CleanOnDrop<D, F>
    where
        F: FnOnce(&mut D)
{
    fn drop(&mut self) {
        if let Some(mut cleanup) = self.cleanup.take() {
            cleanup(&mut self.data);
        }
    }
}

impl<D, F> Deref for CleanOnDrop<D, F>
    where
        F: FnOnce(&mut D),
{
    type Target = D;
    fn deref(&self) -> &D {
        &self.data
    }
}

impl<D, F> DerefMut for CleanOnDrop<D, F>
    where
        F: FnOnce(&mut D),
{
    fn deref_mut(&mut self) -> &mut D {
        &mut self.data
    }
}