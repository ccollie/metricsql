//! A `Future` value that resolves once it's explicitly completed, potentially
//! from a different thread or task similar to Java's `CompletableFuture`.
//!
//! Currently, this is implemented using the `BiLock` from the `futures` crate.
//!
//! # Examples
//!
//! Create an incomplete `ManualFuture` and explicitly complete it with the
//! completer:
//! ```
//! # use metricsql_common::async_runtime::ManualFuture;
//! # use futures::executor::block_on;
//! let (future, completer) = ManualFuture::<i32>::new();
//! block_on(async { completer.complete(5).await });
//! assert_eq!(block_on(future), 5);
//! ```
//!
//! Create an initially complete `ManualFuture` that can be immediately
//! resolved:
//! ```
//! # use metricsql_common::async_runtime::ManualFuture;
//! # use futures::executor::block_on;
//! assert_eq!(block_on(ManualFuture::new_completed(10)), 10);
//! ```
#![warn(clippy::pedantic)]

/// Source: https://github.com/dmarcuse/manual_future/tree/master
/// Copyright (c) 2023 David Marcuse
/// Licensed under the MIT License
///


use std::future::Future;
use std::marker::Unpin;
use std::pin::Pin;
use std::task::Waker;
use std::task::{Context, Poll};
use futures_util::lock::BiLock;

enum State<T> {
    Incomplete,
    Waiting(Waker),
    Complete(Option<T>),
}

impl<T> State<T> {
    fn new(value: Option<T>) -> Self {
        match value {
            None => Self::Incomplete,
            v @ Some(_) => Self::Complete(v),
        }
    }
}

/// A value that may or may not be completed yet.
///
/// This future will not be resolved until it's been explicitly completed, either
/// with `new_completed` or with `ManualFutureCompleter::complete`.
pub struct ManualFuture<T: Unpin> {
    state: BiLock<State<T>>,
}

/// Used to complete a `ManualFuture` so it can be resolved to a given value.
///
/// Dropping a `ManualFutureCompleter` will cause the associated `ManualFuture`
/// to never complete.
pub struct ManualFutureCompleter<T: Unpin> {
    state: BiLock<State<T>>,
}

impl<T: Unpin> ManualFutureCompleter<T> {
    /// Complete the `ManualFuture` associated with this
    ///
    /// `ManualFutureCompleter`. To prevent cases where a `ManualFuture` is
    /// completed twice, this takes the completer by value.
    pub async fn complete(self, value: T) {
        let mut state = self.state.lock().await;

        match std::mem::replace(&mut *state, State::Complete(Some(value))) {
            State::Incomplete => {}
            State::Waiting(w) => w.wake(),
            _ => panic!("future already completed"),
        }
    }
}

impl<T: Unpin> ManualFuture<T> {
    /// Create a new `ManualFuture` which will be resolved once the associated
    /// `ManualFutureCompleter` is used to set the value.
    pub fn new() -> (Self, ManualFutureCompleter<T>) {
        let (a, b) = BiLock::new(State::new(None));
        (Self { state: a }, ManualFutureCompleter { state: b })
    }

    /// Create a new `ManualFuture` which has already been completed with the
    /// given value.
    ///
    /// Because the `ManualFuture` is already completed, a
    /// `ManualFutureCompleter` won't be returned.
    pub fn new_completed(value: T) -> Self {
        let (state, _) = BiLock::new(State::new(Some(value)));
        Self { state }
    }
}

impl<T: Unpin> Future for ManualFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let mut state = match self.state.poll_lock(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(v) => v,
        };

        match &mut *state {
            s @ State::Incomplete => *s = State::Waiting(cx.waker().clone()),
            State::Waiting(w) if w.will_wake(cx.waker()) => {}
            s @ State::Waiting(_) => *s = State::Waiting(cx.waker().clone()),
            State::Complete(v) => match v.take() {
                Some(v) => return Poll::Ready(v),
                None => panic!("future already polled to completion"),
            },
        }

        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use futures::future::join;
    use futures::future::ready;
    use futures::future::select;
    use futures::future::Either;
    use futures::pin_mut;
    use std::thread::sleep;
    use std::thread::spawn;
    use std::time::Duration;

    #[tokio::test]
    async fn test_not_completed() {
        let (future, _) = ManualFuture::<()>::new();
        pin_mut!(future);
        let timeout_future = ready(());
        let result = block_on(select(future, timeout_future));
        match result {
            Either::Left(_) => panic!("should not complete"),
            Either::Right(_) => {} // Expected timeout
        }
    }

    #[tokio::test]
    async fn test_manual_completed() {
        let (future, completer) = ManualFuture::<()>::new();
        assert_eq!(block_on(join(future, completer.complete(()))), ((), ()));
    }

    #[tokio::test]
    async fn test_pre_completed() {
        assert_eq!(block_on(ManualFuture::new_completed(())), ());
    }

    #[tokio::test]
    async fn test_threaded() {
        let (future, completer) = ManualFuture::<()>::new();

        let t1 = spawn(move || {
            assert_eq!(block_on(future), ());
        });

        let t2 = spawn(move || {
            sleep(Duration::from_millis(100));
            block_on(async {
                completer.complete(()).await;
            });
        });

        t1.join().unwrap();
        t2.join().unwrap();
    }
}