// Copyright 2015-2024 Swim Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use static_assertions::assert_obj_safe;
use std::fmt::Debug;
use std::marker::PhantomData;
use thiserror::Error;

/// An event handler that does nothing.
#[derive(Clone, Copy, Default, Debug)]
pub struct NoHandler;

/// Wraps a [`FnMut`] instance to use as an event handler.
#[derive(Clone, Copy, Default, Debug)]
pub struct FnMutHandler<F>(pub F);

/// Wraps a [`Fn`] instance to use as an event handler.
#[derive(Clone, Copy, Default, Debug)]
pub struct FnHandler<F>(pub F);

/// Wraps a [`Fn`] instance that borrows its input in an arbitrary way to use as an event handler.
/// For example, if an event produces a value of type [`String`], this type can write a closure
/// that will take `&str` as a parameter rather than `&String`, avoiding a double indirection.
pub struct BorrowHandler<F, B: ?Sized>(F, PhantomData<fn(B)>);

impl<F: Clone, B: ?Sized> Clone for BorrowHandler<F, B> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1)
    }
}

impl<F: Copy, B: ?Sized> Copy for BorrowHandler<F, B> {}

impl<F: Default, B: ?Sized> Default for BorrowHandler<F, B> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<F: Debug, B: ?Sized> Debug for BorrowHandler<F, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BorrowHandler").field(&self.0).finish()
    }
}

impl<F, B: ?Sized> BorrowHandler<F, B> {
    pub fn new(f: F) -> Self {
        BorrowHandler(f, PhantomData)
    }
}

impl<F, B: ?Sized> AsRef<F> for BorrowHandler<F, B> {
    fn as_ref(&self) -> &F {
        &self.0
    }
}

/// The context type passed to every call to [`HandlerAction::step`] that provides access to the
/// underlying. Some of the methods on this type are not intended for use in user supplied handler
/// implementations and so can only be used from this crate.
pub struct ActionContext<'a, Context> {
    data: &'a Context,
}

impl<'a, Context> ActionContext<'a, Context> {
    pub fn new(agent_context: &'a Context) -> Self {
        ActionContext {
            data: agent_context,
        }
    }
}

/// Trait to describe an action to be taken, within the context of an agent, when an event occurs.
/// This could be expressed using generators from the standard library after this feature is stabilized.
/// A handler instance can be used exactly once.
///
/// # Type Parameters
/// * `Context` - The context within which the handler executes. Typically, this will be a struct type where
///    each field is a lane of an agent.
pub trait HandlerAction<Context> {
    /// The result of executing the handler to completion.
    type Completion;

    /// Run one step of the handler. This can result in either the handler suspending execution, completing
    /// with a result or returning an error.
    ///
    /// # Arguments
    /// * `suspend` - Allows for futures to be suspended into the agent task. The future will result in another event handler
    ///    which will be executed by the agent task upon completion.
    /// * `context` - The execution context of the handler (providing access to the lanes of the agent).
    fn step(
        &mut self,
        action_context: &mut ActionContext<Context>,
        context: &Context,
    ) -> StepResult<Self::Completion>;
}

/// A [`HandlerAction`] that does not produce a result.
pub trait EventHandler<Context>: HandlerAction<Context, Completion = ()> {}

assert_obj_safe!(EventHandler<()>);

impl<Context, H> EventHandler<Context> for H where H: HandlerAction<Context, Completion = ()> {}

impl<'a, H, Context> HandlerAction<Context> for &'a mut H
where
    H: HandlerAction<Context>,
{
    type Completion = H::Completion;

    fn step(
        &mut self,
        action_context: &mut ActionContext<Context>,
        context: &Context,
    ) -> StepResult<Self::Completion> {
        (*self).step(action_context, context)
    }
}

impl<H: ?Sized, Context> HandlerAction<Context> for Box<H>
where
    H: HandlerAction<Context>,
{
    type Completion = H::Completion;

    fn step(
        &mut self,
        action_context: &mut ActionContext<Context>,
        context: &Context,
    ) -> StepResult<Self::Completion> {
        (**self).step(action_context, context)
    }
}

/// Error type for fallible [`HandlerAction`]s. A handler produces an error when a fatal problem occurs and it
/// cannot produce its result. In most cases this will result in the agent terminating.
#[derive(Debug, Error)]
pub enum EventHandlerError {
    /// Handlers can only be used once. If a handler is stepped after it produces its value, this error will be raised.
    #[error("Event handler stepped after completion.")]
    SteppedAfterComplete,
    /// An incoming command message was incomplete and could not be deserialized.
    #[error("An incoming message was incomplete.")]
    IncompleteCommand,
    /// The `on_cue` lifecycle handler is mandatory for demand lanes. If it is not defined this error will be raised.
    #[error("The cue operation for a demand lane was undefined.")]
    DemandCueUndefined,
    /// If a GET request is made to a HTTP lane but it does not handle it, this error is raised. (This will not
    /// terminate the agent but will cause an error HTTP response to be sent).
    #[error("No GET handler was defined for an HTTP lane.")]
    HttpGetUndefined,
    /// The event handler has explicitly requested that the agent stop.
    #[error("The event handler has instructed the agent to stop.")]
    StopInstructed,
}

/// When a handler completes or suspends it can indicate that is has modified the
/// state of an item.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Modification {
    /// The ID of the item.
    pub(crate) item_id: u64,
}

impl Modification {
    pub(crate) fn trigger_only(item_id: u64) -> Self {
        Modification { item_id }
    }
}

/// The result of running a single step of an event handler.
#[derive(Debug)]
pub enum StepResult<C> {
    /// The event handler has suspended.
    Continue {
        /// Indicates if an item has been modified,
        modified_item: Option<Modification>,
    },
    /// The handler has failed and will never now produce a result.
    Fail(EventHandlerError),
    /// The handler has completed successfully. All further attempts to step
    /// will result in an error.
    Complete {
        /// Indicates if an item has been modified.
        modified_item: Option<Modification>,
        /// The result of the handler.
        result: C,
    },
}

impl<C> StepResult<C> {
    /// Create a result that produces a value. The handler should no longer be stepped after this.
    pub fn done(result: C) -> Self {
        Self::Complete {
            modified_item: None,
            result,
        }
    }

    /// Transform the result of this result (if it has one).
    pub fn map<F, D>(self, f: F) -> StepResult<D>
    where
        F: FnOnce(C) -> D,
    {
        match self {
            StepResult::Continue { modified_item } => StepResult::Continue { modified_item },
            StepResult::Fail(err) => StepResult::Fail(err),
            StepResult::Complete {
                modified_item,
                result,
            } => StepResult::Complete {
                modified_item,
                result: f(result),
            },
        }
    }
}

/// Type that is returned by the `map` method on the [`HandlerActionExt`] trait.
pub struct Map<H, F>(Option<(H, F)>);

impl<H, F> Default for Map<H, F> {
    fn default() -> Self {
        Map(None)
    }
}

/// An alternative to [`FnOnce`] that allows for named implementations.
#[doc(hidden)]
pub trait HandlerTrans<In> {
    type Out;
    fn transform(self, input: In) -> Self::Out;
}

impl<In, Out, F> HandlerTrans<In> for F
where
    F: FnOnce(In) -> Out,
{
    type Out = Out;

    fn transform(self, input: In) -> Self::Out {
        self(input)
    }
}

/// Transformation within a context.
#[doc(hidden)]
pub trait ContextualTransform<Context, In> {
    type Out;
    fn transform(self, context: &Context, input: In) -> Self::Out;
}

impl<Context, In, Out, F> ContextualTransform<Context, In> for F
where
    F: FnOnce(&Context, In) -> Out,
{
    type Out = Out;

    fn transform(self, context: &Context, input: In) -> Self::Out {
        self(context, input)
    }
}

impl<Context, H> HandlerAction<Context> for Option<H>
where
    H: HandlerAction<Context>,
{
    type Completion = Option<H::Completion>;

    fn step(
        &mut self,
        action_context: &mut ActionContext<Context>,
        context: &Context,
    ) -> StepResult<Self::Completion> {
        if let Some(inner) = self {
            inner.step(action_context, context).map(Option::Some)
        } else {
            StepResult::done(None)
        }
    }
}
