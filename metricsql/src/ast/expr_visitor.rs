// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Expression visitor

use crate::ast::{AggrFuncExpr, BinaryOpExpr, DurationExpr, Expression, FuncExpr, ParensExpr, RollupExpr, WithExpr};
use crate::parser::ParseResult;

/// Controls how the visitor recursion should proceed.
pub enum Recursion<V: ExpressionVisitor> {
    /// Attempt to visit all the children, recursively, of this expression.
    Continue(V),
    /// Do not visit the children of this expression, though the walk
    /// of parents of this expression will not be affected
    Stop(V),
}

/// Encode the traversal of an expression tree. When passed to
/// `Expression::accept`, `ExpressionVisitor::visit` is invoked
/// recursively on all nodes of an expression tree. See the comments
/// on `Expression::accept` for details on its use
pub trait ExpressionVisitor<E: ExprVisitable = Expression>: Sized {
    /// Invoked before any children of `expr` are visited.
    fn pre_visit(self, expr: &E) -> ParseResult<Recursion<Self>>
        where
            Self: ExpressionVisitor;

    /// Invoked after all children of `expr` are visited. Default
    /// implementation does nothing.
    fn post_visit(self, _expr: &E) -> ParseResult<Self> {
        Ok(self)
    }
}

/// trait for types that can be visited by [`ExpressionVisitor`]
pub trait ExprVisitable: Sized {
    /// accept a visitor, calling `visit` on all children of this
    fn accept<V: ExpressionVisitor<Self>>(&self, visitor: V) -> ParseResult<V>;
}

impl ExprVisitable for Expression {
    /// Performs a depth first walk of an expression and
    /// its children, calling [`ExpressionVisitor::pre_visit`] and
    /// `visitor.post_visit`.
    ///
    /// Implements the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern) to
    /// separate expression algorithms from the structure of the
    /// `Expression` tree and make it easier to add new types of expressions
    /// and algorithms that walk the tree.
    ///
    /// For an expression tree such as
    /// ```text
    /// BinaryExpr (GT)
    ///    left: String("foo")
    ///    right: String("bar")
    /// ```
    ///
    /// The nodes are visited using the following order
    /// ```text
    /// pre_visit(BinaryExpr(GT))
    /// pre_visit(String("foo"))
    /// post_visit(String("foo"))
    /// pre_visit(String("bar"))
    /// post_visit(String("bar"))
    /// post_visit(BinaryExpr(GT))
    /// ```
    ///
    /// If an Err result is returned, recursion is stopped immediately
    ///
    /// If `Recursion::Stop` is returned on a call to pre_visit, no
    /// children of that expression are visited, nor is post_visit
    /// called on that expression
    ///
    fn accept<V: ExpressionVisitor>(&self, visitor: V) -> ParseResult<V> {
        let mut visitor = match visitor.pre_visit(self)? {
            Recursion::Continue(visitor) => visitor,
            // If the recursion should stop, do not visit children
            Recursion::Stop(visitor) => return Ok(visitor),
        };

        // recurse (and cover all expression types)
        let visitor = match self {
            Expression::Number(_)
            | Expression::String(_)
            | Expression::Duration(_)
            | Expression::MetricExpression(_) => self.accept(visitor),
            Expression::Parens(ParensExpr { expressions, .. }) => expressions
                .iter()
                .fold(Ok(visitor), |v, e| v.and_then(|v| e.accept(v))),
            Expression::BinaryOperator(BinaryOpExpr { left, right, .. }) => {
                let visitor = left.accept(visitor)?;
                right.accept(visitor)
            },
            Expression::Function( FuncExpr {  args, .. } ) => args
                .iter()
                .try_fold(visitor, |visitor, arg| arg.accept(visitor)),
            Expression::Aggregation( AggrFuncExpr { args, .. }) => {
                args.iter()
                    .try_fold(visitor, |visitor, arg| arg.accept(visitor))
            }
            Expression::Rollup(RollupExpr {
                                   expr,
                                   window,
                                   step,
                                   offset,
                                   at, ..
                               }) => {

                visitor = expr.accept(visitor)?;
                if let Some(at) = at {
                    visitor = at.accept(visitor)?
                }
                visitor = accept_optional_duration(window, visitor)?;
                visitor = accept_optional_duration(step, visitor)?;
                accept_optional_duration(offset, visitor)
            }
            Expression::With(WithExpr{ was, expr, .. }) => {
                visitor = expr.accept(visitor)?;
                for arg in was.iter() {
                    visitor = arg.expr.accept(visitor)?
                }
                Ok(visitor)
            }
        }?;

        visitor.post_visit(self)
    }
}

fn accept_optional_duration<V: ExpressionVisitor>(e: &Option<DurationExpr>, visitor: V) -> ParseResult<V> {
    if let Some(duration) = e {
        // hacky !!
        return Expression::Duration(duration.clone()).accept(visitor)
    }
    Ok(visitor)
}