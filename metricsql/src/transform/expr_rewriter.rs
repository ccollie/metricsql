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

//! Expression rewriter

use crate::ast::{
    AggrFuncExpr,
    BExpression,
    BinaryOpExpr,
    Expression,
    FuncExpr,
    ParensExpr,
    RollupExpr,
    WithArgExpr,
    WithExpr
};
use crate::parser::ParseResult;


/// Controls how the [ExprRewriter] recursion should proceed.
pub enum RewriteRecursion {
    /// Continue rewrite / visit this expression.
    Continue,
    /// Call [ExprRewriter::mutate()] immediately and return.
    Mutate,
    /// Do not rewrite / visit the children of this expression.
    Stop,
    /// Keep recursive but skip mutate on this expression
    Skip,
}

/// Trait for potentially recursively rewriting an [`Expression`] expression
/// tree. When passed to `Expression::rewrite`, `ExpressionVisitor::mutate` is
/// invoked recursively on all nodes of an expression tree. See the
/// comments on `Expression::rewrite` for details on its use
pub trait ExprRewriter<E: ExprRewritable = Expression>: Sized {
    /// Invoked before any children of `expr` are rewritten /
    /// visited. Default implementation returns `Ok(RewriteRecursion::Continue)`
    fn pre_visit(&mut self, _expr: &E) -> ParseResult<RewriteRecursion> {
        Ok(RewriteRecursion::Continue)
    }

    /// Invoked after all children of `expr` have been mutated and
    /// returns a potentially modified expr.
    fn mutate(&mut self, expr: E) -> ParseResult<E>;
}

/// a trait for marking types that are rewritable by [ExprRewriter]
pub trait ExprRewritable: Sized {
    /// rewrite the expression tree using the given [ExprRewriter]
    fn rewrite<R: ExprRewriter<Self>>(self, rewriter: &mut R) -> ParseResult<Self>;
}

impl ExprRewritable for Expression {
    /// Performs a depth first walk of an expression and its children
    /// to rewrite an expression, consuming `self` producing a new
    /// [`Expression`].
    ///
    /// Implements a modified version of the [visitor
    /// pattern](https://en.wikipedia.org/wiki/Visitor_pattern) to
    /// separate algorithms from the structure of the `Expression` tree and
    /// make it easier to write new, efficient expression
    /// transformation algorithms.
    ///
    /// For an expression tree such as
    /// ```text
    /// BinaryExpr (GT)
    ///    left: Number(1024)
    ///    right: MetricExpression("bar")
    /// ```
    ///
    /// The nodes are visited using the following order
    /// ```text
    /// pre_visit(BinaryExpr(GT))
    /// pre_visit(Number(1024))
    /// mutate(Number(1024))
    /// pre_visit(MetricExpression("bar"))
    /// mutate(MetricExpression("bar"))
    /// mutate(BinaryExpr(GT))
    /// ```
    ///
    /// If an Err result is returned, recursion is stopped immediately
    ///
    /// If [`false`] is returned on a call to pre_visit, no
    /// children of that expression are visited, nor is mutate
    /// called on that expression
    ///
    fn rewrite<R>(self, rewriter: &mut R) -> ParseResult<Self>
        where
            R: ExprRewriter<Self>,
    {
        let need_mutate = match rewriter.pre_visit(&self)? {
            RewriteRecursion::Mutate => return rewriter.mutate(self),
            RewriteRecursion::Stop => return Ok(self),
            RewriteRecursion::Continue => true,
            RewriteRecursion::Skip => false,
        };

        // recurse into all sub expressions(and cover all expression types)
        let expr = match self {
            Expression::Duration(_) |
            Expression::Number(_) |
            Expression::String(_) |
            Expression::MetricExpression(_) => self.clone(),
            Expression::Parens(ParensExpr { expressions, span }) => {
                Expression::Parens(
                    ParensExpr {
                        expressions: rewrite_vec(expressions, rewriter)?,
                        span
                    }
                )
            },
            Expression::Function(FuncExpr {
                                     function,
                                     name,
                                     args,
                                     keep_metric_names,
                                     is_scalar,
                                     span,
                                     with_name, }) => {

                Expression::Function(
                    FuncExpr {
                        function: function.clone(),
                        name,
                        args: rewrite_vec(args, rewriter)?,
                        keep_metric_names,
                        is_scalar,
                        span,
                        with_name
                    }
                )
            },
            Expression::Aggregation(AggrFuncExpr {
                                        function,
                                        name,
                                        args,
                                        modifier,
                                        limit,
                                        keep_metric_names,
                                        span }) => {

                Expression::Aggregation(
                    AggrFuncExpr {
                        function,
                        name,
                        args: rewrite_vec(args, rewriter)?,
                        modifier,
                        limit,
                        keep_metric_names,
                        span
                    }
                )

            },
            Expression::Rollup(RollupExpr {
                                   expr,
                                   window,
                                   step,
                                   offset,
                                   inherit_step,
                                   at,
                                   span
                               }) => {

                Expression::Rollup(
                    RollupExpr {
                        expr: rewrite_boxed(expr, rewriter)?,
                        window,
                        step,
                        offset,
                        inherit_step,
                        at: rewrite_option_box(at, rewriter)?,
                        span
                    }
                )

            },
            Expression::BinaryOperator(BinaryOpExpr {
                                           group_modifier,
                                           join_modifier,
                                           left,
                                           op,
                                           right,
                                           bool_modifier,
                                           span, }) => {
                Expression::BinaryOperator(
                    BinaryOpExpr {
                        left: rewrite_boxed(left, rewriter)?,
                        op,
                        bool_modifier,
                        group_modifier,
                        right: rewrite_boxed(right, rewriter)?,
                        join_modifier,
                        span
                    }
                )
            },
            Expression::With(WithExpr{ was, expr, span }) => {
                let mut was_new: Vec<WithArgExpr> = Vec::with_capacity(was.len());
                for wae in was.iter() {
                    let WithArgExpr{ name, expr, args} = wae;
                    was_new.push(
                        WithArgExpr {
                                name: name.clone(),
                                args: args.clone(),
                                expr: rewrite_boxed(expr.clone(), rewriter)?
                            }
                    );
                }

                Expression::With(
                    WithExpr {
                        was: was_new,
                        expr: rewrite_boxed(expr, rewriter)?,
                        span
                    }
                )
            }
        };

        // now rewrite this expression itself
        if need_mutate {
            rewriter.mutate(expr)
        } else {
            Ok(expr)
        }
    }
}

#[allow(clippy::boxed_local)]
fn rewrite_boxed<R>(boxed_expr: Box<Expression>, rewriter: &mut R) -> ParseResult<Box<Expression>>
    where
        R: ExprRewriter,
{
    // TODO: It might be possible to avoid an allocation (the
    // Box::new) below by reusing the box.
    let expr: Expression = *boxed_expr;
    let rewritten_expr = expr.rewrite(rewriter)?;
    Ok(Box::new(rewritten_expr))
}

fn rewrite_option_box<R>(
    option_box: Option<Box<Expression>>,
    rewriter: &mut R,
) -> ParseResult<Option<Box<Expression>>>
    where
        R: ExprRewriter,
{
    option_box
        .map(|expr| rewrite_boxed(expr, rewriter))
        .transpose()
}

/// rewrite a `Vec` of `Expression`s with the rewriter
fn rewrite_vec<R>(v: Vec<BExpression>, rewriter: &mut R) -> ParseResult<Vec<BExpression>>
    where
        R: ExprRewriter,
{
    let mut res: Vec<BExpression> = Vec::with_capacity(v.len());
    for expr in v.iter() {
        // hacky. Since the vec is of references, we need to clone. Can we avoid this ?
        let cloned = expr.clone();
        res.push( Box::new( cloned.rewrite(rewriter)? ))
    }
    Ok(res)
}
