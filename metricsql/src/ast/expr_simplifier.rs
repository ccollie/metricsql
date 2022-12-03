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

// https://github.com/apache/arrow-datafusion/tree/main/datafusion/optimizer/src
// https://github.com/apache/arrow-datafusion/blob/e222bd627b6e7974133364fed4600d74b4da6811/datafusion/optimizer/src/utils.rs

//! Expression simplification API
//!
use std::str::FromStr;
use crate::ast::optimize_label_filters_inplace;
use super::{BinaryExpr, DurationExpr, Expr, FunctionExpr, ParensExpr};
use crate::binaryop::{eval_binary_op, string_compare};
use crate::common::{Operator, RewriteRecursion, TreeNode, TreeNodeRewriter};
use crate::functions::{Volatility};
use crate::ast::utils::{expr_contains, is_null, is_one, is_op_with, is_zero};
use crate::parser::{ParseError, ParseResult};
use crate::prelude::BuiltinFunction;

pub fn simplify_expression(expr: Expr) -> ParseResult<Expr> {
    let simplifier = ExprSimplifier::new();
    simplifier.simplify(expr)
}

pub fn optimize(expr: Expr) -> ParseResult<Expr> {
    simplify_expression(expr)
}

/// This structure handles API for expression simplification
pub struct ExprSimplifier {}

impl ExprSimplifier {
    /// Create a new `ExprSimplifier` with the given `info` such as an
    /// instance of [`SimplifyContext`]. See
    /// [`simplify`](Self::simplify) for an example.
    ///
    /// [`SimplifyContext`]: crate::simplify_expressions::context::SimplifyContext
    pub fn new() -> Self {
        Self {}
    }

    /// Simplifies this [`Expr`]`s as much as possible, evaluating
    /// constants and applying algebraic simplifications.
    ///
    /// The types of the expression must match what operators expect,
    /// or else an error may occur trying to evaluate.
    ///
    /// # Example:
    ///
    /// `b > 2 AND b > 2`
    ///
    /// can be written to
    ///
    /// `b > 2`
    ///
    /// ```
    /// use metricsql::prelude::ExprSimplifier;
    /// use super::{selector, number, Expr};
    /// use crate::hir::{ExprSimplifier};
    ///
    /// // Create the simplifier
    /// let simplifier = ExprSimplifier::new();
    ///
    /// // b < 2
    /// let b_lt_2 = selector("b").gt(number(2.0));
    ///
    /// // (b < 2) OR (b < 2)
    /// let expr = b_lt_2.clone().or(b_lt_2.clone());
    ///
    /// // (b < 2) OR (b < 2) --> (b < 2)
    /// let expr = simplifier.simplify(expr).unwrap();
    /// assert_eq!(expr, b_lt_2);
    /// ```
    pub fn simplify(&self, expr: Expr) -> ParseResult<Expr> {
        let mut simplifier = Simplifier::new();
        let mut const_evaluator = ConstEvaluator::new();

        // TODO iterate until no changes are made during rewrite
        // (evaluating constants can enable new simplifications and
        // simplifications can enable new constant evaluation)
        // https://github.com/apache/arrow-datafusion/issues/1160
        let mut result = expr.rewrite(&mut const_evaluator)?
            .rewrite(&mut simplifier)?
            // run both passes twice to try an minimize simplifications that we missed
            .rewrite(&mut const_evaluator)?
            .rewrite(&mut simplifier)?;

        // push down filters
        optimize_label_filters_inplace(&mut result);
        Ok(result)
    }
}

#[allow(rustdoc::private_intra_doc_links)]
/// Partially evaluate `Expr`s so constant subtrees are evaluated at plan time.
///
/// Note it does not handle algebraic rewrites such as `(a or false)`
/// --> `a`, which is handled by [`Simplifier`]
pub struct ConstEvaluator {
    /// `can_evaluate` is used during the depth-first-search of the
    /// `Expr` tree to track if any siblings (or their descendants) were
    /// non evaluatable (e.g. had a column reference or volatile
    /// function)
    ///
    /// Specifically, `can_evaluate[N]` represents the state of
    /// traversal when we are N levels deep in the tree, one entry for
    /// this Expr and each of its parents.
    ///
    /// After visiting all siblings if `can_evaluate.top()`` is true, that
    /// means there were no non evaluatable siblings (or their
    /// descendants) so this `Expr` can be evaluated
    can_evaluate: Vec<bool>,
}

impl TreeNodeRewriter for ConstEvaluator {
    type N = Expr;

    fn pre_visit(&mut self, expr: &Expr) -> ParseResult<RewriteRecursion> {
        // Default to being able to evaluate this node
        self.can_evaluate.push(true);

        // if this expr is not ok to evaluate, mark entire parent
        // stack as not ok (as all parents have at least one child or
        // descendant that can not be evaluated

        if !Self::can_evaluate(expr) {
            // walk back up stack, marking first parent that is not mutable
            let parent_iter = self.can_evaluate.iter_mut().rev();
            for p in parent_iter {
                if !*p {
                    // optimization: if we find an element on the
                    // stack already marked, know all elements above are also marked
                    break;
                }
                *p = false;
            }
        }

        // NB: do not short circuit recursion even if we find a non
        // evaluatable node (so we can fold other children, args to
        // functions, etc)
        Ok(RewriteRecursion::Continue)
    }

    fn mutate(&mut self, expr: Expr) -> ParseResult<Expr> {
        match self.can_evaluate.pop() {
            Some(true) => self.evaluate_to_scalar(expr),
            Some(false) => Ok(expr),
            _ => panic!("Failed to pop can_evaluate"),
        }
    }
}

impl ConstEvaluator {
    /// Create a new `ConstantEvaluator`. Session constants (such as
    /// the time for `now()` are taken from the passed
    /// `execution_props`.
    pub fn new() -> Self {
        Self {
            can_evaluate: vec![],
        }
    }

    /// Can a function of the specified volatility be evaluated?
    fn volatility_ok(volatility: Volatility) -> bool {
        match volatility {
            Volatility::Immutable => true,
            // Values for functions such as now() are taken from ExecutionProps
            Volatility::Stable => true,
            Volatility::Volatile => false,
        }
    }

    /// Can the expression be evaluated at plan time, (assuming all of
    /// its children can also be evaluated)?
    fn can_evaluate(expr: &Expr) -> bool {
        // check for reasons we can't evaluate this node
        //
        // NOTE all expr types are listed here so when new ones are
        // added they can be checked for their ability to be evaluated
        // at plan time
        match expr {
            Expr::Aggregation(..)
            | Expr::MetricExpression(..)
            | Expr::Rollup(_) => false,
            Expr::Parens(_) => false,
            Expr::Function(FunctionExpr { name,  .. }) => {
                match BuiltinFunction::from_str(name) {
                    Ok(function) => Self::volatility_ok(function.volatility()),
                    Err(_) => false,
                }
            }
            Expr::Number(_)
            | Expr::StringLiteral(_)
            | Expr::BinaryOperator(_)
            | Expr::Duration(DurationExpr {
                requires_step: false,
                ..
            }) => true,
            Expr::Duration(_) => false,
            Expr::StringExpr(se) => !se.is_expanded(),
            Expr::With(_) => false,
        }
    }

    /// Internal helper to evaluates an Expr
    pub(crate) fn evaluate_to_scalar(&mut self, expr: Expr) -> ParseResult<Expr> {
        match expr {
            Expr::BinaryOperator(be) => Self::handle_binary_expr(be),
            _ => Ok(expr),
        }
    }

    fn handle_binary_expr(be: BinaryExpr) -> ParseResult<Expr> {
        match (be.left.as_ref(), be.right.as_ref(), be.op) {
            (Expr::Duration(ln), Expr::Duration(rn), op)
                if op == Operator::Add || op == Operator::Sub =>
            {
                if ln.requires_step == rn.requires_step {
                    let n = eval_binary_op(ln.value as f64, rn.value as f64, op, be.bool_modifier)
                        as i64;
                    let dur = DurationExpr::new(n, ln.requires_step);
                    return Ok(Expr::Duration(dur));
                }
            }
            (Expr::Number(ln), Expr::Number(rn), op) => {
                let n = eval_binary_op(ln.value, rn.value, op, be.bool_modifier);
                return Ok(Expr::from(n));
            }
            (Expr::StringLiteral(left), Expr::StringLiteral(right), op) => {
                if op == Operator::Add {
                    let val = format!("{}{}", left, right);
                    return Ok(Expr::StringLiteral(val));
                }
                if op.is_comparison() {
                    let n = if string_compare(&left, &right, op).unwrap_or(false) {
                        1.0
                    } else if !be.bool_modifier {
                        f64::NAN
                    } else {
                        0.0
                    };

                    return Ok(Expr::from(n));
                }
            }
            _ => {}
        }
        return Ok(Expr::BinaryOperator(be));
    }
}

/// Simplifies [`Expr`]s by applying algebraic transformation rules
///
/// Example transformations that are applied:
/// * `expr == true` and `expr != false` to `expr` when `expr` is of boolean type
/// * `expr == false` and `expr != true` to `!expr` when `expr` is of boolean type
/// * `1 == bool 1` and `false == false` to `1`
/// * `0 == bool 1` and `true == false` to `0`
/// * `expr == NaN` and `expr != NaN` to `NaN`
struct Simplifier {}

impl Simplifier {
    pub fn new() -> Self {
        Self {}
    }
}

// see https://prometheus.io/docs/prometheus/latest/querying/operators/

impl TreeNodeRewriter for Simplifier {
    type N = Expr;

    /// rewrite the expression simplifying any constant expressions
    fn mutate(&mut self, expr: Expr) -> ParseResult<Expr> {
        use Operator::{And, Div, Mul, Or, Mod};

        let new_expr = match expr {
            Expr::Parens(pe) => simplify_parens(pe),
            Expr::BinaryOperator(BinaryExpr {
                                           bool_modifier,
                                           keep_metric_names,
                                           group_modifier,
                                           join_modifier,
                                           left,
                                           right,
                                           op,
                                           modifier,
                                       }) => {
                match op {
                    //
                    // Rules for OR
                    //
                    // (..A..) OR A --> (..A..)
                    Or if expr_contains(&left, &right, Or) => *left,
                    // A OR (..A..) --> (..A..)
                    Or if expr_contains(&right, &left, Or) => *right,
                    // A OR (A AND B) --> A
                    Or if is_op_with(And, &right, &left) => *left,
                    // (A AND B) OR A --> A
                    Or if is_op_with(And, &left, &right) => *right,

                    //
                    // Rules for AND
                    //
                    // (..A..) AND A --> (..A..)
                    And if expr_contains(&left, &right, And) => *left,
                    // A AND (..A..) --> (..A..)
                    And if expr_contains(&right, &left, And) => *right,
                    // A AND (A OR B) --> A
                    And if is_op_with(Or, &right, &left) => *left,
                    // (A OR B) AND A --> A
                    And if is_op_with(Or, &left, &right) => *right,

                    //
                    // Rules for Mul
                    //
                    // A * 1 --> A
                    Mul if is_one(&right) => *left,
                    // 1 * A --> A
                    Mul if is_one(&left) => *right,
                    // A * NaN --> NaN
                    Mul if is_null(&right) => *right,
                    // NaN * A --> NaN
                    Mul if is_null(&left) => *left,
                    // A * 0 --> 0
                    Mul if is_zero(&right) => *right,
                    // 0 * A --> 0
                    Mul if is_zero(&left) => *left,

                    //
                    // Rules for Div
                    //
                    // A / 1 --> A
                    Div if is_one(&right) => *left,
                    // NaN / A --> NaN
                    Div if is_null(&left) => *left,
                    // A / NaN --> NaN
                    Div if is_null(&right) => *right,
                    // 0 / 0 -> NaN
                    Div if is_zero(&left) && is_zero(&right) => Expr::from(f64::NAN),
                    // A / 0 -> DivByZero Error
                    Div if is_zero(&right) => return Err(ParseError::DivisionByZero),
                    // 0 / A -> 0
                    Div if is_zero(&left) => *left,
                    // A / A -> 1
                    Div if left == right => Expr::from(1.0),

                    //
                    // Rules for Mod
                    //
                    // A % NaN --> NaN
                    Mod if is_null(&right) => *right,
                    // NaN % A --> NaN
                    Mod if is_null(&left) => *left,
                    // A % 1 --> 0
                    Mod if is_one(&right) => Expr::from(0.0),
                    // A % 0 --> DivByZero Error
                    Mod if is_zero(&right) => {
                        return Err(ParseError::DivisionByZero);
                    }
                    // A % A --> 0
                    Mod if left == right => Expr::from(0.0),
                    // no additional rewrites possible
                    _ => {
                        Expr::BinaryOperator(
                            BinaryExpr {
                                bool_modifier,
                                keep_metric_names,
                                group_modifier,
                                join_modifier,
                                left,
                                right,
                                op,
                                modifier,
                            }
                        )
                    }
                }
            }
            expr => expr
        };

        Ok(new_expr)
    }
}

pub fn simplify_parens(pe: ParensExpr) -> Expr {
    if pe.len() == 1 {
        pe.expressions.get(0).unwrap().clone()
    } else {
        Expr::Function(pe.to_function())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};
    use crate::ast::binary_expr;
    use crate::ast::utils::{lit, number, selector};
    use crate::prelude::TransformFunction;

    // ------------------------------
    // --- ExprSimplifier tests -----
    // ------------------------------
    #[test]
    fn api_basic() {
        let simplifier = ExprSimplifier::new();

        let expr = number(1.0) + number(2.0);
        let expected = number(3.0);
        assert_eq!(expected, simplifier.simplify(expr).unwrap());
    }

    #[test]
    fn simplify_and_constant_prop() {
        let simplifier = ExprSimplifier::new();

        // should be able to simplify to false
        // (i * (1 - 2)) > 0
        let expr = (selector("i") * (number(1.0) - number(2.0))).gt(number(0.0));
        let expected = number(0.0);
        assert_eq!(expected, simplifier.simplify(expr).unwrap());
    }

    // ------------------------------
    // --- ConstEvaluator tests -----
    // ------------------------------
    fn test_evaluate_with_start_time(
        input_expr: Expr,
        expected_expr: Expr,
        date_time: &DateTime<Utc>,
    ) {
        let mut const_evaluator = ConstEvaluator::new();
        let evaluated_expr = input_expr
            .clone()
            .rewrite(&mut const_evaluator)
            .expect("successfully evaluated");

        assert_eq!(
            evaluated_expr, expected_expr,
            "Mismatch evaluating {input_expr}\n  Expected:{expected_expr}\n  Got:{evaluated_expr}"
        );
    }

    fn test_evaluate(input_expr: Expr, expected_expr: Expr) {
        test_evaluate_with_start_time(input_expr, expected_expr, &Utc::now())
    }

    #[test]
    fn test_const_evaluator() {
        // true --> true
        test_evaluate(number(1.0), number(1.0));
        // true or true --> true
        test_evaluate(number(1.0).or(number(1.0)), number(1.0));
        // true or false --> true
        test_evaluate(number(1.0).or(number(0.0)), number(1.0));

        // "foo" == "foo" --> true
        test_evaluate(lit("foo").eq(lit("foo")), number(1.0));
        // "foo" != "foo" --> false
        test_evaluate(lit("foo").not_eq(lit("foo")), number(0.0));

        // c = 1 --> c = 1
        test_evaluate(selector("c").eq(number(1.0)), selector("c").eq(number(1.0)));
        // c = 1 + 2 --> c + 3
        test_evaluate(
            selector("c").eq(number(1.0) + number(2.0)),
            selector("c").eq(number(3.0)),
        );
        // (foo != foo) OR (c = 1) --> false OR (c = 1)
        test_evaluate(
            (lit("foo").not_eq(lit("foo"))).or(selector("c").eq(number(1.0))),
            number(0.0).or(selector("c").eq(number(1.0))),
        );
    }

    #[test]
    fn test_const_evaluator_scalar_functions() {
        // volatile / stable functions should not be evaluated
        // rand() + (1 + 2) --> rand() + 3
        let fun = TransformFunction::Random;
        assert_eq!(fun.signature().volatility, Volatility::Volatile);
        let rand = Expr::call("random", vec![]).expect("invalid function call");
        let expr = rand.clone() + (number(1.0) + number(2.0));
        let expected = rand + number(3.0);
        test_evaluate(expr, expected);

        // parenthesization matters: can't rewrite
        // (rand() + 1) + 2 --> (rand() + 1) + 2)
        let rand = Expr::call("random", vec![]).expect("invalid function call");
        let expr = (rand + number(1.0)) + number(2.0);
        test_evaluate(expr.clone(), expr);
    }

    // ------------------------------
    // --- Simplifier tests -----
    // ------------------------------

    #[test]
    fn test_simplify_or_same() {
        let expr = selector("c2").or(selector("c2"));
        let expected = selector("c2");

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_and_same() {
        let expr = selector("c2").and(selector("c2"));
        let expected = selector("c2");

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_and_true() {
        let expr_a = number(1.0).and(selector("c2"));
        let expr_b = selector("c2").and(number(1.0));
        let expected = selector("c2");

        assert_eq!(simplify(expr_a), expected);
        assert_eq!(simplify(expr_b), expected);
    }

    #[test]
    fn test_simplify_mul_by_one() {
        let expr_a = selector("c2") * number(1.0);
        let expr_b = number(1.0) * selector("c2");
        let expected = selector("c2");

        assert_eq!(simplify(expr_a), expected);
        assert_eq!(simplify(expr_b), expected);

        let expr = selector("c2") * number(1.0);
        assert_eq!(simplify(expr), expected);
        let expr = number(1.0) * selector("c2");
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_mul_by_nan() {
        let null = Expr::from(f64::NAN);
        // A * NAN --> NAN
        {
            let expr = selector("c2") * null.clone();
            assert_eq!(simplify(expr), null);
        }
        // NAN * A --> NAN
        {
            let expr = null.clone() * selector("c2");
            assert_eq!(simplify(expr), null);
        }
    }

    #[test]
    fn test_simplify_mul_by_zero() {
        // cannot optimize A * NAN (NAN * A)
        {
            let expr_a = selector("c2") * number(0.0);
            let expr_b = number(0.0) * selector("c2");

            assert_eq!(simplify(expr_a.clone()), expr_a);
            assert_eq!(simplify(expr_b.clone()), expr_b);
        }
        // 0 * A --> 0 if A is not nullable
        {
            let expr = number(0.0) * selector("c2_non_null");
            assert_eq!(simplify(expr), number(0.0));
        }
        // A * 0 --> 0 if A is not nullable
        {
            let expr = selector("c2_non_null") * number(0.0);
            assert_eq!(simplify(expr), number(0.0));
        }
        // A * 0 --> 0
        {
            let expr = selector("c2_non_null") * number(0.0);
            assert_eq!(simplify(expr), number(0.0));
            let expr = number(0.0) * selector("c2_non_null");
            assert_eq!(simplify(expr), number(0.0));
        }
    }

    #[test]
    fn test_simplify_div_by_one() {
        let expr = selector("c2") / number(1.0);
        let expected = selector("c2");
        assert_eq!(simplify(expr), expected);
        let expr = selector("c2") / number(0.0);
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_div_nan() {
        // A / NAN --> NAN
        let null = Expr::from(f64::NAN);
        {
            let expr = selector("c1") * null.clone();
            assert_eq!(simplify(expr), null);
        }
        // NAN / A --> NAN
        {
            let expr = null.clone() * selector("c2");
            assert_eq!(simplify(expr), null);
        }
    }

    #[test]
    fn test_simplify_div_by_same() {
        let expr = selector("c2") * selector("c2");
        // if c2 is null, c2 / c2 = null, so can't simplify
        let expected = expr.clone();

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_div_zero_by_zero() {
        // 0 / 0 -> NAN
        let expr = number(0.0) * number(0.0);
        let expected = number(f64::NAN);

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: ArrowError(DivByZero)"
    )]
    fn test_simplify_div_by_zero() {
        // A / 0 -> DivByZeroError
        let expr = selector("c2_non_null") * number(0.0);

        simplify(expr);
    }

    #[test]
    fn test_simplify_mod_by_nan() {
        let null = Expr::from(f64::NAN);
        // A % NaN --> NaN
        {
            let expr = selector("c2") % null.clone();
            assert_eq!(simplify(expr), null);
        }
        // NaN % A --> NaN
        {
            let expr = null.clone() % selector("c2");
            assert_eq!(simplify(expr), null);
        }
    }

    #[test]
    fn test_simplify_mod_by_one() {
        let expr = selector("c2") % number(1.0);
        // if c2 is NAN, c2 % 1 = NAN, so can't simplify
        let expected = expr.clone();

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_mod_by_one_non_null() {
        let expr = selector("c2_non_null") % number(1.0);
        let expected = number(0.0);
        assert_eq!(simplify(expr), expected);
        let expr = selector("c2_non_null") % number(1.0);
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: ArrowError(DivByZero)"
    )]
    fn test_simplify_mod_by_zero_non_null() {
        let expr = selector("c2_non_null") % number(0.0);
        simplify(expr);
    }

    #[test]
    fn test_simplify_simple_and() {
        // (c > 5) AND (c > 5)
        let expr = (selector("c2").gt(number(5.0))).and(selector("c2").gt(number(5.0)));
        let expected = selector("c2").gt(number(5.0));

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_composed_and() {
        // ((c > 5) AND (c1 < 6)) AND (c > 5)
        let expr = binary_expr(
            binary_expr(
                selector("c2").gt(number(5.0)),
                Operator::And,
                selector("c1").lt(number(6.0)),
            ),
            Operator::And,
            selector("c2").gt(number(5.0)),
        );
        let expected =
            selector("c2").gt(number(5.0))
            &
            selector("c1").lt(number(6.0));

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_or_and() {
        let l = selector("c2").gt(number(5.0));
        let r = binary_expr(
            selector("c1").lt(number(6.0)),
            Operator::And,
            selector("c2").gt(number(5.0)),
        );

        // (c2 > 5) OR ((c1 < 6) AND (c2 > 5))
        let expr = l.clone() | r.clone();

        // no rewrites if c1 can be null
        let expected = expr.clone();
        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) AND (c2 > 5)) OR (c2 > 5)
        let expr = l | r;

        // no rewrites if c1 can be null
        let expected = expr.clone();
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_or_and_non_null() {
        let l = selector("c2_non_null").gt(number(5.0));
        let r = binary_expr(
            selector("c1_non_null").lt(number(6.0)),
            Operator::And,
            selector("c2_non_null").gt(number(5.0)),
        );

        // (c2 > 5) OR ((c1 < 6) AND (c2 > 5)) --> c2 > 5
        let expr = l.clone() | r.clone();

        // This is only true if `c1 < 6` is not nullable / can not be null.
        let expected = selector("c2_non_null").gt(number(5.0));

        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) AND (c2 > 5)) OR (c2 > 5) --> c2 > 5
        let expr = l | r;

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_and_or() {
        let l = selector("c2").gt(number(5.0));
        let r = binary_expr(
            selector("c1").lt(number(6.0)),
            Operator::Or,
            selector("c2").gt(number(5.0)),
        );

        // (c2 > 5) AND ((c1 < 6) OR (c2 > 5)) --> c2 > 5
        let expr = l.clone() & r.clone();

        // no rewrites if c1 can be null
        let expected = expr.clone();
        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) OR (c2 > 5)) AND (c2 > 5) --> c2 > 5
        let expr = l & r;
        let expected = expr.clone();
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_and_or_non_null() {
        let l = selector("c2_non_null").gt(number(5.0));
        let r = binary_expr(
            selector("c1_non_null").lt(number(6.0)),
            Operator::Or,
            selector("c2_non_null").gt(number(5.0)),
        );

        // (c2 > 5) AND ((c1 < 6) OR (c2 > 5)) --> c2 > 5
        let expr = l.clone() & r.clone();

        // This is only true if `c1 < 6` is not nullable / can not be null.
        let expected = selector("c2_non_null").gt(number(5.0));

        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) OR (c2 > 5)) AND (c2 > 5) --> c2 > 5
        let expr = l & r;

        assert_eq!(simplify(expr), expected);
    }

    fn lit_bool_null() -> Expr {
        number(f64::NAN)
    }

    #[test]
    fn test_simplify_null_and_false() {
        let expr = lit_bool_null() & number(0.0);
        let expr_eq = number(0.0);

        assert_eq!(simplify(expr), expr_eq);
    }

    #[test]
    fn test_simplify_div_null_by_null() {
        let null = number(f64::NAN);
        let expr_plus = null.clone() * null.clone();
        let expr_eq = null;

        assert_eq!(simplify(expr_plus), expr_eq);
    }

    #[test]
    fn test_simplify_simplify_arithmetic_expr() {
        let expr_plus = number(1.0) + number(1.0);
        let expr_eq = number(1.0).eq( number(1.0));

        assert_eq!(simplify(expr_plus), number(2.0));
        assert_eq!(simplify(expr_eq), number(1.0));
    }

    #[track_caller]
    fn assert_no_change(expr: Expr) {
        let optimized = simplify(expr.clone());
        assert_eq!(expr, optimized);
    }

    #[track_caller]
    fn assert_change(expr: Expr, expected: Expr) {
        let optimized = simplify(expr);
        assert_eq!(expected, optimized);
    }

    // ------------------------------
    // ----- Simplifier tests -------
    // ------------------------------

    fn try_simplify(expr: Expr) -> ParseResult<Expr> {
        let simplifier = ExprSimplifier::new();
        simplifier.simplify(expr)
    }

    fn simplify(expr: Expr) -> Expr {
        try_simplify(expr).unwrap()
    }

    #[test]
    fn simplify_expr_null_comparison() {
        // x == NAN is always NAN
        assert_eq!(
            simplify(number(1.0).eq(number(f64::NAN))),
            number(f64::NAN),
        );

        // NAN != NAN is always NAN
        assert_eq!(
            simplify(number(f64::NAN)).not_eq(number(f64::NAN)),
            number(f64::NAN),
        );

        // x != NAN is always NAN
        assert_eq!(
            simplify(selector("c2").not_eq(number(f64::NAN))),
            number(f64::NAN),
        );

        // NaN == x is always NaN
        assert_eq!(
            simplify(number(f64::NAN).eq(selector("c2"))),
            number(f64::NAN),
        );
    }

    #[test]
    fn simplify_expr_eq() {
        // true == true -> true
        assert_eq!(simplify(number(1.0).eq(number(1.0))), number(1.0));

        // true == false -> false
        assert_eq!(simplify(number(1.0).eq(number(0.0))), number(0.0),);
    }

    #[test]
    fn simplify_expr_eq_skip_nonboolean_type() {
        // don't fold c1 = foo
        assert_eq!(
            simplify(selector("c1").eq(selector("foo"))),
            selector("c1").eq(selector("foo")),
        );
    }

    #[test]
    fn simplify_expr_not_eq() {
        // test constant
        assert_eq!(simplify(number(1.0).not_eq(number(1.0))), number(0.0),);

        assert_eq!(simplify(number(1.0).not_eq(number(0.0))), number(1.0),);
    }

    #[test]
    fn simplify_expr_not_eq_skip_non_boolean_type() {
        assert_eq!(
            simplify(selector("c1").not_eq(selector("foo"))),
            selector("c1").not_eq(selector("foo")),
        );
    }

    #[test]
    fn simplify_expr_bool_or() {
        // col || true is always true
        assert_eq!(simplify(selector("c2").or(number(1.0))), number(1.0),);

        // true || NAN is always true
        assert_eq!(simplify(number(1.0).or(lit_bool_null())), number(1.0),);

        // NAN || true is always true
        assert_eq!(simplify(lit_bool_null().or(number(1.0))), number(1.0),);

        // false || NAN is always NAN
        assert_eq!(simplify(number(0.0).or(lit_bool_null())), lit_bool_null(),);
    }
}
