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

//! Expression simplification API

use super::utils::*;
use crate::ast::expr_rewriter::{ExprRewritable, ExprRewriter, RewriteRecursion};
use crate::ast::{BinaryExpr, Operator, DurationExpr, Expression, FuncExpr, NumberExpr};
use crate::functions::Volatility;
use crate::parser::{ParseError, ParseResult};
use crate::prelude::constant_fold_binary_expression;

/// This structure handles API for expression simplification
pub struct ExprSimplifier {
}

impl ExprSimplifier {
    /// Create a new `ExprSimplifier` with the given `info` such as an
    /// instance of [`SimplifyContext`]. See
    /// [`simplify`](Self::simplify) for an example.
    ///
    /// [`SimplifyContext`]: crate::simplify_expressions::context::SimplifyContext
    pub fn new() -> Self {
        Self { }
    }

    /// Simplifies this [`Expression`]`s as much as possible, evaluating
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
    /// use datafusion_expr::{selector, number, Expr};
    /// use datafusion_common::Result;
    /// use datafusion_optimizer::simplify_expressions::{ExprSimplifier};
    ///
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
    pub fn simplify(&self, expr: Expression) -> ParseResult<Expression> {
        let mut simplifier = Simplifier::new();
        let mut const_evaluator = ConstEvaluator::new();

        // TODO iterate until no changes are made during rewrite
        // (evaluating constants can enable new simplifications and
        // simplifications can enable new constant evaluation)
        // https://github.com/apache/arrow-datafusion/issues/1160
        expr.rewrite(&mut const_evaluator)?
            .rewrite(&mut simplifier)?
            // run both passes twice to try an minimize simplifications that we missed
            .rewrite(&mut const_evaluator)?
            .rewrite(&mut simplifier)
    }
}

#[allow(rustdoc::private_intra_doc_links)]
/// Partially evaluate `Expr`s so constant subtrees are evaluated at plan time.
///
/// Note it does not handle algebraic rewrites such as `(a or false)`
/// --> `a`, which is handled by [`Simplifier`]
struct ConstEvaluator {
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

impl ExprRewriter for ConstEvaluator {
    fn pre_visit(&mut self, expr: &Expression) -> ParseResult<RewriteRecursion> {
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
    
    fn mutate(&mut self, expr: Expression) -> ParseResult<Expression> {
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
        Self { can_evaluate: vec![], }
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
    fn can_evaluate(expr: &Expression) -> bool {
        // check for reasons we can't evaluate this node
        //
        // NOTE all expr types are listed here so when new ones are
        // added they can be checked for their ability to be evaluated
        // at plan time
        match expr {
            // Has no runtime cost, but needed during planning
            Expression::Aggregation { .. }
            | Expression::MetricExpression{ .. }
            | Expression::Rollup { .. }
            | Expression::Parens(_)
            | Expression::With(_) => false,
            Expression::Function(FuncExpr { function, .. }) => Self::volatility_ok(
                function.volatility()),
            Expression::Function(_) => false,
            Expression::Number(_)
            | Expression::String(_)
            | Expression::BinaryOperator { .. }
            | Expression::Duration(DurationExpr{ requires_step: false, .. }) => true,
            Expression::Duration(_) => false,
        }
    }

    /// Internal helper to evaluates an Expr
    pub(crate) fn evaluate_to_scalar(&mut self, expr: Expression) -> ParseResult<Expression> {
        match expr {
            Expression::Number(_) | Expression::String(_) => Ok(expr),
            Expression::Duration(d) => Ok(Expression::from(d.value)),
            Expression::BinaryOperator(ref be) => {
                if let Some(const_value) = constant_fold_binary_expression(be) {
                    Ok(const_value)
                } else {
                    Ok(expr)
                }
            },
            _ => Ok(expr)
        }
    }
}

/// Simplifies [`Expr`]s by applying algebraic transformation rules
///
/// Example transformations that are applied:
/// * `expr == true` and `expr != false` to `expr` when `expr` is of boolean type
/// * `expr == false` and `expr != true` to `!expr` when `expr` is of boolean type
/// * `true == true` and `false == false` to `true`
/// * `false == true` and `true == false` to `false`
/// * `!!expr` to `expr`
/// * `expr = null` and `expr != null` to `null`
struct Simplifier {
}

impl Simplifier {
    pub fn new() -> Self {
        Self { }
    }
}

// see https://prometheus.io/docs/prometheus/latest/querying/operators/

impl ExprRewriter for Simplifier {
    /// rewrite the expression simplifying any constant expressions
    fn mutate(&mut self, expr: Expression) -> ParseResult<Expression> {
        use Operator::{
            And, Div, Mul, Or,
        };

        let new_expr = match expr {
            Expression::Parens(pe) => {
                if pe.len() == 1 {
                    std::mem::take(pe.expressions[0].as_mut())
                } else {
                    // Treat parensExpr as a function with empty name, i.e. union()
                    // todo: how to avoid clone
                    let fe = FuncExpr::new("", pe.expressions.clone(), expr.span())?;
                    Expression::Function(fe)
                }
            },

            //
            // Rules for OR
            //

            // (..A..) OR A --> (..A..)
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Or,
                                 right, ..
                             }) if expr_contains(&left, &right, Or) => *left,
            // A OR (..A..) --> (..A..)
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                               op: Or,
                                 right, ..
                             }) if expr_contains(&right, &left, Or) => *right,
            // A OR (A AND B) --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Or,
                                 right, ..
                             }) if is_op_with(And, &right, &left) => *left,
            // (A AND B) OR A --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                op: Or,
                                 right, ..
                             }) if is_op_with(And, &left, &right) => *right,

            //
            // Rules for AND
            //

            // (..A..) AND A --> (..A..)
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: And,
                                 right, ..
                             }) if expr_contains(&left, &right, And) => *left,
            // A AND (..A..) --> (..A..)
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: And,
                                 right, ..
                             }) if expr_contains(&right, &left, And) => *right,
            // A AND (A OR B) --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: And,
                                 right, ..
                             }) if is_op_with(Or, &right, &left) => *left,
            
            // (A OR B) AND A --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: And,
                                 right, ..
                             }) if is_op_with(Or, &left, &right) => *right,

            //
            // Rules for Multiply
            //

            // A * 1 --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Operator::Mul,
                                 right, ..
                             }) if is_one(&right) => *left,

            // 1 * A --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Operator::Mul,
                                 right, ..
                             }) if is_one(&left) => *right,

            // A * NaN --> NaN
            Expression::BinaryOperator(BinaryExpr {
                                 left: _,
                                 op: Operator::Mul,
                                 right, ..
                             }) if is_null(&right) => *right,

            // NaN * A --> NaN
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Operator::Mul,
                                 right: _,
                                 ..
                             }) if is_null(&left) => *left,

            // A * 0 --> 0
            Expression::BinaryOperator(BinaryExpr {
                                 left: _,
                                 op: Mul,
                                 right, ..
                             }) if is_zero(&right) => *right,
            
            // 0 * A --> 0
            Expression::BinaryOperator(BinaryExpr { left, op: Mul, right: _, .. })
                if is_zero(&left) => *left,

            //
            // Rules for Divide
            //

            // A / 1 --> A
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Div,
                                 right, ..
                             }) if is_one(&right) => *left,
            
            // NaN / A --> NaN
            Expression::BinaryOperator(BinaryExpr { left, op: Div, right: _, .. })
                if is_null(&left) => *left,
            
            // A / NaN --> NaN
            Expression::BinaryOperator(BinaryExpr {
                                 left: _,
                                 op: Div,
                                 right, ..
                             }) if is_null(&right) => *right,
            
            // 0 / 0 -> NaN
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Div,
                                 right, ..
                             }) if is_zero(&left) && is_zero(&right) => {
                Expression::Number(NumberExpr::from(f64::NAN))
            }
            // A / 0 -> DivideByZero Error
            Expression::BinaryOperator(BinaryExpr {
                                 left: _,
                                 op: Divide,
                                 right, ..
                             }) if is_zero(&right) => {
                return Err(ParseError::DivisionByZero);
            }

            //
            // Rules for Modulo
            //

            // A % NaN --> NaN
            Expression::BinaryOperator(BinaryExpr {
                                 left: _,
                                 op: Operator::Mod,
                                 right, ..
                             }) if is_null(&right) => *right,

            // NaN % A --> NaN
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Operator::Mod,
                                 right: _,
                                 ..
                             }) if is_null(&left) => *left,
            // A % 1 --> 0
            Expression::BinaryOperator(BinaryExpr {
                                 left: _,
                                 op: Operator::Mod,
                                 right, ..
                             }) if is_one(&right) => Expression::from(0.0),
            // A % 0 --> DivideByZero Error
            Expression::BinaryOperator(BinaryExpr {
                                 left,
                                 op: Operator::Mod,
                                 right, ..
                             }) if is_zero(&right) => {
                                    return Err(ParseError::DivisionByZero);
                                }

            // no additional rewrites possible
            expr => expr,
        };
        Ok(new_expr)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, f64, sync::Arc};

    use crate::simplify_expressions::{
        utils::for_test::{now_expr, to_timestamp_expr}
    };

    use super::*;
    use chrono::{DateTime, TimeZone, Utc};
    use datafusion_common::{assert_contains, cast::as_int32_array};
    use datafusion_expr::*;

    // ------------------------------
    // --- ExprSimplifier tests -----
    // ------------------------------
    #[test]
    fn api_basic() {
        let props = ExecutionProps::new();
        let simplifier = ExprSimplifier::new();

        let expr = number(1.0) + number(2.0);
        let expected = number(3);
        assert_eq!(expected, simplifier.simplify(expr).unwrap());
    }

    #[test]
    fn simplify_and_constant_prop() {
        let simplifier = ExprSimplifier::new();

        // should be able to simplify to false
        // (i * (1 - 2)) > 0
        let expr = (selector("i") * (number(1.0) - number(1.0))).gt(number(0.0));
        let expected = lit(false);
        assert_eq!(expected, simplifier.simplify(expr).unwrap());
    }

    // ------------------------------
    // --- ConstEvaluator tests -----
    // ------------------------------
    fn test_evaluate_with_start_time(
        input_expr: Expression,
        expected_expr: Expression,
        date_time: &DateTime<Utc>,
    ) {
        let execution_props = ExecutionProps {
            query_execution_start_time: *date_time,
            var_providers: None,
        };

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

    fn test_evaluate(input_expr: Expression, expected_expr: Expression) {
        test_evaluate_with_start_time(input_expr, expected_expr, &Utc::now())
    }

    #[test]
    fn test_const_evaluator() {
        // true --> true
        test_evaluate(lit(true), lit(true));
        // true or true --> true
        test_evaluate(lit(true).or(lit(true)), lit(true));
        // true or false --> true
        test_evaluate(lit(true).or(lit(false)), lit(true));

        // "foo" == "foo" --> true
        test_evaluate(lit("foo").eq(lit("foo")), lit(true));
        // "foo" != "foo" --> false
        test_evaluate(lit("foo").not_eq(lit("foo")), lit(false));

        // c = 1 --> c = 1
        test_evaluate(selector("c").eq(number(1.0)), selector("c").eq(number(1.0)));
        // c = 1 + 2 --> c + 3
        test_evaluate(selector("c").eq(number(1.0) + lit(2)), selector("c").eq(lit(3)));
        // (foo != foo) OR (c = 1) --> false OR (c = 1)
        test_evaluate(
            (lit("foo").not_eq(lit("foo"))).or(selector("c").eq(number(1.0))),
            lit(false).or(selector("c").eq(number(1.0))),
        );
    }

    #[test]
    fn test_const_evaluator_scalar_functions() {
        // volatile / stable functions should not be evaluated
        // rand() + (1 + 2) --> rand() + 3
        let fun = BuiltinScalarFunction::Random;
        assert_eq!(fun.volatility(), Volatility::Volatile);
        let rand = Expression::ScalarFunction { args: vec![], fun };
        let expr = rand.clone() + (number(1.0) + lit(2));
        let expected = rand + lit(3);
        test_evaluate(expr, expected);

        // parenthesization matters: can't rewrite
        // (rand() + 1) + 2 --> (rand() + 1) + 2)
        let fun = BuiltinScalarFunction::Random;
        let rand = Expression::ScalarFunction { args: vec![], fun };
        let expr = (rand + number(1.0)) + lit(2);
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
        let expr_a = lit(true).and(col("c2"));
        let expr_b = col("c2").and(lit(true));
        let expected = col("c2");

        assert_eq!(simplify(expr_a), expected);
        assert_eq!(simplify(expr_b), expected);
    }

    #[test]
    fn test_simplify_multiply_by_one() {
        let expr_a = binary_expr(col("c2"), Operator::Multiply, number(1));
        let expr_b = binary_expr(number(1.0), Operator::Multiply, col("c2"));
        let expected = col("c2");

        assert_eq!(simplify(expr_a), expected);
        assert_eq!(simplify(expr_b), expected);

        let expr = binary_expr(
            selector("c2"),
            Operator::Multiply,
            number(1)
        );
        assert_eq!(simplify(expr), expected);
        let expr = binary_expr(
            number(1),
            Operator::Multiply,
            selector("c2"),
        );
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_multiply_by_null() {
        let null = Expression::from(f64::NAN);
        // A * null --> null
        {
            let expr = binary_expr(selector("c2"), Operator::Multiply, null.clone());
            assert_eq!(simplify(expr), null);
        }
        // null * A --> null
        {
            let expr = binary_expr(null.clone(), Operator::Multiply, selector("c2"));
            assert_eq!(simplify(expr), null);
        }
    }

    #[test]
    fn test_simplify_multiply_by_zero() {
        // cannot optimize A * null (null * A)
        {
            let expr_a = binary_expr(selector("c2"), Operator::Multiply, number(0.0));
            let expr_b = binary_expr(number(0), Operator::Multiply, selector("c2"));

            assert_eq!(simplify(expr_a.clone()), expr_a);
            assert_eq!(simplify(expr_b.clone()), expr_b);
        }
        // 0 * A --> 0 if A is not nullable
        {
            let expr = binary_expr(number(0.0), Operator::Multiply, selector("c2_non_null"));
            assert_eq!(simplify(expr), number(0.0));
        }
        // A * 0 --> 0 if A is not nullable
        {
            let expr = binary_expr(selector("c2_non_null"), Operator::Multiply, number(0));
            assert_eq!(simplify(expr), number(0.0));
        }
        // A * 0 --> 0
        {
            let expr = binary_expr(
                selector("c2_non_null"),
                Operator::Multiply,
                number(0.0),
            );
            assert_eq!(
                simplify(expr),
                number(0.0)
            );
            let expr = binary_expr(
                number(0.0),
                Operator::Multiply,
                selector("c2_non_null"),
            );
            assert_eq!(
                simplify(expr),
                number(0.0)
            );
        }
    }

    #[test]
    fn test_simplify_divide_by_one() {
        let expr = binary_expr(selector("c2"), Operator::Divide, number(1.0));
        let expected = selector("c2");
        assert_eq!(simplify(expr), expected);
        let expr = binary_expr(
            selector("c2"),
            Operator::Divide,
            number(0.0)
        );
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_divide_null() {
        // A / null --> null
        let null = Expression::Literal(ScalarValue::Null);
        {
            let expr = binary_expr(selector("c"), Operator::Divide, null.clone());
            assert_eq!(simplify(expr), null);
        }
        // null / A --> null
        {
            let expr = binary_expr(null.clone(), Operator::Divide, selector("c"));
            assert_eq!(simplify(expr), null);
        }
    }

    #[test]
    fn test_simplify_divide_by_same() {
        let expr = binary_expr(selector("c2"), Operator::Divide, selector("c2"));
        // if c2 is null, c2 / c2 = null, so can't simplify
        let expected = expr.clone();

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_divide_zero_by_zero() {
        // 0 / 0 -> null
        let expr = binary_expr(lit(0), Operator::Divide, lit(0));
        let expected = Expression::Literal(ScalarValue::Int32(None));

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    #[should_panic(
    expected = "called `Result::unwrap()` on an `Err` value: ArrowError(DivideByZero)"
    )]
    fn test_simplify_divide_by_zero() {
        // A / 0 -> DivideByZeroError
        let expr = binary_expr(selector("c2_non_null"), Operator::Divide, lit(0));

        simplify(expr);
    }

    #[test]
    fn test_simplify_modulo_by_null() {
        let null = Expression::from(f64::NAN);
        // A % NaN --> NaN
        {
            let expr = binary_expr(selector("c2"), Operator::Modulo, null.clone());
            assert_eq!(simplify(expr), null);
        }
        // NaN % A --> NaN
        {
            let expr = binary_expr(null.clone(), Operator::Modulo, selector("c2"));
            assert_eq!(simplify(expr), null);
        }
    }

    #[test]
    fn test_simplify_modulo_by_one() {
        let expr = binary_expr(selector("c2"), Operator::Modulo, number(1.0));
        // if c2 is null, c2 % 1 = null, so can't simplify
        let expected = expr.clone();

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_modulo_by_one_non_null() {
        let expr = binary_expr(selector("c2_non_null"), Operator::Modulo, number(1.0));
        let expected = lit(0);
        assert_eq!(simplify(expr), expected);
        let expr = binary_expr(
            selector("c2_non_null"),
            Operator::Modulo,
            Expression::Literal(ScalarValue::Decimal128(Some(10000000000), 31, 10)),
        );
        assert_eq!(simplify(expr), expected);
    }

    #[test]
    #[should_panic(
    expected = "called `Result::unwrap()` on an `Err` value: ArrowError(DivideByZero)"
    )]
    fn test_simplify_modulo_by_zero_non_null() {
        let expr = binary_expr(selector("c2_non_null"), Operator::Modulo, number(0.0));
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
            binary_expr(selector("c2").gt(number(5.0)), Operator::And, selector("c1").lt(number(6.0))),
            Operator::And,
            selector("c2").gt(number(5.0)),
        );
        let expected =
            binary_expr(selector("c2").gt(number(5.0)), Operator::And, selector("c1").lt(number(6.0)));

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_or_and() {
        let l = selector("c2").gt(number(5.0));
        let r = binary_expr(selector("c1").lt(number(6.0)), Operator::And, selector("c2").gt(number(5.0)));

        // (c2 > 5) OR ((c1 < 6) AND (c2 > 5))
        let expr = binary_expr(l.clone(), Operator::Or, r.clone());

        // no rewrites if c1 can be null
        let expected = expr.clone();
        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) AND (c2 > 5)) OR (c2 > 5)
        let expr = binary_expr(l, Operator::Or, r);

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
        let expr = binary_expr(l.clone(), Operator::Or, r.clone());

        // This is only true if `c1 < 6` is not nullable / can not be null.
        let expected = selector("c2_non_null").gt(number(5.0));

        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) AND (c2 > 5)) OR (c2 > 5) --> c2 > 5
        let expr = binary_expr(l, Operator::Or, r);

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_and_or() {
        let l = selector("c2").gt(number(5.0));
        let r = binary_expr(selector("c1").lt(number(6.0)), Operator::Or, selector("c2").gt(number(5.0)));

        // (c2 > 5) AND ((c1 < 6) OR (c2 > 5)) --> c2 > 5
        let expr = binary_expr(l.clone(), Operator::And, r.clone());

        // no rewrites if c1 can be null
        let expected = expr.clone();
        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) OR (c2 > 5)) AND (c2 > 5) --> c2 > 5
        let expr = binary_expr(l, Operator::And, r);
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
        let expr = binary_expr(l.clone(), Operator::And, r.clone());

        // This is only true if `c1 < 6` is not nullable / can not be null.
        let expected = selector("c2_non_null").gt(number(5.0));

        assert_eq!(simplify(expr), expected);

        // ((c1 < 6) OR (c2 > 5)) AND (c2 > 5) --> c2 > 5
        let expr = binary_expr(l, Operator::And, r);

        assert_eq!(simplify(expr), expected);
    }

    #[test]
    fn test_simplify_null_and_false() {
        let expr = binary_expr(lit_bool_null(), Operator::And, lit(false));
        let expr_eq = lit(false);

        assert_eq!(simplify(expr), expr_eq);
    }

    #[test]
    fn test_simplify_divide_null_by_null() {
        let null = number(f64::NAN);
        let expr_plus = binary_expr(null.clone(), Operator::Divide, null.clone());
        let expr_eq = null;

        assert_eq!(simplify(expr_plus), expr_eq);
    }

    #[test]
    fn test_simplify_simplify_arithmetic_expr() {
        let expr_plus = binary_expr(number(1.0), Operator::Plus, number(1.0));
        let expr_eq = binary_expr(number(1.0), Operator::Eq, number(1.0));

        assert_eq!(simplify(expr_plus), lit(2));
        assert_eq!(simplify(expr_eq), lit(true));
    }

    #[track_caller]
    fn assert_no_change(expr: Expr) {
        let optimized = simplify(expr.clone());
        assert_eq!(expr, optimized);
    }

    #[track_caller]
    fn assert_change(expr: Expr, expected: Expr) {
        let optimized = simplify(expr);
        assert_eq!(optimized, expected);
    }
    
    // ------------------------------
    // ----- Simplifier tests -------
    // ------------------------------

    fn try_simplify(expr: Expression) -> ParseResult<Expression> {
        let execution_props = ExecutionProps::new();
        let simplifier = ExprSimplifier::new();
        simplifier.simplify(expr)
    }

    fn simplify(expr: Expr) -> Expr {
        try_simplify(expr).unwrap()
    }
    

    #[test]
    fn simplify_expr_not_not() {
        assert_eq!(simplify(selector("c2").not().not().not()), selector("c2").not(),);
    }

    #[test]
    fn simplify_expr_null_comparison() {
        // x == NAN is always NAN
        assert_eq!(
            simplify(lit(true).eq(lit(ScalarValue::Boolean(None)))),
            number(f64::NAN),
        );

        // null != null is always null
        assert_eq!(
            simplify(
                lit(ScalarValue::Boolean(None)).not_eq(lit(ScalarValue::Boolean(None)))
            ),
            number(f64::NAN),
        );

        // x != null is always null
        assert_eq!(
            simplify(selector("c2").not_eq(lit(ScalarValue::Boolean(None)))),
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
        assert_eq!(simplify(lit(true).eq(number(1.0))), lit(true));

        // true == false -> false
        assert_eq!(simplify(lit(true).eq(lit(false))), lit(false),);
    }

    #[test]
    fn simplify_expr_eq_skip_nonboolean_type() {
        // don't fold c1 = foo
        assert_eq!(simplify(selector("c1").eq(lit("foo"))), selector("c1").eq(lit("foo")),);
    }

    #[test]
    fn simplify_expr_not_eq() {
        // test constant
        assert_eq!(simplify(lit(true).not_eq(lit(true))), lit(false),);

        assert_eq!(simplify(lit(true).not_eq(lit(false))), lit(true),);
    }

    #[test]
    fn simplify_expr_not_eq_skip_nonboolean_type() {
        assert_eq!(
            simplify(selector("c1").not_eq(lit("foo"))),
            selector("c1").not_eq(lit("foo")),
        );
    }

    #[test]
    fn simplify_expr_bool_or() {
        // col || true is always true
        assert_eq!(simplify(selector("c2").or(lit(true))), lit(true),);

        // col || false is always col
        assert_eq!(simplify(selector("c2").or(lit(false))), selector("c2"),);

        // true || null is always true
        assert_eq!(simplify(lit(true).or(lit_bool_null())), lit(true),);

        // null || true is always true
        assert_eq!(simplify(lit_bool_null().or(lit(true))), lit(true),);

        // false || null is always null
        assert_eq!(simplify(lit(false).or(lit_bool_null())), lit_bool_null(),);

        // null || false is always null
        assert_eq!(simplify(lit_bool_null().or(lit(false))), lit_bool_null(),);
    }

    #[test]
    fn simplify_expr_bool_and() {
        // col & true is always col
        assert_eq!(simplify(selector("c2").and(lit(true))), selector("c2"),);
        // col & false is always false
        assert_eq!(simplify(selector("c2").and(lit(false))), lit(false),);

        // true && NaN is always NaN
        assert_eq!(simplify(lit(true).and(lit_bool_null())), lit_bool_null(),);

        // NaN && true is always NaN
        assert_eq!(simplify(lit_bool_null().and(lit(true))), lit_bool_null(),);

        // false && NaN is always false
        assert_eq!(simplify(lit(false).and(lit_bool_null())), lit(false),);

        // NaN && false is always false
        assert_eq!(simplify(lit_bool_null().and(lit(false))), lit(false),);
    }

}