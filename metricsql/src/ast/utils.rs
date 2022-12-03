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

//! Utility functions for expression simplification

use crate::common::Operator;
use crate::ast::{BinaryExpr, Expr, NumberExpr};
use crate::prelude::MetricExpr;


/// Create a selector expression based on a qualified or unqualified column name
///
/// example:
/// ```
/// use crate::hir::*;
/// let c = selector("latency");
/// ```
pub fn selector(ident: impl Into<String>) -> Expr {
    Expr::MetricExpression(MetricExpr::new(ident.into()))
}

pub fn lit(str: &str) -> Expr {
    Expr::StringLiteral(str.to_string())
}

pub fn number(val: f64) -> Expr {
    Expr::from(val)
}

/// returns true if `needle` is found in a chain of search_op
/// expressions. Such as: (A AND B) AND C
pub fn expr_contains(expr: &Expr, needle: &Expr, search_op: Operator) -> bool {
    match expr {
        Expr::BinaryOperator(BinaryExpr {
                                 left, op, right, ..
                             }) if *op == search_op => {
            expr_contains(left, needle, search_op) || expr_contains(right, needle, search_op)
        }
        _ => expr == needle,
    }
}

pub fn is_number_value(s: &Expr, val: f64) -> bool {
    match s {
        Expr::Number(NumberExpr { value, .. }) => *value == val,
        _ => false,
    }
}

pub fn is_zero(s: &Expr) -> bool {
    is_number_value(s, 0.0)
}

pub fn is_one(s: &Expr) -> bool {
    is_number_value(s, 1.0)
}

pub fn is_null(expr: &Expr) -> bool {
    is_number_value(expr, f64::NAN)
}

/// returns true if `haystack` looks like (needle OP X) or (X OP needle)
pub fn is_op_with(target_op: Operator, haystack: &Expr, needle: &Expr) -> bool {
    matches!(haystack, Expr::BinaryOperator(BinaryExpr { left, op, right, .. })
        if op == &target_op && (needle == left.as_ref() || needle == right.as_ref()))
}

/// Combines an array of filter expressions into a single filter
/// expression consisting of the input filter expressions joined with
/// logical AND.
///
/// Returns None if the filters array is empty.
///
/// # Example
/// ```
/// # use crate::hir::expr_fn::{selector, number};
/// # use crate::hir::utils::conjunction;
/// // a=1 AND b=2
/// let expr = selector("a").eq(number(1)).and(selector("b").eq(number(2)));
///
/// // [a=1, b=2]
/// let split = vec![
///   selector("a").eq(number(1)),
///   selector("b").eq(number(2)),
/// ];
///
/// // use conjunction to join them together with `AND`
/// assert_eq!(conjunction(split), Some(expr));
/// ```
pub fn conjunction(filters: impl IntoIterator<Item =Expr>) -> Option<Expr> {
    filters.into_iter().reduce(|accum, expr| accum.and(expr))
}

/// Combines an array of filter expressions into a single filter
/// expression consisting of the input filter expressions joined with
/// logical OR.
///
/// Returns None if the filters array is empty.
pub fn disjunction(filters: impl IntoIterator<Item =Expr>) -> Option<Expr> {
    filters.into_iter().reduce(|accum, expr| accum.or(expr))
}


// todo: use expr_visitor instead
pub fn visit_all(e: &mut Expr, visitor: fn(&mut Expr) -> ()) {
    match e {
        Expr::BinaryOperator(be) => {
            visit_all(&mut be.left, visitor);
            visit_all(&mut be.right, visitor);
        }
        Expr::Function(fe) => {
            for arg in fe.args.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expr::Aggregation(ae) => {
            for arg in ae.args.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expr::With(we) => {
            visit_all(&mut we.expr, visitor);
            for wa in we.was.iter_mut() {
                visit_all(&mut wa.expr, visitor)
            }
        }
        Expr::Parens(pe) => {
            for arg in pe.expressions.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expr::Rollup(re) => {
            visit_all(&mut re.expr, visitor);
            if let Some(at) = &mut re.at {
                visit_all(at, visitor)
            }
        },
        _ => {}
    }
    visitor(e);
}


#[cfg(test)]
pub mod tests {
    use crate::ast::utils::{selector, conjunction, disjunction};

    #[test]
    fn test_conjunction_empty() {
        assert_eq!(conjunction(vec![]), None);
    }

    #[test]
    fn test_conjunction() {
        // `[A, B, C]`
        let expr = conjunction(vec![selector("a"), selector("b"), selector("c")]);

        // --> `(A AND B) AND C`
        assert_eq!(expr, Some(selector("a").and(selector("b")).and(selector("c"))));

        // which is different than `A AND (B AND C)`
        assert_ne!(expr, Some(selector("a").and(selector("b").and(selector("c")))));
    }

    #[test]
    fn test_disjunction_empty() {
        assert_eq!(disjunction(vec![]), None);
    }

    #[test]
    fn test_disjunction() {
        // `[A, B, C]`
        let expr = disjunction(vec![selector("a"), selector("b"), selector("c")]);

        // --> `(A OR B) OR C`
        assert_eq!(expr, Some(selector("a").or(selector("b")).or(selector("c"))));

        // which is different than `A OR (B OR C)`
        assert_ne!(expr, Some(selector("a").or(selector("b").or(selector("c")))));
    }
}
