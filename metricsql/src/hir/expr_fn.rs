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

//! Functions for creating logical expressions

use crate::common::{Operator, ReturnType};
use crate::hir::{BinaryExpr, Expression, MetricExpr};

/// Create a selector expression based on a qualified or unqualified column name
///
/// example:
/// ```
/// use crate::hir::*;
/// let c = selector("latency");
/// ```
pub fn selector(ident: impl Into<String>) -> Expression {
    Expression::MetricExpression(MetricExpr::new(ident.into()))
}

pub fn number(val: f64) -> Expression {
    Expression::from(val)
}

/// Return a new expression `left <op> right`
pub fn binary_expr(left: Expression, op: Operator, right: Expression) -> Expression {
    let expr = BinaryExpr {
        op,
        left: Box::new(left),
        right: Box::new(right),
        cardinality: None, // todo: !!!
        bool_modifier: op.is_comparison(),
        group_modifier: None,
        join_modifier: None,
        return_type: ReturnType::Scalar, // todo: !!!
    };
    Expression::BinaryOperator(expr)
}

/// Return a new expression with a logical AND
pub fn and(left: Expression, right: Expression) -> Expression {
    binary_expr(left, Operator::And, right)
}

/// Return a new expression with a logical OR
pub fn or(left: Expression, right: Expression) -> Expression {
    binary_expr(left, Operator::Or, right)
}

/// Create is true expression
pub fn is_true(expr: Expression) -> Expression {
    let res = BinaryExpr {
        op: Operator::Eql,
        left: Box::new(expr),
        right: Box::new(number(1.0)),
        cardinality: None,
        bool_modifier: true,
        group_modifier: None,
        join_modifier: None,
        return_type: ReturnType::Scalar,
    };
    Expression::BinaryOperator(res)
}

/// Create is not false expression
pub fn is_not_false(expr: Expression) -> Expression {
    let res = BinaryExpr {
        op: Operator::NotEq,
        left: Box::new(expr),
        right: Box::new(number(0.0)),
        cardinality: None,
        bool_modifier: true,
        group_modifier: None,
        join_modifier: None,
        return_type: ReturnType::Scalar,
    };
    Expression::BinaryOperator(res)
}
