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

use crate::ast::{BinaryExpr, Operator, Expression};


/// returns true if `needle` is found in a chain of search_op
/// expressions. Such as: (A AND B) AND C
pub fn expr_contains(expr: &Expression, needle: &Expression, search_op: Operator) -> bool {
    match expr {
        Expression::BinaryOperator(BinaryExpr { left, op, right, .. }) if *op == search_op => {
            expr_contains(left, needle, search_op)
                || expr_contains(right, needle, search_op)
        }
        _ => expr == needle,
    }
}

pub fn is_zero(s: &Expression) -> bool {
    match s {
        Expression::Number(number) => number.value == 0.0,
        _ => false,
    }
}

pub fn is_one(s: &Expression) -> bool {
    match s {
        Expression::Number(number) => number.value == 1.0,
        _ => false,
    }
}

pub fn is_true(expr: &Expression) -> bool {
    match expr {
        Expression::Number(number) => number.value == 1.0,
        _ => false,
    }
}

/// Return a literal NULL value of Boolean data type
pub fn lit_bool_null() -> Expression {
    Expression::from(f64::NAN)
}

pub fn is_null(expr: &Expression) -> bool {
    match expr {
        Expression::Number(v) => v.value.is_nan(),
        _ => false,
    }
}

/// returns true if `haystack` looks like (needle OP X) or (X OP needle)
pub fn is_op_with(target_op: Operator, haystack: &Expression, needle: &Expression) -> bool {
    matches!(haystack, Expression::BinaryOperator(BinaryExpr { left, op, right, .. })
        if op == &target_op && (needle == left.as_ref() || needle == right.as_ref()))
}

#[cfg(test)]
pub mod for_test {
    use arrow::datatypes::DataType;
    use datafusion_expr::{call_fn, lit, Cast, Expr};

    pub fn now_expr() -> Expr {
        call_fn("now", vec![]).unwrap()
    }

    pub fn to_timestamp_expr(arg: impl Into<String>) -> Expr {
        call_fn("to_timestamp", vec![lit(arg.into())]).unwrap()
    }
}