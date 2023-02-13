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

use crate::expr::{AggregateFunction, BinaryExpr, Cast, GroupingSet, TryCast};
use crate::{
    aggregate_function, built_in_function, conditional_expressions::CaseBuilder,
    logical_plan::Subquery, AccumulatorFunctionImplementation, AggregateUDF,
    BuiltinScalarFunction, Expr, Operator, ReturnTypeFunction,
    ScalarFunctionImplementation, ScalarUDF, Signature, StateTypeFunction, Volatility,
};
use arrow::datatypes::DataType;
use std::sync::Arc;

/// Create a column expression based on a qualified or unqualified column name
///
/// example:
/// ```
/// # use datafusion_expr::col;
/// let c = col("my_column");
/// ```
pub fn selector(ident: impl Into<Column>) -> Expr {
    Expr::Column(ident.into())
}

pub fn number(val: f64) -> Expression {
    Expression::Number(NumberExpr::new(val))
}

/// Return a new expression `left <op> right`
pub fn binary_expr(left: Expression, op: Operator, right: Expression) -> Expression {
    Expression::BinaryExpr(BinaryExpr::new(Box::new(left), op, Box::new(right)))
}

/// Return a new expression with a logical AND
pub fn and(left: Expression, right: Expression) -> Expression {
    binary_expr(left, BinaryOp::And, right)
}

/// Return a new expression with a logical OR
pub fn or(left: Expression, right: Expression) -> Expression {
    binary_expr(left, BinaryOp::Or, right)
}

/// Create an expression to represent the min() aggregate function
pub fn min(expr: Expression) -> Expression {
    Expr::AggregateFunction(AggregateFunction::new(
        aggregate_function::AggregateFunction::Min,
        vec![expr],
        false,
        None,
    ))
}

/// Create an expression to represent the max() aggregate function
pub fn max(expr: Expr) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new(
        aggregate_function::AggregateFunction::Max,
        vec![expr],
        false,
        None,
    ))
}

/// Create an expression to represent the sum() aggregate function
pub fn sum(expr: Expr) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new(
        aggregate_function::AggregateFunction::Sum,
        vec![expr],
        false,
        None,
    ))
}

/// Create an expression to represent the avg() aggregate function
pub fn avg(expr: Expr) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new(
        aggregate_function::AggregateFunction::Avg,
        vec![expr],
        false,
        None,
    ))
}

/// Create an expression to represent the count() aggregate function
pub fn count(expr: Expr) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new(
        aggregate_function::AggregateFunction::Count,
        vec![expr],
        false,
        None,
    ))
}

/// Create a grouping set for rollup
pub fn rollup(exprs: Vec<Expr>) -> Expr {
    Expr::GroupingSet(GroupingSet::Rollup(exprs))
}

/// Create is true expression
pub fn is_true(expr: Expression) -> Expression {
    let mut res = BinaryExpr::new(
        BinaryOp::Eq,
        Box::new(expr),
        Box::new( number(1.0) )
    );
    res.bool_modifier = true;
    res
}

/// Create is not false expression
pub fn is_not_false(expr: Expr) -> Expr {
    let mut res = BinaryExpr::new(
        BinaryOp::Neq,
        Box::new(expr),
        Box::new( number(0.0) )
    );
    res.bool_modifier = true;
    res
}

macro_rules! scalar_expr {
    ($ENUM:ident, $FUNC:ident, $($arg:ident)*, $DOC:expr) => {
        #[doc = $DOC ]
        pub fn $FUNC($($arg: Expr),*) -> Expr {
            Expr::ScalarFunction {
                fun: built_in_function::BuiltinScalarFunction::$ENUM,
                args: vec![$($arg),*],
            }
        }
    };
}

macro_rules! nary_scalar_expr {
    ($ENUM:ident, $FUNC:ident, $DOC:expr) => {
        #[doc = $DOC ]
        pub fn $FUNC(args: Vec<Expr>) -> Expr {
            Expr::ScalarFunction {
                fun: built_in_function::BuiltinScalarFunction::$ENUM,
                args,
            }
        }
    };
}


nary_scalar_expr!(Concat, concat_expr, "concatenates several strings");

/// Creates a new UDF with a specific signature and specific return type.
/// This is a helper function to create a new UDF.
/// The function `create_udf` returns a subset of all possible `ScalarFunction`:
/// * the UDF has a fixed return type
/// * the UDF has a fixed signature (e.g. [f64, f64])
pub fn create_udf(
    name: &str,
    input_types: Vec<DataType>,
    return_type: Arc<DataType>,
    volatility: Volatility,
    fun: ScalarFunctionImplementation,
) -> ScalarUDF {
    let return_type: ReturnTypeFunction = Arc::new(move |_| Ok(return_type.clone()));
    ScalarUDF::new(
        name,
        &Signature::exact(input_types, volatility),
        &return_type,
        &fun,
    )
}

/// Creates a new UDAF with a specific signature, state type and return type.
/// The signature and state type must match the `Accumulator's implementation`.
#[allow(clippy::rc_buffer)]
pub fn create_udaf(
    name: &str,
    input_type: DataType,
    return_type: Arc<DataType>,
    volatility: Volatility,
    accumulator: AccumulatorFunctionImplementation,
    state_type: Arc<Vec<DataType>>,
) -> AggregateUDF {
    let return_type: ReturnTypeFunction = Arc::new(move |_| Ok(return_type.clone()));
    let state_type: StateTypeFunction = Arc::new(move |_| Ok(state_type.clone()));
    AggregateUDF::new(
        name,
        &Signature::exact(vec![input_type], volatility),
        &return_type,
        &accumulator,
        &state_type,
    )
}

/// Calls a named built in function
/// ```
/// use datafusion_expr::{col, lit, call_fn};
///
/// // create the expression sin(x) < 0.2
/// let expr = call_fn("sin", vec![col("x")]).unwrap().lt(lit(0.2));
/// ```
pub fn call_fn(name: impl AsRef<str>, args: Vec<Expr>) -> Result<Expr> {
    match name.as_ref().parse::<BuiltinFunction>() {
        Ok(fun) => Ok(Expression::Function { FuncExpr::new(fun, args })),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::lit;

    #[test]
    fn filter_is_null_and_is_not_null() {
        let col_null = selector("col1");
        let col_not_null = selector("col2");
        assert_eq!(format!("{:?}", col_null.is_null()), "col1 IS NULL");
        assert_eq!(
            format!("{:?}", col_not_null.is_not_null()),
            "col2 IS NOT NULL"
        );
    }

    macro_rules! test_unary_scalar_expr {
        ($ENUM:ident, $FUNC:ident) => {{
            if let Expr::ScalarFunction { fun, args } = $FUNC(col("tableA.a")) {
                let name = built_in_function::BuiltinScalarFunction::$ENUM;
                assert_eq!(name, fun);
                assert_eq!(1, args.len());
            } else {
                assert!(false, "unexpected");
            }
        }};
    }

    macro_rules! test_scalar_expr {
        ($ENUM:ident, $FUNC:ident, $($arg:ident),*) => {
            let expected = vec![$(stringify!($arg)),*];
            let result = $FUNC(
                $(
                    col(stringify!($arg.to_string()))
                ),*
            );
            if let Expr::ScalarFunction { fun, args } = result {
                let name = built_in_function::BuiltinScalarFunction::$ENUM;
                assert_eq!(name, fun);
                assert_eq!(expected.len(), args.len());
            } else {
                assert!(false, "unexpected: {:?}", result);
            }
        };
    }

    macro_rules! test_nary_scalar_expr {
        ($ENUM:ident, $FUNC:ident, $($arg:ident),*) => {
            let expected = vec![$(stringify!($arg)),*];
            let result = $FUNC(
                vec![
                    $(
                        col(stringify!($arg.to_string()))
                    ),*
                ]
            );
            if let Expr::ScalarFunction { fun, args } = result {
                let name = built_in_function::BuiltinScalarFunction::$ENUM;
                assert_eq!(name, fun);
                assert_eq!(expected.len(), args.len());
            } else {
                assert!(false, "unexpected: {:?}", result);
            }
        };
    }

    #[test]
    fn scalar_function_definitions() {
        test_unary_scalar_expr!(Sqrt, sqrt);
        test_unary_scalar_expr!(Sin, sin);
        test_unary_scalar_expr!(Cos, cos);
        test_unary_scalar_expr!(Tan, tan);
        test_unary_scalar_expr!(Asin, asin);
        test_unary_scalar_expr!(Acos, acos);
        test_unary_scalar_expr!(Atan, atan);
        test_unary_scalar_expr!(Floor, floor);
        test_unary_scalar_expr!(Ceil, ceil);
        test_unary_scalar_expr!(Round, round);
        test_unary_scalar_expr!(Trunc, trunc);
        test_unary_scalar_expr!(Abs, abs);
        test_unary_scalar_expr!(Signum, signum);
        test_unary_scalar_expr!(Exp, exp);
        test_unary_scalar_expr!(Log2, log2);
        test_unary_scalar_expr!(Log10, log10);
        test_unary_scalar_expr!(Ln, ln);
        test_scalar_expr!(Atan2, atan2, y, x);

        test_scalar_expr!(Ascii, ascii, input);
        test_scalar_expr!(BitLength, bit_length, string);
        test_nary_scalar_expr!(Btrim, btrim, string);
        test_nary_scalar_expr!(Btrim, btrim, string, characters);
        test_scalar_expr!(CharacterLength, character_length, string);
        test_scalar_expr!(Chr, chr, string);
        test_scalar_expr!(Digest, digest, string, algorithm);
        test_scalar_expr!(InitCap, initcap, string);
        test_scalar_expr!(Left, left, string, count);
        test_scalar_expr!(Lower, lower, string);
        test_nary_scalar_expr!(Lpad, lpad, string, count);
        test_nary_scalar_expr!(Lpad, lpad, string, count, characters);
        test_scalar_expr!(Ltrim, ltrim, string);
        test_scalar_expr!(MD5, md5, string);
        test_scalar_expr!(OctetLength, octet_length, string);
        test_nary_scalar_expr!(RegexpMatch, regexp_match, string, pattern);
        test_nary_scalar_expr!(RegexpMatch, regexp_match, string, pattern, flags);
        test_nary_scalar_expr!(
            RegexpReplace,
            regexp_replace,
            string,
            pattern,
            replacement
        );
        test_nary_scalar_expr!(
            RegexpReplace,
            regexp_replace,
            string,
            pattern,
            replacement,
            flags
        );
        test_scalar_expr!(Replace, replace, string, from, to);
        test_scalar_expr!(Repeat, repeat, string, count);
        test_scalar_expr!(Reverse, reverse, string);
        test_scalar_expr!(Right, right, string, count);
        test_nary_scalar_expr!(Rpad, rpad, string, count);
        test_nary_scalar_expr!(Rpad, rpad, string, count, characters);
        test_scalar_expr!(Rtrim, rtrim, string);
        test_scalar_expr!(SHA224, sha224, string);
        test_scalar_expr!(SHA256, sha256, string);
        test_scalar_expr!(SHA384, sha384, string);
        test_scalar_expr!(SHA512, sha512, string);
        test_scalar_expr!(SplitPart, split_part, expr, delimiter, index);
        test_scalar_expr!(StartsWith, starts_with, string, characters);
        test_scalar_expr!(Strpos, strpos, string, substring);
        test_scalar_expr!(Substr, substr, string, position);
        test_scalar_expr!(Substr, substring, string, position, count);
        test_scalar_expr!(ToHex, to_hex, string);
        test_scalar_expr!(Translate, translate, string, from, to);
        test_scalar_expr!(Trim, trim, string);
        test_scalar_expr!(Upper, upper, string);

        test_scalar_expr!(DatePart, date_part, part, date);
        test_scalar_expr!(DateTrunc, date_trunc, part, date);
        test_scalar_expr!(DateBin, date_bin, stride, source, origin);
        test_scalar_expr!(FromUnixtime, from_unixtime, unixtime);

        test_unary_scalar_expr!(ArrowTypeof, arrow_typeof);
    }

    #[test]
    fn uuid_function_definitions() {
        if let Expr::ScalarFunction { fun, args } = uuid() {
            let name = built_in_function::BuiltinScalarFunction::Uuid;
            assert_eq!(name, fun);
            assert_eq!(0, args.len());
        } else {
            unreachable!();
        }
    }

    #[test]
    fn digest_function_definitions() {
        if let Expr::ScalarFunction { fun, args } = digest(selector("tableA.a"), lit("md5")) {
            let name = BuiltinScalarFunction::Digest;
            assert_eq!(name, fun);
            assert_eq!(2, args.len());
        } else {
            unreachable!();
        }
    }
}