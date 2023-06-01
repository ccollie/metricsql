use crate::ast::{Expr, FunctionExpr};
use crate::common::ValueType;
use crate::functions::BuiltinFunction;
use crate::parser::tokens::Token;
use crate::parser::{ParseError, ParseResult, Parser};
use std::ops::Deref;

pub(super) fn parse_func_expr(p: &mut Parser) -> ParseResult<Expr> {
    let name = p.expect_identifier()?;
    let args = p.parse_arg_list()?;

    // with (f(x) = sum(x * 2))  f(x{a="b"}) => sum(x{a="b"}) * 2)
    // check if we have a function with the same name in the with stack
    if p.can_lookup() {
        let args_clone = args.clone();
        if let Some(expr) = p.resolve_ident( &name, args_clone)? {
            return Ok(expr);
        }
    }

    let mut fe = FunctionExpr::new(&name, args)?;
    fe.keep_metric_names = if p.at(&Token::KeepMetricNames) {
        p.bump();
        true
    } else {
        false
    };

    // TODO: !!!! fix validate args
    // validate_args(&fe.function, &fe.args)?;

    Ok(Expr::Function(fe))
}

pub fn validate_function_args(func: &BuiltinFunction, args: &[Expr]) -> ParseResult<()> {
    let expect = |actual: ValueType, expected: ValueType, index: usize| -> ParseResult<()> {
        if actual != expected {
            return Err(ParseError::ArgumentError(format!(
                "Invalid argument #{} to {}. {} expected",
                index + 1,
                func,
                expected
            )));
        }
        Ok(())
    };

    let validate_return_type =
        |return_type: ValueType, expected: ValueType, index: usize| -> ParseResult<()> {
            match expected {
                ValueType::RangeVector => {
                    return expect(return_type, ValueType::RangeVector, index);
                }
                ValueType::InstantVector => {
                    return match return_type {
                        // scalar can be converted to InstantVector
                        ValueType::Scalar | ValueType::InstantVector => Ok(()),
                        _ => expect(return_type, ValueType::InstantVector, index),
                    };
                }
                ValueType::Scalar => {
                    if !return_type.is_operator_valid() {
                        return Err(ParseError::ArgumentError(format!(
                            "Invalid argument #{} to {}. Scalar or InstantVector expected",
                            index + 1,
                            func
                        )));
                    }
                }
                ValueType::String => {
                    return expect(return_type, ValueType::String, index);
                }
            }
            Ok(())
        };

    // validate function args
    let sig = func.signature();

    sig.validate_arg_count(&func.name(), args.len())?;
    let mut i = 0;
    for (expected_type, actual) in sig.types().zip(args.iter()) {
        validate_return_type(actual.return_type(), expected_type.clone(), 0)?;

        match *actual.deref() {
            // technically should not occur as a function parameter
            Expr::Duration(_) | Expr::Number(_) => {
                if expected_type.is_scalar() {
                    continue;
                }
            }
            Expr::StringLiteral(_) => {
                if expected_type == ValueType::String {
                    continue;
                }
            }
            _ => {}
        }

        validate_return_type(actual.return_type(), expected_type, i)?;
        i += 1;
    }
    Ok(())
}
