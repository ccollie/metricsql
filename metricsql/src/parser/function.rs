use crate::ast::{Expr, FunctionExpr, WithArgExpr};
use crate::common::ValueType;
use crate::functions::{BuiltinFunction};
use crate::parser::expr::parse_arg_list;
use crate::parser::{ParseError, ParseResult, Parser};
use std::ops::Deref;
use crate::parser::tokens::Token;

pub(super) fn parse_func_expr(p: &mut Parser) -> ParseResult<Expr> {
    let name = p.expect_identifier()?;
    let args = parse_arg_list(p)?;

    let mut keep_metric_names = false;
    if p.at(&Token::KeepMetricNames) {
        keep_metric_names = true;
        p.bump();
    }

    let mut fe = FunctionExpr::new(&name, args)?;
    fe.keep_metric_names = keep_metric_names;

    // TODO: !!!! fix validate args
    // validate_args(&fe.function, &fe.args)?;

    Ok(Expr::Function(fe))
}

pub(super) fn parse_template_func_expr(p: &mut Parser) -> ParseResult<Expr> {
    let name = p.expect_identifier()?;
    let func = p.resolve_template_function(&name);
    if func.is_none() {
        // todo: raise error
    }

    let func = func.unwrap().clone(); // todo: avoid this

    let args = parse_arg_list(p)?;

    let frame = func
        .args
        .iter()
        .zip(args.iter())
        .map(|(arg, expr)| WithArgExpr {
            name: arg.clone(),
            args: vec![],
            expr: expr.clone(),
            is_function: false,
        })
        .collect::<Vec<WithArgExpr>>();

    p.with_stack.push(frame);

    let fe = FunctionExpr::new(&name, args)?;

    // TODO: !!!! fix validate args
    // validate_args(&fe.function, &fe.args)?;

    Ok(Expr::Function(fe))
}

pub fn validate_function_args(func: &BuiltinFunction, args: &[Expr]) -> ParseResult<()> {
    let expect = |actual: ValueType, expected: ValueType, index: usize| -> ParseResult<()> {
        if actual != expected {
            return Err(ParseError::ArgumentError(format!(
                "Invalid argument #{} to {}. {} expected",
                index, func, expected
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
                            index, func
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
            Expr::Duration(_) |
            Expr::Number(_) => {
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