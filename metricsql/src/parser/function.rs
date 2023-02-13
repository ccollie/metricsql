use std::ops::Deref;
use crate::ast::{BExpression, Expression, ExpressionNode, FuncExpr, ReturnType, WithArgExpr};
use crate::functions::{BuiltinFunction, DataType};
use crate::lexer::{TokenKind, unescape_ident};
use crate::parser::{ParseError, Parser, ParseResult};
use crate::parser::expr::parse_arg_list;


pub(super) fn parse_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let token = p.expect_token(TokenKind::Ident)?;
    let name = unescape_ident(token.text);
    let mut span= token.span;

    let args = parse_arg_list(p)?;

    let mut keep_metric_names = false;
    if p.at(TokenKind::KeepMetricNames) {
        keep_metric_names = true;
        p.update_span(&mut span);
        p.bump();
    }

    let mut fe = FuncExpr::new(&name, args, span)?;
    fe.keep_metric_names = keep_metric_names;

    // TODO: !!!! fix validate args
    // validate_args(&fe.function, &fe.args)?;

    Ok(fe.cast())
}

pub(super) fn parse_template_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let token = p.expect_token(TokenKind::Ident)?;
    let name = unescape_ident(token.text);
    let mut span= token.span;

    let func = p.resolve_template_function(&name);
    if func.is_none() {
        // todo: raise error
    }

    let func = func.unwrap();

    let args = parse_arg_list(p)?;
    
    let frame = func.was.iter().zip(args.iter()).map(|(arg, expr)| {
        WithArgExpr {
            name: arg.name.clone(),
            args: vec![],
            expr: expr.clone(),
            is_function: false,
        }
    }).collect::<Vec<WithArgExpr>>();

    p.with_stack.push(frame);

    let mut fe = FuncExpr::new(&name, args, span)?;

    // TODO: !!!! fix validate args
    // validate_args(&fe.function, &fe.args)?;

    Ok(fe.cast())
}


pub(crate) fn validate_args(func: &BuiltinFunction, args: &[BExpression]) -> ParseResult<()> {
    use ReturnType::*;

    let expect = |actual: ReturnType, expected: ReturnType, index: usize| -> ParseResult<()> {
        // Note: we don't use == because we're blocked from deriving PartialEq on ReturnValue because
        // of the Unknown variant
        if actual.to_string() != expected.to_string() {
            return Err(ParseError::ArgumentError(
                format!("Invalid argument #{} to {}. {} expected", index, func, expected)
            ))
        }
        Ok(())
    };

    let validate_return_type = |return_type: ReturnType, expected: DataType, index: usize| -> ParseResult<()> {
        match return_type {
            Unknown(u) => {
                return Err(ParseError::ArgumentError(
                    format!("Bug: Cannot determine type of argument #{} to {}. {}", index, func, u.message)
                ))
            },
            _ => {}
        }
        match expected {
            DataType::RangeVector => {
                return expect(return_type, RangeVector, index);
            }
            DataType::InstantVector => {
                return match return_type {
                    // scalar can be converted to InstantVector
                    Scalar | InstantVector => Ok(()),
                    _ => expect(return_type, InstantVector, index)
                }
            }
            DataType::Scalar => {
                if !return_type.is_operator_valid() {
                    return Err(ParseError::ArgumentError(
                        format!("Invalid argument #{} to {}. Scalar or InstantVector expected", index, func)
                    ))
                }
            }
            DataType::String => {
                return expect(return_type, String, index);
            }
        }
        Ok(())
    };

    // validate function args
    let sig = func.signature();
    sig.validate_arg_count(&func.name(), args.len())?;

    let (arg_types, _) = sig.expand_types();

    for (i, arg) in args.iter().enumerate() {
        let expected = arg_types[i];
        match *arg.deref() {
            Expression::Number(_) => {
                if expected.is_numeric() {
                    continue;
                }
            }
            Expression::String(_) => {
                if expected == DataType::String {
                    continue;
                }
            }
            Expression::Duration(_) => {
                // technically should not occur as a function parameter
                if expected.is_numeric() {
                    continue;
                }
            },
            _ => {}
        }

        validate_return_type(arg.return_type(), expected, i)?
    }
    Ok(())
}
