use crate::ast::{AggregateModifier, AggregationExpr, Expr};
use crate::functions::{AggregateFunction, BuiltinFunction, FunctionMeta};
use crate::parser::function::validate_function_args;
use crate::parser::tokens::Token;
use crate::parser::{ParseError, ParseResult, Parser};

impl Parser<'_> {
    /// parse_aggr_func_expr parses an aggregation Expr.
    ///
    ///    <aggr_op> (<Vector_expr>) [by|without <labels>] [limit number]
    ///
    ///    <aggr_op> [by|without <labels>] (<Vector_expr>) [limit number]
    ///
    ///     e.g. `sum(rate(foo[5m])) by (job) limit 10`
    ///
    pub fn parse_aggr_func_expr(&mut self) -> ParseResult<Expr> {
        let tok = self.expect_identifier()?;

        let func = get_aggregation_function(&tok)?;

        fn handle_prefix(p: &mut Parser, func: AggregateFunction) -> ParseResult<Expr> {
            let modifier = Some(p.parse_aggregate_modifier()?);
            handle_args(p, func, modifier)
        }

        fn handle_args(
            p: &mut Parser,
            func: AggregateFunction,
            modifier: Option<AggregateModifier>,
        ) -> ParseResult<Expr> {
            let args = p.parse_arg_list()?;

            validate_function_args(&BuiltinFunction::Aggregate(func), &args)?;
            let mut ae = AggregationExpr::new(func, args);
            let kind = p.peek_kind();
            // Verify whether func suffix exists.
            ae.modifier = if modifier.is_none() && kind.is_aggregate_modifier() {
                Some(p.parse_aggregate_modifier()?)
            } else {
                modifier
            };

            if p.at(&Token::Limit) {
                ae.limit = p.parse_limit()?;
            }

            Ok(Expr::Aggregation(ae))
        }

        let kind = self.peek_kind();
        if kind.is_aggregate_modifier() {
            handle_prefix(self, func)
        } else if kind == Token::LeftParen {
            handle_args(self, func, None)
        } else {
            Err(self.token_error(&[Token::By, Token::Without, Token::LeftParen]))
        }
    }

    fn parse_aggregate_modifier(&mut self) -> ParseResult<AggregateModifier> {
        let tok = self.expect_one_of(&[Token::By, Token::Without])?.kind;

        let mut args = self.parse_ident_list()?;
        args.sort();

        let res = match tok {
            Token::By => AggregateModifier::By(args),
            Token::Without => AggregateModifier::Without(args),
            _ => unreachable!("unexpected aggregate modifier token"),
        };

        Ok(res)
    }

    fn parse_limit(&mut self) -> ParseResult<usize> {
        self.expect(&Token::Limit)?;
        let v = self.parse_number()?;
        if v < 0.0 || !v.is_finite() {
            let msg = format!("LIMIT should be a positive integer. Found {v} ");
            return Err(self.syntax_error(&msg));
        }
        Ok(v as usize)
    }
}

fn get_aggregation_function(name: &str) -> ParseResult<AggregateFunction> {
    if let Some(meta) = FunctionMeta::lookup(name) {
        if let BuiltinFunction::Aggregate(af) = meta.function {
            return Ok(af);
        }
    }
    Err(ParseError::InvalidFunction(format!("aggregation::{name}")))
}
