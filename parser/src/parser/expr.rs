use crate::ast::{
    BinModifier, BinaryExpr, Expr, Operator, StringExpr, UnaryExpr, VectorMatchCardinality,
    VectorMatchModifier,
};
use crate::common::ValueType;
use crate::functions::FunctionMeta;
use crate::label::Labels;
use crate::parser::parse_error::unexpected;
use crate::parser::tokens::{Token, IDENT_LIKE_TOKENS};
use crate::parser::{extract_string_value, parse_number, ParseResult, Parser};

impl Parser<'_> {
    fn parse_number_expr(&mut self) -> ParseResult<Expr> {
        let value = self.parse_number()?;
        Ok(Expr::from(value))
    }

    fn parse_duration_expr(&mut self) -> ParseResult<Expr> {
        let duration = self.parse_duration()?;
        Ok(Expr::Duration(duration))
    }

    fn parse_single_expr(&mut self) -> ParseResult<Expr> {
        let expr = self.parse_single_expr_without_rollup_suffix()?;
        if self.peek_kind().is_rollup_start() {
            let re = self.parse_rollup_expr(expr)?;
            return Ok(re);
        }
        Ok(expr)
    }

    pub fn parse_expression(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_single_expr()?;
        loop {
            if self.at_end() {
                break;
            }
            let token = self.current_token()?;
            let mut op_token = token.kind;

            // Hack incoming:
            // there is some ambiguity because of how the lexer handles negative numbers. In other words
            // -25 is parsed as [-25] as opposed to [Operator(Minus), 25]. So for example `time()-1` is
            // parsed as [time(), -1]. So we need to check for this case here.
            let mut right_scalar: Option<Expr> = None;
            if token.kind == Token::Number {
                if let Ok(right) = parse_number(token.text) {
                    if right < 0_f64 {
                        // we have something like `time()-1000`
                        right_scalar = Some(Expr::from(right.abs()));
                        op_token = Token::OpMinus;
                    }
                }
            }

            if !op_token.is_operator() {
                return Ok(left);
            }

            let operator = Operator::try_from(op_token)?;

            self.bump();

            let mut modifier = BinModifier::default();

            if right_scalar.is_none() && self.at(&Token::Bool) {
                if !operator.is_comparison() {
                    let msg = format!("bool modifier cannot be applied to {operator}");
                    return Err(self.syntax_error(&msg));
                }
                modifier.return_bool = true;
                self.bump();
            }

            if self.at_set(&[Token::On, Token::Ignoring]) {
                self.parse_vector_match_modifier(&mut modifier)?;
                // join modifier
                let token = self.current_token()?;
                if [Token::GroupLeft, Token::GroupRight].contains(&token.kind) {
                    if operator.is_set_operator() {
                        let msg = format!("modifier {} cannot be applied to {operator}", token.text);
                        return Err(self.syntax_error(&msg));
                    }
                    self.parse_vector_match_cardinality(&mut modifier)?;
                }
            }

            let right = if let Some(right) = right_scalar {
                right
            } else {
                self.parse_single_expr()?
            };

            if self.at(&Token::KeepMetricNames) {
                self.bump();
                modifier.keep_metric_names = true;
            }

            // HACK: in PromQL, the `^` (pow) operator is right-associative, which is handled properly
            // below in the `balance_binary_op` function. However, this causes an ambiguity with
            // unary expressions. In other words, `-x^3` up to this point is parsed as `(-x)^3` as opposed
            // to `-(x^3)` which PromQL expects, so we need to handle this case here.
            if operator.is_right_associative() {
                left = match left {
                    Expr::UnaryOperator(uop) => Expr::BinaryOperator(BinaryExpr {
                        left: Box::new(Expr::from(0.0)),
                        right: uop.expr,
                        op: Operator::Sub,
                        modifier: None,
                    }),
                    Expr::NumberLiteral(num) if num.value < 0.0 => Expr::BinaryOperator(BinaryExpr {
                        left: Box::new(Expr::from(0.0)),
                        right: Box::new(Expr::from(num.value * -1.0)),
                        op: Operator::Sub,
                        modifier: None,
                    }),
                    _ => left,
                }
            }

            let be = BinaryExpr {
                left: Box::new(left),
                right: Box::new(right),
                op: operator,
                modifier: if modifier.is_default() {
                    None
                } else {
                    Some(std::mem::take(&mut modifier))
                },
            };

            left = balance_binary_op(be);
        }

        Ok(left)
    }

    fn parse_vector_match_modifier(&mut self, modifier: &mut BinModifier) -> ParseResult<()> {
        let tok = self.expect_one_of(&[Token::Ignoring, Token::On])?;
        let kind = tok.kind;
        let labels = self.parse_ident_list()?;

        modifier.matching = Some(VectorMatchModifier::new(labels, kind == Token::On));

        Ok(())
    }

    fn parse_vector_match_cardinality(&mut self, modifier: &mut BinModifier) -> ParseResult<()> {
        let tok = self.expect_one_of(&[Token::GroupLeft, Token::GroupRight])?;
        let kind = tok.kind;

        let labels = if !self.at(&Token::LeftParen) {
            // join modifier may ignore ident list.
            vec![]
        } else {
            self.parse_ident_list()?
        };

        let label_set = Labels::from(labels);

        modifier.card = match kind {
            Token::GroupLeft => VectorMatchCardinality::ManyToOne(label_set),
            Token::GroupRight => VectorMatchCardinality::OneToMany(label_set),
            _ => unreachable!(),
        };

        Ok(())
    }

    pub(super) fn parse_single_expr_without_rollup_suffix(&mut self) -> ParseResult<Expr> {
        use Token::*;

        let tok = self.current_token()?;
        match &tok.kind {
            StringLiteral => {
                let extracted = extract_string_value(tok.text)?;
                let value = Expr::string_literal(&extracted);
                self.bump();
                Ok(value)
            }
            Identifier => self.parse_ident_expr(),
            Number => self.parse_number_expr(),
            LeftParen => self.parse_parens_expr(),
            LeftBrace => self.parse_metric_expr(),
            Interval | RateInterval | Duration => self.parse_duration_expr(),
            OpPlus => self.parse_unary_plus_expr(),
            OpMinus => self.parse_unary_minus_expr(),
            _ => {
                if IDENT_LIKE_TOKENS.contains(&tok.kind) {
                    self.parse_ident_expr()
                } else {
                    Err(unexpected(
                        "",
                        &tok.kind.to_string(),
                        "Expr",
                        Some(&tok.span),
                    ))
                }
            }
        }
    }

    fn parse_unary_plus_expr(&mut self) -> ParseResult<Expr> {
        self.expect(&Token::OpPlus)?;
        let expr = self.parse_single_expr()?;
        /*
        let t = checkAST(p, &expr)?;
        match t {
            ReturnType::Scalar | ReturnType::InstantVector => Ok(expr),
            _ => {
                let msg = format!("unary Expr only allowed on expressions of type scalar or instant vector, got {:?}", t);
                Err(self.syntax_error(msg))
            }
        }
         */
        Ok(expr)
    }

    fn parse_unary_minus_expr(&mut self) -> ParseResult<Expr> {
        use ValueType::*;
        // assert(self.at(TokenKind::Minus)
        let span = self.last_token_range().unwrap();
        self.bump();
        let expr = self.parse_single_expr()?;

        let rt = expr.return_type();
        if !matches!(rt, InstantVector | Scalar) {
            let msg = format!(
                "unary Expr only allowed on expressions of type scalar or instant vector, got {:?}",
                rt
            );
            return Err(unexpected("", &rt.to_string(), &msg, Some(&span)));
        }

        let unary_expr = UnaryExpr::new(expr);
        Ok(Expr::UnaryOperator(unary_expr))
    }

    pub(super) fn parse_string_expr(&mut self) -> ParseResult<StringExpr> {
        let str = self.parse_string_expression(false)?;
        // todo: make sure
        Ok(str)
    }

    /// parses expressions starting with `identifier` token. 
    fn parse_ident_expr(&mut self) -> ParseResult<Expr> {
        use Token::*;

        fn handle_metric_expression(p: &mut Parser) -> ParseResult<Expr> {
            p.back();
            p.parse_metric_expr()
        }

        let name = self.expect_identifier_ex()?;

        // Look into the next token in order to determine how to parse
        // the current Expr.
        let kind = self.peek_kind();
        match kind {
            Eof | Offset => return handle_metric_expression(self),
            By | Without | LeftParen => {
                let is_left_paren = kind == LeftParen;
                self.back();
                if is_aggr_func(&name) {
                    return self.parse_aggr_func_expr();
                }
                if is_left_paren {
                    return self.parse_func_expr();
                }
                return self.parse_metric_expr();
            }
            LeftBrace | LeftBracket | RightParen | Comma | At | KeepMetricNames => {
                return handle_metric_expression(self);
            }
            _ => {
                if kind.is_operator() {
                    return handle_metric_expression(self);
                }
            }
        }

        let msg = format!("expecting identifier, found \"{}\"", &kind.to_string());
        Err(self.syntax_error(&msg))
    }
}

fn balance_binary_op(mut be: BinaryExpr) -> Expr {
    match be.left.as_ref() {
        Expr::BinaryOperator(left) => {
            let rp = be.op.precedence();
            let lp = left.op.precedence();
            if rp < lp {
                return Expr::BinaryOperator(be);
            }
            if rp == lp && !be.op.is_right_associative() {
                return Expr::BinaryOperator(be);
            }
            let mut bel = left.clone();
            be.left = bel.right;
            bel.right = Box::new(balance_binary_op(be));
            Expr::BinaryOperator(bel)
        }
        _ => Expr::BinaryOperator(be),
    }
}

fn is_aggr_func(name: &str) -> bool {
    if let Some(meta) = FunctionMeta::lookup(name) {
        return meta.is_aggregation();
    }
    false
}
