use std::collections::{HashSet};
use std::default::Default;
use std::ops::Deref;
use std::result::Result;
use std::str::FromStr;
use logos::Source;

use once_cell::sync::OnceCell;
use text_size::{TextRange};

use crate::ast::*;
use crate::functions::{AggregateFunction, BuiltinFunction, DataType};
use crate::lexer::{Lexer, parse_float, quote, Token, TokenKind, unescape_ident};
use crate::parser::expand_with::expand_with_expr;
use crate::parser::parse_error::{InvalidTokenError, ParseError};
use crate::parser::ParseResult;
use crate::parser::simplify::simplify_expr;

/// parser parses MetricsQL expression.
///
/// preconditions for all parser.parse* funcs:
/// - self.lex.token should point to the first token to parse.
///
/// post-conditions for all parser.parse* funcs:
/// - self.lex.token should point to the next token after the parsed token.
pub struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    cursor: usize,
    parsing_with: bool,
}

impl<'a> Parser<'a> {
    pub(crate) fn from_tokens(tokens: Vec<Token<'a>>) -> Self {
        let tokens: Vec<_> = tokens.into_iter().filter(|x| !x.kind.is_trivia()).collect();
        Self {
            cursor: 0,
            tokens,
            parsing_with: false,
        }
    }

    pub fn new(input: &'a str) -> Self {
        let lexer = Lexer::new(input);
        let tokens = lexer.collect();

        Parser::from_tokens(tokens)
    }

    fn next_token(&mut self) -> Option<&Token<'a>> {
        if self.is_eof() {
            return None
        }

        self.cursor += 1;
        self.tokens.get(self.cursor)
    }

    fn prev_token(&mut self) -> Option<&Token<'a>> {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        if self.cursor == 0 {
            return None;
        }

        let token = self.tokens.get(self.cursor)?;
        Some(token)
    }

    pub(crate) fn peek_token(&self) -> Option<&Token<'a>> {
        self.tokens.get(self.cursor)
    }

    pub(crate) fn current_token(&self) -> ParseResult<&Token<'a>> {
        match self.tokens.get(self.cursor) {
            Some(t) => {
                if t.kind.is_error_token() {
                    let error_msg = format!("{}: {}", t.kind, t.text);
                    return Err(ParseError::General(error_msg.to_string()));
                }
                Ok(t)
            },
            None => Err(ParseError::UnexpectedEOF)
        }
    }

    fn is_eof(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(crate) fn last_token_range(&self) -> Option<TextRange> {
        let index = if self.is_eof() {
            self.tokens.len() - 1
        } else {
            self.cursor
        };
        self.tokens.get(index).map(|Token { span, .. }| *span)
    }

    fn update_span(&self, span: &mut TextRange) -> bool {
        if let Some(end_span) = self.last_token_range() {
            span.intersect(end_span);
            return true;
        }
        false
    }

    pub fn parse_expression(&mut self) -> ParseResult<Expression> {
        parse_expression(self)
    }

    pub fn parse_label_filter(&mut self) -> ParseResult<LabelFilter> {
        let filter = self.parse_label_filter_expr()?;
        Ok(filter.to_label_filter())
    }

    pub(crate) fn expect(&mut self, kind: TokenKind) -> ParseResult<()> {
        if self.at(kind) {
            self.bump();
            Ok(())
        } else {
            Err(self.token_error(&[kind]))
        }
    }

    /// Consume the next token if it matches the expected token, otherwise return false
    pub fn consume_token(&mut self, expected: &TokenKind) -> bool {
        if self.at(*expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_token(&mut self, kind: TokenKind) -> ParseResult<&Token<'a>> {
        if self.at(kind) {
            self.cursor += 1;
            let tok = self.tokens.get(self.cursor - 1).unwrap();
            Ok(tok)
        } else {
            Err(self.token_error(&[kind]))
        }
    }

    fn expect_one_of(&mut self, kinds: &[TokenKind]) -> ParseResult<&Token<'a>> {
        if self.at_set(kinds) {
            // todo: weirdness to avoid borrowing
            self.cursor += 1;
            let tok = self.tokens.get(self.cursor - 1).unwrap();
            Ok(tok)
        } else {
            Err(self.token_error(kinds))
        }
    }

    fn token_error(&self, expected: &[TokenKind]) -> ParseError {
        let current_token = self.peek_token();

        let (found, range) = if let Some(Token { kind, span, .. }) = current_token {
            (Some(*kind), *span)
        } else {
            // If we’re at the end of the input we use the range of the very last token in the
            // input.
            (None, self.last_token_range().unwrap_or_default())
        };

        let inner = InvalidTokenError::new(expected, found, &range);

        ParseError::InvalidToken(inner)
    }

    fn parse_arg_list(&mut self) -> ParseResult<Vec<BExpression>> {
        use TokenKind::*;
        self.expect(LeftParen)?;
        self.parse_comma_separated(&[RightParen], |p| {
                Ok(Box::new(p.parse_expression()?))
            })
    }

    fn parse_ident_list(&mut self) -> ParseResult<Vec<String>> {
        use TokenKind::*;

        self.expect(LeftParen)?;
        self.parse_comma_separated(&[RightParen],|parser| {
            let tok = parser.expect_token(Ident)?;
            Ok(unescape_ident(tok.text))
        })

    }

    /// Parse a comma-separated list of 1+ items accepted by `F`
    pub fn parse_comma_separated<T, F>(&mut self, stop_tokens: &[TokenKind], mut f: F) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Parser<'a>) -> ParseResult<T>,
    {
        let mut values = Vec::with_capacity(4);
        loop {
            if self.at_set(stop_tokens) {
                self.bump();
                break
            }
            let item = f(self)?;
            values.push(item);
            let kind = self.peek_kind();
            if kind == TokenKind::Comma {
                self.bump();
                continue
            } else if stop_tokens.contains(&kind) {
                self.bump();
                break;
            } else {
                let mut expected = Vec::from(stop_tokens);
                expected.insert(0, TokenKind::Comma);
                return Err(self.token_error(&expected));
            }
        }
        Ok(values)
    }

    fn bump(&mut self) {
        if self.cursor < self.tokens.len() {
            self.cursor += 1;
        }
    }

    fn back(&mut self) -> &mut Self {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        self
    }

    pub(crate) fn at(&self, kind: TokenKind) -> bool {
        self.peek_kind() == kind
    }

    fn at_set(&self, set: &[TokenKind]) -> bool {
        let kind = self.peek_kind();
        set.contains(&kind )
    }

    pub(crate) fn at_end(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    fn peek_kind(&self) -> TokenKind {
        if self.at_end() {
            return TokenKind::Eof
        }
        let tok = self.tokens.get(self.cursor);
        tok.expect("BUG: invalid index out of bounds").kind
    }

    fn parse_single_expr(&mut self) -> ParseResult<Expression> {
        if self.at(TokenKind::With) {
            let with = parse_with_expr(self)?;
            return Ok(Expression::With(with));
        }
        let e = parse_single_expr_without_rollup_suffix(self)?;
        if self.peek_kind().is_rollup_start() {
            let re = parse_rollup_expr(self, e)?;
            return Ok(re);
        }
        Ok(e)
    }

    fn parse_at_expr(&mut self) -> ParseResult<Expression> {
        self.expect(TokenKind::At)?;
        let expr = parse_single_expr_without_rollup_suffix(self)?;
        // validate result type
        match expr.return_value() {
            ReturnValue::InstantVector | ReturnValue::Scalar => Ok(expr),
            ReturnValue::Unknown(cause) => {
                // todo: pass span
                Err(ParseError::InvalidExpression(cause.message))
            }
            _ => Err(ParseError::InvalidExpression(
                String::from("@ modifier expression must return a scalar or instant vector")
            )) // todo: have InvalidReturnType enum variant
        }
    }

    fn parse_duration(&mut self) -> ParseResult<DurationExpr> {
        let tok = self.current_token()?;
        let span = tok.span;
        match tok.kind {
            TokenKind::Duration => Ok(DurationExpr::new(tok.text, span)),
            TokenKind::Number => {
                // default to seconds if no unit is specified
                Ok(DurationExpr::new(format!("{}s", tok.text).as_str(), span))
            },
            _ => {
                return Err(
                    self.token_error(&[TokenKind::Number, TokenKind::Duration]),
                )
            },
        }
    }

    /// positive_duration_value returns positive duration in milliseconds for the given s
    /// and the given step.
    ///
    /// Duration in s may be combined, i.e. 2h5m or 2h-5m.
    ///
    /// Error is returned if the duration in s is negative.
    pub fn parse_positive_duration(&mut self) -> ParseResult<DurationExpr> {
        // Verify the duration in seconds without explicit suffix.
        let duration = self.parse_duration()?;
        let val = duration.duration(1);
        if val < 0 {
            Err(ParseError::InvalidDuration(duration.s))
        } else {
            Ok(duration)
        }
    }

    fn parse_limit(&mut self) -> ParseResult<usize> {
        self.expect(TokenKind::Limit)?;

        let v = parse_number(self)?;
        if v < 0.0 || v.is_nan() {
            // invalid value
            let msg = format!("LIMIT should be a positive integer. Found {} ", v);
            return Err(ParseError::General(msg));
        }
        Ok(v as usize)
    }

    fn parse_with_arg_expr(&mut self) -> ParseResult<WithArgExpr> {

        let tok = self.expect_token(TokenKind::Ident)?;
        let name = unescape_ident(tok.text);

        let args = if self.at(TokenKind::LeftParen) {
            // Parse func args.
            let args = self.parse_ident_list()?;

            // Make sure all the args have different names
            let mut m: HashSet<String> = HashSet::with_capacity(4);
            for arg in args.iter() {
                if m.contains(arg) {
                    let msg = format!("withArgExpr: duplicate arg name: {}", arg);
                    return Err(ParseError::General(msg));
                }
                m.insert(arg.clone());
            }

            args
        } else {
            vec![]
        };

        self.expect(TokenKind::Equal)?;
        let expr: Expression = match parse_expression(self) {
            Ok(e) => e,
            Err(e) => {
                let msg = format!("withArgExpr: cannot parse expression for {}: {:?}", name, e);
                return Err(ParseError::General(msg));
            }
        };

        Ok(WithArgExpr::new(name, expr, args))
    }

    fn parse_label_filter_expr(&mut self) -> ParseResult<LabelFilterExpr> {
        use TokenKind::*;

        let token = self.expect_token(Ident)?;
        let label = unescape_ident(token.text);

        let op: LabelFilterOp;

        match self.next_token() {
            None => {
                return Err(self.token_error(&[Equal, OpNotEqual, RegexEqual, RegexNotEqual]))
            }
            Some(t) => {
                match t.kind {
                    Equal => op = LabelFilterOp::Equal,
                    OpNotEqual => op = LabelFilterOp::NotEqual,
                    RegexEqual => op = LabelFilterOp::RegexEqual,
                    RegexNotEqual => op = LabelFilterOp::RegexNotEqual,
                    _ => {
                        return Err(self.token_error(&[Equal, OpNotEqual, RegexEqual, RegexNotEqual]))
                    }
                }
            }
        }

        let se = parse_string_expr(self)?;
        Ok(LabelFilterExpr::new(label, se, op))
    }
}

pub fn parse(input: &str) -> Result<Expression, ParseError> {
    let mut parser = Parser::new(input);
    let tok = parser.peek_kind();
    if tok == TokenKind::Eof {
        let msg = format!("cannot parse the first token {}", input);
        return Err(ParseError::General(msg));
    }
    let expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    let was = get_default_with_arg_exprs();
    match expand_with_expr(was, &expr) {
        Ok(expr) => simplify_expr(&expr),
        Err(e) => Err(e),
    }
}

/// Expands WITH expressions inside q and returns the resulting
/// PromQL without WITH expressions.
pub fn expand_with_exprs(q: &str) -> Result<String, ParseError> {
    let e = parse(q)?;
    Ok(format!("{}", e))
}

static DEFAULT_EXPRS: [&str; 4] = [
    // ru - resource utilization
    "ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100",
    // ttf - time to fuckup
    "ttf(freev) = smooth_exponential(
        clamp_max(clamp_max(-freev, 0) / clamp_max(deriv_fast(freev), 0), 365*24*3600),
        clamp_max(step()/300, 1)
    )",
    "median_over_time(m) = quantile_over_time(0.5, m)",
    "range_median(q) = range_quantile(0.5, q)"
];

fn get_default_with_arg_exprs() -> &'static [WithArgExpr; 4] {
    static INSTANCE: OnceCell<[WithArgExpr; 4]> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let was: [WithArgExpr; 4] =
            DEFAULT_EXPRS.map(|expr| {
                let res = must_parse_with_arg_expr(expr);
                res.unwrap()
            });

        if let Err(err) = check_duplicate_with_arg_names(&was) {
            panic!("BUG: {:?}", err)
        }
        was
    })
}

fn parse_function<'a>(p: &mut Parser<'a>, name: &str) -> ParseResult<Expression> {
    match BuiltinFunction::new(name) {
        Ok(pf) => {
            match pf {
                BuiltinFunction::Aggregate(_) => parse_aggr_func_expr(p),
                _ => parse_func_expr(p)
            }
        },
        Err(e) => Err(e)
    }
}

fn parse_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let token = p.expect_token(TokenKind::Ident)?;
    let name = unescape_ident(token.text);
    let mut span= token.span;

    let args = parse_arg_list_expr(p)?;

    let mut keep_metric_names = false;
    if p.at(TokenKind::KeepMetricNames) {
        keep_metric_names = true;
        p.update_span(&mut span);
        p.bump();
    }

    let mut fe = FuncExpr::new(&name, args, span)?;
    fe.keep_metric_names = keep_metric_names;

    validate_args(&fe.function, &fe.args)?;

    Ok(fe.cast())
}

fn parse_aggr_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let tok = p.current_token()?;
    let mut span = tok.span;

    let func: AggregateFunction = AggregateFunction::from_str(tok.text)?;

    fn handle_prefix(p: &mut Parser, ae: &mut AggrFuncExpr) -> ParseResult<()> {
        p.expect_one_of(&[TokenKind::By, TokenKind::Without])?;
        ae.modifier = Some(parse_aggregate_modifier(p)?);
        handle_args(p, ae)
    }

    fn handle_args(p: &mut Parser, ae: &mut AggrFuncExpr) -> ParseResult<()> {
        ae.args = p.parse_arg_list()?;

        validate_args(&BuiltinFunction::Aggregate(ae.function), &ae.args)?;
        let kind = p.peek_kind();
        // Verify whether func suffix exists.
        if ae.modifier.is_none() && kind.is_aggregate_modifier() {
            ae.modifier = Some(parse_aggregate_modifier(p)?);
        }

        if p.at(TokenKind::Limit) {
            ae.limit = p.parse_limit()?;
        }

        Ok(())
    }

    let mut ae: AggrFuncExpr = AggrFuncExpr::new(&func);
    p.bump();

    match p.peek_kind() {
        TokenKind::Ident => handle_prefix(p, &mut ae)?,
        TokenKind::LeftParen => handle_args(p, &mut ae)?,
        _=> return Err(p.token_error(&[TokenKind::Ident, TokenKind::LeftParen])),
    }

    p.update_span(&mut span);
    ae.span = span;

    Ok(Expression::Aggregation(ae))
}

fn parse_expression(p: &mut Parser) -> ParseResult<Expression> {
    let mut e = p.parse_single_expr()?;

    let start = e.span();

    loop {
        if p.at_end() {
            break;
        }
        let token = p.current_token()?;
        if !token.kind.is_operator() {
            return Ok(e);
        }

        let operator = BinaryOp::try_from(token.text)?;

        p.bump();

        let mut is_bool = false;
        if p.at(TokenKind::Bool) {
            if !operator.is_comparison() {
                let msg = format!("bool modifier cannot be applied to {}", operator);
                return Err(ParseError::General(msg));
            }
            is_bool = true;
            p.bump();
        }

        let mut group_modifier: Option<GroupModifier> = None;
        let mut join_modifier: Option<JoinModifier> = None;

        if p.at_set(&[TokenKind::On, TokenKind::Ignoring]) {
            group_modifier = Some(parse_group_modifier(p)?);
            // join modifier
            let token = p.current_token()?;
            if [TokenKind::GroupLeft, TokenKind::GroupRight].contains(&token.kind) {
                if operator.is_set_operator() {
                    let msg = format!("modifier {} cannot be applied to {}", token.text, operator);
                    return Err(ParseError::General(msg));
                }
                let join = parse_join_modifier(p)?;
                join_modifier = Some(join);
            }
        }

        let right = p.parse_single_expr()?;
        let span = start.cover(right.span());

        let mut be = BinaryOpExpr::new(operator, e, right)?;
        be.group_modifier = group_modifier;
        be.join_modifier = join_modifier;
        be.bool_modifier = is_bool;
        be.span = span;

        e = balance_binary_op(&be);
    }

    Ok(e)
}


#[inline]
fn balance_binary_op(be: &BinaryOpExpr) -> Expression {
    return match &be.left.as_ref() {
        Expression::BinaryOperator(left) => {
            let lp = left.op.precedence();
            let rp = be.op.precedence();
            let mut res = be.clone();

            if rp < lp || (rp == lp && !be.op.is_right_associative()) {
                return Expression::BinaryOperator(res);
            }
            std::mem::swap(&mut res.left, &mut res.right);
            let balanced = balance_binary_op(&res);
            let _ = std::mem::replace(&mut res.right, Box::new(balanced));

            res.left.cast()
        }
        _ => be.clone().cast(),
    };
}

fn parse_single_expr_without_rollup_suffix(p: &mut Parser) -> ParseResult<Expression> {
    use TokenKind::*;

    match p.peek_kind() {
        LiteralString => match parse_string_expr(p) {
            Ok(s) => Ok(Expression::String(s)),
            Err(e) => Err(e),
        },
        Ident => parse_ident_expr(p),
        Inf | NaN | Number => parse_number_expr(p),
        Duration => match p.parse_positive_duration() {
            Ok(s) => Ok(Expression::Duration(s)),
            Err(e) => Err(e),
        },
        LeftBrace => parse_metric_expr(p),
        LeftParen => parse_parens_expr(p),
        OpPlus => {
            // unary plus
            p.bump();
            p.parse_single_expr()
        },
        OpMinus => parse_unary_minus_expr(p),
        _ => {
            let valid: [TokenKind; 9] = [
                LiteralString,
                Ident,
                Number,
                Inf,
                NaN,
                Duration,
                LeftParen,
                LeftBrace,
                OpMinus,
            ];
            Err(p.token_error(&valid))
        }
    }
}

fn parse_number(p: &mut Parser) -> ParseResult<f64> {
    let tok = p.expect_one_of(&[TokenKind::Number, TokenKind::NaN, TokenKind::Inf])?;
    parse_float(tok.text)
}

fn parse_number_expr(p: &mut Parser) -> ParseResult<Expression> {
    let tok = p.expect_one_of(&[TokenKind::Number, TokenKind::NaN, TokenKind::Inf])?;
    let value = parse_float(tok.text)?;
    Ok(Expression::Number(NumberExpr::new(value, tok.span)))
}

fn parse_group_modifier(p: &mut Parser) -> Result<GroupModifier, ParseError> {
    let tok = p.current_token()?;
    let mut span: TextRange = tok.span;

    let op: GroupModifierOp;

    match tok.kind {
        TokenKind::Ignoring => op = GroupModifierOp::Ignoring,
        TokenKind::On => op = GroupModifierOp::On,
        _ => {
            return Err(p.token_error(&[TokenKind::Ignoring, TokenKind::On]));
        }
    }

    p.bump();

    let args = p.parse_ident_list()?;
    let mut res = GroupModifier::new(op, args, span);
    if p.update_span(&mut span) {
        res.span(span);
    }

    Ok(res)
}

fn parse_join_modifier(p: &mut Parser) -> ParseResult<JoinModifier> {
    let tok = p.current_token()?;
    let mut span: TextRange = tok.span;

    let op = match tok.kind {
        TokenKind::GroupLeft => JoinModifierOp::GroupLeft,
        TokenKind::GroupRight => JoinModifierOp::GroupRight,
        _ => {
            return Err(p.token_error(&[TokenKind::GroupLeft, TokenKind::GroupRight]));
        }
    };

    p.bump();

    let mut res = JoinModifier::new(op);
    if !p.at(TokenKind::LeftParen) {
        // join modifier may miss ident list.
    } else {
        res.labels = p.parse_ident_list()?;
    }
    if p.update_span(&mut span) {
        res.span(span);
    }
    Ok(res)
}

fn parse_aggregate_modifier(p: &mut Parser) -> ParseResult<AggregateModifier> {
    let op = match p.peek_kind() {
        TokenKind::By => AggregateModifierOp::By,
        TokenKind::Without => AggregateModifierOp::Without,
        _ => {
            return Err(p.token_error(&[TokenKind::By, TokenKind::Without]));
        }
    };

    let args = p.parse_ident_list()?;
    let res = AggregateModifier::new(op, args);

    Ok(res)
}

fn parse_ident_expr(p: &mut Parser) -> ParseResult<Expression> {
    use TokenKind::*;

    // Look into the next-next token in order to determine how to parse
    // the current expression.
    let tok = p.next_token();
    if tok.is_none() || tok.unwrap().kind == Offset {
        p.back();
        return parse_metric_expr(p);
    }

    let tok = tok.unwrap();

    // the following weirdness is to avoid issues with borrowing later
    let is_ident = tok.kind == Ident;
    let name = {
        if is_ident {
            tok.text.clone()
        } else {
            ""
        }
    };

    if is_ident {
        let _ = p.back();
        return match BuiltinFunction::new(name) {
            Ok(bf) => {
                match bf {
                    BuiltinFunction::Aggregate(_) => parse_aggr_func_expr(p),
                    _ => parse_func_expr(p)
                }
            },
            Err(_) => parse_metric_expr(p)
        }
    }

    if tok.kind.is_operator() {
        p.back();
        return parse_metric_expr(p);
    }

    match tok.kind {
        LeftParen => {
            // unwrap is safe here since LeftParen is the previous token
            // clone is to stop holding on to p
            let name = p.prev_token().unwrap().text.clone();
            parse_function(p, name)
        }
        LeftBrace | LeftBracket | RightParen | Comma | At => {
            p.back();
            parse_metric_expr(p)
        },
        _ => {
            const VALID: [TokenKind; 8] = [
                LeftParen,
                Ident,
                Offset,
                LeftBrace,
                LeftBracket,
                RightParen,
                Comma,
                At,
            ];
            Err(p.token_error(&VALID))
        }
    }
}

pub(crate) fn parse_unary_minus_expr(p: &mut Parser) -> ParseResult<Expression> {
    // assert(p.at(TokenKind::Minus)
    let mut span = p.last_token_range().unwrap();
    // Unary minus. Substitute `-expr` with `0 - expr`
    p.bump();
    let e = p.parse_single_expr()?;
    span = span.cover(e.span());
    let b = BinaryOpExpr::new_unary_minus(e, span)?;
    Ok(Expression::BinaryOperator(b))
}

pub(crate) fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expression> {
    let mut me= MetricExpr::default();

    let mut span = p.last_token_range().unwrap();
    if p.at(TokenKind::Ident) {
        let tok = p.current_token()?;

        let tokens = vec![quote(&unescape_ident(tok.text))];
        let value = StringExpr {
            s: "".to_string(),
            tokens: Some(tokens),
            span: tok.span.clone()
        };
        let lfe = LabelFilterExpr {
            label: NAME_LABEL.to_string(),
            value,
            op: LabelFilterOp::Equal,
        };
        me.label_filter_exprs.push(lfe);

        p.bump();
        if !p.at(TokenKind::LeftBrace) {
            return Ok(Expression::MetricExpression(me));
        }
    }
    me.label_filter_exprs = parse_label_filters(p)?;
    p.update_span(&mut span);
    me.span = span;
    Ok(Expression::MetricExpression(me))
}

fn parse_rollup_expr(p: &mut Parser, e: Expression) -> ParseResult<Expression> {
    let mut re = RollupExpr::new(e);
    let tok = p.current_token()?;
    let start = Some(tok.span);

    let mut at: Option<Expression> = None;
    if p.at(TokenKind::LeftBracket) {
        let (window, step, inherit_step) = parse_window_and_step(p)?;
        re.window = window;
        re.step = step;
        re.inherit_step = inherit_step;
    }

    if p.at(TokenKind::At) {
        at = Some(p.parse_at_expr()?);
    }

    if p.at(TokenKind::Offset) {
        re.offset = Some(parse_offset(p)?);
    }

    if p.at(TokenKind::At) {
        if at.is_some() {
            let msg = "RollupExpr: duplicate '@' token".to_string();
            return Err(ParseError::General(msg));
        }
        at = Some(p.parse_at_expr()?);
    }

    if let Some(v) = at {
        re.at = Some(Box::new(v))
    }

    re.span = update_range(p, start)?;

    Ok(Expression::Rollup(re))
}

fn parse_parens_expr(p: &mut Parser) -> ParseResult<Expression> {
    let list = p.parse_arg_list()?;
    Ok(Expression::from(list))
}

fn parse_arg_list_expr(p: &mut Parser) -> ParseResult<Vec<BExpression>> {
    use TokenKind::*;

    p.expect(LeftParen)?;
    p.parse_comma_separated(&[RightParen], |p| {
        let expr = p.parse_expression()?;
        Ok(Box::new(expr))
    })
}

/// parses `WITH (withArgExpr...) expr`.
fn parse_with_expr(p: &mut Parser) -> ParseResult<WithExpr> {
    use TokenKind::*;

    let start = p.last_token_range();

    p.expect(With)?;
    p.expect(LeftParen)?;

    let was_in_with = p.parsing_with;
    p.parsing_with = true;

    let was = p.parse_comma_separated(&[RightParen], Parser::parse_with_arg_expr)?;

    if !was_in_with {
        p.parsing_with = false
    }


    // end:
    check_duplicate_with_arg_names(&was)?;

    let expr = parse_expression(p)?;
    let span = update_range(p, start)?;
    Ok(WithExpr::new(expr, was, span))
}

fn parse_offset(p: &mut Parser) -> ParseResult<DurationExpr> {
    p.expect(TokenKind::Offset)?;
    p.parse_duration()
}

fn parse_window_and_step(
    p: &mut Parser,
) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool), ParseError> {
    p.expect(TokenKind::LeftBracket)?;

    let mut window: Option<DurationExpr> = None;

    if !p.at(TokenKind::Colon) {
        window = Some(p.parse_positive_duration()?);
    }
    let mut step: Option<DurationExpr> = None;
    let mut inherit_step = false;

    if p.at(TokenKind::Colon) {
        p.bump();
        // Parse step
        if p.at(TokenKind::RightBracket) {
            inherit_step = true;
        }
        if !p.at(TokenKind::RightBracket) {
            step = Some(p.parse_positive_duration()?);
        }
    }
    p.expect(TokenKind::RightBracket)?;

    Ok((window, step, inherit_step))
}

fn parse_string_expr(p: &mut Parser) -> ParseResult<StringExpr> {
    use TokenKind::*;

    let mut tokens = Vec::with_capacity(1);
    let mut tok = p.current_token()?;
    let mut span = tok.span;

    loop {
        match tok.kind {
            LiteralString => {
                if tok.text.len() == 2 {
                    tokens.push("".to_string());
                } else {
                    let slice = tok.text.slice(1 .. tok.text.len() - 1);
                    tokens.push( slice.unwrap().to_string() ); // ??
                }
                span = span.cover(tok.span)
            }
            Ident => {
                tokens.push(tok.text.to_string() );
                span = span.cover(tok.span)
            }
            _ => {
                // todo: better err
                return Err( ParseError::InvalidToken(InvalidTokenError{
                    expected: vec![LiteralString],
                    found: Some(tok.kind),
                    range: tok.span,
                    context: "string expr".to_string()
                }));
            }
        }

        match p.next_token() {
            Some(t) => {
                if t.kind != OpPlus {
                    break;
                }
            },
            None => break
        };

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.

        if let Some(t) = p.next_token() {
            tok = t;

            if tok.kind == LiteralString {
                // "s1" + "s2"
                continue;
            }

            if tok.kind != Ident {
                // "s" + unknownToken
                p.back();
                break
            }

            // Look after ident
            match p.next_token() {
                None => break,
                Some(t) => {
                    tok = t;
                    if tok.kind == LeftParen || tok.kind == LeftBrace {
                        p.back();
                        p.back();
                        // `"s" + m(` or `"s" + m{`
                        break;
                    }
                }
            }

        } else {
            break;
        }

        // "s" + ident
        tok = p.prev_token().unwrap();
    }

    Ok( StringExpr::from_tokens(tokens, span) )
}

pub(crate) fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use TokenKind::*;
    p.expect(LeftBrace)?;
    p.parse_comma_separated(&[RightBrace], |p| {
        p.parse_label_filter_expr()
    })
}

fn must_parse_with_arg_expr(s: &str) -> ParseResult<WithArgExpr> {
    let mut p = Parser::new(s);
    let tok = p.peek_kind();
    if tok == TokenKind::Eof {
        return Err(ParseError::UnexpectedEOF);
    }
    let expr = p.parse_with_arg_expr()?;
    if !p.is_eof() {
        let msg = format!("BUG: cannot parse {}: unparsed data", s);
        return Err(ParseError::General(msg));
    }
    Ok(expr)
}

fn check_duplicate_with_arg_names(was: &[WithArgExpr]) -> ParseResult<()> {
    let mut m: HashSet<String> = HashSet::with_capacity(was.len());

    for wa in was {
        if m.contains(&wa.name) {
            return Err(ParseError::DuplicateArgument(wa.name.clone()));
        }
        m.insert(wa.name.clone());
    }
    Ok(())
}

pub(crate) fn validate_args(func: &BuiltinFunction, args: &[BExpression]) -> ParseResult<()> {
    let expect = |actual: ReturnValue, expected: ReturnValue, index: usize| -> ParseResult<()> {
        // Note: we don't use == because we're blocked from deriving PartialEq on ReturnValue because
        // of the Unknown variant
        if actual.to_string() != expected.to_string() {
            return Err(ParseError::ArgumentError(
                format!("Invalid argument #{} to {}. {} expected", index, func, expected)
            ))
        }
        Ok(())
    };

    let validate_return_type = |return_type: ReturnValue, expected: DataType, index: usize| -> ParseResult<()> {
        match return_type {
            ReturnValue::Unknown(u) => {
                return Err(ParseError::ArgumentError(
                    format!("Bug: Cannot determine type of argument #{} to {}. {}", index, func, u.message)
                ))
            },
            _ => {}
        }
        match expected {
            DataType::RangeVector => {
                return expect(return_type, ReturnValue::RangeVector, index);
            }
            DataType::InstantVector => {
                return expect(return_type, ReturnValue::InstantVector, index);
            }
            DataType::Vector => {
                // ?? should we accept RangeVector and flatten ?
                return expect(return_type, ReturnValue::InstantVector, index);
            }
            DataType::Scalar => {
                if !return_type.is_operator_valid() {
                    return Err(ParseError::ArgumentError(
                        format!("Invalid argument #{} to {}. Scalar or InstantVector expected", index, func)
                    ))
                }
            }
            DataType::String => {
                return expect(return_type, ReturnValue::String, index);
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

        validate_return_type(arg.return_value(), expected, i)?
    }
    Ok(())
}

fn is_one_of(tok: Option<Token>, set: &[TokenKind]) -> bool {
    if let Some(token) = tok {
        set.contains(&token.kind)
    } else {
        false
    }
}

fn update_range(p: &Parser, start: Option<TextRange>) -> ParseResult<TextRange> {
    // Note: if we've made it thus far, start is Some
    let start = start.unwrap();
    let token = p.current_token()?;
    match start.intersect(token.span) {
        None => {
            Err(ParseError::General("Bug: error fetching range".to_string()))
        }
        Some(span) => Ok(span)
    }
}


#[cfg(test)]
mod tests {
    use crate::parser::parser::must_parse_with_arg_expr;

    #[test]
    fn test_must_parse_with_arg_expr() {
        let expr = "ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100";
        must_parse_with_arg_expr(expr).unwrap();
    }
}