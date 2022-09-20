use std::collections::HashSet;
use std::result::Result;
use std::str::FromStr;

use once_cell::sync::OnceCell;
use text_size::TextRange;

use crate::ast::*;
use crate::binaryop::eval_binary_op;
use crate::functions::{AggregateFunction, BuiltinFunction, is_aggr_func};
use crate::lexer::{Lexer, parse_float, quote, Token, TokenKind, unescape_ident};
use crate::parser::expand_with::expand_with_expr;
use crate::parser::parse_error::{InvalidTokenError, ParseError};
use crate::parser::ParseResult;

/// parser parses MetricsQL expression.
///
/// preconditions for all parser.parse* funcs:
/// - self.lex.token should point to the first token to parse.
///
/// post-conditions for all parser.parse* funcs:
/// - self.lex.token should point to the next token after the parsed token.
pub struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
    expected_kinds: Vec<TokenKind>,
    parsing_with: bool
}

impl Parser {
    pub(crate) fn from_tokens(tokens: Vec<Token>) -> Self {
        Self {
            expected_kinds: Vec::new(),
            cursor: 0,
            tokens,
            parsing_with: false
        }
    }

    pub fn new(input: &str) -> Self {
        let tokens: Vec<_> = Lexer::new(input).collect();
        Parser::from_tokens(tokens)
    }

    fn next_token(&mut self) -> Option<&Token> {
        self.eat_trivia();

        let token = self.tokens.get(self.cursor)?;
        self.cursor += 1;

        Some(token)
    }

    fn peek_kind_n(&mut self, n: u8) -> Option<TokenKind> {
        let save_cursor = self.cursor;
        let mut kind: Option<TokenKind> = None;
        for _ in 0..n {
            kind = self.peek_kind();
            if kind.is_none() {
                break;
            }
        }
        self.cursor = save_cursor;
        kind
    }

    pub(crate) fn peek_kind(&mut self) -> Option<TokenKind> {
        self.eat_trivia();
        self.peek_kind_raw()
    }

    pub(crate) fn peek_token(&mut self) -> Option<&Token> {
        self.eat_trivia();
        self.peek_token_raw()
    }

    fn at_trivia(&mut self) -> bool {
        self.peek_kind_raw().map_or(false, |x| x.is_trivia())
    }

    fn eat_trivia(&mut self) {
        while self.at_trivia() {
            self.cursor += 1;
        }
    }

    fn peek_token_raw(&mut self) -> Option<&Token> {
        self.tokens.get(self.cursor)
    }

    fn peek_kind_raw(&mut self) -> Option<TokenKind> {
        self.peek_token_raw().map(|Token { kind, .. }| *kind)
    }

    fn is_eof(&self) -> bool {
        self.cursor > self.tokens.len() - 1
    }

    pub(crate) fn last_token_range(&self) -> Option<TextRange> {
        self.tokens.last().map(|Token { range, .. }| *range)
    }

    fn parse(mut self) -> ParseResult<Expression> {
        // self.start();
        parse_expression(&mut self)
    }

    pub fn parse_expression(&mut self) -> ParseResult<Expression> {
        parse_expression(self)
    }

    pub fn parse_label_filter(&mut self) -> Result<LabelFilter, ParseError> {
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

    fn expect_token(&mut self, kind: TokenKind) -> ParseResult<Token> {
        if self.at(kind) {
            let tok = self.tokens.get(self.cursor).unwrap().clone();
            self.bump();
            Ok(tok)
        } else {
            Err(self.token_error(&[kind]))
        }
    }

    fn token_error(&mut self, expected: &[TokenKind]) -> ParseError {
        let current_token = self.peek_token();

        let (found, range) = if let Some(Token { kind, range, .. }) = current_token {
            (Some(*kind), *range)
        } else {
            // If we’re at the end of the input we use the range of the very last token in the
            // input.
            (None, self.last_token_range().unwrap())
        };

        let inner = InvalidTokenError::new(expected.to_vec(), found, range);

        if !self.at_end() {
            self.bump();
        }

        ParseError::InvalidToken(inner)
    }

    fn parse_arg_list(&mut self, need_parens: bool) -> ParseResult<Vec<BExpression>> {
        if need_parens {
            self.expect_token(TokenKind::LeftParen)?;
        }
        let args = self
            .parse_comma_separated(Parser::parse_expression)?
            .into_iter()
            .map(Box::new)
            .collect::<_>();
        if need_parens {
            self.expect_token(TokenKind::RightParen)?;
        }
        Ok(args)
    }

    fn parse_ident_list(&mut self) -> ParseResult<Vec<String>> {
        use TokenKind::*;

        self.expect_token(LeftParen)?;

        let idents = self.parse_comma_separated(|parser| {
            let tok = parser.expect_token(Ident)?;
            Ok(unescape_ident(tok.text().as_str()))
        });

        self.expect_token(RightParen)?;

        idents
    }

    /// Parse a comma-separated list of 1+ items accepted by `F`
    pub fn parse_comma_separated<T, F>(&mut self, mut f: F) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Parser) -> ParseResult<T>,
    {
        let mut values = Vec::with_capacity(1);
        loop {
            values.push(f(self)?);
            if !self.consume_token(&TokenKind::Comma) {
                break;
            }
        }
        Ok(values)
    }

    fn bump(&mut self) -> &mut Parser {
        self.expected_kinds.clear();
        self.next_token().unwrap();
        self
    }

    pub(crate) fn at(&mut self, kind: TokenKind) -> bool {
        self.expected_kinds.push(kind);
        self.peek() == Some(kind)
    }

    fn at_set(&mut self, set: &[TokenKind]) -> bool {
        self.expected_kinds.extend_from_slice(set);
        self.peek().map_or(false, |k| set.contains(&k))
    }

    fn expect_one_token_of(&mut self, set: &[TokenKind]) -> ParseResult<Token> {
        match self.optionally_expect_one_token_of(set) {
            Some(token) => Ok(token),
            None => Err(self.token_error(set)),
        }
    }

    fn optionally_expect_one_token_of(&mut self, set: &[TokenKind]) -> Option<Token> {
        let kind = self.peek_kind();
        if kind.map_or(false, |k| set.contains(&k)) {
            let tok = self.peek_token()?.clone();
            self.bump();
            return Some(tok);
        }
        None
    }

    fn expect_one_of(&mut self, set: &[TokenKind]) -> ParseResult<TokenKind> {
        let kind = self.peek();
        if kind.map_or(false, |k| set.contains(&k)) {
            self.bump();
            return Ok(kind.unwrap());
        }
        Err(self.token_error(set))
    }

    pub(crate) fn at_end(&mut self) -> bool {
        self.peek_kind().is_none()
    }

    fn peek(&mut self) -> Option<TokenKind> {
        self.peek_kind()
    }

    fn parse_single_expr(&mut self) -> ParseResult<Expression> {
        if self.consume_token(&TokenKind::With) && self.at(TokenKind::LeftParen) {
            let with = parse_with_expr(self)?;
            return Ok(Expression::With(with));
        }
        let e = parse_single_expr_without_rollup_suffix(self)?;
        if let Some(kind) = self.peek() {
            if kind.is_rollup_start() {
                let re = parse_rollup_expr(self, e)?;
                return Ok(re);
            }
        }
        Ok(e)
    }

    fn parse_at_expr(&mut self) -> ParseResult<Expression> {
        self.expect_token(TokenKind::At)?;
        parse_single_expr_without_rollup_suffix(self)
    }

    fn parse_duration(&mut self) -> ParseResult<DurationExpr> {
        let tok = self.expect_one_token_of(&[TokenKind::Number, TokenKind::Duration])?;
        let span = tok.range;
        match tok.kind {
            TokenKind::Duration => Ok(DurationExpr::new(tok.text(), span)),
            TokenKind::Number => {
                // default to seconds if no unit is specified
                Ok(DurationExpr::new(format!("{}s", tok.text()), span))
            }
            _ => unreachable!(),
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
        if v < 0.0 {
            // invalid value
            let msg = format!("LIMIT should be a positive integer. Found {} ", v);
            return Err(ParseError::General(msg));
        }
        Ok(v as usize)
    }

    fn parse_with_arg_expr(&mut self) -> ParseResult<WithArgExpr> {
        let args: Vec<String> = vec![];

        let tok = self.expect_token(TokenKind::Ident)?;
        let name = unescape_ident(tok.text().as_str());

        self.bump();

        if self.at(TokenKind::LeftParen) {
            // Parse func args.
            let args = self.parse_ident_list()?;

            // Make sure all the args have different names
            let mut m: HashSet<String> = HashSet::with_capacity(4);
            for arg in args {
                if m.contains(&arg) {
                    let msg = format!("withArgExpr: duplicate arg name: {}", arg);
                    return Err(ParseError::General(msg));
                }
                m.insert(arg);
            }
        }

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
        let label = unescape_ident(token.text().as_str());

        let mut op: LabelFilterOp = LabelFilterOp::Equal;

        match self.expect_one_of(&[Equal, NotEqual, RegexEqual, RegexNotEqual])? {
            Equal => op = LabelFilterOp::Equal,
            NotEqual => op = LabelFilterOp::NotEqual,
            RegexEqual => op = LabelFilterOp::RegexEqual,
            RegexNotEqual => op = LabelFilterOp::RegexNotEqual,
            _ => {
                // unreachable
            }
        }

        let se = parse_string_expr(self)?;
        Ok(LabelFilterExpr::new(label, se, op))
    }
}

pub fn parse(input: &str) -> Result<Expression, ParseError> {
    let mut parser = Parser::new(input);
    let tok = parser.peek();
    if tok.is_none() {
        let msg = format!("cannot parse the first token {}", input);
        return Err(ParseError::General(msg));
    }
    let expr = parser.parse_expression()?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    let was = get_default_with_arg_exprs();
    match expand_with_expr(was, &expr) {
        Ok(expr) => {
            let exp = remove_parens_expr(&expr);
            // if we have a parens expr, simplify it further
            let res = match exp {
                Expression::Parens(pe) => simplify_parens_expr(pe)?,
                _ => exp,
            };
            simplify_constants(&res)
        }
        Err(e) => Err(e),
    }
}

/// Expands WITH expressions inside q and returns the resulting
/// PromQL without WITH expressions.
pub fn expand_with_exprs(q: &str) -> Result<String, ParseError> {
    let e = parse(q)?;
    Ok(format!("{}", e))
}

static DEFAULT_EXPRS: [&str; 5] = [
    // ru - resource utilization
    "ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100",
    // ttf - time to fuckup
    "ttf(freev) = smooth_exponential(
        clamp_max(clamp_max(-freev, 0) / clamp_max(deriv_fast(freev), 0), 365*24*3600),
        clamp_max(step()/300, 1)
    )",
    "median_over_time(m) = quantile_over_time(0.5, m)",
    "range_median(q) = range_quantile(0.5, q)",
    "alias(q, name) = label_set(q, \"__name__\", name)",
];

fn get_default_with_arg_exprs() -> &'static [WithArgExpr; 5] {
    static INSTANCE: OnceCell<[WithArgExpr; 5]> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let was: [WithArgExpr; 5] =
            DEFAULT_EXPRS.map(|expr| must_parse_with_arg_expr(expr).unwrap());

        if let Err(err) = check_duplicate_with_arg_names(&was) {
            panic!("BUG: {:?}", err)
        }
        was
    })
}

fn parse_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let span = p.last_token_range();
    let token = p.expect_token(TokenKind::Ident)?;
    let name = unescape_ident(token.text().as_str());

    let args = p.parse_arg_list(true)?;

    let mut keep_metric_names = false;
    if p.at(TokenKind::KeepMetricNames) {
        keep_metric_names = true;
        p.bump();
    }

    let mut fe = FuncExpr::new(&name, args)?;
    fe.keep_metric_names = keep_metric_names;

    // todo: move this to function constructor
    validate_args(&fe.function, &fe.args)?;

    match span {
        Some(mut span) => {
            if let Some(x) = p.last_token_range() {
                span = Span::new(span.start(), x.end());
                fe.span = Some(span)
            }
        },
        None => {

        }
    }

    Ok(fe.cast())
}

fn parse_aggr_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let tok = p.peek_token().unwrap();

    let func: AggregateFunction = AggregateFunction::from_str(tok.text().as_str())?;

    fn handle_prefix(p: &mut Parser, ae: &mut AggrFuncExpr) -> Result<(), ParseError> {
        p.expect_one_of(&[TokenKind::By, TokenKind::Without])?;
        ae.modifier = Some(parse_aggregate_modifier(p)?);
        handle_args(p, ae)
    }

    fn handle_args(p: &mut Parser, ae: &mut AggrFuncExpr) -> ParseResult<()> {
        ae.args = p.parse_arg_list(false)?;

        validate_args(&BuiltinFunction::Aggregate(ae.function), &ae.args)?;
        let tok = p.peek_token().unwrap();
        // Verify whether func suffix exists.
        if ae.modifier.is_none() && tok.kind.is_aggregate_modifier() {
            ae.modifier = Some(parse_aggregate_modifier(p)?);
        }

        if p.at(TokenKind::Limit) {
            ae.limit = p.parse_limit()?;
        }

        Ok(())
    }

    let start: TextRange;
    {
        start = tok.range;
    }

    let mut ae: AggrFuncExpr = AggrFuncExpr::new(&func);
    p.bump();

    if p.at_set(&[TokenKind::Ident, TokenKind::LeftParen]) {
        match p.peek().unwrap() {
            TokenKind::Ident => handle_prefix(p, &mut ae)?,
            TokenKind::LeftParen => handle_args(p, &mut ae)?,
            _ => unreachable!(),
        }
    } else {
        return Err(p.token_error(&[TokenKind::Ident, TokenKind::LeftParen]));
    }

    ae.span = p.last_token_range().unwrap().intersect(start).unwrap();

    Ok(Expression::Aggregation(ae))
}

fn parse_expression(p: &mut Parser) -> ParseResult<Expression> {
    let e = p.parse_single_expr()?;
    let mut is_bool = false;

    if !p.at_end() {
        if let Some(kind) = p.peek() {
            if !kind.is_operator() {
                return Ok(e);
            }
        }

        let token = p.peek_token().unwrap();

        let binop = BinaryOp::try_from(token.text().as_str())?;

        p.bump();

        if p.at(TokenKind::Bool) {
            if !binop.is_comparison() {
                let msg = format!("bool modifier cannot be applied to {}", binop);
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
            if p.at_set(&[TokenKind::GroupLeft, TokenKind::GroupRight]) {
                let tok = p.peek_token().unwrap();
                if binop.is_binary_op_logical_set() {
                    let msg = format!("modifier {} cannot be applied to {}", tok.text(), binop);
                    return Err(ParseError::General(msg));
                }
                let join = parse_join_modifier(p)?;
                join_modifier = Some(join);
            }
        }

        let right = p.parse_single_expr()?;

        let mut be = BinaryOpExpr::new(binop, e, right);
        be.group_modifier = group_modifier;
        be.join_modifier = join_modifier;
        be.bool_modifier = is_bool;

        let expr = balance_binary_op(&be);
        return Ok(expr);
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
    let valid: [TokenKind; 8] = [
        QuotedString,
        Ident,
        Number,
        Duration,
        LeftParen,
        LeftBrace,
        OpMinus,
        OpMinus,
    ];

    if !p.at_set(&valid) {
        return Err(p.token_error(&valid));
    }

    let span_start = p.last_token_range().unwrap();

    let tok = p.peek_token().unwrap();
    match tok.kind {
        QuotedString => match parse_string_expr(p) {
            Ok(s) => Ok(Expression::String(s)),
            Err(e) => Err(e),
        },
        Ident => parse_ident_expr(p),
        Number => {
            let num = parse_number(p)?;
            p.bump();
            Ok(Expression::Number(NumberExpr::new(num)))
        },
        Duration => match p.parse_positive_duration() {
            Ok(s) => Ok(Expression::Duration(s)),
            Err(e) => Err(e),
        },
        LeftParen | LeftBrace => parse_parens_expr(p),
        OpPlus => p.parse_single_expr(),
        OpMinus => {
            // Unary minus. Substitute `-expr` with `0 - expr`
            p.bump();
            let e = p.parse_single_expr()?;
            let mut b = BinaryOpExpr::new_unary_minus(e);
            b.span = p.last_token_range().unwrap().intersect(span_start).unwrap();
            Ok(Expression::BinaryOperator(b))
        }
        _ => {
            unreachable!()
        }
    }
}

fn parse_number(p: &mut Parser) -> ParseResult<f64> {
    let tok = p.expect_token(TokenKind::Number)?;
    parse_float(tok.text().as_str())
}

fn parse_group_modifier(p: &mut Parser) -> Result<GroupModifier, ParseError> {
    match p.expect_one_of(&[TokenKind::Ignoring, TokenKind::On]) {
        Err(e) => Err(e),
        Ok(kind) => {
            let op = match kind {
                TokenKind::Ignoring => GroupModifierOp::Ignoring,
                TokenKind::On => GroupModifierOp::On,
                _ => {
                    unreachable!()
                }
            };

            let args = p.parse_ident_list()?;
            let res = GroupModifier::new(op, args);

            Ok(res)
        }
    }
}

fn parse_join_modifier(p: &mut Parser) -> Result<JoinModifier, ParseError> {
    match p.expect_one_of(&[TokenKind::GroupLeft, TokenKind::GroupRight]) {
        Err(e) => Err(e),
        Ok(token) => {
            let op = match token {
                TokenKind::GroupLeft => JoinModifierOp::GroupLeft,
                TokenKind::GroupRight => JoinModifierOp::GroupRight,
                _ => {
                    unreachable!()
                }
            };
            let mut res = JoinModifier::new(op);
            if !p.at(TokenKind::LeftParen) {
                // join modifier may miss ident list.
                Ok(res)
            } else {
                res.labels = p.parse_ident_list()?;
                Ok(res)
            }
        }
    }
}

fn parse_aggregate_modifier(p: &mut Parser) -> Result<AggregateModifier, ParseError> {
    match p.expect_one_of(&[TokenKind::By, TokenKind::Without]) {
        Err(e) => Err(e),
        Ok(kind) => {
            let op = match kind {
                TokenKind::By => AggregateModifierOp::By,
                TokenKind::Without => AggregateModifierOp::Without,
                _ => {
                    unreachable!()
                }
            };

            let args = p.parse_ident_list()?;
            let res = AggregateModifier::new(op, args);

            Ok(res)
        }
    }
}

fn parse_ident_expr(p: &mut Parser) -> ParseResult<Expression> {
    use TokenKind::*;

    // Look into the next-next token in order to determine how to parse
    // the current expression.
    let tok = p.peek_kind_n(2);
    if p.is_eof() || tok.is_none() || tok.unwrap() == Offset {
        return parse_metric_expr(p);
    }

    let kind = tok.unwrap();
    if kind.is_operator() {
        return parse_metric_expr(p);
    }

    match kind {
        LeftParen => {
            p.bump();
            let tok = p.peek_token().unwrap();
            if is_aggr_func(tok.text().as_str()) {
                parse_aggr_func_expr(p)
            } else {
                parse_func_expr(p)
            }
        }
        Ident => {
            let tok = p.peek_token().unwrap();
            if is_aggr_func(tok.text().as_str()) {
                return parse_aggr_func_expr(p);
            }
            parse_metric_expr(p)
        }
        Offset => parse_metric_expr(p),
        LeftBrace | LeftBracket | RightParen | Comma | At => parse_metric_expr(p),
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

pub(crate) fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expression> {
    let mut me = MetricExpr {
        label_filters: vec![],
        label_filter_exprs: vec![],
        span: None,
    };

    if p.at(TokenKind::Ident) {
        let tok = p.peek_token().unwrap();

        let tokens = vec![quote(&unescape_ident(tok.text().as_str()))];
        let value = StringExpr {
            s: "".to_string(),
            tokens: Some(tokens),
        };
        let lfe = LabelFilterExpr {
            label: "__name__".to_string(),
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
    Ok(Expression::MetricExpression(me))
}

fn parse_rollup_expr(p: &mut Parser, e: Expression) -> ParseResult<Expression> {
    let mut re = RollupExpr::new(e);
    let tok = p.peek_token().unwrap();
    re.span = Some(tok.range);

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

    Ok(Expression::Rollup(re))
}

fn parse_parens_expr(p: &mut Parser) -> ParseResult<Expression> {
    let exprs = p.parse_arg_list(true)?;
    Ok(Expression::Parens(ParensExpr::new(exprs)))
}

/// parses `WITH (withArgExpr...) expr`.
fn parse_with_expr(p: &mut Parser) -> Result<WithExpr, ParseError> {
    use TokenKind::*;

    p.expect_token(With)?;
    p.expect_token(LeftParen)?;

    let was_in_with = p.parsing_with;
    p.parsing_with = true;

    let was = p.parse_comma_separated(Parser::parse_with_arg_expr)?;

    if !was_in_with {
        p.parsing_with = false
    }

    p.expect_token(RightParen)?;

    // end:
    check_duplicate_with_arg_names(&was)?;

    let expr = parse_expression(p)?;
    Ok(WithExpr::new(expr, was))
}

fn parse_offset(p: &mut Parser) -> ParseResult<DurationExpr> {
    p.expect(TokenKind::Offset)?;
    p.parse_duration()
}

fn parse_window_and_step(
    p: &mut Parser,
) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool), ParseError> {
    p.expect_token(TokenKind::LeftBracket)?;

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
    p.expect_token(TokenKind::RightBracket)?;

    Ok((window, step, inherit_step))
}

fn parse_string_expr(p: &mut Parser) -> ParseResult<StringExpr> {
    use TokenKind::*;
    let mut se = StringExpr::new("");
    let mut tokens = Vec::with_capacity(1);

    let mut tok = p.peek_token().unwrap();
    loop {
        match tok.kind {
            QuotedString | Ident => {
                tokens.push(tok.text());
            }
            _ => {
                let msg = format!("stringExpr: unexpected token {}; want string", tok.text());
                return Err(ParseError::General(msg));
            }
        }

        tok = p.peek_token().unwrap();
        if tok.kind != OpPlus {
            break;
        }

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.
        tok = p.peek_token().unwrap();
        if tok.kind == QuotedString {
            // "s1" + "s2"
            continue;
        }
        if tok.kind != Ident {
            // "s" + unknownToken
            break;
        }
        // Look after ident
        tok = p.peek_token().unwrap();
        if tok.kind == LeftParen || tok.kind == LeftBrace {
            // `"s" + m(` or `"s" + m{`
            break;
        }
        // "s" + ident
        tok = p.peek_token().unwrap();
    }

    se.tokens = Some(tokens);
    Ok(se)
}

pub(crate) fn parse_label_filters(p: &mut Parser) -> Result<Vec<LabelFilterExpr>, ParseError> {
    use TokenKind::*;
    p.expect(LeftBrace)?;
    let lfes = p.parse_comma_separated(Parser::parse_label_filter_expr)?;
    p.expect(RightBrace)?;

    Ok(lfes)
}

/// removes parensExpr for (Expr) case.
fn remove_parens_expr(e: &Expression) -> Expression {
    fn remove_parens_args(args: &[BExpression]) -> Vec<BExpression> {
        return args
            .iter()
            .map(|x| Box::new(remove_parens_expr(x)))
            .collect();
    }

    match e {
        Expression::Rollup(re) => {
            let expr = remove_parens_expr(&*re.expr);
            let at: Option<BExpression> = match &re.at {
                Some(at) => {
                    let expr = remove_parens_expr(at);
                    Some(Box::new(expr))
                }
                None => None,
            };
            let mut res = re.clone();
            res.at = at;
            res.expr = Box::new(expr);
            Expression::Rollup(res)
        }
        Expression::BinaryOperator(be) => {
            let left = remove_parens_expr(&be.left);
            let right = remove_parens_expr(&be.right);
            let mut res = be.clone();
            res.left = Box::new(left);
            res.right = Box::new(right);
            Expression::BinaryOperator(res)
        }
        Expression::Aggregation(agg) => {
            let mut expr = agg.clone();
            expr.args = remove_parens_args(&agg.args);
            Expression::Aggregation(expr)
        }
        Expression::Function(f) => {
            let mut res = f.clone();
            res.args = remove_parens_args(&res.args);
            Expression::Function(res)
        }
        Expression::Parens(parens) => {
            let mut res = parens.clone();
            res.expressions = remove_parens_args(&res.expressions);
            Expression::Parens(res)
        }
        _ => e.clone(),
    }
}

fn simplify_parens_expr(expr: ParensExpr) -> ParseResult<Expression> {
    if expr.len() == 1 {
        let res = *expr.expressions[0].clone();
        return Ok(res);
    }
    // Treat parensExpr as a function with empty name, i.e. union()
    let fe = FuncExpr::new("union", expr.expressions)?;
    Ok(Expression::Function(fe))
}

// todo: use a COW?
fn simplify_constants(expr: &Expression) -> ParseResult<Expression> {
    match expr {
        Expression::Rollup(re) => {
            let mut clone = re.clone();
            let expr = simplify_constants(&re.expr)?;
            clone.expr = Box::new(expr);
            match &re.at {
                Some(at) => {
                    let simplified = simplify_constants(at)?;
                    clone.at = Some(Box::new(simplified));
                }
                None => {}
            }
            Ok(Expression::Rollup(clone))
        }
        Expression::BinaryOperator(be) => {
            let left = simplify_constants(&*be.left)?;
            let right = simplify_constants(&*be.right)?;

            match (left, right) {
                (Expression::Number(ln), Expression::Number(rn)) => {
                    let n = eval_binary_op(ln.value, rn.value, be.op, be.bool_modifier);
                    Ok(Expression::from(n))
                }
                (Expression::String(left), Expression::String(right)) => {
                    if be.op == BinaryOp::Add {
                        let val = format!("{}{}", left.s, right.s);
                        return Ok(Expression::from(val))
                    }
                    let n = if string_compare(&left.s, &right.s, be.op) {
                        1.0
                    } else if !be.bool_modifier {
                        f64::NAN
                    } else {
                        0.0
                    };
                    Ok(Expression::from(n))
                }
                _ => Ok(expr.clone().cast()),
            }
        }
        Expression::Aggregation(agg) => {
            let mut res = agg.clone();
            res.args = simplify_args_constants(&res.args)?;
            Ok(Expression::Aggregation(res))
        }
        Expression::Function(fe) => {
            let mut res = fe.clone();
            res.args = simplify_args_constants(&res.args)?;
            Ok(Expression::Function(res))
        }
        Expression::Parens(parens) => {
            let args = simplify_args_constants(&parens.expressions)?;
            if parens.len() == 1 {
                let expr = args.into_iter().next();
                return Ok(*expr.unwrap());
            }
            // Treat parensExpr as a function with empty name, i.e. union()
            Ok(Expression::Function(FuncExpr::new("union", args)?))
        }
        _ => Ok(expr.clone().cast()),
    }
}

fn simplify_args_constants(args: &[BExpression]) -> ParseResult<Vec<BExpression>> {
    let mut res: Vec<BExpression> = Vec::with_capacity(args.len());
    for arg in args {
        let simple = simplify_constants(arg)?;
        res.push(Box::new(simple));
    }

    Ok(res)
}

fn must_parse_with_arg_expr(s: &str) -> Result<WithArgExpr, ParseError> {
    let mut p = Parser::new(s);
    let tok = p.peek();
    if tok.is_none() {
        let msg = format!("BUG: cannot find first token in {}", s);
        return Err(ParseError::General(msg));
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

fn string_compare(a: &str, b: &str, op: BinaryOp) -> bool {
    match op {
        BinaryOp::Eql => a == b,
        BinaryOp::Neq => a != b,
        BinaryOp::Lt => a < b,
        BinaryOp::Gt => a > b,
        BinaryOp::Lte => a <= b,
        BinaryOp::Gte => a >= b,
        _ => panic!("unexpected operator {} in string comparison", op),
    }
}

pub(crate) fn validate_args(func: &BuiltinFunction, args: &[BExpression]) -> ParseResult<()> {
    // validate function args
    let sig = func.signature();
    sig.validate_arg_count(&func.name(), args.len())?;
    // todo: validate types

    let (arg_types, _) = sig.expand_types();

    for (i, arg)  in args.iter().enumerate()  {
        let expected = arg_types[i];
        // let actual = arg.return_type();

    }
    Ok(())
}