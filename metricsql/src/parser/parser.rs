use std::collections::{HashMap, HashSet};
use crate::error::{Error, Result};
use crate::{eval_binary_op, types};
use crate::lexer::{Lexer, parse_float, quote, Token, TokenKind, unescape_ident};
use crate::parser::aggr::{is_aggr_func};
use crate::parser::expand_with::expand_with_expr;
use crate::types::*;
use once_cell::sync::OnceCell;

// parser parses MetricsQL expression.
//
// preconditions for all parser.parse* funcs:
// - self.lex.token should point to the first token to parse.
//
// post-conditions for all parser.parse* funcs:
// - self.lex.token should point to the next token after the parsed token.
pub struct Parser<'a> {
    lex: Lexer<'a>
}

pub fn parse(input: &str) -> Result<Expression> {
    let mut parser = Parser::new(input);
    let tok = parser.next();
    if tok.is_none() {
        let msg = format!("cannot parse the first token {}", input);
        return Err(Error::new(msg));
    }
    let expr = parser.parse_expression()?;
    if !parser.eof() {
        let msg = format!("unparsed data {}", parser.lex.token);
        return Err(msg.into());
    }
    let was = get_default_with_arg_exprs()?;
    match expand_with_expr(was, &expr) {
        Ok(expr) => {
            remove_parens_expr(expr);
            // if we have a parens expr, simplify it further
            let mut res = match expr {
                Expression::Parens(pe) => {
                    simplify_parens_expr(pe)
                },
                _ => expr
            };
            let simple = simplify_constants(&res);
            Ok(simple)
        },
        Err(e) => {
            let msg = format!("cannot expand with expr {}", e);
            Err(Error::new(msg))
        }
    }
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
        let was: [WithArgExpr; 5] = DEFAULT_EXPRS.map(|expr| must_parse_with_arg_expr(expr).unwrap() );

        match check_duplicate_with_arg_names(&was) {
            Err(err) => {
                panic!("BUG: {}", err)
            }
            _ => ()
        }

        was
    })
}



impl<'a> Parser<'a> {
    pub fn new<S: Into<String>>(input: S) -> Self {
        Parser {
            lex: Lexer::new(input)
        }
    }

    // todo: restrict visibility to lex module.
    pub fn next(&mut self) -> Option<Token> {
        Some(self.lex.next()?)
    }

    pub fn is_eof(&self) -> bool {
        self.lex.isEOF()
    }

    pub fn parse_expression(&mut self) -> Result<Expression> {
        let mut expr = parse_expression(&self.lex)?;
        if !self.is_eof() {
            return Err(Error::new(None));
        }
        Ok(expr)
    }

    pub fn parse_with_expr(&mut self) -> Result<WithExpr> {
        let expr = parse_with_expr(&self.lex)?;
        Ok(expr)
    }

    pub fn parse_label_filter(&mut self) -> Result<LabelFilter> {
        let filter = parse_label_filter_expr(&self.lex)?;
        Ok(filter.to_label_filter())
    }

    pub fn parse_duration(&self) -> Result<DurationExpr> {
        parse_duration(&self.lex)
    }
}

fn parse_func_expr(mut lex: &Lexer) -> Result<FuncExpr> {
    let mut tok = lex.next().unwrap();
    if tok.kind != TokenKind::Ident {
        let msg = format!("funcExpr: unexpected token {}; want function name", tok.text);
        return Err(Error::new(msg));
    }

    let name = unescape_ident(tok.text);
    tok = lex.next().unwrap();
    if tok.kind != TokenKind::LeftParen {
        let msg = format!("funcExpr: unexpected token {}; want '('", lex.token);
        return Err(Error::new(msg));
    }
    let args = parse_arg_list(lex)?;

    let mut keep_metric_names = false;
    tok = lex.token().unwrap();
    if tok.kind == TokenKind::KeepMetricNames {
        keep_metric_names = true;
        lex.next()?;
    }
    let mut fe = FuncExpr::new(name, args);
    fe.keep_metric_names = keep_metric_names;

    Ok(fe)
}

fn parse_arg_list(mut lex: &Lexer) -> Result<Vec<Expression>> {
    let mut tok = consume(lex, "argList", TokenKind::LeftParen, "(")?;
    let mut args = Vec::new();

    while !lex.is_eof() && tok.kind != TokenKind::RightParen {
        let expr = parse_expression(lex)?;
        args.push(expr);
        tok = lex.token().unwrap();
        if tok.kind != TokenKind::Comma {
            break;
        }

        tok = lex.next().unwrap();
    }
    Ok(args)
}

fn parse_aggr_func_expr(mut lex: &Lexer) -> Result<AggrFuncExpr> {
    let mut tok = lex.token().unwrap();
    if !is_aggr_func(tok.text) {
        let msg = format!("AggrFuncExpr: unexpected token {}; want aggregate func", tok.text);
        return Err(Error::new(msg));
    }

    fn handle_prefix(mut lex: &Lexer, ae: &mut AggrFuncExpr) -> Result<()> {
        let tok = lex.token().unwrap();
        if !tok.kind.is_aggregate_modifier() {
            let msg = format!("AggrFuncExpr: unexpected token {}; want aggregate func modifier", tok.text);
            return Err(Error::new(msg));
        }
        ae.modifier = Some(parse_aggregate_modifier(lex)?);

        handle_args(lex, ae)
    }

    fn handle_args(mut lex: &Lexer, ae: &mut AggrFuncExpr) -> Result<()> {
        ae.args = parse_arg_list(lex)?;
        let tok = lex.token().unwrap();
        // Verify whether func suffix exists.
        if ae.modifier.is_none() && tok.kind.is_aggregate_modifier() {
            ae.modifier = Some(parse_aggregate_modifier(lex)?);
        }

        parse_limit(lex, ae)
    }

    fn parse_limit(mut lex: &Lexer, ae: &mut AggrFuncExpr) -> Result<()> {
        lex.next()?;
        let tok = lex.token().unwrap();
        let mut limit: usize = 0;
        match tok.text.parse() {
            Ok(f) => {
                if limit < 0 {
                    let msg = format!("limit must be positive, found {}", tok.text);
                    return Err(Error::new(msg));
                }
                limit = f;
            },
            Err(e) => {
                let msg = format!("cannot parse limit {}: {}", tok.text, e);
                return Err(Error::new(msg));
            }
        }
        lex.next()?;
        ae.limit = limit;
        Ok(())
    }

    let name = unescape_ident(tok.text).to_lowercase().as_str();
    let mut ae: AggrFuncExpr = AggrFuncExpr::new(name);

    tok = lex.next().unwrap();

    match tok.kind {
        TokenKind::Ident => {
            handle_prefix(&lex, &mut ae)?;
        },
        TokenKind::LeftParen => {
            handle_args(&lex, &mut ae)?;
        },
        _ => {
            let msg = format!("AggrFuncExpr: unexpected token {}; want '('", tok.text);
            return Err(Error::new(msg));
        }
    }

    Ok(ae)
}

fn parse_expression(mut lex: &Lexer) -> Result<impl ExpressionNode> {
    let mut e = parse_single_expr(lex)?;
    let mut bool = false;

    while !lex.is_eof() {
        let token = lex.next().unwrap();
        if !token.kind.is_operator() {
            return Ok(e);
        }

        let binop = BinaryOp::try_from(token.text)?;

        let mut tok = lex.next().unwrap();

        if tok.kind == TokenKind::Bool {
            if !binop.is_comparison() {
                let msg = format!("bool modifier cannot be applied to {}", binop);
                return Err(Error::new(msg));
            }
            bool = true;
            tok = lex.next().unwrap();
        }

        let mut group_modifier: Option<GroupModifier> = None;
        let mut join_modifier: Option<JoinModifier> = None;

        if tok.kind.is_group_modifier() {
            group_modifier = Some(parse_group_modifier(&mut lex)?);
            tok = lex.next().unwrap();
            if tok.kind.is_join_modifier() {
                if binop.is_binary_op_logical_set() {
                    let msg = format!("modifier {} cannot be applied to {}", tok.text, binop);
                    return Err(Error::new(msg));
                }
                let join = parse_join_modifier(lex)?;
                join_modifier = Some(join);
            }
        }

        let right = parse_single_expr(lex)?;

        let mut be = BinaryOpExpr::new(binop, e, right);
        be.group_modifier = group_modifier;
        be.join_modifier = join_modifier;

        return Ok(be.balance());
    }

    return Ok(e);
}

pub fn parse_single_expr(mut lex: &Lexer) -> Result<impl ExpressionNode> {
    let mut tok = lex.token().unwrap();
    if tok.kind == TokenKind::With {
        let next_token = lex.peek();
        match next_token {
            Some(..) => {
                if tok.kind == TokenKind::LeftParen {
                    return parse_with_expr(lex);
                }
            },
            None => {
                return Err(Error::new("parse_single_expr: unexpected end of stream"))
            }
        }
        tok = lex.peek().unwrap();
        if tok.kind == TokenKind::LeftParen {
            return Ok(parse_with_expr(lex)?);
        }
    }
    let e = parse_single_expr_without_rollup_suffix(lex)?;
    let tok = lex.token().unwrap();
    if !tok.kind.is_rollup_start() {
        // There is no rollup expression.
        return Ok(e);
    }
    return parse_rollup_expr(lex, e);
}

fn parse_single_expr_without_rollup_suffix(mut lex: &Lexer) -> Result<impl ExpressionNode> {
    let mut token = lex.next().unwrap();

    match token.kind {
        TokenKind::QuotedString => {
            return parse_string_expr(lex);
        }
        TokenKind::Ident => {
            return parse_ident_expr(lex);
        },
        TokenKind::Number => {
            let num = parse_float(token.text)?;
            return Ok( NumberExpr::new(num) );
        },
        TokenKind::Duration => {
            return parse_positive_duration(lex);
        },
        TokenKind::LeftParen |
        TokenKind::LeftBrace => parse_parens_expr(lex),
        TokenKind::OpPlus => {
            // Unary plus
            lex.next()?;
            parse_single_expr(lex)
        },
        TokenKind::OpMinus => {
            // Unary minus. Substitute `-expr` with `0 - expr`
            lex.next()?;
            let e = parse_single_expr(lex)?;
            let lhs = Expression::cast(NumberExpr::new(0.0)).unwrap();
            let be = BinaryOpExpr::new(BinaryOp::Sub, lhs, e);
            Ok(be)
        },
        _ => {
            let msg = format!("singleExpr: unexpected token {}; want '(', '{{', '-', '+'", lex.token);
            Err(Error::new(msg))
        }
    }
}

fn parse_group_modifier(mut lex: &Lexer) -> Result<GroupModifier> {
    let mut token = lex.token().unwrap();
    let op = match token.kind {
        TokenKind::Ignoring => {
            GroupModifierOp::Ignoring
        },
        TokenKind::On => {
            GroupModifierOp::On
        },
        _  => {
            let msg = format!("GroupModifier: unexpected token {}; want 'on' or 'ignoring", token.text);
            return Err(Error::new(msg));
        }
    };

    token = lex.next().unwrap();
    let mut res = GroupModifier::new(op.unwrap());
    let args = parse_ident_list(lex)?;
    res.set_labels(args);

    return Ok(res);
}

fn parse_join_modifier(mut lex: &Lexer) -> Result<JoinModifier> {
    let mut token = lex.token().unwrap();

    let op = match token.kind {
        TokenKind::GroupLeft => {
            JoinModifierOp::GroupLeft
        },
        TokenKind::GroupRight => {
            JoinModifierOp::GroupRight
        },
        _ => {
            let msg = format!("joinModifier: unexpected token {}; want 'group_left' or 'group_right", token.text);
            return Err(Error::from(msg));
        }
    };
    token = lex.next().unwrap();

    let mut res = JoinModifier::new(op);
    if token.kind != TokenKind::LeftParen {
        // join modifier may miss ident list.
        Ok(res)
    } else {
        res.labels = parse_ident_list(lex)?;
        Ok(res)
    }
}

fn parse_aggregate_modifier(mut lex: &Lexer) -> Result<AggregateModifier> {
    let mut token = lex.token().unwrap();

    let op = match token.kind {
        TokenKind::By => {
            AggregateModifierOp::By
        },
        TokenKind::Without => {
            AggregateModifierOp::Without
        },
        _ => {
            let msg = format!("aggregateModifier: unexpected token {}; want 'group_left' or 'group_right", token.text);
            return Err(Error::from(msg));
        }
    };
    token = lex.next().unwrap();

    let res = AggregateModifier::new(op);
    let args = parse_ident_list(lex)?;
    res.set_labels(&args);
    Ok(res)
}

fn parse_ident_expr(mut lex: &Lexer) -> Result<impl ExpressionNode> {
    use TokenKind::*;

    // Look into the next-next token in order to determine how to parse
    // the current expression.
    let mut tok = lex.peek().unwrap();
    if lex.is_eof() || tok.kind == Offset {
        return parse_metric_expr(lex);
    }
    if tok.kind.is_operator() {
        return parse_metric_expr(lex);
    }
    match tok.kind {
        LeftParen => {
            tok = lex.next().unwrap();
            return if is_aggr_func(tok.text) {
                parse_aggr_func_expr(lex)
            } else {
                parse_func_expr(lex)
            }
        },
        Ident => {
            if is_aggr_func(tok.text) {
                return parse_aggr_func_expr(lex);
            }
            return parse_metric_expr(lex);
        },
        Offset => {
            return parse_metric_expr(lex);
        },
        LeftBrace | LeftBracket | RightParen | Comma | At => {
            return parse_metric_expr(lex)
        },
        _ => {
            let msg = format!("IdentExpr: unexpected token {}; want '(', '{{', '[', ')', ',' or '@'", tok.text);
            Err(Error::new(msg))
        }
    }
 }

fn parse_metric_expr(mut lex: &Lexer) -> Result<MetricExpr> {
    let mut me = MetricExpr {
        label_filters: vec![],
        label_filter_exprs: vec![],
        span: lex.span()
    };
    let mut tok = lex.token().unwrap();
    if tok.kind == TokenKind::Ident {
        let tokens = vec![quote(&unescape_ident(tok.text))];
        let value = StringExpr { s: "".as_str(), tokens: Some(tokens) };
        let lfe = LabelFilterExpr { label: "__name__", value, op: types::LabelFilterOp::Equal };
        me.label_filter_exprs.push(lfe);

        tok = lex.next().unwrap();
        if tok.kind != TokenKind::LeftBrace {
            return Ok(me);
        }
    }
    me.label_filter_exprs = parse_label_filters(lex)?;
    Ok(me)
}

fn parse_rollup_expr(mut lex: &Lexer, e: Expression) -> Result<impl ExpressionNode> {
    let mut re = RollupExpr::new(e);
    let mut tok = lex.token().unwrap();
    re.span = Some(tok.span());

    if tok.kind == TokenKind::LeftBrace {
        let (window, step, inherit_step) = parse_window_and_step(&mut lex);
        re.window = window;
        re.step = step;
        re.inherit_step = inherit_step;

        tok = lex.token().unwrap();
        if tok.kind != TokenKind::SymbolAt && tok.kind != TokenKind::Offset {
            return Ok(re);
        }
    }
    tok = lex.token().unwrap();
    if tok.kind == TokenKind::At {
        re.set_at( parse_at_expr(lex)? );
    }
    tok = lex.token().unwrap();
    if tok.kind == TokenKind::Offset {
        re.offset = Some(parse_offset(lex)?);
    }
    tok = lex.token().unwrap();
    if tok.kind == TokenKind::At {
        if re.at.is_some() {
            let msg = format!("RollupExpr: duplicate '@' token");
            return Err(Error::new(msg));
        }
        re.set_at( parse_at_expr(lex)? );
    }

    Ok(re)
}

fn parse_parens_expr(mut lex: &Lexer) -> Result<ParensExpr> {
    let mut tok = consume(lex, "parensExpr", TokenKind::LeftParen, "(")?;

    let mut exprs: Vec<Expression> = vec![];
    while !lex.is_eof() && tok.kind != TokenKind::RightParen {
        let expr = parse_expression(lex)?;
        exprs.push(expr);
        tok = lex.token().unwrap();
        match tok.kind {
            TokenKind::Comma => continue,
            TokenKind::RightParen => break,
            _ => {
                return Err(Error::new(format!("parensExpr: unexpected token {}; want ',' or ')'", lex.token)));
            }
        }
    }
    lex.next()?;
    Ok( ParensExpr::new(exprs) )
}


fn parse_with_expr(mut lex: &Lexer) -> Result<WithExpr> {
    let mut token = consume(lex, "withExpr", TokenKind::With, "with")?;
    token = consume(lex, "withExpr", TokenKind::LeftParen, "(")?;

    // todo: SmallVec
    let mut was: Vec<WithArgExpr> = Vec::with_capacity(1);
    while !lex.is_eof() && token.kind != TokenKind::RightParen {

        let wa = parse_with_arg_expr(lex)?;
        was.push(wa);

        token = lex.token().unwrap();

        match token.kind {
            TokenKind::Comma => (),
            TokenKind::RightParen => break,
            _ => {
                let msg = format!("withExpr: unexpected token {}; want ',' or ')'", lex.token);
                return Err(Error::from(msg));
            }
        }
    }

    // end:
    check_duplicate_with_arg_names(&was)?;

    lex.next()?;

    let expr = parse_expression(lex)?;
    Ok( WithExpr::new(expr, was) )
}

fn parse_with_arg_expr(mut lex: &Lexer) -> Result<WithArgExpr> {
    let args: Vec<String> = vec![];

    let mut tok = lex.token().unwrap();
    if tok.kind != TokenKind::Ident {
        let msg = format!("withArgExpr: unexpected token {}; want identifier", lex.token);
        return Err(Error::from(msg));
    }
    let name = unescape_ident(lex.token);

    tok = lex.next().unwrap();
    if tok.kind == TokenKind::LeftParen {
        // Parse func args.
        let args = match parse_ident_list(lex) {
            Ok(args) => args,
            Err(e) => {
                let msg = format!("withArgExpr: cannot parse args for {}: {:?}", name, e );
                return Err(Error::from(msg, e))
            }
        };
        // Make sure all the args have different names
        let m = HashSet::new(); // todo: set initial capacity
        for arg in args {
            if m.contains(&arg) {
                let msg = format!("withArgExpr: duplicate arg name: {}", arg);
                return Err(Error::from(msg));
            }
            m.add(arg);
        }
    }
    tok = lex.token().unwrap();
    if tok.kind != TokenKind::Equal {
        let msg = format!("withArgExpr: unexpected token {}; want '='", tok.text);
        return Err(Error::from(msg));
    }
    tok = lex.next().unwrap();
    let expr: Expression = match parse_expression(lex) {
        Ok(e) => e,
        Err(e) => {
            let msg = format!("withArgExpr: cannot parse expression for {}: {}", name, e);
            return Err(Error::from(msg, e));
        }
    };
    return Ok( WithArgExpr { name, expr: Box::new(expr), args } );
}

fn parse_at_expr(mut lex: &Lexer) -> Result<impl ExpressionNode> {
    if lex.token() != '@' {
        let msg = format!("atExpr: unexpected token {}; want '@'", lex.token);
        return Err(Error::from(msg));
    }
    lex.next()?;
    match parse_single_expr_without_rollup_suffix(lex) {
        Ok(e) => Ok(e),
        Err(e) => {
            let msg = format!("cannot parse '@' expression: {}", lex.token);
            Err(Error::new(msg))
        }
    }
}

fn parse_offset(mut lex: &Lexer) -> Result<DurationExpr> {
    let tok = consume(lex, "offset", TokenKind::Offset, "offset")?;
    return parse_duration(lex);
}

fn parse_window_and_step(mut lex: &mut Lexer) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool)> {
    let mut tok = consume(lex, "window", TokenKind::LeftBracket, "[")?;

    let mut window: Option<DurationExpr> = None;

    if tok.kind != TokenKind::Colon {
        window = Some(parse_positive_duration(lex)?);
    }
    let mut step: Option<DurationExpr> = None;
    let mut inherit_step = false;
    tok = lex.token().unwrap();

    if tok.kind == TokenKind::Colon {
        tok = lex.next().unwrap();
        // Parse step
        if lex.token.len() == 0 {
            tok = lex.next().unwrap();
            if tok.kind == TokenKind::RightBracket {
                inherit_step = true;
            }
        }
        if tok.kind != TokenKind::RightBracket {
            step = Some(parse_positive_duration(lex)?);
        }
    }
    if tok.kind != TokenKind::RightBracket {
        return Err(Error::new(format!("windowAndStep: unexpected token {}; want ']'", lex.token)));
    }
    lex.next()?;

    Ok((window, step, inherit_step))
}

fn parse_duration(mut lex: &Lexer) -> Result<DurationExpr> {
    let tok = lex.token().expect("Expecting a duration expression");
    match tok.kind {
        TokenKind::Duration => {
            Ok( DurationExpr::new(tok.text) )
        },
        TokenKind::Number => {
            // default to seconds if no unit is specified
            Ok( DurationExpr::new(format!("{}s", tok.text)) )
        },
        _ => {
            Err(Error::new(format!("Expected duration: got {}", tok.text)))
        }
    }
}

pub fn parse_positive_duration(mut lex: &Lexer) -> Result<DurationExpr> {
    // Verify the duration in seconds without explicit suffix.
    let duration = parse_duration(lex)?;
    let val = duration.duration(1);
    if val < 0 {
        Err(Error::new(format!("Expected positive duration: found {}", duration.s)))
    } else {
        Ok(duration)
    }
}


fn consume<'a>(mut lex: &Lexer, func_name: &str, expected: TokenKind, token_text: &str) -> Result<Token<'a>> {
    let tok = lex.token().unwrap(); // todo: use expect
    if tok.kind != expected {
        let msg = format!("{}: unexpected token {}; want '{}'", func_name, tok.text, token_text);
        return Err(Error::new(msg));
    }
    let next = lex.next().unwrap();
    Ok(next)
}

fn parse_ident_list(mut lex: &Lexer) -> Result<Vec<String>> {
    let mut tok = consume(lex, "identlist", TokenKind::LeftParen, "(")?;
    let mut idents = Vec::new();
    while !lex.is_eof() && tok.kind != TokenKind::RightParen {

        if tok.kind != TokenKind::Ident {
            let msg = format!("identList: unexpected token {}; want identifier", tok.text);
            return Err(Error::from(msg));
        }
        idents.push(unescape_ident(lex.token));
        let token = lex.next().unwrap();

        match token.kind {
            TokenKind::Comma => {
                continue;
            },
            TokenKind::RightParen => {
                break;
            },
            _ => {
                let msg = format!("identList: unexpected token {}; want ',' or ')'", lex.token);
                return Err(Error::new(msg));
            }
        }
    }

    lex.next()?;
    return Ok(idents);
}

fn parse_string_expr(mut lex: &Lexer) -> Result<StringExpr> {
    let mut se = StringExpr::new("");

    let mut token= lex.token().unwrap();

    while !lex.is_eof() {
        match token.kind {
            TokenKind::QuotedString | TokenKind::Ident => {
                se.tokens.push(token.text);
            },
            _ => {
                let msg = format!("stringExpr: unexpected token {}; want string", token.text);
                return Err(Error::new(msg));
            }
        }

        let mut tok = lex.peek().unwrap();
        if tok.kind != TokenKind::OpPlus {
            return Ok(se);
        }

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.
        tok = lex.peek().unwrap();
        if tok.kind == TokenKind::QuotedString {
            // "s1" + "s2"
            continue;
        }
        if tok.kind != TokenKind::Ident {
            // "s" + unknownToken
            return Ok(se);
        }
        // Look after ident
        tok = lex.peek().unwrap();
        if tok.kind == TokenKind::LeftParen || tok.kind == TokenKind::LeftBrace {
            // `"s" + m(` or `"s" + m{`
            return Ok(se);
        }
        // "s" + ident
        tok = lex.next().unwrap();
    }

    return Ok(se);
}

fn parse_label_filters(mut lex: &Lexer) -> Result<Vec<LabelFilterExpr>> {
    let mut tok = consume(lex, "label_filters", TokenKind::LeftBrace, "{")?;
    let mut lfes: Vec<LabelFilterExpr> = vec![];

    while !lex.is_eof() && tok.kind != TokenKind::RightBrace {

        let lfe = parse_label_filter_expr(lex)?;
        lfes.push(lfe);

        tok = lex.token().unwrap();
        if tok.kind == TokenKind::RightBrace {
            // goto closeBracesLabel
            break;
        } else if tok.kind == TokenKind::Comma {
            continue;
        } else {
            let msg = format!("label_filters: unexpected token {}; want ',' or '}}'", lex.token);
            return Err(Error::new(msg));
        }
        // lex.next().unwrap();
    }
    lex.next()?;
    Ok(lfes)
}

fn parse_label_filter_expr(mut lex: &Lexer) -> Result<LabelFilterExpr> {
    use TokenKind::*;
    use types::*;

    let mut tok = consume(lex, "labelFilterExpr", Ident, "label name")?;
    let label = unescape_ident(lex.token);

    tok = lex.next().unwrap();
    let mut op: LabelFilterOp;
    match tok.kind {
        Equal => {
            op = LabelFilterOp::Equal;
        },
        OpNotEqual => {
            op = LabelFilterOp::NotEqual;
        },
        OpRegexEqual => {
            op = LabelFilterOp::RegexEqual;
        },
        OpRegexNotEqual => {
            op = LabelFilterOp::RegexNotEqual;
        },
        _ => {
            let msg = format!("labelFilterExpr: unexpected token {}; want '=', '!=', '=~' or '!~'", tok.text);
            return Err(Error::new(msg));
        }
    }
    lex.next()?;

    let se = parse_string_expr(lex)?;
    Ok( LabelFilterExpr{ label, value: se, op } )
}


// remove_parens_expr removes parensExpr for (Expr) case.
fn remove_parens_expr(mut e: &Expression) {
    match e {
        Expression::Rollup(mut re) => {
            remove_parens_expr(&*re.expr);
            if let Some(at) = re.at {
                remove_parens_expr(&*at);
            }
        },
        Expression::BinaryOperator(mut be) => {
            remove_parens_expr(&mut be.left);
            remove_parens_expr(&mut be.right);
        },
        Expression::Aggregation(mut agg) => {
            for mut arg in agg.args {
                remove_parens_expr(&arg)
            }
        },
        Expression::Function(f) => {
            for mut arg in f.args {
                remove_parens_expr(&arg)
            }
        },
        Expression::Parens(parens) => {
            for mut arg in parens.args {
                remove_parens_expr(&arg)
            }
        }
        _ => ()
    }
}

fn simplify_parens_expr(expr: ParensExpr) -> Expression {
    if expr.len() == 1 {
        return expr.expressions[0].clone();
    }
    // Treat parensExpr as a function with empty name, i.e. union()
    let fe = FuncExpr::new("", expr.expressions);
    return fe as Expression;
}

fn simplify_constants(mut expr: &Expression) -> impl ExpressionNode {
    match expr {
        Expression::Rollup(&mut re) => {
            simplify_constants(&*re.expr);
            if let Some(at) = re.at {
                simplify_constants(&mut *at);
            }
            re
        },
        Expression::BinaryOperator(&mut be) => {
            simplify_constants(&mut be.left);
            simplify_constants(&mut be.right);

            match (be.left, be.right) {
                (Expression::Number(ln), Expression::Number(rn)) => {
                    let n = eval_binary_op(ln.value(), rn.value(), be.op, be.bool);
                    NumberExpr::new(n)
                }
                (Expression::String(left), Expression::String(right)) => {
                    if be.op == BinaryOp::Add {
                        let val = format!("{}{}", left.s, right.s);
                        return StringExpr::new(val);
                    }
                    let ok = string_compare(&left.s, &right.s, be.op);
                    let mut n: f64 = if ok { 1 } else { 0 } as f64;
                    if !be.bool_modifier && n == 0 {
                        n = f64::NAN;
                    }
                    NumberExpr::new(n)
                }
                _ => return be
            }
        },
        Expression::Aggregation(mut agg) => {
            simplify_constants_inplace(&agg.args);
            agg
        },
        Expression::Function(mut fe) => {
            simplify_constants_inplace(&mut fe.args);
            fe
        },
        Expression::Parens(mut parens) => {
            simplify_constants_inplace(&parens.expressions);
            if parens.len() == 1 {
                return parens.expressions[0].clone();
            }
            // Treat parensExpr as a function with empty name, i.e. union()
            FuncExpr::new("", parens.expressions)
        }
        _ => expr
    }
}

fn simplify_constants_inplace(mut args: &[Expression]) {
    for arg in &mut args {
        *arg = simplify_constants(arg);
    }
}

#[inline]
fn is_rollup_start_token(token: &Token) -> bool {
    return token.kind.is_rollup_start();
}

fn must_parse_with_arg_expr(s: &str) -> Result<WithArgExpr> {
    let mut p = Parser::new(s);
    let tok = p.next()?;
    if tok.is_none() {
        let msg = format!("BUG: cannot find firs token in {}", s);
        return Err(Error::new(msg));
    }
    let expr = parse_with_arg_expr(&mut p.lex)?;
    if !p.is_eof() {
        let msg = format!("BUG: cannot parse {}: unparsed data", s);
        return Err(Error::new(msg));
    }
    return Ok(expr);
}

fn check_duplicate_with_arg_names(was: &[WithArgExpr]) -> Result<()> {
    let mut m = HashMap::with_capacity(was.len());

    for wa in &was {
        if m.contains_key(&wa.name) {
            return Err(
                Error::new(format!("duplicate 'with' arg name for: {};", wa))
            );
        }
        m.insert(wa.name.clone(), true);
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
        _ => panic!("unexpected operator {} in string comparison", op)
    }
}