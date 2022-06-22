use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use crate::error::Error;
use crate::{AggrFuncExpr, BinaryOp, BinaryOpExpr, DurationExpr, eval_binary_op, Expression, FuncExpr, GroupModifier, is_binary_op, is_binary_op_group_modifier, is_binary_op_join_modifier, is_binary_op_logical_set, JoinModifier, MetricExpr, NumberExpr, RollupExpr, StringExpr, WithArgExpr, WithExpr};
use crate::parser::aggr::{is_aggr_func, is_aggr_func_modifier};
use crate::parser::expand_with::expand_with_expr;
use crate::parser::is_positive_number_prefix;
use crate::parser::lexer::{is_ident_prefix, is_offset, is_positive_duration, is_string_prefix, isInfOrNaN, Lexer};

// parser parses MetricsQL expression.
//
// preconditions for all parser.parse* funcs:
// - self.lex.token should point to the first token to parse.
//
// post-conditions for all parser.parse* funcs:
// - self.lex.token should point to the next token after the parsed token.
pub struct Parser {
    lex: Lexer
}

pub fn parse(input: &str) -> Result<Expression, Error> {
    let mut parser = Parser::new(input);
    let tok = match parser.next() {
        Ok(tok) => tok,
        Err(e) => {
            let msg = format!("cannot parse the first token {}", e);
            return Err(msg.into());
        }
    };
    let expr = parser.parse_expression()?;
    if !parser.eof() {
        let msg = format!("unparsed data {}", parser.lex.token);
        return Err(msg.into());
    }
    let was = get_default_with_arg_exprs(&expr)?;
    match expand_with_expr(was, &expr) {
        Ok(expr) => {
            let mut res = remove_parens_expr(expr);
            let simple = simplify_constants(&res);
            Ok(simple)
        },
        Err(e) => {
            let msg = format!("cannot expand with expr {}", e);
            Err(msg.into())
        }
    }
}


fn get_default_with_arg_exprs(arg_exprs: &[Expr]) -> Vec<WithArgExpr> {
    let mut args = Vec::new();
    for arg_expr in arg_exprs {
        args.push(arg_expr.clone());
    }
    Expr::Default(args)
}


impl Parser {
    pub fn new(input: &str) -> Self {
        Parser {
            lex: Lexer::new(input)
        }
    }

    // todo: restrict visibility to this module.
    pub fn next(&self) -> Option<Token> {
        Some(self.lex.next()?)
    }

    fn prev(&self) -> Option<Token> {
        self.lex.prev()
    }

    pub fn is_eof(&self) -> bool {
        self.lex.isEOF()
    }

    pub fn parse_expression(&mut self) -> Result<Expression, Error> {
        let mut expr = parse_expression(&self.lex)?;
        if !self.is_eof() {
            return Err(Error::UnexpectedToken(None));
        }
        Ok(expr)
    }

    pub fn parse_with_expr(&mut self) -> Result<WithExpr, Error> {
        let expr = parse_with_expr(&self.lex)?;
        Ok(expr)
    }

    pub fn parse_label_filter(&mut self) -> Result<LabelFilter, Error> {
        let filter = parse_label_filter_expr(&self.lex)?;
        Ok(filter.to_label_filter())
    }

    pub fn parse_duration(&self) -> Result<DurationExpr, Error> {
        parse_duration(&self.lex)
    }
}

pub fn parse_func_expr(mut lex: &Lexer) -> Result<Expression, Error> {
    if !is_ident_prefix(lex.token) {
        let msg = format!("funcExpr: unexpected token {}; want function name", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }

    let name = unescape_ident(lex.token);
    lex.next()?;
    if lex.token != '(' {
        let msg = format!("funcExpr: unexpected token {}; want '('", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    let args = lex.parseArgListExpr();

    let mut keep_metric_names = false;
    if is_keep_metric_names(lex.token) {
        keep_metric_names = true;
        lex.next()?;
    }
    let fe = FuncExpr{ name, args, keep_metric_names };

    Ok(fe)
}

pub fn parse_arg_list(mut lex: &Lexer) -> Result<Vec<Expression>, Error> {
    if lex.token !='(' {
        let msg = format!("argList: unexpected token {}; want '('", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    let mut args = Vec::new();
    lex.next()?;

    while !lex.is_eof() && lex.token != ')' {
        let expr = parse_expression(lex)?;
        args.push(expr);
        if lex.token != ',' {
            break;
        }

        lex.next()?;
    }
    Ok(args)
}

fn parse_aggr_func_expr(mut lex: &Lexer) -> Result<AggrFuncExpr, Error> {
    if !is_aggr_func(lex.token) {
        return Err(
            Error::UnexpectedToken(format!("aggrFuncExpr: unexpected token {}; want 'aggrFunc'", lex.token))
        );
    }

    let name = unescapeIdent(lex.token.to_lowercase().as_str());
    let mut ae = AggrFuncExpr::new(name);
    lex.next()?;

    if is_ident_expr(lex.token) {
        if !is_aggr_func_expr(lex.token) {
            return Err(
                Error::UnexpectedToken(format!("aggrFuncExpr: unexpected token {}; want 'aggrFuncModifier'", lex.token))
            );
        }
        ae.with_modifier( parse_modifier_expr(lex) );
        return Ok(ae);
    }
    if lex.token == '(' {
        ae.args = parse_arg_list(lex)?;

        // Verify whether func suffix exists.
        if ae.modifier.is_none() && is_aggr_func_modifier(lex.token) {
            ae.with_modifier( parse_modifier_expr(lex) );
        }
        // Check for optional limit.
        if lex.token.to_lowercase().as_str() == "limit" {
            lex.next()?;
            let limit = parseInt(lex.token);
            if !limit.is_finite() {
                return Err(Error::from(format!("cannot parse limit {}", lex.token)));
            }
            lex.next()?;
            ae.limit = limit;
        }
        return Ok(ae);
    }
    return Err(
        Error::from(format!("AggrFuncExpr: unexpected token {}; want '('", lex.token))
    );
}

fn parse_expression(mut lex: &Lexer) -> Result<Expression, Error> {
    let mut e = parse_single_expr(lex)?;
    let mut bool = false;

    while !lex.is_eof() {
        let token = &lex.token;
        if !is_binary_op(token) {
            return Ok(e);
        }

        let op = this.token.toLowerCase().as_str();

        lex.next()?;
        if is_binaryOp_bool_modifier(lex.token) {
            if !is_binaryop_cmp(op) {
                let msg = format!("bool modifier cannot be applied to {}", lex.token);
                return Err(Error::UnexpectedToken(msg));
            }
            bool = true;
            lex.next()?;
        }
        let right = parse_single_expr(&lex)?;

        let mut be = BinaryOpExpr::new(op, e, right);

        if group.is_some() {
            be.group_modifier = group;
        }

        if is_binary_op_group_modifier(lex.token) {
            be.group_modifier = parse_group_modifier(lex);
            if is_binary_op_join_modifier(lex.token) {
                if is_binary_op_logical_set(op) {
                    let msg = format!("modifier {} cannot be applied to {}", lex.token, op);
                    return Err(Error::UnexpectedToken(msg));
                }
                let join = parse_join_modifier(lex)?;
                be.join_modifier = Some(join);
            }
        }
        return Ok(be.balance());
    }

    return Ok(e);
}

pub fn parse_single_expr(mut lex: &Lexer) -> Result<Expression, Error> {
    if is_with(lex.token) {
        lex.next()?;
        let is_lparen = lex.token == '(';
        lex.prev();
        if is_lparen {
            return Ok(parse_with_expr(lex)?)
        }
    }
    let e = parse_single_expr_without_rollup_suffix(lex)?;
    if !is_rollup_start_token(lex.token) {
        // There is no rollup expression.
        return Ok(e);
    }
    return parse_rollup_expr(lex, e);
}

fn parse_single_expr_without_rollup_suffix(mut lex: &Lexer) -> Result<Expression, Error> {
    if is_positive_duration(lex.token) {
        return parse_positive_duration(lex)?
    }
    if is_string_prefix(lex.token) {
        return parse_string_expr(lex)
    }
    if is_positive_number_prefix(lex.Token) || isInfOrNaN(lex.Token) {
        return parse_positive_number_expr(lex)
    }
    if is_ident_prefix(lex.Token) {
        return parse_ident_expr(lex);
    }
    match lex.token {
        "(" => parse_parens_expr(lex),
        "{" => parse_metric_expr(lex),
        "+" => {
            // Unary plus
            lex.next()?;
            parse_single_expr(lex)
        },
        "-" => {
            // Unary minus. Substitute `-expr` with `0 - expr`
            lex.next()?;
            let e = parse_single_expr(lex)?;
            let be = BinaryExpr(BinaryOp::Sub, NumberExpr::new(0.0), e);
            Ok(be)
        },
        _ => {
            let msg = format!("singleExpr: unexpected token {}; want '(', '{{', '-', '+'", lex.token);
            Err(msg)
        }
    }
}

fn parse_group_modifier(mut lex: &Lexer) -> Result<GroupModifier, Error> {
    if !is_ident_prefix(lex.token) {
        let msg = format!("modifierExpr: unexpected token {}; want 'ident'", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }

    let op = GroupModifierOp.try_from(lex.token);
    if op.is_error() {
        let msg = format!("GroupModifier: unexpected token {}; want 'on' or 'ignoring", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    lex.next()?;
    let mut res = GroupModifier::new(op);
    if lex.token != '(' {
        // join modifier may miss ident list.
        return Ok(res);
    }
    let args = parse_ident_list(lex)?;
    res.set_labels(args);

    return Ok(res);
}

fn parse_join_modifier(mut lex: &Lexer) -> Result<JoinModifier, Error> {
    if !is_ident_prefix(lex.token) {
        let msg = format!("modifierExpr: unexpected token {}; want 'ident'", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }

    let op = JoinModifierOp.try_from(lex.token);
    if op.is_error() {
        let msg = format!("joinModifier: unexpected token {}; want 'group_left' or 'group_right", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    lex.next()?;

    let res = JoinModifier::new(op);
    if lex.token != '(' {
        // join modifier may miss ident list.
        Ok(res)
    } else {
        let args = parse_ident_list(lex)?;
        res.set_labels(args);
        Ok(res)
    }
}

fn parse_ident_expr(mut lex: &Lexer) -> Result<Expression, Error> {
    // Look into the next-next token in order to determine how to parse
    // the current expression.
    lex.next()?;
    if lex.is_eof() || is_offset(lex.token) {
        lex.prev();
        return parse_metric_expr(lex);
    }
    if is_ident_prefix(lex.token) {
        lex.prev();
        if is_aggr_func(lex.token) {
            return parse_aggr_func_expr(lex);
        }
        return parse_metric_expr(lex);
    }
    if is_binary_op(lex.token) {
        lex.prev();
        return parse_metric_expr(lex);
    }
    match lex.token {
        "" => {
            lex.next()?;
            if is_aggr_func(lex.token) {
                parse_aggr_func_expr(lex)
            } else {
                parse_func_expr(lex)
            }
        },
        "{" | "[" | ")" | "," | "@" => {
            lex.prev();
            parse_metric_expr(lex)
        },
        _ => {
            let msg = format!("IdentExpr: unexpected token {}; want '(', '{', '[', ')', ',' or '@'", lex.token);
            Err(Error::UnexpectedToken(msg))
        }
    }
}

fn parse_metric_expr(mut lex: &Lexer) -> Result<MetricExpr, Error> {
    let mut me = MetricExpr {
        label_filters: vec![],
        label_filter_exprs: vec![]
    };
    if is_ident_prefix(this.token) {
        let tokens = Vec![quote(unescapeIdent(this.token))];
        let value = StringExpr { s: "".as_str(), tokens };
        let lfe = LabelFilterExpr { label: "__name__", value };
        me.labelFilterExprs.push(lfe);
        this.next();
        if lex.token != '{' {
            return Ok(me);
        }
    }
    let lfes = parse_label_filters(lex);
    for lf in lfes {
        me.label_filter_exprs.push(lf);
    }
    Ok(me)
}

fn parse_rollup_expr(mut lex: &Lexer, e: Expression) -> Result<Expression, Error> {
    let mut re = RollupExpr::new(e);
    if lex.token == '[' {
        let (window, step, inherit_step) = parse_window_and_step(&mut lex);
        re.window = window;
        re.step = step;
        re.inheritStep = inherit_step;
        if !is_offset(this.token) && lex.token != '@' {
            return Ok(re);
        }
    }
    if this.token == '@' {
        re.set_at( parse_at_expr(lex)? );
    }
    if is_offset(lex.token) {
        re.offset = Option::from(parse_offset(lex)?);
    }
    if lex.token == '@' {
        if re.at.is_some() {
            let msg = format!("RollupExpr: duplicate '@' token");
            return Err(Error::UnexpectedToken(msg));
        }
        re.set_at( parse_at_expr(lex)? );
    }
    Ok(re)
}

fn parse_parens_expr(mut lex: &Lexer) -> Result<ParensExpr, Error> {
    if lex.token != '(' {
        return Err(Error::InvalidToken(forma!("parensExpr: unexpected token {}; want '('", self.token)));
    }
    let mut exprs: Vec<Expression> = Vec![];
    while !lex.is_eof() {
        lex.next()?;
        if lex.token == ')' {
            break;
        }
        let expr = parse_expression(lex)?;
        exprs.push(expr);
        if lex.token == ',' {
            continue;
        }
        if lex.token == ')' {
            break;
        }
        return Err(Error::InvalidToken(forma!("parensExpr: unexpected token {}; want ',' or ')'", self.token)));
    }
    lex.next()?;
    Ok( ParensExpr { exprs } )
}


fn parse_with_expr(mut lex: &Lexer) -> Result<WithExpr, Error> {
    if !is_with(lex.token) {
        let msg = format!("withExpr: unexpected token {}; want 'WITH'", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    this.next();

    let mut was: Vec<WithArgExpr> = Vec![];

    if lex.token != '(' {
        let msg = format!("withExpr: unexpected token {}; want '('", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    while !lex.is_eof() {
        lex.next()?;

        if lex.token == ')' {
            break;
        }

        let wa = parse_with_arg_expr(lex)?;
        was.push(wa);

        if lex.token == ',' {
            // do nothing
        } else if lex.token == ')' {
            break;
        } else {
            let msg = format!("withExpr: unexpected token {}; want ',' or ')'", lex.token);
            return Err(Error::UnexpectedToken(msg));
        }
    }

    // end:
    check_duplicate_with_arg_names(&was)?;

    this.next();

    let expr = parse_expression(lex)?;
    Ok( WithExpr::new(expr, was) )
}

fn parse_with_arg_expr(mut lex: &Lexer) -> Result<WithArgExpr, Error> {
    let args: Vec<String> = Vec![];

    if !is_ident_prefix(lex.token) {
        let msg = format!("withArgExpr: unexpected token {}; want identifier", this.token);
        return Err(Error::UnexpectedToken(msg));
    }
    let name = unescape_ident(lex.token);

    this.next();
    if lex.token == '(' {
        // Parse func args.
        let args = match parse_ident_list(lex) {
            Ok(args) => args,
            Err(e) => {
                let msg = format!("withArgExpr: cannot parse args for {}: {:?}", name, e );
                return Err(e)
            }
        };
        // Make sure all the args have different names
        let m = HashSet::new(); // todo: set initial capacity
        for arg in args {
            if m.contains(&arg) {
                let msg = format!("withArgExpr: duplicate arg name: {}", arg);
                return Err(Error::UnexpectedToken(msg));
            }
            m.add(arg);
        }
    }
    if lex.token != '=' {
        let msg = format!("withArgExpr: unexpected token {}; want '='", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    lex.next()?;
    let expr: Expression = match parse_expression(kex) {
        Ok(e) => e,
        Err(e) => {
            let msg = format!("withArgExpr: cannot parse expression for ${name}: ${e}");
            return Err(Error::UnexpectedToken(msg));
        }
    };
    return WithArgExpr { name, expr, args };
}

fn parse_at_expr(mut lex: &Lexer) -> Result<Expression, Error> {
    if lex.token != '@' {
        let msg = format!("atExpr: unexpected token {}; want '@'", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    this.next();
    match parse_single_expr_without_rollup_suffix(lex) {
        Ok(e) => Ok(e),
        Err(e) => {
            let msg = format!("cannot parse "@" expression: {}", lex.token);
            Err(Error::UnexpectedToken(msg))
        }
    }
}

fn parse_offset(mut lex: &Lexer) -> Result<DurationExpr, Error> {
    if !is_offset(lex.token) {
        let msg = format!("offset: unexpected token {}; want offset", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    lex.next()?;
    return parse_duration(lex);
}

fn parse_window_and_step(mut lex: &mut Lexer) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool), Error> {
    if lex.token != "window" {
        return Err(Error::UnexpectedToken(format!("windowAndStep: unexpected token {}; want '['", lex.token)));
    }

    lex.next()?;
    let mut window: Option<DurationExpr> = None;

    if !lex.token.starts_with(':') {
        window = Some(parse_positive_duration(lex)?);
    }
    let mut step: Option<DurationExpr> = None;
    let mut inherit_step = false;
    if lex.token.starts_with(':') {
        // Parse step
        lex.token = &lex.token[1..];
        if lex.token.len() == 0 {
            lex.next()?;
            if lex.token == ']' {
                inherit_step = true;
            }
        }
        if lex.token != ']' {
            step = Some(parse_positive_duration(lex)?);
        }
    }
    if lex.token != ']' {
        return Err(Error::UnexpectedToken(format!("windowAndStep: unexpected token {}; want ']'", lex.token)));
    }
    lex.next()?;

    Ok((window, step, inherit_step))
}

fn parse_duration(mut lex: &Lexer) -> Result<DurationExpr, Error> {
    let d = lex.duration()?;
    Ok( DurationExpression{ d } )
}

pub fn parse_positive_duration(mut lex: &Lexer) -> Result<DurationExpr, Error> {
    // Verify the duration in seconds without explicit suffix.
    let val = match lex.positive_duration() {
        Ok(res) => res,
        Err(e) => Err(
            Error::UnexpectedToken(format!("cannot parse duration; expression {} : {}", s,  e))
        ),
    };
    Ok(DurationExpr::new(&val))
}


fn parse_ident_list(mut lex: &Lexer) -> Result<Vec<String>, Error> {
    if lex.token != '(' {
        let msg = format!("identList: unexpected token {}; want '('", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }
    let mut idents = Vec::new();
    while !lex.is_eof() {
        this.next();
        if lex.token == ')' {
            break;
        }
        if !is_ident_prefix(lex.token) {
            let msg = format!("identList: unexpected token {}; want identifier", lex.token);
            return Err(Error::UnexpectedToken(msg));
        }
        idents.push(unescapeIdent(this.token));
        lex.next()?;

        if lex.token == ',' {
            continue;
        } else if lex.token == ')' {
            break;
        } else {
            let msg = format!("identList: unexpected token {}; want ',' or ')'", lex.token);
            return Err(Error::UnexpectedToken(msg));
        }
    }

    lex.next()?;
    return Ok(idents);
}

fn parse_string_expr(mut lex: &Lexer) -> Result<StringExpr, Error> {
    let mut se = StringExpr::new("");

    while !lex.is_eof() {
        let mut token = this.lex.token;

        if is_string_prefix(token) || is_ident_prefix(token) {
            se.tokens.push(token);
        } else {
            let msg = format!("stringExpr: unexpected token {}; want string", token);
            return Err(Error::UnexpectedToken(msg));
        }
        this.next();
        if lex.token != '+' {
            return Ok(se);
        }

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.
        lex.next()?;
        if is_string_prefix(lex.token) {
            // "s1" + "s2"
            continue;
        }
        if !is_ident_prefix(lex.token) {
            // "s" + unknownToken
            this.prev();
            return Ok(se);
        }
        // Look after ident
        this.next();
        if lex.token == '(' || lex.token == '{' {
            // `"s" + m(` or `"s" + m{`
            this.prev();
            this.prev();
            return Ok(se);
        }
        // "s" + ident
        this.prev();
    }

    return Ok(se);
}

fn parse_label_filters(mut lex: &Lexer) -> Result<Vec<LabelFilterExpr>, Error> {
    if lex.token != '{' {
        let msg = format!("label_filters: unexpected token {}; want '{'", lex.token);
        return Err(Error::UnexpectedToken(msg));
    }

    let mut lfes: Vec<LabelFilterExpr> = Vec![];
    while !lex.is_eof() {
        this.next();
        if lex.token == '}' {
            // goto closeBracesLabel
            break;
        }
        lfe = parse_label_filter_expr(lex);
        lfes.push(lfe);
        if lex.token == '}' {
            // goto closeBracesLabel
            break;
        } else if lex.token == ',' {
            continue;
        } else {
            let msg = format!("label_filters: unexpected token {}; want ',' or '}}'", lex.token);
            return Err(Error::UnexpectedToken(msg));
        }
    }
    this.next();
    Ok(lfes)
}

fn parse_label_filter_expr(mut lex: &Lexer) -> Result<LabelFilterExpr, Error> {
    if !is_ident_prefix(lex.token) {
        Err(Error::UnexpectedToken(format!("labelFilterExpr: unexpected token {}; want label name", lex.token)));
    }
    let mut is_negative = false;
    let mut is_regex = false;
    let label = unescapeIdent(this.token);

    lex.next()?;
    let op = LabelFilterOp = LabelFilterOp::Equal;
    match lex.token {
        "=" => {
            // do nothing
        },
        "!=" => {
            is_negative = true;
        },
        "=~" => {
            is_regex = true;
            is_negative = true;
        },
        "!~" => {
            is_regex = true;
            is_negative = false;
        },
        _ => {
            let msg = format!("labelFilterExpr: unexpected token {}; want '=', '!=', '=~' or '!~'", lex.token);
            return Err(Error::UnexpectedToken(msg));
        }
    }
    lex.next()?;

    let se = parse_string_expr(lex);
    Ok( LabelFilterExpr{ label, value: se, is_regex, is_negative } )
}


// remove_parens_expr removes parensExpr for (Expr) case.
fn remove_parens_expr(mut e: &Expression) -> Expression {
    match e {
        Expression::Rollup(mut re) => {
            re.expr = remove_parens_expr(re.expr);
            if let Some(at) = re.at {
                re.at = remove_parens_expr(&*at);
            }
            return re
        },
        Expression::BinaryOperator(mut be) => {
            be.left = remove_parens_expr(&mut be.left);
            be.right = remove_parens_expr(&mut be.right);
            return be;
        },
        Expression::Aggregation(mut agg) => {
            for arg in agg.args {
                arg = remove_parens_expr(arg)
            }
            return agg;
        },
        Expression::Function(f) => {
            for mut arg in f.args {
                arg = remove_parens_expr(&arg)
            }
            return f;
        },
        Expression::Parens(args) => {
            for mut arg in args {
                args[i] = remove_parens_expr(arg)
            }
            if args.len() == 1 {
                return args[0]
            }
            // Treat parensExpr as a function with empty name, i.e. union()
            let fe = FuncExpr::new("", args);
            return fe
        }
        _ => e
    }
}

fn simplify_constants(mut expr: &Expression) -> Expression {
    match expr {
        Expression::Rollup(mut re) => {
            re.expr = simplify_constants(&*re.expr);
            if let Some(at) = re.at {
                re.at = simplify_constants(at);
            }
            return re
        },
        Expression::BinaryOperator(mut be) => {
            be.left = simplify_constants(&mut be.left);
            be.right = simplify_constants(&mut be.right);

            return match (be.left, be.right) {
                (Expression::Number(ln), Expression::Number(rn)) => {
                    let n = eval_binary_op(ln.value(), rn.value(), be.op, be.bool);
                    NumberExpr::new(n)
                }
                (Expression::String(left), Expression::String(right)) => {
                    if be.op == BinaryOp::Add {
                        return StringExpr::new(left.s + right.s);
                    }
                    let ok = string_compare(left.s, be.op, right.s);
                    let mut n: f64 = if ok { 1 } else { 0 } as f64;
                    if !be.bool_modifier && n == 0 {
                        n = f64::NAN;
                    }
                    NumberExpr::new(n)
                }
                _ => expr
            }
        },
        Expression::Aggregation(agg) => {
            simplify_constants_inplace(agg.args.as_mut());
            return agg;
        },
        Expression::Function(mut f) => {
            simplify_constants_inplace(f.args.as_mut());
            return f;
        },
        Expression::Parens(args) => {
            if args.len() == 1 {
                return simplify_constants_inplace(args[0])
            }
            simplify_constants_inplace(args.as_mut());
            return args;
        }
        _ => e
    }
}

fn simplify_constants_inplace(mut args: &Vec<Expression>) {
    for arg in &mut args {
        *arg = simplify_constants(arg);
    }
}


fn is_rollup_start_token(token: &str) -> bool {
    return token == "[" || token == "@" || is_offset(token)
}

fn prepare_with_arg_exprs(ss: &Vec<String>) -> Vec<WithArgExpr> {
    let mut was = Vec::with_capacity(ss.len());
    for s in ss {
        let parsed = must_parse_with_arg_expr(s)?;
        // panic(fmt.Errorf("BUG: %s", err))
        was.push(WithArgExpr::new(s));
    }
    return was
}

fn must_parse_with_arg_expr(s: &str) -> Result<WithArgExpr, Error> {
    let p = Parser::new(s);
    let tok = p.next()?;
    if tok.is_none() {
        let msg = format!("BUG: cannot find firs token in {}", s);
        return Err(Error::UnexpectedToken(msg));
    }
    let expr = p.parseWithArgExpr();
    if !p.is_eof() {
        let msg = format!("BUG: cannot parse {}: unparsed data", s);
        return Err(Error::UnexpectedToken(msg));
    }
    return Ok(expr);
}

fn check_duplicate_with_arg_names(was: &Vec<WithArgExpr>) -> Result<(), Error> {
    let mut m = HashMap::with_capacity(was.len());

    for wa in &was {
        if m.contains_key(&x.name) {
            return Err(
                Error::from(format!("duplicate 'with' arg name for: {};", wa))
            );
        }
        m.insert(x.name.clone(), true);
    }
    Ok(())
}

fn string_compare(a: &str, b: &str, op: BinaryOp) -> bool {
    match op {
        BinaryOp::Eq => a == b,
        BinaryOp::Neq => a != b,
        BinaryOp::Lt => a < b,
        BinaryOp::Gt => a > b,
        BinaryOp::Le => a <= b,
        BinaryOp::GE => a >= b,
        _ => panic!(format!("unexpected operator {}", op))
    }
}

#[inline]
fn is_with(s: &str) -> bool {
    let lower = s.to_lowercase().as_str();
    return lower == "with";
}

#[inline]
fn is_keep_metric_names(token: &str) -> bool {
    let lower = token.to_lowercase().as_str();
    return lower == "keep_metric_names";
}
