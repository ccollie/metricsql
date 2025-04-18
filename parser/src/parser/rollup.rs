use crate::ast::{DurationExpr, Expr, RollupExpr};
use crate::common::ValueType;
use crate::parser::tokens::Token;
use crate::parser::{syntax_error, ParseError, ParseResult, Parser};

impl Parser<'_> {
    pub(super) fn parse_rollup_expr(&mut self, e: Expr) -> ParseResult<Expr> {
        let mut re = RollupExpr::new(e);

        let mut at: Option<Expr> = None;
        if self.at(&Token::LeftBracket) {
            let (window, step, inherit_step) = self.parse_window_and_step()?;
            re.window = window;
            re.step = step;
            re.inherit_step = inherit_step;
        }

        if self.at(&Token::At) {
            at = Some(self.parse_at_expr()?);
        }

        if self.at(&Token::Offset) {
            re.offset = Some(self.parse_offset()?);
        }

        if self.at(&Token::At) {
            if at.is_some() {
                let span = self.last_token_range().unwrap_or_default();
                let msg = "duplicate '@' token".to_string();
                return Err(syntax_error(&msg, &span, "".to_string()));
            }
            at = Some(self.parse_at_expr()?);
        }

        if let Some(v) = at {
            re.at = Some(Box::new(v))
        }

        Ok(Expr::Rollup(re))
    }

    fn parse_at_expr(&mut self) -> ParseResult<Expr> {
        use Token::*;

        self.expect(&At)?;

        let span = self.last_token_range().unwrap_or_default();
        let expr = self.parse_single_expr_without_rollup_suffix()?;

        match expr.return_type() {
            ValueType::InstantVector | ValueType::Scalar => Ok(expr),
            _ => Err(syntax_error(
                "@ modifier Expr must return a scalar or instant vector",
                &span,
                "".to_string(),
            )),
        }
    }

    fn parse_offset(&mut self) -> ParseResult<DurationExpr> {
        self.expect(&Token::Offset)?;
        self.parse_duration()
    }

    fn parse_window_and_step(
        &mut self,
    ) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool), ParseError> {
        self.expect(&Token::LeftBracket)?;

        let mut window: Option<DurationExpr> = None;

        if !self.at(&Token::Colon) {
            window = Some(self.parse_positive_duration()?);
        }

        let mut step: Option<DurationExpr> = None;
        let mut inherit_step = false;

        if self.at(&Token::Colon) {
            self.bump();
            // Parse step
            if self.at(&Token::RightBracket) {
                inherit_step = true;
            }
            if !self.at(&Token::RightBracket) {
                step = Some(self.parse_positive_duration()?);
            }
        }
        self.expect(&Token::RightBracket)?;

        Ok((window, step, inherit_step))
    }
}