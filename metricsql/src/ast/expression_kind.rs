use std::fmt;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ExpressionKind {
    Aggregate,
    Binop,
    Function,
    Group,
    Metric,
    Number,
    String,
    Duration,
    With,
    Parens,
    Rollup,
}

impl fmt::Display for ExpressionKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ExpressionKind::*;
        match self {
            Aggregate => write!(f, "aggregate")?,
            Binop => write!(f, "binop")?,
            Function => write!(f, "function")?,
            Group => write!(f, "group")?,
            Metric => write!(f, "metric")?,
            Number => write!(f, "number")?,
            String => write!(f, "string")?,
            Duration => write!(f, "duration")?,
            With => write!(f, "with")?,
            Parens => write!(f, "group")?,
            Rollup => write!(f, "rollup")?,
        }
        Ok(())
    }
}
