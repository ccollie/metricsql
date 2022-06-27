use std::fmt;

pub enum ExpressionKind {
    Aggregate,
    Binop,
    Function,
    Metric,
    Number,
    String,
    Duration,
    With,
    Parens,
    Rollup
}

impl fmt::Display for ExpressionKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ExpressionKind::*;
        match self {
            Aggregate => write!(f, "aggregate")?,
            Binop => write!(f, "binop")?,
            Function => write!(f, "function")?,
            Metric => write!(f, "metric")?,
            Number => write!(f, "number")?,
            String => write!(f, "string")?,
            Duration => write!(f, "duration")?,
            With => write!(f, "with")?,
            Parens => write!(f, "aggregate")?,
            Rollup => write!(f, "rollup")?,
        }
        Ok(())
    }
}

