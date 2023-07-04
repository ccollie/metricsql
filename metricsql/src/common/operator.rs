use std::fmt;

use phf::phf_map;
use serde::{Deserialize, Serialize};

use crate::parser::tokens::Token;
use crate::parser::ParseError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operator {
    Add,
    And,
    Atan2,
    Default,
    Div,
    Eql,
    Mod,
    Mul,
    Pow,
    Sub,
    Gt,
    Gte,
    If,
    IfNot,
    Lt,
    Lte,
    NotEq,
    Or,
    Unless,
}

pub static BINARY_OPS_MAP: phf::Map<&'static str, Operator> = phf_map! {
    "+" => Operator::Add,
    "-" => Operator::Sub,
    "*" => Operator::Mul,
    "/" => Operator::Div,
    "%" => Operator::Mod,
    "^" => Operator::Pow,

    // See https://github.com/prometheus/prometheus/pull/9248
    "atan2" => Operator::Atan2,

    // cmp ops
    "==" => Operator::Eql,
    "!=" => Operator::NotEq,
    "<" => Operator::Lt,
    ">" => Operator::Gt,
    "<=" => Operator::Lte,
    ">=" => Operator::Gte,

    // logic set ops
    "and" => Operator::And,
    "or" => Operator::Or,
    "unless" => Operator::Unless,

    // New ops for MetricsQL
    "if" => Operator::If,
    "ifnot" => Operator::IfNot,
    "default" => Operator::Default,
};

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum BinaryOpKind {
    Arithmetic,
    Comparison,
    Logical,
}

pub type Precedence = usize;

impl Operator {
    #[inline]
    pub fn precedence(self) -> Precedence {
        use Operator::*;

        match self {
            Default => 0,
            If | IfNot => 1,
            // See https://prometheus.io/docs/prometheus/latest/querying/operators/#binary-operator-precedence
            Or => 10,
            And | Unless => 20,
            Eql | Gte | Gt | Lt | Lte | NotEq => 30,
            Add | Sub => 40,
            Mul | Div | Mod | Atan2 => 50,
            Pow => 60,
        }
    }

    #[inline]
    pub fn kind(self) -> BinaryOpKind {
        use BinaryOpKind::*;
        use Operator::*;

        match self {
            Add | Sub | Mul | Div | Mod | Pow | Atan2 => Arithmetic,
            Eql | Gte | Gt | Lt | Lte | NotEq => Comparison,
            And | Unless | Or | If | IfNot | Default => Logical,
        }
    }

    // See https://prometheus.io/docs/prometheus/latest/querying/operators/#binary-operator-precedence
    pub fn is_right_associative(self) -> bool {
        self == Operator::Pow
    }

    pub fn is_logical_op(&self) -> bool {
        self.kind() == BinaryOpKind::Logical
    }

    pub fn is_comparison(&self) -> bool {
        self.kind() == BinaryOpKind::Comparison
    }

    pub fn is_valid_string_op(&self) -> bool {
        use Operator::*;
        matches!(self, Add | Eql | Gte | Gt | Lt | Lte | NotEq)
    }

    pub fn get_reverse_cmp(&self) -> Operator {
        match self {
            Operator::Gt => Operator::Lt,
            Operator::Lt => Operator::Gt,
            Operator::Gte => Operator::Lte,
            Operator::Lte => Operator::Gte,
            // there is no need in changing `==` and `!=`.
            _ => *self,
        }
    }

    #[inline]
    pub fn is_set_operator(&self) -> bool {
        use Operator::*;
        matches!(self, And | Or | Unless)
    }

    pub fn as_str(&self) -> &'static str {
        use Operator::*;
        match self {
            Add => "+",
            And => "and",
            Atan2 => "atan2",
            Default => "default",
            Div => "/",
            Eql => "==",
            Gt => ">",
            Gte => ">=",
            If => "if",
            IfNot => "ifNot",
            Mod => "%",
            Mul => "*",
            Lt => "<",
            Lte => "<=",
            NotEq => "!=",
            Or => "or",
            Pow => "^",
            Sub => "-",
            Unless => "unless",
        }
    }
}

impl TryFrom<&str> for Operator {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        match BINARY_OPS_MAP.get(op.to_lowercase().as_str()) {
            Some(op) => Ok(*op),
            None => {
                let result = Err(ParseError::General(format!("Unknown binary op {}", op)));
                result
            }
        }
    }
}

impl TryFrom<Token> for Operator {
    type Error = ParseError;

    fn try_from(token: Token) -> Result<Self, Self::Error> {
        match token {
            Token::OpAnd => Ok(Operator::And),
            Token::OpAtan2 => Ok(Operator::Atan2),
            Token::OpDefault => Ok(Operator::Default),
            Token::OpDiv => Ok(Operator::Div),
            Token::OpEqual => Ok(Operator::Eql),
            Token::OpGreaterThan => Ok(Operator::Gt),
            Token::OpGreaterThanOrEqual => Ok(Operator::Gte),
            Token::OpIf => Ok(Operator::If),
            Token::OpIfNot => Ok(Operator::IfNot),
            Token::OpMod => Ok(Operator::Mod),
            Token::OpMul => Ok(Operator::Mul),
            Token::OpMinus => Ok(Operator::Sub),
            Token::OpLessThan => Ok(Operator::Lt),
            Token::OpLessThanOrEqual => Ok(Operator::Lte),
            Token::OpNotEqual => Ok(Operator::NotEq),
            Token::OpOr => Ok(Operator::Or),
            Token::OpPow => Ok(Operator::Pow),
            Token::OpUnless => Ok(Operator::Unless),
            Token::OpPlus => Ok(Operator::Add),
            _ => Err(ParseError::General(format!(
                "Unknown binary op {:?}",
                token
            ))),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

pub fn is_binary_op(op: &str) -> bool {
    BINARY_OPS_MAP.contains_key(op.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use crate::common::is_binary_op;

    #[test]
    fn test_is_binary_op_success() {
        let f = |s: &str| assert!(is_binary_op(s), "expecting valid binaryOp: {}", s);

        f("and");
        f("AND");
        f("unless");
        f("unleSS");
        f("==");
        f("!=");
        f(">=");
        f("<=");
        f("or");
        f("Or");
        f("+");
        f("-");
        f("*");
        f("/");
        f("%");
        f("atan2");
        f("^");
        f(">");
        f("<");
    }

    #[test]
    fn test_is_binary_op_error() {
        let f = |s: &str| {
            assert!(!is_binary_op(s), "unexpected valid binaryOp: {}", s);
        };

        f("foobar");
        f("=~");
        f("!~");
        f("=");
        f("<==");
        f("234");
    }
}
