use crate::parser::ParseError;
use phf::phf_map;
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
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
    Neq,
    Or,
    Unless,
}

pub static BINARY_OPS_MAP: phf::Map<&'static str, BinaryOp> = phf_map! {
    "+" => BinaryOp::Add,
    "-" => BinaryOp::Sub,
    "*" => BinaryOp::Mul,
    "/" => BinaryOp::Div,
    "%" => BinaryOp::Mod,
    "^" => BinaryOp::Pow,

    // See https://github.com/prometheus/prometheus/pull/9248
    "atan2" => BinaryOp::Atan2,

    // cmp ops
    "==" => BinaryOp::Eql,
    "!=" => BinaryOp::Neq,
    "<" => BinaryOp::Lt,
    ">" => BinaryOp::Gt,
    "<=" => BinaryOp::Lte,
    ">=" => BinaryOp::Gte,

    // logic set ops
    "and" => BinaryOp::And,
    "or" => BinaryOp::Or,
    "unless" => BinaryOp::Unless,

    // New ops for MetricsQL
    "if" => BinaryOp::If,
    "ifnot" => BinaryOp::IfNot,
    "default" => BinaryOp::Default,
};

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum BinaryOpKind {
    Arithmetic,
    Comparison,
    Logical,
}

pub type Precedence = usize;

impl BinaryOp {
    #[inline]
    pub fn precedence(self) -> Precedence {
        use BinaryOp::*;

        match self {
            Default => 0,
            If | IfNot => 1,
            Or => 10,
            And | Unless => 20,
            Eql | Gte | Gt | Lt | Lte | Neq => 30,
            Add | Sub => 40,
            Mul | Div | Mod | Atan2 => 50,
            Pow => 60,
        }
    }

    #[inline]
    pub fn kind(self) -> BinaryOpKind {
        use BinaryOp::*;
        use BinaryOpKind::*;

        match self {
            Add | Sub | Mul | Div | Mod | Pow | Atan2 => Arithmetic,
            Eql | Gte | Gt | Lt | Lte | Neq => Comparison,
            And | Unless | Or | If | IfNot | Default => Logical,
        }
    }

    // See https://prometheus.io/docs/prometheus/latest/querying/operators/#binary-operator-precedence
    pub fn is_right_associative(self) -> bool {
        self == BinaryOp::Pow
    }

    pub fn is_logical_op(&self) -> bool {
        self.kind() == BinaryOpKind::Logical
    }

    pub fn is_comparison(&self) -> bool {
        self.kind() == BinaryOpKind::Comparison
    }

    pub fn get_reverse_cmp(&self) -> BinaryOp {
        match self {
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::Gte => BinaryOp::Lte,
            BinaryOp::Lte => BinaryOp::Gte,
            // there is no need in changing `==` and `!=`.
            _ => *self,
        }
    }

    #[inline]
    pub fn is_binary_op_logical_set(&self) -> bool {
        use BinaryOp::*;
        matches!(self, And | Or | Unless)
    }

    pub fn as_str<'a>(self) -> &'a str {
        use BinaryOp::*;
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
            Neq => "!=",
            Or => "or",
            Pow => "^",
            Sub => "-",
            Unless => "unless",
        }
    }
}

impl TryFrom<&str> for BinaryOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        match BINARY_OPS_MAP.get(op.to_lowercase().as_str()) {
            Some(op) => Ok(*op),
            None => Err(ParseError::General(format!("Unknown, binary op {}", op))),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

pub fn is_binary_op(op: &str) -> bool {
    BINARY_OPS_MAP.contains_key(op.to_lowercase().as_str())
}
