use std::fmt;
use phf::phf_map;
use metrix::error::Error;
use crate::error::Error;

#[derive(Copy, Clone, Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
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
        use BinaryOp::*;
        return self == Pow;
    }

    pub fn is_logical_set_op(self) -> bool {
        use BinaryOp::*;
        return self == And || self == Or || self == Unless;
    }

    pub fn is_comparison(&self) -> bool {
        self.kind == BinaryOpKind::Comparison
    }
}

impl TryFrom<&str> for BinaryOp {
    type Error = Error;

    fn try_from(op: &str) -> Result<Self, E> {
        match BINARY_OPS_MAP.get(op.to_lowercase().as_str()) {
            Some(op) => Ok(*op),
            None => Err(Error::new(format!("Unknown, binary op {}", op)))
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
        Ok(())
    }
}

pub fn is_binary_op(op: &str) -> bool {
    BINARY_OPS_MAP.contains_key(op.to_lowercase().as_str())
}

pub fn binary_op_priority(op: &str) -> i8 {
    let opp = BinaryOp::try_from(op);
    return if opp.is_ok() {
        op.precedence()
    } else {
        -1
    };
}

pub fn scan_binary_op_prefix(s: &str) -> usize {
    let mut n = 0;
    for op in BINARY_OPS_MAP.keys() {
        let op_len = op.len();
        if s.len() < op_len {
            continue;
        }
        let ss = &s[0..op_len].to_lowercase();
        if ss == *k && op_len > n {
            n = op_len;
        }
        n += 1;
    }
    return n;
}

pub fn is_binary_op_cmp(op: &str) -> bool {
    op == "==" || op == "!=" || op == "<" || op == ">" || op == "<=" || op == ">="
}

pub fn is_right_associative_binary_op(op: &str) -> bool {
    // See https://prometheus.io/docs/prometheus/latest/querying/operators/#binary-operator-precedence
    return op == "^"
}

pub fn is_binary_op_group_modifier(s: &str) -> bool {
    let lower = s.to_ascii_lowercase().as_str();
    // See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
    return lower == "on" || lower == "ignoring";
}

pub fn is_binary_op_join_modifier(s: &str) -> bool {
    let lower = s.to_ascii_lowercase().as_str();
    // See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
    return lower == "group_left" || lower == "group_right";
}

pub fn is_binary_op_logical_set(s: &str) -> bool {
    let lower = s.to_ascii_lowercase().as_str();
    return lower == "and" || lower == "or" || lower == "unless";
}

pub fn is_binary_op_bool_modifier(s: &str) -> bool {
    let lower = s.to_ascii_lowercase().as_str();
    return lower == "bool";
}