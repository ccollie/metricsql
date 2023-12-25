use std::fmt::{Display, Formatter};
use std::str::FromStr;

// todo: move to common_query ??
pub enum SqlDialect {
    Sqlite,
    MySql,
    Postgres,
}

impl Display for SqlDialect {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SqlDialect::Sqlite => write!(f, "sqlite"),
            SqlDialect::MySql => write!(f, "mysql"),
            SqlDialect::Postgres => write!(f, "postgres"),
        }
    }
}

impl FromStr for SqlDialect {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            s if s.eq_ignore_ascii_case("sqlite") => Ok(SqlDialect::Sqlite),
            s if s.eq_ignore_ascii_case("mysql") => Ok(SqlDialect::MySql),
            s if s.eq_ignore_ascii_case("postgres") => Ok(SqlDialect::Postgres),
            s if s.eq_ignore_ascii_case("postgresql") => Ok(SqlDialect::Postgres),
            _ => Err(format!("unsupported sql dialect: {}", s)),
        }
    }
}
