use std::{env,
          error,
          fmt,
          net,
          result,
          str,
          string};

pub const DEFAULT_ERROR_EXIT_CODE: i32 = 1;

pub type Result<T> = result::Result<T, Error>;

pub enum Error {
    UnexpectedToken(Option<String>),
    UnexpectedEOF,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match *self {
            Error::UnexpectedToken(ref s) => {
                if let Some(s) = s {
                    format!("{}", s)
                } else {
                    "unexpected token".to_string()
                }
            },
            Error::UnexpectedEOF => "unexpected end of file".to_string(),
        };
    }
}