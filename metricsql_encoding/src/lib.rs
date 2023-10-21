use std::{error, fmt, io};

pub mod encoders;
pub mod marshal;
mod reader;
pub mod utils;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum BlockType {
    Float,
    Integer,
    Bool,
    Str,
    Unsigned,
}

impl TryFrom<u8> for BlockType {
    type Error = EncodingError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Float),
            1 => Ok(Self::Integer),
            2 => Ok(Self::Bool),
            3 => Ok(Self::Str),
            4 => Ok(Self::Unsigned),
            _ => Err(EncodingError {
                description: format!("{value:?} is invalid block type"),
            }),
        }
    }
}

/// `Block` holds information about location and time range of a block of data.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Block {
    pub min_time: i64,
    pub max_time: i64,
    pub offset: u64,
    pub size: u32,
    pub typ: BlockType,

    // This index is used to track an associated reader needed to decode the
    // data this block holds.
    pub reader_idx: usize,
}

impl Block {
    /// Determines if this block overlaps the provided block.
    ///
    /// Blocks overlap when the time-range of the data within the block can
    /// overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.min_time <= other.max_time && other.min_time <= self.max_time
    }
}

// MAX_BLOCK_VALUES is the maximum number of values a TSM block can store.
const MAX_BLOCK_VALUES: usize = 1000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodingError {
    pub description: String,
}

impl EncodingError {
    pub fn new(description: String) -> Self {
        Self { description }
    }
}

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl error::Error for EncodingError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

impl From<io::Error> for EncodingError {
    fn from(e: io::Error) -> Self {
        Self {
            description: format!("TODO - io error: {e} ({e:?})"),
        }
    }
}

impl From<std::str::Utf8Error> for EncodingError {
    fn from(e: std::str::Utf8Error) -> Self {
        Self {
            description: format!("TODO - utf8 error: {e} ({e:?})"),
        }
    }
}

pub type EncodingResult<T> = Result<T, EncodingError>;
