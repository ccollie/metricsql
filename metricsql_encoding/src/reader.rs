//! Types for reading and writing TSM files produced by InfluxDB >= 2.x

use integer_encoding::VarInt;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::u64;

use super::*;

/// A BlockDecoder is capable of decoding a block definition into block data
/// (timestamps and value vectors).
pub trait BlockDecoder {
    fn decode(&mut self, block: &Block) -> Result<BlockData, EncodingError>;
}

impl<T> BlockDecoder for &mut T
where
    T: BlockDecoder,
{
    fn decode(&mut self, block: &Block) -> Result<BlockData, EncodingError> {
        (&mut **self).decode(block)
    }
}

/// MockBlockDecoder implements the BlockDecoder trait. It uses the `min_time`
/// value in a provided `Block` definition as a key to a map of block data,
/// which should be provided on initialisation.
#[derive(Debug, Clone)]
pub struct MockBlockDecoder {
    blocks: BTreeMap<i64, BlockData>,
}

impl MockBlockDecoder {
    pub fn new(blocks: BTreeMap<i64, BlockData>) -> Self {
        Self { blocks }
    }
}

impl BlockDecoder for MockBlockDecoder {
    fn decode(&mut self, block: &Block) -> Result<BlockData, EncodingError> {
        self.blocks
            .get(&block.min_time)
            .cloned()
            .ok_or(EncodingError {
                description: "block not found".to_string(),
            })
    }
}

/// `BlockData` describes the various types of block data that can be held
/// within a TSM file.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockData {
    Float {
        i: usize,
        ts: Vec<i64>,
        values: Vec<f64>,
    },
    Integer {
        i: usize,
        ts: Vec<i64>,
        values: Vec<i64>,
    },
    Bool {
        i: usize,
        ts: Vec<i64>,
        values: Vec<bool>,
    },
    Str {
        i: usize,
        ts: Vec<i64>,
        values: Vec<Vec<u8>>,
    },
    Unsigned {
        i: usize,
        ts: Vec<i64>,
        values: Vec<u64>,
    },
}

impl BlockData {
    /// Initialise an empty `BlockData` with capacity `other.len()` values.
    fn new_from_data(other: &Self) -> Self {
        match other {
            Self::Float { .. } => Self::Float {
                i: 0,
                ts: Vec::with_capacity(other.len()),
                values: Vec::with_capacity(other.len()),
            },
            Self::Integer { .. } => Self::Integer {
                i: 0,
                ts: Vec::with_capacity(other.len()),
                values: Vec::with_capacity(other.len()),
            },
            Self::Bool { .. } => Self::Bool {
                i: 0,
                ts: Vec::with_capacity(other.len()),
                values: Vec::with_capacity(other.len()),
            },
            Self::Str { .. } => Self::Str {
                i: 0,
                ts: Vec::with_capacity(other.len()),
                values: Vec::with_capacity(other.len()),
            },
            Self::Unsigned { .. } => Self::Unsigned {
                i: 0,
                ts: Vec::with_capacity(other.len()),
                values: Vec::with_capacity(other.len()),
            },
        }
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        match self {
            Self::Float { ts, values, .. } => {
                ts.reserve_exact(additional);
                values.reserve_exact(additional);
            }
            Self::Integer { ts, values, .. } => {
                ts.reserve_exact(additional);
                values.reserve_exact(additional);
            }
            Self::Bool { ts, values, .. } => {
                ts.reserve_exact(additional);
                values.reserve_exact(additional);
            }
            Self::Str { ts, values, .. } => {
                ts.reserve_exact(additional);
                values.reserve_exact(additional);
            }
            Self::Unsigned { ts, values, .. } => {
                ts.reserve_exact(additional);
                values.reserve_exact(additional);
            }
        }
    }

    /// Pushes the provided time-stamp value tuple onto the block data.
    pub fn push(&mut self, pair: ValuePair) {
        match pair {
            ValuePair::F64((t, v)) => {
                if let Self::Float { ts, values, .. } = self {
                    ts.push(t);
                    values.push(v);
                } else {
                    panic!("unsupported variant for BlockData::Float");
                }
            }
            ValuePair::I64((t, v)) => {
                if let Self::Integer { ts, values, .. } = self {
                    ts.push(t);
                    values.push(v);
                } else {
                    panic!("unsupported variant for BlockData::Integer");
                }
            }
            ValuePair::Bool((t, v)) => {
                if let Self::Bool { ts, values, .. } = self {
                    ts.push(t);
                    values.push(v);
                } else {
                    panic!("unsupported variant for BlockData::Bool");
                }
            }
            ValuePair::Str((t, v)) => {
                if let Self::Str { ts, values, .. } = self {
                    ts.push(t);
                    values.push(v); // TODO(edd): figure out
                } else {
                    panic!("unsupported variant for BlockData::Str");
                }
            }
            ValuePair::U64((t, v)) => {
                if let Self::Unsigned { ts, values, .. } = self {
                    ts.push(t);
                    values.push(v);
                } else {
                    panic!("unsupported variant for BlockData::Unsigned");
                }
            }
        }
    }

    pub fn next_pair(&mut self) -> Option<ValuePair> {
        if self.is_empty() {
            return None;
        }

        match self {
            Self::Float { i, ts, values } => {
                let idx = *i;
                *i += 1;
                Some(ValuePair::F64((ts[idx], values[idx])))
            }
            Self::Integer { i, ts, values } => {
                let idx = *i;
                *i += 1;
                Some(ValuePair::I64((ts[idx], values[idx])))
            }
            Self::Bool { i, ts, values } => {
                let idx = *i;
                *i += 1;
                Some(ValuePair::Bool((ts[idx], values[idx])))
            }
            Self::Str { i, ts, values } => {
                let idx = *i;
                *i += 1;
                Some(ValuePair::Str((ts[idx], values[idx].clone()))) // TODO - figure out
            }
            Self::Unsigned { i, ts, values } => {
                let idx = *i;
                *i += 1;
                Some(ValuePair::U64((ts[idx], values[idx])))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self {
            Self::Float { i, ts, .. } => *i == ts.len(),
            Self::Integer { i, ts, .. } => *i == ts.len(),
            Self::Bool { i, ts, .. } => *i == ts.len(),
            Self::Str { i, ts, .. } => *i == ts.len(),
            Self::Unsigned { i, ts, .. } => *i == ts.len(),
        }
    }

    pub fn len(&self) -> usize {
        match &self {
            Self::Float { ts, .. } => ts.len(),
            Self::Integer { ts, .. } => ts.len(),
            Self::Bool { ts, .. } => ts.len(),
            Self::Str { ts, .. } => ts.len(),
            Self::Unsigned { ts, .. } => ts.len(),
        }
    }

    /// Merges multiple blocks of data together.
    ///
    /// For values within the block that have identical timestamps, `merge`
    /// overwrites previous values. Therefore, in order to have "last write
    /// wins" semantics it is important that the provided vector of blocks
    /// is ordered by the wall-clock time the blocks were created.
    #[allow(clippy::manual_flatten)]
    pub fn merge(mut blocks: Vec<Self>) -> Self {
        if blocks.is_empty() {
            panic!("merge called with zero blocks");
        } else if blocks.len() == 1 {
            return blocks.remove(0); // only one block; no merging.
        }

        // The merged output block data to be returned
        let mut block_data = Self::new_from_data(blocks.first().unwrap());

        // buf will hold the next candidates from each of the sorted input
        // blocks.
        let mut buf = vec![None; blocks.len()];

        // TODO(edd): perf - this simple iterator approach will likely be sped
        // up by batch merging none-overlapping sections of candidate inputs.
        loop {
            match Self::refill_buffer(&mut blocks, &mut buf) {
                Some(min_ts) => {
                    let mut next_pair = None;
                    // deduplicate points that have same timestamp.
                    for pair in &mut buf {
                        if let Some(vp) = pair {
                            if vp.timestamp() == min_ts {
                                // remove the data from the candidate buffer so it
                                // can be refilled next time around
                                next_pair = pair.take();
                            }
                        }
                    }

                    if let Some(vp) = next_pair {
                        block_data.push(vp);
                    } else {
                        // TODO(edd): it feels like we should be able to re-jig
                        // this so the compiler can prove that there is always
                        // a next_pair.
                        panic!("value pair missing from buffer");
                    }
                }
                None => return block_data, // all inputs drained
            }
        }
    }

    fn refill_buffer(blocks: &mut [Self], dst: &mut Vec<Option<ValuePair>>) -> Option<i64> {
        let mut min_ts = None;
        for (block, dst) in blocks.iter_mut().zip(dst) {
            if dst.is_none() {
                *dst = block.next_pair();
            }

            if let Some(pair) = dst {
                match min_ts {
                    Some(min) => {
                        if pair.timestamp() < min {
                            min_ts = Some(pair.timestamp());
                        }
                    }
                    None => min_ts = Some(pair.timestamp()),
                }
            };
        }
        min_ts
    }
}

// ValuePair represents a single timestamp-value pair from a TSM block.
#[derive(Debug, PartialEq, Clone)]
pub enum ValuePair {
    F64((i64, f64)),
    I64((i64, i64)),
    Bool((i64, bool)),
    Str((i64, Vec<u8>)),
    U64((i64, u64)),
}

impl ValuePair {
    // The timestamp associated with the value pair.
    pub fn timestamp(&self) -> i64 {
        match *self {
            Self::F64((ts, _)) => ts,
            Self::I64((ts, _)) => ts,
            Self::Bool((ts, _)) => ts,
            Self::Str((ts, _)) => ts,
            Self::U64((ts, _)) => ts,
        }
    }
}

/// `TSMBlockReader` allows you to read and decode TSM blocks from within a TSM
/// file.
#[derive(Debug)]
pub struct TsmBlockReader<R>
where
    R: Read + Seek,
{
    readers: Vec<R>,
}

impl<R> TsmBlockReader<R>
where
    R: Read + Seek,
{
    pub fn new(r: R) -> Self {
        Self { readers: vec![r] }
    }

    pub fn add_reader(&mut self, r: R) {
        self.readers.push(r);
    }
}

impl<R> BlockDecoder for TsmBlockReader<R>
where
    R: Read + Seek,
{
    /// decode a block whose location is described by the provided
    /// `Block`.
    ///
    /// The components of the returned `BlockData` are guaranteed to have
    /// identical lengths.
    fn decode(&mut self, block: &Block) -> Result<BlockData, EncodingError> {
        match self.readers.get_mut(block.reader_idx) {
            Some(r) => {
                r.seek(SeekFrom::Start(block.offset))?;

                let mut data: Vec<u8> = vec![0; block.size as usize];
                r.read_exact(&mut data)?;

                // TODO(edd): skip 32-bit CRC checksum at beginning of block for now
                let mut idx = 4;

                // determine the block type
                let block_type = BlockType::try_from(data[idx])?;
                idx += 1;

                // first decode the timestamp block.
                let mut ts = Vec::with_capacity(MAX_BLOCK_VALUES); // 1000 is the max block size
                                                                   // size of timestamp block
                let (len, n) = u64::decode_var(&data[idx..]).ok_or_else(|| EncodingError {
                    description: "unable to decode timestamp".into(),
                })?;

                idx += n;
                encoders::timestamp::decode(&data[idx..idx + (len as usize)], &mut ts).map_err(
                    |e| EncodingError {
                        description: e.to_string(),
                    },
                )?;
                idx += len as usize;

                match block_type {
                    BlockType::Float => {
                        // values will be same length as time-stamps.
                        let mut values = Vec::with_capacity(ts.len());
                        encoders::float::decode_influxdb(&data[idx..], &mut values).map_err(
                            |e| EncodingError {
                                description: e.to_string(),
                            },
                        )?;

                        Ok(BlockData::Float { i: 0, ts, values })
                    }
                    BlockType::Integer => {
                        // values will be same length as time-stamps.
                        let mut values = Vec::with_capacity(ts.len());
                        encoders::integer::decode(&data[idx..], &mut values).map_err(|e| {
                            EncodingError {
                                description: e.to_string(),
                            }
                        })?;

                        Ok(BlockData::Integer { i: 0, ts, values })
                    }
                    BlockType::Bool => {
                        // values will be same length as time-stamps.
                        let mut values = Vec::with_capacity(ts.len());
                        encoders::boolean::decode(&data[idx..], &mut values).map_err(|e| {
                            EncodingError {
                                description: e.to_string(),
                            }
                        })?;

                        Ok(BlockData::Bool { i: 0, ts, values })
                    }
                    BlockType::Str => {
                        // values will be same length as time-stamps.
                        let mut values = Vec::with_capacity(ts.len());
                        encoders::string::decode(&data[idx..], &mut values).map_err(|e| {
                            EncodingError {
                                description: e.to_string(),
                            }
                        })?;
                        Ok(BlockData::Str { i: 0, ts, values })
                    }
                    BlockType::Unsigned => {
                        // values will be same length as time-stamps.
                        let mut values = Vec::with_capacity(ts.len());
                        encoders::unsigned::decode(&data[idx..], &mut values).map_err(|e| {
                            EncodingError {
                                description: e.to_string(),
                            }
                        })?;
                        Ok(BlockData::Unsigned { i: 0, ts, values })
                    }
                }
            }
            None => Err(EncodingError {
                description: format!("cannot decode block {:?} with no associated decoder", block),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use flate2::read::GzDecoder;
    use std::fs::File;
    use std::io::BufReader;
    use std::io::Cursor;
    use std::io::Read;

    use super::*;

    #[test]
    fn decode_tsm_blocks() {
        let file = File::open("../tests/fixtures/000000000000005-000000002.tsm.gz");
        let mut decoder = GzDecoder::new(file.unwrap());
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf).unwrap();
        let r = Cursor::new(buf);

        let mut block_reader = TsmBlockReader::new(BufReader::new(r));

        let block_defs = vec![
            Block {
                min_time: 1590585530000000000,
                max_time: 1590590600000000000,
                offset: 5339,
                size: 153,
                typ: BlockType::Float,
                reader_idx: 0,
            },
            Block {
                min_time: 1590585520000000000,
                max_time: 1590590600000000000,
                offset: 190770,
                size: 30,
                typ: BlockType::Integer,
                reader_idx: 0,
            },
        ];

        let mut blocks = vec![];
        for def in block_defs {
            blocks.push(block_reader.decode(&def).unwrap());
        }

        for block in blocks {
            // The first integer block in the value should have 509 values in it.
            match block {
                BlockData::Float { ts, values, .. } => {
                    assert_eq!(ts.len(), 507);
                    assert_eq!(values.len(), 507);
                }
                BlockData::Integer { ts, values, .. } => {
                    assert_eq!(ts.len(), 509);
                    assert_eq!(values.len(), 509);
                }
                other => panic!("should not have decoded {:?}", other),
            }
        }
    }

    #[test]
    fn refill_buffer() {
        let mut buf = vec![None; 2];
        let mut blocks = vec![
            BlockData::Float {
                i: 0,
                ts: vec![1, 2, 3],
                values: vec![1.2, 2.3, 4.4],
            },
            BlockData::Float {
                i: 0,
                ts: vec![2],
                values: vec![20.2],
            },
        ];

        let mut min_ts = BlockData::refill_buffer(&mut blocks, &mut buf);
        assert_eq!(min_ts.unwrap(), 1);
        assert_eq!(buf[0].take().unwrap(), ValuePair::F64((1, 1.2)));
        assert_eq!(buf[1].take().unwrap(), ValuePair::F64((2, 20.2)));

        // input buffer drained via take calls above - refill
        min_ts = BlockData::refill_buffer(&mut blocks, &mut buf);
        assert_eq!(min_ts.unwrap(), 2);
        assert_eq!(buf[0].take().unwrap(), ValuePair::F64((2, 2.3)));
        assert_eq!(buf[1].take(), None);
    }

    #[test]
    fn merge_blocks() {
        let res = BlockData::merge(vec![
            BlockData::Integer {
                i: 0,
                ts: vec![1, 2, 3],
                values: vec![10, 20, 30],
            },
            BlockData::Integer {
                i: 0,
                ts: vec![2, 4],
                values: vec![200, 300],
            },
        ]);

        assert_eq!(
            res,
            BlockData::Integer {
                i: 0,
                ts: vec![1, 2, 3, 4],
                values: vec![10, 200, 30, 300],
            },
        );
    }
}
