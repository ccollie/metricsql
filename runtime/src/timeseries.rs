use std::borrow::BorrowMut;
use std::fmt::Debug;
use std::sync::Arc;

use byte_slice_cast::{AsByteSlice, AsSliceOf};
use integer_encoding::VarInt;
use lockfree_object_pool::LinearObjectPool;
use once_cell::sync::OnceCell;

use lib::{marshal_var_int, unmarshal_uint16};

use crate::MetricName;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::traits::{Timestamp, TimestampTrait};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Timeseries {
    pub metric_name: MetricName,
    pub values: Vec<f64>,
    pub timestamps: Arc<Vec<i64>>, //Arc used vs Rc since Rc is !Send
}

impl Timeseries {
    pub fn new(timestamps: Vec<i64>, values: Vec<f64>) -> Self {
        Timeseries {
            metric_name: MetricName::default(),
            values,
            timestamps: Arc::new(timestamps),
        }
    }

    pub fn copy(src: &Timeseries) -> Self {
        Timeseries {
            timestamps: src.timestamps.clone(),
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
        }
    }

    pub fn copy_from_metric_name(src: &Timeseries) -> Self {
        let ts = Timeseries {
            timestamps: src.timestamps.clone(),
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
        };
        ts
    }


    pub fn with_shared_timestamps(timestamps: &Arc<Vec<i64>>, values: &[f64]) -> Self {
        Timeseries {
            metric_name: MetricName::default(),
            values: Vec::from(values),
            // see https://pkolaczk.github.io/server-slower-than-a-laptop/ under the section #the fix
            timestamps: Arc::new(timestamps.as_ref().clone()),   // clones the value under Arc and wraps it in a new counter
        }
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.timestamps = Arc::new(vec![]);
        self.metric_name.reset();
    }

    pub fn copy_from_shallow_timestamps(src: &Timeseries) -> Self {
        Timeseries {
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
            timestamps: Arc::clone(&src.timestamps),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// unmarshal_timeseries_fast unmarshals timeseries from src.
    ///
    /// The returned timeseries refer to src, so it is unsafe to modify it
    /// until timeseries are in use.
    pub fn unmarshal_fast(src: &[u8]) -> RuntimeResult<Vec<Timeseries>> {
        let mut timestamps: Vec<i64> = Vec::with_capacity(4);
        let mut src = unmarshal_fast_timestamps(src, &mut timestamps)?;

        let mut tss: Vec<Timeseries> = Vec::with_capacity(1);
        let timestamps: Arc<Vec<i64>> = Arc::new(timestamps.into());

        while src.len() > 0 {
            let (ts, tail) = Timeseries::unmarshal_fast_no_timestamps(src, &timestamps)?;
            src = tail;

            tss.push(ts)
        }

        Ok(tss)
    }

    /// appends marshaled ts to dst and returns the result.
    ///
    /// It doesn't marshal timestamps.
    ///
    /// The result must be unmarshalled with unmarshal_fast_no_timestamps.
    pub fn marshal_fast_no_timestamps(&self, dst: &mut Vec<u8>) {
        marshal_bytes_fast(dst, self.metric_name.metric_group.as_bytes());
        marshal_var_int::<usize>(dst, self.tag_count() as usize);
        // There is no need in tags' sorting - they must be sorted after unmarshalling.
        self.metric_name.marshal_tags_fast(dst);

        // do not marshal ts.values.len(), since it is already encoded as ts.timestamps.len()
        // during marshal_fast_timestamps.
        if self.values.len() > 0 {
            let values_buf = self.values.as_byte_slice();
            dst.extend_from_slice(values_buf);
        }
    }


    /// marshaled_fast_size_no_timestamps returns the size of marshaled ts
    /// returned from marshal_fast_no_timestamps.
    fn marshaled_fast_size_no_timestamps(&self) -> usize {
        let mut n = self.metric_name.serialized_size();
        n += 8 * self.values.len();
        return n
    }

    /// unmarshalFastNoTimestamps unmarshal ts from src, so ts members reference src.
    ///
    /// It is expected that ts.timestamps is already unmarshaled.
    pub fn unmarshal_fast_no_timestamps<'a>(src: &'a [u8], timestamps: &Arc<Vec<i64>>) -> RuntimeResult<(Timeseries, &'a [u8])> {

        let (metric_name, tail) = MetricName::unmarshal_fast(src)?;
        let src = tail;

        let mut ts = Timeseries {
            metric_name,
            values: vec![],
            timestamps: Arc::clone(&timestamps)
        };

        if timestamps.len() == 0 {
            return Ok((ts, src))
        }

        let buf_size = timestamps.len() * 8;
        if src.len() < buf_size {
            return Err(RuntimeError::SerializationError(
                format!("cannot unmarshal values; got {} bytes; want at least {} bytes", src.len(), buf_size)
            ));
        }

        match src[0..buf_size].as_slice_of::<f64>() {
            Err(e) => {
                return Err(RuntimeError::SerializationError(
                    format!("cannot unmarshal Timestamp::values from byte slice: {:?}", e)
                ));
            },
            Ok(values) => {
                ts.values = Vec::from(values);
            }
        }

        let src = &src[buf_size .. ];

        Ok((ts, src))
    }

    #[inline]
    pub fn tag_count(&self) -> usize {
        self.metric_name.get_tag_count()
    }
}

pub fn marshal_timeseries_fast(dst: &mut Vec<u8>, tss: &[Timeseries], max_size: usize, step: i64) -> RuntimeResult<()> {
    if tss.len() == 0 {
        return Err(RuntimeError::ArgumentError("BUG: tss cannot be empty".to_string()))
    }

    // Calculate the required size for marshaled tss.
    let mut size = 0;
    for ts in tss {
        size += ts.marshaled_fast_size_no_timestamps()
    }
    // timestamps are stored only once for all the tss, since they are identical.
    assert_identical_timestamps(tss, step)?;
    size += 8 * tss[0].timestamps.len();

    if size > max_size {
        // do not marshal tss, since it would occupy too much space
        return Ok(());
    }

    // Allocate the buffer for the marshaled tss before its' marshaling.
    // This should reduce memory fragmentation and memory usage.
    dst.reserve(size);
    marshal_fast_timestamps(dst, &tss[0].timestamps);
    for ts in tss {
        ts.marshal_fast_no_timestamps(dst);
    }
    Ok(())
}

/// unmarshal_timeseries_fast unmarshals timeseries from src.
pub(crate) fn unmarshal_timeseries_fast(src: &[u8]) -> RuntimeResult<Vec<Timeseries>> {
    let mut timestamps: Vec<i64> = Vec::with_capacity(4);
    let tail = unmarshal_fast_timestamps(src, &mut timestamps)?;
    let mut src = tail;

    let mut tss: Vec<Timeseries> = Vec::with_capacity(1);
    let shared_timestamps: Arc<Vec<i64>> = Arc::new(timestamps.into());

    while src.len() > 0 {
        let (ts, tail) = Timeseries::unmarshal_fast_no_timestamps(src, &shared_timestamps)?;
        tss.push(ts);
        src = tail;
    }

    Ok(tss)
}

pub fn marshal_fast_timestamps(dst: &mut Vec<u8>, timestamps: &Vec<i64>) {
    marshal_var_int(dst, timestamps.len());
    if timestamps.len() > 0 {
        let timestamps_buf = timestamps.as_byte_slice();
        dst.extend_from_slice(timestamps_buf);
    }
}

/// it is unsafe modifying src while the returned timestamps is in use.
pub fn unmarshal_fast_timestamps<'a>(src: &'a [u8], dst: &mut Vec<i64>) -> RuntimeResult<&'a [u8]> {
    unmarshal_fast_values(src, dst)
}

pub fn unmarshal_fast_values<'a, T>(src: &'a [u8], dst: &mut Vec<T>) -> RuntimeResult<&'a [u8]>
where T: Clone + byte_slice_cast::FromByteSlice
{
    if src.len() < 4 {
        return Err(RuntimeError::from(
            format!("cannot decode len(values); got {} bytes; want at least {} bytes", src.len(), 4)
        ));
    }

    let mut src = src;
    let mut item_count: usize = 0;

    match i32::decode_var(src) {
        Some((v, size)) => {
            src = &src[size..];
            item_count = v as usize;
        },
        None => {

        }
    }

    let mut src = &src[4..];
    if item_count == 0 {
        return Ok(src)
    }

    let buf_size = item_count * 8;
    if src.len() < buf_size {
        return Err(RuntimeError::from(
            format!("cannot unmarshal values; got {} bytes; want at least {} bytes", src.len(), buf_size)
        ));
    }

    let buf = &src[0..buf_size];
    dst.extend_from_slice(buf.as_slice_of::<T>().unwrap() );
    src = &src[buf_size..];

    Ok(src)
}

/// unmarshal_fast_no_timestamps unmarshals ts from src, so ts members reference src.
///
/// It is expected that ts.timestamps is already unmarshaled.
pub(crate) fn unmarshal_fast_no_timestamps<'a>(ts: &'a mut Timeseries, src: &'a [u8]) -> RuntimeResult<&'a [u8]> {
    // ts members point to src, so they cannot be re-used.

    let tail;
    match ts.metric_name.unmarshal_fast_internal(src) {
        Err(e) => return Err(
            RuntimeError::from(format!("cannot unmarshal MetricName: {:?}", e))
        ),
        Ok(t) => tail = t
    }

    let src = tail;
    let values_count = ts.timestamps.len();
    if values_count == 0 {
        return Ok(src);
    }
    let buf_size = values_count * 8;
    if src.len() < buf_size {
        let msg = format!("cannot unmarshal values; got {} bytes; want at least {} bytes", src.len(), buf_size);
        return Err(RuntimeError::from(msg));
    }

    unmarshal_fast_values(tail,&mut ts.values)
}


pub fn marshal_bytes_fast(dst: &mut Vec<u8>, s: &[u8]) {
    marshal_var_int::<usize>(dst,s.len());
    dst.extend_from_slice(s);
}

pub fn unmarshal_bytes_fast(src: &[u8]) -> RuntimeResult<(&[u8], &[u8])> {
    if src.len() < 2 {
        return Err(RuntimeError::SerializationError(
            format!("cannot decode size; it must be at least 2 bytes")
        ));
    }

    match unmarshal_uint16(src) {
        Ok((len, tail)) => {
            if tail.len() < len as usize {
                return Err(RuntimeError::SerializationError(
                    format!("too short src; it must be at least {} bytes", len)
                ));
            }
            Ok(src.split_at(len as usize))
        },
        Err(_) => {
            return Err(RuntimeError::SerializationError("error reading byte length".to_string()));
        }
    }
}

/// returns a copy of arg with shallow copies of MetricNames,
/// Timestamps and Values.
pub(crate) fn copy_timeseries_shallow(arg: &Vec<Timeseries>) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(arg.len());
    for src in arg.iter() {
        let dst = Timeseries::copy_from_shallow_timestamps(&src);
        rvs.push(dst);
    }
    return rvs;
}

pub(super) fn assert_identical_timestamps(tss: &[Timeseries], step: i64) -> RuntimeResult<()> {
    if tss.len() == 0 {
        return Ok(())
    }
    let ts_golden = &tss[0];
    if ts_golden.values.len() != ts_golden.timestamps.len() {
        let msg = format!("BUG: ts_golden.values.leen() must match ts_golden.timestamps.len(); got {} vs {}",
                      ts_golden.values.len(), ts_golden.timestamps.len());
        return Err(RuntimeError::from(msg));
    }
    if ts_golden.timestamps.len() > 0 {
        let mut prev_timestamp = ts_golden.timestamps[0];
        for timestamp in ts_golden.timestamps[1..].iter() {
            if timestamp - prev_timestamp != step {
                let msg = format!("BUG: invalid step between timestamps; got {}; want {};",
                                  timestamp - prev_timestamp, step);
                return Err(RuntimeError::from(msg));
            }
            prev_timestamp = *timestamp
        }
    }
    for ts in tss.iter() {
        if ts.values.len() != ts_golden.values.len() {
            let msg = format!("BUG: unexpected ts.values.len(); got {}; want {}; ts.values={}",
                          ts.values.len(),
                          ts_golden.values.len(),
                          ts.values.len());
            return Err(RuntimeError::from(msg));
        }
        if ts.timestamps.len() != ts_golden.timestamps.len() {
            let msg = format!("BUG: unexpected ts.timestamps.len(); got {}; want {};",
                          ts.timestamps.len(), ts_golden.timestamps.len());
            return Err(RuntimeError::from(msg));
        }
        if ts.timestamps.len() == 0 {
            continue
        }
        if &ts.timestamps[0] == &ts_golden.timestamps[0] {
            // Fast path - shared timestamps.
            continue
        }
        for i in 0 .. ts.timestamps.len() {
            if ts.timestamps[i] != ts_golden.timestamps[i] {
                let msg = format!("BUG: timestamps mismatch at position {}; got {}; want {};",
                              i, ts.timestamps[i], ts_golden.timestamps[i]);
                return Err(RuntimeError::from(msg));
            }
        }
    }
    Ok(())
}


#[derive(Clone)]
struct TimeseriesPoolEntry {
    timeseries: Timeseries,
    last_reset_time: Timestamp
}

impl TimeseriesPoolEntry {
    fn reset(mut self) {
        reset_timeseries(&mut self);
    }
}

impl Default for TimeseriesPoolEntry {
    fn default() -> Self {
        Self {
            timeseries: Timeseries::default(),
            last_reset_time: Timestamp::now()
        }
    }
}

fn reset_timeseries(v: &mut TimeseriesPoolEntry) {
    v.timeseries.values.clear();
    let current_time = Timestamp::now();
    // ts.timestamps points to shared_timestamps. Zero it, so it can be re-used.
    v.timeseries.timestamps = Arc::new(vec![]);
    let cap = v.timeseries.values.capacity();
    if cap > 1024*1024 && (4 * v.timeseries.values.len()) < cap &&
        (current_time - v.last_reset_time) > 100 {
        // Reset r.rs in order to preserve memory usage after processing big time series with
        // millions of rows.
        let ts_slice = &v.timeseries.timestamps[0..1023];
        v.timeseries.values.shrink_to(1024);
        v.timeseries.timestamps = Arc::new(ts_slice.to_vec());
    }
    v.last_reset_time = current_time;
}

fn timeseries_pool() -> &'static LinearObjectPool<TimeseriesPoolEntry> {
    static INSTANCE: OnceCell<LinearObjectPool<TimeseriesPoolEntry>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        LinearObjectPool::<TimeseriesPoolEntry>::new(
            || TimeseriesPoolEntry::default(),
            |v| {
                reset_timeseries(v)
            }
        )
    })
}

pub(crate) fn get_timeseries() -> &'static mut Timeseries {
    timeseries_pool().pull().timeseries.borrow_mut()
}