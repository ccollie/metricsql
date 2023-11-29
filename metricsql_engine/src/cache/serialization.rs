use std::io::Write;
use std::mem::size_of;
use std::sync::Arc;

use q_compress::data_types::NumberLike;
use q_compress::errors::QCompressError;
use q_compress::wrapped::{ChunkSpec, Compressor, Decompressor};
use q_compress::CompressorConfig;

use crate::common::encoding::{marshal_var_i64, read_usize, write_usize};
use crate::{MetricName, RuntimeError, RuntimeResult, Timeseries, Timestamp};

fn get_value_compressor(values: &[f64]) -> Compressor<f64> {
    Compressor::<f64>::from_config(q_compress::auto_compressor_config(
        values,
        q_compress::DEFAULT_COMPRESSION_LEVEL,
    ))
}

pub(crate) fn compress_series(series: &[Timeseries], buf: &mut Vec<u8>) -> RuntimeResult<()> {
    const DATA_PAGE_SIZE: usize = 3000;

    let series_count = series.len();

    // write out series count
    write_usize(buf, series_count);

    if series_count == 0 {
        return Ok(());
    }

    // Invariant: the length of all
    let q_timestamps = &series[0].timestamps;
    let mut count = q_timestamps.len();

    let mut data_page_sizes = Vec::with_capacity(count / DATA_PAGE_SIZE + 1);

    while count > 0 {
        let page_size = count.min(DATA_PAGE_SIZE);
        data_page_sizes.push(page_size);
        count -= page_size;
    }

    let chunk_spec = ChunkSpec::default().with_page_sizes(data_page_sizes.clone());

    // the caller ensures that timestamps are equally spaced, so we can use delta encoding
    let ts_compressor_config = CompressorConfig::default().with_delta_encoding_order(2);

    let mut t_compressor = Compressor::<i64>::from_config(ts_compressor_config);

    let mut value_compressors: Vec<_> = series
        .iter()
        .map(|s| get_value_compressor(&s.values))
        .collect();

    // timestamp metadata
    write_metadata(&mut t_compressor, buf, q_timestamps, &chunk_spec)?;

    // write out value chunk metadata
    for (compressor, series) in value_compressors.iter_mut().zip(series.iter()) {
        write_metadata(compressor, buf, &series.values, &chunk_spec)?;
    }

    // write out series labels
    for series in series.iter() {
        series.metric_name.marshal(buf);
    }

    let mut idx = 0;
    for page_size in data_page_sizes.iter() {
        // Each page consists of
        // 1. count
        // 2. timestamp min and max (for fast decompression filtering)
        // 3. timestamp compressed body size
        // 4. timestamp page
        // 5. values compressed body size
        // 6. values page

        // 1.
        write_usize(buf, *page_size);

        // 2.
        // There's no reason you have to serialize the timestamp metadata in the
        // same way as q_compress. Here we do it out of convenience.
        let t_min = q_timestamps[idx];
        idx += page_size;
        let t_max = q_timestamps[idx - 1];
        write_timestamp(buf, t_min);
        write_timestamp(buf, t_max);

        if *page_size == 0 {
            continue;
        }

        // for ease of seeking by date range, we write out the full compressed data page size (timestamps and data)
        t_compressor.data_page().map_err(map_err)?;
        for v_compressor in value_compressors.iter_mut() {
            v_compressor.data_page().map_err(map_err)?;
        }

        let data_size = t_compressor.byte_size()
            + size_of::<usize>()
            + value_compressors
                .iter_mut()
                .map(|c| c.byte_size() + size_of::<usize>())
                .sum::<usize>();

        write_usize(buf, data_size);

        // 3.
        write_usize(buf, t_compressor.byte_size());

        // 4.
        buf.extend(t_compressor.drain_bytes());

        for v_compressor in value_compressors.iter_mut() {
            // 5.
            write_usize(buf, v_compressor.byte_size());

            // 6.
            buf.extend(v_compressor.drain_bytes());
        }
    }

    Ok(())
}

pub(crate) fn deserialize_series_between(
    compressed: &[u8],
    t0: i64,
    t1: i64,
) -> RuntimeResult<Vec<Timeseries>> {
    // read series count
    let (compressed, series_count) = read_usize(compressed, "series count")?;
    if series_count == 0 {
        return Ok(vec![]);
    }

    // read timestamp metadata
    let mut t_decompressor = Decompressor::<i64>::default();
    let mut compressed = read_metadata(&mut t_decompressor, compressed)?;

    let mut v_decompressors: Vec<_> = (0..series_count)
        .map(|_| Decompressor::<f64>::default())
        .collect();

    for v_decompressor in v_decompressors.iter_mut() {
        compressed = read_metadata(v_decompressor, compressed)?;
    }

    // read series labels
    let mut labels = Vec::with_capacity(series_count);
    for _ in 0..series_count {
        let (c, s) = MetricName::unmarshal(compressed)?;
        compressed = c;
        labels.push(s);
    }

    // todo: init capacity
    let mut timestamps = Vec::new();

    let mut res: Vec<Timeseries> = labels
        .into_iter()
        .map(|metric_name| Timeseries {
            metric_name,
            timestamps: Arc::new(vec![]),
            values: vec![],
        })
        .collect();

    #[allow(unused_assignments)]
    let mut size: usize = 0;

    while !compressed.is_empty() {
        (compressed, size) = read_usize(compressed, "data length")?;
        let n = size;

        #[allow(unused_assignments)]
        let mut ts: i64 = 0;

        (compressed, ts) = read_timestamp(compressed)?;

        if ts > t1 {
            break;
        }

        (compressed, ts) = read_timestamp(compressed)?;

        // size of data segment (timestamps and values)
        (compressed, size) = read_usize(compressed, "data segment length")?;
        let data_size = size;

        if ts < t0 {
            // we can skip this data
            compressed = &compressed[data_size..];
        } else {
            // we need to filter and append this data
            let (page_t, remaining) = read_timestamp_page(&mut t_decompressor, compressed, n)?;
            compressed = remaining;

            if page_t.is_empty() {
                continue;
            }

            let first = page_t[0];
            if first > t1 {
                break;
            }

            let (ts_start_index, ts_end_index) = get_timestamp_index_bounds(&page_t, t0, t1);
            timestamps.extend_from_slice(&page_t[ts_start_index..=ts_end_index]);

            for (v_decompressor, series) in v_decompressors.iter_mut().zip(res.iter_mut()) {
                let (page_v, remaining) = read_values_page(v_decompressor, compressed, n)?;

                compressed = remaining;

                series
                    .values
                    .extend_from_slice(&page_v[ts_start_index..=ts_end_index]);
            }

            let end = page_t[page_t.len() - 1];
            if end > t1 {
                break;
            }
        }
    }

    let timestamps = Arc::new(timestamps);
    for series in res.iter_mut() {
        series.timestamps = Arc::clone(&timestamps);
    }

    Ok(res)
}

pub(crate) fn get_timestamp_index_bounds(
    timestamps: &[i64],
    start_ts: Timestamp,
    end_ts: Timestamp,
) -> (usize, usize) {
    let stamps = &timestamps[0..];
    let start_idx = match stamps.binary_search(&start_ts) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };

    let end_idx = match stamps.binary_search(&end_ts) {
        Ok(idx) => idx,
        Err(idx) => idx - 1,
    };

    (start_idx, end_idx)
}

pub(super) fn estimate_size(tss: &[Timeseries]) -> usize {
    if tss.is_empty() {
        return 0;
    }
    // estimate size of labels
    let labels_size = tss
        .iter()
        .fold(0, |acc, ts| acc + ts.metric_name.serialized_size());
    let value_size = tss.iter().fold(0, |acc, ts| acc + ts.values.len() * 8);
    let timestamp_size = 8 * tss[0].timestamps.len();

    // Calculate the required size for marshaled tss.
    labels_size + value_size + timestamp_size
}

fn write_metadata<T: NumberLike>(
    compressor: &mut Compressor<T>,
    dest: &mut Vec<u8>,
    values: &[T],
    chunk_spec: &ChunkSpec,
) -> RuntimeResult<()> {
    // timestamp metadata
    compressor.header().map_err(map_err)?;
    compressor
        .chunk_metadata(values, chunk_spec)
        .map_err(map_err)?;
    write_usize(dest, compressor.byte_size());
    dest.extend(compressor.drain_bytes());
    Ok(())
}

fn read_metadata<'a, T: NumberLike>(
    decompressor: &mut Decompressor<T>,
    compressed: &'a [u8],
) -> RuntimeResult<&'a [u8]> {
    let (mut tail, size) = read_usize(compressed, "metadata length")?;
    decompressor.write_all(&tail[..size]).unwrap();
    decompressor.header().map_err(map_err)?;
    decompressor.chunk_metadata().map_err(map_err)?;
    tail = &tail[size..];
    Ok(tail)
}

fn read_timestamp_page<'a>(
    decompressor: &mut Decompressor<i64>,
    compressed: &'a [u8],
    page: usize,
) -> RuntimeResult<(Vec<i64>, &'a [u8])> {
    let (mut compressed, size) = read_usize(compressed, "timestamp data size")?;
    decompressor.write_all(&compressed[..size]).map_err(|e| {
        let msg = format!("error reading timestamp data: {:?}", e);
        RuntimeError::SerializationError(msg)
    })?;
    let page = decompressor.data_page(page, size).map_err(map_err)?;

    compressed = &compressed[size..];
    Ok((page, compressed))
}

fn read_values_page<'a>(
    decompressor: &mut Decompressor<f64>,
    compressed: &'a [u8],
    page: usize,
) -> RuntimeResult<(Vec<f64>, &'a [u8])> {
    let (mut compressed, size) = read_usize(compressed, "value data size")?;
    decompressor.write_all(&compressed[..size]).map_err(|e| {
        let msg = format!("error reading value data: {:?}", e);
        RuntimeError::SerializationError(msg)
    })?;
    let page = decompressor.data_page(page, size).map_err(map_err)?;

    compressed = &compressed[size..];
    Ok((page, compressed))
}

fn map_err(e: QCompressError) -> RuntimeError {
    RuntimeError::SerializationError(e.to_string())
}

fn read_timestamp(compressed: &[u8]) -> RuntimeResult<(&[u8], i64)> {
    crate::common::encoding::read_i64(compressed, "timestamp")
}

fn write_timestamp(dest: &mut Vec<u8>, ts: i64) {
    marshal_var_i64(dest, ts);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_series(tss: &[Timeseries]) {
        let mut buffer: Vec<u8> = Vec::new();
        compress_series(tss, &mut buffer).unwrap();
        let tss2 = deserialize_series_between(&buffer, 0, 100000).unwrap();

        assert_eq!(
            tss, tss2,
            "unexpected timeseries unmarshaled\ngot\n{:?}\nwant\n{:?}",
            tss2[0], tss[0]
        )
    }

    // Single series
    #[test]
    fn single_series() {
        let series = Timeseries::default();
        test_series(&[series]);

        let mut series = Timeseries::default();
        series.metric_name.metric_group = "foobar".to_string();
        series.metric_name.add_tag("tag1", "value1");
        series.metric_name.add_tag("tag2", "value2");

        series.values = vec![1.0, 2.0, 3.234];
        series.timestamps = Arc::new(vec![10, 20, 30]);
        test_series(&[series]);
    }

    #[test]
    fn multiple_series() {
        let mut series = Timeseries::default();
        series.metric_name.metric_group = "foobar".to_string();
        series.metric_name.add_tag("tag1", "value1");
        series.metric_name.add_tag("tag2", "value2");

        series.values = vec![1.0, 2.34, -33.0];
        series.timestamps = Arc::new(vec![0, 10, 20]);

        let mut series2 = Timeseries::default();
        series2.metric_name.metric_group = "baz".to_string();
        series2.metric_name.add_tag("tag12", "value13");

        series2.values = vec![4.0, 1.0, -2.34];
        series2.timestamps = Arc::new(vec![0, 10, 20]);

        test_series(&[series, series2]);
    }
}
