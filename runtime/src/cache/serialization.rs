use std::mem::size_of;
use std::sync::Arc;

use pco::data_types::Number;
use pco::errors::PcoError;
use pco::standalone::{simple_compress, simple_compress_into, simple_decompress_into};
use pco::{ChunkConfig, DeltaSpec, PagingSpec};

use crate::common::encoding::marshal_var_i64;
use crate::types::{MetricName, SeriesSlice, Timeseries, Timestamp};
use crate::{RuntimeError, RuntimeResult};

const MAGIC_HEADER: [u8; 4] = [77, 67, 83, 49]; // MCS1

// todo: move elsewhere
#[allow(dead_code)]
pub(crate) fn compress_series(series: &[Timeseries], buf: &mut Vec<u8>) -> RuntimeResult<()> {
    let series_slices: Vec<SeriesSlice> = series
        .iter()
        .map(|s| SeriesSlice {
            metric_name: &s.metric_name,
            timestamps: &s.timestamps,
            values: &s.values,
        })
        .collect();

    compress_series_slice(&series_slices, buf)
}

pub(crate) fn compress_series_slice(
    series: &[SeriesSlice],
    buf: &mut Vec<u8>,
) -> RuntimeResult<()> {
    const DATA_PAGE_SIZE: usize = 1000;

    // write magic header
    buf.extend_from_slice(&MAGIC_HEADER);

    let series_count = series.len();

    // write out series count
    write_usize(buf, series_count);

    if series_count == 0 {
        return Ok(());
    }

    // Invariant: the length of all series is the same
    // Invariant: the timestamps are equal between series
    let q_timestamps = &series[0].timestamps;

    let config = ChunkConfig::default();

    // the caller ensures that timestamps are equally spaced, so we can use delta encoding
    let ts_config = ChunkConfig::default().with_delta_spec(DeltaSpec::TryConsecutive(2));

    // write out value chunk metadata

    // write out series labels
    for series in series.iter() {
        series.metric_name.marshal(buf);
    }

    // Each page consists of
    // 1. count
    // 2. timestamp min and max (for fast decompression filtering)
    // 3. Total data size (timestamps and values). Allows for fast seeking by date range.
    // 4. compressed timestamps body size
    // 5. timestamp page
    // 6. series compressed values size
    // 7. compressed values page

    // 6/7 are repeated for each series
    let mut start_idx = 0usize;
    for timestamps in q_timestamps.chunks(DATA_PAGE_SIZE) {
        let len = timestamps.len();

        // 1.
        write_usize(buf, len);

        // 2.
        let t_min = timestamps[0];
        let t_max = timestamps[len - 1];
        write_timestamp(buf, t_min);
        write_timestamp(buf, t_max);

        // 3.
        // add placeholder for total data size
        let placeholder_offset = buf.len();

        write_usize(buf, 0);
        let data_size_offset = buf.len();

        if len == 0 {
            continue;
        }

        // 4/5
        write_data(buf, timestamps, &ts_config)?;

        for series in series.iter() {
            let values = &series.values[start_idx..start_idx + len];
            // 6/7
            write_data(buf, values, &config)?;
        }

        // patch in the data size
        let data_size = buf.len() - data_size_offset;
        buf[placeholder_offset..placeholder_offset + size_of::<usize>()]
            .copy_from_slice(&data_size.to_le_bytes());

        start_idx += len;
    }

    Ok(())
}


pub(crate) fn deserialize_series_between(
    compressed: &[u8],
    start_ts: i64,
    end_ts: i64,
) -> RuntimeResult<Vec<Timeseries>> {
    if compressed.is_empty() {
        return Ok(vec![]);
    }

    let mut compressed = compressed;

    // check the magic header
    if compressed.len() < 4 || &compressed[..4] != MAGIC_HEADER {
        return Err(RuntimeError::SerializationError(
            "invalid magic header".to_string(),
        ));
    }
    compressed = &compressed[MAGIC_HEADER.len()..];

    // read series count
    let series_count = read_usize(&mut compressed, "series count")?;
    if series_count == 0 {
        return Ok(vec![]);
    }

    // read series labels
    let mut res: Vec<Timeseries> = Vec::with_capacity(series_count);
    for _ in 0..series_count {
        let (c, s) = MetricName::unmarshal(compressed)?;
        compressed = c;
        res.push(Timeseries {
            metric_name: s,
            timestamps: Arc::new(vec![]),
            values: vec![],
        });
    }

    // todo: init capacity
    let mut timestamps = Vec::new();

    // decompression scratch buffers to minimize allocations
    let mut page_t: Vec<i64> = Vec::new();
    let mut page_v: Vec<f64> = Vec::new();

    while !compressed.is_empty() {
        let page_size = read_usize(&mut compressed, "data length")?;

        if page_size == 0 {
            break;
        }

        page_v.resize(page_size, 0.0);
        page_t.resize(page_size, 0);

        let t_min = read_timestamp(&mut compressed)?;
        let t_max = read_timestamp(&mut compressed)?;

        if t_min > end_ts {
            break;
        }

        // size of data segment (timestamps and values)
        let data_size = read_usize(&mut compressed, "data segment length")?;

        if t_max < start_ts {
            // we can skip this data
            compressed = &compressed[data_size..];
        } else {
            // we need to filter and append this data
            let count = read_timestamp_page(&mut compressed, &mut page_t)?;
            if count != page_size {
                return Err(RuntimeError::SerializationError(
                    "incomplete timestamp page".to_string(),
                ));
            }

            if page_t.is_empty() {
                continue;
            }

            let first = page_t[0];
            if first > end_ts {
                break;
            }

            let (ts_start_index, ts_end_index) = get_timestamp_index_bounds(&page_t, start_ts, end_ts);
            let ts_slice = &page_t[ts_start_index..=ts_end_index];
            timestamps.extend_from_slice(ts_slice);

            for series in res.iter_mut() {
                let count = read_values_page(&mut compressed, &mut page_v)?;

                if count != page_size {
                    return Err(RuntimeError::SerializationError(
                        "incomplete data page".to_string(),
                    ));
                }

                series
                    .values
                    .extend_from_slice(&page_v[ts_start_index..=ts_end_index]);
            }

            let end = page_t[page_t.len() - 1];
            if end >= end_ts {
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
    if timestamps.is_empty() {
        return (0, 0);
    }


    let mut start_idx = timestamps.binary_search(&start_ts).unwrap_or_else(|idx| idx);
    let right = &timestamps[start_idx..];

    let idx = find_last_ge_index(right, end_ts);
    let end_idx = start_idx + idx;

    // imagine this scenario:
    // samples = &[10, 20, 30, 40]
    // start = 25, end = 25
    // we have a situation where start_index == end_index (2), yet samples[2] is greater than end,
    if start_idx == end_idx {
        // todo: get_unchecked
        if timestamps[start_idx] > end_ts {
            start_idx = start_idx.saturating_sub(1);
        }
    }

    (start_idx, end_idx)
}

pub fn find_last_ge_index(arr: &[i64], val: i64) -> usize {
    if arr.len() <= 16 {
        return arr.iter().rposition(|&x| val >= x).map_or(0, |idx| {
            if arr[idx] > val {
                idx.saturating_sub(1)
            } else {
                idx
            }
        });
    }
    arr.binary_search(&val)
        .unwrap_or_else(|x| x.saturating_sub(1))
}

fn write_data<T: Number>(
    dest: &mut Vec<u8>,
    values: &[T],
    config: &ChunkConfig,
) -> RuntimeResult<usize> {
    let buf = simple_compress(values, config).map_err(map_err)?;
    write_usize(dest, buf.len());
    dest.extend(&buf);
    Ok(buf.len())
}

fn read_timestamp_page(compressed: &mut &[u8], dst: &mut [i64]) -> RuntimeResult<usize> {
    let size = read_usize(compressed, "timestamp data size")?;
    let progress = simple_decompress_into(compressed, dst).map_err(map_err)?;
    if !progress.finished {
        return Err(RuntimeError::SerializationError(
            "incomplete timestamp data".to_string(),
        ));
    }
    *compressed = &compressed[size..];
    Ok(progress.n_processed)
}

fn read_values_page(compressed: &mut &[u8], dst: &mut [f64]) -> RuntimeResult<usize> {
    let size = read_usize(compressed, "value data size")?;
    let progress = simple_decompress_into(compressed, dst).map_err(map_err)?;
    if !progress.finished {
        return Err(RuntimeError::SerializationError(
            "incomplete value data".to_string(),
        ));
    }

    *compressed = &compressed[size..];
    Ok(progress.n_processed)
}

fn map_err(e: PcoError) -> RuntimeError {
    RuntimeError::SerializationError(e.to_string())
}

fn read_timestamp(compressed: &mut &[u8]) -> RuntimeResult<i64> {
    let (remaining, value) = crate::common::encoding::read_i64(compressed, "timestamp")?;
    *compressed = remaining;
    Ok(value)
}

fn write_timestamp(dest: &mut Vec<u8>, ts: i64) {
    marshal_var_i64(dest, ts);
}

fn write_usize(slice: &mut Vec<u8>, size: usize) {
    slice.extend_from_slice(&size.to_le_bytes());
}

fn read_usize(input: &mut &[u8], field: &str) -> RuntimeResult<usize> {
    let (int_bytes, rest) = input.split_at(size_of::<usize>());
    let buf = int_bytes.try_into().map_err(|_| {
        RuntimeError::SerializationError(format!("invalid usize reading {field}").to_string())
    })?;

    *input = rest;
    Ok(usize::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use rand::prelude::ThreadRng;
    use rand::Rng;
    use super::*;

    fn create_test_metric_name(suffix: &str) -> MetricName {
        let mut rng = ThreadRng::default();
        let mut metric_name = MetricName::default();
        metric_name.measurement = format!("test_metric_{}", suffix);
        metric_name.add_label(&format!("label_{}", rng.random::<u16>()), "value1");
        metric_name.add_label(&format!("label_{}_a", rng.random::<u16>()), "value2");
        metric_name
    }

    fn create_test_series_slice(count: usize, metric_suffix: &str) -> (Vec<i64>, Vec<f64>, MetricName) {
        let mut rng = ThreadRng::default();
        let timestamps: Vec<i64> = (0..count).map(|i| (i as i64) * 1000).collect();
        let values: Vec<f64> = (0..count).map(|_| rng.random_range(100.0..1000.0)).collect();
        let metric_name = create_test_metric_name(metric_suffix);
        (timestamps, values, metric_name)
    }

    fn create_series_slices(count: usize, series_count: usize) -> (Vec<Vec<i64>>, Vec<Vec<f64>>, Vec<MetricName>) {
        let mut all_timestamps = Vec::new();
        let mut all_values = Vec::new();
        let mut all_metric_names = Vec::new();

        // Create shared timestamps for all series
        let shared_timestamps: Vec<i64> = (0..count).map(|i| (i as i64) * 1000).collect();

        for i in 0..series_count {
            let (_, values, metric_name) = create_test_series_slice(count, &format!("series_{}", i));
            all_timestamps.push(shared_timestamps.clone());
            all_values.push(values);
            all_metric_names.push(metric_name);
        }

        (all_timestamps, all_values, all_metric_names)
    }

    #[test]
    fn test_serialize_multiple_series_slices() {
        let (all_timestamps, all_values, all_metric_names) = create_series_slices(200, 3);

        let series_slices: Vec<SeriesSlice> = all_timestamps.iter()
            .zip(all_values.iter())
            .zip(all_metric_names.iter())
            .map(|((ts, vals), metric)| SeriesSlice {
                metric_name: metric,
                timestamps: ts,
                values: vals,
            })
            .collect();

        let mut buffer = Vec::new();
        compress_series_slice(&series_slices, &mut buffer).unwrap();

        let result = deserialize_series_between(&buffer, 0, 200000).unwrap();

        assert_eq!(result.len(), 3);
        for (i, series) in result.iter().enumerate() {
            assert_eq!(series.metric_name, all_metric_names[i]);
            assert_eq!(series.timestamps.as_ref(), &all_timestamps[i]);
            assert_eq!(series.values, all_values[i]);
        }
    }

    #[test]
    fn test_serialize_large_series_slice() {
        // Test with data that spans multiple pages (> 1000 data points)
        let (timestamps, values, metric_name) = create_test_series_slice(2500, "large");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        let result = deserialize_series_between(&buffer, 0, 2500000).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metric_name, metric_name);
        assert_eq!(result[0].timestamps.as_ref(), &timestamps);
        assert_eq!(result[0].values, values);
    }

    #[test]
    fn test_serialize_with_time_range_filtering() {
        let (timestamps, values, metric_name) = create_test_series_slice(100, "filtered");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        // Test filtering: only get data between timestamp 10000 and 50000
        let start_ts = 10000;
        let end_ts = 50000;
        let result = deserialize_series_between(&buffer, start_ts, end_ts).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metric_name, metric_name);

        // Check that all timestamps are within the requested range
        for &ts in result[0].timestamps.iter() {
            assert!(ts >= start_ts && ts <= end_ts,
                    "Timestamp {} is outside range [{}, {}]", ts, start_ts, end_ts);
        }

        // Check that the values match the expected filtered values
        let expected_indices: Vec<usize> = timestamps.iter()
            .enumerate()
            .filter(|(_, &ts)| ts >= start_ts && ts <= end_ts)
            .map(|(i, _)| i)
            .collect();

        let expected_values: Vec<f64> = expected_indices.iter()
            .map(|&i| values[i])
            .collect();

        assert_eq!(result[0].values, expected_values);
    }

    #[test]
    fn test_serialize_edge_cases() {
        // Test with single data point
        let (timestamps, values, metric_name) = create_test_series_slice(1, "single_point");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        let result = deserialize_series_between(&buffer, 0, 1000).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timestamps.len(), 1);
        assert_eq!(result[0].values.len(), 1);
        assert_eq!(result[0].timestamps[0], 0);
        assert_eq!(result[0].values[0], values[0]);
    }

    #[test]
    fn test_serialize_no_matching_time_range() {
        let (timestamps, values, metric_name) = create_test_series_slice(100, "no_match");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        // Request data outside the available range
        let result = deserialize_series_between(&buffer, 200000, 300000).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].timestamps.is_empty());
        assert!(result[0].values.is_empty());
    }

    #[test]
    fn test_serialize_exact_page_boundary() {
        // Test with exactly 1000 data points (page boundary)
        let (timestamps, values, metric_name) = create_test_series_slice(1000, "boundary");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        let result = deserialize_series_between(&buffer, 0, 1000000).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timestamps.as_ref(), &timestamps);
        assert_eq!(result[0].values, values);
    }

    #[test]
    fn test_serialize_multiple_pages() {
        // Test with exactly 2000 data points (2 pages)
        let (timestamps, values, metric_name) = create_test_series_slice(2000, "two_pages");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        // Test filtering across page boundaries
        let start_ts = 500000;  // Middle of first page
        let end_ts = 1500000;   // Middle of second page
        let result = deserialize_series_between(&buffer, start_ts, end_ts).unwrap();

        assert_eq!(result.len(), 1);

        // Verify all timestamps are in range and in order
        for &ts in result[0].timestamps.iter() {
            assert!(ts >= start_ts && ts <= end_ts);
        }

        // Verify the result is not empty (should span across pages)
        assert!(!result[0].timestamps.is_empty());
        assert!(!result[0].values.is_empty());
    }

    #[test]
    fn test_serialize_mixed_series_sizes() {
        // Test multiple series with the same timestamp count but different data
        let series_count = 100;
        let (all_timestamps, all_values, all_metric_names) = create_series_slices(series_count, 4);

        let series_slices: Vec<SeriesSlice> = all_timestamps.iter()
            .zip(all_values.iter())
            .zip(all_metric_names.iter())
            .map(|((ts, vals), metric)| SeriesSlice {
                metric_name: metric,
                timestamps: ts,
                values: vals,
            })
            .collect();

        let mut buffer = Vec::new();
        compress_series_slice(&series_slices, &mut buffer).unwrap();

        let result = deserialize_series_between(&buffer, 0, 100000).unwrap();

        assert_eq!(result.len(), 4);

        // Verify that all series have the same timestamps (shared)
        for i in 1..result.len() {
            assert_eq!(result[0].timestamps, result[i].timestamps);
        }

        // Verify that each series has its own unique values
        for (i, series) in result.iter().enumerate() {
            assert_eq!(series.values, all_values[i]);
            assert_eq!(series.metric_name, all_metric_names[i]);
        }
    }

    #[test]
    fn test_invalid_magic_header() {
        let mut buffer = vec![1, 2, 3, 4]; // Invalid magic header
        buffer.extend_from_slice(&0usize.to_le_bytes()); // series count

        let result = deserialize_series_between(&buffer, 0, 100000);

        assert!(result.is_err());
        match result.unwrap_err() {
            RuntimeError::SerializationError(msg) => {
                assert!(msg.contains("invalid magic header"));
            }
            _ => panic!("Expected SerializationError with magic header message"),
        }
    }

    #[test]
    fn test_buffer_compression_size() {
        // Test that compression actually reduces size for repetitive data
        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        // Create repetitive data that should compress well
        for i in 0..1000 {
            timestamps.push(i * 1000);
            values.push(100.0); // Same value repeated
        }

        let metric_name = create_test_metric_name("compression_test");
        let series_slice = SeriesSlice {
            metric_name: &metric_name,
            timestamps: &timestamps,
            values: &values,
        };

        let mut buffer = Vec::new();
        compress_series_slice(&[series_slice], &mut buffer).unwrap();

        // Compressed size should be significantly smaller than uncompressed
        let uncompressed_size = timestamps.len() * 8 + values.len() * 8; // rough estimate
        assert!(buffer.len() < uncompressed_size / 2,
                "Compression should reduce size significantly for repetitive data");
    }
}