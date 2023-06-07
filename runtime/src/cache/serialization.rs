use std::io::Write;
use crate::{
    MetricName, RuntimeError, RuntimeResult, Timeseries
};
use q_compress::data_types::NumberLike;
use q_compress::errors::{QCompressError};
use q_compress::wrapped::{ChunkSpec, Compressor, Decompressor};
use std::mem::size_of;
use std::sync::Arc;
use lib::{marshal_var_i64, unmarshal_var_i64};
use lib::error::Error;
use crate::utils::{read_usize, write_usize};

fn get_value_compressor(values: &[f64]) -> Compressor<f64> {
    Compressor::<f64>::from_config(q_compress::auto_compressor_config(
        &values,
        q_compress::DEFAULT_COMPRESSION_LEVEL,
    ))
}

pub(crate) fn compress_series(series: &[Timeseries]) -> RuntimeResult<Vec<u8>> {
    const DATA_PAGE_SIZE: usize = 3000;

    let series_count = series.len();
    let q_timestamps = &series[0].timestamps;
    let mut count = q_timestamps.len();

    let mut data_page_sizes = Vec::with_capacity(count / DATA_PAGE_SIZE + 1);

    while count > 0 {
        let page_size = count.min(DATA_PAGE_SIZE);
        data_page_sizes.push(page_size);
        count -= page_size;
    }

    let chunk_spec = ChunkSpec::default().with_page_sizes(data_page_sizes.clone());

    let mut t_compressor = Compressor::<i64>::from_config(q_compress::auto_compressor_config(
        &q_timestamps,
        q_compress::DEFAULT_COMPRESSION_LEVEL,
    ));

    let mut value_compressors: Vec<_> = series
        .iter()
        .map(|s| get_value_compressor(&s.values))
        .collect();

    let mut res = Vec::with_capacity(512);

    // timestamp metadata
    write_metatadata(&mut t_compressor, &mut res, &q_timestamps, &chunk_spec)?;

    // write out series count
    write_usize(&mut res, series_count);

    // write out value chunk metadata
    for (compressor, series) in value_compressors.iter_mut().zip(series.iter()) {
        write_metatadata(compressor, &mut res, &series.values, &chunk_spec)?;
    }

    // write out series labels
    for series in series.iter() {
        series.metric_name.marshal(&mut res);
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
        write_usize(&mut res, *page_size);

        // 2.
        // There's no reason you have to serialize the timestamp metadata in the
        // same way as q_compress. Here we do it out of convenience.
        let t_min = q_timestamps[idx];
        idx += page_size;
        let t_max = q_timestamps[idx - 1];
        write_timestamp(&mut res, t_min);
        write_timestamp(&mut res, t_max);

        // for ease of seeking by date range, we write out the full compressed data page size (timestamps and data)
        t_compressor.data_page().map_err(map_err)?;
        for v_compressor in value_compressors.iter_mut() {
            v_compressor.data_page().map_err(map_err)?;
        }

        let data_size = t_compressor.byte_size()
            + size_of::<usize>()
            + value_compressors
                .iter_mut()
                .map(|mut c| c.byte_size() + size_of::<usize>())
                .sum::<usize>();

        write_usize(&mut res, data_size);

        // 3.
        write_usize(&mut res, t_compressor.byte_size());

        // 4.
        res.extend(t_compressor.drain_bytes());

        for v_compressor in value_compressors.iter_mut() {
            // 5.
            write_usize(&mut res, v_compressor.byte_size());

            // 6.
            res.extend(v_compressor.drain_bytes());
        }
    }

    Ok(res)
}

pub(crate) fn deserialize_series_between(
    mut compressed: &[u8],
    t0: i64,
    t1: i64,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut t_decompressor = Decompressor::<i64>::default();

    #[warn(unused_assignments)]
    let mut size = 0;

    // read timestamp metadata
    compressed = read_metadata(&mut t_decompressor, compressed)?;

    // read series count
    (compressed, size) = read_usize(compressed, "series count")?;

    let series_count = size;

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
    let tmp_ts = Arc::new(Vec::new());

    let mut res: Vec<Timeseries> = labels
        .into_iter()
        .map(|lbl| Timeseries {
            metric_name: lbl,
            timestamps: tmp_ts.clone(),
            values: Vec::new(),
        })
        .collect();

    while !compressed.is_empty() {
        (compressed, size) = read_usize(compressed, "data length")?;
        let n = size;

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
            (compressed, size) = read_usize(compressed, "timestamp data size")?;
            t_decompressor.write_all(&compressed[..size]).unwrap();
            let page_t = t_decompressor.data_page(n, size).map_err(map_err)?;

            compressed = &compressed[size..];

            for (v_decompressor, series) in v_decompressors.iter_mut().zip(res.iter_mut()) {
                (compressed, size) = read_usize(compressed, "value data size")?;
                v_decompressor.write_all(&compressed[..size]).unwrap();
                let page_v = v_decompressor.data_page(n, size).map_err(map_err)?;

                compressed = &compressed[size..];

                // todo: this can be faster
                let filtered = page_t
                    .iter()
                    .zip(page_v)
                    .filter(|(t, _)| (t0..t1).contains(t))
                    .collect::<Vec<_>>();

                timestamps.extend(filtered.iter().map(|(t, _)| *t));
                series.values.extend(filtered.into_iter().map(|(_, v)| v));
            }
        }
    }

    let timestamps = Arc::new(timestamps);
    for series in res.iter_mut() {
        series.timestamps = Arc::clone(&timestamps);
    }

    Ok(res)
}

fn write_metatadata<T: NumberLike>(compressor: &mut Compressor<T>,
                                   dest: &mut Vec<u8>,
                                   values: &[T],
                                   chunk_spec: &ChunkSpec) -> RuntimeResult<()> {
    // timestamp metadata
    compressor.header().map_err(map_err)?;
    compressor.chunk_metadata(values, chunk_spec).map_err(map_err)?;
    write_usize(dest, compressor.byte_size());
    dest.extend(compressor.drain_bytes());
    Ok(())
}

fn read_metadata<'a, T: NumberLike>(decompressor: &mut Decompressor<T>, compressed: &'a [u8]) -> RuntimeResult<&'a [u8]> {
    let (mut tail, size) = read_usize(compressed, "metatdata length")?;
    decompressor.write_all(&tail[..size]).unwrap();
    decompressor.header().map_err(map_err)?;
    decompressor.chunk_metadata().map_err(map_err)?;
    tail = &tail[size..];
    Ok(tail)
}

fn map_err(e: QCompressError) -> RuntimeError {
    RuntimeError::SerializationError(e.to_string())
}

fn map_marshal_err(e: Error, what: &str) -> RuntimeError {
    let msg = format!("error writing {}: {:?}", what, e);
    RuntimeError::SerializationError(msg)
}

fn map_unmarshal_err(e: Error, what: &str) -> RuntimeError {
    let msg = format!("error reading {}: {:?}", what, e);
    RuntimeError::SerializationError(msg)
}

fn read_timestamp(compressed: &[u8]) -> RuntimeResult<(&[u8], i64)> {
    let (val, tail) = unmarshal_var_i64(compressed).map_err(|e| map_marshal_err(e, "timestamp"))?;
    Ok((tail, val))
}

fn write_timestamp(dest: &mut Vec<u8>, ts: i64) {
    marshal_var_i64(dest, ts);
}



#[cfg(test)]
mod tests {
    use super::*;

    fn test_series(tss: &[Timeseries]) {
        let data = compress_series(tss).unwrap();
        let tss2 = deserialize_series_between(&data, 0, 100000).unwrap();

        assert_eq!(tss, tss2,
            "unexpected timeseries unmarshaled\ngot\n{:?}\nwant\n{:?}", tss2[0], tss[0])
    }


    // Single series
    #[test]
    fn single_series() {
        let mut series = Timeseries::default();
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