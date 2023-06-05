use std::io::Write;
use std::sync::Arc;
use byte_slice_cast::AsByteSlice;
use q_compress::data_types::{NumberLike};
use q_compress::errors::QCompressResult;
use q_compress::wrapped::{ChunkSpec, Compressor, Decompressor};
use crate::{marshal_bytes_fast, MetricName, RuntimeResult, Tag, Timeseries, unmarshal_bytes_fast};


pub(crate) fn compress_series(series: &[Timeseries]) -> QCompressResult<Vec<u8>> {
    const DATA_PAGE_SIZE: usize = 3000;

    let series_count = series.len();
    let q_timestamps = &series[0].timestamps;
    let mut count = series.iter().fold(0, |acc, s| acc + s.len());
    let mut values = Vec::with_capacity(count);

    // flatten values. Dont know if i like this
    for i in 0 .. q_timestamps.len() {
        for s in series {
            values.push(s.values[i]); // can we eliminate this bound check?
        }
    }

    let mut data_page_sizes = Vec::with_capacity(count / DATA_PAGE_SIZE + 1);
    // for ease of implementation, round DATA_PAGE_SIZE up to the nearest multiple of series_count
    let data_page_size =  DATA_PAGE_SIZE + (series_count - (DATA_PAGE_SIZE % series_count));

    while count > 0 {
        let page_size = count.min(data_page_size);
        data_page_sizes.push(page_size);
        count -= page_size;
    }

    let chunk_spec = ChunkSpec::default().with_page_sizes(data_page_sizes.clone());

    let mut t_compressor =
        Compressor::<i64>::from_config(q_compress::auto_compressor_config(
            &q_timestamps,
            q_compress::DEFAULT_COMPRESSION_LEVEL,
        ));

    let mut v_compressor = Compressor::<f64>::from_config(q_compress::auto_compressor_config(
        &values,
        q_compress::DEFAULT_COMPRESSION_LEVEL,
    ));

    let mut res = Vec::with_capacity(512);

    // write out series count
    write_usize(&mut res, series_count);

    // timestamp metadata
    t_compressor.header()?;
    t_compressor.chunk_metadata(&q_timestamps, &chunk_spec)?;
    write_usize(&mut res, t_compressor.byte_size());
    res.extend(t_compressor.drain_bytes());

    // write out value chunk metadata
    v_compressor.header()?;
    v_compressor.chunk_metadata(&values, &chunk_spec)?;
    write_usize(&mut res, v_compressor.byte_size());
    res.extend(v_compressor.drain_bytes());

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
        let start = idx % series_count;
        let t_min = q_timestamps[start];
        idx += page_size;
        let end = idx % series_count;
        let t_max = q_timestamps[end - 1];
        res.extend(t_min.to_bytes());
        res.extend(t_max.to_bytes());

        // 3.
        t_compressor.data_page()?;
        write_usize(&mut res, t_compressor.byte_size());

        // 4.
        res.extend(t_compressor.drain_bytes());

        // 5.
        v_compressor.data_page()?;
        write_usize(&mut res, v_compressor.byte_size());

        // 6.
        res.extend(v_compressor.drain_bytes());
    }

    Ok(res)
}


pub(crate) fn decompress_time_series_between(
    mut compressed: &[u8],
    t0: i64,
    t1: i64,
) -> QCompressResult<Vec<Timeseries>> {
    let mut series = Timeseries::default();
    let vs = &mut series.values;

    let mut t_decompressor = Decompressor::<i64>::default();
    let mut v_decompressor = Decompressor::<f64>::default();

    #[warn(unused_assignments)]
    let mut size = 0;

    (compressed, size) = read_usize(compressed);

    let series_count = size;

    (compressed, size) = read_usize(compressed);
    t_decompressor.write_all(&compressed[..size]).unwrap();
    t_decompressor.header()?;
    t_decompressor.chunk_metadata()?;
    compressed = &compressed[size..];

    (compressed, size) = read_usize(compressed);
    v_decompressor.write_all(&compressed[..size]).unwrap();
    v_decompressor.header()?;
    v_decompressor.chunk_metadata()?;
    compressed = &compressed[size..];

    // todo: init capacity
    let mut timestamps = Vec::new();
    let tmp_ts = Arc::new(Vec::new());

    let mut res: Vec<Timeseries> = (0..series_count).map(|_| {
        Timeseries {
            metric_name: Default::default(),
            timestamps: tmp_ts.clone(),
            values: Vec::new(),
        }
    }).collect();

    while !compressed.is_empty() {
        (compressed, size) = read_usize(compressed);
        let n = size;

        let mut ts: i64 = 0;

        (compressed, ts) = read_timestamp(compressed)?;

        if ts > t1 {
            break;
        }

        (compressed, ts) = read_timestamp(compressed)?;

        if ts < t0 {
            // we can skip this data
            (compressed, size) = read_usize(compressed);
            compressed = &compressed[size..];
            (compressed, size) = read_usize(compressed);
            compressed = &compressed[size..];
        } else {
            // we need to filter and append this data
            (compressed, size) = read_usize(compressed);
            t_decompressor.write_all(&compressed[..size]).unwrap();
            let page_t = t_decompressor.data_page(n, size)?;
            compressed = &compressed[size..];

            (compressed, size) = read_usize(compressed);
            v_decompressor.write_all(&compressed[..size]).unwrap();
            let page_v = v_decompressor.data_page(n, size)?;
            compressed = &compressed[size..];

            let filtered = page_t
                .into_iter()
                .zip(page_v)
                .filter(|(t, _)| (t0..t1).contains(t))
                .collect::<Vec<_>>();

            timestamps.extend(filtered.iter().map(|(t, _)| *t));
            // filtered.into_iter().map(|(_, v)| v).for_each(|v| {
            //     values.push(*v)
            // });
            vs.extend(filtered.into_iter().map(|(_, v)| v));
        }
    }

    Ok(res)
}

fn write_usize(slice: &mut [u8], size: usize) {
    slice[0..4].copy_from_slice(&(size as u32).to_be_bytes());
}

fn read_usize(slice: &[u8]) -> (&[u8], usize) {
    let byte_size = u32::from_be_bytes(slice[0..4].try_into().unwrap());
    (&slice[4..], byte_size as usize)
}

fn read_timestamp(compressed: &[u8]) -> QCompressResult<(&[u8], i64)> {
    let ts = i64::from_bytes(compressed[..8].as_byte_slice())?;
    Ok((&compressed[8..], ts))
}

fn write_metric_name(mn: &MetricName, dst: &mut Vec<u8>) {
    write_usize(dst, mn.tags.len());
    marshal_bytes_fast(dst, mn.metric_group.as_bytes());
    for Tag { key: k, value: v } in mn.tags.iter() {
        marshal_bytes_fast(dst, k.as_bytes());
        marshal_bytes_fast(dst, v.as_bytes());
    }
}

// todo: use a faster marshaller (maybe serde?)
fn read_metric_name(src: &[u8]) -> RuntimeResult<(&[u8], MetricName)> {
    let (src, tag_count) = read_usize(src);
    let (mut src, metric_group) = unmarshal_bytes_fast(src)?;
    let mut tags = Vec::with_capacity(tag_count);

    let mut key: &[u8] = &[];
    let mut value: &[u8] = &[];

    for _ in 0..tag_count {
        (src, key) = unmarshal_bytes_fast(src)?;
        (src, value) = unmarshal_bytes_fast(src)?;
        tags.push(Tag {
            key: String::from_utf8_lossy(key).into_owned(),
            value: String::from_utf8_lossy(value).into_owned(),
        });
    }
    let mut mn = MetricName::default();
    mn.metric_group = String::from_utf8_lossy(metric_group).to_string();
    mn.tags = tags;

    Ok((src, mn))
}

