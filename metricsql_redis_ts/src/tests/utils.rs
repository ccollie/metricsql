use std::env;
use std::ops::Range;

use rustis::commands::{FlushingMode, ServerCommands, TsAddOptions, TsCreateOptions, TsEncoding};
use rustis::{client::Client, commands::TimeSeriesCommands};

use metricsql_engine::prelude::generators::RandAlgo::MackeyGlass;
use metricsql_engine::prelude::generators::{generate_series_data, GeneratorOptions};
use metricsql_engine::prelude::SeriesData;
use metricsql_engine::{MetricName, Tag, Timestamp, METRIC_NAME_LABEL};

const BATCH_SIZE: usize = 1000;

pub async fn create_series(client: &Client, mn: &MetricName) {
    let mut options = TsCreateOptions::default().encoding(TsEncoding::Compressed);

    let mut labels: Vec<(String, String)> = Vec::with_capacity(mn.tags.len());
    labels.push((METRIC_NAME_LABEL.to_string(), mn.metric_group.clone()));
    for Tag { key, value } in mn.tags.iter() {
        labels.push((key.clone(), value.clone()));
    }
    options = options.labels(labels);
    client
        .ts_create(metric_name_to_key(mn), options)
        .await
        .unwrap();
}

pub async fn write_data(client: &Client, key: &str, data: &SeriesData) {
    let pipeline = client.create_pipeline();
    let options: TsAddOptions = Default::default();

    for (i, sample) in data.iter().enumerate() {
        let _ = pipeline.ts_add(key, sample.timestamp, sample.value, options.clone());
        if i % BATCH_SIZE == 0 {
            let _ = pipeline.execute().await.unwrap();
        }
    }
    let _ = pipeline.execute().await.unwrap();
}

pub fn generate_data(start: Timestamp, end: Timestamp, range: Range<f64>) -> SeriesData {
    let options = GeneratorOptions {
        typ: MackeyGlass,
        start,
        end: Some(end),
        range,
        ..Default::default()
    };
    generate_series_data(&options).expect("Failed to generate data")
}

pub fn metric_name_to_key(mn: &MetricName) -> String {
    let mut res = vec![mn.metric_group.clone()];
    // tags should be sorted
    for Tag { key, .. } in mn.tags.iter() {
        res.push(key.clone());
    }
    res.join(":")
}

pub async fn flush_all(client: &Client) {
    let _ = client.flushall(FlushingMode::Sync).await;
}

pub async fn get_client() -> Client {
    let redis_url = get_redis_url();
    let client = Client::connect(&redis_url).unwrap();
    client
}

pub fn get_redis_url() -> String {
    let redis_host_key = "REDIS_HOST";
    let redis_host_port = "REDIS_PORT";

    let redis_host = match env::var(redis_host_key) {
        Ok(host) => host,
        _ => "localhost".to_string(),
    };

    let redis_port = match env::var(redis_host_port) {
        Ok(port) => port,
        _ => "6379".to_string(),
    };

    format!("redis://{}:{}/", redis_host, redis_port)
}
