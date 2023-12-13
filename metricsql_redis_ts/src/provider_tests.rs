#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rustis::client::Client;

    use metricsql_engine::execution::{exec, Context, EvalConfig};
    use metricsql_engine::{test_results_equal, MetricName, QueryResult, Tag, Timestamp};

    use crate::tests::{
        create_series, generate_data, get_redis_url, metric_name_to_key, write_data,
    };
    use crate::RedisMetricsQLProvider;

    const TEST_ITERATIONS: usize = 5;

    const REGIONS: [&str; 5] = [
        "us-east-1",
        "us-west-1",
        "us-west-2",
        "eu-west-1",
        "us-gov-west-1",
    ];

    async fn create_context() -> Context {
        let server_url = get_redis_url();
        let provider = RedisMetricsQLProvider::new(&server_url).await.unwrap();
        let mut context = Context::default().with_metric_storage(Arc::new(provider));
        context
    }

    async fn populate_data(client: &Client, start: Timestamp, end: Timestamp) {
        let metrics = REGIONS.iter().map(|r| {
            let mut mn = MetricName::new("requests_per_sec");
            mn.tags.push(Tag {
                key: "region".to_string(),
                value: r.to_string(),
            });
            mn
        });
        for mn in metrics.iter() {
            let key = metric_name_to_key(mn);
            let data = generate_data(start, end, 15_000.0..100_000.0);
            create_series(client, mn).await;
            write_data(client, &key, &data).await;
        }
    }

    async fn test_query(
        q: &str,
        start: Timestamp,
        end: Timestamp,
        step: i64,
        expected: Vec<QueryResult>,
    ) {
        let mut ec = EvalConfig::new(start, end, step);
        let context = create_context().await;

        for _ in 0..TEST_ITERATIONS {
            match exec(&context, &mut ec, q, false) {
                Ok(result) => test_results_equal(&result, &expected),
                Err(e) => {
                    panic!("{}", e)
                }
            }
        }
    }
}
