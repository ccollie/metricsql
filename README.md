# metrix

`metrix` implements a [MetricsQL](https://github.com/VictoriaMetrics/VictoriaMetrics/wiki/MetricsQL)
and [PromQL](https://medium.com/@valyala/promql-tutorial-for-beginners-9ab455142085) execution engine in Rust.
Define your datasource as a trait and execute timeseries queries against your data.

### Usage

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define your datasource
    let provider = Arc::new(MyDataProvider::new());
    // Create a Context, which will be used to provides session level config and services like caching and query stats
    let ctx = Context::default().with_provider(provider);

    let mut builder = QueryBuilder::default()
        .start(Utc::now() - Duration::minutes(5))
        .end(Utc::now())
        .step(Duration::minutes(1))
        .enable_tracing()
        .query(r#"sum(rate(foo{bar="baz"}[5m])) by (job)"#);

    let query_params = builder.build(&context)?;

    // run arbitrary query against your datasource
    let result = runtime::query_range(&context, query_params)?;
}
```

### Under Heavy Development !!!!

This code is under heavy development and is not ready for production ore even casual use. The code base is
also in heavy flux and will change frequently.

### Features

- Handles PromQL as well as a superset (MetricsQL). Note, however that 100% PromQL compatibility is not a goal.
- Over 200 supported functions (Aggregation, Rollup and Transformation)
- Builtin support for query rollup caching
- Builtin support for query tracing
- Uses [Rayon](https://docs.rs/rayon/latest/rayon/) for query execution parallelization

### Roadmap

- [x] Implement query parsing
- [x] Implement basic query execution
- [x] Implement query functions
- [ ] Test coverage
- [ ] [Datafusion](https://arrow.apache.org/datafusion/) based provider. Expected support for `postgres`, `mysql`,
  and `sqlite` as well as file based sources like `csv`, `json` and `parquet`

### Contributing

Contributions are welcome. Please open an issue to discuss your ideas before submitting a PR.

### Acknowledgements

This project started as heavily modded `rust` port of parts
of [VictoriaMetrics](https://github.com/VictoriaMetrics/VictoriaMetrics).
The original code is licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

Inspiration was also taken from

- [Prometheus](https://github.com/prometheus/prometheus)
- [GrepTime DB](https://github.com/GreptimeTeam/greptimedb)
- [InfluxDB](https://github.com/influxdata/influxdb_iox/tree/main)
- [OpenObserve](https://github.com/openobserve/openobserve)

### License

This project is licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).