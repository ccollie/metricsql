# metricsql

`metricsql` implements a [MetricsQL](https://github.com/VictoriaMetrics/VictoriaMetrics/wiki/MetricsQL)
and [PromQL](https://medium.com/@valyala/promql-tutorial-for-beginners-9ab455142085) parser in Rust.

### Usage

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = r#"
        sum(rate(node_cpu_seconds_total{mode="user"}[5m])) by (mode) 
        or 
        sum(rate(node_cpu_seconds_total{mode="system"}[5m])) by (mode)
    "#;
    let expr = metricsql_parser::parse(query)?;
    // Now expr contains parsed MetricsQL as `Expr` enum variants.
    // See parse examples for more details.
}
```

See [docs](https://godoc.org/github.com/VictoriaMetrics/metricsql) for more details.