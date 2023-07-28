# metricsql

`metricsql` implements a [MetricsQL](https://github.com/VictoriaMetrics/VictoriaMetrics/wiki/MetricsQL)
and [PromQL](https://medium.com/@valyala/promql-tutorial-for-beginners-9ab455142085) parser in Rust.

### Usage

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expr = metricsql::parse(r#"sum(rate(foo{bar="baz"}[5m])) by (job)"#)?;
    // Now expr contains parsed MetricsQL as `Expr` enum variants.
    // See parse examples for more details.
}
```

See [docs](https://godoc.org/github.com/VictoriaMetrics/metricsql) for more details.