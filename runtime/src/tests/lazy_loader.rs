use std::time::Duration;

use crate::tests::load_cmd::LoadCmd;
use crate::tests::test_storage::TestStorage;
use crate::tests::types::CancelFunc;
use crate::{RuntimeError, RuntimeResult};

/// LazyLoader lazily loads samples into storage.
/// This is specifically implemented for unit testing of rules.
pub(crate) struct LazyLoader {
    load_cmd: LoadCmd,
    storage: TestStorage,
    subquery_interval: Duration,
    query_engine: Engine,
    context: Context,
    cancel_ctx: CancelFunc,
    opts: LazyLoaderOpts,
}

/// LazyLoaderOpts are options for the lazy loader.
pub struct LazyLoaderOpts {
    /// Both of these must be set to true for regular PromQL (as of
    /// Prometheus v2.33). They can still be disabled here for legacy and
    /// other uses.
    enable_at_modifier: bool,
    enable_negative_offset: bool,
}

impl LazyLoader {
    /// new returns an initialized empty LazyLoader.
    fn new(input: &str, opts: LazyLoaderOpts) -> RuntimeResult<LazyLoader> {
        let ll = LazyLoader {
            load_cmd: LoadCmd {
                gap: Default::default(),
                metrics: Default::default(),
                defs: Default::default(),
            },
            storage: (),
            subquery_interval: Default::default(),
            query_engine: (),
            context: (),
            cancel_ctx: (),
            opts,
        };

        ll.parse(input)?;
        ll.clear();

        Ok(ll)
    }

    /// parse the given load command.
    fn parse(&mut self, input: &str) -> RuntimeResult<()> {
        let lines = get_lines(input);
        /// Accepts only 'load' command.
        for line in lines.iter() {
            if line.is_empty() {
                continue;
            }
            if strings.ToLower(patSpace.Split(l, 2)[0]) == "load" {
                self.load_cmd = parse_load(lines, i)?;
                Ok(())
            }
            return raise(i, format!("invalid command {}", l));
        }
        return RuntimeError::from("no \"load\" command found");
    }

    // clear the current test storage of all inserted samples.
    pub fn clear(&mut self) -> RuntimeResult<()> {
        match self.storage.close() {
            Ok(_) => {}
            Err(e) => {
                return Err(RuntimeError::from(
                    "Unexpected error while closing test storage.",
                ))
            }
        }
        if let Some(cancel_func) = self.cancel_ctx {
            (cancel_func)();
        }
        self.storage = TestStorage::new(ll);

        let opts = EngineOpts {
            max_samples: 10000,
            timeout: 100 * time.Second,
            NoStepSubqueryIntervalFn: Duration::from_millis(self.SubqueryInterval),
        };

        self.queryEngine =
            NewEngine(opts)(self.context, self.cancelCtx) = context.WithCancel(context.Background())
    }

    /// append_till appends the defined time series to the storage till the given timestamp
    /// (in milliseconds).
    fn append_till(&mut self, ts: i64) -> RuntimeResult<()> {
        let app = self.storage.get_appender(self.context);
        let to_remove: Vec<u64> = vec![];

        for (h, samples) in self.load_cmd.defs.iter() {
            let m = self.load_cmd.metrics.get(h);
            for (i, s) in samples.iter() {
                if s.t > ts {
                    // Removing the already added samples.
                    self.load_cmd.defs[h] = &samples[i..];
                    break;
                }
                app.append(0, m, s.t, s.v)?;
                if i == samples.len() - 1 {
                    self.load_cmd.defs[h] = nil
                }
            }
        }
        return app.commit();
    }

    // Close closes resources associated with the LazyLoader.
    pub fn close(&mut self) -> RuntimeResult<()> {
        self.cancelCtx();
        self.storage.close();
        require.NoError(ll.T, err, "Unexpected error while closing test storage.")
    }

    // with_samples_till loads the samples till given timestamp and executes the given function.
    fn with_samples_till(&mut self, ts: timestamp, func: fn(RuntimeError)) {
        let ts_milli = ts.Sub(time.Unix(0, 0).UTC()) / time.Millisecond;
        func(self.append_till(ts_milli));
    }

    /// Queryable allows querying the LazyLoader's data.
    /// Note: only the samples till the max timestamp used
    /// in `with_samples_till` can be queried.
    pub fn queryable(&self) -> &Queryable {
        &self.storage
    }
}
