use rand::prelude::StdRng;
use rand::Rng;
use rand::{distributions::Distribution, thread_rng, SeedableRng};
use rand_distr::{Exp1, StandardNormal};

use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult, Timeseries};

fn create_rng(tfa: &mut TransformFuncArg) -> RuntimeResult<StdRng> {
    if tfa.args.len() == 1 {
        return match tfa.args[0].get_int() {
            Err(e) => Err(e),
            Ok(val) => match u64::try_from(val) {
                Err(_) => Err(RuntimeError::ArgumentError(
                    format!("invalid rand seed {}", val).to_string(),
                )),
                Ok(seed) => Ok(StdRng::seed_from_u64(seed)),
            },
        };
    }
    match StdRng::from_rng(thread_rng()) {
        Err(e) => Err(RuntimeError::ArgumentError(
            format!("Error constructing rng {:?}", e).to_string(),
        )),
        Ok(rng) => Ok(rng),
    }
}

fn rand_fn_inner<F>(tfa: &mut TransformFuncArg, f: F) -> RuntimeResult<Vec<Timeseries>>
where
    F: Fn(&mut StdRng) -> f64,
{
    let mut rng: StdRng = create_rng(tfa)?;
    let mut tss = eval_number(tfa.ec, 0.0)?;
    for value in tss[0].values.iter_mut() {
        *value = f(&mut rng);
    }
    Ok(tss)
}

macro_rules! create_rand_func {
    ($name: ident, $f:expr) => {
        pub(crate) fn $name(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            rand_fn_inner(tfa, $f)
        }
    };
}

create_rand_func!(rand, |r: &mut StdRng| r.gen::<f64>());

create_rand_func!(rand_norm, |r: &mut StdRng| {
    <StandardNormal as Distribution<f64>>::sample::<StdRng>(&StandardNormal, r)
});

create_rand_func!(rand_exp, |r: &mut StdRng| {
    <Exp1 as Distribution<f64>>::sample::<StdRng>(&Exp1, r)
});
