// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::borrow::Borrow;
use std::hash::Hash;

use clone_dyn::clone_dyn;

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{QueryValue, Timeseries};

#[inline]
pub(crate) fn get_single_timeseries(series: &Vec<Timeseries>) -> RuntimeResult<&Timeseries> {
    if series.len() != 1 {
        let msg = format!(
            "arg must contain a single timeseries; got {} timeseries",
            series.len()
        );
        return Err(RuntimeError::TypeCastError(msg));
    }
    Ok(&series[0])
}

pub fn get_scalar_param_value(
    param: &QueryValue,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<f64> {
    match param {
        QueryValue::Scalar(val) => Ok(*val),
        _ => {
            let msg = format!(
                "expected scalar arg for parameter \"{}\" of function {}; Got {}",
                param_name,
                func_name,
                param.data_type_name()
            );
            return Err(RuntimeError::TypeCastError(msg));
        }
    }
}

#[inline]
pub fn get_string_param_value(
    param: &QueryValue,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<String> {
    match param {
        QueryValue::String(val) => Ok(val.clone()),
        _ => {
            let msg = format!(
                "expected string arg for parameter \"{}\" of function {}; Got {}",
                param_name,
                func_name,
                param.data_type_name()
            );
            return Err(RuntimeError::TypeCastError(msg));
        }
    }
}

#[clone_dyn]
pub trait FunctionImplementation<P: ?Sized, R>: Fn(&mut P) -> R
where
    P: Send + Sync,
{
}

impl<T, P: ?Sized, R> FunctionImplementation<P, R> for T
where
    T: Fn(&mut P) -> R + 'static,
    P: Send + Sync,
{
}

pub trait FunctionRegistry<K, P: ?Sized, R>
where
    K: Eq + Hash,
{
    fn into_vec(self) -> Vec<(K, Box<dyn FunctionImplementation<P, R, Output = R>>)>;

    fn remove<Q: ?Sized>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Eq + Hash;

    fn insert(&mut self, key: K, item: Box<dyn FunctionImplementation<P, R, Output = R>>);

    fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + Hash;

    fn get<Q: ?Sized>(&self, key: &Q) -> Option<&Box<dyn FunctionImplementation<P, R, Output = R>>>
    where
        K: Borrow<Q>,
        Q: Eq + Hash;

    fn len(&self) -> usize;
}
