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

//! Signature module contains foundational types that are used to represent signatures, types,
//! and return types of functions in metrix.
use std::borrow::{Borrow, Cow};
use std::hash::{Hash};
use std::str::FromStr;

use clone_dyn::clone_dyn;

use metricsql::functions::DataType;
use crate::eval::{eval_number};

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, Timeseries};

pub enum AnyValue {
    RangeVector(Vec<Timeseries>),
    InstantVector(Vec<Timeseries>),
    Scalar(f64),
    String(String),
}

impl AnyValue {

    pub fn nan() -> Self {
        AnyValue::Scalar(f64::NAN)
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            AnyValue::Scalar(_) => true,
            _ => false
        }
    }

    pub fn data_type(&self) -> DataType {
        match &self {
            AnyValue::RangeVector(_) => DataType::RangeVector,
            AnyValue::Scalar(_) => DataType::Scalar,
            AnyValue::String(_) => DataType::String,
            AnyValue::InstantVector(_) => DataType::InstantVector
        }
    }

    pub fn data_type_name(&self) -> &'static str {
        match &self {
            AnyValue::RangeVector(_) => "RangeVector",
            AnyValue::Scalar(_) => "Scalar",
            AnyValue::String(_) => "String",
            AnyValue::InstantVector(_) => "InstantVector"
        }
    }

    pub fn get_matrix(&self) -> &Vec<Timeseries> {
        match self {
            AnyValue::RangeVector(val) => val,
            _ => panic!("BUG: range selection value expected ")
        }
    }

    pub fn get_scalar(&self) -> RuntimeResult<f64> {
        match self {
            AnyValue::Scalar(val) => Ok(*val),
            AnyValue::InstantVector(series) => {
                let ts = get_single_timeseries(series)?;
                if ts.values.len() != 1 {
                    let msg = format!("expected a vector of size 1; got {}", ts.values.len());
                    return Err(RuntimeError::ArgumentError(msg))
                }
                Ok(ts.values[0])
            },
            _ => {
                return Err(RuntimeError::TypeCastError(
                    format!("{} cannot be converted to a scalar", self.data_type())
                ))
            }
        }
    }

    pub fn get_int(&self) -> RuntimeResult<i64> {
        match self {
            AnyValue::Scalar(val) => Ok(*val as i64),
            _=> {
                match self.get_scalar() {
                    Ok(val) => Ok(val as i64),
                    Err(e) => Err(e)
                }
            }
        }
    }

    pub fn get_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            AnyValue::InstantVector(val) => {
                // TODO: into
                Ok(val.clone())
            }, // ????
            AnyValue::RangeVector(val) => {
                // TODO: into
                Ok(val.clone())
            }, // ????
            _ => Err(RuntimeError::InvalidNumber("vector parameter expected ".to_string()))
        }
    }


    // todo: get_series_into()
    pub fn get_instant_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            AnyValue::InstantVector(val) => Ok(val.clone()), // ????
            _ => panic!("BUG: invalid series parameter")
        }
    }

    pub fn get_range_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            AnyValue::RangeVector(val) => Ok(val.clone()), // ????
            _ => panic!("BUG: invalid series parameter")
        }
    }

    pub fn into_instant_vector(self, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            AnyValue::InstantVector(mut val) => Ok(std::mem::take(&mut val)), // use .into instead ????
            AnyValue::Scalar(n) => Ok(eval_number(ec, n)),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                return Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn get_string(&self) -> RuntimeResult<String> {
        match self {
            AnyValue::String(s) => {
                Ok(s.to_string())
            }
            AnyValue::InstantVector(series) => {
                let ts = get_single_timeseries(series)?;
                if ts.values.len() > 0 {
                    let all_nan = series[0].values.iter().all(|x| x.is_nan());
                    if !all_nan {
                        let msg = format!("series contains non-string timeseries");
                        return Err(RuntimeError::ArgumentError(msg));
                    }
                }
                let res = ts.metric_name.metric_group.clone();
                // todo: return reference
                Ok(res)
            }
            _ => {
                let msg = format!("cannot cast {} to a string", self.data_type());
                return Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn as_instant_vec(&self, ec: &EvalConfig) -> RuntimeResult<Cow<Vec<Timeseries>>> {
        match self {
            AnyValue::InstantVector(v) => Ok(Cow::Borrowed(v)),
            AnyValue::Scalar(n) => {
                Ok(
                    Cow::Owned(eval_number(ec, *n))
                )
            },
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                return Err(RuntimeError::TypeCastError(msg))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            AnyValue::InstantVector(val) => val.is_empty(), // ????
            AnyValue::RangeVector(val) => val.is_empty(),
            AnyValue::String(v) => v.is_empty(),
            _ => false
        }
    }

    pub fn empty_vec() -> Self {
        AnyValue::InstantVector(vec![])
    }
}

impl Default for AnyValue {
    fn default() -> Self {
        AnyValue::Scalar(0_f64)
    }
}

impl FromStr for AnyValue {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AnyValue::String(s.to_string()))
    }
}

impl From<f64> for AnyValue {
    fn from(v: f64) -> Self {
        AnyValue::Scalar(v)
    }
}

impl From<i64> for AnyValue {
    fn from(v: i64) -> Self {
        AnyValue::Scalar(v as f64)
    }
}

impl From<Vec<Timeseries>> for AnyValue {
    fn from(vec: Vec<Timeseries>) -> Self {
        AnyValue::InstantVector(vec)
    }
}

impl Clone for AnyValue {
    fn clone(&self) -> Self {
        match self {
            AnyValue::RangeVector(m) => {
                AnyValue::RangeVector(m.clone())
            }
            AnyValue::InstantVector(series) => {
                AnyValue::InstantVector(series.clone())
            }
            AnyValue::Scalar(f) => AnyValue::Scalar(*f),
            AnyValue::String(s) => { AnyValue::String(s.clone()) }
        }
    }
}

#[inline]
fn get_single_timeseries(series: &Vec<Timeseries>) -> RuntimeResult<&Timeseries> {
    if series.len() != 1 {
        let msg = format!(
            "arg must contain a single timeseries; got {} timeseries",
            series.len()
        );
        return Err(RuntimeError::TypeCastError(msg))
    }
    Ok(&series[0])
}

pub fn get_scalar_param_value(param: &AnyValue, func_name: &str, param_name: &str) -> RuntimeResult<f64> {
    match param {
        AnyValue::Scalar(val) => Ok(*val),
        _ => {
            let msg = format!(
                "expected scalar arg for parameter \"{}\" of function {}; Got {}",
                param_name, func_name, param.data_type_name()
            );
            return Err(RuntimeError::TypeCastError(msg))
        }
    }
}

#[inline]
pub fn get_string_param_value(param: &AnyValue, func_name: &str, param_name: &str) -> RuntimeResult<String> {
    match param {
        AnyValue::String(val) => Ok(val.clone()),
        _ => {
            let msg = format!(
                "expected string arg for parameter \"{}\" of function {}; Got {}",
                param_name, func_name, param.data_type_name()
            );
            return Err(RuntimeError::TypeCastError(msg))
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
    fn into_vec(self) -> Vec<(K, Box<dyn FunctionImplementation<P, R, Output=R>>)>;

    fn remove<Q: ?Sized>(&mut self, key: &Q)
        where
            K: Borrow<Q>,
            Q: Eq + Hash;

    fn insert(&mut self, key: K, item: Box<dyn FunctionImplementation<P, R, Output=R>>);

    fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where
            K: Borrow<Q>,
            Q: Eq + Hash;

    fn get<Q: ?Sized>(&self, key: &Q) -> Option<&Box<dyn FunctionImplementation<P, R, Output=R>>>
        where
            K: Borrow<Q>,
            Q: Eq + Hash;

    fn len(&self) -> usize;
}