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
//! and return types of functions in DataFusion.
use std::borrow::Borrow;
use std::hash::Hash;
use std::str::FromStr;

use clone_dyn::clone_dyn;

use metricsql::functions::DataType;

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::Timeseries;

pub(crate) static MAX_ARG_COUNT: usize = 32;

pub enum ParameterValue {
    Matrix(Vec<Vec<Timeseries>>),
    Series(Vec<Timeseries>),
    Vector(Vec<f64>),
    Float(f64),
    Int(i64),
    String(String),
}

impl ParameterValue {
    pub fn is_numeric(&self) -> bool {
        match self {
            ParameterValue::Float(_) |
            ParameterValue::Int(_) => true,
            _ => false
        }
    }

    pub fn data_type(&self) -> DataType {
        match &self {
            ParameterValue::Matrix(_) => DataType::Matrix,
            ParameterValue::Float(_) => DataType::Float,
            ParameterValue::String(_) => DataType::String,
            ParameterValue::Int(_) => DataType::Int,
            ParameterValue::Vector(_) => DataType::Vector,
            ParameterValue::Series(_) => DataType::Series
        }
    }

    pub fn get_matrix(&self) -> &Vec<Vec<Timeseries>> {
        match self {
            ParameterValue::Matrix(val) => val,
            _ => panic!("BUG: range selection value expected ")
        }
    }

    pub fn get_float(&self) -> RuntimeResult<f64> {
        match self {
            ParameterValue::Series(series) => {
                let ts = get_single_timeseries(series)?;
                if ts.values.len() != 1 {
                    let msg = format!("expected a vector of size 1; got {}", ts.values.len());
                    return Err(RuntimeError::ArgumentError(msg))
                }
                Ok(ts.values[0])
            },
            ParameterValue::Vector(vec) => {
                if vec.len() != 1 {
                    let msg = format!("expected a vector of size 1; got {}", vec.len());
                    return Err(RuntimeError::ArgumentError(msg))
                }
                Ok(vec[0])
            },
            ParameterValue::Float(val) => Ok(*val),
            ParameterValue::Int(val) => Ok(*val as f64),
            ParameterValue::String(s) => {
                match f64::from_str(s) {
                    Err(e) => {
                        return Err(RuntimeError::TypeCastError(
                            format!("{} cannot be converted to a float", s)
                        ))
                    },
                    Ok(val) => Ok(val)
                }
            },
            _ => {
                return Err(RuntimeError::TypeCastError(
                    format!("{} cannot be converted to a float", self.data_type())
                ))
            }
        }
    }

    pub fn get_int(&self) -> RuntimeResult<i64> {
        match self {
            ParameterValue::Int(val) => Ok(*val),
            ParameterValue::Float(val) => Ok(*val as i64),
            _=> {
                match self.get_float() {
                    Ok(val) => Ok(val as i64),
                    Err(e) => Err(e)
                }
            }
        }
    }

    pub fn get_vector<'a>(&self) -> RuntimeResult<&'a Vec<f64>> {
        match self {
            ParameterValue::Vector(val) => Ok(val),
            _ => Err(RuntimeError::InvalidNumber("vector parameter expected ".to_string()))
        }
    }


    pub fn get_str(&self) -> RuntimeResult<&str> {
        let str = self.get_string()?;
        Ok(str.as_str())
    }

    pub fn get_series<'a>(&self) -> &'a Vec<Timeseries> {
        match self {
            ParameterValue::Series(val) => val.as_ref(),
            _ => panic!("BUG: invalid series parameter")
        }
    }

    pub fn get_series_mut<'a>(&self) -> &'a mut Vec<Timeseries> {
        match self {
            ParameterValue::Series(mut val) => val.as_mut(),
            _ => panic!("BUG: invalid series parameter")
        }
    }

    pub fn get_string(&self) -> RuntimeResult<String> {
        match self {
            ParameterValue::String(s) => {
                Ok(s.to_string())
            }
            ParameterValue::Series(series) => {
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
}

impl Default for ParameterValue {
    fn default() -> Self {
        ParameterValue::Int(0)
    }
}

impl FromStr for ParameterValue {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ParameterValue::String(s.to_string()))
    }
}

impl From<f64> for ParameterValue {
    fn from(v: f64) -> Self {
        ParameterValue::Float(v)
    }
}

impl From<i64> for ParameterValue {
    fn from(v: i64) -> Self {
        ParameterValue::Int(v)
    }
}


#[inline]
fn get_single_series_from_param(param: &ParameterValue) -> RuntimeResult<&Timeseries> {
    match param {
        ParameterValue::Series(series) => get_single_timeseries(series),
        _ => {
            let msg = format!("expected a timeseries vector; got {}", param.data_type());
            return Err(RuntimeError::TypeCastError(msg))
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
    fn into_vec(self) -> Vec<(K, Box<dyn FunctionImplementation<P, R>>)>;

    fn remove<Q: ?Sized>(&mut self, key: &Q)
        where
            K: Borrow<Q>,
            Q: Eq + Hash;

    fn insert(&mut self, key: K, item: Box<dyn FunctionImplementation<P, R>>);

    fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where
            K: Borrow<Q>,
            Q: Eq + Hash;

    fn get<Q: ?Sized>(&self, key: &Q) -> Option<&Box<dyn FunctionImplementation<P, R>>>
        where
            K: Borrow<Q>,
            Q: Eq + Hash;

    fn len(&self) -> usize;
}