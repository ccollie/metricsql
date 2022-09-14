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
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::Timeseries;
use clone_dyn::clone_dyn;

pub(crate) static MAX_ARG_COUNT: usize = 32;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum DataType {
    Matrix,
    Series,
    /// Vec<f64> (normally Timeseries::values)
    Vector,
    /// A 64-bit floating point number.
    Float,
    /// A 64-bit int.
    Int,
    /// An owned String
    String
}

impl DataType {
    /// Returns true if this type is numeric: (Int or Float).
    pub fn is_numeric(&self) -> bool {
        *self == DataType::Float || *self == DataType::Int
    }
}

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

    pub fn get_int(&self) -> i64 {
        match self {
            ParameterValue::Int(val) => *val,
            _ => panic!("BUG: int parameter expected ")
        }
    }

    pub fn get_float(&self) -> f64 {
        match self {
            ParameterValue::Float(val) => *val,
            _ => panic!("BUG: float parameter expected ")
        }
    }

    pub fn get_vector<'a>(&self) -> &'a Vec<f64> {
        match self {
            ParameterValue::Vector(val) => val,
            _ => panic!("BUG: vector parameter expected ")
        }
    }

    pub fn get_str(&self) -> &str {
        match self {
            ParameterValue::String(val) => val.as_str(),
            _ => panic!("BUG: invalid string parameter")
        }
    }

    pub fn get_string(&self) -> String {
        match self {
            ParameterValue::String(val) => val.to_string(),
            ParameterValue::Int(int) => int.to_string(),
            ParameterValue::Float(f) => f.to_string(),
            _ => panic!("BUG: invalid string parameter")
        }
    }

    pub fn get_series<'a>(&self) -> &'a Vec<Timeseries> {
        match self {
            ParameterValue::Series(val) => val.as_ref(),
            _ => panic!("BUG: invalid series parameter")
        }
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

///A function's volatility, which defines the functions eligibility for certain optimizations
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Volatility {
    /// Immutable - An immutable function will always return the same output when given the same
    /// input. An example of this is [super::BuiltinScalarFunction::Cos].
    Immutable,
    /// Stable - A stable function may return different values given the same input across different
    /// queries but must return the same value for a given input within a query. An example of
    /// this is [super::BuiltinScalarFunction::Now].
    Stable,
    /// Volatile - A volatile function may change the return value from evaluation to evaluation.
    /// Multiple invocations of a volatile function may return different results when used in the
    /// same query. An example of this is [super::BuiltinScalarFunction::Random].
    Volatile,
}

/// A function's type signature, which defines the function's supported argument types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeSignature {
    /// arbitrary number of arguments of an common type out of a list of valid types
    /// second element is the min number of arguments
    /// A function such as `concat` is `Variadic(vec![DataType::String, DataType::Float], 1)`
    Variadic(Vec<DataType>, usize),
    /// arbitrary number of arguments of an arbitrary but equal type, with possible minimum
    // A function such as `array` is `VariadicEqual`
    VariadicEqual(DataType, usize),
    /// fixed number of arguments of an arbitrary type out of a list of valid types
    // A function of one argument of f64 is `Uniform(1, Float)`
    Uniform(usize, DataType),
    /// exact number of arguments of an exact type
    Exact(Vec<DataType>),
    /// fixed number of arguments of arbitrary types
    Any(usize)
}

impl TypeSignature {
    /// Validate argument counts matches the `signature`.
    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> RuntimeResult<()> {

        fn expect_arg_count(name: &str, arg_len: usize, expected: usize) -> RuntimeResult<()> {
            if arg_len != expected {
                return Err(RuntimeError::ArgumentError(format!(
                    "The function {} expected {} arguments but received {}",
                    name,
                    expected,
                    arg_len
                )));
            }
            Ok(())
        }

        fn expect_min_args(name: &str, args_len: usize, min: usize) -> RuntimeResult<()> {
            if args_len < min {
                return Err(RuntimeError::ArgumentError(format!(
                    "The function {} expected a minimum of {} arguments but received {}",
                    name,
                    min,
                    args_len
                )));
            }
            Ok(())
        }

        return match self {
            TypeSignature::VariadicEqual(data_type_, min) => {
                expect_min_args(name,arg_len, *min)
            },
            TypeSignature::Variadic(valid_types, min) => {
                if valid_types.len() < *min || arg_len > valid_types.len() {
                    return Err(RuntimeError::ArgumentError(format!(
                        "The function {} expected between {} and {} argument, but received {}",
                        name,
                        min,
                        valid_types.len(),
                        arg_len
                    )));
                }
                Ok(())
            },
            TypeSignature::Uniform(number, valid_type_) => {
                expect_arg_count(name, arg_len, *number)
            },
            TypeSignature::Exact(valid_types) => {
                expect_arg_count(name, arg_len, valid_types.len())
            },
            TypeSignature::Any(number) => {
                expect_arg_count(name, arg_len, *number)
            }
        }
    }
}


///The Signature of a function defines its supported input types as well as its volatility.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Signature {
    /// type_signature - The types that the function accepts. See [TypeSignature] for more information.
    pub type_signature: TypeSignature,
    /// volatility - The volatility of the function. See [Volatility] for more information.
    pub volatility: Volatility,
}

impl Signature {
    /// new - Creates a new Signature from any type signature and the volatility.
    pub fn new(type_signature: TypeSignature, volatility: Volatility) -> Self {
        Signature {
            type_signature,
            volatility,
        }
    }
    /// variadic - Creates a variadic signature that represents an arbitrary number of arguments all from a type in common_types.
    pub fn variadic(common_types: Vec<DataType>, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Variadic(common_types, 0),
            volatility,
        }
    }

    /// variadic - Creates a variadic signature that represents an arbitrary number of arguments all from a type in common_types.
    pub fn variadic_min(common_types: Vec<DataType>, min: usize, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Variadic(common_types, min),
            volatility,
        }
    }

    /// variadic_equal - Creates a variadic signature that represents an arbitrary number of arguments of the same type.
    pub fn variadic_equal(valid_type: DataType, min: usize, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::VariadicEqual(valid_type, min),
            volatility,
        }
    }
    /// uniform - Creates a function with a fixed number of arguments of the same type, which must be from valid_types.
    pub fn uniform(
        arg_count: usize,
        valid_type: DataType,
        volatility: Volatility,
    ) -> Self {
        Self {
            type_signature: TypeSignature::Uniform(arg_count, valid_type),
            volatility,
        }
    }
    /// exact - Creates a signature which must match the types in exact_types in order.
    pub fn exact(exact_types: Vec<DataType>, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Exact(exact_types),
            volatility,
        }
    }
    /// any - Creates a signature which can a be made of any type but of a specified number
    pub fn any(arg_count: usize, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Any(arg_count),
            volatility,
        }
    }

    /// Validate argument counts matches the `signature`.
    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> RuntimeResult<()> {
        self.type_signature.validate_arg_count(name, arg_len)
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