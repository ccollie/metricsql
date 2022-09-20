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
use crate::functions::data_type::DataType;
use crate::functions::MAX_ARG_COUNT;
use crate::parser::ParseError;

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
    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> Result<(), ParseError> {

        fn expect_arg_count(name: &str, arg_len: usize, expected: usize) -> Result<(), ParseError> {
            if arg_len != expected {
                return Err(ParseError::ArgumentError(format!(
                    "The function {} expected {} arguments but received {}",
                    name,
                    expected,
                    arg_len
                )));
            }
            Ok(())
        }

        fn expect_min_args(name: &str, args_len: usize, min: usize) -> Result<(), ParseError> {
            if args_len < min {
                return Err(ParseError::ArgumentError(format!(
                    "The function {} expected a minimum of {} arguments but received {}",
                    name,
                    min,
                    args_len
                )));
            }
            Ok(())
        }

        return match self {
            TypeSignature::VariadicEqual(_data_type_, min) => {
                expect_min_args(name,arg_len, *min)
            },
            TypeSignature::Variadic(valid_types, min) => {
                if valid_types.len() < *min || arg_len > valid_types.len() {
                    return Err(ParseError::ArgumentError(format!(
                        "The function {} expected between {} and {} argument, but received {}",
                        name,
                        min,
                        valid_types.len(),
                        arg_len
                    )));
                }
                Ok(())
            },
            TypeSignature::Uniform(number, _valid_type_) => {
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
    /// uniform - Creates a signature with a fixed number of arguments of the same type, which must be from valid_types.
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
    pub fn validate_arg_count(&self, name: &str, arg_len: usize) -> Result<(), ParseError> {
        self.type_signature.validate_arg_count(name, arg_len)
    }

    pub fn expand_types(&self) -> (Vec<DataType>, usize) {  // todo: also return min count
        match &self.type_signature {
            TypeSignature::Variadic(types, min) => {
                (types.clone(), *min)
            }
            TypeSignature::VariadicEqual(data_type, min) => {
                (vec![*data_type;MAX_ARG_COUNT], *min)
            }
            TypeSignature::Uniform(count, data_type) => {
                (vec![*data_type;MAX_ARG_COUNT], *count)
            }
            TypeSignature::Exact(types) => {
                (types.clone(), types.len())
            }
            TypeSignature::Any(count) => {
                (vec![DataType::Series; *count], *count) // TODO:: !!!! have a Datatype::Any
            }
        }
    }
}