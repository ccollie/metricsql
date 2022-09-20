use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;

use metricsql::functions::{DataType, Signature, Volatility};

use crate::functions::types::FunctionImplementation;

/// Logical representation of a UDF.
pub(crate) struct Udf<P, R>
    where
        P: ?Sized + Send + Sync,
{
    /// name
    pub name: String,
    /// signature
    pub signature: Signature,
    /// The fn param is the wrapped function but be aware that the function will
    /// be passed with the slice / vec of values (either scalar or array)
    /// with the exception of zero param function, where a singular element vec
    /// will be passed.
    pub fun: Box<dyn FunctionImplementation<P, R>>,
}

impl <P: ?Sized + Send + Sync, R>Debug for Udf<P, R>
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("UDF")
            .field("name", &self.name)
            .field("signature", &self.signature)
            .field("fun", &"<FUNC>")
            .finish()
    }
}

impl <P: ?Sized, R>PartialEq for Udf<P, R>
    where
        P: Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.signature == other.signature
    }
}

impl <P: ?Sized, R>Hash for Udf<P, R>
    where
        P: Send + Sync,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.signature.hash(state);
    }
}

impl <P: ?Sized , R>Udf<P, R>
    where
        P: Send + Sync,
{
    /// Create a new UDF
    pub fn new(
        name: &str,
        signature: Signature,
        fun: impl FunctionImplementation<P, R>,
    ) -> Self {
        Self {
            name: name.to_owned(),
            signature,
            fun: Box::new(fun),
        }
    }

    fn create_basic(name: &str, arg_count: usize, f: impl FunctionImplementation<P, R>) -> Self {
        let mut types: Vec<DataType> = Vec::with_capacity(arg_count);
        types.push(DataType::Series);
        for i in 1 .. arg_count {
            types.push(DataType::Vector);
        }
        let sig: Signature = Signature::exact(types, Volatility::Immutable);
        Self::new(name, sig, f)
    }
}