use std::borrow::Cow;
use std::str::FromStr;
use metricsql::common::{Value, ValueType};
use crate::{EvalConfig, RuntimeError, RuntimeResult, Timeseries};
use crate::eval::eval_number;
use crate::functions::types::get_single_timeseries;

#[derive(Debug, PartialEq)]
pub enum QueryValue {
    RangeVector(Vec<Timeseries>),
    InstantVector(Vec<Timeseries>),
    Scalar(f64),
    String(String),
}

impl QueryValue {
    pub fn nan() -> Self {
        QueryValue::Scalar(f64::NAN)
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            QueryValue::Scalar(_) => true,
            _ => false,
        }
    }

    pub fn data_type(&self) -> ValueType {
        match &self {
            QueryValue::RangeVector(_) => ValueType::RangeVector,
            QueryValue::Scalar(_) => ValueType::Scalar,
            QueryValue::String(_) => ValueType::String,
            QueryValue::InstantVector(_) => ValueType::InstantVector,
        }
    }

    pub fn data_type_name(&self) -> &'static str {
        match &self {
            QueryValue::RangeVector(_) => "RangeVector",
            QueryValue::Scalar(_) => "Scalar",
            QueryValue::String(_) => "String",
            QueryValue::InstantVector(_) => "InstantVector",
        }
    }

    pub fn get_matrix(&self) -> &Vec<Timeseries> {
        match self {
            QueryValue::RangeVector(val) => val,
            _ => panic!("BUG: range selection value expected "),
        }
    }

    pub fn get_scalar(&self) -> RuntimeResult<f64> {
        match self {
            QueryValue::Scalar(val) => Ok(*val),
            QueryValue::InstantVector(series) => {
                let ts = get_single_timeseries(series)?;
                if ts.values.len() != 1 {
                    let msg = format!("expected a vector of size 1; got {}", ts.values.len());
                    return Err(RuntimeError::ArgumentError(msg));
                }
                Ok(ts.values[0])
            }
            _ => {
                return Err(RuntimeError::TypeCastError(format!(
                    "{} cannot be converted to a scalar",
                    self.data_type()
                )))
            }
        }
    }

    pub fn get_int(&self) -> RuntimeResult<i64> {
        match self {
            QueryValue::Scalar(val) => Ok(*val as i64),
            _ => match self.get_scalar() {
                Ok(val) => Ok(val as i64),
                Err(e) => Err(e),
            },
        }
    }

    pub fn get_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::InstantVector(val) => {
                // TODO: into
                Ok(val.clone())
            } // ????
            QueryValue::RangeVector(val) => {
                // TODO: into
                Ok(val.clone())
            } // ????
            _ => Err(RuntimeError::InvalidNumber(
                "vector parameter expected ".to_string(),
            )),
        }
    }

    // todo: get_series_into()
    pub fn get_instant_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::InstantVector(val) => Ok(val.clone()), // ????
            _ => panic!("BUG: invalid series parameter"),
        }
    }

    pub fn get_range_vector(&self) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::RangeVector(val) => Ok(val.clone()), // ????
            _ => panic!("BUG: invalid series parameter"),
        }
    }

    pub fn into_instant_vector(self, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        match self {
            QueryValue::InstantVector(mut val) => Ok(std::mem::take(&mut val)), // use .into instead ????
            QueryValue::Scalar(n) => Ok(eval_number(ec, n)),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                return Err(RuntimeError::TypeCastError(msg));
            }
        }
    }

    pub fn get_string(&self) -> RuntimeResult<String> {
        match self {
            QueryValue::String(s) => Ok(s.to_string()),
            QueryValue::InstantVector(series) => {
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
                return Err(RuntimeError::TypeCastError(msg));
            }
        }
    }

    pub fn as_instant_vec(&self, ec: &EvalConfig) -> RuntimeResult<Cow<Vec<Timeseries>>> {
        match self {
            QueryValue::InstantVector(v) => Ok(Cow::Borrowed(v)),
            QueryValue::Scalar(n) => Ok(Cow::Owned(eval_number(ec, *n))),
            _ => {
                let msg = format!("cannot cast {} to an instant vector", self.data_type());
                return Err(RuntimeError::TypeCastError(msg));
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            QueryValue::InstantVector(val) => val.is_empty(), // ????
            QueryValue::RangeVector(val) => val.is_empty(),
            QueryValue::String(v) => v.is_empty(),
            _ => false,
        }
    }

    pub fn empty_vec() -> Self {
        QueryValue::InstantVector(vec![])
    }
}

impl Value for QueryValue {
    fn value_type(&self) -> ValueType {
        match &self {
            QueryValue::RangeVector(_) => ValueType::RangeVector,
            QueryValue::Scalar(_) => ValueType::Scalar,
            QueryValue::String(_) => ValueType::String,
            QueryValue::InstantVector(_) => ValueType::InstantVector,
        }
    }
}

impl Default for QueryValue {
    fn default() -> Self {
        QueryValue::Scalar(0_f64)
    }
}

impl FromStr for QueryValue {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(QueryValue::String(s.to_string()))
    }
}

impl From<f64> for QueryValue {
    fn from(v: f64) -> Self {
        QueryValue::Scalar(v)
    }
}

impl From<i64> for QueryValue {
    fn from(v: i64) -> Self {
        QueryValue::Scalar(v as f64)
    }
}

impl From<Vec<Timeseries>> for QueryValue {
    fn from(vec: Vec<Timeseries>) -> Self {
        QueryValue::InstantVector(vec)
    }
}

impl Clone for QueryValue {
    fn clone(&self) -> Self {
        match self {
            QueryValue::RangeVector(m) => QueryValue::RangeVector(m.clone()),
            QueryValue::InstantVector(series) => QueryValue::InstantVector(series.clone()),
            QueryValue::Scalar(f) => QueryValue::Scalar(*f),
            QueryValue::String(s) => QueryValue::String(s.clone()),
        }
    }
}