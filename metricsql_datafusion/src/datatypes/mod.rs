use std::sync::Arc;
use arrow::array::ArrayRef;
use arrow_schema::{DataType, ArrowError};
use datafusion_common::ScalarValue;
use datafusion_expr::TableType;

pub mod error;

pub fn is_timestamp_compatible(field: DataType) -> bool {
    matches!(
        field,
        DataType::Timestamp(_, _) |
        DataType::Date32 |
        DataType::Date64 |
        DataType::Int64
    )
}

/// Try to cast an arrow scalar value into Array
pub fn try_array_from_scalar_value(value: ScalarValue, length: usize) -> Result<ArrayRef, ArrowError> {
    use arrow::array::*;

    let array: Arc<dyn Array> = match value {
        ScalarValue::Null => Arc::new(NullArray::new(length)),
        ScalarValue::Boolean(v) => Arc::new(BooleanArray::from(vec![v; length])),
        ScalarValue::Float32(v) => Arc::new(Float32Array::from(vec![v; length])),
        ScalarValue::Float64(v) => Arc::new(Float64Array::from(vec![v; length])),
        ScalarValue::Int8(v) => Arc::new(Int8Array::from(vec![v; length])),
        ScalarValue::Int16(v) => Arc::new(Int16Array::from(vec![v; length])),
        ScalarValue::Int32(v) => Arc::new(Int32Array::from(vec![v; length])),
        ScalarValue::Int64(v) => Arc::new(Int64Array::from(vec![v; length])),
        ScalarValue::UInt8(v) => Arc::new(UInt8Array::from(vec![v; length])),
        ScalarValue::UInt16(v) => Arc::new(UInt16Array::from(vec![v; length])),
        ScalarValue::UInt32(v) => Arc::new(UInt32Array::from(vec![v; length])),
        ScalarValue::UInt64(v) => Arc::new(UInt64Array::from(vec![v; length])),
        ScalarValue::Utf8(v) | ScalarValue::LargeUtf8(v) => {
            Arc::new(StringArray::from(vec![v; length]))
        },
        ScalarValue::Date32(v) => Arc::new(Date32Array::from(vec![v; length])),
        ScalarValue::Date64(v) => Arc::new(Date64Array::from(vec![v; length])),
        ScalarValue::TimestampSecond(v, _) => {
            // Timezone is unimplemented now.
            Arc::new(TimestampSecondArray::from(vec![v; length]))
        }
        ScalarValue::TimestampMillisecond(v, _) => {
            // Timezone is unimplemented now.
            Arc::new(TimestampMillisecondArray::from(vec![v; length]))
        }
        ScalarValue::TimestampMicrosecond(v, _) => {
            // Timezone is unimplemented now.
            Arc::new(TimestampMicrosecondArray::from(vec![v; length]))
        }
        ScalarValue::TimestampNanosecond(v, _) => {
            // Timezone is unimplemented now.
            Arc::new(TimestampNanosecondArray::from(vec![v; length]))
        }
        ScalarValue::Time32Second(v) => {
            Arc::new(Time32SecondArray::from(vec![v; length]))
        }
        ScalarValue::Time32Millisecond(v) => {
            Arc::new(Time32MillisecondArray::from(vec![v; length]))
        }
        ScalarValue::Time64Microsecond(v) => Arc::new(Time64MicrosecondArray::from(vec![v; length])),
        ScalarValue::Time64Nanosecond(v) => {
            Arc::new(Time64NanosecondArray::from(vec![v; length]))
        }
        ScalarValue::IntervalYearMonth(v) =>
            Arc::new(IntervalYearMonthArray::from(vec![v; length])),
        ScalarValue::IntervalDayTime(v) =>
            Arc::new(IntervalDayTimeArray::from(vec![v; length])),
        ScalarValue::IntervalMonthDayNano(v) => {
            Arc::new(IntervalMonthDayNanoArray::from(vec![v; length]))
        }
        ScalarValue::DurationSecond(v) => Arc::new(DurationSecondArray::from(vec![v; length])),
        ScalarValue::DurationMillisecond(v) => Arc::new(DurationMillisecondArray::from(vec![v; length])),
        ScalarValue::DurationMicrosecond(v) => Arc::new(DurationMicrosecondArray::from(vec![v; length])),
        ScalarValue::DurationNanosecond(v) => Arc::new(DurationNanosecondArray::from(vec![v; length])),
        ScalarValue::Decimal128(v, p, s) => {
            let arr = Decimal128Array::from(vec![v])
                .with_precision_and_scale(p, s)?;
            Arc::new(arr)
        }
        _ => {
            let err = ArrowError::CastError(format!("Unsupported scalar value: {value}").to_string());
            return Err(err)
        }
    };

    Ok(Arc::new(array))
}


pub(crate) const fn table_type_name(table_type: TableType) -> &'static str {
    match table_type {
        TableType::Base => "Base",
        TableType::Temporary => "Temporary",
        TableType::View => "View",
    }
}

pub(crate) const fn data_type_name(data_type: &DataType) -> &str {
    match data_type {
        DataType::Boolean => "Boolean",
        DataType::UInt8 => "UInt8",
        DataType::UInt16 => "UInt16",
        DataType::UInt32 => "UInt32",
        DataType::UInt64 => "UInt64",
        DataType::Int8 => "Int8",
        DataType::Int16 => "Int16",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Utf8 => "Utf8",
        DataType::Timestamp(_, _) => "Timestamp",
        DataType::List(_) => "List",
        DataType::Struct(_) => "Struct",
        DataType::Dictionary(_, _) => "Dictionary",
        DataType::Interval(_) => "Interval",
        DataType::Binary => "Binary",
        DataType::FixedSizeBinary(_) => "FixedSizeBinary",
        DataType::FixedSizeList(_, _) => "FixedSizeList",
        DataType::Date32 => "Date32",
        DataType::Date64 => "Date64",
        DataType::Time32(_) => "Time32",
        DataType::Time64(_) => "Time64",
        DataType::Duration(_) => "Duration",
        DataType::Decimal128(_, _) => "Decimal128",
        DataType::Decimal256(_, _) => "Decimal256",
        DataType::Null => "Null",
        DataType::LargeBinary => "LargeBinary",
        DataType::LargeUtf8 => "LargeUtf8",
        DataType::LargeList(_) => "LargeList",
        DataType::Map(_, _) => "Map",
        DataType::Union(_, ..) => "Union",
        DataType::Float16 => "Float16",
    }
}