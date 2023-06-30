use datafusion::arrow::datatypes::DataType::Timestamp;
use datafusion::arrow::datatypes::TimeUnit;
use datafusion::common::ScalarValue;

/// Convert [ScalarValue] to [Timestamp].
/// Return `None` if given scalar value cannot be converted to a valid timestamp.
pub fn scalar_value_to_timestamp(scalar: &ScalarValue) -> Option<Timestamp> {
    match scalar {
        ScalarValue::Int64(val) => val.map(Timestamp::new_millisecond),
        ScalarValue::Utf8(Some(s)) => match Timestamp::from_str(s) {
            Ok(t) => Some(t),
            Err(e) => {
                logging::error!(e;"Failed to convert string literal {s} to timestamp");
                None
            }
        },
        ScalarValue::TimestampSecond(v, _) => v.map(Timestamp::new_second),
        ScalarValue::TimestampMillisecond(v, _) => v.map(Timestamp::new_millisecond),
        ScalarValue::TimestampMicrosecond(v, _) => v.map(Timestamp::new_microsecond),
        ScalarValue::TimestampNanosecond(v, _) => v.map(Timestamp::new_nanosecond),
        _ => None,
    }
}

pub fn timestamp_to_scalar_value(unit: TimeUnit, val: Option<i64>) -> ScalarValue {
    match unit {
        TimeUnit::Second => ScalarValue::TimestampSecond(val, None),
        TimeUnit::Millisecond => ScalarValue::TimestampMillisecond(val, None),
        TimeUnit::Microsecond => ScalarValue::TimestampMicrosecond(val, None),
        TimeUnit::Nanosecond => ScalarValue::TimestampNanosecond(val, None),
    }
}