use arrow_schema::DataType;

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