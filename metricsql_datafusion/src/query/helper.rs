// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Vector helper functions, inspired by databend Series mod
use arrow::array::{ArrayRef, StringArray};
use arrow::compute;
use arrow::compute::kernels::comparison;
use arrow::array::BooleanArray;
use snafu::{OptionExt, ResultExt};

use crate::datatypes::error::{ArrowComputeSnafu, Result};

/// Helper functions for `Vector`.
pub struct Helper;

impl Helper {
    /// Perform SQL like operation on `names` and a scalar `s`.
    pub fn like_utf8(names: Vec<String>, s: &str) -> Result<ArrayRef> {
        let array = StringArray::from(names);

        let s = StringArray::new_scalar(s);
        let filter = comparison::like(&array, &s).context(ArrowComputeSnafu)?;

        let result = compute::filter(&array, &filter).context(ArrowComputeSnafu)?;
        Helper::try_into_vector(result)
    }

    pub fn like_utf8_filter(names: Vec<String>, s: &str) -> Result<(ArrayRef, BooleanArray)> {
        let array = StringArray::from(names);
        let s = StringArray::new_scalar(s);
        let filter = comparison::like(&array, &s).context(ArrowComputeSnafu)?;
        let result = compute::filter(&array, &filter).context(ArrowComputeSnafu)?;
        let vector = Helper::try_into_vector(result)?;

        Ok((vector, BooleanArray::from(filter)))
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{
        ArrayRef, BooleanArray
    };
    use arrow_array::Array;

    use super::*;

    #[test]
    fn test_like_utf8() {
        fn assert_vector(expected: Vec<&str>, actual: &ArrayRef) {
            let actual = actual.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(*actual, StringArray::from(expected));
        }

        let names: Vec<String> = vec!["greptime", "hello", "public", "world"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();

        let ret = Helper::like_utf8(names.clone(), "%ll%").unwrap();
        assert_vector(vec!["hello"], &ret);

        let ret = Helper::like_utf8(names.clone(), "%time").unwrap();
        assert_vector(vec!["greptime"], &ret);

        let ret = Helper::like_utf8(names.clone(), "%ld").unwrap();
        assert_vector(vec!["world"], &ret);

        let ret = Helper::like_utf8(names, "%").unwrap();
        assert_vector(vec!["greptime", "hello", "public", "world"], &ret);
    }

    #[test]
    fn test_like_utf8_filter() {
        fn assert_vector(expected: Vec<&str>, actual: &ArrayRef) {
            let actual = actual.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(*actual, StringArray::from(expected));
        }

        fn assert_filter(array: Vec<String>, s: &str, expected_filter: &BooleanArray) {
            let array = StringArray::from(array);
            let s = StringArray::new_scalar(s);
            let actual_filter = comparison::like(&array, &s).unwrap();
            assert_eq!(BooleanArray::from(actual_filter), *expected_filter);
        }

        let names: Vec<String> = vec!["greptime", "timeseries", "cloud", "database"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();

        let (table, filter) = Helper::like_utf8_filter(names.clone(), "%ti%").unwrap();
        assert_vector(vec!["greptime", "timeseries"], &table);
        assert_filter(names.clone(), "%ti%", &filter);

        let (tables, filter) = Helper::like_utf8_filter(names.clone(), "%lou").unwrap();
        assert_vector(vec![], &tables);
        assert_filter(names.clone(), "%lou", &filter);

        let (tables, filter) = Helper::like_utf8_filter(names.clone(), "%d%").unwrap();
        assert_vector(vec!["cloud", "database"], &tables);
        assert_filter(names.clone(), "%d%", &filter);
    }

}
