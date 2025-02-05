// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// https://github.com/prometheus/prometheus/blob/main/tsdb/index/postings_test.go
// https://github.com/prometheus/prometheus/blob/main/tsdb/querier_test.go
#[cfg(test)]
mod tests {
    use crate::types::MetricName;
    use crate::{MemoryPostings, SeriesRef};
    use ahash::AHashSet;
    use metricsql_parser::label::{Label, MatchOp, Matcher, Matchers};
    use std::collections::{HashMap, HashSet};

    fn hash_labels(labels: &[Label]) -> AHashSet<String> {
        labels.iter().map(|l| l.to_string()).collect()
    }

    fn labels_from_strings<S: Into<String> + Clone>(ss: &[S]) -> Vec<Label> {
        if ss.is_empty() {
            return vec![];
        }
        if ss.len() % 2 != 0 {
            panic!("labels_from_strings: odd number of strings")
        }
        let mut labels = vec![];
        for i in (0..ss.len()).step_by(2) {
            let name = <S as Clone>::clone(&ss[i]).into();
            let value = <S as Clone>::clone(&ss[i + 1]).into();
            labels.push(Label { name, value })
        }
        labels
    }

    fn add_series(ix: &mut MemoryPostings,
                  labels_map: &mut HashMap<SeriesRef, Vec<Label>>,
                  series_ref: SeriesRef, labels: &Vec<Label>) {
        let mut mn = MetricName::default();
        for label in labels.iter() {
            mn.set(&label.name, &label.value);
        }
        
        ix.add_posting(series_ref, &mn);
        labels_map.insert(series_ref, labels.clone());
    }

    fn to_label_vec(labels: &[Vec<Label>]) -> Vec<Vec<Label>> {
        labels
            .iter()
            .map(|items| {
                let mut items = items.clone();
                items.sort();
                items
            })
            .collect::<Vec<_>>()
    }

    fn get_labels_by_matcher(
        ix: &MemoryPostings,
        matchers: &[Matcher],
        series_data: &HashMap<SeriesRef, Vec<Label>>,
    ) -> Vec<Vec<Label>> {
        let filter = Matchers::new(Vec::from(matchers.clone()));
        let p = ix.postings_for_matchers(&filter).unwrap();
        let actual: Vec<_> = p
            .iter()
            .flat_map(|id| series_data.get(&id))
            .cloned()
            .collect();

        actual
    }

    fn label_vec_to_string(labels: &[Label]) -> String {
        let mut items = labels.to_vec();
        items.sort();

        items
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    #[test]
    fn test_postings_for_matchers() {
        use MatchOp::*;
        let mut ix: MemoryPostings = MemoryPostings::new();
        let mut labels_map: HashMap<SeriesRef, Vec<Label>> = HashMap::new();

        let series_data = HashMap::from([
            (1, labels_from_strings(&["n", "1"])),
            (2, labels_from_strings(&["n", "1", "i", "a"])),
            (3, labels_from_strings(&["n", "1", "i", "b"])),
            (4, labels_from_strings(&["n", "1", "i", "\n"])),
            (5, labels_from_strings(&["n", "2"])),
            (6, labels_from_strings(&["n", "2.5"])),
        ]);

        for (series_ref, labels) in series_data.iter() {
            add_series(&mut ix, &mut labels_map,  *series_ref, labels);
        }

        struct TestCase {
            matchers: Vec<Matcher>,
            exp: Vec<Vec<Label>>,
        }

        let cases = vec![
            TestCase {
                matchers: vec![Matcher::new(Equal, "n", "1").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(Equal, "i", "a").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1", "i", "a"])],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(Equal, "i", "missing").unwrap(),
                ],
                exp: vec![],
            },
            TestCase {
                matchers: vec![Matcher::new(Equal, "missing", "").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(NotEqual, "n", "1").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(NotEqual, "i", "").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(NotEqual, "missing", "").unwrap()],
                exp: vec![],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(Equal, "i", "missing").unwrap(),
                ],
                exp: vec![],
            },
            TestCase {
                matchers: vec![Matcher::new(Equal, "missing", "").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            // Not equals.
            TestCase {
                matchers: vec![Matcher::new(NotEqual, "n", "1").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(NotEqual, "i", "").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(NotEqual, "missing", "").unwrap()],
                exp: vec![],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(NotEqual, "i", "a").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(NotEqual, "i", "").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            // Regex.
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "n", "^1$").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexEqual, "i", "^a$").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1", "i", "a"])],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexEqual, "i", "^a?$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "i", "^$").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexEqual, "i", "^$").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1"])],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexEqual, "i", "^.*$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexEqual, "i", "^.+$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            // Not regex.
            TestCase {
                matchers: vec![Matcher::new(RegexNotEqual, "i", "").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexNotEqual, "n", "^1$").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexNotEqual, "n", "1").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexNotEqual, "n", "1|2.5").unwrap()],
                exp: vec![labels_from_strings(&["n", "2"])],
            },
            TestCase {
                matchers: vec![Matcher::new(RegexNotEqual, "n", "(1|2.5)").unwrap()],
                exp: vec![labels_from_strings(&["n", "2"])],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexNotEqual, "i", "^a$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexNotEqual, "i", "^a?$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexNotEqual, "i", "^$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                ]),
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexNotEqual, "i", "^.*$").unwrap(),
                ],
                exp: vec![],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexNotEqual, "i", "^.+$").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1"])],
            },
            // Combinations.
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(NotEqual, "i", "").unwrap(),
                    Matcher::new(Equal, "i", "a").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1", "i", "a"])],
            },
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(NotEqual, "i", "b").unwrap(),
                    Matcher::new(RegexEqual, "i", "^(b|a).*$").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1", "i", "a"])],
            },
            // Set optimization for Regex.
            // Refer to https://github.com/prometheus/prometheus/issues/2651.
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "n", "1|2").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                    labels_from_strings(&["n", "2"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "i", "a|b").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "i", "(a|b)").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "n", "x1|2").unwrap()],
                exp: vec![labels_from_strings(&["n", "2"])],
            },
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "n", "2|2\\.5").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            // Empty value.
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "i", "c||d").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "i", "(c||d)").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            // Test shortcut for i=~".*"
            TestCase {
                matchers: vec![Matcher::new(RegexEqual, "i", ".*").unwrap()],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            // Test shortcut for n=~".*" and i=~"^.*$"
            TestCase {
                matchers: vec![
                    Matcher::new(RegexEqual, "n", ".*").unwrap(),
                    Matcher::new(RegexEqual, "i", "^.*$").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "1", "i", "a"]),
                    labels_from_strings(&["n", "1", "i", "b"]),
                    labels_from_strings(&["n", "1", "i", "\n"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            // Test shortcut for n=~"^.*$"
            TestCase {
                matchers: vec![
                    Matcher::new(RegexEqual, "n", "^.*$").unwrap(),
                    Matcher::new(Equal, "i", "a").unwrap(),
                ],
                exp: vec![labels_from_strings(&["n", "1", "i", "a"])],
            },
            // Test shortcut for i!~".*"
            TestCase {
                matchers: vec![Matcher::new(RegexNotEqual, "i", ".*").unwrap()],
                exp: vec![],
            },
            // Test shortcut for n!~"^.*$",  i!~".*". First one triggers empty result.
            TestCase {
                matchers: vec![
                    Matcher::new(RegexNotEqual, "n", "^.*$").unwrap(),
                    Matcher::new(RegexNotEqual, "i", ".*").unwrap(),
                ],
                exp: vec![],
            },
            // Test shortcut i!~".*"
            TestCase {
                matchers: vec![
                    Matcher::new(RegexEqual, "n", ".*").unwrap(),
                    Matcher::new(RegexNotEqual, "i", ".*").unwrap(),
                ],
                exp: vec![],
            },
            // Test shortcut i!~".+"
            TestCase {
                matchers: vec![
                    Matcher::new(RegexEqual, "n", ".*").unwrap(),
                    Matcher::new(RegexNotEqual, "i", ".+").unwrap(),
                ],
                exp: to_label_vec(&[
                    labels_from_strings(&["n", "1"]),
                    labels_from_strings(&["n", "2"]),
                    labels_from_strings(&["n", "2.5"]),
                ]),
            },
            // Test shortcut i!~"^.*$"
            TestCase {
                matchers: vec![
                    Matcher::new(Equal, "n", "1").unwrap(),
                    Matcher::new(RegexNotEqual, "i", "^.*$").unwrap(),
                ],
                exp: vec![],
            },
        ];

        let mut exp: HashSet<String> = HashSet::new();

        for case in cases {
            let name = case
                .matchers
                .iter()
                .map(|matcher| matcher.to_string())
                .collect::<Vec<_>>()
                .join(", ");

            let name = format!("{{{}}}", name);
            exp.clear();

            for labels in case.exp {
                let val = label_vec_to_string(&labels);
                exp.insert(val);
            }

            let actual = get_labels_by_matcher(&ix, &case.matchers, &series_data);
            for labels in actual {
                let actual = label_vec_to_string(&labels);
                let found = exp.remove(&actual);
                assert!(found, "Evaluating {name}\n unexpected result {actual}");
            }

            assert!(
                exp.is_empty(),
                "Evaluating {name}\nextra result(s): {exp:?}"
            );
        }
    }
}
