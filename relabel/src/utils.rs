use metricsql_common::prelude::StringMatchHandler;
use regex::Regex;
use metricsql_common::prelude::Label;
use metricsql_common::regex_util::simplify;
use metricsql_parser::ast::Expr;
use metricsql_parser::label::Matchers;
use metricsql_parser::parser::parse;
use crate::relabel_error::{RelabelError, RelabelResult};

/// `new_labels_from_string` creates labels from s, which can have the form `metric{labels}`.
///
/// This function must be used only in non performance-critical code, since it allocates too much
pub fn new_labels_from_string(metric_with_labels: &str) -> Result<Vec<Label>, String> {
    let mut metric_with_labels = metric_with_labels.to_string();

    if metric_with_labels.starts_with('{') {
        metric_with_labels = format!("dummy_metric{}", metric_with_labels);
    }

    // add a value to metric_with_labels, so it could be parsed by prometheus protocol parser.
    let filters = parse_metric_selector(&metric_with_labels)
        .map_err(|err| format!("cannot parse metric selector {:?}: {}", metric_with_labels, err))?;


    let mut labels: Vec<Label> = vec![];
    for tag_list in filters.iter() {
        for tag in tag_list {
            labels.push(Label{
                name: tag.label.clone(),
                value: tag.value.clone(),
            });
        }
    }

    // if !strip_dummy_metric {
    //     labels.push(Label {
    //         name: "__name__"
    //     });
    // }

    Ok(labels)
}

pub fn concat_label_values(labels: &[Label], label_names: &[String], separator: &str) -> String {
    if label_names.is_empty() {
        return "".to_string();
    }
    let mut need_truncate = false;
    let mut dst = String::with_capacity(64); // todo: get from pool
    for label_name in label_names.iter() {
        if let Some(label) = labels.iter().find(|lbl| &lbl.name == label_name) {
            dst.push_str(&label.value);
            dst.push_str(separator);
            need_truncate = true;
        }
    }
    if need_truncate {
        dst.truncate(dst.len() - separator.len());
    }
    dst
}

pub fn set_label_value(labels: &mut Vec<Label>, labels_offset: usize, name: &str, value: String) {
    let sub = &mut labels[labels_offset..];
    for label in sub.iter_mut() {
        if label.name == name {
            label.value = value;
            return;
        }
    }
    labels.push(Label {
        name: name.to_string(),
        value,
    })
}

static EMPTY_STRING: &str = "";

pub fn get_label_value<'a>(labels: &'a [Label], name: &str) -> &'a str {
    for label in labels.iter() {
        if label.name == name {
            return &label.value;
        }
    }
    &EMPTY_STRING
}

/// returns label with the given name from labels.
pub fn get_label_by_name<'a>(labels: &'a mut [Label], name: &str) -> Option<&'a mut Label> {
    for label in labels.iter_mut() {
        if &label.name == name {
            return Some(label);
        }
    }
    None
}

pub fn are_equal_label_values(labels: &[Label], label_names: &[String]) -> bool {
    if label_names.len() < 2 {
        // logger.Panicf("BUG: expecting at least 2 label_names; got {}", label_names.len());
        return false;
    }
    let label_value = get_label_value(labels, &label_names[0]);
    for labelName in &label_names[1..] {
        let v = get_label_value(labels, labelName);
        if v != label_value {
            return false;
        }
    }
    true
}

pub fn contains_all_label_values(labels: &[Label], target_label: &str, source_labels: &[String]) -> bool {
    let target_label_value = get_label_value(labels, target_label);
    for sourceLabel in source_labels.iter() {
        let v = get_label_value(labels, sourceLabel);
        if !target_label_value.contains(v) {
            return false
        }
    }
    true
}

pub(super) fn get_regex_literal_prefix(regex: &Regex) -> (String, bool) {
    let (prefix, suffix) = simplify(regex.as_str())
        .unwrap_or((EMPTY_STRING.into(), EMPTY_STRING.into()));
    (prefix, suffix.is_empty())
}


pub(crate) fn is_regex_matcher(matcher: &StringMatchHandler) -> bool {
    match matcher {
        StringMatchHandler::FastRegex(_) => true,
        _ => false,
    }
}

/// parse_metric_selector parses s containing PromQL metric selector and returns the corresponding
/// LabelFilters.
pub fn parse_metric_selector(s: &str) -> RelabelResult<Matchers> {
    match parse(s) {
        Ok(expr) => match expr {
            Expr::MetricExpression(me) => {
                if me.is_empty() {
                    let msg = "labelFilters cannot be empty".to_string();
                    return Err(RelabelError::InvalidSeriesSelector(msg));
                }
                Ok(me.matchers)
            }
            _ => {
                let msg = format!("expecting metric selector; got {expr}").to_string();
                Err(RelabelError::InvalidSeriesSelector(msg))
            }
        },
        Err(err) => Err(RelabelError::ParseError(err)),
    }
}
