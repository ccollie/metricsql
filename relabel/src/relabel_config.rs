use super::{labels_to_string, new_graphite_label_rules, DebugStep, GraphiteLabelRule, GraphiteMatchTemplate, IfExpression, ParsedRelabelConfig, RelabelActionType};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::str::FromStr;
use dynamic_lru_cache::DynamicCache;
use metricsql_common::prelude::{remove_start_end_anchors, Label};
use metricsql_common::prelude::{simplify, PromRegex};

pub const METRIC_NAME_LABEL: &str = "__name__";

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RelabelAction {
    Drop,
    DropEqual,
    DropIfContains,
    DropIfEqual,
    DropMetrics,
    Graphite,
    HashMod,
    Keep,
    KeepEqual,
    KeepIfContains,
    KeepIfEqual,
    KeepMetrics,
    Lowercase,
    LabelMap,
    LabelMapAll,
    LabelDrop,
    LabelKeep,
    #[default]
    Replace,
    ReplaceAll,
    Uppercase,
}

impl RelabelAction {
    pub fn as_str(&self) -> &'static str {
        use RelabelAction::*;
        match self {
            Graphite => "graphite",
            Replace => "replace",
            ReplaceAll => "replace_all",
            KeepIfEqual => "keep_if_equal",
            DropIfEqual => "drop_if_equal",
            KeepEqual => "keepequal",
            DropEqual => "dropequal",
            Keep => "keep",
            Drop => "drop",
            DropIfContains => "drop_if_contains",
            DropMetrics => "drop_metrics",
            HashMod => "hashmod",
            KeepMetrics => "keep_metrics",
            Uppercase => "uppercase",
            Lowercase => "lowercase",
            LabelMap => "labelmap",
            LabelMapAll => "labelmap_all",
            LabelDrop => "labeldrop",
            LabelKeep => "labelkeep",
            KeepIfContains => "keep_if_contains",
        }
    }
}

impl Display for RelabelAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for RelabelAction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use RelabelAction::*;
        match s.to_lowercase().as_str() {
            "graphite" => Ok(Graphite),
            "replace" => Ok(Replace),
            "replace_all" => Ok(ReplaceAll),
            "keep_if_equal" => Ok(KeepIfEqual),
            "drop" => Ok(Drop),
            "drop_equal" | "dropequal" => Ok(DropEqual),
            "drop_if_equal" => Ok(DropIfEqual),
            "drop_if_contains" => Ok(DropIfContains),
            "drop_metrics" => Ok(DropMetrics),
            "keep_equal" | "keepequal" => Ok(KeepEqual),
            "keep" => Ok(Keep),
            "hashmod" => Ok(HashMod),
            "keep_metrics" => Ok(KeepMetrics),
            "keep_if_contains" => Ok(KeepIfContains),
            "lowercase" => Ok(Lowercase),
            "labelmap" => Ok(LabelMap),
            "labelmap_all" => Ok(LabelMapAll),
            "labeldrop" | "label_drop" => Ok(LabelDrop),
            "labelkeep" | "label_keep" => Ok(LabelKeep),
            "uppercase" => Ok(Uppercase),
            _ => Err(format!("unknown action: {}", s)),
        }
    }
}

/// RelabelConfig represents relabel config.
///
/// See https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config
#[derive(Debug, Clone)]
pub(crate) struct RelabelConfig {
    pub if_expr: Option<IfExpression>,
    pub action: RelabelActionType,
    pub source_labels: Vec<String>,
    pub separator: String,
    pub target_label: String,
    pub regex: Option<String>,
    pub modulus: u64,
    pub replacement: String,

    /// match is used together with Labels for `action: graphite`. For example:
    /// - action: graphite
    ///   match: 'foo.*.*.bar'
    ///   labels:
    ///     job: '$1'
    ///     instance: '${2}:8080'
    pub r#match: String,

    /// Labels is used together with match for `action: graphite`. For example:
    /// - action: graphite
    ///   match: 'foo.*.*.bar'
    ///   labels:
    ///     job: '$1'
    ///     instance: '${2}:8080'
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Default, Clone)]
pub struct ParsedConfigs(pub Vec<ParsedRelabelConfig>);

impl ParsedConfigs {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// apply_debug applies pcs to labels in debug mode.
    ///
    /// It returns DebugStep list - one entry per each applied relabeling step.
    pub fn apply_debug(&self, labels: &mut Vec<Label>) -> Vec<DebugStep> {
        self.apply_internal(labels, 0, true)
    }

    /// applies self to labels starting from the labelsOffset.
    pub fn apply(&self, labels: &mut Vec<Label>, labels_offset: usize) {
        let _ = self.apply_internal(labels, labels_offset, false);
    }

    fn apply_internal(
        &self,
        labels: &mut Vec<Label>,
        labels_offset: usize,
        debug: bool,
    ) -> Vec<DebugStep> {
        let mut dss: Vec<DebugStep> = Vec::with_capacity(labels.len());
        let mut in_str: String = "".to_string();
        if debug {
            in_str = labels_to_string(&labels[labels_offset..])
        }
        for prc in self.0.iter() {
            prc.apply(labels, labels_offset);
            if debug {
                let out_str = labels_to_string(&labels[labels_offset..]);
                dss.push(DebugStep {
                    rule: prc.to_string(),
                    r#in: in_str,
                    out: out_str.clone(),
                });
                in_str = out_str
            }
            if labels.len() == labels_offset {
                // All the labels have been removed.
                return dss;
            }
        }

        remove_empty_labels(labels, labels_offset);
        if debug {
            let out_str = labels_to_string(&labels[labels_offset..]);
            if out_str != in_str {
                dss.push(DebugStep {
                    rule: "remove empty labels".to_string(),
                    r#in: in_str,
                    out: out_str,
                });
            }
        }

        dss
    }
}

// todo:
// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/lib/promrelabel/config.go#L123
impl Display for ParsedConfigs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        for prc in &self.0 {
            s.push_str(&prc.to_string());
        }
        write!(f, "{}", s)
    }
}

/// parses relabel configs from the given data.
// pub fn parse_relabel_configs_data(data: &str) -> Result<ParsedConfigs, String> {
//     let rcs: Vec<RelabelConfig> =  serde_yaml::from_str(data)
//         .map_err(|e| format!("cannot parse relabel configs from data: {:?}", e))?;
//     parse_relabel_configs(rcs)
// }

/// parse_relabel_configs parses rcs to dst.
pub fn parse_relabel_configs(rcs: Vec<RelabelConfig>) -> Result<ParsedConfigs, String> {
    if rcs.is_empty() {
        return Ok(ParsedConfigs::default());
    }
    let mut prcs: Vec<ParsedRelabelConfig> = Vec::with_capacity(rcs.len());
    for (i, item) in rcs.into_iter().enumerate() {
        let prc = parse_relabel_config(item);
        if let Err(err) = prc {
            return Err(format!(
                "error when parsing `relabel_config` #{}: {:?}",
                i + 1,
                err
            ));
        }
        prcs.push(prc.unwrap());
    }
    Ok(ParsedConfigs(prcs))
}

const DEFAULT_REGEX_STR_FOR_RELABEL_CONFIG: &str = "^(?:(.*))$";
const DEFAULT_ORIGINAL_REGEX_STR_FOR_RELABEL_CONFIG: &str = "(.*)" ;
//const DEFAULT_REGEX_STR_FOR_RELABEL_CONFIG: &'static str = "^(.*)$";


// todo: use OnceLock
lazy_static! {
    pub static ref DEFAULT_ORIGINAL_REGEX_FOR_RELABEL_CONFIG: Regex = Regex::new(".*").unwrap();
    pub static ref DEFAULT_REGEX_FOR_RELABEL_CONFIG: Regex = Regex::new("^(.*)$").unwrap();
}

pub(crate) fn is_default_regex_for_config(regex: &Regex) -> bool {
    regex.as_str() == "^(.*)$"
}

fn validate_labels(
    action: RelabelActionType,
    source_labels: &Vec<String>,
    target_label: &str,
) -> Result<(), String> {
    if source_labels.is_empty() {
        return Err(format!("missing `source_labels` for `action={action}`"));
    }
    if target_label.is_empty() {
        return Err(format!("missing `target_label` for `action={action}`"));
    }
    Ok(())
}

pub fn parse_relabel_config(rc: RelabelConfig) -> Result<ParsedRelabelConfig, String> {
    use RelabelActionType::*;

    let mut source_labels = rc.source_labels;
    let mut separator = ";";

    if !rc.separator.is_empty() {
        separator = &rc.separator;
    }

    let target_label = rc.target_label;
    let mut reg_str = rc.regex.unwrap_or_default();
    let (regex_anchored, regex_original_compiled, prom_regex) =
        if !is_empty_regex_str(&reg_str) && !is_default_regex(&reg_str) {
            let regex = &reg_str[0..];

            let mut regex_orig = regex;
            if rc.action != ReplaceAll && rc.action != LabelMapAll {
                let stripped = remove_start_end_anchors(&regex);
                regex_orig = stripped;
                reg_str = format!("^(?:{stripped})$");
            }

            let regex_anchored =
                Regex::new(&reg_str).map_err(|e| format!("cannot parse `regex` {reg_str}: {:?}", e))?;

            let regex_original_compiled = Regex::new(&regex_orig)
                .map_err(|e| format!("cannot parse `regex` {}: {:?}", regex_orig, e))?;

            let prom_regex = PromRegex::new(&regex_orig).map_err(|err| {
                format!(
                    "BUG: cannot parse already parsed regex {}: {:?}",
                    regex_orig, err
                )
            })?;

            (regex_anchored, regex_original_compiled, prom_regex)
        } else {
            (
                DEFAULT_REGEX_FOR_RELABEL_CONFIG.clone(),
                DEFAULT_ORIGINAL_REGEX_FOR_RELABEL_CONFIG.clone(),
                PromRegex::new(".*").unwrap(),
            )
        };


    let modulus = rc.modulus;
    let replacement = if !rc.replacement.is_empty() {
        rc.replacement.clone()
    } else {
        "$1".to_string()
    };
    let mut graphite_match_template: Option<GraphiteMatchTemplate> = None;
    if !rc.r#match.is_empty() {
        graphite_match_template = Some(GraphiteMatchTemplate::new(&rc.r#match));
    }
    let mut graphite_label_rules: Vec<GraphiteLabelRule> = vec![];
    if !rc.labels.is_empty() {
        graphite_label_rules = new_graphite_label_rules(&rc.labels)
    }

    let mut action = rc.action.clone();
    match rc.action {
        Graphite => {
            if graphite_match_template.is_none() {
                return Err("missing `match` for `action=graphite`; see https://docs.victoriametrics.com/vmagent.html#graphite-relabeling".to_string());
            }
            if graphite_label_rules.is_empty() {
                return Err("missing `labels` for `action=graphite`; see https://docs.victoriametrics.com/vmagent.html#graphite-relabeling".to_string());
            }
            if !source_labels.is_empty() {
                return Err("`source_labels` cannot be used with `action=graphite`; see https://docs.victoriametrics.com/vmagent.html#graphite-relabeling".to_string());
            }
            if !target_label.is_empty() {
                return Err("`target_label` cannot be used with `action=graphite`; see https://docs.victoriametrics.com/vmagent.html#graphite-relabeling".to_string());
            }
            if !replacement.is_empty() {
                return Err("`replacement` cannot be used with `action=graphite`; see https://docs.victoriametrics.com/vmagent.html#graphite-relabeling".to_string());
            }
            if rc.regex.is_some() {
                return Err("`regex` cannot be used for `action=graphite`; see https://docs.victoriametrics.com/vmagent.html#graphite-relabeling".to_string());
            }
        }
        Replace => {
            if target_label.is_empty() {
                return Err("missing `target_label` for `action=replace`".to_string());
            }
        }
        ReplaceAll => validate_labels(rc.action, &source_labels, &target_label)?,
        DropIfContains | DropIfEqual | KeepIfContains | KeepIfEqual | LabelMap | LabelMapAll
        | LabelDrop | LabelKeep => {
            validate_labels(rc.action, &source_labels, &target_label)?;
            if rc.regex.is_some() {
                return Err(format!("`regex` cannot be used for `action={}`", rc.action));
            }
        }
        KeepEqual | DropEqual => validate_labels(rc.action, &source_labels, &target_label)?,
        Keep => {
            if source_labels.is_empty() && rc.if_expr.is_none() {
                return Err("missing `source_labels` for `action=keep`".to_string());
            }
        }
        Drop => {
            if source_labels.is_empty() && rc.if_expr.is_none() {
                return Err("missing `source_labels` for `action=drop`".to_string());
            }
        }
        HashMod => {
            validate_labels(rc.action, &source_labels, &target_label)?;
            if modulus < 1 {
                return Err(format!(
                    "unexpected `modulus` for `action=hashmod`: {modulus}; must be greater than 0"
                ));
            }
        }
        KeepMetrics => {
            if is_empty_regex(&rc.regex) && rc.if_expr.is_none() {
                return Err("`regex` must be non-empty for `action=keep_metrics`".to_string());
            }
            if source_labels.len() > 0 {
                return Err(format!(
                    "`source_labels` must be empty for `action=keep_metrics`; got {:?}",
                    source_labels
                ));
            }
            source_labels = vec![METRIC_NAME_LABEL.to_string()];
            action = Keep;
        }
        DropMetrics => {
            if is_empty_regex(&rc.regex) && rc.if_expr.is_none() {
                return Err("`regex` must be non-empty for `action=drop_metrics`".to_string());
            }
            if source_labels.len() > 0 {
                return Err(format!(
                    "`source_labels` must be empty for `action=drop_metrics`; got {:?}",
                    source_labels
                ));
            }
            source_labels = vec![METRIC_NAME_LABEL.to_string()];
            action = Drop;
        }
        Uppercase | Lowercase => {
            validate_labels(rc.action, &source_labels, &target_label)?;
        }
    }
    if action != Graphite {
        if graphite_match_template.is_some() {
            return Err(format!("`match` config cannot be applied to `action={}`; it is applied only to `action=graphite`", action));
        }
        if !graphite_label_rules.is_empty() {
            return Err(format!("`labels` config cannot be applied to `action={}`; it is applied only to `action=graphite`", action));
        }
    }

    // let rule_original = match serde_yaml::to_string(&rc) {
    //     Ok(data) => data,
    //     Err(err) => {
    //         panic!("BUG: cannot marshal RelabelConfig to yaml: {:?}", err);
    //     }
    // };

    // TODO:
    let rule_original = format!("{:?}", rc);

    let prc = ParsedRelabelConfig {
        rule_original: rule_original.to_string(),
        source_labels,
        separator: separator.to_string(),
        target_label: target_label.to_string(),
        regex_anchored,
        modulus,
        action,
        r#if: rc.if_expr.clone(),
        graphite_match_template,
        graphite_label_rules,
        string_replacer_cache: DynamicCache::new(64), // todo: pass in config ?
        regex: prom_regex,
        regex_original: regex_original_compiled,
        has_capture_group_in_target_label: target_label.contains("$"),
        has_capture_group_in_replacement: replacement.contains("$"),
        has_label_reference_in_replacement: replacement.contains("{{"),
        replacement,
        submatch_cache: DynamicCache::new(64),
    };
    Ok(prc)
}

fn is_default_regex(expr: &str) -> bool {
    match simplify(expr) {
        Ok((prefix, suffix)) => prefix == "" && suffix == ".*",
        _ => false,
    }
}

fn is_empty_regex(regex: &Option<String>) -> bool {
    if let Some(regex) = regex {
        return is_empty_regex_str(regex)
    }
    true
}

fn is_empty_regex_str(regex: &str) -> bool {
    regex.is_empty() || regex == "(?:)"
}

fn remove_empty_labels(labels: &mut Vec<Label>, labels_offset: usize) {
    let mut i: usize = labels.len() - 1;
    while i >= labels_offset {
        let label = &labels[i];
        if label.name.is_empty() || label.value.is_empty() {
            labels.remove(i);
        }
        i -= 1;
    }
}
