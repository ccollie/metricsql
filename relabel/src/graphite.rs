use std::collections::HashMap;
use std::fmt::Display;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphiteMatches(pub Vec<String>);

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphiteMatchTemplate {
    s_orig: String,
    parts: Vec<String>,
}

impl GraphiteMatchTemplate {
    pub fn new(s: &str) -> Self {
        let s_orig = s.to_string();
        let mut parts: Vec<String> = Vec::new();

        let mut s = s;
        loop {
            if let Some(n) = s.find('*') {
                append_graphite_match_template_parts(&mut parts, &s[0..n]);
                append_graphite_match_template_parts(&mut parts, "*");
                s = &s[n + 1..];
            } else {
                append_graphite_match_template_parts(&mut parts, s);
                break;
            }
        }

        Self {
            s_orig,
            parts,
        }
    }

    /// Match matches s against gmt.
    ///
    /// On success it adds matched captures to dst and returns it with true.
    /// On failure it returns false.
    pub fn is_match(&self, dst: &mut Vec<String>, s: &str) -> bool {
        dst.push(s.to_string());
        let mut s = s;
        if !self.parts.is_empty() {
            let p = &self.parts[self.parts.len() - 1];
            if p != "*" && !s.ends_with(p) {
                // fast path - suffix mismatch
                return false;
            }
        }

        let mut i: usize = 0;
        while i < self.parts.len() {
            let part = &self.parts[i];
            if part != "*" {
                if !s.starts_with(part) {
                    // Cannot match the current part
                    return false;
                }
                s = &s[part.len()..];
                continue;
            }
            // Search for the matching substring for '*' part.
            if i + 1 >= self.parts.len() {
                // Matching the last part.
                if s.contains('.') {
                    // The '*' cannot match string with dots.
                    return false;
                }
                dst.push(s.to_string());
                return true;
            }
            // Search for the start of the next part.
            let p = &self.parts[i + 1];
            i += 1;

            if let Some(n) = s.find(p) {
                let tmp = &s[0..n];
                if tmp.contains('.') {
                    // The '*' cannot match string with dots.
                    return false;
                }
                dst.push(tmp.to_string());
                s = &s[n + p.len()..];
            } else {
                // Cannot match the next part
                return false;
            }

            i += 1;
        }
        s.is_empty()
    }
}

impl Display for GraphiteMatchTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.s_orig)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphiteLabelRule {
    grt: GraphiteReplaceTemplate,
    pub target_label: String,
}

impl GraphiteLabelRule {
    pub fn expand(&self, matches: &Vec<String>) -> String {
        self.grt.expand(matches)
    }
}

impl Display for GraphiteLabelRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "replaceTemplate={}, target_label={}", self.grt, self.target_label)
    }
}

pub fn new_graphite_label_rules(m: &HashMap<String, String>) -> Vec<GraphiteLabelRule> {
    let mut result = Vec::with_capacity(m.len());
    for (labelName, replaceTemplate) in m.iter() {
        result.push(GraphiteLabelRule {
            grt: GraphiteReplaceTemplate::new(replaceTemplate),
            target_label: labelName.clone(),
        })
    }
    result
}

fn append_graphite_match_template_parts(dst: &mut Vec<String>, s: &str) {
    if !s.is_empty() {
        // Skip empty part
        dst.push(s.to_string());
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphiteReplaceTemplate {
    s_orig: String,
    parts: Vec<GraphiteReplaceTemplatePart>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphiteReplaceTemplatePart {
    n: i32,
    s: String,
}

impl GraphiteReplaceTemplate {
    pub fn new(s: &str) -> Self {
        let s_orig = s.to_string();
        let mut s = s;
        let mut parts: Vec<GraphiteReplaceTemplatePart> = vec![];
        loop {
            if let Some(n) = s.find('$') {
                if n > 0 {
                    append_graphite_replace_template_parts(&mut parts, &s[0..n], -1)
                }
                s = &s[n + 1..];
                if !s.is_empty() && s.starts_with('$') {
                    // The index in the form ${123}
                    if let Some(n) = s.find('}') {
                        let idx_str = &s[1..n];
                        s = &s[n + 1..];
                        let val = format!("${{{idx_str}}}");
                        if let Ok(idx) = idx_str.parse::<i32>() {
                            append_graphite_replace_template_parts(&mut parts, &val, idx)
                        } else {
                            append_graphite_replace_template_parts(&mut parts, &val, -1)
                        }
                    } else {
                        let val = format!("${s}");
                        append_graphite_replace_template_parts(&mut parts, &val, -1);
                        break;
                    }
                } else {
                    // The index in the form $123
                    let mut n = 0;
                    for ch in s.chars() {
                        if ch < '0' || ch > '9' {
                            break;
                        }
                        n += 1;
                    }
                    let idx_str = &s[0..n];
                    s = &s[n..];
                    let val = format!("${}", idx_str);
                    if let Ok(idx) = idx_str.parse::<i32>() {
                        append_graphite_replace_template_parts(&mut parts, &val, idx)
                    } else {
                        append_graphite_replace_template_parts(&mut parts, &val, -1)
                    }
                }
            } else {
                append_graphite_replace_template_parts(&mut parts, s, -1);
                break;
            }
        }

        GraphiteReplaceTemplate {
            s_orig,
            parts,
        }
    }

    /// Expand expands the template with the given matches into dst and returns it.
    pub fn expand(&self, matches: &Vec<String>) -> String {
        // todo: just a WAG
        let capacity = matches.len() * 16;
        let mut dst = String::with_capacity(capacity);
        for part in self.parts.iter() {
            let n = part.n;
            if n >= 0 && n < matches.len() as i32 {
                dst.push_str(&matches[n as usize].to_string());
            } else {
                dst.push_str(&part.s);
            }
        }
        dst
    }
}

impl Display for GraphiteReplaceTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.s_orig)
    }
}

fn append_graphite_replace_template_parts(dst: &mut Vec<GraphiteReplaceTemplatePart>, s: &str, n: i32) {
    if !s.is_empty() {
        dst.push(GraphiteReplaceTemplatePart {
            s: s.to_string(),
            n,
        })
    }
}