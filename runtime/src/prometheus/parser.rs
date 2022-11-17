use std::borrow::Cow;
use std::ops::Deref;
use std::slice::Iter;

use metricsql::utils::parse_float;

use crate::{RuntimeError, RuntimeResult};

/// Rows contains parsed Prometheus rows.
#[derive(Default, Clone)]
pub struct Rows {
    pub rows: Vec<Row>,
}

impl Rows {
    pub fn new(s: &str) -> RuntimeResult<Self> {
        let mut res = Self {
            rows: vec![]
        };
        res.unmarshal(s)?;
        Ok(res)
    }
    
    /// Reset resets rs.
    pub fn reset(&mut self) {
        self.rows.clear();
    }

    /// unmarshal unmarshals Prometheus exposition text rows from s.
    ///
    /// See https://github.com/prometheus/docs/blob/master/content/docs/instrumenting/exposition_formats.md#text-format-details
    ///
    /// s shouldn't be modified while rs is in use.
    pub fn unmarshal(&mut self, s: &str) -> RuntimeResult<()> {
        let no_escapes = s.find('\\').is_none();
        unmarshal_rows(&mut self.rows, s, no_escapes)
    }
    
    pub fn iter(&self) -> Iter<'_, Row> {
        self.rows.iter()
    }
}

impl TryFrom<&str> for Rows {
    type Error = RuntimeError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Rows::new(value)
    }
}

impl Deref for Rows {
    type Target = Vec<Row>;

    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}

/// Row is a single Prometheus row.
#[derive(Default, Clone)]
pub struct Row {
    pub metric: String,
    pub tags: Vec<Tag>,
    pub value: f64,
    pub timestamp: i64,
}

impl Row {
    pub fn reset(&mut self) {
        self.metric = "".to_string();
        self.tags.clear();
        self.value = 0_f64;
        self.timestamp = 0
    }

    pub(self) fn unmarshal(&mut self, s: &str, no_escapes: bool) -> RuntimeResult<()> {
        self.reset();
        let mut s = s;
        s = skip_leading_whitespace(s);
        if let Some(n) = s.find('{') {
            // Tags found. Parse them.
            self.metric = skip_trailing_whitespace(&s[0..n]).to_string();
            s = &s[n + 1..];
            match unmarshal_tags(&mut self.tags, s, no_escapes) {
                Err(err) => {
                    return Err(RuntimeError::from(format!("cannot unmarshal tags: {:?}", err)));
                }
                Ok(v) => s = v
            }
            if !s.is_empty() && s.chars().next().unwrap() == ' ' {
                // Fast path - skip whitespace.
                s = &s[1..];
            }
        } else {
            // Tags weren't found. Search for value after whitespace
            if let Some(n) = next_whitespace(s) {
                self.metric = s[0..n].to_string();
                s = &s[n + 1..];
            } else {
                return Err(RuntimeError::from("missing value"));
            }
        }
        if self.metric.is_empty() {
            return Err(RuntimeError::from("metric cannot be empty"));
        }
        s = skip_leading_whitespace(s);
        s = skip_trailing_whitespace(s);
        if s.len() == 0 {
            return Err(RuntimeError::from("value cannot be empty"));
        }
        let n = next_whitespace(s);
        if n.is_none() {
            // There is no timestamp.
            match parse_float(s) {
                Err(err) => {
                    return Err(RuntimeError::from(format!("cannot parse value {}: {:?}", s, err)));
                }
                Ok(v) => self.value = v
            }
            return Ok(());
        }
        let n = n.unwrap();
        // There is a timestamp.
        let ts_part = &s[0..n];
        match parse_float(&s[0..n]) {
            Err(err) => {
                return Err(RuntimeError::from(format!("cannot parse value {}: {:?}", ts_part, err)));
            }
            Ok(v) => self.value = v
        }
        s = skip_leading_whitespace(&s[n + 1..]);
        if s.len() == 0 {
            // There is no timestamp - just a whitespace after the value.
            return Ok(());
        }
        // There are some whitespaces after timestamp
        s = skip_trailing_whitespace(s);
        return match parse_float(s) {
            Err(err) => {
                Err(RuntimeError::from(format!("cannot parse timestamp {}: {:?}", s, err)))
            }
            Ok(v) => {
                let mut ts = v as i64;
                if ts >= -1 << 31 && ts < 1 << 31 {
                    // This looks like OpenMetrics timestamp in Unix seconds.
                    // Convert it to milliseconds.
                    //
                    // See https://github.com/OpenObservability/OpenMetrics/blob/master/specification/OpenMetrics.md#timestamps
                    ts *= 1000
                }
                self.timestamp = ts;
                Ok(())
            }
        };
    }
}


fn skip_trailing_comment(s: &str) -> &str {
    match s.rfind('#') {
        None => s,
        Some(n) => {
            &s[0..n]
        }
    }
}

fn skip_leading_whitespace(s: &str) -> &str {
    // Prometheus treats ' ' and '\t' as whitespace
    // according to https://github.com/prometheus/docs/blob/master/content/docs/instrumenting/exposition_formats.md#text-format-details
    let x: &[_] = &[' ', '\t'];
    s.trim_start_matches(x)
}

fn skip_trailing_whitespace(s: &str) -> &str {
    // Prometheus treats ' ' and '\t' as whitespace
    // according to https://github.com/prometheus/docs/blob/master/content/docs/instrumenting/exposition_formats.md#text-format-details
    let x: &[_] = &[' ', '\t'];
    s.trim_end_matches(x)
}

fn next_whitespace(s: &str) -> Option<usize> {
    let n = s.find(' ');
    if n.is_none() {
        return s.find('\t');
    }

    let n = n.unwrap();
    return Some(
        match s.find('\t') {
            None => n,
            Some(n1) => {
                if n1 > n { n } else { n1 }
            }
        }
    );
}


fn unmarshal_rows(dst: &mut Vec<Row>, s: &str, no_escapes: bool) -> RuntimeResult<()> {
    let mut s = s;
    while !s.is_empty() {
        match s.find('\n') {
            None => {
                // The last line.
                unmarshal_row(dst, s, no_escapes)?;
                break;
            }
            Some(n) => {
                unmarshal_row(dst, &s[0..n], no_escapes.clone())?;
                s = &s[n + 1..];
            }
        }
    }
    Ok(())
}

fn unmarshal_row(dst: &mut Vec<Row>, s: &str, no_escapes: bool) -> RuntimeResult<()> {
    let mut s = s.trim_end_matches(&['\r']);
    s = skip_leading_whitespace(s);
    if s.is_empty() {
        // Skip empty line
        return Ok(());
    }
    if s.chars().next().unwrap() == '#' {
        // Skip comment
        return Ok(());
    }
    let mut r = Row::default();
    match r.unmarshal(s, no_escapes) {
        Err(err) => {
            let msg = format!("cannot unmarshal Prometheus line {}: {:?}", s, err);
            Err(RuntimeError::from(msg))
        }
        _ => {
            dst.push(r);
            Ok(())
        }
    }
}


fn unmarshal_tags<'a>(dst: &mut Vec<Tag>, s: &'a str, no_escapes: bool) -> RuntimeResult<&'a str> {
    let mut s = s;

    let add_tag = |dst: &mut Vec<Tag>, k: &str, v: &str| {
        if k.len() > 0 {
            // Allow empty values (value.len()==0) - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/453
            let tag = Tag {
                key: k.to_string(),
                value: v.to_string(),
            };
            dst.push(tag);
        }
    };

    loop {
        s = skip_leading_whitespace(s);
        if !s.is_empty() && s.chars().next().unwrap() == '}' {
            // End of tags found.
            return Ok(&s[1..]);
        }

        let n: usize;
        match s.find('=') {
            None => return Err(RuntimeError::from(format!("missing value for tag {}", s))),
            Some(pos) => n = pos
        }

        let key = skip_trailing_whitespace(&s[0..n]);
        s = skip_leading_whitespace(&s[n + 1..]);
        if s.is_empty() || s.chars().next().unwrap() != '"' {
            return Err(RuntimeError::from(format!("expecting quoted value for tag {}; got {}", key, s)));
        }
        let mut value = &s[1..];
        if no_escapes {
            // Fast path - the line has no escape chars
            match value.find('"') {
                None => {
                    return Err(RuntimeError::from(format!("missing closing quote for tag value {}", s)));
                }
                Some(pos) => {
                    (value, s) = s.split_at(pos);
                    add_tag(dst, key, value);
                }
            }
        } else {
            // Slow path - the line contains escape chars
            match find_closing_quote(s) {
                None => {
                    return Err(RuntimeError::from(format!("missing closing quote for tag value {}", s)));
                }
                Some(n) => {
                    // todo: fix this
                    let v = unescape_value(&s[1..n]);
                    add_tag(dst, key, &*v);
                    s = &s[n + 1..];
                }
            }
        }

        s = skip_leading_whitespace(s);
        if !s.is_empty() && s.chars().next().unwrap() == '}' {
            // End of tags found.
            return Ok(&s[1..]);
        }
        if s.is_empty() || s.chars().next().unwrap() != ',' {
            return Err(RuntimeError::from(format!("missing comma after tag {}={}", key, value)));
        }
        s = &s[1..]
    }
}

/// Tag is a Prometheus tag.
#[derive(Default, Clone)]
pub struct Tag {
    pub key: String,
    pub value: String,
}

impl Tag {
    pub fn reset(&mut self) {
        self.key = "".to_string();
        self.value = "".to_string();
    }
}


fn find_closing_quote(s: &str) -> Option<usize> {
    let mut s = s;
    if s.is_empty() || s.chars().next().unwrap() != '"' {
        return None;
    }
    let mut off = 1;
    s = &s[1..];
    loop {
        match s.find('"') {
            None => return None,
            Some(n) => {
                if prev_backslashes_count(&s[0..n]) % 2 == 0 {
                    return Some(off + n);
                }
                off += n + 1;
                s = &s[n.clone() + 1..];
            }
        }
    }
}

fn unescape_value(s: &str) -> Cow<str> {
    let n = s.find('\\');
    if n.is_none() {
        // Fast path - nothing to unescape
        return Cow::Borrowed(s);
    }
    let mut s = s;
    let mut n = n.unwrap();
    let mut b: Vec<u8> = Vec::with_capacity(s.len());
    loop {
        b.extend_from_slice((&s[0..n]).as_ref());
        s = &s[n + 1..];
        if s.is_empty() {
            b.push(b'\\');
            break;
        }
        // label_value can be any sequence of UTF-8 characters, but the backslash (\), double-quote ("),
        // and line feed (\n) characters have to be escaped as \\, \", and \n, respectively.
        // See https://github.com/prometheus/docs/blob/master/content/docs/instrumenting/exposition_formats.md
        let ch = s.chars().next().unwrap();
        match ch {
            '\\' => b.push(b'\\'),
            '"' => b.push(u8::try_from('"').unwrap()),
            'n' => b.push(b'\n'),
            _ => {
                b.push(b'\\');
                b.push(u8::try_from(ch).unwrap());
            }
        }
        s = &s[1..];
        match s.find('\\') {
            None => {
                b.extend_from_slice(s.as_bytes());
                break;
            }
            Some(pos) => n = pos
        }
    }

    let v = String::from_utf8_lossy(&b).to_string();
    Cow::Owned(v)
}

fn append_escaped_value(dst: &mut Vec<u8>, s: &str) {
    let mut s = s;
    // label_value can be any sequence of UTF-8 characters, but the backslash (\), double-quote ("),
    // and line feed (\n) characters have to be escaped as \\, \", and \n, respectively.
    // See https://github.com/prometheus/docs/blob/master/content/docs/instrumenting/exposition_formats.md
    let chars: &[_] = &['\\', '\"', '\n'];
    loop {
        match s.find(chars) {
            None => {
                dst.extend_from_slice(s.as_bytes());
                return;
            }
            Some(n) => {
                let (left, right) = s.split_at(n);
                dst.extend_from_slice(left.as_ref());
                if right.is_empty() {
                    return;
                }
                let ch = right.chars().next().unwrap();
                match ch {
                    '\\' => dst.extend_from_slice(r#"\\"#.as_bytes()),
                    '"' => dst.extend_from_slice(r#"\""#.as_bytes()),
                    '\n' => dst.extend_from_slice(r"\n".as_bytes()),
                    _ => {}
                }
                s = right;
            }
        }
    }
}

pub fn prev_backslashes_count(s: &str) -> i64 {
    let mut n = 0;

    // Note: Not unicode safe, but i don't think the prometheus text protocol supports non-escaped
    // non-western code points
    for ch in s.chars().rev() {
        if ch == '\\' {
            n += 1
        } else {
            break;
        }
    }
    n
}


fn marshal_metric_name_with_tags(dst: &mut Vec<u8>, r: &Row) {
    dst.extend_from_slice(r.metric.as_bytes());
    if r.tags.len() == 0 {
        return;
    }
    dst.push(b'{');
    for (i, t) in r.tags.iter().enumerate() {
        dst.extend_from_slice(t.key.as_bytes());
        dst.extend_from_slice(b"=\"");
        append_escaped_value(dst, &t.value);
        dst.push(b'"');
        if i + 1 < r.tags.len() {
            dst.push(b',');
        }
    }
    dst.push(b'}');
}