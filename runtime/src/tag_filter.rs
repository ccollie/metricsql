// tagFilter represents a filter used for filtering tags.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct TagFilter {
    key: Vec<u8>,
    value: Vec<u8>,
    is_negative: bool,
    is_regexp:   bool,

    // match_cost is a cost for matching a filter against a single string.
    match_cost: u64,

    // contains the prefix for regexp filter if is_regexp==true.
    regexpPrefix: String,

    // Prefix contains either {nsPrefixTagToMetricIDs, key} or {nsPrefixDateTagToMetricIDs, date, key}.
    // Additionally it contains:
    //  - value if !is_regexp.
    //  - regexpPrefix if is_regexp.
    prefix: Vec<u8>,

    // `or` values obtained from regexp suffix if it equals to "foo|bar|..."
    //
    // the regexp prefix is stored in regexpPrefix.
    //
    // This array is also populated with matching Graphite metrics if key="__graphite__"
    or_suffixes: Vec<String>,

    // Matches regexp suffix.
    re_suffix_match: fn(b: Vec<u8>) -> bool,

    // Set to true for filters matching empty value.
    is_empty_match: bool,

    // Contains reverse suffix for Graphite wildcard.
    // I.e. for `{__name__=~"foo\\.[^.]*\\.bar\\.baz"}` the value will be `zab.rab.`
    graphiteReverseSuffix: Vec<u8>
}


fn get_common_prefix(ss: &[String]) -> (string, Vec<String>) {
    if ss.len() == 0 {
        return ("", vec![]);
    }
    let mut prefix = ss[0];
    for s in ss[1..] {
        let mut i = 0;
        for i < len(s) && i < prefix.len() && s[i] == prefix[i] {
            i = i + 1
        }
        prefix = prefix[0..i];
        if prefix.len() == 0 {
            return ("", ss)
        }
    }
    let result = Vec::with_capacity(ss.len());
    for s in ss {
        result.push_str(  = s[len(prefix):]
    }
    return (prefix, result)
}