const ParseCacheMaxLen: usize = 10e3;

pub(super) struct ParseCacheValue {
    e: Box<Expression>,
    err: Box<Error>
}
