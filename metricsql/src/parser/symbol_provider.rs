use crate::parser::with_expr::must_parse_with_arg_expr;
use crate::parser::ParseResult;
use crate::prelude::WithArgExpr;
use std::collections::HashMap;
use std::sync::Arc;

pub type SymbolProviderRef = Arc<dyn SymbolProvider>;

pub trait SymbolProvider {
    fn size(&self) -> usize;
    fn get_symbol(&self, name: &str) -> Option<&WithArgExpr>;
}

#[derive(Debug)]
pub struct HashMapSymbolProvider(HashMap<String, WithArgExpr>);

impl HashMapSymbolProvider {
    pub fn new() -> Self {
        HashMapSymbolProvider(HashMap::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        HashMapSymbolProvider(HashMap::with_capacity(capacity))
    }

    pub fn register(&mut self, key: &str, expr: WithArgExpr) -> Option<WithArgExpr> {
        // todo: ensure valid symbol name
        self.0.insert(key.to_string(), expr)
    }

    pub fn register_number(&mut self, name: &str, value: f64) -> Option<WithArgExpr> {
        self.register(name, WithArgExpr::new_number(name, value))
    }

    pub fn register_string<K: Into<String>>(
        &mut self,
        name: &str,
        value: K,
    ) -> Option<WithArgExpr> {
        self.register(name, WithArgExpr::new_string(name, value.into()))
    }

    pub fn register_lambda(&mut self, expr: &str) -> ParseResult<Option<WithArgExpr>> {
        let func = must_parse_with_arg_expr(expr)?;
        Ok(self.0.insert(func.name.to_string(), func))
    }

    pub fn has(&self, key: &str) -> bool {
        self.0.get(key).is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl SymbolProvider for HashMapSymbolProvider {
    fn size(&self) -> usize {
        self.0.len()
    }

    fn get_symbol(&self, name: &str) -> Option<&WithArgExpr> {
        self.0.get(name)
    }
}
