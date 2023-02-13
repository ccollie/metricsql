mod expand_with_ext;
mod optimizer;
mod simplify;

pub use optimizer::*;
pub use simplify::*;
pub(crate) use expand_with_ext::expand_with_expr;


#[cfg(test)]
mod expand_with_test;

#[cfg(test)]
mod optimizer_test;
mod optimize;

