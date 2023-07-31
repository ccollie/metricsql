use std::fmt;
use std::fmt::Display;

use crate::tests::test::Test;
use crate::RuntimeResult;

/// clearCmd is a command that wipes the test's storage state.
pub(crate) struct ClearCmd {}

impl ClearCmd {
    pub fn new() -> Self {
        ClearCmd {}
    }

    pub(crate) fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        t.clear()
    }
}

impl Display for ClearCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "clear()")
    }
}
