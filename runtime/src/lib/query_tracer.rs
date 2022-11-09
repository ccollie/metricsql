pub struct Tracer {
    pub enabled: bool
}

impl Tracer {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled
        }
    }
}