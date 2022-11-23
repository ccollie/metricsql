use minitrace::prelude::*;
use minstant::Instant;

pub struct Tracer {
    /// startTime is the time when the Tracer was created
    pub start_time: Instant,
    pub message: String,
    pub name: String,
    pub span: Span,
    pub enabled: bool
}

impl Tracer {
    pub fn new(enabled: bool) -> Self {
        let (root_span, _ ) = Span::root("root");
        let now = Instant::now();
        Self {
            start_time: now,
            message: "".to_string(),
            name: "root".to_string(),
            span: root_span,
            enabled
        }
    }

    pub fn new_child(&mut self, label: &str) {
        let mut sg2 = Span::enter_with_parent("a span", &self.span);

    }

    fn enter(&mut self) -> &mut Self {
        self.start_time = Instant::now();
        self
    }

    pub fn add_property(&mut self, label: &'static str, value: &str) {
        self.span.add_property(|| (label, value.to_owned()))
    }

    #[inline]
    pub fn add_properties<I>(&mut self, properties: I)
        where
            I: IntoIterator<Item = (&'static str, String)>
    {
        self.span.add_properties(|| properties)
    }

    // Donef appends the given message to t and finished it.
    //
    // Donef cannot be called multiple times.
    // Other Tracer functions cannot be called after Donef call.
    pub fn donef(&mut self, message: &str) {
        todo!();
       // self.end_time = time.Now()
    }

}