use minitrace::prelude::*;
use minstant::Instant;

pub struct Tracer {
    pub span: Span,
}

struct TracerInner {
    span: Span,
}

impl TracerInner {
    pub fn new_child(&mut self, label: &'static str, properties: I) -> TracerInner
        where
            I: IntoIterator<Item = (&'static str, impl Into<String>)>
    {
        let mut span = Span::enter_with_parent(label, &self.span);
        for (label, value) in properties.into_iter() {
            span.add_property(|| (label, value.into()))
        }

        TracerInner {
            span
        }
    }

    pub fn add_property(&mut self, label: &'static str, value: impl Into<String>) {
        self.span.add_property(|| (label, value.into()))
    }

}

impl Tracer {
    pub fn new(event: &'static str) -> Self {
        let (root_span, _ ) = Span::root(event);
        Self {
            span: root_span,
        }
    }

    pub fn new_child(&mut self, label: &'static str, properties: I) -> Tracer
        where
            I: IntoIterator<Item = (&'static str, impl Into<String>)>
    {
        let mut span = Span::enter_with_parent(label, &self.span);
        for (label, value) in properties.into_iter() {
            span.add_property(|| (label, value.into()))
        }

        Tracer {
            span
        }
    }

    pub fn add_property(&mut self, label: &'static str, value: impl Into<String>) {
        self.span.add_property(|| (label, value.into()))
    }

    pub fn add_float_property(&mut self, label: &'static str, value: f64) {
        self.span.add_property(|| (label, value.to_string()))
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