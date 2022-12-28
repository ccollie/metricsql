use std::fmt;
use std::fmt::Formatter;

// Original Code: Polars
// https://github.com/pola-rs/polars/blob/master/polars/polars-core/src/fmt.rs
// License Apache-2.0

const SIZES_NS: [i64; 4] = [
    86_400_000_000_000,
    3_600_000_000_000,
    60_000_000_000,
    1_000_000_000,
];
const NAMES: [&str; 4] = ["d", "h", "m", "s"];
const SIZES_US: [i64; 4] = [86_400_000_000, 3_600_000_000, 60_000_000, 1_000_000];
const SIZES_MS: [i64; 4] = [86_400_000, 3_600_000, 60_000, 1_000];
const SIZES_S: [i64; 4] = [86_400, 3_600, 60, 1];

fn fmt_duration_ns(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0ns");
    }
    format_duration(f, v, SIZES_NS.as_slice(), NAMES.as_slice())?;
    if v % 1000 != 0 {
        write!(f, "{}ns", v % 1_000_000_000)?;
    } else if v % 1_000_000 != 0 {
        write!(f, "{}µs", (v % 1_000_000_000) / 1000)?;
    } else if v % 1_000_000_000 != 0 {
        write!(f, "{}ms", (v % 1_000_000_000) / 1_000_000)?;
    }
    Ok(())
}

fn fmt_duration_us(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0µs");
    }
    format_duration(f, v, SIZES_US.as_slice(), NAMES.as_slice())?;
    if v % 1000 != 0 {
        write!(f, "{}µs", (v % 1_000_000))?;
    } else if v % 1_000_000 != 0 {
        write!(f, "{}ms", (v % 1_000_000) / 1_000)?;
    }
    Ok(())
}

pub fn fmt_duration_ms(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0ms");
    }
    format_duration(f, v, SIZES_MS.as_slice(), NAMES.as_slice())?;
    if v % 1_000 != 0 {
        write!(f, "{}ms", (v % 1_000))?;
    }
    Ok(())
}

pub fn fmt_duration_s(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0s");
    }
    format_duration(f, v, SIZES_S.as_slice(), NAMES.as_slice())
}

pub fn format_duration(f: &mut Formatter, v: i64, sizes: &[i64], names: &[&str]) -> fmt::Result {
    for i in 0..4 {
        let whole_num = if i == 0 {
            v / sizes[i]
        } else {
            (v % sizes[i - 1]) / sizes[i]
        };
        if whole_num <= -1 || whole_num >= 1 {
            write!(f, "{}{}", whole_num, names[i])?;
            if v % sizes[i] != 0 {
                write!(f, " ")?;
            }
        }
    }
    Ok(())
}