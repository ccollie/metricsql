
pub fn get_scalar_iter(value: f64, n: usize) -> impl Iterator<Item = f64> {
    std::iter::repeat(value).take(n)
}