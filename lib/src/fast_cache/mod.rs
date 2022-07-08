mod fast_cache;



pub fn compare<T: Ord>(a: &[T], b: &[T]) -> std::cmp::Ordering {
    let mut p1 = a.iter();
    let mut p2 = b.iter();
    let mut min_len = std::cmp::min(a.len(), b.len());
    while min_len > 0 {
        let v = p1.next().unwrap();
        let w = p2.next().unwrap();
        if v != w {
            return v.cmp(w);
        }
        min_len -= 1;
    }
    return a.len().cmp(&b.len());
}