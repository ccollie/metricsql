#[cfg(test)]
mod tests {
    use crate::eval::validate_max_points_per_timeseries;

    #[test]
    fn test_validate_max_points_per_series_failure() {
        let f = |start: i64, end: i64, step: i64, max_points: usize| {
            match validate_max_points_per_timeseries(start, end, step, max_points) {
                Err(_) => {
                    panic!("expecting non-nil error for validate_max_points_per_series(start={}, end={}, step={}, max_points={})", start, end, step, max_points)
                }
                _ => {}
            }
        };
        // zero step
        f(0, 0, 0, 0);
        f(0, 0, 0, 1);
        // the maxPoints is smaller than the generated points
        f(0, 1, 1, 0);
        f(0, 1, 1, 1);
        f(1659962171908, 1659966077742, 5000, 700)
    }

    #[test]
    fn test_validate_max_points_per_series_success() {
        let f = |start: i64, end: i64, step: i64, max_points: usize| {
            match validate_max_points_per_timeseries(start, end, step, max_points) {
                Err(err) => panic!("unexpected error in validate_max_points_per_series(start={}, end={}, step={}, maxPoints={}): {:?}",
                                 start, end, step, max_points, err),
                _ => {}
            }
        };

        f(1, 1, 1, 2);
        f(1659962171908, 1659966077742, 5000, 800);
        f(1659962150000, 1659966070000, 10000, 393)
    }
}
