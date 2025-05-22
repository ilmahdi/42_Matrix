use std::ops::{AddAssign, MulAssign, SubAssign};

pub fn lerp<V: Clone + AddAssign + SubAssign + MulAssign<f32>>(u: V, v: V, t: f32) -> V {
    assert!(
        t >= 0.0 && t <= 1.0,
        "Interpolation factor t must be in the range [0, 1]."
    );
    let mut result = v.clone();
    result -= u.clone();
    result *= t;
    result += u;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector; // adjust this path to your project structure

    #[test]
    fn test_lerp_vector() {
        let u = Vector::from([1.0, 2.0, 3.0]);
        let v = Vector::from([4.0, 6.0, 8.0]);
        let t = 0.5;

        let result = lerp(u, v, t);
        let expected = Vector::from([2.5, 4.0, 5.5]);

        assert_eq!(result, expected);
    }
}
