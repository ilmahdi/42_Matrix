use crate::matrix::Matrix;
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

pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32> {
    let t = near * (fov / 2.0).tan();
    let r = t * ratio;

    let a = 2.0 * near / (2.0 * r);
    let b = 2.0 * near / (2.0 * t);
    let c = 0.0;
    let d = 0.0;
    let e = -(far + near) / (far - near);
    let f = -(2.0 * far * near) / (far - near);

    Matrix::from(vec![
        vec![a, 0.0, c, 0.0],
        vec![0.0, b, d, 0.0],
        vec![0.0, 0.0, e, f],
        vec![0.0, 0.0, -1.0, 0.0],
    ])
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
