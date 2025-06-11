pub mod complex;
pub mod matrix;
pub mod scalar;
pub mod vector;
use std::ops::{AddAssign, MulAssign, SubAssign};

pub use complex::Complex;
pub use matrix::Matrix;
pub use scalar::Scalar;
pub use vector::Vector;

pub fn linear_combination<K: Scalar>(u: &[Vector<K>], coefs: &[K]) -> Vector<K> {
    let mut result_data = vec![K::default(); u[0].size()];

    for (v, &c) in u.iter().zip(coefs) {
        for (res, &val) in result_data.iter_mut().zip(v.data()) {
            *res = res.mul_add(c, val);
        }
    }

    Vector::from(result_data)
}

pub fn lerp<V: Clone + AddAssign + SubAssign + MulAssign<f32>>(u: V, v: V, t: f32) -> V {
    let mut result = v;
    result -= u.clone();
    result *= t;
    result += u;
    result
}

pub fn angle_cos<K: Scalar>(u: &Vector<K>, v: &Vector<K>) -> f32 {
    let mut res: f32 = u.dot(v).to_f32() / (u.norm() * v.norm());
    if res < -1.0 {
        res = -1.0;
    } else if res > 1.0 {
        res = 1.0;
    }
    res
}

pub fn angle_cos_complex(u: &Vector<Complex>, v: &Vector<Complex>) -> f32 {
    let mut dot = Complex::new(0.0, 0.0);
    let n: usize = u.data().len();

    for i in 0..n {
        dot += u[i].conj() * v[i];
    }

    let norm_u = u.norm();
    let norm_v = v.norm();

    if norm_u == 0.0 || norm_v == 0.0 {
        return f32::NAN;
    }

    let denom = Complex::new(norm_u * norm_v, 0.0);
    let cos = dot / denom;

    // Clamp real part
    let mut real = cos.re;
    if real < -1.0 {
        real = -1.0;
    } else if real > 1.0 {
        real = 1.0;
    }

    real
}

pub fn cross_product<K: Scalar>(u: &Vector<K>, v: &Vector<K>) -> Vector<K> {
    let x = u[1] * v[2] - u[2] * v[1];
    let y = u[2] * v[0] - u[0] * v[2];
    let z = u[0] * v[1] - u[1] * v[0];

    Vector::from([x, y, z])
}

pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32> {
    let b = 1.0 / (fov / 2.0).tan();
    let a = b / ratio;
    let nf = 1.0 / (far - near);
    let c = far * nf;
    let d = -(far * near) * nf;

    let m = [
        [a, 0.0, 0.0, 0.0],
        [0.0, b, 0.0, 0.0],
        [0.0, 0.0, c, 1.0],
        [0.0, 0.0, d, 0.0],
    ];

    Matrix::from(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex; // Ensure Complex is imported

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_linear_combination_scalar() {
        let u1 = Vector::from([-10.0, 20.0]);
        let coefs1 = [-2.0];
        let result1 = linear_combination(&[u1], &coefs1);
        assert_eq!(result1.data(), vec![20.0, -40.0]);

        let u2_1 = Vector::from([5.0]);
        let u2_2 = Vector::from([5.0]);
        let u2_3 = Vector::from([10.0]);
        let coefs2 = [1.0, -1.0, 0.0];
        let result2 = linear_combination(&[u2_1, u2_2, u2_3], &coefs2);
        assert_eq!(result2.data(), vec![0.0]);

        let u3_1 = Vector::from([1.0, 2.0]);
        let u3_2 = Vector::from([3.0, 4.0]);
        let u3_3 = Vector::from([5.0, 6.0]);
        let coefs3 = [2.0, -1.0, 0.5];
        let result3 = linear_combination(&[u3_1, u3_2, u3_3], &coefs3);
        assert_eq!(result3.data(), vec![1.5, 3.0]);

        let u4_1 = Vector::from([10.0, 20.0, 30.0]);
        let u4_2 = Vector::from([1.0, 1.0, 1.0]);
        let coefs4 = [0.5, -2.0];
        let result4 = linear_combination(&[u4_1, u4_2], &coefs4);
        assert_eq!(result4.data(), vec![3.0, 8.0, 13.0]);
    }

    #[test]
    fn test_linear_combination_complex() {
        let c_u1 = Vector::from([Complex::new(-1.0, 2.0), Complex::new(3.0, -4.0)]);
        let c_coefs1 = [Complex::new(-2.0, 0.0)];
        let c_result1 = linear_combination(&[c_u1], &c_coefs1);
        assert!((c_result1.data()[0].re - 2.0).abs() < EPSILON);
        assert!((c_result1.data()[0].im - -4.0).abs() < EPSILON);
        assert!((c_result1.data()[1].re - -6.0).abs() < EPSILON);
        assert!((c_result1.data()[1].im - 8.0).abs() < EPSILON);

        let c_u2_1 = Vector::from([Complex::new(1.0, 0.0)]);
        let c_u2_2 = Vector::from([Complex::new(0.0, 1.0)]);
        let c_u2_3 = Vector::from([Complex::new(1.0, 1.0)]);
        let c_coefs2 = [
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(-1.0, 0.0),
        ];
        let c_result2 = linear_combination(&[c_u2_1, c_u2_2, c_u2_3], &c_coefs2);
        assert!((c_result2.data()[0].re - 0.0).abs() < EPSILON);
        assert!((c_result2.data()[0].im - 0.0).abs() < EPSILON);

        let c_u3_1 = Vector::from([Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
        let c_u3_2 = Vector::from([Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)]);
        let c_coefs3 = [Complex::new(2.0, 0.0), Complex::new(-1.0, 0.0)];
        let c_result3 = linear_combination(&[c_u3_1, c_u3_2], &c_coefs3);
        assert!((c_result3.data()[0].re - -1.0).abs() < EPSILON);
        assert!((c_result3.data()[0].im - -1.0).abs() < EPSILON);
        assert!((c_result3.data()[1].re - 0.0).abs() < EPSILON);
        assert!((c_result3.data()[1].im - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_lerp_scalar() {
        assert!((lerp(10.0, 20.0, 0.0) - 10.0).abs() < EPSILON);
        assert!((lerp(10.0, 20.0, 1.0) - 20.0).abs() < EPSILON);
        assert!((lerp(10.0, 50.0, 0.25) - 20.0).abs() < EPSILON);
        assert!((lerp(-100.0, 100.0, 0.75) - 50.0).abs() < EPSILON);
        let v_lerp1 = Vector::from([10.0, 100.0]);
        let v_lerp2 = Vector::from([20.0, 0.0]);
        let result_v_lerp = lerp(v_lerp1, v_lerp2, 0.5);
        assert!((result_v_lerp.data()[0] - 15.0).abs() < EPSILON);
        assert!((result_v_lerp.data()[1] - 50.0).abs() < EPSILON);
    }

    #[test]
    fn test_lerp_complex() {
        let c_lerp1 = Complex::new(1.0, 2.0);
        let c_lerp2 = Complex::new(5.0, 6.0);
        let c_result_s = lerp(c_lerp1, c_lerp2, 0.5);
        assert!((c_result_s.re - 3.0).abs() < EPSILON);
        assert!((c_result_s.im - 4.0).abs() < EPSILON);

        let c_v_lerp1 = Vector::from([Complex::new(1.0, 1.0), Complex::new(10.0, 10.0)]);
        let c_v_lerp2 = Vector::from([Complex::new(3.0, 3.0), Complex::new(0.0, 0.0)]);
        let c_result_v = lerp(c_v_lerp1, c_v_lerp2, 0.5);
        assert!((c_result_v.data()[0].re - 2.0).abs() < EPSILON);
        assert!((c_result_v.data()[0].im - 2.0).abs() < EPSILON);
        assert!((c_result_v.data()[1].re - 5.0).abs() < EPSILON);
        assert!((c_result_v.data()[1].im - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_angle_cos_scalar() {
        let u1 = Vector::from([1.0, 1.0]);
        let v1 = Vector::from([-1.0, 1.0]);
        assert!((angle_cos(&u1, &v1) - 0.0).abs() < EPSILON);

        let u2 = Vector::from([1.0, 0.0]);
        let v2 = Vector::from([1.0, 1.0]);
        assert!((angle_cos(&u2, &v2) - (1.0 / 2.0_f32.sqrt())).abs() < EPSILON);

        let u3 = Vector::from([5.0, 0.0]);
        let v3 = Vector::from([5.0, 0.0]);
        assert!((angle_cos(&u3, &v3) - 1.0).abs() < EPSILON);

        let u4 = Vector::from([1.0, 2.0]);
        let v4 = Vector::from([2.0, 1.0]);
        assert!((angle_cos(&u4, &v4) - 0.8).abs() < EPSILON);

        let u5 = Vector::from([1.0, 0.0]);
        let v5 = Vector::from([-1.0, 0.0]);
        assert!((angle_cos(&u5, &v5) - -1.0).abs() < EPSILON);

        // Test commutativity
        let u_comm = Vector::from([1.0, 2.0]);
        let v_comm = Vector::from([3.0, 4.0]);
        assert!((angle_cos(&u_comm, &v_comm) - angle_cos(&v_comm, &u_comm)).abs() < EPSILON);

        // Test with zero vectors
        let u_zero = Vector::from([0.0, 0.0]);
        let v_non_zero = Vector::from([1.0, 1.0]);
        assert!(angle_cos(&u_zero, &v_non_zero).is_nan());
        assert!(angle_cos(&v_non_zero, &u_zero).is_nan());
        assert!(angle_cos(&u_zero, &u_zero).is_nan());
    }

    #[test]
    fn test_angle_cos_complex() {
        let c_u1 = Vector::from([Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]);
        let c_v1 = Vector::from([Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)]);
        let c_result1 = angle_cos_complex(&c_u1, &c_v1);
        assert!((c_result1 - 0.0).abs() < EPSILON); // expected orthogonal in complex inner product

        let c_u2 = Vector::from([Complex::new(1.0, 1.0)]);
        let c_v2 = Vector::from([Complex::new(1.0, 1.0)]);
        let c_result2 = angle_cos_complex(&c_u2, &c_v2);
        assert!((c_result2 - 1.0).abs() < EPSILON); // same vector

        let c_u3 = Vector::from([Complex::new(1.0, 0.0)]);
        let c_v3 = Vector::from([Complex::new(0.0, 0.0)]);
        assert!(angle_cos_complex(&c_u3, &c_v3).is_nan()); // zero vector
    }

    #[test]
    fn test_cross_product_scalar() {
        let u1 = Vector::from([1.0, 2.0, 3.0]);
        let v1 = Vector::from([0.0, 0.0, 0.0]);
        let expected1 = Vector::from([0.0, 0.0, 0.0]);
        assert_eq!(cross_product(&u1, &v1), expected1);

        let u2 = Vector::from([0.0, 5.0, 0.0]);
        let v2 = Vector::from([0.0, 0.0, 0.0]);
        let expected2 = Vector::from([0.0, 0.0, 0.0]);
        assert_eq!(cross_product(&u2, &v2), expected2);

        let u3 = Vector::from([0.0, 1.0, 0.0]);
        let v3 = Vector::from([0.0, 0.0, 1.0]);
        let expected3 = Vector::from([1.0, 0.0, 0.0]);
        assert_eq!(cross_product(&u3, &v3), expected3);

        let u4 = Vector::from([1.0, 2.0, 3.0]);
        let v4 = Vector::from([4.0, 5.0, 6.0]);
        let expected4 = Vector::from([-3.0, 6.0, -3.0]);
        assert_eq!(cross_product(&u4, &v4), expected4);

        let u5 = Vector::from([5.0, 5.0, 5.0]);
        let v5 = Vector::from([0.0, 0.0, 0.0]);
        let expected5 = Vector::from([0.0, 0.0, 0.0]);
        assert_eq!(cross_product(&u5, &v5), expected5);

        let u6 = Vector::from([2.0, 4.0, 6.0]);
        let v6 = Vector::from([1.0, 2.0, 3.0]);
        let expected6 = Vector::from([0.0, 0.0, 0.0]);
        assert_eq!(cross_product(&u6, &v6), expected6);
    }

    #[test]
    fn test_cross_product_complex() {
        let c_u1 = Vector::from([
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        let c_v1 = Vector::from([
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        let c_expected1 = Vector::from([
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);
        assert_eq!(cross_product(&c_u1, &c_v1), c_expected1);

        let c_u2 = Vector::from([
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, 3.0),
        ]);
        let c_v2 = Vector::from([
            Complex::new(4.0, 4.0),
            Complex::new(5.0, 5.0),
            Complex::new(6.0, 6.0),
        ]);
        let c_expected2 = Vector::from([
            Complex::new(0.0, -6.0),
            Complex::new(0.0, 12.0),
            Complex::new(0.0, -6.0),
        ]);
        assert_eq!(cross_product(&c_u2, &c_v2), c_expected2);

        let c_u3 = Vector::from([
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        let c_v3 = Vector::from([
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
        ]);
        let c_expected3 = Vector::from([
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.0, 0.0),
        ]);
        assert_eq!(cross_product(&c_u3, &c_v3), c_expected3);
    }
}
