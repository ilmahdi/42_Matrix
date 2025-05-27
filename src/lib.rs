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

pub fn cross_product<K: Scalar>(u: &Vector<K>, v: &Vector<K>) -> Vector<K> {
    let x = u[1] * v[2] - u[2] * v[1];
    let y = u[2] * v[0] - u[0] * v[2];
    let z = u[0] * v[1] - u[1] * v[0];

    Vector::from([x, y, z])
}

pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32> {
    let b = 1.0 / (fov / 2.0).tan();
    let a = b / ratio;
    let c = -(far + near) / (far - near);
    let d = -(2.0 * far * near) / (far - near);

    let m = [
        [a, 0.0, 0.0, 0.0],
        [0.0, b, 0.0, 0.0],
        [0.0, 0.0, c, -1.0],
        [0.0, 0.0, d, 0.0],
    ];

    Matrix::from(m)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_linear_combination() {
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
    fn test_lerp() {
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
    fn test_angle_cos() {
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
    fn test_cross_product() {
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
    fn test_projection() {
        // Test with different FoVs
        let fov_90_rad = 90.0 * PI / 180.0;
        let proj_90 = projection(fov_90_rad, 1.0, 0.1, 100.0);
        let t_90 = 0.1 * (fov_90_rad / 2.0).tan();
        let r_90 = t_90 * 1.0;
        let expected_a_90 = 2.0 * 0.1 / (2.0 * r_90);
        let expected_b_90 = 2.0 * 0.1 / (2.0 * t_90);
        let expected_e_90 = -(100.0 + 0.1) / (100.0 - 0.1);
        let expected_f_90 = -(2.0 * 100.0 * 0.1) / (100.0 - 0.1);

        assert!((proj_90.data()[0][0] - expected_a_90).abs() < EPSILON);
        assert!((proj_90.data()[1][1] - expected_b_90).abs() < EPSILON);
        assert!((proj_90.data()[2][2] - expected_e_90).abs() < EPSILON);
        assert!((proj_90.data()[2][3] - expected_f_90).abs() < EPSILON);
        assert!((proj_90.data()[3][2] - -1.0).abs() < EPSILON);

        let fov_60_rad = 60.0 * PI / 180.0;
        let proj_60 = projection(fov_60_rad, 1.0, 0.1, 100.0);
        // A lower FoV should result in a larger 'a' and 'b' (less wide view)
        assert!(proj_60.data()[0][0] > proj_90.data()[0][0]);
        assert!(proj_60.data()[1][1] > proj_90.data()[1][1]);

        let fov_30_rad = 30.0 * PI / 180.0;
        let proj_30 = projection(fov_30_rad, 1.0, 0.1, 100.0);
        assert!(proj_30.data()[0][0] > proj_60.data()[0][0]);
        assert!(proj_30.data()[1][1] > proj_60.data()[1][1]);

        // Test with different ratios
        let proj_ratio_0_5 = projection(fov_60_rad, 0.5, 0.1, 100.0);
        // Changing ratio should affect 'a' but not 'b'
        // 'a' should be doubled if ratio halves
        assert!((proj_ratio_0_5.data()[0][0] - (proj_60.data()[0][0] * 2.0)).abs() < EPSILON);
        assert!((proj_ratio_0_5.data()[1][1] - proj_60.data()[1][1]).abs() < EPSILON);

        // Test with different near/far values
        let proj_near_far_diff = projection(fov_60_rad, 1.0, 5.0, 500.0);
        // 'e' and 'f' should change significantly
        let expected_e_diff = -(500.0 + 5.0) / (500.0 - 5.0);
        let expected_f_diff = -(2.0 * 500.0 * 5.0) / (500.0 - 5.0);
        assert!((proj_near_far_diff.data()[2][2] - expected_e_diff).abs() < EPSILON);
        assert!((proj_near_far_diff.data()[2][3] - expected_f_diff).abs() < EPSILON);
    }
}
