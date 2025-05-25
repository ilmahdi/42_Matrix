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
    let mut result = v.clone();
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
