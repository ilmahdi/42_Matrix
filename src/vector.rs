use crate::Complex;
use crate::scalar::Scalar;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{AddAssign, Index, MulAssign, SubAssign};

/* -------------------------------------------------------------------------- */
//  Vector Struct                                                             //
/* -------------------------------------------------------------------------- */
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<K> {
    data: Vec<K>,
}

/* -------------------------------------------------------------------------- */
//  Traits Implementations                                                    //
/* -------------------------------------------------------------------------- */
impl<K: Clone, const N: usize> From<[K; N]> for Vector<K> {
    fn from(array: [K; N]) -> Self {
        Self {
            data: Vec::from(array),
        }
    }
}

impl<K: Clone> From<Vec<K>> for Vector<K> {
    fn from(vec: Vec<K>) -> Self {
        Self { data: vec }
    }
}

impl<K: Display> Display for Vector<K> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        for item in &self.data {
            writeln!(f, "[ {:^6.1} ]", item)?;
        }
        Ok(())
    }
}

impl<K: Scalar> AddAssign for Vector<K> {
    fn add_assign(&mut self, other: Self) {
        self.add(&other);
    }
}

impl<K: Scalar> SubAssign for Vector<K> {
    fn sub_assign(&mut self, other: Self) {
        self.sub(&other);
    }
}

impl<K: Scalar> MulAssign<K> for Vector<K> {
    fn mul_assign(&mut self, scalar: K) {
        self.scl(scalar)
    }
}

impl MulAssign<f32> for Vector<Complex> {
    fn mul_assign(&mut self, rhs: f32) {
        for elem in &mut self.data {
            *elem *= Complex::new(rhs, 0.0);
        }
    }
}

impl<K: Scalar> Index<usize> for Vector<K> {
    type Output = K;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

/* -------------------------------------------------------------------------- */
//  Member Functions                                                          //
/* -------------------------------------------------------------------------- */

impl<K: Clone> Vector<K> {
    pub fn new(vec: Vec<K>) -> Self {
        Self { data: vec }
    }
}

impl<K: Scalar> Vector<K> {
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<K: Scalar> Vector<K> {
    pub fn data(&self) -> &[K] {
        &self.data
    }
}

impl<K: Scalar> Vector<K> {
    fn zip_apply<F>(&mut self, other: &Vector<K>, mut func: F)
    where
        F: FnMut(&mut K, &K),
    {
        self.data
            .iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| func(a, b));
    }
}

impl<K: Scalar> Vector<K> {
    pub fn add(&mut self, other: &Vector<K>) {
        self.zip_apply(other, |a, b| *a += *b);
    }

    pub fn sub(&mut self, other: &Vector<K>) {
        self.zip_apply(other, |a, b| *a -= *b);
    }

    pub fn scl(&mut self, scalar: K) {
        self.data.iter_mut().for_each(|x| *x *= scalar);
    }
}

impl<K: Scalar> Vector<K> {
    pub fn dot(&self, v: &Vector<K>) -> K {
        let mut res = K::default();
        for (&a, &b) in self.data.iter().zip(v.data.iter()) {
            res = res.mul_add(a.conj(), b);
        }
        res
    }
}

impl<K: Scalar> Vector<K> {
    pub fn norm_1(&self) -> f32 {
        let mut res: f32 = 0.0;
        for &a in self.data.iter() {
            let af32 = a.to_f32();
            res += af32.max(-af32);
        }
        res
    }
    pub fn norm(&self) -> f32 {
        let mut res: f32 = 0.0;
        for &a in self.data.iter() {
            let af32 = a.to_f32();
            res = af32.mul_add(af32, res);
        }
        res.powf(0.5)
    }
    pub fn norm_inf(&self) -> f32 {
        let mut res: f32 = 0.0;
        for &a in self.data.iter() {
            let af32 = a.to_f32();
            res = res.max(af32.max(-af32));
        }
        res
    }
}

/* -------------------------------------------------------------------------- */
//  Unit Tests                                                                //
/* -------------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

    const EPSILON: f32 = 1e-6;

    // Helper functions for cloning and applying operations to avoid modifying original vectors in tests
    impl<K: Scalar> Vector<K> {
        fn add_clone(&self, other: &Vector<K>) -> Self {
            let mut cloned = self.clone();
            cloned.add(other);
            cloned
        }
        fn sub_clone(&self, other: &Vector<K>) -> Self {
            let mut cloned = self.clone();
            cloned.sub(other);
            cloned
        }
        fn scl_clone(&self, scalar: K) -> Self {
            let mut cloned = self.clone();
            cloned.scl(scalar);
            cloned
        }
    }

    #[test]
    fn test_vector_add_sub_scl_scalar() {
        // Provided tests
        let mut v1 = Vector::from([10.0, 20.0, 30.0]);
        let v2 = Vector::from([1.0, 2.0, 3.0]);

        v1.add(&v2);
        assert_eq!(v1.data, vec![11.0, 22.0, 33.0]);

        v1.sub(&v2);
        assert_eq!(v1.data, vec![10.0, 20.0, 30.0]);

        v1.scl(0.5);
        assert_eq!(v1.data, vec![5.0, 10.0, 15.0]);

        // Exercise 00 - Add (vectors)
        assert_eq!(
            Vector::from([5.0, 5.0])
                .add_clone(&Vector::from([3.0, 2.0]))
                .data,
            vec![8.0, 7.0]
        );
        assert_eq!(
            Vector::from([10.0, 0.0])
                .add_clone(&Vector::from([0.0, 10.0]))
                .data,
            vec![10.0, 10.0]
        );
        assert_eq!(
            Vector::from([7.0, 7.0])
                .add_clone(&Vector::from([3.0, 3.0]))
                .data,
            vec![10.0, 10.0]
        );
        assert_eq!(
            Vector::from([100.0, 50.0])
                .add_clone(&Vector::from([20.0, 30.0]))
                .data,
            vec![120.0, 80.0]
        );
        assert_eq!(
            Vector::from([-5.0, 10.0])
                .add_clone(&Vector::from([5.0, -10.0]))
                .data,
            vec![0.0, 0.0]
        );
        assert_eq!(
            Vector::from([1.0, 2.0, 3.0, 4.0, 5.0])
                .add_clone(&Vector::from([5.0, 4.0, 3.0, 2.0, 1.0]))
                .data,
            vec![6.0, 6.0, 6.0, 6.0, 6.0]
        );

        // Exercise 00 - Subtract (vectors)
        assert_eq!(
            Vector::from([10.0, 10.0])
                .sub_clone(&Vector::from([5.0, 5.0]))
                .data,
            vec![5.0, 5.0]
        );
        assert_eq!(
            Vector::from([5.0, 0.0])
                .sub_clone(&Vector::from([0.0, 5.0]))
                .data,
            vec![5.0, -5.0]
        );
        assert_eq!(
            Vector::from([8.0, 8.0])
                .sub_clone(&Vector::from([8.0, 8.0]))
                .data,
            vec![0.0, 0.0]
        );
        assert_eq!(
            Vector::from([30.0, 15.0])
                .sub_clone(&Vector::from([10.0, 5.0]))
                .data,
            vec![20.0, 10.0]
        );
        assert_eq!(
            Vector::from([10.0, -10.0])
                .sub_clone(&Vector::from([5.0, -5.0]))
                .data,
            vec![5.0, -5.0]
        );
        assert_eq!(
            Vector::from([10.0, 9.0, 8.0, 7.0, 6.0])
                .sub_clone(&Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]))
                .data,
            vec![9.0, 7.0, 5.0, 3.0, 1.0]
        );

        // Exercise 00 - Multiply (vector scaling)
        assert_eq!(
            Vector::from([5.0, 10.0]).scl_clone(2.0).data,
            vec![10.0, 20.0]
        );
        assert_eq!(Vector::from([7.0, 7.0]).scl_clone(1.0).data, vec![7.0, 7.0]);
        assert_eq!(
            Vector::from([10.0, 20.0]).scl_clone(0.0).data,
            vec![0.0, 0.0]
        );
        assert_eq!(
            Vector::from([100.0, 200.0]).scl_clone(0.1).data,
            vec![10.0, 20.0]
        );
        assert_eq!(
            Vector::from([-5.0, -10.0]).scl_clone(-2.0).data,
            vec![10.0, 20.0]
        );
    }

    #[test]
    fn test_dot_product_scalar() {
        let u1 = Vector::from([5.0, 5.0]);
        let v1 = Vector::from([0.0, 0.0]);
        assert!((u1.dot(&v1).to_f32() - 0.0).abs() < EPSILON);

        let u2 = Vector::from([7.0, 0.0]);
        let v2 = Vector::from([0.0, 0.0]);
        assert!((u2.dot(&v2).to_f32() - 0.0).abs() < EPSILON);

        let u3 = Vector::from([3.0, 0.0]);
        let v3 = Vector::from([3.0, 0.0]);
        assert!((u3.dot(&v3).to_f32() - 9.0).abs() < EPSILON);

        let u4 = Vector::from([10.0, 0.0]);
        let v4 = Vector::from([0.0, 10.0]);
        assert!((u4.dot(&v4).to_f32() - 0.0).abs() < EPSILON);

        let u5 = Vector::from([5.0, 5.0]);
        let v5 = Vector::from([2.0, 2.0]);
        assert!((u5.dot(&v5).to_f32() - 20.0).abs() < EPSILON);

        let u6 = Vector::from([3.0, 5.0]);
        let v6 = Vector::from([2.0, 4.0]);
        assert!((u6.dot(&v6).to_f32() - 26.0).abs() < EPSILON); // 3*2 + 5*4 = 6 + 20 = 26
    }

    #[test]
    fn test_euclidean_norm_scalar() {
        assert!((Vector::from([0.0]).norm() - 0.0).abs() < EPSILON);
        assert!((Vector::from([5.0]).norm() - 5.0).abs() < EPSILON);
        assert!((Vector::from([0.0, 0.0]).norm() - 0.0).abs() < EPSILON);
        assert!((Vector::from([7.0, 0.0]).norm() - 7.0).abs() < EPSILON);
        assert!((Vector::from([3.0, 4.0]).norm() - 5.0).abs() < EPSILON);
        assert!((Vector::from([6.0, 8.0]).norm() - 10.0).abs() < EPSILON);
        assert!((Vector::from([-6.0, -8.0]).norm() - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_norm_scalar() {
        assert!((Vector::from([0.0]).norm_1() - 0.0).abs() < EPSILON);
        assert!((Vector::from([10.0]).norm_1() - 10.0).abs() < EPSILON);
        assert!((Vector::from([0.0, 0.0]).norm_1() - 0.0).abs() < EPSILON);
        assert!((Vector::from([12.0, 0.0]).norm_1() - 12.0).abs() < EPSILON);
        assert!((Vector::from([5.0, 7.0]).norm_1() - 12.0).abs() < EPSILON);
        assert!((Vector::from([8.0, 3.0]).norm_1() - 11.0).abs() < EPSILON);
        assert!((Vector::from([-8.0, -3.0]).norm_1() - 11.0).abs() < EPSILON);
    }

    #[test]
    fn test_supremum_norm_scalar() {
        // Test the function with several different vectors. Each time, the function
        // must return the component of the vector with the greatest value.
        assert!((Vector::from([0.0]).norm_inf() - 0.0).abs() < EPSILON);
        assert!((Vector::from([20.0]).norm_inf() - 20.0).abs() < EPSILON);
        assert!((Vector::from([-20.0]).norm_inf() - 20.0).abs() < EPSILON);
        assert!((Vector::from([0.0, 0.0]).norm_inf() - 0.0).abs() < EPSILON);
        assert!((Vector::from([15.0, 0.0]).norm_inf() - 15.0).abs() < EPSILON);
        assert!((Vector::from([7.0, 3.0]).norm_inf() - 7.0).abs() < EPSILON);
        assert!((Vector::from([9.0, 5.0]).norm_inf() - 9.0).abs() < EPSILON);
        assert!((Vector::from([-9.0, -5.0]).norm_inf() - 9.0).abs() < EPSILON);
        assert!((Vector::from([-15.0, 8.0, -12.0]).norm_inf() - 15.0).abs() < EPSILON);
        assert!((Vector::from([2.0, -7.0, 12.0]).norm_inf() - 12.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_add_sub_scl_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_v1 = Vector::from([Complex::new(2.0, 1.0), Complex::new(4.0, 3.0)]);
        let c_v2 = Vector::from([Complex::new(1.0, 5.0), Complex::new(2.0, 6.0)]);
        let c_result_add = c_v1.add_clone(&c_v2);
        assert!((c_result_add.data[0].re - 3.0).abs() < EPSILON);
        assert!((c_result_add.data[0].im - 6.0).abs() < EPSILON);
        assert!((c_result_add.data[1].re - 6.0).abs() < EPSILON);
        assert!((c_result_add.data[1].im - 9.0).abs() < EPSILON);

        let c_result_sub = c_v1.sub_clone(&c_v2);
        assert!((c_result_sub.data[0].re - 1.0).abs() < EPSILON);
        assert!((c_result_sub.data[0].im - -4.0).abs() < EPSILON);
        assert!((c_result_sub.data[1].re - 2.0).abs() < EPSILON);
        assert!((c_result_sub.data[1].im - -3.0).abs() < EPSILON);

        let c_result_scl = c_v1.scl_clone(Complex::new(3.0, 0.0)); // Scale by 3
        assert!((c_result_scl.data[0].re - 6.0).abs() < EPSILON);
        assert!((c_result_scl.data[0].im - 3.0).abs() < EPSILON);
        assert!((c_result_scl.data[1].re - 12.0).abs() < EPSILON);
        assert!((c_result_scl.data[1].im - 9.0).abs() < EPSILON);

        let c_result_scl_i = c_v1.scl_clone(Complex::new(0.0, 2.0)); // Scale by 2i
        assert!((c_result_scl_i.data[0].re - -2.0).abs() < EPSILON);
        assert!((c_result_scl_i.data[0].im - 4.0).abs() < EPSILON);
        assert!((c_result_scl_i.data[1].re - -6.0).abs() < EPSILON);
        assert!((c_result_scl_i.data[1].im - 8.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_complex() {
        let u = Vector::from([Complex::new(1.0, 2.0)]);
        let v = Vector::from([Complex::new(3.0, 4.0)]);

        let dot = u.dot(&v);

        assert!(
            (dot.re - 11.0).abs() < EPSILON,
            "Expected real part 11.0 but got {}",
            dot.re
        );
        assert!(
            (dot.im + 2.0).abs() < EPSILON,
            "Expected imag part -2.0 but got {}",
            dot.im
        );
    }

    #[test]
    fn test_euclidean_norm_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_v = Vector::from([Complex::new(6.0, 8.0)]);
        assert!((c_v.norm() - 10.0).abs() < EPSILON);
        let c_v2 = Vector::from([Complex::new(3.0, 0.0), Complex::new(0.0, 4.0)]); // norm = sqrt(3^2 + 4^2) = 5
        assert!((c_v2.norm() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_norm_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_v = Vector::from([Complex::new(6.0, 8.0)]);
        assert!((c_v.norm_1() - 10.0).abs() < EPSILON);
        let c_v2 = Vector::from([Complex::new(-2.0, 0.0), Complex::new(0.0, -3.0)]); // |-2| + |-3i| = 2 + 3 = 5
        assert!((c_v2.norm_1() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_supremum_norm_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_v = Vector::from([Complex::new(6.0, 8.0), Complex::new(-10.0, 0.0)]); // max(|6+8i|, |-10|) = max(10, 10) = 10
        assert!((c_v.norm_inf() - 10.0).abs() < EPSILON);
        let c_v2 = Vector::from([Complex::new(2.0, 3.0), Complex::new(-12.0, -5.0)]); // max(|2+3i|, |-12-5i|) = max(sqrt(13), sqrt(169)) = max(3.605, 13) = 13
        assert!((c_v2.norm_inf() - 13.0).abs() < EPSILON);
    }
}
