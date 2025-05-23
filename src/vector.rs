use crate::scalar::Scalar;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{AddAssign, MulAssign, SubAssign};

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

/* -------------------------------------------------------------------------- */
//  Member Functions                                                          //
/* -------------------------------------------------------------------------- */
impl<K: Scalar> Vector<K> {
    pub fn size(&self) -> usize {
        self.data.len()
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
            res = res.mul_add(a, b);
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
//  Static Functions                                                          //
/* -------------------------------------------------------------------------- */

pub fn linear_combination<K: Scalar>(u: &[Vector<K>], coefs: &[K]) -> Vector<K> {
    let mut result_data = vec![K::default(); u[0].size()];

    for (v, &c) in u.iter().zip(coefs) {
        for (res, &val) in result_data.iter_mut().zip(&v.data) {
            *res = res.mul_add(c, val);
        }
    }

    Vector { data: result_data }
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

/* -------------------------------------------------------------------------- */
//  Unit Tests                                                                //
/* -------------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add_sub_scl() {
        let mut v1 = Vector::from([1.0, 2.0, 3.0]);
        let v2 = Vector::from([4.0, 5.0, 6.0]);

        v1.add(&v2);
        assert_eq!(v1.data, vec![5.0, 7.0, 9.0]);

        v1.sub(&v2);
        assert_eq!(v1.data, vec![1.0, 2.0, 3.0]);

        v1.scl(2.0);
        assert_eq!(v1.data, vec![2.0, 4.0, 6.0]);
    }
}
