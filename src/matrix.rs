use crate::scalar::Scalar;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{AddAssign, MulAssign, SubAssign};

/* -------------------------------------------------------------------------- */
//  Matrix Tests                                                              //
/* -------------------------------------------------------------------------- */
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<K> {
    data: Vec<Vec<K>>,
}

/* -------------------------------------------------------------------------- */
//  Traits Implementations                                                    //
/* -------------------------------------------------------------------------- */

impl<K: Clone, const N: usize, const M: usize> From<[[K; N]; M]> for Matrix<K> {
    fn from(array: [[K; N]; M]) -> Self {
        Self {
            data: array.iter().map(|row| Vec::from(row.clone())).collect(),
        }
    }
}

impl<K: Display> Display for Matrix<K> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        for row in &self.data {
            write!(f, "[ ")?;
            for (i, item) in row.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:^6.1}", item)?;
            }
            writeln!(f, " ]")?;
        }
        Ok(())
    }
}

impl<K: Scalar> AddAssign for Matrix<K> {
    fn add_assign(&mut self, other: Self) {
        self.add(&other);
    }
}

impl<K: Scalar> SubAssign for Matrix<K> {
    fn sub_assign(&mut self, other: Self) {
        self.sub(&other);
    }
}

impl<K: Scalar> MulAssign<K> for Matrix<K> {
    fn mul_assign(&mut self, scalar: K) {
        self.scl(scalar)
    }
}

/* -------------------------------------------------------------------------- */
//  Member Functions                                                          //
/* -------------------------------------------------------------------------- */

impl<K> Matrix<K> {
    pub fn shape(&self) -> (usize, usize) {
        (self.data.len(), self.data.first().map_or(0, |r| r.len()))
    }
}

impl<K> Matrix<K> {
    fn zip_apply<F>(&mut self, other: &Matrix<K>, mut func: F)
    where
        F: FnMut(&mut K, &K),
    {
        let (rows, cols) = self.shape();
        for i in 0..rows {
            for j in 0..cols {
                func(&mut self.data[i][j], &other.data[i][j]);
            }
        }
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn add(&mut self, other: &Matrix<K>) {
        self.zip_apply(other, |a: &mut K, b| *a += *b);
    }

    pub fn sub(&mut self, other: &Matrix<K>) {
        self.zip_apply(other, |a, b| *a -= *b);
    }

    pub fn scl(&mut self, scalar: K) {
        for row in &mut self.data {
            for item in row {
                *item *= scalar;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
//  Static Functions                                                          //
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
//  Unit Tests                                                                //
/* -------------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_add_sub_scl() {
        let mut m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);

        m1.add(&m2);
        assert_eq!(m1.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);

        m1.sub(&m2);
        assert_eq!(m1.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        m1.scl(2.0);
        assert_eq!(m1.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }
}
