use crate::scalar::Scalar;
use crate::vector::Vector;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::result::Result;

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

impl<K: Clone> From<Vec<Vec<K>>> for Matrix<K> {
    fn from(vec: Vec<Vec<K>>) -> Self {
        Self { data: vec }
    }
}

impl<K: Display> Display for Matrix<K> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
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
//  Member private Functions                                                  //
/* -------------------------------------------------------------------------- */

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
    fn swap_rows(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }
    fn scale_row(&mut self, i: usize, scalar: K) {
        for j in 0..self.data[i].len() {
            if self.data[i][j] != K::default() {
                self.data[i][j] *= scalar;
            }
        }
    }
    fn add_scaled_row(&mut self, i: usize, j: usize, scalar: K) {
        for k in 0..self.data[i].len() {
            // to check later
            let hol = self.data[j][k];
            self.data[i][k] += hol * scalar;
        }
    }
}

impl<K: Scalar> Matrix<K> {
    fn row_echelon_with_rank(&mut self, mut rank: Option<&mut usize>) {
        let (row, col) = self.shape();
        let mut pivot = (0, 0);

        'outer: while pivot.0 != row && pivot.1 != col {
            let mut i = pivot.0;
            while self.data[i][pivot.1] == K::default() {
                i += 1;
                if i == row {
                    pivot.1 += 1;
                    continue 'outer;
                }
            }
            if i != pivot.0 {
                self.swap_rows(pivot.0, i);
            }
            if self.data[pivot.0][pivot.1] != K::one() {
                self.scale_row(pivot.0, K::one() / self.data[pivot.0][pivot.1]);
            }
            for j in pivot.0 + 1..row {
                if self.data[j][pivot.1] != K::default() {
                    self.add_scaled_row(j, pivot.0, -self.data[j][pivot.1]);
                }
            }
            pivot.0 += 1;
            pivot.1 += 1;
            if let Some(ref mut val) = rank {
                **val += 1;
            }
        }
    }
}

impl<K: Scalar> Matrix<K> {
    fn row_echelon_v2(&mut self) -> u32 {
        let (n, _) = self.shape();
        let mut pivot = (0, 0);
        let mut swap_track = 0;

        'outer: while pivot.0 != n && pivot.1 != n {
            let mut i = pivot.0;
            while self.data[i][pivot.1] == K::default() {
                i += 1;
                if i == n {
                    pivot.1 += 1;
                    continue 'outer;
                }
            }
            if i != pivot.0 {
                self.swap_rows(pivot.0, i);
                swap_track += 1;
            }
            for j in pivot.0 + 1..n {
                if self.data[j][pivot.1] != K::default() {
                    self.add_scaled_row(
                        j,
                        pivot.0,
                        -self.data[j][pivot.1] / self.data[pivot.0][pivot.1],
                    );
                }
            }
            pivot.0 += 1;
            pivot.1 += 1;
        }
        swap_track
    }
}

impl<K: Scalar> Matrix<K> {
    fn reduced_row_echelon_for_inverse(mat_a: &mut Matrix<K>, mat_i: &mut Matrix<K>) -> bool {
        let (n, _) = mat_a.shape();
        let mut pivot = 0;

        while pivot != n {
            let mut i = pivot;
            while mat_a.data[i][pivot] == K::default() {
                i += 1;
                if i == n {
                    return false;
                }
            }
            if i != pivot {
                mat_a.swap_rows(pivot, i);
                mat_i.swap_rows(pivot, i);
            }
            let pivot_val = mat_a.data[pivot][pivot];
            if pivot_val == K::default() {
                return false; // Singular matrix
            }
            if pivot_val != K::one() {
                let inv = K::one() / pivot_val;
                mat_a.scale_row(pivot, inv);
                mat_i.scale_row(pivot, inv);
            }
            for j in 0..n {
                if j != pivot && mat_a.data[j][pivot] != K::default() {
                    let factor = -mat_a.data[j][pivot];
                    mat_a.add_scaled_row(j, pivot, factor);
                    mat_i.add_scaled_row(j, pivot, factor);
                }
            }
            pivot += 1;
        }
        return true;
    }
}

/* -------------------------------------------------------------------------- */
//  Member public Functions                                                   //
/* -------------------------------------------------------------------------- */

impl<K: Clone> Matrix<K> {
    pub fn new(vec: Vec<Vec<K>>) -> Self {
        Self { data: vec }
    }
}

impl<K> Matrix<K> {
    pub fn shape(&self) -> (usize, usize) {
        (self.data.len(), self.data.first().map_or(0, |r| r.len()))
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![K::default(); n]; n];
        for i in 0..n {
            data[i][i] = K::one();
        }
        Matrix::from(data)
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

impl<K: Scalar> Matrix<K> {
    pub fn mul_vec(&mut self, vec: &Vector<K>) -> Vector<K> {
        let (rows, cols) = self.shape();
        let mut res = vec![K::default(); rows];
        for i in 0..rows {
            for j in 0..cols {
                res[i] = res[i].mul_add(self.data[i][j], vec[j]);
            }
        }
        Vector::from(res)
    }
    pub fn mul_mat(&self, mat: &Matrix<K>) -> Matrix<K> {
        let (m, n) = self.shape();
        let (_, p) = mat.shape();
        let mut res = vec![vec![K::default(); p]; m];
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    res[i][j] = res[i][j].mul_add(self.data[i][k], mat.data[k][j]);
                }
            }
        }
        Matrix::from(res)
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn trace(&self) -> K {
        let (n, _) = self.shape();

        let mut res = K::default();
        for i in 0..n {
            res += self.data[i][i];
        }
        res
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn transpose(&self) -> Matrix<K> {
        let (m, n) = self.shape();
        let mut res = vec![vec![K::default(); n]; m];

        for i in 0..m {
            for j in 0..n {
                res[j][i] = self.data[i][j];
            }
        }
        Matrix::from(res)
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn row_echelon(&self) -> Matrix<K> {
        let mut res = self.clone();
        res.row_echelon_with_rank(None);
        res
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn determinant(&self) -> K {
        let mut res;
        let (n, _) = self.shape();
        let mut mat = self.clone();
        let swap_track = mat.row_echelon_v2();
        if swap_track % 2 == 1 {
            res = -K::one();
        } else {
            res = K::one();
        }

        for i in 0..n {
            res *= mat.data[i][i];
        }
        res
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn inverse(&self) -> Result<Matrix<K>, String> {
        let (n, _) = self.shape();
        let mut res = self.clone();
        let mut identity = Matrix::<K>::identity(n);
        if Matrix::<K>::reduced_row_echelon_for_inverse(&mut res, &mut identity) {
            return Ok(identity);
        } else {
            return Err("Matrix is singular".to_string());
        }
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn rank(&self) -> usize {
        let mut res = self.clone();
        let mut rank = 0;
        res.row_echelon_with_rank(Some(&mut rank));
        rank
    }
}

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
