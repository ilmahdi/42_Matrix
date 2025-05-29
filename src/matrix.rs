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

impl<K: Scalar> Matrix<K> {
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
    fn row_echelon_v2_for_rank(&mut self) -> usize {
        let (row, col) = self.shape();
        let mut pivot = (0, 0);
        let mut rank_track = 0;

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
            for j in pivot.0 + 1..row {
                if j != pivot.0 && self.data[j][pivot.1] != K::default() {
                    self.add_scaled_row(
                        j,
                        pivot.0,
                        -self.data[j][pivot.1] / self.data[pivot.0][pivot.1],
                    );
                }
            }
            pivot.0 += 1;
            pivot.1 += 1;
            rank_track += 1;
        }
        rank_track
    }
}

impl<K: Scalar> Matrix<K> {
    fn row_echelon_v2_for_determinant(&mut self) -> u32 {
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
    pub fn data(&self) -> &Vec<Vec<K>> {
        &self.data
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
    pub fn mul_vec(&self, vec: &Vector<K>) -> Vector<K> {
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
        let mut res = vec![vec![K::default(); m]; n];

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

        let (row, col) = res.shape();
        let mut pivot = (0, 0);

        'outer: while pivot.0 != row && pivot.1 != col {
            let mut i = pivot.0;
            while res.data[i][pivot.1] == K::default() {
                i += 1;
                if i == row {
                    pivot.1 += 1;
                    continue 'outer;
                }
            }
            if i != pivot.0 {
                res.swap_rows(pivot.0, i);
            }
            if res.data[pivot.0][pivot.1] != K::one() {
                res.scale_row(pivot.0, K::one() / res.data[pivot.0][pivot.1]);
            }
            for j in 0..row {
                if j != pivot.0 && res.data[j][pivot.1] != K::default() {
                    res.add_scaled_row(j, pivot.0, -res.data[j][pivot.1]);
                }
            }
            pivot.0 += 1;
            pivot.1 += 1;
        }
        res
    }
}

impl<K: Scalar> Matrix<K> {
    pub fn determinant(&self) -> K {
        let mut res;
        let (n, _) = self.shape();
        let mut mat = self.clone();
        let swap_track = mat.row_echelon_v2_for_determinant();
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
        res.row_echelon_v2_for_rank()
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

    // Helper functions for cloning and applying operations to avoid modifying original matrices in tests
    impl<K: Scalar> Matrix<K> {
        fn add_clone(&self, other: &Matrix<K>) -> Self {
            let mut cloned = self.clone();
            cloned.add(other);
            cloned
        }
        fn sub_clone(&self, other: &Matrix<K>) -> Self {
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

    // Helper for comparing matrices with floating point numbers
    fn assert_matrix_approx_eq<K: Scalar>(m1: &Matrix<K>, m2: &Matrix<K>, epsilon: f32) {
        let (rows1, cols1) = m1.shape();
        let (rows2, cols2) = m2.shape();
        assert_eq!(rows1, rows2, "Matrices have different number of rows");
        assert_eq!(cols1, cols2, "Matrices have different number of columns");

        for i in 0..rows1 {
            for j in 0..cols1 {
                assert!(
                    (m1.data[i][j].to_f32() - m2.data[i][j].to_f32()).abs() < epsilon,
                    "Matrices differ at [{}][{}]: left = {}, right = {}",
                    i,
                    j,
                    m1.data[i][j],
                    m2.data[i][j]
                );
            }
        }
    }

    #[test]
    fn test_matrix_add_sub_scl_scalar() {
        let mut m1 = Matrix::from([[10.0, 20.0], [30.0, 40.0]]);
        let m2 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);

        // Test addition
        m1.add(&m2);
        assert_eq!(*m1.data(), vec![vec![11.0, 22.0], vec![33.0, 44.0]]);

        // Test subtraction (using the new m1, which is now the result of the addition)
        m1.sub(&m2);
        assert_eq!(*m1.data(), vec![vec![10.0, 20.0], vec![30.0, 40.0]]);

        // Test scaling (using the new m1, which is back to its initial value)
        m1.scl(3.0);
        assert_eq!(*m1.data(), vec![vec![30.0, 60.0], vec![90.0, 120.0]]);

        // Exercise 00 - Add (matrices) - these remain as they were, demonstrating various addition cases
        assert_eq!(
            Matrix::from([[0.0, 0.0], [0.0, 0.0]])
                .add_clone(&Matrix::from([[0.0, 0.0], [0.0, 0.0]]))
                .data(),
            &vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );
        assert_eq!(
            Matrix::from([[1.0, 0.0], [0.0, 1.0]])
                .add_clone(&Matrix::from([[0.0, 0.0], [0.0, 0.0]]))
                .data(),
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
        assert_eq!(
            Matrix::from([[1.0, 1.0], [1.0, 1.0]])
                .add_clone(&Matrix::from([[1.0, 1.0], [1.0, 1.0]]))
                .data(),
            &vec![vec![2.0, 2.0], vec![2.0, 2.0]]
        );
        assert_eq!(
            Matrix::from([[21.0, 21.0], [21.0, 21.0]])
                .add_clone(&Matrix::from([[21.0, 21.0], [21.0, 21.0]]))
                .data(),
            &vec![vec![42.0, 42.0], vec![42.0, 42.0]]
        );

        // Exercise 00 - Subtract (matrices)
        assert_eq!(
            Matrix::from([[0.0, 0.0], [0.0, 0.0]])
                .sub_clone(&Matrix::from([[0.0, 0.0], [0.0, 0.0]]))
                .data(),
            &vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );
        assert_eq!(
            Matrix::from([[1.0, 0.0], [0.0, 1.0]])
                .sub_clone(&Matrix::from([[0.0, 0.0], [0.0, 0.0]]))
                .data(),
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
        assert_eq!(
            Matrix::from([[1.0, 1.0], [1.0, 1.0]])
                .sub_clone(&Matrix::from([[1.0, 1.0], [1.0, 1.0]]))
                .data(),
            &vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );
        assert_eq!(
            Matrix::from([[21.0, 21.0], [21.0, 21.0]])
                .sub_clone(&Matrix::from([[21.0, 21.0], [21.0, 21.0]]))
                .data(),
            &vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );

        // Exercise 00 - Multiply (matrix scaling)
        assert_eq!(
            Matrix::from([[0.0, 0.0], [0.0, 0.0]]).scl_clone(0.0).data(),
            &vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );
        assert_eq!(
            Matrix::from([[1.0, 0.0], [0.0, 1.0]]).scl_clone(1.0).data(),
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
        assert_eq!(
            Matrix::from([[1.0, 2.0], [3.0, 4.0]]).scl_clone(2.0).data(),
            &vec![vec![2.0, 4.0], vec![6.0, 8.0]]
        );
        assert_eq!(
            Matrix::from([[21.0, 21.0], [21.0, 21.0]])
                .scl_clone(0.5)
                .data(),
            &vec![vec![10.5, 10.5], vec![10.5, 10.5]]
        );
    }

    #[test]
    fn test_matrix_add_sub_scl_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m1 = Matrix::from([
            [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)],
            [Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)],
        ]);
        let c_m2 = Matrix::from([
            [Complex::new(5.0, 5.0), Complex::new(6.0, 6.0)],
            [Complex::new(7.0, 7.0), Complex::new(8.0, 8.0)],
        ]);
        let c_result_add = c_m1.add_clone(&c_m2);
        assert!((c_result_add.data[0][0].re - 6.0).abs() < EPSILON);
        assert!((c_result_add.data[0][0].im - 6.0).abs() < EPSILON);
        assert!((c_result_add.data[1][1].re - 12.0).abs() < EPSILON);
        assert!((c_result_add.data[1][1].im - 12.0).abs() < EPSILON);

        let c_result_sub = c_m1.sub_clone(&c_m2);
        assert!((c_result_sub.data[0][0].re - -4.0).abs() < EPSILON);
        assert!((c_result_sub.data[0][0].im - -4.0).abs() < EPSILON);
        assert!((c_result_sub.data[1][1].re - -4.0).abs() < EPSILON);
        assert!((c_result_sub.data[1][1].im - -4.0).abs() < EPSILON);

        let c_result_scl = c_m1.scl_clone(Complex::new(2.0, 0.0)); // Scale by 2
        assert!((c_result_scl.data[0][0].re - 2.0).abs() < EPSILON);
        assert!((c_result_scl.data[0][0].im - 2.0).abs() < EPSILON);
        assert!((c_result_scl.data[1][1].re - 8.0).abs() < EPSILON);
        assert!((c_result_scl.data[1][1].im - 8.0).abs() < EPSILON);
    }

    #[test]
    fn test_linear_transform_mul_vec_scalar() {
        let m1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        let v1 = Vector::from([1.0, 2.0]);
        assert_eq!(m1.mul_vec(&v1).data(), vec![0.0, 0.0]);

        let m2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        let v2 = Vector::from([5.0, 10.0]);
        assert_eq!(m2.mul_vec(&v2).data(), vec![5.0, 10.0]);

        let m3 = Matrix::from([[1.0, 1.0], [1.0, 1.0]]);
        let v3 = Vector::from([4.0, 2.0]);
        assert_eq!(m3.mul_vec(&v3).data(), vec![6.0, 6.0]);

        let m4 = Matrix::from([[2.0, 0.0], [0.0, 2.0]]);
        let v4 = Vector::from([2.0, 1.0]);
        assert_eq!(m4.mul_vec(&v4).data(), vec![4.0, 2.0]);

        let m5 = Matrix::from([[0.5, 0.0], [0.0, 0.5]]);
        let v5 = Vector::from([4.0, 2.0]);
        assert_eq!(m5.mul_vec(&v5).data(), vec![2.0, 1.0]);
    }

    #[test]
    fn test_linear_transform_mul_vec_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)],
        ]);
        let c_v = Vector::from([Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let c_result = c_m.mul_vec(&c_v);
        assert!((c_result.data()[0].re - -3.0).abs() < EPSILON);
        assert!((c_result.data()[0].im - 5.0).abs() < EPSILON);
        assert!((c_result.data()[1].re - 1.0).abs() < EPSILON);
        assert!((c_result.data()[1].im - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_mul_mat_scalar() {
        // Test basic multiplication
        let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
        let expected = Matrix::from([[19.0, 22.0], [43.0, 50.0]]);
        assert_eq!(m1.mul_mat(&m2), expected);

        // Test identity matrix multiplication
        let identity = Matrix::identity(2);
        assert_eq!(m1.mul_mat(&identity), m1);
        assert_eq!(identity.mul_mat(&m1), m1);

        // Test with zero matrix
        let zero_matrix = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        let expected_zero = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(m1.mul_mat(&zero_matrix), expected_zero);
        assert_eq!(zero_matrix.mul_mat(&m1), expected_zero);

        // Test 3x3 matrices
        let m3x3_a = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let m3x3_b = Matrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
        let expected_3x3 =
            Matrix::from([[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]);
        assert_eq!(m3x3_a.mul_mat(&m3x3_b), expected_3x3);
    }

    #[test]
    fn test_mul_mat_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m_a = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)],
        ]);
        let c_m_b = Matrix::from([
            [Complex::new(1.0, 1.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(1.0, 1.0)],
        ]);
        let c_expected = Matrix::from([
            [Complex::new(1.0, 1.0), Complex::new(-1.0, 1.0)],
            [Complex::new(-1.0, 1.0), Complex::new(1.0, 1.0)],
        ]);
        let c_result = c_m_a.mul_mat(&c_m_b);
        assert!((c_result.data[0][0].re - c_expected.data[0][0].re).abs() < EPSILON);
        assert!((c_result.data[0][0].im - c_expected.data[0][0].im).abs() < EPSILON);
        assert!((c_result.data[0][1].re - c_expected.data[0][1].re).abs() < EPSILON);
        assert!((c_result.data[0][1].im - c_expected.data[0][1].im).abs() < EPSILON);
        assert!((c_result.data[1][0].re - c_expected.data[1][0].re).abs() < EPSILON);
        assert!((c_result.data[1][0].im - c_expected.data[1][0].im).abs() < EPSILON);
        assert!((c_result.data[1][1].re - c_expected.data[1][1].re).abs() < EPSILON);
        assert!((c_result.data[1][1].im - c_expected.data[1][1].im).abs() < EPSILON);
    }

    #[test]
    fn test_trace_scalar() {
        let m1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        assert!((m1.trace().to_f32() - 0.0).abs() < EPSILON);

        let m2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert!((m2.trace().to_f32() - 2.0).abs() < EPSILON);

        let m3 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        assert!((m3.trace().to_f32() - 5.0).abs() < EPSILON);

        let m4 = Matrix::from([[8.0, -7.0], [4.0, 2.0]]);
        assert!((m4.trace().to_f32() - 10.0).abs() < EPSILON);

        let m5 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!((m5.trace().to_f32() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_trace_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)],
            [Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)],
        ]);
        let c_trace = c_m.trace(); // (1+i) + (4+4i) = 5+5i
        assert!((c_trace.re - 5.0).abs() < EPSILON);
        assert!((c_trace.im - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_transpose_scalar() {
        let m1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        let expected1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(m1.transpose(), expected1);

        let m2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        let expected2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(m2.transpose(), expected2);

        let m3 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let expected3 = Matrix::from([[1.0, 3.0], [2.0, 4.0]]);
        assert_eq!(m3.transpose(), expected3);

        let m4 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let expected4 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(m4.transpose(), expected4);

        let m5 = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let expected5 = Matrix::from(vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]]);
        assert_eq!(m5.transpose(), expected5);
    }

    #[test]
    fn test_transpose_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)],
            [Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)],
        ]);
        let c_expected = Matrix::from([
            [Complex::new(1.0, 1.0), Complex::new(3.0, 3.0)],
            [Complex::new(2.0, 2.0), Complex::new(4.0, 4.0)],
        ]);
        assert_eq!(c_m.transpose(), c_expected);
    }

    #[test]
    fn test_row_echelon_scalar() {
        let m1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        let expected1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        assert_matrix_approx_eq(&m1.row_echelon(), &expected1, EPSILON);

        let m2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        let expected2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert_matrix_approx_eq(&m2.row_echelon(), &expected2, EPSILON);

        let m3 = Matrix::from([[4.0, 2.0], [2.0, 1.0]]);
        let expected3 = Matrix::from([[1.0, 0.5], [0.0, 0.0]]);
        assert_matrix_approx_eq(&m3.row_echelon(), &expected3, EPSILON);

        let m4 = Matrix::from([[-7.0, 2.0], [4.0, 8.0]]);
        let expected4 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert_matrix_approx_eq(&m4.row_echelon(), &expected4, EPSILON);

        let m5 = Matrix::from([[1.0, 2.0], [4.0, 8.0]]);
        let expected5 = Matrix::from([[1.0, 2.0], [0.0, 0.0]]);
        assert_matrix_approx_eq(&m5.row_echelon(), &expected5, EPSILON);

        // Test 3x3 matrix
        let m6 = Matrix::from([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        let expected6 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_matrix_approx_eq(&m6.row_echelon(), &expected6, EPSILON);
    }

    #[test]
    fn test_row_echelon_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
        ]);
        let c_expected = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        ]);
        assert_matrix_approx_eq(&c_m.row_echelon(), &c_expected, EPSILON);
    }

    #[test]
    fn test_determinant_scalar() {
        let m1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        assert!((m1.determinant().to_f32() - 0.0).abs() < EPSILON);

        let m2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert!((m2.determinant().to_f32() - 1.0).abs() < EPSILON);

        let m3 = Matrix::from([[2.0, 0.0], [0.0, 2.0]]);
        assert!((m3.determinant().to_f32() - 4.0).abs() < EPSILON);

        let m4 = Matrix::from([[1.0, 1.0], [1.0, 1.0]]);
        assert!((m4.determinant().to_f32() - 0.0).abs() < EPSILON);

        let m5 = Matrix::from([[0.0, 1.0], [1.0, 0.0]]);
        assert!((m5.determinant().to_f32() - -1.0).abs() < EPSILON);

        let m6 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        assert!((m6.determinant().to_f32() - -2.0).abs() < EPSILON);

        let m7 = Matrix::from([[-7.0, 5.0], [4.0, 6.0]]);
        assert!((m7.determinant().to_f32() - -62.0).abs() < EPSILON);

        let m8 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!((m8.determinant().to_f32() - 1.0).abs() < EPSILON);

        // Test a 3x3 matrix with non-zero determinant
        let m9 = Matrix::from([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        assert!((m9.determinant().to_f32() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_determinant_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)],
        ]);
        let c_det = c_m.determinant();
        assert!((c_det.re - 2.0).abs() < EPSILON);
        assert!((c_det.im - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_inverse_scalar() {
        let m1 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        let expected1 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert_matrix_approx_eq(&m1.inverse().unwrap(), &expected1, EPSILON);

        let m2 = Matrix::from([[2.0, 0.0], [0.0, 2.0]]);
        let expected2 = Matrix::from([[0.5, 0.0], [0.0, 0.5]]);
        assert_matrix_approx_eq(&m2.inverse().unwrap(), &expected2, EPSILON);

        let m3 = Matrix::from([[0.5, 0.0], [0.0, 0.5]]);
        let expected3 = Matrix::from([[2.0, 0.0], [0.0, 2.0]]);
        assert_matrix_approx_eq(&m3.inverse().unwrap(), &expected3, EPSILON);

        let m4 = Matrix::from([[0.0, 1.0], [1.0, 0.0]]);
        let expected4 = Matrix::from([[0.0, 1.0], [1.0, 0.0]]);
        assert_matrix_approx_eq(&m4.inverse().unwrap(), &expected4, EPSILON);

        let m5 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let expected5 = Matrix::from([[-2.0, 1.0], [1.5, -0.5]]);
        assert_matrix_approx_eq(&m5.inverse().unwrap(), &expected5, EPSILON);

        let m6 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let expected6 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_matrix_approx_eq(&m6.inverse().unwrap(), &expected6, EPSILON);

        // Test non-invertible matrix
        let singular_matrix = Matrix::from([[1.0, 1.0], [1.0, 1.0]]);
        assert!(singular_matrix.inverse().is_err());

        // Test multiplication by inverse gives identity
        let m_test = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let inv_m_test = m_test.inverse().unwrap();
        let identity_result = m_test.mul_mat(&inv_m_test);
        let expected_identity = Matrix::identity(2);
        assert_matrix_approx_eq(&identity_result, &expected_identity, EPSILON);
    }

    #[test]
    fn test_inverse_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)],
        ]);
        let c_expected_inv = Matrix::from([
            [Complex::new(0.5, 0.0), Complex::new(0.0, -0.5)],
            [Complex::new(0.0, -0.5), Complex::new(0.5, 0.0)],
        ]);
        assert_matrix_approx_eq(&c_m.inverse().unwrap(), &c_expected_inv, EPSILON);

        let c_singular = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
        ]);
        assert!(c_singular.inverse().is_err());
    }

    #[test]
    fn test_rank_scalar() {
        let m1 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(m1.rank(), 0);

        let m2 = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(m2.rank(), 2);

        let m3 = Matrix::from([[2.0, 0.0], [0.0, 2.0]]);
        assert_eq!(m3.rank(), 2);

        let m4 = Matrix::from([[1.0, 1.0], [1.0, 1.0]]);
        assert_eq!(m4.rank(), 1);

        let m5 = Matrix::from([[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(m5.rank(), 2);

        let m6 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m6.rank(), 2);

        let m7 = Matrix::from([[-7.0, 5.0], [4.0, 6.0]]);
        assert_eq!(m7.rank(), 2);

        let m8 = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(m8.rank(), 3);

        // Test a 3x3 matrix with rank 2
        let m9 = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(m9.rank(), 2);
    }

    #[test]
    fn test_rank_complex() {
        // Test with complex numbers (Bonus Exercise 15)
        let c_m = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
        ]);
        assert_eq!(c_m.rank(), 1);
        let c_m2 = Matrix::from([
            [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
            [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)],
        ]);
        assert_eq!(c_m2.rank(), 2);
    }
}
