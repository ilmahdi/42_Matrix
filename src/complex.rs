use crate::scalar::Scalar;
use std::fmt::{Debug, Display, Formatter, Result};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    fn conjugate(&self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn norm_sq(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    fn norm(&self) -> f32 {
        self.norm_sq().powf(0.5)
    }
}

impl Default for Complex {
    fn default() -> Self {
        Complex::new(0.0, 0.0)
    }
}

impl Display for Complex {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if self.im >= 0.0 {
            write!(f, "{:.1} + {:.1}i", self.re, self.im)
        } else {
            write!(f, "{:.1} - {:.1}i", self.re, -self.im)
        }
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl MulAssign<f32> for Complex {
    fn mul_assign(&mut self, rhs: f32) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let den = other.norm_sq();
        Self::new(
            (self.re * other.re + self.im * other.im) / den,
            (self.im * other.re - self.re * other.im) / den,
        )
    }
}

impl DivAssign for Complex {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Complex::default(), |acc, x| acc + x)
    }
}

impl Scalar for Complex {
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }
    fn to_f32(self) -> f32 {
        self.norm()
    }
    fn conj(self) -> Self {
        self.conjugate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_complex_add() {
        let c1 = Complex::new(1.0, 2.0);
        let c2 = Complex::new(3.0, 4.0);
        let result = c1 + c2;
        assert!((result.re - 4.0).abs() < EPSILON);
        assert!((result.im - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_complex_sub() {
        let c1 = Complex::new(1.0, 2.0);
        let c2 = Complex::new(3.0, 4.0);
        let result = c1 - c2;
        assert!((result.re - (-2.0)).abs() < EPSILON);
        assert!((result.im - (-2.0)).abs() < EPSILON);
    }

    #[test]
    fn test_complex_mul() {
        let c1 = Complex::new(1.0, 2.0);
        let c2 = Complex::new(3.0, 4.0);
        let result = c1 * c2;
        assert!((result.re - (-5.0)).abs() < EPSILON);
        assert!((result.im - 10.0).abs() < EPSILON);

        let i = Complex::new(0.0, 1.0);
        let i_sq = i * i;
        assert!((i_sq.re - (-1.0)).abs() < EPSILON);
        assert!((i_sq.im - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_complex_div() {
        let c1 = Complex::new(1.0, 2.0);
        let c2 = Complex::new(3.0, 4.0);
        let result = c1 / c2;
        assert!((result.re - 0.44).abs() < EPSILON);
        assert!((result.im - 0.08).abs() < EPSILON);
    }

    #[test]
    fn test_complex_neg() {
        let c = Complex::new(1.0, -2.0);
        let result = -c;
        assert!((result.re - (-1.0)).abs() < EPSILON);
        assert!((result.im - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_complex_default_one() {
        let default_c = Complex::default();
        assert!((default_c.re - 0.0).abs() < EPSILON);
        assert!((default_c.im - 0.0).abs() < EPSILON);

        let one_c = Complex::one();
        assert!((one_c.re - 1.0).abs() < EPSILON);
        assert!((one_c.im - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_complex_mul_add() {
        let c1 = Complex::new(1.0, 1.0);
        let a = Complex::new(2.0, 2.0);
        let b = Complex::new(3.0, 3.0);
        let result = c1.mul_add(a, b);
        assert!((result.re - 1.0).abs() < EPSILON);
        assert!((result.im - 13.0).abs() < EPSILON);
    }
}
