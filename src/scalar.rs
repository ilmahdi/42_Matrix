use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
pub trait Scalar:
    Display
    + Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + PartialEq
    + Default
    + Sum<Self>
{
    fn one() -> Self;
    fn to_f32(self) -> f32;
    fn mul_add(self, a: Self, b: Self) -> Self {
        a * b + self
    }
    fn conj(self) -> Self {
        self
    }
}

impl Scalar for f32 {
    fn one() -> Self {
        1.0
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        a.mul_add(b, self)
    }
}

impl Scalar for f64 {
    fn one() -> Self {
        1.0
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        a.mul_add(b, self)
    }
}

impl Scalar for i32 {
    fn one() -> Self {
        1
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for i64 {
    fn one() -> Self {
        1
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for i8 {
    fn one() -> Self {
        1
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for i16 {
    fn one() -> Self {
        1
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for isize {
    fn one() -> Self {
        1
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}
