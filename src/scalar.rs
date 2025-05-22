use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
pub trait Scalar:
    Display
    + Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + PartialEq
    + Default
    + Sum<Self>
{
    // to be checked later
    fn mul_add(self, a: Self, b: Self) -> Self {
        a * b + self
    }
    fn to_f32(self) -> f32;
}

impl Scalar for f32 {
    fn mul_add(self, a: Self, b: Self) -> Self {
        a.mul_add(b, self)
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl Scalar for f64 {
    fn mul_add(self, a: Self, b: Self) -> Self {
        a.mul_add(b, self)
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Scalar for i32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for i64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for i8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for i16 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for isize {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
