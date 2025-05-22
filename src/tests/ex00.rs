use crate::matrix::*;
use crate::vector::*;

fn print_test<T: std::fmt::Display + PartialEq>(label: &str, actual: &T, expected: &T) {
    println!("--- {} ---", label);
    println!("Actual:\n{}", actual);
    println!("Expected:\n{}", expected);
    if actual == expected {
        println!("\x1b[32m✔ Passed\x1b[0m\n");
    } else {
        println!("\x1b[31m✘ Failed\x1b[0m\n");
    }
}

pub fn run() {
    println!("\nRunning vector and matrix tests...\n");

    println!("==== VECTOR TESTS ====");
    let mut v1 = Vector::from([1.0, 2.0, 3.0]);
    let v2 = Vector::from([4.0, 5.0, 6.0]);

    println!("Original v1:\n{}\n", v1);
    println!("Vector to add (v2):\n{}\n", v2);

    v1.add(&v2);
    let expected_add = Vector::from([5.0, 7.0, 9.0]);
    print_test("v1.add(v2)", &v1, &expected_add);

    v1.sub(&v2);
    let expected_sub = Vector::from([1.0, 2.0, 3.0]);
    print_test("v1.sub(v2)", &v1, &expected_sub);

    v1.scl(3.0);
    let expected_scl = Vector::from([3.0, 6.0, 9.0]);
    print_test("v1.scl(3.0)", &v1, &expected_scl);

    println!("==== MATRIX TESTS ====");
    let mut m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);

    println!("Original m1:\n{}\n", m1);
    println!("Matrix to add (m2):\n{}\n", m2);

    m1.add(&m2);
    let expected_add = Matrix::from([[6.0, 8.0], [10.0, 12.0]]);
    print_test("m1.add(m2)", &m1, &expected_add);

    m1.sub(&m2);
    let expected_sub = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    print_test("m1.sub(m2)", &m1, &expected_sub);

    m1.scl(0.5);
    let expected_scl = Matrix::from([[0.5, 1.0], [1.5, 2.0]]);
    print_test("m1.scl(0.5)", &m1, &expected_scl);
}
