use crate::vector::*;

fn print_test<T: std::fmt::Display + PartialEq>(got: &T, expected: &T) {
    println!("Got:\n{got}");
    println!("Expected:\n{expected}");
    if got == expected {
        println!("✅ Passed\n");
    } else {
        println!("❌ Failed\n");
    }
}

pub fn run() {
    println!("\nRunning linear_combination test...\n");

    let v1 = Vector::from([1.0, 2.0, 3.0]);
    let v2 = Vector::from([4.0, 5.0, 6.0]);
    let v3 = Vector::from([7.0, 8.0, 9.0]);

    let vectors = vec![v1.clone(), v2.clone(), v3.clone()];
    let coefs = vec![1.0, 0.0, -1.0];

    let result = linear_combination(&vectors, &coefs);
    let expected = Vector::from([-6.0, -6.0, -6.0]);

    print_test(&result, &expected);
}
