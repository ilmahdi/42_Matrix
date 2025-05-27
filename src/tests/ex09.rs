use matrix::Matrix;

pub fn run() {
    let u = Matrix::from([[1., 0.], [0., 1.]]);
    println!("{}", u.transpose());
    // [1.0, 0.0]
    // [0.0, 1.0]

    let u = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
    println!("{}", u.transpose());
    // [1.0, 4.0]
    // [2.0, 5.0]
    // [3.0, 6.0]

    let u = Matrix::from([[-2., 1., 0.], [1., -23., 6.], [0., 4., 4.]]);
    println!("{}", u.transpose());
    // [-2.0, 1.0, 0.0]
    // [1.0, -23.0, 4.0]
    // [0.0, 6.0, 4.0]
}
