use matrix::Complex;
use matrix::Matrix;
use matrix::Vector;
use matrix::angle_cos;

pub fn run() {
    let c = |re, im| Complex::new(re, im);

    let mut u = Vector::from([c(2., 3.), c(4., -1.)]);
    let v = Vector::from([c(1., 1.), c(0., 2.)]);
    u.add(&v);
    assert_eq!(u, Vector::from([c(3., 4.), c(4., 1.)]));

    let mut u = Vector::from([c(2., 3.), c(4., -1.)]);
    let v = Vector::from([c(1., 1.), c(0., 2.)]);
    u.sub(&v);
    assert_eq!(u, Vector::from([c(1., 2.), c(4., -3.)]));

    let mut u = Vector::from([c(1., 1.), c(2., 0.)]);
    u.scl(c(2., 0.));
    assert_eq!(u, Vector::from([c(2., 2.), c(4., 0.)]));

    let mut u = Matrix::from([[c(1., 2.), c(3., 4.)], [c(5., -1.), c(2., 2.)]]);
    let v = Matrix::from([[c(0., -2.), c(1., 1.)], [c(-1., 1.), c(2., -2.)]]);
    u.add(&v);
    assert_eq!(
        u,
        Matrix::from([[c(1., 0.), c(4., 5.)], [c(4., 0.), c(4., 0.)]])
    );

    let mut u = Matrix::from([[c(1., 2.), c(3., 4.)], [c(5., -1.), c(2., 2.)]]);
    let v = Matrix::from([[c(0., -2.), c(1., 1.)], [c(-1., 1.), c(2., -2.)]]);
    u.sub(&v);
    assert_eq!(
        u,
        Matrix::from([[c(1., 4.), c(2., 3.)], [c(6., -2.), c(0., 4.)]])
    );

    let mut u = Matrix::from([[c(1., 1.), c(0., 0.)], [c(0., 1.), c(1., 0.)]]);
    u.scl(c(2., 0.));
    assert_eq!(
        u,
        Matrix::from([[c(2., 2.), c(0., 0.)], [c(0., 2.), c(2., 0.)]])
    );

    // let e1 = Vector::from([c(1., 0.), c(0., 0.), c(0., 0.)]);
    // let e2 = Vector::from([c(0., 0.), c(1., 0.), c(0., 0.)]);
    // let e3 = Vector::from([c(0., 0.), c(0., 0.), c(1., 0.)]);
    // let result = linear_combination(&[e1, e2, e3], &[c(10., 0.), c(-2., 0.), c(0.5, 0.)]);
    // assert_eq!(result, Vector::from([c(10., 0.), c(-2., 0.), c(0.5, 0.)]));

    // let v1 = Vector::from([c(1., 1.), c(2., -1.), c(3., 0.)]);
    // let v2 = Vector::from([c(0., 0.), c(10., 0.), c(-100., 0.)]);
    // let result = linear_combination(&[v1, v2], &[c(10., 0.), c(-2., 0.)]);
    // assert_eq!(result, Vector::from([c(10., 10.), c(0., -10.), c(230., 0.)]));

    let u = Vector::from([c(1., 0.), c(0., 1.)]);
    let v = Vector::from([c(1., 0.), c(0., -1.)]);
    let dot = u.dot(&v);
    assert_eq!(dot, c(1., 0.) + c(0., 1.) * c(0., -1.)); // 1 + 1 = 2
    assert_eq!(dot, c(2., 0.));

    let u = Vector::from([c(3., 4.)]);
    assert_eq!(u.norm_1(), 7.0);
    assert!((u.norm() - 5.0).abs() < 1e-10);
    assert_eq!(u.norm_inf(), 5.0);

    let u = Vector::from([c(1., 0.), c(0., 1.)]);
    let v = Vector::from([c(0., 1.), c(1., 0.)]);
    let angle_cos_result = angle_cos(&u, &v);
    assert!(angle_cos_result.abs() < 1e-10); // Cosine ~ 0

    let u = Matrix::from([[c(1., 0.), c(0., 0.)], [c(0., 0.), c(1., 0.)]]);
    assert_eq!(u.trace(), c(2., 0.));

    let u = Matrix::from([[c(1., 2.), c(3., 4.)], [c(5., 6.), c(7., 8.)]]);
    let t = u.transpose();
    assert_eq!(
        t,
        Matrix::from([[c(1., 2.), c(5., 6.)], [c(3., 4.), c(7., 8.)]])
    );
}
