use matrix::{projection, Matrix};
use std::f32::consts::PI;

 fn print(mat: &Matrix<f32>) {
        for row in 0..4 {
            for col in 0..4 {
                print!("{:.6}", mat.data()[row][col]);
                if col < 3 {
                    print!(", ");
                }
            }
            println!();
        }
    }
pub fn run() {
    
    let fov = 120.0 * (PI / 180.0); 
    let ratio = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;

    let proj = projection(fov, ratio, near, far);
    print(&proj);
}
