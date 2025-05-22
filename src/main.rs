mod matrix;
mod scalar;
mod shared;
mod vector;
mod tests {
    pub mod ex00;
    pub mod ex01;
    pub mod ex02;
    pub mod ex03;
    pub mod ex04;
    pub mod ex05;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo run [vector|matrix]");
        return;
    }

    match args[1].as_str() {
        "test-ex00" => tests::ex00::run(),
        "test-ex01" => tests::ex01::run(),
        "test-ex02" => tests::ex02::run(),
        "test-ex03" => tests::ex03::run(),
        "test-ex04" => tests::ex04::run(),
        "test-ex05" => tests::ex05::run(),
        _ => eprintln!("Unknown test name: '{}'", args[1]),
    }
}
