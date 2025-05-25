mod vector;
mod matrix;
mod scalar;
mod complex;
mod shared;
mod tests {
    pub mod ex00;
    pub mod ex01;
    pub mod ex02;
    pub mod ex03;
    pub mod ex04;
    pub mod ex05;
    pub mod ex06;
    pub mod ex07;
    pub mod ex08;
    pub mod ex09;
    pub mod ex10;
    pub mod ex11;
    pub mod ex12;
    pub mod ex13;
    pub mod ex14;
    pub mod ex15;
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
        "test-ex06" => tests::ex06::run(),
        "test-ex07" => tests::ex07::run(),
        "test-ex08" => tests::ex08::run(),
        "test-ex09" => tests::ex09::run(),
        "test-ex10" => tests::ex10::run(),
        "test-ex11" => tests::ex11::run(),
        "test-ex12" => tests::ex12::run(),
        "test-ex13" => tests::ex13::run(),
        "test-ex14" => tests::ex14::run(),
        "test-ex15" => tests::ex15::run(),
        _ => eprintln!("\nUnknown test name: '{}'", args[1]),
    }
}
