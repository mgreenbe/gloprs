use std::io::{self, Read};

use lp_data::lp_data::LinearProgram;
use lp_data::sol_reader::parse_sol_string;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let (header, solution) = input.split_once('\n').unwrap();
    let n: usize = header.parse().unwrap();
    let mut lines = solution.splitn(n + 1, '\n');
    let mut model = LinearProgram::new();
    for _ in 0..n {
        let name = lines.next().unwrap();
        let column = model.create_new_variable();
        if name != "<empty>" {
            model.set_variable_name(column, name);
        }
    }
    let solution = lines.next().unwrap_or("");
    match parse_sol_string(solution, &model) {
        Ok(values) => {
            print!("OK");
            for value in values.as_slice() {
                print!(" {:x}", value.to_bits());
            }
            println!();
        }
        Err(error) => println!("ERR {error}"),
    }
}
