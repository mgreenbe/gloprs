use lp_data::lp_data::LinearProgram;
use lp_data::lp_parser::{parse_constraint, parse_lp};
use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let (mode, body) = input.split_once('\n').unwrap();
    let body = body.strip_suffix('\n').unwrap_or(body);
    if mode == "C" {
        match parse_constraint(body) {
            Err(error) => println!("ERR {error}"),
            Ok(c) => {
                println!(
                    "OK {} {:x} {:x} {}",
                    c.name,
                    c.lower_bound.to_bits(),
                    c.upper_bound.to_bits(),
                    c.variable_names.len()
                );
                for (name, coefficient) in c.variable_names.iter().zip(c.coefficients) {
                    println!("{} {:x}", name, coefficient.to_bits());
                }
            }
        }
    } else {
        let mut lp = LinearProgram::new();
        let ok = parse_lp(body, &mut lp);
        print!("{}\n{}", if ok { "OK" } else { "ERR" }, lp.dump());
    }
}
