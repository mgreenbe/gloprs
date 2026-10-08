use std::io::{self, Read};

use lp_data::lp_print_utils::{
    stringify, stringify_default, stringify_monomial, stringify_rational,
};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let count: usize = fields.next().unwrap().parse().unwrap();
    for _ in 0..count {
        let bits = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
        let value = f64::from_bits(bits);
        let number = stringify(value);
        let monomial = stringify_monomial(value, "x");
        let default_number = stringify_default(value);
        let rational = if value.is_finite() && value.abs() <= 1e12 {
            stringify_rational(value, f64::EPSILON)
        } else {
            "SKIP".to_owned()
        };
        println!("{} {number}", number.len());
        println!("{} {monomial}", monomial.len());
        println!("{} {default_number}", default_number.len());
        println!("{} {rational}", rational.len());
    }
}
