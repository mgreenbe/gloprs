use std::io::{self, Read};

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::{MpsFormat, parse_mps_with_format};

struct Fingerprint(u64);

impl Fingerprint {
    fn new() -> Self {
        Self(14_695_981_039_346_656_037)
    }

    fn byte(&mut self, value: u8) {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(1_099_511_628_211);
    }

    fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.byte(byte);
        }
    }

    fn float(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        for byte in value.bytes() {
            self.byte(byte);
        }
    }
}

fn main() {
    let format = match std::env::args().nth(1).as_deref() {
        None | Some("auto") => MpsFormat::AutoDetect,
        Some("free") => MpsFormat::Free,
        Some("fixed") => MpsFormat::Fixed,
        Some(_) => std::process::exit(2),
    };
    let mut source = String::new();
    io::stdin().read_to_string(&mut source).unwrap();
    let Ok((lp, _)) = parse_mps_with_format(&source, format) else {
        println!("error");
        return;
    };

    let mut hash = Fingerprint::new();
    hash.string(lp.name());
    hash.byte(u8::from(lp.is_maximization_problem()));
    hash.float(lp.objective_offset());
    hash.u64(lp.num_constraints().to_usize() as u64);
    hash.u64(lp.num_variables().to_usize() as u64);
    for index in 0..lp.num_constraints().to_usize() {
        let row = RowIndex::from_usize(index);
        hash.string(&lp.constraint_name(row));
        hash.float(lp.constraint_lower_bounds()[row]);
        hash.float(lp.constraint_upper_bounds()[row]);
    }
    for index in 0..lp.num_variables().to_usize() {
        let col = ColIndex::from_usize(index);
        hash.string(&lp.variable_name(col));
        hash.float(lp.objective_coefficients()[col]);
        hash.float(lp.variable_lower_bounds()[col]);
        hash.float(lp.variable_upper_bounds()[col]);
        hash.byte(u8::from(lp.is_variable_integer(col)));
        for entry in lp.matrix().column(col) {
            hash.u64(entry.row().to_usize() as u64);
            hash.float(entry.coefficient());
        }
    }
    println!("ok {:x}", hash.0);
    println!("{}", lp.dimension_string());
    println!("{}", lp.objective_stats_string());
    println!("{}", lp.bounds_stats_string());
    println!("{}", lp.problem_stats());
    print!(
        "pretty-problem-begin\n{}pretty-problem-end\n",
        lp.pretty_problem_stats()
    );
    println!("{}", lp.nonzero_stats());
    print!(
        "pretty-nonzero-begin\n{}pretty-nonzero-end\n",
        lp.pretty_nonzero_stats()
    );
    print!("dump-begin\n{}dump-end\n", lp.dump());
    println!("solution {}", lp.dump_solution(lp.objective_coefficients()));
}
