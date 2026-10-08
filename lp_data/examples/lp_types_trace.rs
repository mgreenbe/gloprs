use std::io::{self, Read};

use lp_data::lp_types::{
    BitVec, ConstraintStatus, EPSILON, INFINITY, ProblemStatus, RANGE_MAX, RowIndex,
    VariableStatus, VariableType, VectorIndex, deterministic_time_for_fp_operations,
};

fn print_bits(name: &str, bits: &BitVec<RowIndex>) {
    print!("{name} {}", bits.len().value());
    for position in 0..bits.len().to_usize() {
        print!(
            "{}",
            i32::from(bits.contains(RowIndex::from_usize(position)))
        );
    }
    print!(" set");
    for position in bits.iter_ones() {
        print!(" {}", position.value());
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let size = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let count: usize = fields.next().unwrap().parse().unwrap();
    let other_size = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let other_count: usize = fields.next().unwrap().parse().unwrap();
    let query = RowIndex::new(fields.next().unwrap().parse().unwrap());
    let mut bits = BitVec::new(size);
    for _ in 0..count {
        bits.set(RowIndex::new(fields.next().unwrap().parse().unwrap()));
    }
    let mut other = BitVec::new(other_size);
    for _ in 0..other_count {
        other.set(RowIndex::new(fields.next().unwrap().parse().unwrap()));
    }

    print_bits("bits", &bits);
    println!("pair {}", i32::from(bits.are_one_of_two_bits_set(query)));
    let mut cleared = bits.clone();
    cleared.clear_two_bits(query);
    print_bits("clear_pair", &cleared);

    let mut content = BitVec::new(size);
    for position in (1..size.to_usize()).step_by(2) {
        content.set(RowIndex::from_usize(position));
    }
    content.set_content_from(&other);
    print_bits("content", &content);

    let mut intersection = bits.clone();
    intersection.intersection(&other);
    print_bits("intersection", &intersection);
    let mut united = bits.clone();
    united.union(&other);
    print_bits("union", &united);

    let mut resized = bits;
    resized.resize(RowIndex::from_usize(size.to_usize() / 2));
    resized.resize(RowIndex::from_usize(size.to_usize() + 5));
    print_bits("resized", &resized);

    for status in [
        ProblemStatus::Optimal,
        ProblemStatus::PrimalInfeasible,
        ProblemStatus::DualInfeasible,
        ProblemStatus::InfeasibleOrUnbounded,
        ProblemStatus::PrimalUnbounded,
        ProblemStatus::DualUnbounded,
        ProblemStatus::Init,
        ProblemStatus::PrimalFeasible,
        ProblemStatus::DualFeasible,
        ProblemStatus::Abnormal,
        ProblemStatus::InvalidProblem,
        ProblemStatus::Imprecise,
    ] {
        println!("problem {status}");
    }
    for variable_type in [
        VariableType::Unconstrained,
        VariableType::LowerBounded,
        VariableType::UpperBounded,
        VariableType::UpperAndLowerBounded,
        VariableType::FixedVariable,
    ] {
        println!("type {variable_type}");
    }
    for status in [
        VariableStatus::Basic,
        VariableStatus::FixedValue,
        VariableStatus::AtLowerBound,
        VariableStatus::AtUpperBound,
        VariableStatus::Free,
    ] {
        println!("status {status} {}", ConstraintStatus::from(status));
    }
    println!(
        "scalars {} {} {} {}",
        RANGE_MAX.to_bits(),
        INFINITY.to_bits(),
        EPSILON.to_bits(),
        deterministic_time_for_fp_operations(123_456_789).to_bits()
    );
}
