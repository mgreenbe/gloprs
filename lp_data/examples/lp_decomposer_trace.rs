use lp_data::lp_data::{LinearProgram, ModelVariableType};
use lp_data::lp_decomposer::LpDecomposer;
use lp_data::lp_types::{ColIndex, DenseRow, RowIndex, VectorIndex};
use std::io::{self, Read};

#[allow(clippy::many_single_char_names)]
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut f = input.split_whitespace();
    let n: usize = f.next().unwrap().parse().unwrap();
    let m: usize = f.next().unwrap().parse().unwrap();
    let count: usize = f.next().unwrap().parse().unwrap();
    let maximize: i32 = f.next().unwrap().parse().unwrap();
    let mut lp = LinearProgram::new();
    lp.set_maximization_problem(maximize != 0);
    for _ in 0..n {
        let name = f.next().unwrap();
        let integer: i32 = f.next().unwrap().parse().unwrap();
        let lower = f.next().unwrap().parse().unwrap();
        let upper = f.next().unwrap().parse().unwrap();
        let objective = f.next().unwrap().parse().unwrap();
        let c = lp.create_new_variable();
        lp.set_variable_name(c, name);
        if integer != 0 {
            lp.set_variable_type(c, ModelVariableType::Integer);
        }
        lp.set_variable_bounds(c, lower, upper);
        lp.set_objective_coefficient(c, objective);
    }
    for _ in 0..m {
        let name = f.next().unwrap();
        let lower = f.next().unwrap().parse().unwrap();
        let upper = f.next().unwrap().parse().unwrap();
        let r = lp.create_new_constraint();
        lp.set_constraint_name(r, name);
        lp.set_constraint_bounds(r, lower, upper);
    }
    for _ in 0..count {
        let r = RowIndex::new(f.next().unwrap().parse().unwrap());
        let c = ColIndex::new(f.next().unwrap().parse().unwrap());
        let value = f.next().unwrap().parse().unwrap();
        lp.set_coefficient(r, c, value);
    }
    let mut d = LpDecomposer::new();
    d.decompose(&lp);
    println!("problems {}", d.number_of_problems());
    let global = DenseRow::from_vec(
        (0..n)
            .map(|c| f64::from(u32::try_from(c).unwrap()) + 0.5)
            .collect(),
    );
    let mut locals = Vec::new();
    for p in 0..d.number_of_problems() {
        let local = d.extract_local_problem(p);
        println!(
            "P {p} {} {} {} {}",
            usize::from(local.is_maximization_problem()),
            local.num_variables().value(),
            local.num_constraints().value(),
            local.num_entries().value()
        );
        for c in 0..local.num_variables().to_usize() {
            let c = ColIndex::from_usize(c);
            println!(
                "V {} {} {:x} {:x} {:x}",
                local.variable_name(c),
                local.variable_type(c) as i8,
                local.variable_lower_bounds()[c].to_bits(),
                local.variable_upper_bounds()[c].to_bits(),
                local.objective_coefficients()[c].to_bits()
            );
            for e in local.sparse_column(c) {
                println!(
                    "E {} {} {:x}",
                    c.value(),
                    e.index().value(),
                    e.coefficient().to_bits()
                );
            }
        }
        for r in 0..local.num_constraints().to_usize() {
            let r = RowIndex::from_usize(r);
            println!(
                "R {} {:x} {:x}",
                local.constraint_name(r),
                local.constraint_lower_bounds()[r].to_bits(),
                local.constraint_upper_bounds()[r].to_bits()
            );
        }
        locals.push(d.extract_local_assignment(p, &global));
        print!("A");
        for value in locals.last().unwrap().as_slice() {
            print!(" {:x}", value.to_bits());
        }
        println!();
    }
    let aggregate = d.aggregate_assignments(&locals);
    print!("G");
    for value in aggregate.as_slice() {
        print!(" {:x}", value.to_bits());
    }
    println!();
}
