use std::io::{self, Read};

use lp_data::lp_data::LinearProgram;
use lp_data::lp_data_utils::LpScalingHelper;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};

fn print(name: &str, values: &[f64]) {
    print!("{name}");
    for value in values {
        print!(" {value:.17e}");
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let columns: usize = fields.next().unwrap().parse().unwrap();
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let mut lp = LinearProgram::new();
    for _ in 0..columns {
        lp.create_new_variable();
    }
    for _ in 0..rows {
        lp.create_new_constraint();
    }
    for _ in 0..entries {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let col = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        lp.set_coefficient(row, col, value);
    }
    for index in 0..columns {
        let col = ColIndex::from_usize(index);
        let objective = fields.next().unwrap().parse().unwrap();
        let lower = fields.next().unwrap().parse().unwrap();
        let upper = fields.next().unwrap().parse().unwrap();
        lp.set_objective_coefficient(col, objective);
        lp.set_variable_bounds(col, lower, upper);
    }
    for index in 0..rows {
        let row = RowIndex::from_usize(index);
        let lower = fields.next().unwrap().parse().unwrap();
        let upper = fields.next().unwrap().parse().unwrap();
        lp.set_constraint_bounds(row, lower, upper);
    }
    lp.clean_up();
    let mut helper = LpScalingHelper::new();
    helper.scale(&mut lp);
    print("objective", lp.objective_coefficients().as_slice());
    print("variable_lower", lp.variable_lower_bounds().as_slice());
    print("variable_upper", lp.variable_upper_bounds().as_slice());
    print("constraint_lower", lp.constraint_lower_bounds().as_slice());
    print("constraint_upper", lp.constraint_upper_bounds().as_slice());
    print!("matrix");
    for index in 0..columns {
        let col = ColIndex::from_usize(index);
        for entry in lp.sparse_column(col) {
            print!(
                " {} {} {:.17e}",
                col.value(),
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();
    println!(
        "factors {:.17e} {:.17e}",
        helper.bounds_scaling_factor(),
        helper.objective_scaling_factor()
    );
}
