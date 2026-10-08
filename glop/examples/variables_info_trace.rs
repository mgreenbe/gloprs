use std::io::{self, Read};

use glop::variables_info::VariablesInfo;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

fn parse_float(token: &str) -> f64 {
    match token {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => token.parse().unwrap(),
    }
}

fn emit(label: &str, info: &VariablesInfo) {
    let n = info.num_columns().to_usize();
    print!("{label}_types");
    for value in info.variable_types() {
        print!(" {value}");
    }
    println!();
    print!("{label}_statuses");
    for value in info.variable_statuses() {
        print!(" {value}");
    }
    println!();
    for (name, bits) in [
        ("increase", info.can_increase()),
        ("decrease", info.can_decrease()),
        ("relevant", info.relevance()),
        ("basic", info.is_basic()),
        ("not_basic", info.not_basic()),
        ("boxed", info.non_basic_boxed_variables()),
    ] {
        print!("{label}_{name}");
        for column in 0..n {
            print!(
                " {}",
                usize::from(bits.contains(ColIndex::from_usize(column)))
            );
        }
        println!();
    }
    println!(
        "{label}_entries {}",
        info.num_entries_in_relevant_columns().value()
    );
    print!("{label}_lower");
    for value in info.lower_bounds() {
        print!(" {value}");
    }
    println!();
    print!("{label}_upper");
    for value in info.upper_bounds() {
        print!(" {value}");
    }
    println!();
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let columns: usize = fields.next().unwrap().parse().unwrap();
    let num_entries: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(RowIndex::from_usize(rows), ColIndex::from_usize(columns));
    for _ in 0..num_entries {
        let row: usize = fields.next().unwrap().parse().unwrap();
        let column: usize = fields.next().unwrap().parse().unwrap();
        let value: f64 = fields.next().unwrap().parse().unwrap();
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .add_entry(RowIndex::from_usize(row), value);
    }
    matrix.clean_up();
    let mut lower = Vec::with_capacity(columns);
    let mut upper = Vec::with_capacity(columns);
    for _ in 0..columns {
        lower.push(parse_float(fields.next().unwrap()));
        upper.push(parse_float(fields.next().unwrap()));
    }
    let reduced_costs: Vec<_> = (0..columns)
        .map(|_| parse_float(fields.next().unwrap()))
        .collect();
    let num_basic: usize = fields.next().unwrap().parse().unwrap();
    let basic: Vec<usize> = (0..num_basic)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();

    let mut info = VariablesInfo::new(&matrix);
    info.load_bounds_and_return_true_if_unchanged(&lower, &upper);
    info.initialize_to_default_status();
    emit("default", &info);
    info.make_boxed_variable_relevant(false);
    emit("unboxed", &info);
    info.make_boxed_variable_relevant(true);
    for column in basic {
        info.update_to_basic_status(ColIndex::from_usize(column));
    }
    emit("basic", &info);
    info.transform_to_dual_phase_one_problem(1e-7, &reduced_costs);
    emit("phase1", &info);
    info.end_dual_phase_one(1e-7, &reduced_costs);
    emit("restored", &info);

    for column in 0..columns {
        let (lower, upper) = match column % 5 {
            0 => (f64::NEG_INFINITY, f64::INFINITY),
            1 => (-2.0, f64::INFINITY),
            2 => (f64::NEG_INFINITY, 3.0),
            3 => (-4.0, 5.0),
            _ => (6.0, 6.0),
        };
        info.mutable_lower_bounds()[column] = lower;
        info.mutable_upper_bounds()[column] = upper;
    }
    info.initialize_from_mutated_state();
    info.initialize_to_default_status();
    emit("mutated", &info);

    let variables = columns - rows;
    let variable_lower = &lower[..variables];
    let variable_upper = &upper[..variables];
    let constraint_lower: Vec<_> = upper[variables..].iter().map(|value| -value).collect();
    let constraint_upper: Vec<_> = lower[variables..].iter().map(|value| -value).collect();
    let mut structural = VariablesInfo::new(&matrix);
    let first_unchanged = structural.load_bounds_from_lp(
        variable_lower,
        variable_upper,
        &constraint_lower,
        &constraint_upper,
    );
    let second_unchanged = structural.load_bounds_from_lp(
        variable_lower,
        variable_upper,
        &constraint_lower,
        &constraint_upper,
    );
    println!(
        "structural_unchanged {} {}",
        usize::from(first_unchanged),
        usize::from(second_unchanged)
    );
    structural.initialize_to_default_status();
    emit("structural", &structural);
}
