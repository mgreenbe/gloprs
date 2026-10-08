use std::io::{self, Read};

use lp_data::lp_data_utils::LpScalingHelper;
use lp_data::lp_types::{ColIndex, DenseRow, RowIndex, TypedVec, VectorIndex};
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};

fn read_values<I: VectorIndex>(
    fields: &mut std::str::SplitWhitespace<'_>,
    n: usize,
) -> TypedVec<I, f64> {
    TypedVec::from_vec(
        (0..n)
            .map(|_| fields.next().unwrap().parse().unwrap())
            .collect(),
    )
}

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
    let n: usize = fields.next().unwrap().parse().unwrap();
    let row_factors: Vec<f64> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let col_factors: Vec<f64> = (0..n)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let mut objective: DenseRow = read_values(&mut fields, n);
    let mut lower: DenseRow = read_values(&mut fields, n);
    let mut upper: DenseRow = read_values(&mut fields, n);
    let solve_values: TypedVec<RowIndex, f64> = read_values(&mut fields, n);
    let pattern_size: usize = fields.next().unwrap().parse().unwrap();
    let pattern: Vec<usize> = (0..pattern_size)
        .map(|_| fields.next().unwrap().parse().unwrap())
        .collect();
    let basis: Vec<ColIndex> = (0..n)
        .map(|_| ColIndex::new(fields.next().unwrap().parse().unwrap()))
        .collect();
    let selected = ColIndex::new(fields.next().unwrap().parse().unwrap());

    let mut helper = LpScalingHelper::new();
    helper.configure_from_factors(&row_factors, &col_factors);
    print!("scalar");
    for index in 0..n {
        let row = RowIndex::from_usize(index);
        let col = ColIndex::from_usize(index);
        let value = solve_values[row];
        print!(
            " {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
            helper.scale_variable_value(col, value),
            helper.scale_reduced_cost(col, value),
            helper.scale_dual_value(row, value),
            helper.scale_constraint_activity(row, value),
            helper.variable_scaling_factor(col),
            helper.variable_scaling_factor_with_slack(ColIndex::from_usize(n + index)),
        );
    }
    println!();

    let mut left = ScatteredRow::new(ColIndex::from_usize(n));
    for index in 0..n {
        left.values_mut()[ColIndex::from_usize(index)] = solve_values[RowIndex::from_usize(index)];
    }
    left.non_zeros_mut()
        .extend(pattern.iter().copied().map(ColIndex::from_usize));
    helper.unscale_unit_row_left_solve(selected, &mut left);
    print("left", left.values().as_slice());

    let mut right = ScatteredColumn::new(RowIndex::from_usize(n));
    right
        .values_mut()
        .as_mut_slice()
        .copy_from_slice(solve_values.as_slice());
    right
        .non_zeros_mut()
        .extend(pattern.iter().copied().map(RowIndex::from_usize));
    helper.unscale_column_right_solve(&basis, selected, &mut right);
    print("right", right.values().as_slice());

    helper.average_cost_scaling(&mut objective);
    print("objective", objective.as_slice());
    println!(
        "objective_factor {:.17e}",
        helper.objective_scaling_factor()
    );
    helper.contain_one_bound_scaling(&mut upper, &mut lower);
    print("lower", lower.as_slice());
    print("upper", upper.as_slice());
    println!("bound_factor {:.17e}", helper.bounds_scaling_factor());
}
