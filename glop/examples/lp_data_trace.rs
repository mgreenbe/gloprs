use std::io::{self, Read};

use lp_data::lp_data::{CostScalingAlgorithm, LinearProgram, ModelVariableType};
use lp_data::lp_types::{
    ColIndex, DenseBooleanColumn, DenseBooleanRow, DenseColumn, DenseRow, RowIndex, TypedVec,
    VectorIndex,
};
use lp_data::permutation::{ColumnPermutation, RowPermutation};
use lp_data::sparse::SparseMatrix;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let rows: usize = fields.next().unwrap().parse().unwrap();
    let columns: usize = fields.next().unwrap().parse().unwrap();
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let maximize: usize = fields.next().unwrap().parse().unwrap();
    let offset = fields.next().unwrap().parse().unwrap();
    let scale = fields.next().unwrap().parse().unwrap();
    let tolerance = fields.next().unwrap().parse().unwrap();
    let mut lp = LinearProgram::new();
    for _ in 0..columns {
        lp.create_new_variable();
    }
    for _ in 0..rows {
        lp.create_new_constraint();
    }
    lp.set_maximization_problem(maximize != 0);
    lp.set_objective_offset(offset);
    lp.set_objective_scaling_factor(scale);
    for position in 0..columns {
        let variable_type = match fields.next().unwrap().parse::<usize>().unwrap() {
            0 => ModelVariableType::Continuous,
            1 => ModelVariableType::Integer,
            _ => ModelVariableType::ImpliedInteger,
        };
        let lower = fields.next().unwrap().parse().unwrap();
        let upper = fields.next().unwrap().parse().unwrap();
        let objective = fields.next().unwrap().parse().unwrap();
        let column = ColIndex::from_usize(position);
        lp.set_variable_type(column, variable_type);
        lp.set_variable_bounds(column, lower, upper);
        lp.set_objective_coefficient(column, objective);
    }
    for position in 0..rows {
        let lower = fields.next().unwrap().parse().unwrap();
        let upper = fields.next().unwrap().parse().unwrap();
        lp.set_constraint_bounds(RowIndex::from_usize(position), lower, upper);
    }
    for _ in 0..entries {
        let row = RowIndex::new(fields.next().unwrap().parse().unwrap());
        let column = ColIndex::new(fields.next().unwrap().parse().unwrap());
        let value = fields.next().unwrap().parse().unwrap();
        lp.set_coefficient(row, column, value);
    }
    let mut solution = DenseRow::from_vec(
        (0..columns)
            .map(|_| fields.next().unwrap().parse().unwrap())
            .collect(),
    );
    lp.clean_up();
    print!("types");
    for position in 0..columns {
        let column = ColIndex::from_usize(position);
        print!(
            " {} {}",
            usize::from(lp.is_variable_integer(column)),
            usize::from(lp.is_variable_binary(column))
        );
    }
    println!();
    println!(
        "feasibility {} {} {} {}",
        usize::from(lp.solution_is_within_variable_bounds(&solution, tolerance)),
        usize::from(lp.solution_is_lp_feasible(&solution, tolerance)),
        usize::from(lp.solution_is_integer(&solution, tolerance)),
        usize::from(lp.solution_is_mip_feasible(&solution, tolerance))
    );
    println!(
        "names {} {} {} {}",
        lp.variable_name(ColIndex::new(0)),
        lp.variable_name(ColIndex::from_usize(columns + 3)),
        lp.constraint_name(RowIndex::new(0)),
        lp.constraint_name(RowIndex::from_usize(rows + 3))
    );
    let scaled = lp.apply_objective_scaling_and_offset(2.25);
    print!(
        "objective {scaled:.17e} {:.17e}",
        lp.remove_objective_scaling_and_offset(scaled)
    );
    for position in 0..columns {
        print!(
            " {:.17e}",
            lp.objective_coefficient_for_minimization(ColIndex::from_usize(position))
        );
    }
    println!();
    println!(
        "integral_bounds {} {}",
        usize::from(lp.bounds_of_integer_variables_are_integer(tolerance)),
        usize::from(lp.bounds_of_integer_constraints_are_integer(tolerance))
    );

    lp.add_slack_variables_where_necessary(true);
    solution.resize(lp.num_variables(), 0.0);
    lp.compute_slack_variable_values(&mut solution);
    print!(
        "slacks {} {} {} {}",
        lp.first_slack_variable().unwrap().value(),
        lp.num_variables().value(),
        lp.num_entries().value(),
        usize::from(lp.is_in_equation_form())
    );
    for position in 0..rows {
        let slack = lp.slack_variable(RowIndex::from_usize(position)).unwrap();
        print!(
            " {} {:.17e} {:.17e} {:.17e} {}",
            slack.value(),
            solution[slack],
            lp.variable_lower_bounds()[slack],
            lp.variable_upper_bounds()[slack],
            usize::from(lp.is_variable_integer(slack))
        );
    }
    println!();
    lp.delete_slack_variables();
    print!(
        "restored {} {} {}",
        lp.num_variables().value(),
        lp.num_constraints().value(),
        lp.num_entries().value()
    );
    for position in 0..rows {
        let row = RowIndex::from_usize(position);
        print!(
            " {:.17e} {:.17e}",
            lp.constraint_lower_bounds()[row],
            lp.constraint_upper_bounds()[row]
        );
    }
    println!();

    let mut appended = SparseMatrix::new();
    appended.populate_from_zero(RowIndex::new(2), lp.num_variables());
    for position in 0..columns {
        let column = ColIndex::from_usize(position);
        if position % 2 == 0 {
            appended
                .mutable_column(column)
                .set_coefficient(RowIndex::new(0), f64::from(column.value()) + 0.5);
        }
        if position % 3 == 1 {
            appended
                .mutable_column(column)
                .set_coefficient(RowIndex::new(1), 1.25 - f64::from(column.value()));
        }
    }
    let appended_lower = DenseColumn::from_vec(vec![-7.0, -3.0]);
    let appended_upper = DenseColumn::from_vec(vec![4.0, 9.0]);
    let appended_names = TypedVec::from_vec(vec!["appended0".to_owned(), "appended1".to_owned()]);
    lp.add_constraints(&appended, &appended_lower, &appended_upper, &appended_names);
    print!(
        "appended {} {} {}",
        lp.num_constraints().value(),
        lp.num_entries().value(),
        usize::from(lp.is_cleaned_up())
    );
    let appended_transpose = lp.transpose_sparse_matrix();
    for position in rows..rows + 2 {
        let row = RowIndex::from_usize(position);
        print!(
            " {:.17e} {:.17e}",
            lp.constraint_lower_bounds()[row],
            lp.constraint_upper_bounds()[row]
        );
        for entry in appended_transpose.column(ColIndex::from_usize(position)) {
            print!(
                " {position}:{}:{:.17e}",
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    drop(appended_transpose);
    println!();

    let mut dual = LinearProgram::new();
    let duplicated_rows = dual.populate_from_dual(&lp);
    print!(
        "dual {} {} {} {} {:.17e} {:.17e}",
        dual.num_constraints().value(),
        dual.num_variables().value(),
        dual.num_entries().value(),
        usize::from(dual.is_maximization_problem()),
        dual.objective_offset(),
        dual.objective_scaling_factor()
    );
    for &column in &duplicated_rows {
        print!(" {}", column.value());
    }
    for position in 0..dual.num_constraints().to_usize() {
        let row = RowIndex::from_usize(position);
        print!(
            " {:.17e} {:.17e}",
            dual.constraint_lower_bounds()[row],
            dual.constraint_upper_bounds()[row]
        );
    }
    for position in 0..dual.num_variables().to_usize() {
        let column = ColIndex::from_usize(position);
        print!(
            " {:.17e} {:.17e} {:.17e}",
            dual.variable_lower_bounds()[column],
            dual.variable_upper_bounds()[column],
            dual.objective_coefficients()[column]
        );
        for entry in dual.sparse_column(column) {
            print!(
                " {position}:{}:{:.17e}",
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();

    print!("scaling");
    for method in [
        CostScalingAlgorithm::NoCostScaling,
        CostScalingAlgorithm::ContainOneCostScaling,
        CostScalingAlgorithm::MeanCostScaling,
        CostScalingAlgorithm::MedianCostScaling,
    ] {
        let mut scaled = LinearProgram::new();
        scaled.populate_from_linear_program(&lp);
        for position in 0..scaled.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            scaled
                .set_objective_coefficient(column, 100.0 * scaled.objective_coefficients()[column]);
        }
        let factor = scaled.scale_objective(method);
        print!(
            " {factor:.17e} {:.17e} {:.17e}",
            scaled.objective_scaling_factor(),
            scaled.objective_offset()
        );
        for &coefficient in scaled.objective_coefficients() {
            print!(" {coefficient:.17e}");
        }
    }
    let mut bound_scaled = LinearProgram::new();
    bound_scaled.populate_from_linear_program(&lp);
    for position in 0..bound_scaled.num_variables().to_usize() {
        let column = ColIndex::from_usize(position);
        bound_scaled.set_variable_bounds(
            column,
            100.0 * bound_scaled.variable_lower_bounds()[column],
            100.0 * bound_scaled.variable_upper_bounds()[column],
        );
    }
    for position in 0..bound_scaled.num_constraints().to_usize() {
        let row = RowIndex::from_usize(position);
        bound_scaled.set_constraint_bounds(
            row,
            100.0 * bound_scaled.constraint_lower_bounds()[row],
            100.0 * bound_scaled.constraint_upper_bounds()[row],
        );
    }
    let bound_factor = bound_scaled.scale_bounds();
    println!(
        " {bound_factor:.17e} {:.17e} {:.17e} {} {}",
        bound_scaled.objective_scaling_factor(),
        bound_scaled.objective_offset(),
        usize::from(bound_scaled.is_valid(1e100)),
        usize::from(bound_scaled.is_valid(1.0))
    );

    let row_permutation = RowPermutation::from_vec(
        (0..lp.num_constraints().to_usize())
            .map(|position| RowIndex::from_usize(lp.num_constraints().to_usize() - 1 - position))
            .collect(),
    );
    let column_permutation = ColumnPermutation::from_vec(
        (0..lp.num_variables().to_usize())
            .map(|position| ColIndex::from_usize((position + 1) % lp.num_variables().to_usize()))
            .collect(),
    );
    let mut permuted = LinearProgram::new();
    permuted.populate_from_permuted_linear_program(&lp, &row_permutation, &column_permutation);
    print!(
        "permuted {} {} {}",
        permuted.num_constraints().value(),
        permuted.num_variables().value(),
        permuted.num_entries().value()
    );
    for position in 0..permuted.num_variables().to_usize() {
        let column = ColIndex::from_usize(position);
        let variable_type = match permuted.variable_types()[column] {
            ModelVariableType::Continuous => 0,
            ModelVariableType::Integer => 1,
            ModelVariableType::ImpliedInteger => 2,
        };
        print!(
            " {:.17e} {:.17e} {:.17e} {variable_type}",
            permuted.variable_lower_bounds()[column],
            permuted.variable_upper_bounds()[column],
            permuted.objective_coefficients()[column]
        );
        for entry in permuted.sparse_column(column) {
            print!(
                " {position}:{}:{:.17e}",
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();
    let mut variables_only = LinearProgram::new();
    variables_only.populate_from_linear_program_variables(&permuted);
    println!(
        "variables_only {} {} {} {:.17e}",
        variables_only.num_constraints().value(),
        variables_only.num_variables().value(),
        variables_only.num_entries().value(),
        variables_only.objective_scaling_factor()
    );

    let mut deleted_columns = DenseBooleanRow::filled(ColIndex::from_usize(columns), false);
    for position in (1..columns).step_by(3) {
        deleted_columns[ColIndex::from_usize(position)] = true;
    }
    let mut deleted_rows = DenseBooleanColumn::filled(RowIndex::from_usize(rows), false);
    for position in (2..rows).step_by(4) {
        deleted_rows[RowIndex::from_usize(position)] = true;
    }
    lp.delete_columns(&deleted_columns);
    lp.delete_rows(&deleted_rows);
    print!(
        "deleted {} {} {}",
        lp.num_variables().value(),
        lp.num_constraints().value(),
        lp.num_entries().value()
    );
    for position in 0..lp.num_variables().to_usize() {
        let column = ColIndex::from_usize(position);
        let variable_type = match lp.variable_types()[column] {
            ModelVariableType::Continuous => 0,
            ModelVariableType::Integer => 1,
            ModelVariableType::ImpliedInteger => 2,
        };
        print!(
            " {:.17e} {:.17e} {:.17e} {variable_type}",
            lp.variable_lower_bounds()[column],
            lp.variable_upper_bounds()[column],
            lp.objective_coefficients()[column]
        );
        for entry in lp.sparse_column(column) {
            print!(
                " {position}:{}:{:.17e}",
                entry.index().value(),
                entry.coefficient()
            );
        }
    }
    println!();
}
