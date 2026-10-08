use std::io::{self, Read};

use glop::basis_representation::BasisRepresentation;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let n: usize = fields.next().unwrap().parse().unwrap();
    let structural_columns: usize = fields.next().unwrap().parse().unwrap();
    let entries: usize = fields.next().unwrap().parse().unwrap();
    let updates: usize = fields.next().unwrap().parse().unwrap();
    let refactorization_period: usize = fields.next().unwrap().parse().unwrap();
    let dynamically_adjust: usize = fields.next().unwrap().parse().unwrap();
    let use_middle_product: usize = fields.next().unwrap().parse().unwrap();
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(
        RowIndex::from_usize(n),
        ColIndex::from_usize(structural_columns + n),
    );
    for _ in 0..entries {
        let row: usize = fields.next().unwrap().parse().unwrap();
        let column: usize = fields.next().unwrap().parse().unwrap();
        let value: f64 = fields.next().unwrap().parse().unwrap();
        matrix
            .mutable_column(ColIndex::from_usize(column))
            .add_entry(RowIndex::from_usize(row), value);
    }
    for row in 0..n {
        matrix
            .mutable_column(ColIndex::from_usize(structural_columns + row))
            .add_entry(RowIndex::from_usize(row), 1.0);
    }
    matrix.clean_up();
    let mut basis_matrix = SparseMatrix::new();
    basis_matrix.populate_from_zero(RowIndex::from_usize(n), ColIndex::from_usize(n));
    for row in 0..n {
        basis_matrix
            .mutable_column(ColIndex::from_usize(row))
            .add_entry(RowIndex::from_usize(row), 1.0);
    }
    basis_matrix.clean_up();
    let parameters = glop::parameters::GlopParameters {
        basis_refactorization_period: i32::try_from(refactorization_period).unwrap_or(i32::MAX),
        dynamically_adjust_refactorization_period: dynamically_adjust != 0,
        use_middle_product_form_update: use_middle_product != 0,
        ..glop::parameters::GlopParameters::default()
    };
    let mut basis = BasisRepresentation::new_with_parameters(basis_matrix, &parameters).unwrap();
    let rhs: Vec<_> = (1..=n).map(|value| value as f64).collect();
    for stage in 0..updates {
        let entering: usize = fields.next().unwrap().parse().unwrap();
        let leaving: usize = fields.next().unwrap().parse().unwrap();
        let entering_column = matrix.column(ColIndex::from_usize(entering)).clone();
        let mut direction =
            lp_data::scattered_vector::ScatteredColumn::new(RowIndex::from_usize(n));
        basis
            .right_solve_for_problem_column(entering, &entering_column, &mut direction)
            .unwrap();
        let mut unit_left_inverse =
            lp_data::scattered_vector::ScatteredRow::new(ColIndex::from_usize(n));
        basis
            .left_solve_for_unit_row(leaving, &mut unit_left_inverse)
            .unwrap();
        if basis
            .replace_column_after_solve(entering, leaving, &direction, entering_column)
            .is_err()
        {
            println!("stage{stage}_error");
            return;
        }
        print!("stage{stage}_right");
        for value in basis.solve(&rhs).unwrap() {
            print!(" {value}");
        }
        println!();
        print!("stage{stage}_left");
        for value in basis.transpose_solve(&rhs).unwrap() {
            print!(" {value}");
        }
        println!();
        println!("stage{stage}_updates {}", basis.num_updates());
        println!(
            "stage{stage}_refactorized {}",
            usize::from(basis.is_refactorized())
        );
        println!(
            "stage{stage}_deterministic_time {}",
            basis.deterministic_time()
        );
        println!(
            "stage{stage}_update_entries {}",
            basis.number_of_entries_in_updates()
        );
        if let Some((u, v)) = basis.last_update_entry_counts() {
            println!("stage{stage}_last_entries {u} {v}");
        }
    }
    basis.force_refactorization().unwrap();
    println!("after_force_time {}", basis.deterministic_time());
    basis.refactorize().unwrap();
    println!("after_noop_refactorize_time {}", basis.deterministic_time());
    print!("special_right_norms");
    for position in 0..structural_columns {
        print!(
            " {}",
            basis
                .right_solve_squared_norm(matrix.column(ColIndex::from_usize(position)))
                .unwrap()
        );
    }
    println!();
    print!("special_dual_norms");
    for row in 0..n {
        print!(" {}", basis.dual_edge_squared_norm(row).unwrap());
    }
    println!();
    print!("temporary_unit_rows");
    let mut temporary_unit_nnz = Vec::with_capacity(n);
    for row in 0..n {
        let mut temporary = lp_data::scattered_vector::ScatteredRow::new(ColIndex::from_usize(n));
        basis
            .temporary_left_solve_for_unit_row(row, &mut temporary)
            .unwrap();
        temporary_unit_nnz.push(temporary.num_non_zeros_estimate());
        for column in 0..n {
            print!(" {}", temporary.value(ColIndex::from_usize(column)));
        }
    }
    println!();
    print!("temporary_unit_nnz");
    for count in temporary_unit_nnz {
        print!(" {count}");
    }
    println!();
    basis.set_column_permutation_to_identity();
    println!("one_norm {}", basis.one_norm());
    println!("infinity_norm {}", basis.infinity_norm());
    println!("inverse_one_norm {}", basis.inverse_one_norm().unwrap());
    println!(
        "inverse_infinity_norm {}",
        basis.inverse_infinity_norm().unwrap()
    );
    println!(
        "one_condition {}",
        basis.one_norm_condition_number().unwrap()
    );
    println!(
        "infinity_condition {}",
        basis.infinity_norm_condition_number().unwrap()
    );
    println!(
        "infinity_condition_bound {}",
        basis.infinity_norm_condition_number_upper_bound()
    );
    println!("lu_entries {}", basis.number_of_entries_in_lu());
    println!("final_deterministic_time {}", basis.deterministic_time());
    print!("stats_hex ");
    for byte in basis.stat_string().bytes() {
        print!("{byte:02x}");
    }
    println!();
    let before_clear_time = basis.deterministic_time();
    basis.clear();
    print!("cleared_right");
    for value in basis.solve(&rhs).unwrap() {
        print!(" {value}");
    }
    println!();
    println!(
        "clear_preserved_time {} {}",
        before_clear_time,
        basis.deterministic_time()
    );
}
