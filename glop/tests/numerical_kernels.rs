use glop::basis_representation::BasisRepresentation;
use glop::dual_edge_norms::{DualEdgeNorms, compute_dual_edge_squared_norms};
use glop::lu_factorization::LuFactorization;
use glop::numerical::relative_residual;
use glop::primal_edge_norms::{PrimalEdgeNorms, compute_primal_edge_squared_norms};
use glop::update_row::{UpdateRow, UpdateRowAlgorithm, compute_update_row};
use lp_data::lp_types::{ColBitVec, ColIndex, RowIndex, VectorIndex};
use lp_data::sparse::SparseMatrix;
use lp_data::sparse_vector::SparseColumn;

struct Generator(u64);

impl Generator {
    #[allow(clippy::cast_precision_loss)]
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 11) as f64 / ((1_u64 << 53) as f64);
        2.0 * unit - 1.0
    }
}

fn sparse_matrix(values: &[Vec<f64>]) -> SparseMatrix {
    let mut matrix = SparseMatrix::new();
    matrix.populate_from_zero(
        RowIndex::from_usize(values.len()),
        ColIndex::from_usize(values[0].len()),
    );
    for (row, entries) in values.iter().enumerate() {
        for (column, &value) in entries.iter().enumerate() {
            if value != 0.0 {
                matrix
                    .mutable_column(ColIndex::from_usize(column))
                    .add_entry(RowIndex::from_usize(row), value);
            }
        }
    }
    matrix.clean_up();
    matrix
}

fn column_bits(values: &[bool]) -> ColBitVec {
    let mut result = ColBitVec::new(ColIndex::from_usize(values.len()));
    for (column, &value) in values.iter().enumerate() {
        if value {
            result.set(ColIndex::from_usize(column));
        }
    }
    result
}

fn transpose_residual(matrix: &[Vec<f64>], solution: &[f64], rhs: &[f64]) -> f64 {
    (0..matrix.len())
        .map(|row| {
            ((0..matrix.len())
                .map(|column| matrix[column][row] * solution[column])
                .sum::<f64>()
                - rhs[row])
                .abs()
        })
        .fold(0.0, f64::max)
}

#[test]
#[allow(clippy::needless_range_loop)]
fn randomized_lu_and_transpose_residuals() {
    let mut generator = Generator(0x4d59_5df4_d0f3_3173);
    for n in 1..=12 {
        for _ in 0..30 {
            let mut values = vec![vec![0.0; n]; n];
            for row in 0..n {
                let mut absolute_sum = 0.0;
                for column in 0..n {
                    if row != column && generator.next().abs() > 0.45 {
                        let value = generator.next();
                        values[row][column] = value;
                        absolute_sum += value.abs();
                    }
                }
                values[row][row] = absolute_sum + 0.5 + generator.next().abs();
            }
            let matrix = sparse_matrix(&values);
            let factorization = LuFactorization::factorize(&matrix, 0.1).unwrap();
            let rhs: Vec<f64> = (0..n).map(|_| generator.next()).collect();
            let solution = factorization.solve(&rhs).unwrap();
            assert!(relative_residual(&matrix, &solution, &rhs) < 1e-13);
            let transpose_solution = factorization.transpose_solve(&rhs).unwrap();
            assert!(transpose_residual(&values, &transpose_solution, &rhs) < 1e-11);
        }
    }
}

#[test]
fn update_rows_and_edge_norms_match_direct_solve_definitions() {
    let values = vec![
        vec![3.0, 1.0, 0.0],
        vec![1.0, 4.0, 1.0],
        vec![0.0, 2.0, 5.0],
    ];
    let matrix = sparse_matrix(&values);
    let basis = BasisRepresentation::new(matrix.clone(), 0.1, 8).unwrap();
    let update = compute_update_row(&basis, &matrix, 1).unwrap();
    assert!(
        update
            .iter()
            .enumerate()
            .all(|(column, value)| (*value - if column == 1 { 1.0 } else { 0.0 }).abs() < 1e-12)
    );

    let primal = compute_primal_edge_squared_norms(&basis, &matrix).unwrap();
    assert!(primal.iter().all(|value| (*value - 2.0).abs() < 1e-12));
    let dual = compute_dual_edge_squared_norms(&basis).unwrap();
    for (row, &norm) in dual.iter().enumerate() {
        let mut unit = vec![0.0; 3];
        unit[row] = 1.0;
        let inverse_row = basis.transpose_solve(&unit).unwrap();
        let expected: f64 = inverse_row.iter().map(|value| value * value).sum();
        assert!((norm - expected).abs() < 1e-14);
    }
}

#[test]
fn all_update_row_kernels_produce_the_same_filtered_coefficients() {
    let matrix = sparse_matrix(&[
        vec![1.0, 0.0, 2.0, 0.0, -1.0],
        vec![0.0, 3.0, 1.0, 0.0, 0.5],
        vec![2.0, 0.0, 0.0, 4.0, 0.0],
    ]);
    let relevant = column_bits(&[true, false, true, true, true]);
    let left_inverse = [0.25, -2.0, 1.5];
    let mut expected: Option<(Vec<usize>, Vec<f64>)> = None;
    for algorithm in [
        UpdateRowAlgorithm::Column,
        UpdateRowAlgorithm::Row,
        UpdateRowAlgorithm::RowHypersparse,
    ] {
        let mut update = UpdateRow::new(&matrix);
        update
            .compute_update_row_for_benchmark(&matrix, &relevant, &left_inverse, algorithm)
            .unwrap();
        if let Some((positions, coefficients)) = &expected {
            assert_eq!(update.non_zero_positions(), positions);
            for &column in positions {
                assert!((update.coefficient(column) - coefficients[column]).abs() < 1e-14);
            }
        } else {
            expected = Some((
                update.non_zero_positions().to_vec(),
                update.coefficients().to_vec(),
            ));
        }
    }
}

#[test]
fn multiple_updates_agree_with_refactorization() {
    let initial = sparse_matrix(&[
        vec![4.0, 1.0, 0.0, 0.0],
        vec![1.0, 4.0, 1.0, 0.0],
        vec![0.0, 1.0, 4.0, 1.0],
        vec![0.0, 0.0, 1.0, 4.0],
    ]);
    let mut updated = BasisRepresentation::new(initial, 0.1, 20).unwrap();
    for (leaving, entries) in [
        [(0, 2.0), (1, -1.0), (2, 1.0), (3, 0.5)],
        [(0, 1.0), (1, 2.0), (2, -0.5), (3, 1.0)],
    ]
    .into_iter()
    .enumerate()
    {
        let mut column = SparseColumn::new();
        for (row, value) in entries {
            column.add_entry(RowIndex::from_usize(row), value);
        }
        updated.replace_column(leaving, column).unwrap();
    }
    let fresh = BasisRepresentation::new(updated.basis().clone(), 0.1, 20).unwrap();
    let rhs = [1.0, -2.0, 0.5, 4.0];
    for (left, right) in updated
        .solve(&rhs)
        .unwrap()
        .iter()
        .zip(fresh.solve(&rhs).unwrap())
    {
        assert!((left - right).abs() < 1e-11);
    }
}

#[test]
fn incremental_dual_edge_norms_agree_with_exact_recomputation() {
    let initial = sparse_matrix(&[
        vec![4.0, 1.0, 0.0, 0.0],
        vec![1.0, 4.0, 1.0, 0.0],
        vec![0.0, 1.0, 4.0, 1.0],
        vec![0.0, 0.0, 1.0, 4.0],
    ]);
    let mut basis = BasisRepresentation::new(initial, 0.1, 20).unwrap();
    let mut norms = DualEdgeNorms::new();
    let _ = norms.edge_squared_norms(&basis).unwrap();

    for (leaving_row, entries) in [
        [(0, 2.0), (1, -1.0), (2, 1.0), (3, 0.5)],
        [(0, 1.0), (1, 2.0), (2, -0.5), (3, 1.0)],
    ]
    .into_iter()
    .enumerate()
    {
        let mut entering = SparseColumn::new();
        let mut dense = vec![0.0; 4];
        for (row, value) in entries {
            entering.add_entry(RowIndex::from_usize(row), value);
            dense[row] = value;
        }
        let direction = basis.solve(&dense).unwrap();
        let mut unit = vec![0.0; 4];
        unit[leaving_row] = 1.0;
        let unit_row_left_inverse = basis.transpose_solve(&unit).unwrap();
        assert!(norms.test_precision(leaving_row, &unit_row_left_inverse));
        norms
            .update_before_basis_pivot(&basis, leaving_row, &direction, &unit_row_left_inverse)
            .unwrap();
        basis.replace_column(leaving_row, entering).unwrap();

        let exact = compute_dual_edge_squared_norms(&basis).unwrap();
        for (&maintained, expected) in norms.edge_squared_norms(&basis).unwrap().iter().zip(exact) {
            assert!((maintained - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn incremental_primal_edge_norms_agree_with_exact_recomputation() {
    let full_matrix = sparse_matrix(&[
        vec![2.0, -1.0, 1.0, 0.0, 0.0],
        vec![-1.0, 0.5, 0.0, 1.0, 0.0],
        vec![1.0, 2.0, 0.0, 0.0, 1.0],
    ]);
    let initial_basis = sparse_matrix(&[
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let mut basis = BasisRepresentation::new(initial_basis, 0.1, 20).unwrap();
    let relevant_before = column_bits(&[true, true, false, false, false]);
    let mut norms = PrimalEdgeNorms::new(&full_matrix);
    let _ = norms.edge_squared_norms(&basis, &relevant_before).unwrap();

    let entering_column = 0;
    let leaving_row = 1;
    let leaving_column = 3;
    let mut entering_dense = vec![0.0; 3];
    let mut entering = SparseColumn::new();
    for entry in full_matrix.column(ColIndex::from_usize(entering_column)) {
        entering_dense[entry.index().to_usize()] = entry.coefficient();
        entering.add_entry(entry.index(), entry.coefficient());
    }
    let direction = basis.solve(&entering_dense).unwrap();
    assert!(norms.test_entering_edge_norm_precision(entering_column, &direction));
    let mut update_row = UpdateRow::new(&full_matrix);
    norms
        .update_before_basis_pivot(
            &basis,
            &relevant_before,
            entering_column,
            leaving_column,
            leaving_row,
            &direction,
            &mut update_row,
        )
        .unwrap();
    basis.replace_column(leaving_row, entering).unwrap();

    let relevant_after = column_bits(&[false, true, false, true, false]);
    let exact = compute_primal_edge_squared_norms(&basis, &full_matrix).unwrap();
    let maintained = norms.edge_squared_norms(&basis, &relevant_after).unwrap();
    for column in [leaving_column, 1] {
        assert!((maintained[column] - exact[column]).abs() < 1e-10);
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn randomized_middle_product_updates_agree_with_refactorization() {
    let mut generator = Generator(0xd1b5_4a32_d192_ed03);
    for n in 2..=10 {
        let mut values = vec![vec![0.0; n]; n];
        for row in 0..n {
            let mut absolute_sum = 0.0;
            for column in 0..n {
                if row != column && generator.next().abs() > 0.5 {
                    values[row][column] = generator.next();
                    absolute_sum += values[row][column].abs();
                }
            }
            values[row][row] = absolute_sum + 1.0;
        }
        let mut updated = BasisRepresentation::new(sparse_matrix(&values), 0.1, 100).unwrap();

        for update_index in 0..30 {
            let leaving = update_index % n;
            let mut entering_dense = vec![0.0; n];
            for entry in updated.basis().column(ColIndex::from_usize(leaving)) {
                entering_dense[entry.index().to_usize()] = entry.coefficient();
            }
            for value in &mut entering_dense {
                *value += 0.025 * generator.next();
            }
            let mut entering = SparseColumn::new();
            for (row, &value) in entering_dense.iter().enumerate() {
                if value != 0.0 {
                    entering.add_entry(RowIndex::from_usize(row), value);
                }
            }

            let mut expected_basis = updated.basis().clone();
            expected_basis.replace_column(ColIndex::from_usize(leaving), entering.clone());
            let Ok(fresh) = BasisRepresentation::new(expected_basis, 0.1, 100) else {
                continue;
            };
            updated.replace_column(leaving, entering).unwrap();

            let rhs: Vec<_> = (0..n).map(|_| generator.next()).collect();
            for (maintained, expected) in updated
                .solve(&rhs)
                .unwrap()
                .iter()
                .zip(fresh.solve(&rhs).unwrap())
            {
                assert!((maintained - expected).abs() < 2e-10);
            }
            for (maintained, expected) in updated
                .transpose_solve(&rhs)
                .unwrap()
                .iter()
                .zip(fresh.transpose_solve(&rhs).unwrap())
            {
                assert!((maintained - expected).abs() < 2e-10);
            }
        }
    }
}

#[test]
fn badly_scaled_and_nearly_singular_systems_are_handled_or_rejected() {
    let scaled = sparse_matrix(&[vec![1e-12, 0.0], vec![0.0, 1e12]]);
    let factorization = LuFactorization::factorize(&scaled, 0.1).unwrap();
    let expected = [2.0, -1.0];
    let rhs = [2e-12, -1e12];
    let solution = factorization.solve(&rhs).unwrap();
    assert!(
        solution
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() < 1e-12)
    );

    let rank_deficient = sparse_matrix(&[
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![0.0, 1.0, 1.0],
    ]);
    assert!(LuFactorization::factorize(&rank_deficient, 0.1).is_err());
}
