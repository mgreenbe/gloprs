use glop::basis_representation::BasisRepresentation;
use glop::dual_edge_norms::compute_dual_edge_squared_norms;
use glop::lu_factorization::LuFactorization;
use glop::numerical::relative_residual;
use glop::primal_edge_norms::compute_primal_edge_squared_norms;
use glop::update_row::compute_update_row;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
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
