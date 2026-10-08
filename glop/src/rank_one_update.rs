//! Product-form inverse (eta) updates used between basis refactorizations.

use std::cell::Cell;

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex, deterministic_time_for_fp_operations};
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};

pub(crate) fn sparse_scalar_product(entries: &[(usize, f64)], values: &[f64]) -> f64 {
    let shifted_end = entries.len().saturating_sub(3);
    let mut entry = 0;
    let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
    while entry < shifted_end {
        result1 = entries[entry].1.mul_add(values[entries[entry].0], result1);
        result2 = entries[entry + 1]
            .1
            .mul_add(values[entries[entry + 1].0], result2);
        result3 = entries[entry + 2]
            .1
            .mul_add(values[entries[entry + 2].0], result3);
        result4 = entries[entry + 3]
            .1
            .mul_add(values[entries[entry + 3].0], result4);
        entry += 4;
    }
    let mut result = result1 + result2 + result3 + result4;
    if entry < entries.len() {
        result = entries[entry].1.mul_add(values[entries[entry].0], result);
        if entry + 1 < entries.len() {
            result = entries[entry + 1]
                .1
                .mul_add(values[entries[entry + 1].0], result);
            if entry + 2 < entries.len() {
                result = entries[entry + 2]
                    .1
                    .mul_add(values[entries[entry + 2].0], result);
            }
        }
    }
    result
}

/// An elementary matrix `T = I + u v^T`, corresponding to upstream
/// `RankOneUpdateElementaryMatrix`.
#[derive(Clone, Debug)]
pub struct RankOneUpdateElementaryMatrix {
    u: Vec<(usize, f64)>,
    v: Vec<(usize, f64)>,
    mu: f64,
}

impl RankOneUpdateElementaryMatrix {
    #[must_use]
    pub fn new(u: Vec<(usize, f64)>, v: Vec<(usize, f64)>, u_dot_v: f64) -> Self {
        Self {
            u,
            v,
            mu: 1.0 + u_dot_v,
        }
    }

    #[must_use]
    pub fn is_singular(&self) -> bool {
        self.mu == 0.0
    }

    pub fn right_solve(&self, values: &mut [f64]) {
        debug_assert!(!self.is_singular());
        let multiplier = -sparse_scalar_product(&self.v, values) / self.mu;
        for &(index, value) in &self.u {
            values[index] = multiplier.mul_add(value, values[index]);
        }
    }

    pub fn left_solve(&self, values: &mut [f64]) {
        debug_assert!(!self.is_singular());
        let multiplier = -sparse_scalar_product(&self.u, values) / self.mu;
        for &(index, value) in &self.v {
            values[index] = multiplier.mul_add(value, values[index]);
        }
    }

    pub fn right_multiply(&self, values: &mut [f64]) {
        let multiplier = sparse_scalar_product(&self.v, values);
        for &(index, value) in &self.u {
            values[index] = multiplier.mul_add(value, values[index]);
        }
    }

    pub fn left_multiply(&self, values: &mut [f64]) {
        let multiplier = sparse_scalar_product(&self.u, values);
        for &(index, value) in &self.v {
            values[index] = multiplier.mul_add(value, values[index]);
        }
    }

    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.u.len() + self.v.len()
    }
}

/// Product `T_0 T_1 ... T_(k-1)` with upstream solve ordering.
#[derive(Clone, Debug)]
pub struct RankOneUpdateFactorization {
    // GLOP stores every u/v pair as consecutive CompactSparseMatrix columns.
    // These packed arrays are the Rust equivalent and avoid one allocation per
    // vector while preserving sequential solve traversal.
    entries: Vec<(usize, f64)>,
    elementary_matrices: Vec<PackedElementaryMatrix>,
    num_entries: usize,
    hypersparse_ratio: f64,
    deterministic_time: Cell<f64>,
}

#[derive(Clone, Copy, Debug)]
struct PackedElementaryMatrix {
    u_start: usize,
    u_end: usize,
    v_start: usize,
    v_end: usize,
    mu: f64,
}

impl Default for RankOneUpdateFactorization {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            elementary_matrices: Vec::new(),
            num_entries: 0,
            hypersparse_ratio: 0.05,
            deterministic_time: Cell::new(0.0),
        }
    }
}

impl RankOneUpdateFactorization {
    pub fn clear(&mut self) {
        self.entries.clear();
        self.elementary_matrices.clear();
        self.num_entries = 0;
    }

    pub fn update(&mut self, matrix: RankOneUpdateElementaryMatrix) {
        self.num_entries += matrix.num_entries();
        let u_start = self.entries.len();
        self.entries.extend(matrix.u);
        let u_end = self.entries.len();
        let v_start = u_end;
        self.entries.extend(matrix.v);
        let v_end = self.entries.len();
        self.elementary_matrices.push(PackedElementaryMatrix {
            u_start,
            u_end,
            v_start,
            v_end,
            mu: matrix.mu,
        });
    }

    pub fn right_solve(&self, values: &mut [f64]) {
        for matrix in &self.elementary_matrices {
            let multiplier =
                -sparse_scalar_product(&self.entries[matrix.v_start..matrix.v_end], values)
                    / matrix.mu;
            for &(index, value) in &self.entries[matrix.u_start..matrix.u_end] {
                values[index] = multiplier.mul_add(value, values[index]);
            }
        }
        self.add_deterministic_time();
    }

    pub fn left_solve(&self, values: &mut [f64]) {
        for matrix in self.elementary_matrices.iter().rev() {
            let multiplier =
                -sparse_scalar_product(&self.entries[matrix.u_start..matrix.u_end], values)
                    / matrix.mu;
            for &(index, value) in &self.entries[matrix.v_start..matrix.v_end] {
                values[index] = multiplier.mul_add(value, values[index]);
            }
        }
        self.add_deterministic_time();
    }

    pub fn right_solve_with_nonzeros(&self, values: &mut ScatteredColumn) {
        if values.non_zeros().is_empty() {
            self.right_solve(values.values_mut().as_mut_slice());
            return;
        }
        values.repopulate_sparse_mask();
        let mut use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
        if use_dense {
            values.non_zeros_mut().clear();
        }
        for matrix in &self.elementary_matrices {
            let multiplier = -sparse_scalar_product(
                &self.entries[matrix.v_start..matrix.v_end],
                values.values().as_slice(),
            ) / matrix.mu;
            if multiplier == 0.0 {
                continue;
            }
            if use_dense {
                let dense = values.values_mut().as_mut_slice();
                for &(index, value) in &self.entries[matrix.u_start..matrix.u_end] {
                    dense[index] = multiplier.mul_add(value, dense[index]);
                }
            } else {
                for &(index, value) in &self.entries[matrix.u_start..matrix.u_end] {
                    values.add(RowIndex::from_usize(index), multiplier * value);
                }
                use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
                if use_dense {
                    values.non_zeros_mut().clear();
                }
            }
        }
        values.clear_sparse_mask();
        values.clear_non_zeros_if_too_dense(self.hypersparse_ratio);
        self.add_deterministic_time();
    }

    pub fn left_solve_with_nonzeros(&self, values: &mut ScatteredRow) {
        if values.non_zeros().is_empty() {
            self.left_solve(values.values_mut().as_mut_slice());
            return;
        }
        values.repopulate_sparse_mask();
        let mut use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
        if use_dense {
            values.non_zeros_mut().clear();
        }
        for matrix in self.elementary_matrices.iter().rev() {
            let multiplier = -sparse_scalar_product(
                &self.entries[matrix.u_start..matrix.u_end],
                values.values().as_slice(),
            ) / matrix.mu;
            if multiplier == 0.0 {
                continue;
            }
            if use_dense {
                let dense = values.values_mut().as_mut_slice();
                for &(index, value) in &self.entries[matrix.v_start..matrix.v_end] {
                    dense[index] = multiplier.mul_add(value, dense[index]);
                }
            } else {
                for &(index, value) in &self.entries[matrix.v_start..matrix.v_end] {
                    values.add(ColIndex::from_usize(index), multiplier * value);
                }
                use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
                if use_dense {
                    values.non_zeros_mut().clear();
                }
            }
        }
        values.clear_sparse_mask();
        values.clear_non_zeros_if_too_dense(self.hypersparse_ratio);
        self.add_deterministic_time();
    }

    pub fn set_hypersparse_ratio(&mut self, value: f64) {
        self.hypersparse_ratio = value;
    }

    #[must_use]
    pub const fn num_entries(&self) -> usize {
        self.num_entries
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.elementary_matrices.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elementary_matrices.is_empty()
    }

    #[must_use]
    pub fn last_entry_counts(&self) -> Option<(usize, usize)> {
        self.elementary_matrices
            .last()
            .map(|matrix| (matrix.u_end - matrix.u_start, matrix.v_end - matrix.v_start))
    }

    #[must_use]
    pub fn deterministic_time_since_last_reset(&self) -> f64 {
        self.deterministic_time.get()
    }

    pub fn reset_deterministic_time(&self) {
        self.deterministic_time.set(0.0);
    }

    fn add_deterministic_time(&self) {
        self.deterministic_time.set(
            self.deterministic_time.get()
                + deterministic_time_for_fp_operations(
                    i64::try_from(self.num_entries).unwrap_or(i64::MAX),
                ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementary_solve_inverts_rank_one_multiply_in_both_directions() {
        let update = RankOneUpdateElementaryMatrix::new(
            vec![(0, 2.0), (2, -1.0)],
            vec![(1, 3.0), (2, 0.5)],
            -0.5,
        );
        let original = [1.0, -2.0, 4.0];
        let mut right = original;
        update.right_multiply(&mut right);
        update.right_solve(&mut right);
        assert!(
            right
                .iter()
                .zip(original)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        let mut left = original;
        update.left_multiply(&mut left);
        update.left_solve(&mut left);
        assert!(
            left.iter()
                .zip(original)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn packed_factorization_sparse_and_dense_solves_agree() {
        let updates = [
            RankOneUpdateElementaryMatrix::new(
                vec![(0, 2.0), (2, -1.0)],
                vec![(1, 3.0), (2, 0.5)],
                -0.5,
            ),
            RankOneUpdateElementaryMatrix::new(
                vec![(1, -0.25), (3, 2.0)],
                vec![(0, 1.5), (3, -0.5)],
                -1.375,
            ),
        ];
        let mut factorization = RankOneUpdateFactorization::default();
        factorization.set_hypersparse_ratio(1.0);
        for update in updates {
            factorization.update(update);
        }

        let rhs = [1.0, 0.0, -2.0, 0.0];
        let mut dense = rhs;
        factorization.right_solve(&mut dense);
        let mut sparse = ScatteredColumn::new(RowIndex::new(4));
        sparse.set(RowIndex::new(0), rhs[0]);
        sparse.set(RowIndex::new(2), rhs[2]);
        factorization.right_solve_with_nonzeros(&mut sparse);
        assert!(dense.iter().enumerate().all(|(index, value)| {
            (*value - sparse.value(RowIndex::from_usize(index))).abs() < 1e-12
        }));

        let mut dense = rhs;
        factorization.left_solve(&mut dense);
        let mut sparse = ScatteredRow::new(ColIndex::new(4));
        sparse.set(ColIndex::new(0), rhs[0]);
        sparse.set(ColIndex::new(2), rhs[2]);
        factorization.left_solve_with_nonzeros(&mut sparse);
        assert!(dense.iter().enumerate().all(|(index, value)| {
            (*value - sparse.value(ColIndex::from_usize(index))).abs() < 1e-12
        }));
        assert!(factorization.deterministic_time_since_last_reset() > 0.0);
        factorization.reset_deterministic_time();
        assert_eq!(
            factorization
                .deterministic_time_since_last_reset()
                .to_bits(),
            0.0_f64.to_bits()
        );
    }
}
