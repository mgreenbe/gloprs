//! Product-form inverse (eta) updates used between basis refactorizations.

use std::cell::Cell;

use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex, deterministic_time_for_fp_operations};
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};
use lp_data::sparse::CompactColumn;

pub(crate) fn sparse_scalar_product(entries: &[(usize, f64)], values: &[f64]) -> f64 {
    // Four-entry chunks prove positional reads of the packed update storage
    // in range. The gathered RHS remains checked: public elementary matrices
    // can carry arbitrary indices and public solves accept arbitrary slices.
    let mut chunks = entries.chunks_exact(4);
    let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
    for chunk in chunks.by_ref() {
        result1 = chunk[0].1.mul_add(values[chunk[0].0], result1);
        result2 = chunk[1].1.mul_add(values[chunk[1].0], result2);
        result3 = chunk[2].1.mul_add(values[chunk[2].0], result3);
        result4 = chunk[3].1.mul_add(values[chunk[3].0], result4);
    }
    let mut result = result1 + result2 + result3 + result4;
    for &(index, coefficient) in chunks.remainder() {
        result = coefficient.mul_add(values[index], result);
    }
    result
}

fn packed_scalar_product(rows: &[u32], coefficients: &[f64], values: &[f64]) -> f64 {
    debug_assert_eq!(rows.len(), coefficients.len());
    let mut row_chunks = rows.chunks_exact(4);
    let mut coefficient_chunks = coefficients.chunks_exact(4);
    let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
    for (r, c) in row_chunks.by_ref().zip(coefficient_chunks.by_ref()) {
        result1 = c[0].mul_add(values[r[0] as usize], result1);
        result2 = c[1].mul_add(values[r[1] as usize], result2);
        result3 = c[2].mul_add(values[r[2] as usize], result3);
        result4 = c[3].mul_add(values[r[3] as usize], result4);
    }
    let mut result = result1 + result2 + result3 + result4;
    for (&row, &coefficient) in row_chunks
        .remainder()
        .iter()
        .zip(coefficient_chunks.remainder())
    {
        result = coefficient.mul_add(values[row as usize], result);
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
    #[inline]
    pub fn is_singular(&self) -> bool {
        self.mu == 0.0
    }

    pub fn right_solve(&self, values: &mut [f64]) {
        debug_assert!(!self.is_singular());
        let multiplier = -sparse_scalar_product(&self.v, values) / self.mu;
        for &(index, value) in &self.u {
            let slot = &mut values[index];
            *slot = multiplier.mul_add(value, *slot);
        }
    }

    pub fn left_solve(&self, values: &mut [f64]) {
        debug_assert!(!self.is_singular());
        let multiplier = -sparse_scalar_product(&self.u, values) / self.mu;
        for &(index, value) in &self.v {
            let slot = &mut values[index];
            *slot = multiplier.mul_add(value, *slot);
        }
    }

    pub fn right_multiply(&self, values: &mut [f64]) {
        let multiplier = sparse_scalar_product(&self.v, values);
        for &(index, value) in &self.u {
            let slot = &mut values[index];
            *slot = multiplier.mul_add(value, *slot);
        }
    }

    pub fn left_multiply(&self, values: &mut [f64]) {
        let multiplier = sparse_scalar_product(&self.u, values);
        for &(index, value) in &self.v {
            let slot = &mut values[index];
            *slot = multiplier.mul_add(value, *slot);
        }
    }

    #[must_use]
    #[inline]
    pub fn num_entries(&self) -> usize {
        self.u.len() + self.v.len()
    }
}

/// Product `T_0 T_1 ... T_(k-1)` with upstream solve ordering.
#[derive(Clone, Debug)]
pub struct RankOneUpdateFactorization {
    // GLOP stores every u/v pair as consecutive CompactSparseMatrix columns,
    // with separate 32-bit rows and f64 coefficients. Keep that layout here.
    entry_rows: Vec<u32>,
    entry_coefficients: Vec<f64>,
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
            entry_rows: Vec::new(),
            entry_coefficients: Vec::new(),
            elementary_matrices: Vec::new(),
            num_entries: 0,
            hypersparse_ratio: 0.05,
            deterministic_time: Cell::new(0.0),
        }
    }
}

impl RankOneUpdateFactorization {
    #[inline]
    fn entries_in(&self, start: usize, end: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.entry_rows[start..end]
            .iter()
            .copied()
            .zip(self.entry_coefficients[start..end].iter().copied())
            .map(|(row, coefficient)| (row as usize, coefficient))
    }

    pub fn clear(&mut self) {
        self.entry_rows.clear();
        self.entry_coefficients.clear();
        self.elementary_matrices.clear();
        self.num_entries = 0;
    }

    /// Append one elementary matrix to the compact update chain.
    ///
    /// # Panics
    ///
    /// Panics if an update row cannot be represented by GLOP's 32-bit index.
    pub fn update(&mut self, matrix: RankOneUpdateElementaryMatrix) {
        self.num_entries += matrix.num_entries();
        let u_start = self.entry_rows.len();
        for (row, coefficient) in matrix.u {
            self.entry_rows
                .push(u32::try_from(row).expect("update row fits 32 bits"));
            self.entry_coefficients.push(coefficient);
        }
        let u_end = self.entry_rows.len();
        let v_start = u_end;
        for (row, coefficient) in matrix.v {
            self.entry_rows
                .push(u32::try_from(row).expect("update row fits 32 bits"));
            self.entry_coefficients.push(coefficient);
        }
        let v_end = self.entry_rows.len();
        self.elementary_matrices.push(PackedElementaryMatrix {
            u_start,
            u_end,
            v_start,
            v_end,
            mu: matrix.mu,
        });
    }

    /// Append GLOP's right scratchpad and already-stored left compact column
    /// directly, avoiding per-pivot temporary u/v allocations.
    pub(crate) fn update_from_compact(
        &mut self,
        right: &ScatteredColumn,
        right_is_dense: bool,
        left: CompactColumn<'_>,
        u_dot_v: f64,
    ) {
        let u_start = self.entry_rows.len();
        if right_is_dense {
            for (row, &coefficient) in right.values().as_slice().iter().enumerate() {
                if coefficient != 0.0 {
                    self.entry_rows
                        .push(u32::try_from(row).expect("update row fits 32 bits"));
                    self.entry_coefficients.push(coefficient);
                }
            }
        } else {
            for entry in right {
                let coefficient = entry.coefficient();
                if coefficient != 0.0 {
                    self.entry_rows.push(
                        u32::try_from(entry.row().to_usize()).expect("update row fits 32 bits"),
                    );
                    self.entry_coefficients.push(coefficient);
                }
            }
        }
        let u_end = self.entry_rows.len();
        let v_start = u_end;
        for (row, coefficient) in left.iter() {
            if coefficient != 0.0 {
                self.entry_rows
                    .push(u32::try_from(row.to_usize()).expect("update row fits 32 bits"));
                self.entry_coefficients.push(coefficient);
            }
        }
        let v_end = self.entry_rows.len();
        self.num_entries += v_end - u_start;
        self.elementary_matrices.push(PackedElementaryMatrix {
            u_start,
            u_end,
            v_start,
            v_end,
            mu: 1.0 + u_dot_v,
        });
    }

    pub fn right_solve(&self, values: &mut [f64]) {
        for matrix in &self.elementary_matrices {
            let multiplier = -packed_scalar_product(
                &self.entry_rows[matrix.v_start..matrix.v_end],
                &self.entry_coefficients[matrix.v_start..matrix.v_end],
                values,
            ) / matrix.mu;
            for (index, value) in self.entries_in(matrix.u_start, matrix.u_end) {
                let slot = &mut values[index];
                *slot = multiplier.mul_add(value, *slot);
            }
        }
        self.add_deterministic_time();
    }

    pub fn left_solve(&self, values: &mut [f64]) {
        for matrix in self.elementary_matrices.iter().rev() {
            let multiplier = -packed_scalar_product(
                &self.entry_rows[matrix.u_start..matrix.u_end],
                &self.entry_coefficients[matrix.u_start..matrix.u_end],
                values,
            ) / matrix.mu;
            for (index, value) in self.entries_in(matrix.v_start, matrix.v_end) {
                let slot = &mut values[index];
                *slot = multiplier.mul_add(value, *slot);
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
        for matrix in &self.elementary_matrices {
            let multiplier = -packed_scalar_product(
                &self.entry_rows[matrix.v_start..matrix.v_end],
                &self.entry_coefficients[matrix.v_start..matrix.v_end],
                values.values().as_slice(),
            ) / matrix.mu;
            if multiplier == 0.0 {
                continue;
            }
            if use_dense {
                let dense = values.values_mut().as_mut_slice();
                for (index, value) in self.entries_in(matrix.u_start, matrix.u_end) {
                    let slot = &mut dense[index];
                    *slot = multiplier.mul_add(value, *slot);
                }
            } else {
                for (index, value) in self.entries_in(matrix.u_start, matrix.u_end) {
                    values.add(RowIndex::from_usize(index), multiplier * value);
                }
                use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
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
        for matrix in self.elementary_matrices.iter().rev() {
            let multiplier = -packed_scalar_product(
                &self.entry_rows[matrix.u_start..matrix.u_end],
                &self.entry_coefficients[matrix.u_start..matrix.u_end],
                values.values().as_slice(),
            ) / matrix.mu;
            if multiplier == 0.0 {
                continue;
            }
            if use_dense {
                let dense = values.values_mut().as_mut_slice();
                for (index, value) in self.entries_in(matrix.v_start, matrix.v_end) {
                    let slot = &mut dense[index];
                    *slot = multiplier.mul_add(value, *slot);
                }
            } else {
                for (index, value) in self.entries_in(matrix.v_start, matrix.v_end) {
                    values.add(ColIndex::from_usize(index), multiplier * value);
                }
                use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
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
    #[inline]
    pub const fn num_entries(&self) -> usize {
        self.num_entries
    }

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.elementary_matrices.len()
    }

    #[must_use]
    #[inline]
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
    #[inline]
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
    fn packed_dot_preserves_four_accumulator_bits() {
        let values = [1.0e16, -3.0, 0.125, -1.0e16, 7.0, -0.5, 2.0];
        for len in 0..29 {
            let entries: Vec<_> = (0..len)
                .map(|position| {
                    (
                        (position * 5 + 3) % values.len(),
                        (f64::from(u32::try_from(position).unwrap()) - 11.0) * 0.125,
                    )
                })
                .collect();
            let rows: Vec<_> = entries
                .iter()
                .map(|&(row, _)| u32::try_from(row).unwrap())
                .collect();
            let coefficients: Vec<_> = entries.iter().map(|&(_, value)| value).collect();
            assert_eq!(
                packed_scalar_product(&rows, &coefficients, &values).to_bits(),
                sparse_scalar_product(&entries, &values).to_bits(),
                "length {len}"
            );
        }
    }

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

    #[test]
    fn public_update_rejects_out_of_range_dense_gathers() {
        let update = RankOneUpdateElementaryMatrix::new(vec![], vec![(2, 1.0)], 0.0);
        assert!(std::panic::catch_unwind(|| update.right_solve(&mut [1.0, 2.0])).is_err());
    }
}
