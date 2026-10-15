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

#[allow(unsafe_code)]
unsafe fn packed_scalar_product(rows: &[u32], coefficients: &[f64], values: &[f64]) -> f64 {
    debug_assert_eq!(rows.len(), coefficients.len());
    let mut row_chunks = rows.chunks_exact(4);
    let mut coefficient_chunks = coefficients.chunks_exact(4);
    let (mut result1, mut result2, mut result3, mut result4) = (0.0, 0.0, 0.0, 0.0);
    for (r, c) in row_chunks.by_ref().zip(coefficient_chunks.by_ref()) {
        // SAFETY: every solve checks the required RHS length once before this
        // packed inner loop, including caller-built updates.
        result1 = c[0].mul_add(unsafe { *values.get_unchecked(r[0] as usize) }, result1);
        result2 = c[1].mul_add(unsafe { *values.get_unchecked(r[1] as usize) }, result2);
        result3 = c[2].mul_add(unsafe { *values.get_unchecked(r[2] as usize) }, result3);
        result4 = c[3].mul_add(unsafe { *values.get_unchecked(r[3] as usize) }, result4);
    }
    let mut result = result1 + result2 + result3 + result4;
    for (&row, &coefficient) in row_chunks
        .remainder()
        .iter()
        .zip(coefficient_chunks.remainder())
    {
        // SAFETY: the same solve-entry length check covers the tail.
        result = coefficient.mul_add(unsafe { *values.get_unchecked(row as usize) }, result);
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

    /// # Panics
    ///
    /// Panics when a stored update row exceeds the right-hand-side length.
    pub fn right_solve(&self, values: &mut [f64]) {
        debug_assert!(!self.is_singular());
        let multiplier = -sparse_scalar_product(&self.v, values) / self.mu;
        for &(index, value) in &self.u {
            let slot = &mut values[index];
            *slot = multiplier.mul_add(value, *slot);
        }
    }

    /// # Panics
    ///
    /// Panics when a stored update row exceeds the right-hand-side length.
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
    required_len: usize,
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
            required_len: 0,
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
        self.required_len = 0;
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
            let row = u32::try_from(row).expect("update row fits 32 bits");
            self.required_len = self.required_len.max(
                (row as usize)
                    .checked_add(1)
                    .expect("update row fits RHS length"),
            );
            self.entry_rows.push(row);
            self.entry_coefficients.push(coefficient);
        }
        let u_end = self.entry_rows.len();
        let v_start = u_end;
        for (row, coefficient) in matrix.v {
            let row = u32::try_from(row).expect("update row fits 32 bits");
            self.required_len = self.required_len.max(
                (row as usize)
                    .checked_add(1)
                    .expect("update row fits RHS length"),
            );
            self.entry_rows.push(row);
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
    ///
    /// # Safety
    ///
    /// `left` must be a column of the same basis dimension as `right`.
    pub(crate) unsafe fn update_from_compact(
        &mut self,
        right: &mut ScatteredColumn,
        left: CompactColumn<'_>,
        u_dot_v: f64,
    ) {
        let u_start = self.entry_rows.len();
        // The solver's scratchpad and compact column share the basis
        // dimension. Record it once, rather than updating a maximum for each
        // stored entry as required by the public update() path.
        let dimension = right.len().to_usize();
        self.required_len = self.required_len.max(dimension);
        // Match CompactSparseMatrix::AddAndClearColumnWithNonZeros(): the
        // touched list can contain duplicates. The first visit stores and
        // clears the final value; subsequent visits see zero.
        let (values, touched) = right.mutable_parts();
        for &row in touched.iter() {
            let index = row.to_usize();
            let coefficient = values[index];
            if coefficient != 0.0 {
                let row = u32::try_from(index).expect("update row fits 32 bits");
                self.entry_rows.push(row);
                self.entry_coefficients.push(coefficient);
                values[index] = 0.0;
            }
        }
        touched.clear();
        let u_end = self.entry_rows.len();
        let v_start = u_end;
        for (row, coefficient) in left.iter() {
            if coefficient != 0.0 {
                debug_assert!(row.to_usize() < dimension);
                let row = u32::try_from(row.to_usize()).expect("update row fits 32 bits");
                self.entry_rows.push(row);
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

    /// # Panics
    ///
    /// Panics when a stored update row exceeds the right-hand-side length.
    #[allow(unsafe_code)]
    pub fn right_solve(&self, values: &mut [f64]) {
        assert!(self.required_len <= values.len());
        for matrix in &self.elementary_matrices {
            // SAFETY: required_len was checked above and bounds every packed row.
            let multiplier = -unsafe {
                packed_scalar_product(
                    &self.entry_rows[matrix.v_start..matrix.v_end],
                    &self.entry_coefficients[matrix.v_start..matrix.v_end],
                    values,
                )
            } / matrix.mu;
            for (index, value) in self.entries_in(matrix.u_start, matrix.u_end) {
                // SAFETY: required_len was checked against values at solve entry.
                let slot = unsafe { values.get_unchecked_mut(index) };
                *slot = multiplier.mul_add(value, *slot);
            }
        }
        self.add_deterministic_time();
    }

    /// # Panics
    ///
    /// Panics when a stored update row exceeds the right-hand-side length.
    #[allow(unsafe_code)]
    pub fn left_solve(&self, values: &mut [f64]) {
        assert!(self.required_len <= values.len());
        for matrix in self.elementary_matrices.iter().rev() {
            // SAFETY: required_len was checked above and bounds every packed row.
            let multiplier = -unsafe {
                packed_scalar_product(
                    &self.entry_rows[matrix.u_start..matrix.u_end],
                    &self.entry_coefficients[matrix.u_start..matrix.u_end],
                    values,
                )
            } / matrix.mu;
            for (index, value) in self.entries_in(matrix.v_start, matrix.v_end) {
                // SAFETY: required_len was checked against values at solve entry.
                let slot = unsafe { values.get_unchecked_mut(index) };
                *slot = multiplier.mul_add(value, *slot);
            }
        }
        self.add_deterministic_time();
    }

    /// # Panics
    ///
    /// Panics when a stored update row exceeds the right-hand-side length.
    #[allow(unsafe_code)]
    pub fn right_solve_with_nonzeros(&self, values: &mut ScatteredColumn) {
        assert!(self.required_len <= values.len().to_usize());
        if values.non_zeros().is_empty() {
            self.right_solve(values.values_mut().as_mut_slice());
            return;
        }
        values.repopulate_sparse_mask();
        let mut use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
        for matrix in &self.elementary_matrices {
            // SAFETY: required_len was checked above and bounds every packed row.
            let multiplier = -unsafe {
                packed_scalar_product(
                    &self.entry_rows[matrix.v_start..matrix.v_end],
                    &self.entry_coefficients[matrix.v_start..matrix.v_end],
                    values.values().as_slice(),
                )
            } / matrix.mu;
            if multiplier == 0.0 {
                continue;
            }
            if use_dense {
                let dense = values.values_mut().as_mut_slice();
                for (index, value) in self.entries_in(matrix.u_start, matrix.u_end) {
                    // SAFETY: required_len was checked against the scattered
                    // vector's dense backing slice at solve entry.
                    let slot = unsafe { dense.get_unchecked_mut(index) };
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

    /// # Panics
    ///
    /// Panics when a stored update row exceeds the right-hand-side length.
    #[allow(unsafe_code)]
    pub fn left_solve_with_nonzeros(&self, values: &mut ScatteredRow) {
        assert!(self.required_len <= values.len().to_usize());
        if values.non_zeros().is_empty() {
            self.left_solve(values.values_mut().as_mut_slice());
            return;
        }
        values.repopulate_sparse_mask();
        let mut use_dense = values.should_use_dense_iteration(self.hypersparse_ratio);
        for matrix in self.elementary_matrices.iter().rev() {
            // SAFETY: required_len was checked above and bounds every packed row.
            let multiplier = -unsafe {
                packed_scalar_product(
                    &self.entry_rows[matrix.u_start..matrix.u_end],
                    &self.entry_coefficients[matrix.u_start..matrix.u_end],
                    values.values().as_slice(),
                )
            } / matrix.mu;
            if multiplier == 0.0 {
                continue;
            }
            if use_dense {
                let dense = values.values_mut().as_mut_slice();
                for (index, value) in self.entries_in(matrix.v_start, matrix.v_end) {
                    // SAFETY: required_len was checked against the scattered
                    // vector's dense backing slice at solve entry.
                    let slot = unsafe { dense.get_unchecked_mut(index) };
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
    use lp_data::sparse::CompactSparseMatrix;

    #[test]
    fn compact_update_consumes_duplicate_touched_rows_once() {
        let mut right = ScatteredColumn::new(RowIndex::new(3));
        right.scatter_entry_without_mask(RowIndex::new(1), 2.0);
        right.values_mut()[RowIndex::new(1)] -= 0.5;
        right.non_zeros_mut().push(RowIndex::new(1));
        right.scatter_entry_without_mask(RowIndex::new(2), -3.0);

        let mut left = CompactSparseMatrix::default();
        left.reset(RowIndex::new(3));
        left.add_entry_to_current_column(RowIndex::new(0), 4.0);
        left.close_current_column();
        let mut updates = RankOneUpdateFactorization::default();
        // SAFETY: both inputs have the same three-row dimension.
        unsafe { updates.update_from_compact(&mut right, left.column(ColIndex::new(0)), 0.0) };

        assert_eq!(updates.entry_rows, [1, 2, 0]);
        assert_eq!(updates.entry_coefficients, [1.5, -3.0, 4.0]);
        assert!(right.values().as_slice().iter().all(|&value| value == 0.0));
        assert!(right.non_zeros().is_empty());
    }

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
                // SAFETY: test rows are generated modulo values.len().
                unsafe { packed_scalar_product(&rows, &coefficients, &values) }.to_bits(),
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
    fn packed_factorization_rejects_short_rhs_before_unchecked_access() {
        let mut factorization = RankOneUpdateFactorization::default();
        factorization.update(RankOneUpdateElementaryMatrix::new(
            vec![(2, 1.0)],
            vec![(1, 2.0)],
            0.0,
        ));
        assert_eq!(factorization.required_len, 3);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                factorization.right_solve(&mut [0.0; 2]);
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                factorization.left_solve_with_nonzeros(&mut ScatteredRow::new(ColIndex::new(2)));
            }))
            .is_err()
        );
        factorization.clear();
        assert_eq!(factorization.required_len, 0);
        factorization.right_solve(&mut [0.0; 2]);
    }

    #[test]
    fn public_update_rejects_out_of_range_dense_gathers() {
        let update = RankOneUpdateElementaryMatrix::new(vec![], vec![(2, 1.0)], 0.0);
        assert!(std::panic::catch_unwind(|| update.right_solve(&mut [1.0, 2.0])).is_err());
    }
}
