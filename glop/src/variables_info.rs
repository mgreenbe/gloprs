//! Variable bounds, statuses, movement flags, and pricing relevance.
//!
//! This follows `ortools/glop/variables_info.{h,cc}`. The matrix is owned here
//! to avoid self-referential solver state in Rust; all maintained invariants
//! and update ordering otherwise mirror GLOP.

#![allow(clippy::float_cmp)]

use lp_data::lp_types::{
    ColBitVec, ColIndex, EntryIndex, INFINITY, RowToColMapping, TypedVec, VariableStatus,
    VariableStatusRow, VariableType, VariableTypeRow, VectorIndex,
};
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasisState {
    pub statuses: VariableStatusRow,
}

impl Default for BasisState {
    fn default() -> Self {
        Self {
            statuses: VariableStatusRow::new(),
        }
    }
}

impl BasisState {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.statuses.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct VariablesInfo {
    matrix: CompactSparseMatrix,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    saved_lower_bounds: Vec<f64>,
    saved_upper_bounds: Vec<f64>,
    variable_status: VariableStatusRow,
    variable_type: VariableTypeRow,
    can_increase: ColBitVec,
    can_decrease: ColBitVec,
    relevance: ColBitVec,
    is_basic: ColBitVec,
    not_basic: ColBitVec,
    non_basic_boxed_variables: ColBitVec,
    num_entries_in_relevant_columns: EntryIndex,
    boxed_variables_are_relevant: bool,
    in_dual_phase_one: bool,
}

impl VariablesInfo {
    #[must_use]
    pub fn new(matrix: &SparseMatrix) -> Self {
        let compact = CompactSparseMatrix::from_sparse(matrix);
        let num_cols = compact.num_cols();
        Self {
            matrix: compact,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            saved_lower_bounds: Vec::new(),
            saved_upper_bounds: Vec::new(),
            variable_status: VariableStatusRow::new(),
            variable_type: VariableTypeRow::new(),
            can_increase: ColBitVec::new(num_cols),
            can_decrease: ColBitVec::new(num_cols),
            relevance: ColBitVec::new(num_cols),
            is_basic: ColBitVec::new(num_cols),
            not_basic: ColBitVec::new(num_cols),
            non_basic_boxed_variables: ColBitVec::new(num_cols),
            num_entries_in_relevant_columns: EntryIndex::new(0),
            boxed_variables_are_relevant: true,
            in_dual_phase_one: false,
        }
    }

    /// Loads equation-form bounds and returns whether they were unchanged.
    ///
    /// # Panics
    ///
    /// Panics when the bound vectors do not match the matrix column count.
    pub fn load_bounds_and_return_true_if_unchanged(
        &mut self,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> bool {
        let num_cols = self.matrix.num_cols().to_usize();
        assert_eq!(lower_bounds.len(), num_cols);
        assert_eq!(upper_bounds.len(), num_cols);
        if self.lower_bounds == lower_bounds && self.upper_bounds == upper_bounds {
            return true;
        }
        self.lower_bounds.clear();
        self.lower_bounds.extend_from_slice(lower_bounds);
        self.upper_bounds.clear();
        self.upper_bounds.extend_from_slice(upper_bounds);
        self.recompute_variable_types();
        false
    }

    /// Loads structural-variable and constraint bounds, adding negated slack
    /// bounds in GLOP's equation-form convention.
    ///
    /// # Panics
    ///
    /// Panics when dimensions or lower/upper bound pairs disagree.
    pub fn load_bounds_from_lp(
        &mut self,
        variable_lower_bounds: &[f64],
        variable_upper_bounds: &[f64],
        constraint_lower_bounds: &[f64],
        constraint_upper_bounds: &[f64],
    ) -> bool {
        assert_eq!(variable_lower_bounds.len(), variable_upper_bounds.len());
        assert_eq!(constraint_lower_bounds.len(), constraint_upper_bounds.len());
        assert_eq!(
            self.matrix.num_cols().to_usize(),
            variable_lower_bounds.len() + constraint_lower_bounds.len()
        );
        let num_cols = self.matrix.num_cols();
        let num_variables = variable_lower_bounds.len();
        let mut is_unchanged = self.lower_bounds.len() == num_cols.to_usize();
        self.lower_bounds.resize(num_cols.to_usize(), 0.0);
        self.upper_bounds.resize(num_cols.to_usize(), 0.0);
        self.variable_type
            .resize(num_cols, VariableType::FixedVariable);

        // Copy structural-variable bounds, recomputing a type only when that
        // column changed. This is the same incremental path as GLOP.
        for column in 0..num_variables {
            let lower = variable_lower_bounds[column];
            let upper = variable_upper_bounds[column];
            if self.lower_bounds[column] != lower || self.upper_bounds[column] != upper {
                self.lower_bounds[column] = lower;
                self.upper_bounds[column] = upper;
                is_unchanged = false;
                let column = ColIndex::from_usize(column);
                self.variable_type[column] = self.compute_variable_type(column);
            }
        }

        // Equation-form slack bounds are the negated constraint bounds in
        // reverse lower/upper order.
        for row in 0..constraint_lower_bounds.len() {
            let column_index = num_variables + row;
            let lower = -constraint_upper_bounds[row];
            let upper = -constraint_lower_bounds[row];
            if self.lower_bounds[column_index] != lower || self.upper_bounds[column_index] != upper
            {
                self.lower_bounds[column_index] = lower;
                self.upper_bounds[column_index] = upper;
                is_unchanged = false;
                let column = ColIndex::from_usize(column_index);
                self.variable_type[column] = self.compute_variable_type(column);
            }
        }
        is_unchanged
    }

    /// Recomputes variable types after callers mutate the internal bound
    /// storage through [`Self::mutable_lower_bounds`] and
    /// [`Self::mutable_upper_bounds`].
    ///
    /// # Panics
    ///
    /// Panics when either bound vector does not match the matrix column count,
    /// or when a mutated lower bound exceeds its corresponding upper bound.
    pub fn initialize_from_mutated_state(&mut self) {
        assert_eq!(self.lower_bounds.len(), self.matrix.num_cols().to_usize());
        assert_eq!(self.upper_bounds.len(), self.matrix.num_cols().to_usize());
        self.variable_type
            .resize(self.matrix.num_cols(), VariableType::Unconstrained);
        for column in 0..self.matrix.num_cols().to_usize() {
            let column = ColIndex::from_usize(column);
            self.variable_type[column] = self.compute_variable_type(column);
        }
    }

    /// Returns the owned lower-bound storage for zero-copy incremental solves.
    pub fn mutable_lower_bounds(&mut self) -> &mut Vec<f64> {
        &mut self.lower_bounds
    }

    /// Returns the owned upper-bound storage for zero-copy incremental solves.
    pub fn mutable_upper_bounds(&mut self) -> &mut Vec<f64> {
        &mut self.upper_bounds
    }

    /// Initializes statuses from a possibly smaller warm-start state.
    ///
    /// # Panics
    ///
    /// Panics when more new columns are specified than precede the slacks.
    pub fn initialize_from_basis_state(
        &mut self,
        first_slack: usize,
        num_new_columns: usize,
        state: &BasisState,
    ) {
        self.reset_status_info();
        let num_cols = self.matrix.num_cols().to_usize();
        assert!(num_new_columns <= first_slack);
        let first_new_column = first_slack - num_new_columns;
        for column in 0..num_cols {
            let status = if column < first_new_column && column < state.statuses.len().to_usize() {
                Some(state.statuses[ColIndex::from_usize(column)])
            } else if column >= first_slack
                && column - num_new_columns < state.statuses.len().to_usize()
            {
                Some(state.statuses[ColIndex::from_usize(column - num_new_columns)])
            } else {
                None
            };
            let column = ColIndex::from_usize(column);
            match status {
                Some(VariableStatus::Basic) => {
                    self.variable_status[column] = VariableStatus::Basic;
                    self.is_basic.set(column);
                }
                Some(VariableStatus::AtLowerBound) => {
                    let corrected = if self.lower_bounds[column.to_usize()]
                        == self.upper_bounds[column.to_usize()]
                    {
                        VariableStatus::FixedValue
                    } else if self.lower_bounds[column.to_usize()] == -INFINITY {
                        self.default_variable_status(column)
                    } else {
                        VariableStatus::AtLowerBound
                    };
                    self.update_to_nonbasic_status(column, corrected);
                }
                Some(VariableStatus::AtUpperBound) => {
                    let corrected = if self.lower_bounds[column.to_usize()]
                        == self.upper_bounds[column.to_usize()]
                    {
                        VariableStatus::FixedValue
                    } else if self.upper_bounds[column.to_usize()] == INFINITY {
                        self.default_variable_status(column)
                    } else {
                        VariableStatus::AtUpperBound
                    };
                    self.update_to_nonbasic_status(column, corrected);
                }
                _ => {
                    let default = self.default_variable_status(column);
                    self.update_to_nonbasic_status(column, default);
                }
            }
        }
    }

    pub fn initialize_to_default_status(&mut self) {
        self.reset_status_info();
        for column in 0..self.matrix.num_cols().to_usize() {
            let column = ColIndex::from_usize(column);
            let status = self.default_variable_status(column);
            self.update_to_nonbasic_status(column, status);
        }
    }

    pub fn change_unused_basic_variables_to_free(&mut self, basis: &RowToColMapping) -> usize {
        self.is_basic.clear_and_resize(self.matrix.num_cols());
        for &column in basis {
            self.update_to_basic_status(column);
        }
        let mut changed = 0;
        for column in 0..self.matrix.num_cols().to_usize() {
            let column = ColIndex::from_usize(column);
            if !self.is_basic.contains(column)
                && self.variable_status[column] == VariableStatus::Basic
            {
                changed += 1;
                let status = if self.variable_type[column] == VariableType::FixedVariable {
                    VariableStatus::FixedValue
                } else {
                    VariableStatus::Free
                };
                self.update_to_nonbasic_status(column, status);
            }
        }
        changed
    }

    pub fn snap_free_variables_to_bound(
        &mut self,
        distance: f64,
        starting_values: &[f64],
    ) -> usize {
        let mut changed = 0;
        for column in 0..self.matrix.num_cols().to_usize() {
            let index = ColIndex::from_usize(column);
            if self.variable_status[index] != VariableStatus::Free
                || self.variable_type[index] == VariableType::Unconstrained
            {
                continue;
            }
            let value = starting_values.get(column).copied().unwrap_or(0.0);
            let difference_upper = self.upper_bounds[column] - value;
            let difference_lower = value - self.lower_bounds[column];
            let status = if difference_lower <= difference_upper && difference_lower <= distance {
                Some(VariableStatus::AtLowerBound)
            } else if difference_upper < difference_lower && difference_upper <= distance {
                Some(VariableStatus::AtUpperBound)
            } else {
                None
            };
            if let Some(status) = status {
                changed += 1;
                self.update_to_nonbasic_status(index, status);
            }
        }
        changed
    }

    pub fn update_to_basic_status(&mut self, column: ColIndex) {
        let index = column.to_usize();
        if self.in_dual_phase_one {
            if self.lower_bounds[index] != 0.0 {
                self.lower_bounds[index] = -INFINITY;
            }
            if self.upper_bounds[index] != 0.0 {
                self.upper_bounds[index] = INFINITY;
            }
            self.variable_type[column] = self.compute_variable_type(column);
        }
        self.variable_status[column] = VariableStatus::Basic;
        self.is_basic.set(column);
        self.not_basic.clear_bit(column);
        self.can_increase.clear_bit(column);
        self.can_decrease.clear_bit(column);
        self.non_basic_boxed_variables.clear_bit(column);
        self.set_relevance(column, false);
    }

    /// Updates all status-dependent bitsets for a nonbasic variable.
    ///
    /// # Panics
    ///
    /// Panics when passed [`VariableStatus::Basic`].
    pub fn update_to_nonbasic_status(&mut self, column: ColIndex, status: VariableStatus) {
        assert_ne!(status, VariableStatus::Basic);
        self.variable_status[column] = status;
        self.is_basic.clear_bit(column);
        self.not_basic.set(column);
        self.can_increase.set_to(
            column,
            matches!(status, VariableStatus::AtLowerBound | VariableStatus::Free),
        );
        self.can_decrease.set_to(
            column,
            matches!(status, VariableStatus::AtUpperBound | VariableStatus::Free),
        );
        let boxed = self.variable_type[column] == VariableType::UpperAndLowerBounded;
        self.non_basic_boxed_variables.set_to(column, boxed);
        self.set_relevance(
            column,
            status != VariableStatus::FixedValue && (self.boxed_variables_are_relevant || !boxed),
        );
    }

    pub fn make_boxed_variable_relevant(&mut self, value: bool) {
        if value == self.boxed_variables_are_relevant {
            return;
        }
        self.boxed_variables_are_relevant = value;
        let boxed: Vec<_> = self.non_basic_boxed_variables.iter_ones().collect();
        for column in boxed {
            self.set_relevance(
                column,
                value && self.variable_type[column] != VariableType::FixedVariable,
            );
        }
    }

    /// Replaces bounds with GLOP's auxiliary dual phase-I bounds.
    ///
    /// # Panics
    ///
    /// Panics if dual phase I is already active or dimensions disagree.
    pub fn transform_to_dual_phase_one_problem(
        &mut self,
        dual_feasibility_tolerance: f64,
        reduced_costs: &[f64],
    ) {
        assert!(!self.in_dual_phase_one);
        assert_eq!(reduced_costs.len(), self.matrix.num_cols().to_usize());
        self.in_dual_phase_one = true;
        self.saved_lower_bounds.clone_from(&self.lower_bounds);
        self.saved_upper_bounds.clone_from(&self.upper_bounds);
        for (column, &reduced_cost) in reduced_costs.iter().enumerate() {
            let index = ColIndex::from_usize(column);
            match self.variable_type[index] {
                VariableType::FixedVariable | VariableType::UpperAndLowerBounded => {
                    self.lower_bounds[column] = 0.0;
                    self.upper_bounds[column] = 0.0;
                    self.variable_type[index] = VariableType::FixedVariable;
                }
                VariableType::LowerBounded => {
                    self.lower_bounds[column] = 0.0;
                    self.upper_bounds[column] = 1.0;
                    self.variable_type[index] = VariableType::UpperAndLowerBounded;
                }
                VariableType::UpperBounded => {
                    self.lower_bounds[column] = -1.0;
                    self.upper_bounds[column] = 0.0;
                    self.variable_type[index] = VariableType::UpperAndLowerBounded;
                }
                VariableType::Unconstrained => {
                    self.lower_bounds[column] = -1000.0;
                    self.upper_bounds[column] = 1000.0;
                    self.variable_type[index] = VariableType::UpperAndLowerBounded;
                }
            }
            if self.variable_type[index] == VariableType::UpperAndLowerBounded {
                if reduced_cost > dual_feasibility_tolerance {
                    self.variable_status[index] = VariableStatus::AtLowerBound;
                } else if reduced_cost < -dual_feasibility_tolerance {
                    self.variable_status[index] = VariableStatus::AtUpperBound;
                }
            }
            self.update_status_for_new_type(index);
        }
    }

    /// Restores the original bounds and reconciles statuses after dual phase I.
    ///
    /// # Panics
    ///
    /// Panics unless dual phase I is active or the reduced-cost size differs.
    pub fn end_dual_phase_one(&mut self, dual_feasibility_tolerance: f64, reduced_costs: &[f64]) {
        assert!(self.in_dual_phase_one);
        assert_eq!(reduced_costs.len(), self.matrix.num_cols().to_usize());
        self.in_dual_phase_one = false;
        std::mem::swap(&mut self.saved_lower_bounds, &mut self.lower_bounds);
        std::mem::swap(&mut self.saved_upper_bounds, &mut self.upper_bounds);
        // Preserve the pinned code's exact two-swap behavior: the phase-I
        // upper allocation is released, while the former lower allocation is
        // left in saved_upper_bounds (despite the upstream comment saying the
        // saved storage is cleared). The next phase-I transform overwrites it.
        let phase_one_lower = std::mem::take(&mut self.saved_lower_bounds);
        self.saved_upper_bounds = phase_one_lower;
        for (column, &reduced_cost) in reduced_costs.iter().enumerate() {
            let index = ColIndex::from_usize(column);
            self.variable_type[index] = self.compute_variable_type(index);
            if self.variable_type[index] == VariableType::UpperAndLowerBounded {
                if reduced_cost > dual_feasibility_tolerance {
                    self.variable_status[index] = VariableStatus::AtLowerBound;
                } else if reduced_cost < -dual_feasibility_tolerance {
                    self.variable_status[index] = VariableStatus::AtUpperBound;
                }
            }
            self.update_status_for_new_type(index);
        }
    }

    #[must_use]
    pub fn variable_types(&self) -> &VariableTypeRow {
        &self.variable_type
    }

    #[must_use]
    pub fn variable_statuses(&self) -> &VariableStatusRow {
        &self.variable_status
    }

    #[must_use]
    pub const fn can_increase(&self) -> &ColBitVec {
        &self.can_increase
    }

    #[must_use]
    pub const fn can_decrease(&self) -> &ColBitVec {
        &self.can_decrease
    }

    #[must_use]
    pub const fn relevance(&self) -> &ColBitVec {
        &self.relevance
    }

    #[must_use]
    pub const fn is_basic(&self) -> &ColBitVec {
        &self.is_basic
    }

    #[must_use]
    pub const fn not_basic(&self) -> &ColBitVec {
        &self.not_basic
    }

    #[must_use]
    pub const fn non_basic_boxed_variables(&self) -> &ColBitVec {
        &self.non_basic_boxed_variables
    }

    #[must_use]
    pub fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds
    }

    #[must_use]
    pub fn upper_bounds(&self) -> &[f64] {
        &self.upper_bounds
    }

    #[must_use]
    pub fn num_columns(&self) -> ColIndex {
        self.matrix.num_cols()
    }

    #[must_use]
    pub const fn num_entries_in_relevant_columns(&self) -> EntryIndex {
        self.num_entries_in_relevant_columns
    }

    #[must_use]
    pub fn bound_difference(&self, column: ColIndex) -> f64 {
        self.upper_bounds[column.to_usize()] - self.lower_bounds[column.to_usize()]
    }

    #[must_use]
    pub fn relevance_as_dense(&self) -> Vec<bool> {
        (0..self.matrix.num_cols().to_usize())
            .map(|column| self.relevance.contains(ColIndex::from_usize(column)))
            .collect()
    }

    fn reset_status_info(&mut self) {
        let num_cols = self.matrix.num_cols();
        self.variable_status = TypedVec::filled(num_cols, VariableStatus::Free);
        self.can_increase.clear_and_resize(num_cols);
        self.can_decrease.clear_and_resize(num_cols);
        self.is_basic.clear_and_resize(num_cols);
        self.not_basic.clear_and_resize(num_cols);
        self.non_basic_boxed_variables.clear_and_resize(num_cols);
        self.boxed_variables_are_relevant = true;
        self.num_entries_in_relevant_columns = EntryIndex::new(0);
        self.relevance.clear_and_resize(num_cols);
    }

    fn recompute_variable_types(&mut self) {
        self.variable_type = TypedVec::filled(self.matrix.num_cols(), VariableType::Unconstrained);
        for column in 0..self.matrix.num_cols().to_usize() {
            let column = ColIndex::from_usize(column);
            self.variable_type[column] = self.compute_variable_type(column);
        }
    }

    fn default_variable_status(&self, column: ColIndex) -> VariableStatus {
        let lower = self.lower_bounds[column.to_usize()];
        let upper = self.upper_bounds[column.to_usize()];
        if lower == upper {
            VariableStatus::FixedValue
        } else if lower == -INFINITY && upper == INFINITY {
            VariableStatus::Free
        } else if lower.abs() <= upper.abs() {
            VariableStatus::AtLowerBound
        } else {
            VariableStatus::AtUpperBound
        }
    }

    fn compute_variable_type(&self, column: ColIndex) -> VariableType {
        let lower = self.lower_bounds[column.to_usize()];
        let upper = self.upper_bounds[column.to_usize()];
        assert!(lower <= upper);
        if lower == -INFINITY {
            if upper == INFINITY {
                VariableType::Unconstrained
            } else {
                VariableType::UpperBounded
            }
        } else if upper == INFINITY {
            VariableType::LowerBounded
        } else if lower == upper {
            VariableType::FixedVariable
        } else {
            VariableType::UpperAndLowerBounded
        }
    }

    fn set_relevance(&mut self, column: ColIndex, relevance: bool) {
        if self.relevance.contains(column) == relevance {
            return;
        }
        self.relevance.set_to(column, relevance);
        let entries = i64::try_from(self.matrix.column(column).len()).unwrap_or(i64::MAX);
        self.num_entries_in_relevant_columns = EntryIndex::new(
            self.num_entries_in_relevant_columns.value()
                + if relevance { entries } else { -entries },
        );
    }

    fn update_status_for_new_type(&mut self, column: ColIndex) {
        let status = self.variable_status[column];
        match status {
            VariableStatus::Basic => self.update_to_basic_status(column),
            VariableStatus::AtLowerBound => {
                if self.lower_bounds[column.to_usize()] == self.upper_bounds[column.to_usize()] {
                    self.update_to_nonbasic_status(column, VariableStatus::FixedValue);
                } else if self.lower_bounds[column.to_usize()] == -INFINITY {
                    let default = self.default_variable_status(column);
                    self.update_to_nonbasic_status(column, default);
                } else {
                    self.update_to_nonbasic_status(column, status);
                }
            }
            VariableStatus::AtUpperBound => {
                if self.lower_bounds[column.to_usize()] == self.upper_bounds[column.to_usize()] {
                    self.update_to_nonbasic_status(column, VariableStatus::FixedValue);
                } else if self.upper_bounds[column.to_usize()] == INFINITY {
                    let default = self.default_variable_status(column);
                    self.update_to_nonbasic_status(column, default);
                } else {
                    self.update_to_nonbasic_status(column, status);
                }
            }
            VariableStatus::FixedValue | VariableStatus::Free => {
                let default = self.default_variable_status(column);
                self.update_to_nonbasic_status(column, default);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use lp_data::lp_types::RowIndex;

    use super::*;

    fn matrix() -> SparseMatrix {
        let mut matrix = SparseMatrix::new();
        matrix.populate_from_zero(RowIndex::new(2), ColIndex::new(5));
        for column in 0..5 {
            matrix
                .mutable_column(ColIndex::new(column))
                .add_entry(RowIndex::new(column % 2), 1.0);
        }
        matrix.clean_up();
        matrix
    }

    #[test]
    fn defaults_and_status_updates_maintain_all_bitsets() {
        let mut info = VariablesInfo::new(&matrix());
        assert!(!info.load_bounds_and_return_true_if_unchanged(
            &[-INFINITY, 0.0, -2.0, 3.0, 1.0],
            &[INFINITY, INFINITY, 5.0, 3.0, 4.0],
        ));
        info.initialize_to_default_status();
        assert_eq!(
            info.variable_statuses()[ColIndex::new(0)],
            VariableStatus::Free
        );
        assert_eq!(
            info.variable_statuses()[ColIndex::new(1)],
            VariableStatus::AtLowerBound
        );
        assert_eq!(
            info.variable_statuses()[ColIndex::new(2)],
            VariableStatus::AtLowerBound
        );
        assert_eq!(
            info.variable_statuses()[ColIndex::new(3)],
            VariableStatus::FixedValue
        );
        assert!(info.relevance().contains(ColIndex::new(4)));
        assert!(!info.relevance().contains(ColIndex::new(3)));
        assert_eq!(info.num_entries_in_relevant_columns().value(), 4);

        info.update_to_basic_status(ColIndex::new(1));
        assert!(info.is_basic().contains(ColIndex::new(1)));
        assert!(!info.not_basic().contains(ColIndex::new(1)));
        assert!(!info.relevance().contains(ColIndex::new(1)));
    }

    #[test]
    fn boxed_relevance_and_dual_phase_one_round_trip_match_upstream_rules() {
        let mut info = VariablesInfo::new(&matrix());
        info.load_bounds_and_return_true_if_unchanged(
            &[-INFINITY, 0.0, -2.0, 3.0, 1.0],
            &[INFINITY, INFINITY, 5.0, 3.0, 4.0],
        );
        info.initialize_to_default_status();
        info.make_boxed_variable_relevant(false);
        assert!(!info.relevance().contains(ColIndex::new(2)));
        assert!(!info.relevance().contains(ColIndex::new(4)));
        let lower = info.lower_bounds().to_vec();
        let upper = info.upper_bounds().to_vec();
        info.transform_to_dual_phase_one_problem(1e-7, &[0.0, 1.0, -1.0, 0.0, 0.0]);
        info.end_dual_phase_one(1e-7, &[0.0, 1.0, -1.0, 0.0, 0.0]);
        assert_eq!(info.lower_bounds(), lower);
        assert_eq!(info.upper_bounds(), upper);
    }

    #[test]
    fn mutated_state_and_structural_slack_loading_match_upstream_contracts() {
        let mut info = VariablesInfo::new(&matrix());
        info.mutable_lower_bounds()
            .extend_from_slice(&[-INFINITY, 0.0, -2.0, 3.0, 1.0]);
        info.mutable_upper_bounds()
            .extend_from_slice(&[INFINITY, INFINITY, 5.0, 3.0, 4.0]);
        info.initialize_from_mutated_state();
        assert_eq!(
            info.variable_types().as_slice(),
            &[
                VariableType::Unconstrained,
                VariableType::LowerBounded,
                VariableType::UpperAndLowerBounded,
                VariableType::FixedVariable,
                VariableType::UpperAndLowerBounded,
            ]
        );

        assert!(!info.load_bounds_from_lp(
            &[-INFINITY, 0.0, -2.0],
            &[INFINITY, INFINITY, 5.0],
            &[-4.0, -4.0],
            &[-3.0, -1.0],
        ));
        assert_eq!(info.lower_bounds(), &[-INFINITY, 0.0, -2.0, 3.0, 1.0]);
        assert_eq!(info.upper_bounds(), &[INFINITY, INFINITY, 5.0, 4.0, 4.0]);
        assert!(info.load_bounds_from_lp(
            &[-INFINITY, 0.0, -2.0],
            &[INFINITY, INFINITY, 5.0],
            &[-4.0, -4.0],
            &[-3.0, -1.0],
        ));

        assert!(BasisState::default().is_empty());
    }
}
