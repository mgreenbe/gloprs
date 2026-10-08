//! Linear-program model representation.
//!
//! This is the solver-facing subset of upstream `ortools/lp_data/lp_data.*`.

use std::cell::{Cell, Ref, RefCell};
use std::collections::HashMap;

use crate::lp_print_utils::{stringify, stringify_default, stringify_monomial};
use crate::lp_types::{
    ColIndex, ConstraintStatus, ConstraintStatusColumn, DenseBooleanColumn, DenseBooleanRow,
    DenseColumn, DenseRow, EPSILON, EntryIndex, Fractional, INFINITY, INVALID_COL, INVALID_ROW,
    ProblemStatus, RowIndex, TypedVec, VariableStatus, VariableStatusRow, VectorIndex,
};
use crate::matrix_scaler::SparseMatrixScaler;
use crate::permutation::{ColumnPermutation, RowPermutation};
use crate::sparse::SparseMatrix;
use crate::sparse_vector::SparseColumn;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ModelVariableType {
    #[default]
    Continuous,
    Integer,
    ImpliedInteger,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CostScalingAlgorithm {
    #[default]
    NoCostScaling,
    ContainOneCostScaling,
    MeanCostScaling,
    MedianCostScaling,
}

#[derive(Clone, Debug)]
pub struct LinearProgram {
    name: String,
    matrix: SparseMatrix,
    constraint_lower_bounds: DenseColumn,
    constraint_upper_bounds: DenseColumn,
    objective_coefficients: DenseRow,
    variable_lower_bounds: DenseRow,
    variable_upper_bounds: DenseRow,
    variable_types: TypedVec<ColIndex, ModelVariableType>,
    variable_names: TypedVec<ColIndex, String>,
    constraint_names: TypedVec<RowIndex, String>,
    variable_ids: HashMap<String, ColIndex>,
    constraint_ids: HashMap<String, RowIndex>,
    objective_offset: Fractional,
    objective_scaling_factor: Fractional,
    maximize: bool,
    first_slack_variable: Option<ColIndex>,
    transpose_matrix: RefCell<SparseMatrix>,
    transpose_matrix_is_consistent: Cell<bool>,
    columns_are_known_to_be_clean: Cell<bool>,
    integer_variables: RefCell<Vec<ColIndex>>,
    binary_variables: RefCell<Vec<ColIndex>>,
    non_binary_integer_variables: RefCell<Vec<ColIndex>>,
    integer_variable_lists_are_consistent: Cell<bool>,
}

impl Default for LinearProgram {
    fn default() -> Self {
        Self {
            name: String::new(),
            matrix: SparseMatrix::new(),
            constraint_lower_bounds: DenseColumn::new(),
            constraint_upper_bounds: DenseColumn::new(),
            objective_coefficients: DenseRow::new(),
            variable_lower_bounds: DenseRow::new(),
            variable_upper_bounds: DenseRow::new(),
            variable_types: TypedVec::new(),
            variable_names: TypedVec::new(),
            constraint_names: TypedVec::new(),
            variable_ids: HashMap::new(),
            constraint_ids: HashMap::new(),
            objective_offset: 0.0,
            objective_scaling_factor: 1.0,
            maximize: false,
            first_slack_variable: None,
            transpose_matrix: RefCell::new(SparseMatrix::new()),
            transpose_matrix_is_consistent: Cell::new(true),
            columns_are_known_to_be_clean: Cell::new(true),
            integer_variables: RefCell::new(Vec::new()),
            binary_variables: RefCell::new(Vec::new()),
            non_binary_integer_variables: RefCell::new(Vec::new()),
            integer_variable_lists_are_consistent: Cell::new(true),
        }
    }
}

impl PartialEq for LinearProgram {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.matrix == other.matrix
            && self.constraint_lower_bounds == other.constraint_lower_bounds
            && self.constraint_upper_bounds == other.constraint_upper_bounds
            && self.objective_coefficients == other.objective_coefficients
            && self.variable_lower_bounds == other.variable_lower_bounds
            && self.variable_upper_bounds == other.variable_upper_bounds
            && self.variable_types == other.variable_types
            && self.variable_names == other.variable_names
            && self.constraint_names == other.constraint_names
            && self.variable_ids == other.variable_ids
            && self.constraint_ids == other.constraint_ids
            && self.objective_offset == other.objective_offset
            && self.objective_scaling_factor == other.objective_scaling_factor
            && self.maximize == other.maximize
            && self.first_slack_variable == other.first_slack_variable
    }
}

impl LinearProgram {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn create_new_variable(&mut self) -> ColIndex {
        debug_assert!(self.first_slack_variable.is_none());
        let column = self.matrix.append_empty_column();
        self.objective_coefficients.push(0.0);
        self.variable_lower_bounds.push(0.0);
        self.variable_upper_bounds.push(INFINITY);
        self.variable_types.push(ModelVariableType::Continuous);
        self.variable_names.push(String::new());
        self.transpose_matrix_is_consistent.set(false);
        column
    }

    pub fn create_new_slack_variable(
        &mut self,
        is_integer: bool,
        lower: Fractional,
        upper: Fractional,
        name: impl Into<String>,
    ) -> ColIndex {
        let column = self.matrix.append_empty_column();
        self.objective_coefficients.push(0.0);
        self.variable_lower_bounds.push(lower);
        self.variable_upper_bounds.push(upper);
        self.variable_types.push(if is_integer {
            ModelVariableType::ImpliedInteger
        } else {
            ModelVariableType::Continuous
        });
        self.variable_names.push(name.into());
        self.transpose_matrix_is_consistent.set(false);
        column
    }

    pub fn create_new_constraint(&mut self) -> RowIndex {
        debug_assert!(self.first_slack_variable.is_none());
        let row = self.constraint_lower_bounds.len();
        self.constraint_lower_bounds.push(0.0);
        self.constraint_upper_bounds.push(0.0);
        self.constraint_names.push(String::new());
        self.matrix.set_num_rows(self.constraint_lower_bounds.len());
        self.transpose_matrix_is_consistent.set(false);
        row
    }

    pub fn find_or_create_variable(&mut self, id: &str) -> ColIndex {
        if let Some(&column) = self.variable_ids.get(id) {
            return column;
        }
        let column = self.create_new_variable();
        id.clone_into(&mut self.variable_names[column]);
        self.variable_ids.insert(id.to_owned(), column);
        column
    }

    pub fn find_or_create_constraint(&mut self, id: &str) -> RowIndex {
        if let Some(&row) = self.constraint_ids.get(id) {
            return row;
        }
        let row = self.create_new_constraint();
        id.clone_into(&mut self.constraint_names[row]);
        self.constraint_ids.insert(id.to_owned(), row);
        row
    }

    pub fn set_variable_name(&mut self, column: ColIndex, name: impl Into<String>) {
        self.variable_names[column] = name.into();
    }

    pub fn set_constraint_name(&mut self, row: RowIndex, name: impl Into<String>) {
        self.constraint_names[row] = name.into();
    }

    pub fn set_variable_type(&mut self, column: ColIndex, variable_type: ModelVariableType) {
        let was_integer = self.is_variable_integer(column);
        self.variable_types[column] = variable_type;
        if self.is_variable_integer(column) != was_integer {
            self.integer_variable_lists_are_consistent.set(false);
        }
    }

    #[must_use]
    pub fn is_variable_integer(&self, column: ColIndex) -> bool {
        matches!(
            self.variable_types[column],
            ModelVariableType::Integer | ModelVariableType::ImpliedInteger
        )
    }

    #[must_use]
    pub fn is_variable_binary(&self, column: ColIndex) -> bool {
        self.is_variable_integer(column)
            && self.variable_lower_bounds[column] < EPSILON
            && self.variable_lower_bounds[column] > -1.0
            && self.variable_upper_bounds[column] > 1.0 - EPSILON
            && self.variable_upper_bounds[column] < 2.0
    }

    fn update_all_integer_variable_lists(&self) {
        if self.integer_variable_lists_are_consistent.get() {
            return;
        }
        let mut integer_variables = self.integer_variables.borrow_mut();
        let mut binary_variables = self.binary_variables.borrow_mut();
        let mut non_binary_integer_variables = self.non_binary_integer_variables.borrow_mut();
        integer_variables.clear();
        binary_variables.clear();
        non_binary_integer_variables.clear();
        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            if self.is_variable_integer(column) {
                integer_variables.push(column);
                if self.is_variable_binary(column) {
                    binary_variables.push(column);
                } else {
                    non_binary_integer_variables.push(column);
                }
            }
        }
        self.integer_variable_lists_are_consistent.set(true);
    }

    #[must_use]
    pub fn integer_variables(&self) -> Ref<'_, [ColIndex]> {
        self.update_all_integer_variable_lists();
        Ref::map(self.integer_variables.borrow(), Vec::as_slice)
    }

    #[must_use]
    pub fn binary_variables(&self) -> Ref<'_, [ColIndex]> {
        self.update_all_integer_variable_lists();
        Ref::map(self.binary_variables.borrow(), Vec::as_slice)
    }

    #[must_use]
    pub fn non_binary_integer_variables(&self) -> Ref<'_, [ColIndex]> {
        self.update_all_integer_variable_lists();
        Ref::map(self.non_binary_integer_variables.borrow(), Vec::as_slice)
    }

    pub fn set_variable_bounds(&mut self, column: ColIndex, lower: Fractional, upper: Fractional) {
        let was_binary = self.is_variable_binary(column);
        self.variable_lower_bounds[column] = lower;
        self.variable_upper_bounds[column] = upper;
        if self.is_variable_binary(column) != was_binary {
            self.integer_variable_lists_are_consistent.set(false);
        }
    }

    pub fn set_constraint_bounds(&mut self, row: RowIndex, lower: Fractional, upper: Fractional) {
        while self.num_constraints() <= row {
            self.create_new_constraint();
        }
        self.constraint_lower_bounds[row] = lower;
        self.constraint_upper_bounds[row] = upper;
    }

    pub fn set_coefficient(&mut self, row: RowIndex, column: ColIndex, value: Fractional) {
        self.matrix
            .mutable_column(column)
            .set_coefficient(row, value);
        self.columns_are_known_to_be_clean.set(false);
        self.transpose_matrix_is_consistent.set(false);
    }

    pub fn set_objective_coefficient(&mut self, column: ColIndex, value: Fractional) {
        self.objective_coefficients[column] = value;
    }

    pub fn set_objective_offset(&mut self, offset: Fractional) {
        self.objective_offset = offset;
    }

    pub fn set_objective_scaling_factor(&mut self, factor: Fractional) {
        self.objective_scaling_factor = factor;
    }

    pub fn set_maximization_problem(&mut self, maximize: bool) {
        self.maximize = maximize;
    }

    pub fn clean_up(&mut self) {
        if self.columns_are_known_to_be_clean.get() {
            return;
        }
        self.matrix.clean_up();
        self.columns_are_known_to_be_clean.set(true);
        self.transpose_matrix_is_consistent.set(false);
    }

    /// Records that callers have restored every mutable column to canonical
    /// sparse order, matching GLOP's `NotifyThatColumnsAreClean()` contract.
    pub fn notify_that_columns_are_clean(&self) {
        debug_assert!((0..self.num_variables().to_usize()).all(|position| {
            self.sparse_column(ColIndex::from_usize(position))
                .is_cleaned_up()
        }));
        self.columns_are_known_to_be_clean.set(true);
    }

    #[must_use]
    pub fn is_cleaned_up(&self) -> bool {
        if self.columns_are_known_to_be_clean.get() {
            return true;
        }
        let cleaned = self.matrix.is_cleaned_up();
        self.columns_are_known_to_be_clean.set(cleaned);
        cleaned
    }

    #[must_use]
    pub fn num_variables(&self) -> ColIndex {
        self.matrix.num_cols()
    }

    #[must_use]
    pub fn num_constraints(&self) -> RowIndex {
        self.matrix.num_rows()
    }

    #[must_use]
    pub fn num_entries(&self) -> EntryIndex {
        self.matrix.num_entries()
    }

    #[must_use]
    pub fn matrix(&self) -> &SparseMatrix {
        &self.matrix
    }

    #[must_use]
    pub fn transpose_sparse_matrix(&self) -> Ref<'_, SparseMatrix> {
        if !self.transpose_matrix_is_consistent.get() {
            self.transpose_matrix
                .borrow_mut()
                .populate_from_transpose(&self.matrix);
            self.transpose_matrix_is_consistent.set(true);
        }
        debug_assert_eq!(
            self.transpose_matrix.borrow().num_rows().value(),
            self.matrix.num_cols().value()
        );
        debug_assert_eq!(
            self.transpose_matrix.borrow().num_cols().value(),
            self.matrix.num_rows().value()
        );
        self.transpose_matrix.borrow()
    }

    pub fn mutable_transpose_sparse_matrix(&mut self) -> &mut SparseMatrix {
        if !self.transpose_matrix_is_consistent.get() {
            self.transpose_matrix
                .get_mut()
                .populate_from_transpose(&self.matrix);
        }
        self.transpose_matrix_is_consistent.set(false);
        self.transpose_matrix.get_mut()
    }

    pub fn use_transpose_matrix_as_reference(&mut self) {
        debug_assert_eq!(
            self.transpose_matrix.get_mut().num_rows().value(),
            self.matrix.num_cols().value()
        );
        debug_assert_eq!(
            self.transpose_matrix.get_mut().num_cols().value(),
            self.matrix.num_rows().value()
        );
        self.matrix
            .populate_from_transpose(self.transpose_matrix.get_mut());
        self.transpose_matrix_is_consistent.set(true);
    }

    pub fn clear_transpose_matrix(&mut self) {
        self.transpose_matrix.get_mut().clear();
        self.transpose_matrix_is_consistent.set(false);
    }

    pub fn populate_from_linear_program(&mut self, source: &Self) {
        self.matrix.populate_from_sparse(&source.matrix);
        if source.transpose_matrix_is_consistent.get() {
            self.transpose_matrix
                .get_mut()
                .populate_from_sparse(&source.transpose_matrix.borrow());
            self.transpose_matrix_is_consistent.set(true);
        } else {
            self.clear_transpose_matrix();
        }
        self.constraint_lower_bounds = source.constraint_lower_bounds.clone();
        self.constraint_upper_bounds = source.constraint_upper_bounds.clone();
        self.constraint_names = source.constraint_names.clone();
        self.constraint_ids.clear();
        self.populate_name_objective_and_variables_from(source);
        self.first_slack_variable = source.first_slack_variable;
    }

    /// Constructs the mathematical dual using GLOP's variable ordering and
    /// returns the duplicate column created for each ranged primal row.
    ///
    /// # Panics
    ///
    /// Panics for a free primal constraint, which upstream reports as a debug
    /// fatal error because this construction does not support free rows.
    #[allow(clippy::float_cmp)] // Bound classes use exact comparisons in GLOP.
    pub fn populate_from_dual(&mut self, primal: &Self) -> TypedVec<RowIndex, ColIndex> {
        let primal_variables = primal.num_variables();
        let primal_constraints = primal.num_constraints();
        self.clear();
        self.set_maximization_problem(true);
        self.set_objective_offset(primal.objective_offset);
        self.set_objective_scaling_factor(primal.objective_scaling_factor);

        // One dual variable y per primal row. Ranged rows initially use their
        // upper bound and receive a duplicate nonnegative column below.
        for position in 0..primal_constraints.to_usize() {
            let row = RowIndex::from_usize(position);
            let column = self.create_new_variable();
            let lower = primal.constraint_lower_bounds[row];
            let upper = primal.constraint_upper_bounds[row];
            if lower == upper {
                self.set_variable_bounds(column, -INFINITY, INFINITY);
                self.set_objective_coefficient(column, lower);
            } else if upper != INFINITY {
                self.set_variable_bounds(column, -INFINITY, 0.0);
                self.set_objective_coefficient(column, upper);
            } else if lower != -INFINITY {
                self.set_variable_bounds(column, 0.0, INFINITY);
                self.set_objective_coefficient(column, lower);
            } else {
                panic!("populate_from_dual does not support free constraints");
            }
        }

        // Lower- and upper-bound columns v and w follow all y columns, exactly
        // in primal-column order within each of the two groups.
        for position in 0..primal_variables.to_usize() {
            let primal_column = ColIndex::from_usize(position);
            let lower = primal.variable_lower_bounds[primal_column];
            if lower != -INFINITY {
                let column = self.create_new_variable();
                self.set_objective_coefficient(column, lower);
                self.set_variable_bounds(column, 0.0, INFINITY);
                self.set_coefficient(RowIndex::from_usize(position), column, 1.0);
            }
        }
        for position in 0..primal_variables.to_usize() {
            let primal_column = ColIndex::from_usize(position);
            let upper = primal.variable_upper_bounds[primal_column];
            if upper != INFINITY {
                let column = self.create_new_variable();
                self.set_objective_coefficient(column, upper);
                self.set_variable_bounds(column, -INFINITY, 0.0);
                self.set_coefficient(RowIndex::from_usize(position), column, 1.0);
            }
        }

        // The leading block is A^T and each dual row is fixed to the primal
        // minimization objective coefficient.
        for position in 0..primal_variables.to_usize() {
            let primal_column = ColIndex::from_usize(position);
            let row = RowIndex::from_usize(position);
            let bound = primal.objective_coefficient_for_minimization(primal_column);
            self.set_constraint_bounds(row, bound, bound);
            for entry in primal.sparse_column(primal_column) {
                self.set_coefficient(
                    row,
                    ColIndex::from_usize(entry.index().to_usize()),
                    entry.coefficient(),
                );
            }
        }

        let mut duplicated_rows = TypedVec::filled(primal_constraints, INVALID_COL);
        for position in 0..primal_constraints.to_usize() {
            let primal_row = RowIndex::from_usize(position);
            let lower = primal.constraint_lower_bounds[primal_row];
            let upper = primal.constraint_upper_bounds[primal_row];
            let free_or_boxed = (lower == -INFINITY && upper == INFINITY)
                || (lower != -INFINITY && upper != INFINITY && lower != upper);
            if free_or_boxed {
                debug_assert!(upper != INFINITY || lower != -INFINITY);
                let column = self.create_new_variable();
                self.set_variable_bounds(column, 0.0, INFINITY);
                self.set_objective_coefficient(column, lower);
                let source = self.matrix.column(ColIndex::from_usize(position)).clone();
                self.matrix
                    .mutable_column(column)
                    .populate_from_sparse_vector(&source);
                duplicated_rows[primal_row] = column;
            }
        }
        self.columns_are_known_to_be_clean.set(true);
        self.transpose_matrix_is_consistent.set(false);
        duplicated_rows
    }

    /// Copies a cleaned model while applying source-to-destination row and
    /// column permutations.
    ///
    /// # Panics
    ///
    /// Panics if the source has slack variables, is not cleaned up, or either
    /// permutation is invalid or has the wrong size.
    pub fn populate_from_permuted_linear_program(
        &mut self,
        source: &Self,
        row_permutation: &RowPermutation,
        column_permutation: &ColumnPermutation,
    ) {
        assert!(source.is_cleaned_up());
        assert_eq!(row_permutation.len(), source.num_constraints());
        assert_eq!(column_permutation.len(), source.num_variables());
        assert!(source.first_slack_variable.is_none());
        assert!(row_permutation.check());
        assert!(column_permutation.check());
        self.clear();

        let mut inverse_column_permutation = ColumnPermutation::default();
        inverse_column_permutation.populate_from_inverse(column_permutation);
        self.matrix.populate_from_permuted_matrix(
            &source.matrix,
            row_permutation,
            &inverse_column_permutation,
        );
        self.clear_transpose_matrix();
        self.constraint_lower_bounds = row_permutation.apply(&source.constraint_lower_bounds);
        self.constraint_upper_bounds = row_permutation.apply(&source.constraint_upper_bounds);
        self.objective_coefficients = column_permutation.apply(&source.objective_coefficients);
        self.variable_lower_bounds = column_permutation.apply(&source.variable_lower_bounds);
        self.variable_upper_bounds = column_permutation.apply(&source.variable_upper_bounds);
        self.variable_types = column_permutation.apply(&source.variable_types);
        self.integer_variable_lists_are_consistent.set(false);
        self.constraint_names = row_permutation.apply(&source.constraint_names);
        self.variable_names = column_permutation.apply(&source.variable_names);
        self.maximize = source.maximize;
        self.objective_offset = source.objective_offset;
        self.objective_scaling_factor = source.objective_scaling_factor;
        self.name.clone_from(&source.name);
    }

    pub fn populate_from_linear_program_variables(&mut self, source: &Self) {
        self.matrix
            .populate_from_zero(RowIndex::new(0), source.num_variables());
        self.first_slack_variable = None;
        self.clear_transpose_matrix();
        self.constraint_lower_bounds.clear();
        self.constraint_upper_bounds.clear();
        self.constraint_names.clear();
        self.constraint_ids.clear();
        self.populate_name_objective_and_variables_from(source);
    }

    fn populate_name_objective_and_variables_from(&mut self, source: &Self) {
        self.objective_coefficients = source.objective_coefficients.clone();
        self.variable_lower_bounds = source.variable_lower_bounds.clone();
        self.variable_upper_bounds = source.variable_upper_bounds.clone();
        self.variable_names = source.variable_names.clone();
        self.variable_types = source.variable_types.clone();
        self.integer_variable_lists_are_consistent
            .set(source.integer_variable_lists_are_consistent.get());
        self.integer_variables
            .get_mut()
            .clone_from(source.integer_variables.borrow().as_ref());
        self.binary_variables
            .get_mut()
            .clone_from(source.binary_variables.borrow().as_ref());
        self.non_binary_integer_variables
            .get_mut()
            .clone_from(source.non_binary_integer_variables.borrow().as_ref());
        self.variable_ids.clear();
        self.maximize = source.maximize;
        self.objective_offset = source.objective_offset;
        self.objective_scaling_factor = source.objective_scaling_factor;
        self.columns_are_known_to_be_clean
            .set(source.columns_are_known_to_be_clean.get());
        self.name.clone_from(&source.name);
    }

    #[must_use]
    pub fn sparse_column(&self, column: ColIndex) -> &SparseColumn {
        self.matrix.column(column)
    }

    pub fn mutable_sparse_column(&mut self, column: ColIndex) -> &mut SparseColumn {
        self.columns_are_known_to_be_clean.set(false);
        self.transpose_matrix_is_consistent.set(false);
        self.matrix.mutable_column(column)
    }

    #[must_use]
    pub fn constraint_lower_bounds(&self) -> &DenseColumn {
        &self.constraint_lower_bounds
    }

    #[must_use]
    pub fn constraint_upper_bounds(&self) -> &DenseColumn {
        &self.constraint_upper_bounds
    }

    #[must_use]
    pub fn objective_coefficients(&self) -> &DenseRow {
        &self.objective_coefficients
    }

    #[must_use]
    pub fn variable_lower_bounds(&self) -> &DenseRow {
        &self.variable_lower_bounds
    }

    #[must_use]
    pub fn variable_upper_bounds(&self) -> &DenseRow {
        &self.variable_upper_bounds
    }

    #[must_use]
    pub fn variable_types(&self) -> &TypedVec<ColIndex, ModelVariableType> {
        &self.variable_types
    }

    #[must_use]
    pub fn variable_type(&self, column: ColIndex) -> ModelVariableType {
        self.variable_types[column]
    }

    #[must_use]
    pub fn variable_name(&self, column: ColIndex) -> String {
        self.variable_names
            .as_slice()
            .get(column.to_usize())
            .filter(|name| !name.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("c{}", column.value()))
    }

    #[must_use]
    pub fn constraint_name(&self, row: RowIndex) -> String {
        self.constraint_names
            .as_slice()
            .get(row.to_usize())
            .filter(|name| !name.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("r{}", row.value()))
    }

    #[must_use]
    pub const fn objective_offset(&self) -> Fractional {
        self.objective_offset
    }

    #[must_use]
    pub const fn objective_scaling_factor(&self) -> Fractional {
        self.objective_scaling_factor
    }

    #[must_use]
    pub const fn is_maximization_problem(&self) -> bool {
        self.maximize
    }

    #[must_use]
    pub fn objective_coefficient_for_minimization(&self, column: ColIndex) -> Fractional {
        if self.maximize {
            -self.objective_coefficients[column]
        } else {
            self.objective_coefficients[column]
        }
    }

    #[must_use]
    pub fn solution_is_within_variable_bounds(
        &self,
        solution: &DenseRow,
        tolerance: Fractional,
    ) -> bool {
        solution.len() == self.num_variables()
            && (0..self.num_variables().to_usize()).all(|position| {
                let column = ColIndex::from_usize(position);
                let value = solution[column];
                value.is_finite()
                    && self.variable_lower_bounds[column] - value <= tolerance
                    && value - self.variable_upper_bounds[column] <= tolerance
            })
    }

    #[must_use]
    pub fn solution_is_lp_feasible(&self, solution: &DenseRow, tolerance: Fractional) -> bool {
        if !self.solution_is_within_variable_bounds(solution, tolerance) {
            return false;
        }
        let transpose = self.transpose_sparse_matrix();
        (0..self.num_constraints().to_usize()).all(|position| {
            let row = RowIndex::from_usize(position);
            let value = crate::lp_utils::sparse_scalar_product(
                solution.as_slice(),
                transpose.column(ColIndex::from_usize(position)),
            );
            value.is_finite()
                && self.constraint_lower_bounds[row] - value <= tolerance
                && value - self.constraint_upper_bounds[row] <= tolerance
        })
    }

    #[must_use]
    pub fn solution_is_integer(&self, solution: &DenseRow, tolerance: Fractional) -> bool {
        solution.len() == self.num_variables()
            && self.integer_variables().iter().copied().all(|column| {
                let value = solution[column];
                value.is_finite() && (value - value.round()).abs() <= tolerance
            })
    }

    #[must_use]
    pub fn solution_is_mip_feasible(&self, solution: &DenseRow, tolerance: Fractional) -> bool {
        self.solution_is_lp_feasible(solution, tolerance)
            && self.solution_is_integer(solution, tolerance)
    }

    #[must_use]
    pub fn apply_objective_scaling_and_offset(&self, value: Fractional) -> Fractional {
        self.objective_scaling_factor * (value + self.objective_offset)
    }

    #[must_use]
    pub fn remove_objective_scaling_and_offset(&self, value: Fractional) -> Fractional {
        value / self.objective_scaling_factor - self.objective_offset
    }

    #[allow(clippy::float_cmp)] // GLOP intentionally tests exact zeros and one.
    pub fn scale_objective(&mut self, method: CostScalingAlgorithm) -> Fractional {
        let nonzero_magnitudes: Vec<_> = self
            .objective_coefficients
            .iter()
            .copied()
            .filter(|&value| value != 0.0)
            .map(Fractional::abs)
            .collect();
        let factor = match method {
            CostScalingAlgorithm::NoCostScaling => 1.0,
            CostScalingAlgorithm::ContainOneCostScaling => {
                let (minimum, maximum) = magnitude_range(&nonzero_magnitudes);
                divisor_so_range_contains_one(minimum, maximum)
            }
            CostScalingAlgorithm::MeanCostScaling => {
                if nonzero_magnitudes.is_empty() {
                    1.0
                } else {
                    nonzero_magnitudes.iter().sum::<Fractional>()
                        / Fractional::from(
                            i32::try_from(nonzero_magnitudes.len()).unwrap_or(i32::MAX),
                        )
                }
            }
            CostScalingAlgorithm::MedianCostScaling => {
                if nonzero_magnitudes.is_empty() {
                    1.0
                } else {
                    let mut magnitudes = nonzero_magnitudes;
                    magnitudes.sort_by(Fractional::total_cmp);
                    magnitudes[magnitudes.len() / 2]
                }
            }
        };
        if factor != 1.0 {
            for coefficient in self.objective_coefficients.as_mut_slice() {
                if *coefficient != 0.0 {
                    *coefficient /= factor;
                }
            }
            self.objective_scaling_factor *= factor;
            self.objective_offset /= factor;
        }
        factor
    }

    #[allow(clippy::float_cmp)] // GLOP intentionally tests exact zeros and one.
    pub fn scale_bounds(&mut self) -> Fractional {
        let mut minimum = INFINITY;
        let mut maximum: Fractional = 0.0;
        for value in self
            .variable_lower_bounds
            .iter()
            .chain(self.variable_upper_bounds.iter())
            .chain(self.constraint_lower_bounds.iter())
            .chain(self.constraint_upper_bounds.iter())
        {
            let magnitude = value.abs();
            if magnitude != 0.0 && magnitude != INFINITY {
                minimum = minimum.min(magnitude);
                maximum = maximum.max(magnitude);
            }
        }
        let factor = divisor_so_range_contains_one(minimum, maximum);
        if factor != 1.0 {
            self.objective_scaling_factor *= factor;
            self.objective_offset /= factor;
            for position in 0..self.num_variables().to_usize() {
                let column = ColIndex::from_usize(position);
                self.set_variable_bounds(
                    column,
                    self.variable_lower_bounds[column] / factor,
                    self.variable_upper_bounds[column] / factor,
                );
            }
            for position in 0..self.num_constraints().to_usize() {
                let row = RowIndex::from_usize(position);
                self.set_constraint_bounds(
                    row,
                    self.constraint_lower_bounds[row] / factor,
                    self.constraint_upper_bounds[row] / factor,
                );
            }
        }
        factor
    }

    /// Scales the matrix and associated vectors using GLOP's default sparse
    /// equilibration factors.
    pub fn scale(&mut self, scaler: &mut SparseMatrixScaler) {
        scaler.init(&self.matrix);
        scaler.scale(&mut self.matrix);
        scaler.scale_row_vector(false, &mut self.objective_coefficients);
        scaler.scale_row_vector(true, &mut self.variable_upper_bounds);
        scaler.scale_row_vector(true, &mut self.variable_lower_bounds);
        scaler.scale_column_vector(false, &mut self.constraint_upper_bounds);
        scaler.scale_column_vector(false, &mut self.constraint_lower_bounds);
        self.transpose_matrix_is_consistent.set(false);
    }

    pub fn update_variable_bounds_to_intersection(
        &mut self,
        lower: &DenseRow,
        upper: &DenseRow,
    ) -> bool {
        debug_assert_eq!(lower.len(), self.num_variables());
        debug_assert_eq!(upper.len(), self.num_variables());
        let mut new_lower = DenseRow::filled(self.num_variables(), 0.0);
        let mut new_upper = DenseRow::filled(self.num_variables(), 0.0);
        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            new_lower[column] = lower[column].max(self.variable_lower_bounds[column]);
            new_upper[column] = upper[column].min(self.variable_upper_bounds[column]);
            if new_lower[column] > new_upper[column] {
                return false;
            }
        }
        self.variable_lower_bounds = new_lower;
        self.variable_upper_bounds = new_upper;
        true
    }

    pub fn remove_near_zero_entries(&mut self, threshold: Fractional) {
        let old_num_entries = self.num_entries();
        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            self.matrix
                .mutable_column(column)
                .remove_near_zero_entries(threshold);
            if self.objective_coefficients[column].abs() <= threshold {
                self.objective_coefficients[column] = 0.0;
            }
        }
        if self.num_entries() != old_num_entries {
            self.transpose_matrix_is_consistent.set(false);
        }
    }

    #[allow(clippy::float_cmp)] // GLOP requires exactly integral coefficients here.
    pub fn add_slack_variables_where_necessary(&mut self, detect_integer_constraints: bool) {
        self.clean_up();
        let rows = self.num_constraints().to_usize();
        let mut integer_slack = vec![detect_integer_constraints; rows];
        if detect_integer_constraints {
            for position in 0..self.num_variables().to_usize() {
                let column = ColIndex::from_usize(position);
                let integer_variable = self.is_variable_integer(column);
                for entry in self.sparse_column(column) {
                    let row = entry.index().to_usize();
                    integer_slack[row] &=
                        integer_variable && entry.coefficient().round() == entry.coefficient();
                }
            }
        }

        let original_variables = self.num_variables();
        for (position, &has_integer_slack) in integer_slack.iter().enumerate() {
            let row = RowIndex::from_usize(position);
            if self
                .slack_variable(row)
                .is_some_and(|column| column < original_variables)
            {
                continue;
            }
            let column = self.create_new_slack_variable(
                has_integer_slack,
                -self.constraint_upper_bounds[row],
                -self.constraint_lower_bounds[row],
                format!("s{position}"),
            );
            self.matrix.mutable_column(column).set_coefficient(row, 1.0);
            self.constraint_lower_bounds[row] = 0.0;
            self.constraint_upper_bounds[row] = 0.0;
        }
        self.first_slack_variable.get_or_insert(original_variables);
        self.transpose_matrix_is_consistent.set(false);
    }

    #[must_use]
    pub const fn first_slack_variable(&self) -> Option<ColIndex> {
        self.first_slack_variable
    }

    #[must_use]
    pub fn slack_variable(&self, row: RowIndex) -> Option<ColIndex> {
        self.first_slack_variable
            .map(|first| ColIndex::new(first.value() + row.value()))
    }

    /// # Panics
    ///
    /// Panics unless slack variables have been added and `solution` has one
    /// entry per variable, matching upstream's checked preconditions.
    pub fn compute_slack_variable_values(&self, solution: &mut DenseRow) {
        let first = self
            .first_slack_variable
            .expect("slack variables have not been added");
        assert_eq!(solution.len(), self.num_variables());
        let transpose = self.transpose_sparse_matrix();
        for position in 0..self.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            let sum = crate::lp_utils::partial_scalar_product(
                solution.as_slice(),
                transpose.column(ColIndex::from_usize(position)),
                first.to_usize(),
            );
            solution[self.slack_variable(row).expect("missing slack variable")] = -sum;
        }
    }

    #[must_use]
    pub fn is_in_equation_form(&self) -> bool {
        self.first_slack_variable.is_some()
            && self
                .constraint_lower_bounds
                .iter()
                .zip(self.constraint_upper_bounds.iter())
                .all(|(&lower, &upper)| lower == 0.0 && upper == 0.0)
            && self.first_slack_variable.is_some_and(|first| {
                self.num_variables().value() - first.value() == self.num_constraints().value()
            })
            && crate::matrix_utils::is_rightmost_square_matrix_identity(&self.matrix)
    }

    #[must_use]
    pub fn bounds_of_integer_variables_are_integer(&self, tolerance: Fractional) -> bool {
        self.integer_variables().iter().copied().all(|column| {
            let lower = self.variable_lower_bounds[column];
            let upper = self.variable_upper_bounds[column];
            (!lower.is_finite() || (lower - lower.round()).abs() <= tolerance)
                && (!upper.is_finite() || (upper - upper.round()).abs() <= tolerance)
        })
    }

    #[must_use]
    #[allow(clippy::float_cmp)] // GLOP requires exactly integral coefficients here.
    pub fn bounds_of_integer_constraints_are_integer(&self, tolerance: Fractional) -> bool {
        let transpose = self.transpose_sparse_matrix();
        (0..self.num_constraints().to_usize()).all(|position| {
            let row = RowIndex::from_usize(position);
            let integer_constraint = transpose
                .column(ColIndex::from_usize(position))
                .into_iter()
                .all(|entry| {
                    self.is_variable_integer(ColIndex::from_usize(entry.index().to_usize()))
                        && entry.coefficient().round() == entry.coefficient()
                });
            if !integer_constraint {
                return true;
            }
            let lower = self.constraint_lower_bounds[row];
            let upper = self.constraint_upper_bounds[row];
            (!lower.is_finite() || (lower - lower.round()).abs() <= tolerance)
                && (!upper.is_finite() || (upper - upper.round()).abs() <= tolerance)
        })
    }

    pub fn delete_columns(&mut self, deleted: &DenseBooleanRow) {
        if deleted.is_empty() {
            return;
        }
        let old_columns = self.num_variables().to_usize();
        let mut destination_by_source = vec![None; old_columns];
        let mut kept = Vec::with_capacity(old_columns);
        for source in 0..old_columns {
            if source >= deleted.as_slice().len() || !deleted[ColIndex::from_usize(source)] {
                destination_by_source[source] = Some(ColIndex::from_usize(kept.len()));
                kept.push(source);
            }
        }
        self.objective_coefficients = DenseRow::from_vec(
            kept.iter()
                .map(|&i| self.objective_coefficients.as_slice()[i])
                .collect(),
        );
        self.variable_lower_bounds = DenseRow::from_vec(
            kept.iter()
                .map(|&i| self.variable_lower_bounds.as_slice()[i])
                .collect(),
        );
        self.variable_upper_bounds = DenseRow::from_vec(
            kept.iter()
                .map(|&i| self.variable_upper_bounds.as_slice()[i])
                .collect(),
        );
        self.variable_types = TypedVec::from_vec(
            kept.iter()
                .map(|&i| self.variable_types.as_slice()[i])
                .collect(),
        );
        self.variable_names = TypedVec::from_vec(
            kept.iter()
                .map(|&i| self.variable_names.as_slice()[i].clone())
                .collect(),
        );
        self.variable_ids.retain(|_, column| {
            if let Some(destination) = destination_by_source[column.to_usize()] {
                *column = destination;
                true
            } else {
                false
            }
        });
        self.matrix.delete_columns(deleted);
        self.integer_variable_lists_are_consistent.set(false);
        if self.transpose_matrix_is_consistent.get() {
            let mut permutation = RowPermutation::new(RowIndex::from_usize(old_columns));
            for (source, destination) in destination_by_source.iter().enumerate() {
                permutation[RowIndex::from_usize(source)] =
                    destination.map_or(INVALID_ROW, |col| RowIndex::from_usize(col.to_usize()));
            }
            self.transpose_matrix
                .get_mut()
                .delete_rows(RowIndex::from_usize(kept.len()), &permutation);
        }
    }

    pub fn delete_rows(&mut self, deleted: &DenseBooleanColumn) {
        if deleted.is_empty() {
            return;
        }
        let old_rows = self.num_constraints().to_usize();
        let mut permutation = RowPermutation::new(self.num_constraints());
        let mut kept = Vec::with_capacity(old_rows);
        for source in 0..old_rows {
            let row = RowIndex::from_usize(source);
            if source >= deleted.as_slice().len() || !deleted[row] {
                permutation[row] = RowIndex::from_usize(kept.len());
                kept.push(source);
            } else {
                permutation[row] = INVALID_ROW;
            }
        }
        self.constraint_lower_bounds = DenseColumn::from_vec(
            kept.iter()
                .map(|&i| self.constraint_lower_bounds.as_slice()[i])
                .collect(),
        );
        self.constraint_upper_bounds = DenseColumn::from_vec(
            kept.iter()
                .map(|&i| self.constraint_upper_bounds.as_slice()[i])
                .collect(),
        );
        self.constraint_names = TypedVec::from_vec(
            kept.iter()
                .map(|&i| self.constraint_names.as_slice()[i].clone())
                .collect(),
        );
        self.constraint_ids.retain(|_, row| {
            let destination = permutation[*row];
            if destination == INVALID_ROW {
                false
            } else {
                *row = destination;
                true
            }
        });
        self.matrix
            .delete_rows(RowIndex::from_usize(kept.len()), &permutation);
        if self.transpose_matrix_is_consistent.get() {
            let deleted_columns = DenseBooleanRow::from_vec(deleted.as_slice().to_vec());
            self.transpose_matrix
                .get_mut()
                .delete_columns(&deleted_columns);
        }
    }

    /// Appends a block of constraints whose coefficients are stored by column.
    ///
    /// This follows GLOP's `AddConstraints()`: the appended columns need not be
    /// cleaned, so the model's clean flag is invalidated and the cached
    /// transpose is released rather than incrementally extended.
    ///
    /// # Panics
    ///
    /// Panics if the coefficient matrix has a different number of columns or
    /// if the bounds and names do not have one entry per appended row.
    pub fn add_constraints(
        &mut self,
        coefficients: &SparseMatrix,
        lower_bounds: &DenseColumn,
        upper_bounds: &DenseColumn,
        names: &TypedVec<RowIndex, String>,
    ) {
        assert_eq!(self.num_variables(), coefficients.num_cols());
        assert_eq!(coefficients.num_rows(), lower_bounds.len());
        assert_eq!(coefficients.num_rows(), upper_bounds.len());
        assert_eq!(coefficients.num_rows(), names.len());
        assert!(self.matrix.append_rows_from_sparse(coefficients));
        self.clear_transpose_matrix();
        self.columns_are_known_to_be_clean.set(false);
        for &bound in lower_bounds {
            self.constraint_lower_bounds.push(bound);
        }
        for &bound in upper_bounds {
            self.constraint_upper_bounds.push(bound);
        }
        for name in names {
            self.constraint_names.push(name.clone());
        }
    }

    pub fn add_constraints_with_slack_variables(
        &mut self,
        coefficients: &SparseMatrix,
        lower_bounds: &DenseColumn,
        upper_bounds: &DenseColumn,
        names: &TypedVec<RowIndex, String>,
        detect_integer_constraints: bool,
    ) {
        self.add_constraints(coefficients, lower_bounds, upper_bounds, names);
        self.add_slack_variables_where_necessary(detect_integer_constraints);
    }

    /// # Panics
    ///
    /// Panics unless slack variables have previously been added.
    pub fn delete_slack_variables(&mut self) {
        let first = self
            .first_slack_variable
            .expect("slack variables have not been added");
        let mut deleted = DenseBooleanRow::filled(self.num_variables(), false);
        for position in first.to_usize()..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            let slack = self.sparse_column(column);
            debug_assert_eq!(slack.num_entries(), 1);
            let row = slack.entry(0).index();
            self.constraint_lower_bounds[row] = -self.variable_upper_bounds[column];
            self.constraint_upper_bounds[row] = -self.variable_lower_bounds[column];
            deleted[column] = true;
        }
        self.delete_columns(&deleted);
        self.first_slack_variable = None;
    }

    /// Checks dimensions, bounds, coefficients, and objective metadata.
    ///
    /// # Errors
    ///
    /// Returns a description of the first inconsistent or invalid value.
    pub fn validate(&self) -> Result<(), String> {
        let columns = self.num_variables().to_usize();
        let rows = self.num_constraints().to_usize();
        for (name, len) in [
            (
                "objective coefficients",
                self.objective_coefficients.as_slice().len(),
            ),
            (
                "variable lower bounds",
                self.variable_lower_bounds.as_slice().len(),
            ),
            (
                "variable upper bounds",
                self.variable_upper_bounds.as_slice().len(),
            ),
            ("variable types", self.variable_types.as_slice().len()),
            ("variable names", self.variable_names.as_slice().len()),
        ] {
            if len != columns {
                return Err(format!(
                    "{name} length {len} differs from {columns} columns"
                ));
            }
        }
        for (name, len) in [
            (
                "constraint lower bounds",
                self.constraint_lower_bounds.as_slice().len(),
            ),
            (
                "constraint upper bounds",
                self.constraint_upper_bounds.as_slice().len(),
            ),
            ("constraint names", self.constraint_names.as_slice().len()),
        ] {
            if len != rows {
                return Err(format!("{name} length {len} differs from {rows} rows"));
            }
        }
        for column in 0..columns {
            let column = ColIndex::from_usize(column);
            validate_bounds(
                self.variable_lower_bounds[column],
                self.variable_upper_bounds[column],
            )?;
            for entry in self.sparse_column(column) {
                if entry.index().to_usize() >= rows || !entry.coefficient().is_finite() {
                    return Err(format!("invalid matrix entry in column {column}"));
                }
            }
        }
        for row in 0..rows {
            let row = RowIndex::from_usize(row);
            validate_bounds(
                self.constraint_lower_bounds[row],
                self.constraint_upper_bounds[row],
            )?;
        }
        if !self.objective_offset.is_finite()
            || !self.objective_scaling_factor.is_finite()
            || self.objective_scaling_factor == 0.0
            || self
                .objective_coefficients
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("invalid objective data".to_owned());
        }
        Ok(())
    }

    #[must_use]
    pub fn is_valid(&self, max_valid_magnitude: Fractional) -> bool {
        if !self.objective_offset.is_finite()
            || self.objective_offset.abs() > max_valid_magnitude
            || !self.objective_scaling_factor.is_finite()
            || self.objective_scaling_factor == 0.0
            || self.objective_scaling_factor.abs() > max_valid_magnitude
        {
            return false;
        }
        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            let lower = self.variable_lower_bounds[column];
            let upper = self.variable_upper_bounds[column];
            if !bounds_are_valid(lower, upper)
                || (lower.is_finite() && lower.abs() > max_valid_magnitude)
                || (upper.is_finite() && upper.abs() > max_valid_magnitude)
            {
                return false;
            }
            let objective = self.objective_coefficients[column];
            if !objective.is_finite() || objective.abs() > max_valid_magnitude {
                return false;
            }
            for entry in self.sparse_column(column) {
                if !entry.coefficient().is_finite()
                    || entry.coefficient().abs() > max_valid_magnitude
                {
                    return false;
                }
            }
        }
        if self.constraint_upper_bounds.len() != self.constraint_lower_bounds.len() {
            return false;
        }
        (0..self.constraint_lower_bounds.len().to_usize()).all(|position| {
            let row = RowIndex::from_usize(position);
            let lower = self.constraint_lower_bounds[row];
            let upper = self.constraint_upper_bounds[row];
            bounds_are_valid(lower, upper)
                && (!lower.is_finite() || lower.abs() <= max_valid_magnitude)
                && (!upper.is_finite() || upper.abs() <= max_valid_magnitude)
        })
    }

    #[must_use]
    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            name: self.name.clone(),
            rows: self.num_constraints().to_usize(),
            columns: self.num_variables().to_usize(),
            nonzeros: self.num_entries().value().try_into().unwrap_or(usize::MAX),
            maximize: self.maximize,
            objective_offset: self.objective_offset,
        }
    }

    #[must_use]
    pub fn dimension_string(&self) -> String {
        let (minimum, maximum) = self.matrix.min_and_max_magnitudes();
        format!(
            "{} rows, {} columns, {} entries with magnitude in [{}, {}]",
            self.num_constraints().value(),
            self.num_variables().value(),
            self.num_entries().value(),
            format_printf_e(minimum),
            format_printf_e(maximum)
        )
    }

    #[must_use]
    pub fn objective_stats_string(&self) -> String {
        let mut count = 0_i64;
        let mut minimum = INFINITY;
        let mut maximum = -INFINITY;
        update_value_stats(
            self.objective_coefficients.as_slice(),
            &mut count,
            &mut minimum,
            &mut maximum,
        );
        if count == 0 {
            "No objective term. This is a pure feasibility problem.".to_owned()
        } else {
            format!(
                "{count} non-zeros, range [{}, {}]",
                format_printf_e(minimum),
                format_printf_e(maximum)
            )
        }
    }

    #[must_use]
    pub fn bounds_stats_string(&self) -> String {
        let mut count = 0_i64;
        let mut minimum = INFINITY;
        let mut maximum = -INFINITY;
        for values in [
            self.variable_lower_bounds.as_slice(),
            self.variable_upper_bounds.as_slice(),
            self.constraint_lower_bounds.as_slice(),
            self.constraint_upper_bounds.as_slice(),
        ] {
            update_value_stats(values, &mut count, &mut minimum, &mut maximum);
        }
        if count == 0 {
            "All variables/constraints bounds are zero or +/- infinity.".to_owned()
        } else {
            format!(
                "{count} non-zeros, range [{}, {}]",
                format_printf_e(minimum),
                format_printf_e(maximum)
            )
        }
    }

    /// Emits the `lp_solve` text representation used by GLOP's `Dump()`.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn dump(&self) -> String {
        use std::fmt::Write;

        let mut output = if self.maximize { "max:" } else { "min:" }.to_owned();
        if self.objective_offset != 0.0 {
            output.push_str(&stringify(self.objective_offset));
        }
        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            output.push_str(&stringify_monomial(
                self.objective_coefficients[column],
                &self.variable_name(column),
            ));
        }
        output.push_str(";\n");

        let transpose = self.transpose_sparse_matrix();
        for position in 0..self.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            let lower = self.constraint_lower_bounds[row];
            let upper = self.constraint_upper_bounds[row];
            write!(output, "{}:", self.constraint_name(row)).unwrap();
            if bounds_are_free_or_boxed(lower, upper) {
                write!(output, " {} <=", stringify(lower)).unwrap();
            }
            for entry in transpose.column(ColIndex::from_usize(position)) {
                output.push_str(&stringify_monomial(
                    entry.coefficient(),
                    &self.variable_name(ColIndex::from_usize(entry.index().to_usize())),
                ));
            }
            if bounds_are_free_or_boxed(lower, upper) {
                write!(output, " <= {}", stringify(upper)).unwrap();
            } else if lower == upper {
                write!(output, " = {}", stringify(upper)).unwrap();
            } else if lower != -INFINITY {
                write!(output, " >= {}", stringify(lower)).unwrap();
            } else if lower != INFINITY {
                write!(output, " <= {}", stringify(upper)).unwrap();
            }
            output.push_str(";\n");
        }
        drop(transpose);

        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            let lower = self.variable_lower_bounds[column];
            let upper = self.variable_upper_bounds[column];
            if bounds_are_free_or_boxed(lower, upper) {
                write!(output, "{} <= ", stringify(lower)).unwrap();
            }
            output.push_str(&self.variable_name(column));
            if bounds_are_free_or_boxed(lower, upper) {
                write!(output, " <= {}", stringify(upper)).unwrap();
            } else if lower == upper {
                write!(output, " = {}", stringify(upper)).unwrap();
            } else if lower != -INFINITY {
                write!(output, " >= {}", stringify(lower)).unwrap();
            } else if lower != INFINITY {
                write!(output, " <= {}", stringify(upper)).unwrap();
            }
            output.push_str(";\n");
        }

        let integers = self.integer_variables();
        if !integers.is_empty() {
            output.push_str("int");
            for &column in integers.iter() {
                write!(output, " {}", self.variable_name(column)).unwrap();
            }
            output.push_str(";\n");
        }
        output
    }

    /// Formats a solution as `name = value` pairs in column order.
    ///
    /// # Panics
    ///
    /// Panics if the vector does not contain one value per variable.
    #[must_use]
    pub fn dump_solution(&self, values: &DenseRow) -> String {
        assert_eq!(values.len(), self.num_variables());
        let mut entries = Vec::with_capacity(values.len().to_usize());
        for position in 0..values.len().to_usize() {
            let column = ColIndex::from_usize(position);
            entries.push(format!(
                "{} = {}",
                self.variable_name(column),
                stringify_default(values[column])
            ));
        }
        entries.join(", ")
    }

    /// Returns GLOP's comma-separated structural problem statistics.
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn problem_stats(&self) -> String {
        let mut objective_nonzeros = 0;
        let mut nonnegative = 0;
        let mut boxed = 0;
        let mut free = 0;
        let mut fixed = 0;
        let mut other = 0;
        for position in 0..self.num_variables().to_usize() {
            let column = ColIndex::from_usize(position);
            objective_nonzeros += usize::from(self.objective_coefficients[column] != 0.0);
            let lower = self.variable_lower_bounds[column];
            let upper = self.variable_upper_bounds[column];
            let lower_bounded = lower != -INFINITY;
            let upper_bounded = upper != INFINITY;
            if !lower_bounded && !upper_bounded {
                free += 1;
            } else if lower == 0.0 && !upper_bounded {
                nonnegative += 1;
            } else if !upper_bounded || !lower_bounded {
                other += 1;
            } else if lower == upper {
                fixed += 1;
            } else {
                boxed += 1;
            }
        }

        let mut ranges = 0;
        let mut less_than = 0;
        let mut greater_than = 0;
        let mut equal = 0;
        let mut rhs_nonzeros = 0;
        for position in 0..self.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            let lower = self.constraint_lower_bounds[row];
            let upper = self.constraint_upper_bounds[row];
            if bounds_are_free_or_boxed(lower, upper) {
                ranges += 1;
            } else if lower == upper {
                equal += 1;
                rhs_nonzeros += usize::from(lower != 0.0);
            } else if lower == -INFINITY {
                less_than += 1;
                rhs_nonzeros += usize::from(upper != 0.0);
            } else if upper == INFINITY {
                greater_than += 1;
                rhs_nonzeros += usize::from(lower != 0.0);
            }
        }
        let integers = self.integer_variables().len();
        let binaries = self.binary_variables().len();
        let nonbinary = self.non_binary_integer_variables().len();
        let continuous = self.num_variables().to_usize() - integers;
        format!(
            "{},{},{},{objective_nonzeros},{rhs_nonzeros},{less_than},{greater_than},{equal},{ranges},{nonnegative},{boxed},{free},{fixed},{other},{integers},{binaries},{nonbinary},{continuous}",
            self.num_constraints().value(),
            self.num_variables().value(),
            self.num_entries().value()
        )
    }

    #[must_use]
    pub fn pretty_problem_stats(&self) -> String {
        let values: Vec<_> = self.problem_stats().split(',').map(str::to_owned).collect();
        let labels = [
            "Number of rows",
            "Number of variables in file",
            "Number of entries (non-zeros)",
            "Number of entries in the objective",
            "Number of entries in the right-hand side",
            "Number of <= constraints",
            "Number of >= constraints",
            "Number of = constraints",
            "Number of range constraints",
            "Number of non-negative variables",
            "Number of boxed variables",
            "Number of free variables",
            "Number of fixed variables",
            "Number of other variables",
            "Number of integer variables",
            "Number of binary variables",
            "Number of non-binary integer variables",
            "Number of continuous variables",
        ];
        let mut output = String::new();
        for (label, value) in labels.into_iter().zip(values) {
            use std::fmt::Write;
            writeln!(output, "{label:<45}: {value}").unwrap();
        }
        output
    }

    /// Returns GLOP's comma-separated matrix-density statistics.
    #[must_use]
    pub fn nonzero_stats(&self) -> String {
        let mut rows = vec![0_i64; self.num_constraints().to_usize()];
        let mut columns = vec![0_i64; self.num_variables().to_usize()];
        let mut entries = 0_i64;
        for (position, column_entries) in columns.iter_mut().enumerate() {
            let column = self.sparse_column(ColIndex::from_usize(position));
            *column_entries = i64::try_from(column.num_entries()).unwrap_or(i64::MAX);
            entries += *column_entries;
            for entry in column {
                rows[entry.index().to_usize()] += 1;
            }
        }
        let height = self.num_constraints().to_usize().max(1);
        let width = self.num_variables().to_usize().max(1);
        #[allow(clippy::cast_precision_loss)]
        let fill = 100.0 * entries as f64 / (height * width) as f64;
        let (row_max, row_average, row_deviation) = integer_vector_stats(&rows);
        let (column_max, column_average, column_deviation) = integer_vector_stats(&columns);
        format!(
            "{fill:.2}%,{row_max},{row_average:.2},{row_deviation:.2},{column_max},{column_average:.2},{column_deviation:.2}"
        )
    }

    #[must_use]
    pub fn pretty_nonzero_stats(&self) -> String {
        let values: Vec<_> = self.nonzero_stats().split(',').map(str::to_owned).collect();
        format!(
            "Fill rate                                    : {}\n\
             Entries in row (Max / average / std. dev.)   : {} / {} / {}\n\
             Entries in column (Max / average / std. dev.): {} / {} / {}\n",
            values[0], values[1], values[2], values[3], values[4], values[5], values[6]
        )
    }

    /// Returns a deterministic fingerprint of bounds and objective metadata.
    /// This is used to compare parser output with the pinned native GLOP model.
    #[must_use]
    pub fn data_fingerprint(&self) -> u64 {
        let mut hash = 14_695_981_039_346_656_037_u64;
        let mut add_byte = |byte: u8| {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        };
        add_byte(u8::from(self.maximize));
        for value in std::iter::once(self.objective_offset)
            .chain(std::iter::once(self.objective_scaling_factor))
            .chain(self.objective_coefficients.iter().copied())
            .chain(self.variable_lower_bounds.iter().copied())
            .chain(self.variable_upper_bounds.iter().copied())
            .chain(self.constraint_lower_bounds.iter().copied())
            .chain(self.constraint_upper_bounds.iter().copied())
        {
            for byte in value.to_bits().to_le_bytes() {
                add_byte(byte);
            }
        }
        hash
    }
}

fn validate_bounds(lower: Fractional, upper: Fractional) -> Result<(), String> {
    if lower.is_nan() || upper.is_nan() || lower == INFINITY || upper == -INFINITY || lower > upper
    {
        Err(format!("invalid bounds [{lower}, {upper}]"))
    } else {
        Ok(())
    }
}

#[allow(clippy::float_cmp)] // GLOP's bound classes are exact.
fn bounds_are_valid(lower: Fractional, upper: Fractional) -> bool {
    !(lower.is_nan()
        || upper.is_nan()
        || lower == INFINITY && upper == INFINITY
        || lower == -INFINITY && upper == -INFINITY)
        && lower <= upper
}

#[allow(clippy::float_cmp)]
fn bounds_are_free_or_boxed(lower: Fractional, upper: Fractional) -> bool {
    (lower == -INFINITY && upper == INFINITY)
        || (lower != -INFINITY && upper != INFINITY && lower != upper)
}

fn update_value_stats(values: &[f64], count: &mut i64, minimum: &mut f64, maximum: &mut f64) {
    for &value in values {
        if value == 0.0 || value == INFINITY || value == -INFINITY {
            continue;
        }
        *minimum = minimum.min(value);
        *maximum = maximum.max(value);
        *count += 1;
    }
}

#[allow(clippy::cast_precision_loss)]
fn integer_vector_stats(values: &[i64]) -> (i64, f64, f64) {
    let maximum = values.iter().copied().max().unwrap_or(0);
    let mut count = 0.0;
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for &value in values {
        let sample = value as f64;
        if sample == 0.0 {
            continue;
        }
        count += 1.0;
        sum += sample;
        square_sum += sample * sample;
    }
    if count == 0.0 {
        (maximum, 0.0, 0.0)
    } else {
        (
            maximum,
            sum / count,
            ((square_sum - sum * sum / count) / count).sqrt(),
        )
    }
}

fn format_printf_e(value: f64) -> String {
    let formatted = format!("{value:.6e}");
    let Some((mantissa, exponent)) = formatted.split_once('e') else {
        return formatted;
    };
    let exponent: i32 = exponent.parse().expect("Rust emitted a valid exponent");
    format!("{mantissa}e{exponent:+03}")
}

fn magnitude_range(values: &[Fractional]) -> (Fractional, Fractional) {
    values
        .iter()
        .copied()
        .fold((INFINITY, 0.0_f64), |(minimum, maximum), value| {
            (minimum.min(value), maximum.max(value))
        })
}

fn divisor_so_range_contains_one(minimum: Fractional, maximum: Fractional) -> Fractional {
    if minimum > 1.0 && minimum < INFINITY {
        minimum
    } else if maximum > 0.0 && maximum < 1.0 {
        maximum
    } else {
        1.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelSummary {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub nonzeros: usize,
    pub maximize: bool,
    pub objective_offset: Fractional,
}

impl std::fmt::Display for ModelSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {} rows, {} columns, {} nonzeros, {}, objective offset {:.17e}",
            self.name,
            self.rows,
            self.columns,
            self.nonzeros,
            if self.maximize {
                "maximize"
            } else {
                "minimize"
            },
            self.objective_offset
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProblemSolution {
    pub status: ProblemStatus,
    pub primal_values: DenseRow,
    pub dual_values: DenseColumn,
    pub variable_statuses: VariableStatusRow,
    pub constraint_statuses: ConstraintStatusColumn,
}

impl ProblemSolution {
    #[must_use]
    pub fn new(num_rows: RowIndex, num_columns: ColIndex) -> Self {
        Self {
            status: ProblemStatus::Optimal,
            primal_values: DenseRow::filled(num_columns, 0.0),
            dual_values: DenseColumn::filled(num_rows, 0.0),
            variable_statuses: VariableStatusRow::filled(num_columns, VariableStatus::Free),
            constraint_statuses: ConstraintStatusColumn::filled(num_rows, ConstraintStatus::Free),
        }
    }

    #[must_use]
    pub fn debug_string(&self) -> String {
        use std::fmt::Write;

        let mut output = format!("Problem status: {}", self.status);
        for position in 0..self.primal_values.len().to_usize() {
            let column = ColIndex::from_usize(position);
            write!(
                output,
                "\n  Var #{}: {} {}",
                column.value(),
                self.variable_statuses[column],
                stringify_default(self.primal_values[column])
            )
            .unwrap();
        }
        output.push_str("\n------------------------------");
        for position in 0..self.dual_values.len().to_usize() {
            let row = RowIndex::from_usize(position);
            write!(
                output,
                "\n  Constraint #{}: {} {}",
                row.value(),
                self.constraint_statuses[row],
                stringify_default(self.dual_values[row])
            )
            .unwrap();
        }
        output
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn model_defaults_validation_and_solution_statuses_match_upstream() {
        let mut model = LinearProgram::new();
        let column = model.find_or_create_variable("x");
        let row = model.find_or_create_constraint("r");
        model.set_coefficient(row, column, 2.0);
        model.clean_up();
        assert!(model.validate().is_ok());
        assert_eq!(model.variable_lower_bounds()[column], 0.0);
        assert_eq!(model.variable_upper_bounds()[column], INFINITY);

        let solution = ProblemSolution::new(RowIndex::new(1), ColIndex::new(1));
        assert_eq!(solution.variable_statuses[column], VariableStatus::Free);
        assert_eq!(solution.constraint_statuses[row], ConstraintStatus::Free);
    }

    #[test]
    fn lazy_model_caches_follow_upstream_invalidation_and_adoption() {
        let mut model = LinearProgram::default();
        assert_eq!(model.objective_scaling_factor(), 1.0);
        let x = model.create_new_variable();
        let y = model.create_new_variable();
        let r0 = model.create_new_constraint();
        let r1 = model.create_new_constraint();
        model.set_coefficient(r0, x, 2.0);
        model.set_coefficient(r1, y, 3.0);
        model.clean_up();

        assert_eq!(
            model
                .transpose_sparse_matrix()
                .look_up_value(RowIndex::from_usize(x.to_usize()), ColIndex::new(0)),
            2.0
        );
        model.mutable_sparse_column(x).set_coefficient(r1, 4.0);
        model.clean_up();
        assert_eq!(
            model
                .transpose_sparse_matrix()
                .look_up_value(RowIndex::from_usize(x.to_usize()), ColIndex::new(1)),
            4.0
        );

        model.set_variable_type(x, ModelVariableType::Integer);
        model.set_variable_bounds(x, 0.0, 1.0);
        assert_eq!(&*model.integer_variables(), &[x]);
        assert_eq!(&*model.binary_variables(), &[x]);
        assert!(model.non_binary_integer_variables().is_empty());
        model.set_variable_bounds(x, 0.0, 2.0);
        assert!(model.binary_variables().is_empty());
        assert_eq!(&*model.non_binary_integer_variables(), &[x]);

        model
            .mutable_transpose_sparse_matrix()
            .mutable_column(ColIndex::new(0))
            .set_coefficient(RowIndex::from_usize(y.to_usize()), 5.0);
        model.use_transpose_matrix_as_reference();
        assert_eq!(model.sparse_column(y).look_up_coefficient(r0), 5.0);
    }

    #[test]
    fn invalid_bounds_are_rejected() {
        let mut model = LinearProgram::new();
        let column = model.create_new_variable();
        model.set_variable_bounds(column, 2.0, 1.0);
        assert!(model.validate().is_err());
    }
}
