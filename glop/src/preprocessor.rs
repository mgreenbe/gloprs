// Copyright 2010-2025 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Incremental port of pinned `ortools/glop/preprocessor.{h,cc}`.

#![allow(clippy::float_cmp)] // Pinned GLOP uses exact bound/status comparisons.

use lp_data::lp_data::{CostScalingAlgorithm as ModelCostScaling, LinearProgram, ProblemSolution};
use lp_data::lp_data_utils::LpScalingParameters;
use lp_data::lp_types::{
    ColIndex, ConstraintStatus, DenseBooleanColumn, DenseBooleanRow, DenseRow, INVALID_COL,
    INVALID_ROW, ProblemStatus, RowIndex, VariableStatus, VectorIndex,
};
use lp_data::lp_utils::{
    SumWithNegativeInfiniteAndOneMissing, SumWithPositiveInfiniteAndOneMissing,
};
use lp_data::matrix_scaler::SparseMatrixScaler;
use lp_data::matrix_utils::find_proportional_columns;
use lp_data::sparse_vector::SparseColumn;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::parameters::{CostScalingAlgorithm, GlopParameters, ScalingAlgorithm, SolverBehavior};
use crate::revised_simplex::RevisedSimplex;
use crate::time_limit::TimeLimit;

#[derive(Debug, Default)]
struct ColumnsSaver {
    saved_columns_index: HashMap<ColIndex, usize>,
    saved_columns: Vec<SparseColumn>,
}

impl ColumnsSaver {
    fn save_column(&mut self, col: ColIndex, column: &SparseColumn) {
        let index = self.saved_columns.len();
        assert!(self.saved_columns_index.insert(col, index).is_none());
        self.saved_columns.push(column.clone());
    }

    fn saved_column(&self, col: ColIndex) -> &SparseColumn {
        &self.saved_columns[self.saved_columns_index[&col]]
    }

    fn saved_or_empty_column(&self, col: ColIndex) -> Option<&SparseColumn> {
        self.saved_columns_index
            .get(&col)
            .map(|&index| &self.saved_columns[index])
    }

    fn save_column_if_not_already_done(&mut self, col: ColIndex, column: &SparseColumn) {
        if let std::collections::hash_map::Entry::Vacant(entry) =
            self.saved_columns_index.entry(col)
        {
            entry.insert(self.saved_columns.len());
            self.saved_columns.push(column.clone());
        }
    }
}

fn is_smaller_within_tolerance(x: f64, y: f64, tolerance: f64) -> bool {
    if y.is_infinite() {
        return x <= y;
    }
    x <= tolerance.mul_add(1.0_f64.max(x.abs().min(y.abs())), y)
}

fn min_in_magnitude_or_zero_if_infinite(lower: f64, upper: f64) -> f64 {
    let value = if lower.abs() < upper.abs() {
        lower
    } else {
        upper
    };
    if value.is_finite() { value } else { 0.0 }
}

/// Upstream's prefix-sized deletion map. A deleted variable's exact value and
/// status are kept for postsolve; no matrix columns are copied into this map.
#[derive(Debug)]
struct ColumnDeletionHelper {
    deleted: DenseBooleanRow,
    stored_values: DenseRow,
    stored_statuses: lp_data::lp_types::VariableStatusRow,
}

impl Default for ColumnDeletionHelper {
    fn default() -> Self {
        Self {
            deleted: DenseBooleanRow::new(),
            stored_values: DenseRow::new(),
            stored_statuses: lp_data::lp_types::VariableStatusRow::new(),
        }
    }
}

impl ColumnDeletionHelper {
    fn mark(&mut self, col: ColIndex) {
        self.mark_with_state(col, 0.0, VariableStatus::Free);
    }

    fn is_marked(&self, col: ColIndex) -> bool {
        col.to_usize() < self.deleted.len().to_usize() && self.deleted[col]
    }

    fn mark_with_state(&mut self, col: ColIndex, value: f64, status: VariableStatus) {
        if col.to_usize() >= self.deleted.len().to_usize() {
            let size = ColIndex::from_usize(col.to_usize() + 1);
            self.deleted.resize(size, false);
            self.stored_values.resize(size, 0.0);
            self.stored_statuses.resize(size, VariableStatus::Free);
        }
        self.deleted[col] = true;
        self.stored_values[col] = value;
        self.stored_statuses[col] = status;
    }

    fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }

    fn restore(&self, solution: &mut ProblemSolution) {
        let mut primal = DenseRow::new();
        let mut statuses = lp_data::lp_types::VariableStatusRow::new();
        let mut old = 0;
        for position in 0..self.deleted.len().to_usize() {
            let col = ColIndex::from_usize(position);
            if self.deleted[col] {
                primal.push(self.stored_values[col]);
                statuses.push(self.stored_statuses[col]);
            } else {
                let source = ColIndex::from_usize(old);
                primal.push(solution.primal_values[source]);
                statuses.push(solution.variable_statuses[source]);
                old += 1;
            }
        }
        while old < solution.primal_values.len().to_usize() {
            let source = ColIndex::from_usize(old);
            primal.push(solution.primal_values[source]);
            statuses.push(solution.variable_statuses[source]);
            old += 1;
        }
        solution.primal_values = primal;
        solution.variable_statuses = statuses;
    }
}

#[derive(Debug, Default)]
struct RowDeletionHelper {
    deleted: DenseBooleanColumn,
}

impl RowDeletionHelper {
    fn unmark(&mut self, row: RowIndex) {
        if row.to_usize() < self.deleted.len().to_usize() {
            self.deleted[row] = false;
        }
    }

    fn is_marked(&self, row: RowIndex) -> bool {
        row.to_usize() < self.deleted.len().to_usize() && self.deleted[row]
    }

    fn mark(&mut self, row: RowIndex) {
        if row.to_usize() >= self.deleted.len().to_usize() {
            self.deleted
                .resize(RowIndex::from_usize(row.to_usize() + 1), false);
        }
        self.deleted[row] = true;
    }

    fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }

    fn restore(&self, solution: &mut ProblemSolution) {
        let mut dual = lp_data::lp_types::DenseColumn::new();
        let mut statuses = lp_data::lp_types::ConstraintStatusColumn::new();
        let mut old = 0;
        for position in 0..self.deleted.len().to_usize() {
            let row = RowIndex::from_usize(position);
            if self.deleted[row] {
                dual.push(0.0);
                statuses.push(ConstraintStatus::Basic);
            } else {
                let source = RowIndex::from_usize(old);
                dual.push(solution.dual_values[source]);
                statuses.push(solution.constraint_statuses[source]);
                old += 1;
            }
        }
        while old < solution.dual_values.len().to_usize() {
            let source = RowIndex::from_usize(old);
            dual.push(solution.dual_values[source]);
            statuses.push(solution.constraint_statuses[source]);
            old += 1;
        }
        solution.dual_values = dual;
        solution.constraint_statuses = statuses;
    }
}

fn subtract_column_multiple_from_constraint_bounds(
    lp: &mut LinearProgram,
    col: ColIndex,
    multiple: f64,
) {
    lp.subtract_column_multiple_from_constraint_bounds(col, multiple);
}

#[derive(Clone, Copy, Debug)]
struct SingletonEntry {
    row: RowIndex,
    col: ColIndex,
    coeff: f64,
}

#[derive(Clone, Copy, Debug)]
enum SingletonOperation {
    Row,
    ZeroCostColumn,
    ColumnInEquality,
    MakeEquality,
}

#[derive(Clone, Copy, Debug)]
struct SingletonUndo {
    operation: SingletonOperation,
    entry: SingletonEntry,
    maximization: bool,
    cost: f64,
    variable_lower: f64,
    variable_upper: f64,
    constraint_lower: f64,
    constraint_upper: f64,
    constraint_status: ConstraintStatus,
}

impl SingletonUndo {
    fn new(
        operation: SingletonOperation,
        lp: &LinearProgram,
        entry: SingletonEntry,
        constraint_status: ConstraintStatus,
    ) -> Self {
        Self {
            operation,
            entry,
            maximization: lp.is_maximization_problem(),
            cost: lp.objective_coefficients()[entry.col],
            variable_lower: lp.variable_lower_bounds()[entry.col],
            variable_upper: lp.variable_upper_bounds()[entry.col],
            constraint_lower: lp.constraint_lower_bounds()[entry.row],
            constraint_upper: lp.constraint_upper_bounds()[entry.row],
            constraint_status,
        }
    }

    fn undo(
        &self,
        parameters: &GlopParameters,
        saved_column: Option<&SparseColumn>,
        saved_row: Option<&SparseColumn>,
        solution: &mut ProblemSolution,
    ) {
        match self.operation {
            SingletonOperation::Row => {
                self.undo_row(saved_column.expect("singleton row column saved"), solution);
            }
            SingletonOperation::ZeroCostColumn => self.undo_zero_cost_column(
                parameters,
                saved_row.expect("singleton column row saved"),
                solution,
            ),
            SingletonOperation::ColumnInEquality => {
                self.undo_zero_cost_column(
                    parameters,
                    saved_row.expect("singleton equality row saved"),
                    solution,
                );
                solution.dual_values[self.entry.row] += self.cost / self.entry.coeff;
                if solution.constraint_statuses[self.entry.row] == ConstraintStatus::Basic {
                    solution.variable_statuses[self.entry.col] = VariableStatus::Basic;
                    solution.constraint_statuses[self.entry.row] = ConstraintStatus::FixedValue;
                }
            }
            SingletonOperation::MakeEquality => {
                if solution.constraint_statuses[self.entry.row] == ConstraintStatus::FixedValue {
                    solution.constraint_statuses[self.entry.row] = self.constraint_status;
                }
            }
        }
    }

    #[allow(clippy::nonminimal_bool)] // Mirrors three independent native early exits.
    fn undo_row(&self, saved_column: &SparseColumn, solution: &mut ProblemSolution) {
        let e = self.entry;
        let status = solution.variable_statuses[e.col];
        if matches!(status, VariableStatus::Basic | VariableStatus::Free) {
            return;
        }
        let mut implied_lower = self.constraint_lower / e.coeff;
        let mut implied_upper = self.constraint_upper / e.coeff;
        if e.coeff < 0.0 {
            std::mem::swap(&mut implied_lower, &mut implied_upper);
        }
        let lower_changed = implied_lower > self.variable_lower;
        let upper_changed = implied_upper < self.variable_upper;
        if (!lower_changed && !upper_changed)
            || (status == VariableStatus::AtLowerBound && !lower_changed)
            || (status == VariableStatus::AtUpperBound && !upper_changed)
        {
            return;
        }
        let reduced_cost = self.cost
            - lp_data::lp_utils::sparse_scalar_product(
                solution.dual_values.as_slice(),
                saved_column,
            );
        let min_cost = if self.maximization {
            -reduced_cost
        } else {
            reduced_cost
        };
        if status == VariableStatus::FixedValue {
            if min_cost >= 0.0 && !lower_changed {
                solution.variable_statuses[e.col] = VariableStatus::AtLowerBound;
                return;
            }
            if min_cost <= 0.0 && !upper_changed {
                solution.variable_statuses[e.col] = VariableStatus::AtUpperBound;
                return;
            }
        }
        solution.dual_values[e.row] = reduced_cost / e.coeff;
        let mut constraint_status = match status {
            VariableStatus::AtLowerBound => ConstraintStatus::AtLowerBound,
            VariableStatus::AtUpperBound => ConstraintStatus::AtUpperBound,
            VariableStatus::FixedValue => ConstraintStatus::FixedValue,
            VariableStatus::Free => ConstraintStatus::Free,
            VariableStatus::Basic => ConstraintStatus::Basic,
        };
        if status == VariableStatus::FixedValue && (!lower_changed || !upper_changed) {
            constraint_status = if lower_changed {
                ConstraintStatus::AtLowerBound
            } else {
                ConstraintStatus::AtUpperBound
            };
        }
        if e.coeff < 0.0 {
            constraint_status = match constraint_status {
                ConstraintStatus::AtLowerBound => ConstraintStatus::AtUpperBound,
                ConstraintStatus::AtUpperBound => ConstraintStatus::AtLowerBound,
                other => other,
            };
        }
        solution.variable_statuses[e.col] = VariableStatus::Basic;
        solution.constraint_statuses[e.row] = constraint_status;
    }

    #[allow(clippy::too_many_lines, clippy::nonminimal_bool)]
    fn undo_zero_cost_column(
        &self,
        parameters: &GlopParameters,
        saved_row: &SparseColumn,
        solution: &mut ProblemSolution,
    ) {
        let e = self.entry;
        if self.variable_upper == self.variable_lower {
            solution.primal_values[e.col] = self.variable_lower;
            solution.variable_statuses[e.col] = VariableStatus::FixedValue;
            return;
        }
        let constraint_status = solution.constraint_statuses[e.row];
        if constraint_status == ConstraintStatus::FixedValue {
            let corrected_dual = if self.maximization {
                -solution.dual_values[e.row]
            } else {
                solution.dual_values[e.row]
            };
            if corrected_dual > 0.0 {
                solution.primal_values[e.col] = self.variable_lower;
                solution.variable_statuses[e.col] = VariableStatus::AtLowerBound;
            } else {
                solution.primal_values[e.col] = self.variable_upper;
                solution.variable_statuses[e.col] = VariableStatus::AtUpperBound;
            }
            return;
        }
        if matches!(
            constraint_status,
            ConstraintStatus::AtLowerBound | ConstraintStatus::AtUpperBound
        ) {
            if (constraint_status == ConstraintStatus::AtUpperBound && e.coeff > 0.0)
                || (constraint_status == ConstraintStatus::AtLowerBound && e.coeff < 0.0)
            {
                solution.primal_values[e.col] = self.variable_lower;
                solution.variable_statuses[e.col] = VariableStatus::AtLowerBound;
            } else {
                solution.primal_values[e.col] = self.variable_upper;
                solution.variable_statuses[e.col] = VariableStatus::AtUpperBound;
            }
            if self.constraint_upper == self.constraint_lower {
                solution.constraint_statuses[e.row] = ConstraintStatus::FixedValue;
            }
            return;
        }
        let activity =
            lp_data::lp_utils::sparse_scalar_product(solution.primal_values.as_slice(), saved_row);
        if self.variable_lower != f64::NEG_INFINITY {
            let activity_at_bound = e.coeff.mul_add(self.variable_lower, activity);
            if is_smaller_within_tolerance(
                self.constraint_lower,
                activity_at_bound,
                parameters.preprocessor_zero_tolerance,
            ) && is_smaller_within_tolerance(
                activity_at_bound,
                self.constraint_upper,
                parameters.preprocessor_zero_tolerance,
            ) {
                solution.primal_values[e.col] = self.variable_lower;
                solution.variable_statuses[e.col] = VariableStatus::AtLowerBound;
                return;
            }
        }
        if self.variable_upper != f64::INFINITY {
            let activity_at_bound = e.coeff.mul_add(self.variable_upper, activity);
            if is_smaller_within_tolerance(
                self.constraint_lower,
                activity_at_bound,
                parameters.preprocessor_zero_tolerance,
            ) && is_smaller_within_tolerance(
                activity_at_bound,
                self.constraint_upper,
                parameters.preprocessor_zero_tolerance,
            ) {
                solution.primal_values[e.col] = self.variable_upper;
                solution.variable_statuses[e.col] = VariableStatus::AtUpperBound;
                return;
            }
        }
        if self.constraint_lower == f64::NEG_INFINITY && self.constraint_upper == f64::INFINITY {
            solution.primal_values[e.col] = 0.0;
            solution.variable_statuses[e.col] = VariableStatus::Free;
            return;
        }
        solution.variable_statuses[e.col] = VariableStatus::Basic;
        if self.constraint_lower == self.constraint_upper {
            solution.primal_values[e.col] = (self.constraint_lower - activity) / e.coeff;
            solution.constraint_statuses[e.row] = ConstraintStatus::FixedValue;
            return;
        }
        let choose_lower = if self.constraint_lower == f64::NEG_INFINITY {
            false
        } else if self.constraint_upper == f64::INFINITY {
            true
        } else {
            let to_lower = (self.constraint_lower - activity) / e.coeff;
            let to_upper = (self.constraint_upper - activity) / e.coeff;
            (self.variable_lower - to_lower).max(to_lower - self.variable_upper)
                < (self.variable_lower - to_upper).max(to_upper - self.variable_upper)
        };
        if choose_lower {
            solution.primal_values[e.col] = (self.constraint_lower - activity) / e.coeff;
            solution.constraint_statuses[e.row] = ConstraintStatus::AtLowerBound;
        } else {
            solution.primal_values[e.col] = (self.constraint_upper - activity) / e.coeff;
            solution.constraint_statuses[e.row] = ConstraintStatus::AtUpperBound;
        }
    }
}

/// Pinned GLOP singleton presolve. The matrix and its transpose remain the
/// immutable reference graph while degrees account for deferred deletions.
#[derive(Debug)]
pub struct SingletonPreprocessor {
    status: ProblemStatus,
    columns: ColumnDeletionHelper,
    rows: RowDeletionHelper,
    undo: Vec<SingletonUndo>,
    row_sum_cached: Vec<bool>,
    row_lower_sum: Vec<SumWithNegativeInfiniteAndOneMissing>,
    row_upper_sum: Vec<SumWithPositiveInfiniteAndOneMissing>,
    saved_columns: ColumnsSaver,
    saved_rows: ColumnsSaver,
    parameters: GlopParameters,
}

impl Default for SingletonPreprocessor {
    fn default() -> Self {
        Self {
            status: ProblemStatus::Init,
            columns: ColumnDeletionHelper::default(),
            rows: RowDeletionHelper::default(),
            undo: Vec::new(),
            row_sum_cached: Vec::new(),
            row_lower_sum: Vec::new(),
            row_upper_sum: Vec::new(),
            saved_columns: ColumnsSaver::default(),
            saved_rows: ColumnsSaver::default(),
            parameters: GlopParameters::default(),
        }
    }
}

impl SingletonPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    fn update_constraint_bounds(e: SingletonEntry, lp: &mut LinearProgram) {
        let mut lower_delta = -e.coeff * lp.variable_upper_bounds()[e.col];
        let mut upper_delta = -e.coeff * lp.variable_lower_bounds()[e.col];
        if e.coeff < 0.0 {
            std::mem::swap(&mut lower_delta, &mut upper_delta);
        }
        lp.set_constraint_bounds(
            e.row,
            lp.constraint_lower_bounds()[e.row] + lower_delta,
            lp.constraint_upper_bounds()[e.row] + upper_delta,
        );
    }

    fn delete_row(&mut self, e: SingletonEntry, lp: &mut LinearProgram) {
        let mut implied_lower = lp.constraint_lower_bounds()[e.row] / e.coeff;
        let mut implied_upper = lp.constraint_upper_bounds()[e.row] / e.coeff;
        if e.coeff < 0.0 {
            std::mem::swap(&mut implied_lower, &mut implied_upper);
        }
        let old_lower = lp.variable_lower_bounds()[e.col];
        let old_upper = lp.variable_upper_bounds()[e.col];
        let potential_error = (self.parameters.preprocessor_zero_tolerance / e.coeff).abs();
        let new_lower = if implied_lower - potential_error > old_lower {
            implied_lower
        } else {
            old_lower
        };
        let mut new_upper = if implied_upper + potential_error < old_upper {
            implied_upper
        } else {
            old_upper
        };
        if new_upper == f64::NEG_INFINITY || new_lower == f64::INFINITY {
            self.status = ProblemStatus::PrimalInfeasible;
            return;
        }
        if new_upper < new_lower {
            if !is_smaller_within_tolerance(
                new_lower,
                new_upper,
                self.parameters.solution_feasibility_tolerance,
            ) {
                self.status = ProblemStatus::PrimalInfeasible;
                return;
            }
            new_upper = new_lower;
        }
        self.rows.mark(e.row);
        self.undo.push(SingletonUndo::new(
            SingletonOperation::Row,
            lp,
            e,
            ConstraintStatus::Free,
        ));
        self.saved_columns
            .save_column_if_not_already_done(e.col, lp.sparse_column(e.col));
        lp.set_variable_bounds(e.col, new_lower, new_upper);
    }

    fn delete_zero_cost_column(&mut self, e: SingletonEntry, lp: &mut LinearProgram) {
        self.undo.push(SingletonUndo::new(
            SingletonOperation::ZeroCostColumn,
            lp,
            e,
            ConstraintStatus::Free,
        ));
        {
            let transpose = lp.transpose_sparse_matrix();
            self.saved_rows.save_column_if_not_already_done(
                ColIndex::from_usize(e.row.to_usize()),
                transpose.column(ColIndex::from_usize(e.row.to_usize())),
            );
        }
        Self::update_constraint_bounds(e, lp);
        self.columns.mark(e.col);
    }

    #[allow(clippy::too_many_lines)] // Direct correspondence with the native presolve rule.
    fn make_equality_if_possible(&mut self, e: SingletonEntry, lp: &mut LinearProgram) -> bool {
        let lower = lp.constraint_lower_bounds()[e.row];
        let upper = lp.constraint_upper_bounds()[e.row];
        if lower == upper {
            return true;
        }
        if lower == f64::NEG_INFINITY && upper == f64::INFINITY {
            return false;
        }
        let row = e.row.to_usize();
        if self.row_sum_cached.len() <= row {
            self.row_sum_cached.resize(row + 1, false);
            self.row_lower_sum
                .resize(row + 1, SumWithNegativeInfiniteAndOneMissing::default());
            self.row_upper_sum
                .resize(row + 1, SumWithPositiveInfiniteAndOneMissing::default());
        }
        if !self.row_sum_cached[row] {
            self.row_sum_cached[row] = true;
            self.row_lower_sum[row].add(lower);
            self.row_upper_sum[row].add(upper);
            let transpose = lp.transpose_sparse_matrix();
            for entry in transpose.column(ColIndex::from_usize(row)) {
                let col = ColIndex::from_usize(entry.index().to_usize());
                if self.columns.is_marked(col) {
                    continue;
                }
                let coeff = entry.coefficient();
                if coeff > 0.0 {
                    self.row_lower_sum[row].add(-coeff * lp.variable_upper_bounds()[col]);
                    self.row_upper_sum[row].add(-coeff * lp.variable_lower_bounds()[col]);
                } else {
                    self.row_lower_sum[row].add(-coeff * lp.variable_lower_bounds()[col]);
                    self.row_upper_sum[row].add(-coeff * lp.variable_upper_bounds()[col]);
                }
            }
        }
        let variable_lower = lp.variable_lower_bounds()[e.col];
        let variable_upper = lp.variable_upper_bounds()[e.col];
        let implied_lower = if e.coeff > 0.0 {
            self.row_lower_sum[row].sum_without_lb(-e.coeff * variable_upper) / e.coeff
        } else {
            self.row_upper_sum[row].sum_without_ub(-e.coeff * variable_upper) / e.coeff
        };
        let implied_upper = if e.coeff > 0.0 {
            self.row_upper_sum[row].sum_without_ub(-e.coeff * variable_lower) / e.coeff
        } else {
            self.row_lower_sum[row].sum_without_lb(-e.coeff * variable_lower) / e.coeff
        };
        let cost = lp.objective_coefficient_for_minimization(e.col);
        let mut relaxed = ConstraintStatus::FixedValue;
        if cost < 0.0
            && is_smaller_within_tolerance(
                implied_upper,
                variable_upper,
                self.parameters.preprocessor_zero_tolerance,
            )
        {
            if e.coeff > 0.0 {
                if upper == f64::INFINITY {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                relaxed = ConstraintStatus::AtUpperBound;
                lp.set_constraint_bounds(e.row, upper, upper);
            } else {
                if lower == f64::NEG_INFINITY {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                relaxed = ConstraintStatus::AtLowerBound;
                lp.set_constraint_bounds(e.row, lower, lower);
            }
            lp.set_variable_bounds(e.col, variable_lower, f64::INFINITY);
        }
        if cost > 0.0
            && is_smaller_within_tolerance(
                variable_lower,
                implied_lower,
                self.parameters.preprocessor_zero_tolerance,
            )
        {
            if e.coeff > 0.0 {
                if lower == f64::NEG_INFINITY {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                relaxed = ConstraintStatus::AtLowerBound;
                lp.set_constraint_bounds(e.row, lower, lower);
            } else {
                if upper == f64::INFINITY {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                relaxed = ConstraintStatus::AtUpperBound;
                lp.set_constraint_bounds(e.row, upper, upper);
            }
            lp.set_variable_bounds(e.col, f64::NEG_INFINITY, lp.variable_upper_bounds()[e.col]);
        }
        if lp.constraint_lower_bounds()[e.row] == lp.constraint_upper_bounds()[e.row] {
            self.undo.push(SingletonUndo::new(
                SingletonOperation::MakeEquality,
                lp,
                e,
                relaxed,
            ));
            return true;
        }
        false
    }

    fn delete_column_in_equality(&mut self, e: SingletonEntry, lp: &mut LinearProgram) {
        self.undo.push(SingletonUndo::new(
            SingletonOperation::ColumnInEquality,
            lp,
            e,
            ConstraintStatus::Free,
        ));
        let row = ColIndex::from_usize(e.row.to_usize());
        {
            let transpose = lp.transpose_sparse_matrix();
            self.saved_rows
                .save_column_if_not_already_done(row, transpose.column(row));
        }
        let multiplier = lp.objective_coefficients()[e.col] / e.coeff;
        let rhs = lp.constraint_upper_bounds()[e.row];
        lp.set_objective_offset(rhs.mul_add(multiplier, lp.objective_offset()));
        // The saved native-order row is already needed for postsolve. Reuse it
        // directly, without allocating another copy just to update costs.
        for entry in self.saved_rows.saved_column(row) {
            let col = ColIndex::from_usize(entry.index().to_usize());
            if self.columns.is_marked(col) {
                continue;
            }
            let mut new_cost =
                (-entry.coefficient()).mul_add(multiplier, lp.objective_coefficients()[col]);
            if new_cost.abs() < self.parameters.preprocessor_zero_tolerance {
                new_cost = 0.0;
            }
            lp.set_objective_coefficient(col, new_cost);
        }
        Self::update_constraint_bounds(e, lp);
        self.columns.mark(e.col);
    }

    #[allow(clippy::too_many_lines)] // Preserve native queue traversal in one visible block.
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        self.parameters = parameters.clone();
        let mut column_degree: Vec<usize> = (0..lp.num_variables().to_usize())
            .map(|c| lp.sparse_column(ColIndex::from_usize(c)).num_entries())
            .collect();
        let mut column_queue: Vec<ColIndex> = column_degree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 1)
            .map(|(c, _)| ColIndex::from_usize(c))
            .collect();
        let mut row_degree: Vec<usize> = {
            let transpose = lp.transpose_sparse_matrix();
            (0..lp.num_constraints().to_usize())
                .map(|r| transpose.column(ColIndex::from_usize(r)).num_entries())
                .collect()
        };
        let mut row_queue: Vec<RowIndex> = row_degree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 1)
            .map(|(r, _)| RowIndex::from_usize(r))
            .collect();
        while self.status == ProblemStatus::Init
            && (!column_queue.is_empty() || !row_queue.is_empty())
        {
            while self.status == ProblemStatus::Init {
                let Some(col) = column_queue.pop() else {
                    break;
                };
                if column_degree[col.to_usize()] == 0 {
                    continue;
                }
                let Some(entry) = lp
                    .sparse_column(col)
                    .iter()
                    .find(|entry| !self.rows.is_marked(entry.index()))
                else {
                    self.status = ProblemStatus::Abnormal;
                    break;
                };
                let e = SingletonEntry {
                    row: entry.index(),
                    col,
                    coeff: entry.coefficient(),
                };
                if lp.objective_coefficients()[col] == 0.0 {
                    self.delete_zero_cost_column(e, lp);
                } else {
                    if e.coeff.abs() < parameters.preprocessor_zero_tolerance {
                        continue;
                    }
                    if !self.make_equality_if_possible(e, lp) {
                        continue;
                    }
                    self.delete_column_in_equality(e, lp);
                }
                row_degree[e.row.to_usize()] -= 1;
                if row_degree[e.row.to_usize()] == 1 {
                    row_queue.push(e.row);
                }
            }
            while self.status == ProblemStatus::Init {
                let Some(row) = row_queue.pop() else {
                    break;
                };
                if row_degree[row.to_usize()] == 0 {
                    continue;
                }
                let e = {
                    let transpose = lp.transpose_sparse_matrix();
                    transpose
                        .column(ColIndex::from_usize(row.to_usize()))
                        .iter()
                        .find(|entry| {
                            !self
                                .columns
                                .is_marked(ColIndex::from_usize(entry.index().to_usize()))
                        })
                        .map(|entry| SingletonEntry {
                            row,
                            col: ColIndex::from_usize(entry.index().to_usize()),
                            coeff: entry.coefficient(),
                        })
                };
                let Some(e) = e else {
                    self.status = ProblemStatus::Abnormal;
                    break;
                };
                self.delete_row(e, lp);
                column_degree[e.col.to_usize()] -= 1;
                if column_degree[e.col.to_usize()] == 1 {
                    column_queue.push(e.col);
                }
            }
        }
        if self.status != ProblemStatus::Init {
            return false;
        }
        lp.delete_columns(&self.columns.deleted);
        lp.delete_rows(&self.rows.deleted);
        !self.columns.is_empty() || !self.rows.is_empty()
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.columns.restore(solution);
        self.rows.restore(solution);
        for undo in self.undo.iter().rev() {
            undo.undo(
                &self.parameters,
                self.saved_columns.saved_or_empty_column(undo.entry.col),
                self.saved_rows
                    .saved_or_empty_column(ColIndex::from_usize(undo.entry.row.to_usize())),
                solution,
            );
        }
    }
}

#[derive(Debug)]
enum AppliedPreprocessor {
    Shift(Box<ShiftVariableBoundsPreprocessor>),
    Fixed(Box<FixedVariablePreprocessor>),
    Singleton(Box<SingletonPreprocessor>),
    Forcing(Box<ForcingAndImpliedFreeConstraintPreprocessor>),
    FreeConstraint(Box<FreeConstraintPreprocessor>),
    ImpliedFree(Box<ImpliedFreePreprocessor>),
    Unconstrained(Box<UnconstrainedVariablePreprocessor>),
    DoubletonFree(Box<DoubletonFreeColumnPreprocessor>),
    DoubletonEquality(Box<DoubletonEqualityRowPreprocessor>),
    EmptyColumn(Box<EmptyColumnPreprocessor>),
    EmptyConstraint(Box<EmptyConstraintPreprocessor>),
    ProportionalColumn(Box<ProportionalColumnPreprocessor>),
    ProportionalRow(Box<ProportionalRowPreprocessor>),
    Dualizer(Box<DualizerPreprocessor>),
    SingletonSign(Box<SingletonColumnSignPreprocessor>),
    Scaling(Box<ScalingPreprocessor>),
}

impl AppliedPreprocessor {
    fn run(
        &mut self,
        lp: &mut LinearProgram,
        parameters: &GlopParameters,
    ) -> (bool, ProblemStatus) {
        match self {
            Self::Shift(p) => match p.run(lp) {
                Ok(changed) => (changed, ProblemStatus::Init),
                Err(status) => (false, status),
            },
            Self::Fixed(p) => (p.run(lp), ProblemStatus::Init),
            Self::Singleton(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::Forcing(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::FreeConstraint(p) => (p.run(lp), ProblemStatus::Init),
            Self::ImpliedFree(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::Unconstrained(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::DoubletonFree(p) => (p.run(lp, parameters), ProblemStatus::Init),
            Self::DoubletonEquality(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::EmptyColumn(p) => {
                let changed = p.run(lp);
                (changed, p.status())
            }
            Self::EmptyConstraint(p) => {
                let changed = p.run(lp, parameters.solution_feasibility_tolerance);
                (changed, p.status())
            }
            Self::ProportionalColumn(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::ProportionalRow(p) => {
                let changed = p.run(lp, parameters);
                (changed, p.status())
            }
            Self::Dualizer(p) => (p.run(lp, parameters), ProblemStatus::Init),
            Self::SingletonSign(p) => (p.run(lp), ProblemStatus::Init),
            Self::Scaling(p) => (p.run(lp, parameters), ProblemStatus::Init),
        }
    }

    fn recover_solution(&self, solution: &mut ProblemSolution) {
        match self {
            Self::Shift(p) => p.recover_solution(solution),
            Self::Fixed(p) => p.recover_solution(solution),
            Self::Singleton(p) => p.recover_solution(solution),
            Self::Forcing(p) => p.recover_solution(solution),
            Self::FreeConstraint(p) => p.recover_solution(solution),
            Self::ImpliedFree(p) => p.recover_solution(solution),
            Self::Unconstrained(p) => p.recover_solution(solution),
            Self::DoubletonFree(p) => p.recover_solution(solution),
            Self::DoubletonEquality(p) => p.recover_solution(solution),
            Self::EmptyColumn(p) => p.recover_solution(solution),
            Self::EmptyConstraint(p) => p.recover_solution(solution),
            Self::ProportionalColumn(p) => p.recover_solution(solution),
            Self::ProportionalRow(p) => p.recover_solution(solution),
            Self::Dualizer(p) => p.recover_solution(solution),
            Self::SingletonSign(p) => p.recover_solution(solution),
            Self::Scaling(p) => p.recover_solution(solution),
        }
    }
}

/// Runs the pinned `MainLpPreprocessor::Run()` sequence, retaining only passes
/// whose transformations require reverse-order postsolve.
#[derive(Debug)]
pub struct MainLpPreprocessor {
    preprocessors: Vec<AppliedPreprocessor>,
    status: ProblemStatus,
}

impl Default for MainLpPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MainLpPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            preprocessors: Vec::new(),
            status: ProblemStatus::Init,
        }
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    fn run_and_push(
        &mut self,
        mut preprocessor: AppliedPreprocessor,
        lp: &mut LinearProgram,
        parameters: &GlopParameters,
        time_limit: &mut TimeLimit,
    ) {
        if self.status != ProblemStatus::Init || time_limit.limit_reached() {
            return;
        }
        if lp.num_variables().to_usize() == 0 && lp.num_constraints().to_usize() == 0 {
            self.status = ProblemStatus::Optimal;
            return;
        }
        let (changed, status) = preprocessor.run(lp, parameters);
        self.status = status;
        if changed {
            self.preprocessors.push(preprocessor);
        }
    }

    #[allow(clippy::too_many_lines)] // Exactly the order in pinned MainLpPreprocessor::Run.
    pub fn run(
        &mut self,
        lp: &mut LinearProgram,
        parameters: &GlopParameters,
        time_limit: &mut TimeLimit,
    ) -> bool {
        macro_rules! run {
            ($variant:ident, $type:ident) => {
                self.run_and_push(
                    AppliedPreprocessor::$variant(Box::new($type::new())),
                    lp,
                    parameters,
                    time_limit,
                );
            };
        }
        if parameters.use_preprocessing {
            run!(Shift, ShiftVariableBoundsPreprocessor);
            for _ in 0..20 {
                let old_size = self.preprocessors.len();
                run!(Fixed, FixedVariablePreprocessor);
                run!(Singleton, SingletonPreprocessor);
                run!(Forcing, ForcingAndImpliedFreeConstraintPreprocessor);
                run!(FreeConstraint, FreeConstraintPreprocessor);
                run!(ImpliedFree, ImpliedFreePreprocessor);
                run!(Unconstrained, UnconstrainedVariablePreprocessor);
                run!(DoubletonFree, DoubletonFreeColumnPreprocessor);
                run!(DoubletonEquality, DoubletonEqualityRowPreprocessor);
                if self.preprocessors.len() == old_size {
                    break;
                }
            }
            run!(EmptyColumn, EmptyColumnPreprocessor);
            run!(EmptyConstraint, EmptyConstraintPreprocessor);
            run!(ProportionalColumn, ProportionalColumnPreprocessor);
            run!(ProportionalRow, ProportionalRowPreprocessor);
            let old_size = self.preprocessors.len();
            run!(Dualizer, DualizerPreprocessor);
            if self.preprocessors.len() != old_size {
                run!(Singleton, SingletonPreprocessor);
                run!(FreeConstraint, FreeConstraintPreprocessor);
                run!(Unconstrained, UnconstrainedVariablePreprocessor);
                run!(EmptyColumn, EmptyColumnPreprocessor);
                run!(EmptyConstraint, EmptyConstraintPreprocessor);
            }
            run!(SingletonSign, SingletonColumnSignPreprocessor);
        }
        run!(Scaling, ScalingPreprocessor);
        !self.preprocessors.is_empty()
    }

    /// Reverses the applied passes while retaining the stack for another
    /// solution, as in GLOP's `RecoverSolution()`.
    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        for preprocessor in self.preprocessors.iter().rev() {
            preprocessor.recover_solution(solution);
        }
    }

    /// Reverses the applied passes and frees each one immediately, matching
    /// GLOP's lower-peak-memory `DestructiveRecoverSolution()`.
    pub fn destructive_recover_solution(&mut self, solution: &mut ProblemSolution) {
        while let Some(preprocessor) = self.preprocessors.pop() {
            preprocessor.recover_solution(solution);
        }
    }
}

/// Removes fixed columns, preserving their values/statuses for postsolve.
#[derive(Debug, Default)]
pub struct FixedVariablePreprocessor {
    deleted: ColumnDeletionHelper,
}

impl FixedVariablePreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self, lp: &mut LinearProgram) -> bool {
        self.deleted = ColumnDeletionHelper::default();
        for position in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(position);
            let lower = lp.variable_lower_bounds()[col];
            if lower == lp.variable_upper_bounds()[col] {
                subtract_column_multiple_from_constraint_bounds(lp, col, lower);
                self.deleted
                    .mark_with_state(col, lower, VariableStatus::FixedValue);
            }
        }
        lp.delete_columns(&self.deleted.deleted);
        !self.deleted.is_empty()
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.deleted.restore(solution);
    }
}

/// GLOP's forcing-row and implied-free-row transformation. Saved sparse
/// columns are used only for postsolve, as in the upstream implementation.
#[derive(Debug)]
pub struct ForcingAndImpliedFreeConstraintPreprocessor {
    columns_saver: ColumnsSaver,
    column_deletion_helper: ColumnDeletionHelper,
    row_deletion_helper: RowDeletionHelper,
    is_forcing_up: Vec<bool>,
    costs: DenseRow,
    lp_is_maximization_problem: bool,
    status: ProblemStatus,
}

impl Default for ForcingAndImpliedFreeConstraintPreprocessor {
    fn default() -> Self {
        Self {
            columns_saver: ColumnsSaver::default(),
            column_deletion_helper: ColumnDeletionHelper::default(),
            row_deletion_helper: RowDeletionHelper::default(),
            is_forcing_up: Vec::new(),
            costs: DenseRow::new(),
            lp_is_maximization_problem: false,
            status: ProblemStatus::Init,
        }
    }
}

impl ForcingAndImpliedFreeConstraintPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_lines)] // Preserve the upstream pass's ordered phases.
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        let rows = lp.num_constraints().to_usize();
        let cols = lp.num_variables().to_usize();
        let mut implied_lower_bounds = vec![0.0; rows];
        let mut implied_upper_bounds = vec![0.0; rows];
        let mut row_degree = vec![0; rows];
        for position in 0..cols {
            let col = ColIndex::from_usize(position);
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            for entry in lp.sparse_column(col) {
                let row = entry.index().to_usize();
                let coeff = entry.coefficient();
                if coeff > 0.0 {
                    implied_lower_bounds[row] = lower.mul_add(coeff, implied_lower_bounds[row]);
                    implied_upper_bounds[row] = upper.mul_add(coeff, implied_upper_bounds[row]);
                } else {
                    implied_lower_bounds[row] = upper.mul_add(coeff, implied_lower_bounds[row]);
                    implied_upper_bounds[row] = lower.mul_add(coeff, implied_upper_bounds[row]);
                }
                row_degree[row] += 1;
            }
        }

        self.is_forcing_up.resize(rows, false);
        let mut is_forcing_down = vec![false; rows];
        let mut num_forcing = 0;
        for position in 0..rows {
            if row_degree[position] == 0 {
                continue;
            }
            let row = RowIndex::from_usize(position);
            let lower = lp.constraint_lower_bounds()[row];
            let upper = lp.constraint_upper_bounds()[row];
            if !is_smaller_within_tolerance(
                lower,
                implied_upper_bounds[position],
                parameters.solution_feasibility_tolerance,
            ) || !is_smaller_within_tolerance(
                implied_lower_bounds[position],
                upper,
                parameters.solution_feasibility_tolerance,
            ) {
                self.status = ProblemStatus::PrimalInfeasible;
                return false;
            }
            if is_smaller_within_tolerance(
                implied_upper_bounds[position],
                lower,
                parameters.preprocessor_zero_tolerance,
            ) {
                is_forcing_down[position] = true;
                num_forcing += 1;
                continue;
            }
            if is_smaller_within_tolerance(
                upper,
                implied_lower_bounds[position],
                parameters.preprocessor_zero_tolerance,
            ) {
                self.is_forcing_up[position] = true;
                num_forcing += 1;
                continue;
            }
            if is_smaller_within_tolerance(
                lower,
                implied_lower_bounds[position],
                parameters.preprocessor_zero_tolerance,
            ) && is_smaller_within_tolerance(
                implied_upper_bounds[position],
                upper,
                parameters.preprocessor_zero_tolerance,
            ) {
                lp.set_constraint_bounds(row, f64::NEG_INFINITY, f64::INFINITY);
            }
        }

        if num_forcing > 0 {
            self.lp_is_maximization_problem = lp.is_maximization_problem();
            self.costs.resize(lp.num_variables(), 0.0);
            for position in 0..cols {
                let col = ColIndex::from_usize(position);
                let lower = lp.variable_lower_bounds()[col];
                let upper = lp.variable_upper_bounds()[col];
                let mut is_forced = false;
                let mut target_bound = 0.0;
                for entry in lp.sparse_column(col) {
                    let row = entry.index().to_usize();
                    let candidate = if is_forcing_down[row] {
                        Some(if entry.coefficient() < 0.0 {
                            lower
                        } else {
                            upper
                        })
                    } else if self.is_forcing_up[row] {
                        Some(if entry.coefficient() < 0.0 {
                            upper
                        } else {
                            lower
                        })
                    } else {
                        None
                    };
                    if let Some(candidate) = candidate {
                        if is_forced && candidate != target_bound {
                            if is_smaller_within_tolerance(
                                upper,
                                lower,
                                parameters.preprocessor_zero_tolerance,
                            ) {
                                target_bound = if lower.abs() < upper.abs() {
                                    lower
                                } else {
                                    upper
                                };
                                continue;
                            }
                            self.status = ProblemStatus::PrimalInfeasible;
                            return false;
                        }
                        target_bound = candidate;
                        is_forced = true;
                    }
                }
                if is_forced {
                    subtract_column_multiple_from_constraint_bounds(lp, col, target_bound);
                    self.column_deletion_helper.mark_with_state(
                        col,
                        target_bound,
                        status_at_value(target_bound, lower, upper),
                    );
                    self.columns_saver.save_column(col, lp.sparse_column(col));
                    self.costs[col] = lp.objective_coefficients()[col];
                }
            }
            for (position, &forcing_down) in is_forcing_down.iter().enumerate() {
                if forcing_down || self.is_forcing_up[position] {
                    self.row_deletion_helper
                        .mark(RowIndex::from_usize(position));
                }
            }
        }
        lp.delete_columns(&self.column_deletion_helper.deleted);
        lp.delete_rows(&self.row_deletion_helper.deleted);
        !self.column_deletion_helper.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.column_deletion_helper.restore(solution);
        self.row_deletion_helper.restore(solution);
        let mut entries = Vec::new();
        for position in 0..self.column_deletion_helper.deleted.len().to_usize() {
            let col = ColIndex::from_usize(position);
            if !self.column_deletion_helper.is_marked(col) {
                continue;
            }
            let mut last = None;
            for entry in self.columns_saver.saved_column(col) {
                if self.row_deletion_helper.is_marked(entry.index()) {
                    last = Some((entry.index(), entry.coefficient()));
                }
            }
            if let Some((row, coefficient)) = last {
                entries.push((row, col, coefficient));
            }
        }
        entries.sort_unstable_by_key(|&(row, col, _)| (row, col));
        let mut i = 0;
        while i < entries.len() {
            let row = entries[i].0;
            let mut new_dual = 0.0;
            let mut new_basic = None;
            while i < entries.len() && entries[i].0 == row {
                let (_, col, coefficient) = entries[i];
                let scalar_product = self
                    .columns_saver
                    .saved_column(col)
                    .scalar_product(&solution.dual_values);
                let bound = (self.costs[col] - scalar_product) / coefficient;
                if self.is_forcing_up[row.to_usize()] != self.lp_is_maximization_problem {
                    if bound < new_dual {
                        new_dual = bound;
                        new_basic = Some(col);
                    }
                } else if bound > new_dual {
                    new_dual = bound;
                    new_basic = Some(col);
                }
                i += 1;
            }
            if let Some(col) = new_basic {
                solution.dual_values[row] = new_dual;
                solution.variable_statuses[col] = VariableStatus::Basic;
                solution.constraint_statuses[row] = if self.is_forcing_up[row.to_usize()] {
                    ConstraintStatus::AtUpperBound
                } else {
                    ConstraintStatus::AtLowerBound
                };
            }
        }
    }
}

#[derive(Debug)]
pub struct ImpliedFreePreprocessor {
    postsolve_status_of_free_variables: lp_data::lp_types::VariableStatusRow,
    variable_offsets: DenseRow,
    status: ProblemStatus,
}

impl Default for ImpliedFreePreprocessor {
    fn default() -> Self {
        Self {
            postsolve_status_of_free_variables: lp_data::lp_types::VariableStatusRow::new(),
            variable_offsets: DenseRow::new(),
            status: ProblemStatus::Init,
        }
    }
}

impl ImpliedFreePreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)] // Match GLOP's lower/upper arithmetic and ordered scan.
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        if !parameters.use_implied_free_preprocessor {
            return false;
        }
        let rows = lp.num_constraints().to_usize();
        let cols = lp.num_variables().to_usize();
        let mut lb_sums = vec![SumWithNegativeInfiniteAndOneMissing::default(); rows];
        let mut ub_sums = vec![SumWithPositiveInfiniteAndOneMissing::default(); rows];
        for position in 0..cols {
            let col = ColIndex::from_usize(position);
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            for entry in lp.sparse_column(col) {
                let mut entry_lb = entry.coefficient() * lower;
                let mut entry_ub = entry.coefficient() * upper;
                if entry.coefficient() < 0.0 {
                    std::mem::swap(&mut entry_lb, &mut entry_ub);
                }
                let row = entry.index().to_usize();
                lb_sums[row].add(entry_lb);
                ub_sums[row].add(entry_ub);
            }
        }
        for position in 0..rows {
            let row = RowIndex::from_usize(position);
            lb_sums[position].add(-lp.constraint_upper_bounds()[row]);
            ub_sums[position].add(-lp.constraint_lower_bounds()[row]);
        }

        let mut used_rows = vec![false; rows];
        self.postsolve_status_of_free_variables
            .resize(lp.num_variables(), VariableStatus::Free);
        self.variable_offsets.resize(lp.num_variables(), 0.0);
        let mut col_by_degree = Vec::with_capacity(cols);
        for position in 0..cols {
            let col = ColIndex::from_usize(position);
            col_by_degree.push((lp.sparse_column(col).num_entries(), col));
        }
        col_by_degree.sort_unstable();

        let mut num_implied_free = 0;
        for (_, col) in col_by_degree {
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            if (!lower.is_finite() && !upper.is_finite()) || lower == upper {
                continue;
            }
            let mut overall_lb = f64::NEG_INFINITY;
            let mut overall_ub = f64::INFINITY;
            for entry in lp.sparse_column(col) {
                let row = entry.index().to_usize();
                if used_rows[row] {
                    continue;
                }
                let coeff = entry.coefficient();
                let mut entry_lb = coeff * lower;
                let mut entry_ub = coeff * upper;
                if coeff < 0.0 {
                    std::mem::swap(&mut entry_lb, &mut entry_ub);
                }
                let implied_lb = if coeff > 0.0 {
                    -ub_sums[row].sum_without_ub(entry_ub) / coeff
                } else {
                    -lb_sums[row].sum_without_lb(entry_lb) / coeff
                };
                let implied_ub = if coeff > 0.0 {
                    -lb_sums[row].sum_without_lb(entry_lb) / coeff
                } else {
                    -ub_sums[row].sum_without_ub(entry_ub) / coeff
                };
                overall_lb = overall_lb.max(implied_lb);
                overall_ub = overall_ub.min(implied_ub);
            }
            if !is_smaller_within_tolerance(
                overall_lb,
                upper,
                parameters.solution_feasibility_tolerance,
            ) || !is_smaller_within_tolerance(
                lower,
                overall_ub,
                parameters.solution_feasibility_tolerance,
            ) || !is_smaller_within_tolerance(
                overall_lb,
                overall_ub,
                parameters.solution_feasibility_tolerance,
            ) {
                self.status = ProblemStatus::PrimalInfeasible;
                return false;
            }
            if is_smaller_within_tolerance(
                upper,
                overall_lb,
                parameters.preprocessor_zero_tolerance,
            ) || is_smaller_within_tolerance(
                overall_ub,
                lower,
                parameters.preprocessor_zero_tolerance,
            ) || is_smaller_within_tolerance(
                overall_ub,
                overall_lb,
                parameters.preprocessor_zero_tolerance,
            ) {
                continue;
            }
            if is_smaller_within_tolerance(
                lower,
                overall_lb,
                parameters.preprocessor_zero_tolerance,
            ) && is_smaller_within_tolerance(
                overall_ub,
                upper,
                parameters.preprocessor_zero_tolerance,
            ) {
                num_implied_free += 1;
                lp.set_variable_bounds(col, f64::NEG_INFINITY, f64::INFINITY);
                for entry in lp.sparse_column(col) {
                    used_rows[entry.index().to_usize()] = true;
                }
                let offset = if lower.abs() < upper.abs() {
                    lower
                } else {
                    upper
                };
                let offset = if offset.is_finite() { offset } else { 0.0 };
                if offset != 0.0 {
                    self.variable_offsets[col] = offset;
                    subtract_column_multiple_from_constraint_bounds(lp, col, offset);
                }
                self.postsolve_status_of_free_variables[col] =
                    status_at_value(offset, lower, upper);
            }
        }
        num_implied_free > 0
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        for position in 0..solution.variable_statuses.len().to_usize() {
            let col = ColIndex::from_usize(position);
            let restored = self.postsolve_status_of_free_variables[col];
            if restored == VariableStatus::Free {
                continue;
            }
            if solution.variable_statuses[col] == VariableStatus::Free {
                solution.variable_statuses[col] = restored;
            } else {
                debug_assert_eq!(solution.variable_statuses[col], VariableStatus::Basic);
            }
            solution.primal_values[col] += self.variable_offsets[col];
        }
    }
}

#[derive(Debug)]
struct DoubletonFreeRestoreInfo {
    col: ColIndex,
    rows: [RowIndex; 2],
    coefficients: [f64; 2],
    objective_coefficient: f64,
    deleted_row_as_column: SparseColumn,
}

/// GLOP's elimination of an unbounded column with exactly two row entries.
#[derive(Debug, Default)]
pub struct DoubletonFreeColumnPreprocessor {
    row_deletion_helper: RowDeletionHelper,
    restore_stack: Vec<DoubletonFreeRestoreInfo>,
}

impl DoubletonFreeColumnPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_lines)] // Keep the upstream sparse-transpose update order visible.
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        let cols = lp.num_variables().to_usize();
        let mut transpose = lp.take_mutable_transpose_sparse_matrix();
        for position in 0..cols {
            let col = ColIndex::from_usize(position);
            if lp.sparse_column(col).num_entries() != 2
                || lp.variable_lower_bounds()[col] != f64::NEG_INFINITY
                || lp.variable_upper_bounds()[col] != f64::INFINITY
            {
                continue;
            }
            let mut rows = [RowIndex::from_usize(0); 2];
            let mut coefficients = [0.0; 2];
            let mut count = 0;
            for entry in lp.sparse_column(col) {
                if self.row_deletion_helper.is_marked(entry.index()) {
                    break;
                }
                rows[count] = entry.index();
                coefficients[count] = entry.coefficient();
                count += 1;
            }
            if count != 2 {
                continue;
            }
            debug_assert_eq!(
                coefficients[0],
                transpose
                    .column(ColIndex::from_usize(rows[0].to_usize()))
                    .look_up_coefficient(RowIndex::from_usize(position))
            );
            debug_assert_eq!(
                coefficients[1],
                transpose
                    .column(ColIndex::from_usize(rows[1].to_usize()))
                    .look_up_coefficient(RowIndex::from_usize(position))
            );
            if coefficients[0].abs() < coefficients[1].abs() {
                coefficients.swap(0, 1);
                rows.swap(0, 1);
            }
            let mut deleted_row_as_column = SparseColumn::new();
            deleted_row_as_column
                .swap(transpose.mutable_column(ColIndex::from_usize(rows[0].to_usize())));
            let mut variable_lower = lp.constraint_lower_bounds()[rows[0]] / coefficients[0];
            let mut variable_upper = lp.constraint_upper_bounds()[rows[0]] / coefficients[0];
            if coefficients[0] < 0.0 {
                std::mem::swap(&mut variable_lower, &mut variable_upper);
            }
            lp.set_variable_bounds(col, variable_lower, variable_upper);
            deleted_row_as_column.add_multiple_to_sparse_vector_and_ignore_common_index(
                -coefficients[1] / coefficients[0],
                RowIndex::from_usize(position),
                parameters.drop_tolerance,
                transpose.mutable_column(ColIndex::from_usize(rows[1].to_usize())),
            );
            let objective_coefficient = lp.objective_coefficients()[col];
            if objective_coefficient != 0.0 {
                for entry in &deleted_row_as_column {
                    let changed_col = ColIndex::from_usize(entry.index().to_usize());
                    if changed_col == col {
                        continue;
                    }
                    let new_objective = lp.objective_coefficients()[changed_col]
                        - entry.coefficient() * objective_coefficient / coefficients[0];
                    lp.set_objective_coefficient(
                        changed_col,
                        if new_objective.abs() > parameters.drop_tolerance {
                            new_objective
                        } else {
                            0.0
                        },
                    );
                }
            }
            self.row_deletion_helper.mark(rows[0]);
            self.restore_stack.push(DoubletonFreeRestoreInfo {
                col,
                rows,
                coefficients,
                objective_coefficient,
                deleted_row_as_column,
            });
        }
        if self.row_deletion_helper.is_empty() {
            lp.restore_owned_transpose_sparse_matrix(transpose);
            return false;
        }
        lp.use_owned_transpose_matrix_as_reference(transpose);
        lp.delete_rows(&self.row_deletion_helper.deleted);
        true
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.row_deletion_helper.restore(solution);
        for restore in self.restore_stack.iter().rev() {
            let deleted = restore.rows[0];
            let modified = restore.rows[1];
            let pivot = restore.coefficients[0];
            solution.constraint_statuses[deleted] = match solution.variable_statuses[restore.col] {
                VariableStatus::FixedValue => ConstraintStatus::FixedValue,
                VariableStatus::AtUpperBound => {
                    if pivot > 0.0 {
                        ConstraintStatus::AtUpperBound
                    } else {
                        ConstraintStatus::AtLowerBound
                    }
                }
                VariableStatus::AtLowerBound => {
                    if pivot > 0.0 {
                        ConstraintStatus::AtLowerBound
                    } else {
                        ConstraintStatus::AtUpperBound
                    }
                }
                VariableStatus::Free => ConstraintStatus::Free,
                VariableStatus::Basic => ConstraintStatus::Basic,
            };
            let mut value = solution.primal_values[restore.col];
            for entry in &restore.deleted_row_as_column {
                let col = ColIndex::from_usize(entry.index().to_usize());
                if col != restore.col {
                    value = (-(entry.coefficient() / pivot))
                        .mul_add(solution.primal_values[col], value);
                }
            }
            solution.primal_values[restore.col] = value;
            if solution.variable_statuses[restore.col] == VariableStatus::Basic {
                debug_assert_eq!(solution.dual_values[deleted], 0.0);
            } else {
                solution.variable_statuses[restore.col] = VariableStatus::Basic;
                let reduced_cost = (-restore.coefficients[1]).mul_add(
                    solution.dual_values[modified],
                    restore.objective_coefficient,
                );
                solution.dual_values[deleted] = reduced_cost / pivot;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundBacktracking {
    col_choice: usize,
    status: VariableStatus,
    value: f64,
}

#[derive(Debug)]
struct DoubletonEqualityRestoreInfo {
    row: RowIndex,
    cols: [ColIndex; 2],
    coefficients: [f64; 2],
    rhs: f64,
    lower_backtracking: BoundBacktracking,
    upper_backtracking: BoundBacktracking,
}

#[derive(Debug)]
pub struct DoubletonEqualityRowPreprocessor {
    columns_saver: ColumnsSaver,
    column_deletion_helper: ColumnDeletionHelper,
    row_deletion_helper: RowDeletionHelper,
    restore_stack: Vec<DoubletonEqualityRestoreInfo>,
    saved_row_lower_bounds: lp_data::lp_types::DenseColumn,
    saved_row_upper_bounds: lp_data::lp_types::DenseColumn,
    saved_objective: DenseRow,
    status: ProblemStatus,
}

impl Default for DoubletonEqualityRowPreprocessor {
    fn default() -> Self {
        Self {
            columns_saver: ColumnsSaver::default(),
            column_deletion_helper: ColumnDeletionHelper::default(),
            row_deletion_helper: RowDeletionHelper::default(),
            restore_stack: Vec::new(),
            saved_row_lower_bounds: lp_data::lp_types::DenseColumn::new(),
            saved_row_upper_bounds: lp_data::lp_types::DenseColumn::new(),
            saved_objective: DenseRow::new(),
            status: ProblemStatus::Init,
        }
    }
}

impl DoubletonEqualityRowPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        self.saved_row_lower_bounds = lp.constraint_lower_bounds().clone();
        self.saved_row_upper_bounds = lp.constraint_upper_bounds().clone();
        self.saved_objective = lp.objective_coefficients().clone();

        // Move, rather than clone, the original transpose. It must remain
        // unchanged while substitutions mutate the primary column matrix.
        let original_transpose = lp.take_mutable_transpose_sparse_matrix();
        let mut sorted_rows = Vec::new();
        for position in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            let original_row = original_transpose.column(ColIndex::from_usize(position));
            if original_row.num_entries() != 2
                || lp.constraint_lower_bounds()[row] != lp.constraint_upper_bounds()[row]
            {
                continue;
            }
            let mut score = 0;
            for entry in original_row {
                score += lp
                    .sparse_column(ColIndex::from_usize(entry.index().to_usize()))
                    .num_entries();
            }
            sorted_rows.push((score, row));
        }
        sorted_rows.sort_unstable();

        for (_, row) in sorted_rows {
            let original_row = original_transpose.column(ColIndex::from_usize(row.to_usize()));
            let mut cols = [ColIndex::from_usize(0); 2];
            let mut coefficients = [0.0; 2];
            let mut count = 0;
            for entry in original_row {
                let col = ColIndex::from_usize(entry.index().to_usize());
                if self.column_deletion_helper.is_marked(col) {
                    continue;
                }
                cols[count] = col;
                coefficients[count] = entry.coefficient();
                count += 1;
            }
            if count < 2 {
                continue;
            }
            let bounds_lower = [
                lp.variable_lower_bounds()[cols[0]],
                lp.variable_lower_bounds()[cols[1]],
            ];
            let bounds_upper = [
                lp.variable_upper_bounds()[cols[0]],
                lp.variable_upper_bounds()[cols[1]],
            ];
            if bounds_lower[0] == bounds_upper[0] || bounds_lower[1] == bounds_upper[1] {
                continue;
            }
            let rhs = lp.constraint_lower_bounds()[row];
            let objective_coefficients = [
                lp.objective_coefficients()[cols[0]],
                lp.objective_coefficients()[cols[1]],
            ];
            let carry_offset = rhs / coefficients[1];
            let carry_factor = -coefficients[0] / coefficients[1];
            if !carry_offset.is_finite() || !carry_factor.is_finite() || carry_factor == 0.0 {
                self.status = ProblemStatus::Abnormal;
                break;
            }
            let mut new_lower = bounds_lower[1];
            let mut new_upper = bounds_upper[1];
            let mut carried_lower = bounds_lower[0].mul_add(carry_factor, carry_offset);
            let mut carried_upper = bounds_upper[0].mul_add(carry_factor, carry_offset);
            if carry_factor < 0.0 {
                std::mem::swap(&mut carried_lower, &mut carried_upper);
            }
            let lower_backtracking = if carried_lower <= new_lower {
                BoundBacktracking {
                    col_choice: 1,
                    status: VariableStatus::AtLowerBound,
                    value: new_lower,
                }
            } else {
                new_lower = carried_lower;
                BoundBacktracking {
                    col_choice: 0,
                    status: if carry_factor > 0.0 {
                        VariableStatus::AtLowerBound
                    } else {
                        VariableStatus::AtUpperBound
                    },
                    value: if carry_factor > 0.0 {
                        bounds_lower[0]
                    } else {
                        bounds_upper[0]
                    },
                }
            };
            let upper_backtracking = if carried_upper >= new_upper {
                BoundBacktracking {
                    col_choice: 1,
                    status: VariableStatus::AtUpperBound,
                    value: new_upper,
                }
            } else {
                new_upper = carried_upper;
                BoundBacktracking {
                    col_choice: 0,
                    status: if carry_factor > 0.0 {
                        VariableStatus::AtUpperBound
                    } else {
                        VariableStatus::AtLowerBound
                    },
                    value: if carry_factor > 0.0 {
                        bounds_upper[0]
                    } else {
                        bounds_lower[0]
                    },
                }
            };
            if is_smaller_within_tolerance(
                new_upper,
                new_lower,
                parameters.preprocessor_zero_tolerance,
            ) {
                continue;
            }
            lp.set_variable_bounds(cols[1], new_lower, new_upper);
            let restore = DoubletonEqualityRestoreInfo {
                row,
                cols,
                coefficients,
                rhs,
                lower_backtracking,
                upper_backtracking,
            };
            self.restore_stack.push(restore);
            let substitution_factor = -coefficients[1] / coefficients[0];
            let constant_offset_factor = rhs / coefficients[0];
            if !substitution_factor.is_finite()
                || substitution_factor == 0.0
                || !constant_offset_factor.is_finite()
            {
                self.status = ProblemStatus::Abnormal;
                break;
            }
            for &col in &cols {
                self.columns_saver
                    .save_column_if_not_already_done(col, lp.sparse_column(col));
            }
            lp.add_multiple_between_columns_and_delete_common_index(
                cols[0],
                cols[1],
                substitution_factor,
                row,
                parameters.drop_tolerance,
            );
            let new_objective =
                substitution_factor.mul_add(objective_coefficients[0], objective_coefficients[1]);
            lp.set_objective_coefficient(
                cols[1],
                if new_objective.abs() > parameters.drop_tolerance {
                    new_objective
                } else {
                    0.0
                },
            );
            subtract_column_multiple_from_constraint_bounds(lp, cols[0], constant_offset_factor);
            lp.mutable_sparse_column(cols[0]).clear_and_release();
            self.column_deletion_helper.mark(cols[0]);
            self.row_deletion_helper.mark(row);
        }
        if self.status != ProblemStatus::Init {
            return false;
        }
        if self.column_deletion_helper.is_empty() {
            lp.restore_owned_transpose_sparse_matrix(original_transpose);
        }
        lp.delete_columns(&self.column_deletion_helper.deleted);
        lp.delete_rows(&self.row_deletion_helper.deleted);
        !self.column_deletion_helper.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    /// Reconstructs primal values, basis statuses, and duals in reverse order.
    ///
    /// # Panics
    ///
    /// Panics if the restored basis violates upstream's singleton elimination
    /// invariant for the deleted doubleton rows.
    #[allow(clippy::too_many_lines)]
    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.column_deletion_helper.restore(solution);
        self.row_deletion_helper.restore(solution);
        let mut new_basic_columns = vec![false; solution.variable_statuses.len().to_usize()];
        for restore in self.restore_stack.iter().rev() {
            let modified = restore.cols[1];
            let deleted = restore.cols[0];
            match solution.variable_statuses[modified] {
                VariableStatus::FixedValue => {}
                VariableStatus::Free | VariableStatus::Basic => {
                    solution.variable_statuses[deleted] = VariableStatus::Basic;
                    new_basic_columns[deleted.to_usize()] = true;
                }
                VariableStatus::AtLowerBound | VariableStatus::AtUpperBound => {
                    let backtracking =
                        if solution.variable_statuses[modified] == VariableStatus::AtLowerBound {
                            restore.lower_backtracking
                        } else {
                            restore.upper_backtracking
                        };
                    let bounded = restore.cols[backtracking.col_choice];
                    let basic = restore.cols[1 - backtracking.col_choice];
                    solution.variable_statuses[bounded] = backtracking.status;
                    solution.primal_values[bounded] = backtracking.value;
                    solution.variable_statuses[basic] = VariableStatus::Basic;
                    new_basic_columns[basic.to_usize()] = true;
                }
            }
            if solution.variable_statuses[deleted] == VariableStatus::Basic {
                solution.primal_values[deleted] = (-solution.primal_values[modified])
                    .mul_add(restore.coefficients[1], restore.rhs)
                    / restore.coefficients[0];
            }
            solution.constraint_statuses[restore.row] = ConstraintStatus::FixedValue;
        }
        let num_cols = solution.variable_statuses.len().to_usize();
        let mut col_to_index = vec![BTreeSet::new(); num_cols];
        for (index, restore) in self.restore_stack.iter().enumerate() {
            col_to_index[restore.cols[1].to_usize()].insert(index);
            col_to_index[restore.cols[0].to_usize()].insert(index);
        }
        let mut singleton_cols = Vec::new();
        for (position, &is_new_basic) in new_basic_columns.iter().enumerate() {
            if is_new_basic && col_to_index[position].len() == 1 {
                singleton_cols.push(ColIndex::from_usize(position));
            }
        }
        while let Some(col) = singleton_cols.pop() {
            if !new_basic_columns[col.to_usize()] || col_to_index[col.to_usize()].is_empty() {
                continue;
            }
            assert_eq!(col_to_index[col.to_usize()].len(), 1);
            let index = *col_to_index[col.to_usize()].first().unwrap();
            let restore = &self.restore_stack[index];
            let choice = usize::from(restore.cols[1] == col);
            let saved_col = self.columns_saver.saved_column(restore.cols[choice]);
            let current_reduced_cost = self.saved_objective[restore.cols[choice]]
                - lp_data::lp_utils::precise_sparse_scalar_product(
                    solution.dual_values.as_slice(),
                    saved_col,
                );
            solution.dual_values[restore.row] = current_reduced_cost / restore.coefficients[choice];
            for &affected in &restore.cols {
                let set = &mut col_to_index[affected.to_usize()];
                set.remove(&index);
                if set.len() == 1 {
                    singleton_cols.push(affected);
                }
            }
        }
        for position in 0..solution.constraint_statuses.len().to_usize() {
            let row = RowIndex::from_usize(position);
            if solution.constraint_statuses[row] == ConstraintStatus::FixedValue
                && self.saved_row_lower_bounds[row] != self.saved_row_upper_bounds[row]
            {
                solution.constraint_statuses[row] = if solution.dual_values[row] > 0.0 {
                    ConstraintStatus::AtLowerBound
                } else {
                    ConstraintStatus::AtUpperBound
                };
            }
        }
    }
}

#[derive(Debug)]
pub struct ProportionalRowPreprocessor {
    row_deletion_helper: RowDeletionHelper,
    row_factors: Vec<f64>,
    lower_bound_sources: Vec<RowIndex>,
    upper_bound_sources: Vec<RowIndex>,
    lp_is_maximization_problem: bool,
    status: ProblemStatus,
}

impl Default for ProportionalRowPreprocessor {
    fn default() -> Self {
        Self {
            row_deletion_helper: RowDeletionHelper::default(),
            row_factors: Vec::new(),
            lower_bound_sources: Vec::new(),
            upper_bound_sources: Vec::new(),
            lp_is_maximization_problem: false,
            status: ProblemStatus::Init,
        }
    }
}

impl ProportionalRowPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        let rows = lp.num_constraints().to_usize();
        let (mut mapping, factors) = {
            let transpose = lp.transpose_sparse_matrix();
            let mut factors = vec![0.0; rows];
            for (position, factor) in factors.iter_mut().enumerate() {
                let row = transpose.column(ColIndex::from_usize(position));
                if !row.is_empty() {
                    *factor = row.first_coefficient();
                }
            }
            (
                find_proportional_columns(&transpose, parameters.preprocessor_zero_tolerance),
                factors,
            )
        };
        self.row_factors = factors;
        let mut lower_bounds = vec![f64::NEG_INFINITY; rows];
        let mut upper_bounds = vec![f64::INFINITY; rows];
        self.lower_bound_sources = vec![INVALID_ROW; rows];
        self.upper_bound_sources = vec![INVALID_ROW; rows];
        let mut is_representative = vec![false; rows];
        for position in 0..rows {
            let row_as_col = ColIndex::from_usize(position);
            let representative = mapping[row_as_col];
            if representative != INVALID_COL {
                mapping[representative] = representative;
                is_representative[representative.to_usize()] = true;
            }
        }
        for position in 0..rows {
            let row = RowIndex::from_usize(position);
            let representative_as_col = mapping[ColIndex::from_usize(position)];
            if representative_as_col == INVALID_COL {
                continue;
            }
            self.row_deletion_helper.mark(row);
            let representative = RowIndex::from_usize(representative_as_col.to_usize());
            let factor = self.row_factors[representative.to_usize()] / self.row_factors[position];
            let mut implied_lower = factor * lp.constraint_lower_bounds()[row];
            let mut implied_upper = factor * lp.constraint_upper_bounds()[row];
            if factor < 0.0 {
                std::mem::swap(&mut implied_lower, &mut implied_upper);
            }
            if implied_lower >= lower_bounds[representative.to_usize()] {
                lower_bounds[representative.to_usize()] = implied_lower;
                self.lower_bound_sources[representative.to_usize()] = row;
            }
            if implied_upper <= upper_bounds[representative.to_usize()] {
                upper_bounds[representative.to_usize()] = implied_upper;
                self.upper_bound_sources[representative.to_usize()] = row;
            }
        }
        for (position, &representative) in is_representative.iter().enumerate() {
            if !representative {
                continue;
            }
            let row = RowIndex::from_usize(position);
            let lower_source = self.lower_bound_sources[position];
            let upper_source = self.upper_bound_sources[position];
            self.lower_bound_sources[position] = INVALID_ROW;
            self.upper_bound_sources[position] = INVALID_ROW;
            debug_assert_ne!(lower_source, INVALID_ROW);
            debug_assert_ne!(upper_source, INVALID_ROW);
            if lower_source == upper_source {
                self.row_deletion_helper.unmark(lower_source);
                continue;
            }
            if !is_smaller_within_tolerance(
                lower_bounds[position],
                upper_bounds[position],
                parameters.solution_feasibility_tolerance,
            ) {
                self.status = ProblemStatus::PrimalInfeasible;
                return false;
            }
            if lp.constraint_lower_bounds()[lower_source]
                == lp.constraint_upper_bounds()[lower_source]
            {
                self.row_deletion_helper.unmark(lower_source);
                continue;
            }
            if lp.constraint_lower_bounds()[upper_source]
                == lp.constraint_upper_bounds()[upper_source]
            {
                self.row_deletion_helper.unmark(upper_source);
                continue;
            }
            let mut new_representative = lower_source;
            let mut other = upper_source;
            if self.row_factors[new_representative.to_usize()].abs()
                < self.row_factors[other.to_usize()].abs()
            {
                std::mem::swap(&mut new_representative, &mut other);
            }
            let factor = self.row_factors[new_representative.to_usize()]
                / self.row_factors[other.to_usize()];
            let mut new_lower = factor * lp.constraint_lower_bounds()[other];
            let mut new_upper = factor * lp.constraint_upper_bounds()[other];
            if factor < 0.0 {
                std::mem::swap(&mut new_lower, &mut new_upper);
            }
            self.lower_bound_sources[new_representative.to_usize()] = new_representative;
            self.upper_bound_sources[new_representative.to_usize()] = new_representative;
            if new_lower > lp.constraint_lower_bounds()[new_representative] {
                self.lower_bound_sources[new_representative.to_usize()] = other;
            } else {
                new_lower = lp.constraint_lower_bounds()[new_representative];
            }
            if new_upper < lp.constraint_upper_bounds()[new_representative] {
                self.upper_bound_sources[new_representative.to_usize()] = other;
            } else {
                new_upper = lp.constraint_upper_bounds()[new_representative];
            }
            let new_lower_source = self.lower_bound_sources[new_representative.to_usize()];
            if new_lower_source == self.upper_bound_sources[new_representative.to_usize()] {
                self.row_deletion_helper.unmark(new_lower_source);
                self.lower_bound_sources[new_representative.to_usize()] = INVALID_ROW;
                self.upper_bound_sources[new_representative.to_usize()] = INVALID_ROW;
                continue;
            }
            debug_assert!(is_smaller_within_tolerance(
                new_lower,
                new_upper,
                parameters.solution_feasibility_tolerance,
            ));
            if new_lower > new_upper {
                if self.lower_bound_sources[new_representative.to_usize()] == new_representative {
                    new_upper = lp.constraint_lower_bounds()[new_representative];
                } else {
                    new_lower = lp.constraint_upper_bounds()[new_representative];
                }
            }
            self.row_deletion_helper.unmark(new_representative);
            lp.set_constraint_bounds(new_representative, new_lower, new_upper);
            let _ = row;
        }
        self.lp_is_maximization_problem = lp.is_maximization_problem();
        lp.delete_rows(&self.row_deletion_helper.deleted);
        !self.row_deletion_helper.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.row_deletion_helper.restore(solution);
        for position in 0..solution.dual_values.len().to_usize() {
            let row = RowIndex::from_usize(position);
            let lower_source = self.lower_bound_sources[position];
            let upper_source = self.upper_bound_sources[position];
            if lower_source == INVALID_ROW && upper_source == INVALID_ROW {
                continue;
            }
            let mut status = solution.constraint_statuses[row];
            if status == ConstraintStatus::Basic {
                continue;
            }
            if status == ConstraintStatus::FixedValue {
                let corrected = if self.lp_is_maximization_problem {
                    -solution.dual_values[row]
                } else {
                    solution.dual_values[row]
                };
                if corrected != 0.0 {
                    status = if corrected > 0.0 {
                        ConstraintStatus::AtLowerBound
                    } else {
                        ConstraintStatus::AtUpperBound
                    };
                }
            }
            if lower_source != row && status == ConstraintStatus::AtLowerBound {
                let factor = self.row_factors[position] / self.row_factors[lower_source.to_usize()];
                solution.dual_values[lower_source] = factor * solution.dual_values[row];
                solution.dual_values[row] = 0.0;
                solution.constraint_statuses[row] = ConstraintStatus::Basic;
                solution.constraint_statuses[lower_source] = if factor > 0.0 {
                    ConstraintStatus::AtLowerBound
                } else {
                    ConstraintStatus::AtUpperBound
                };
            }
            if upper_source != row && status == ConstraintStatus::AtUpperBound {
                let factor = self.row_factors[position] / self.row_factors[upper_source.to_usize()];
                solution.dual_values[upper_source] = factor * solution.dual_values[row];
                solution.dual_values[row] = 0.0;
                solution.constraint_statuses[row] = ConstraintStatus::Basic;
                solution.constraint_statuses[upper_source] = if factor > 0.0 {
                    ConstraintStatus::AtUpperBound
                } else {
                    ConstraintStatus::AtLowerBound
                };
            }
            if solution.constraint_statuses[row] == ConstraintStatus::FixedValue {
                solution.constraint_statuses[row] = if lower_source == row {
                    ConstraintStatus::AtLowerBound
                } else {
                    ConstraintStatus::AtUpperBound
                };
            }
        }
    }
}

#[derive(Debug)]
pub struct ProportionalColumnPreprocessor {
    column_deletion_helper: ColumnDeletionHelper,
    column_factors: DenseRow,
    merged_columns: Vec<ColIndex>,
    lower_bounds: DenseRow,
    upper_bounds: DenseRow,
    new_lower_bounds: DenseRow,
    new_upper_bounds: DenseRow,
    status: ProblemStatus,
}

impl Default for ProportionalColumnPreprocessor {
    fn default() -> Self {
        Self {
            column_deletion_helper: ColumnDeletionHelper::default(),
            column_factors: DenseRow::new(),
            merged_columns: Vec::new(),
            lower_bounds: DenseRow::new(),
            upper_bounds: DenseRow::new(),
            new_lower_bounds: DenseRow::new(),
            new_upper_bounds: DenseRow::new(),
            status: ProblemStatus::Init,
        }
    }
}

impl ProportionalColumnPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        let mut mapping =
            find_proportional_columns(lp.matrix(), parameters.preprocessor_zero_tolerance);
        let mut proportional_columns = Vec::new();
        for position in 0..mapping.len().to_usize() {
            let col = ColIndex::from_usize(position);
            let representative = mapping[col];
            if representative != INVALID_COL {
                if mapping[representative] == INVALID_COL {
                    proportional_columns.push(representative);
                    mapping[representative] = representative;
                }
                proportional_columns.push(col);
            }
        }
        if proportional_columns.is_empty() {
            return false;
        }
        let cols = lp.num_variables().to_usize();
        self.column_factors.resize(lp.num_variables(), 0.0);
        for &col in &proportional_columns {
            self.column_factors[col] = lp.sparse_column(col).first_coefficient();
        }
        let mut slope_lower_bound = vec![f64::NEG_INFINITY; cols];
        let mut slope_upper_bound = vec![f64::INFINITY; cols];
        for &col in &proportional_columns {
            let representative = mapping[col].to_usize();
            let mut upper_bounded = lp.variable_upper_bounds()[col] == f64::INFINITY;
            let mut lower_bounded = lp.variable_lower_bounds()[col] == f64::NEG_INFINITY;
            if self.column_factors[col] < 0.0 {
                std::mem::swap(&mut lower_bounded, &mut upper_bounded);
            }
            let slope = lp.objective_coefficient_for_minimization(col) / self.column_factors[col];
            if lower_bounded {
                slope_lower_bound[representative] = slope_lower_bound[representative].max(slope);
            }
            if upper_bounded {
                slope_upper_bound[representative] = slope_upper_bound[representative].min(slope);
            }
        }
        for &col in &proportional_columns {
            let representative = mapping[col];
            if representative == col
                && !is_smaller_within_tolerance(
                    slope_lower_bound[col.to_usize()],
                    slope_upper_bound[col.to_usize()],
                    parameters.solution_feasibility_tolerance,
                )
            {
                self.status = ProblemStatus::InfeasibleOrUnbounded;
                return false;
            }
        }
        for &col in &proportional_columns {
            let representative = mapping[col].to_usize();
            let slope = lp.objective_coefficient_for_minimization(col) / self.column_factors[col];
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            let target = if !is_smaller_within_tolerance(
                slope_lower_bound[representative],
                slope,
                parameters.solution_feasibility_tolerance,
            ) {
                Some(if self.column_factors[col] >= 0.0 {
                    upper
                } else {
                    lower
                })
            } else if !is_smaller_within_tolerance(
                slope,
                slope_upper_bound[representative],
                parameters.solution_feasibility_tolerance,
            ) {
                Some(if self.column_factors[col] >= 0.0 {
                    lower
                } else {
                    upper
                })
            } else {
                None
            };
            if let Some(target) = target {
                mapping[col] = INVALID_COL;
                if !target.is_finite() {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                subtract_column_multiple_from_constraint_bounds(lp, col, target);
                self.column_deletion_helper.mark_with_state(
                    col,
                    target,
                    status_at_value(target, lower, upper),
                );
            }
        }
        let mut sorted_columns = Vec::new();
        for &col in &proportional_columns {
            if mapping[col] != INVALID_COL {
                sorted_columns.push((
                    col,
                    mapping[col],
                    lp.objective_coefficients()[col] / self.column_factors[col],
                ));
            }
        }
        sorted_columns.sort_unstable_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| {
                    left.2
                        .partial_cmp(&right.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.0.cmp(&right.0))
        });
        self.merged_columns = vec![INVALID_COL; cols];
        self.lower_bounds
            .resize(lp.num_variables(), f64::NEG_INFINITY);
        self.upper_bounds.resize(lp.num_variables(), f64::INFINITY);
        self.new_lower_bounds
            .resize(lp.num_variables(), f64::NEG_INFINITY);
        self.new_upper_bounds
            .resize(lp.num_variables(), f64::INFINITY);
        let mut position = 0;
        while position < sorted_columns.len() {
            let (target_col, target_representative, target_scaled_cost) = sorted_columns[position];
            self.lower_bounds[target_col] = lp.variable_lower_bounds()[target_col];
            self.upper_bounds[target_col] = lp.variable_upper_bounds()[target_col];
            position += 1;
            let mut num_merged = 0;
            while position < sorted_columns.len() {
                let (col, representative, scaled_cost) = sorted_columns[position];
                if representative != target_representative
                    || (scaled_cost - target_scaled_cost).abs()
                        >= parameters.preprocessor_zero_tolerance
                {
                    break;
                }
                position += 1;
                num_merged += 1;
                let lower = lp.variable_lower_bounds()[col];
                let upper = lp.variable_upper_bounds()[col];
                self.lower_bounds[col] = lower;
                self.upper_bounds[col] = upper;
                self.merged_columns[col.to_usize()] = target_col;
                let bound_factor = self.column_factors[col] / self.column_factors[target_col];
                let target_value = min_in_magnitude_or_zero_if_infinite(lower, upper);
                let mut lower_diff = (lower - target_value) * bound_factor;
                let mut upper_diff = (upper - target_value) * bound_factor;
                if bound_factor < 0.0 {
                    std::mem::swap(&mut lower_diff, &mut upper_diff);
                }
                lp.set_variable_bounds(
                    target_col,
                    lp.variable_lower_bounds()[target_col] + lower_diff,
                    lp.variable_upper_bounds()[target_col] + upper_diff,
                );
                subtract_column_multiple_from_constraint_bounds(lp, col, target_value);
                self.column_deletion_helper.mark_with_state(
                    col,
                    target_value,
                    status_at_value(target_value, lower, upper),
                );
            }
            if num_merged > 0 {
                self.merged_columns[target_col.to_usize()] = target_col;
                let target_value = min_in_magnitude_or_zero_if_infinite(
                    self.lower_bounds[target_col],
                    self.upper_bounds[target_col],
                );
                lp.set_variable_bounds(
                    target_col,
                    lp.variable_lower_bounds()[target_col] - target_value,
                    lp.variable_upper_bounds()[target_col] - target_value,
                );
                subtract_column_multiple_from_constraint_bounds(lp, target_col, target_value);
                self.new_lower_bounds[target_col] = lp.variable_lower_bounds()[target_col];
                self.new_upper_bounds[target_col] = lp.variable_upper_bounds()[target_col];
            }
        }
        lp.delete_columns(&self.column_deletion_helper.deleted);
        !self.column_deletion_helper.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    #[allow(clippy::too_many_lines)]
    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.column_deletion_helper.restore(solution);
        let cols = self.merged_columns.len();
        let mut is_representative_basic = vec![false; cols];
        let mut is_distance_to_upper_bound = vec![false; cols];
        let mut distance_to_bound = vec![0.0; cols];
        let mut wanted_value = vec![0.0; cols];
        for position in 0..cols {
            let col = ColIndex::from_usize(position);
            if self.merged_columns[position] != col {
                continue;
            }
            let value = solution.primal_values[col];
            let upper_distance = self.new_upper_bounds[col] - value;
            let lower_distance = value - self.new_lower_bounds[col];
            if upper_distance < lower_distance {
                distance_to_bound[position] = upper_distance;
                is_distance_to_upper_bound[position] = true;
            } else {
                distance_to_bound[position] = lower_distance;
            }
            is_representative_basic[position] =
                solution.variable_statuses[col] == VariableStatus::Basic;
            wanted_value[position] = value;
            let initial_value = min_in_magnitude_or_zero_if_infinite(
                self.lower_bounds[col],
                self.upper_bounds[col],
            );
            solution.primal_values[col] = initial_value;
            solution.variable_statuses[col] = status_at_value(
                initial_value,
                self.lower_bounds[col],
                self.upper_bounds[col],
            );
        }
        for position in 0..cols {
            let col = ColIndex::from_usize(position);
            let representative = self.merged_columns[position];
            if representative == INVALID_COL {
                continue;
            }
            let rep = representative.to_usize();
            if distance_to_bound[rep].is_finite() {
                let bound_factor = self.column_factors[col] / self.column_factors[representative];
                let scaled_distance = distance_to_bound[rep] / bound_factor.abs();
                let width = self.upper_bounds[col] - self.lower_bounds[col];
                let to_upper_bound = (bound_factor > 0.0) == is_distance_to_upper_bound[rep];
                if width <= scaled_distance {
                    let value = if to_upper_bound {
                        self.lower_bounds[col]
                    } else {
                        self.upper_bounds[col]
                    };
                    solution.primal_values[col] = value;
                    solution.variable_statuses[col] =
                        status_at_value(value, self.lower_bounds[col], self.upper_bounds[col]);
                    distance_to_bound[rep] -= width * bound_factor.abs();
                } else {
                    let value = if to_upper_bound {
                        self.upper_bounds[col] - scaled_distance
                    } else {
                        self.lower_bounds[col] + scaled_distance
                    };
                    solution.primal_values[col] = value;
                    solution.variable_statuses[col] = if is_representative_basic[rep] {
                        VariableStatus::Basic
                    } else {
                        status_at_value(value, self.lower_bounds[col], self.upper_bounds[col])
                    };
                    distance_to_bound[rep] = 0.0;
                    is_representative_basic[rep] = false;
                }
            } else {
                let error = wanted_value[rep];
                if error == 0.0 {
                    if is_representative_basic[rep] {
                        solution.variable_statuses[col] = VariableStatus::Basic;
                        is_representative_basic[rep] = false;
                    }
                } else {
                    let bound_factor =
                        self.column_factors[col] / self.column_factors[representative];
                    let use_this_variable = if error * bound_factor > 0.0 {
                        self.upper_bounds[col] == f64::INFINITY
                    } else {
                        self.lower_bounds[col] == f64::NEG_INFINITY
                    };
                    if use_this_variable {
                        wanted_value[rep] = 0.0;
                        solution.primal_values[col] += error / bound_factor;
                        if is_representative_basic[rep] {
                            solution.variable_statuses[col] = VariableStatus::Basic;
                            is_representative_basic[rep] = false;
                        } else {
                            solution.variable_statuses[col] = VariableStatus::Free;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct DualizerPreprocessor {
    primal_num_cols: ColIndex,
    primal_num_rows: RowIndex,
    primal_is_maximization_problem: bool,
    variable_lower_bounds: DenseRow,
    variable_upper_bounds: DenseRow,
    dual_status_correspondence: Vec<VariableStatus>,
    slack_or_surplus_mapping: Vec<ColIndex>,
    duplicated_rows: lp_data::lp_types::TypedVec<RowIndex, ColIndex>,
}

impl DualizerPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Converts the LP to its dual after shifting each bounded column.
    ///
    /// # Panics
    ///
    /// Panics if a free constraint reaches this pass; upstream requires the
    /// earlier free-constraint pass to have removed it.
    #[allow(clippy::too_many_lines)]
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        if parameters.solve_dual_problem == SolverBehavior::NeverDo {
            return false;
        }
        self.primal_num_cols = lp.num_variables();
        self.primal_num_rows = lp.num_constraints();
        self.primal_is_maximization_problem = lp.is_maximization_problem();
        if parameters.solve_dual_problem == SolverBehavior::LetSolverDecide
            && (f64::from(self.primal_num_rows.value()))
                < parameters.dualizer_threshold * f64::from(self.primal_num_cols.value())
        {
            return false;
        }
        self.variable_lower_bounds.resize(lp.num_variables(), 0.0);
        self.variable_upper_bounds.resize(lp.num_variables(), 0.0);
        for position in 0..self.primal_num_cols.to_usize() {
            let col = ColIndex::from_usize(position);
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            self.variable_lower_bounds[col] = lower;
            self.variable_upper_bounds[col] = upper;
            let value = min_in_magnitude_or_zero_if_infinite(lower, upper);
            if value != 0.0 {
                lp.set_variable_bounds(col, lower - value, upper - value);
                subtract_column_multiple_from_constraint_bounds(lp, col, value);
            }
        }
        for position in 0..self.primal_num_rows.to_usize() {
            let row = RowIndex::from_usize(position);
            let lower = lp.constraint_lower_bounds()[row];
            let upper = lp.constraint_upper_bounds()[row];
            let status = if lower == upper {
                VariableStatus::FixedValue
            } else if upper != f64::INFINITY {
                VariableStatus::AtUpperBound
            } else if lower != f64::NEG_INFINITY {
                VariableStatus::AtLowerBound
            } else {
                panic!("GLOP dualizer requires free rows to be removed first");
            };
            self.dual_status_correspondence.push(status);
        }
        for position in 0..self.primal_num_cols.to_usize() {
            let col = ColIndex::from_usize(position);
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            if lower != f64::NEG_INFINITY {
                self.dual_status_correspondence.push(if upper == lower {
                    VariableStatus::FixedValue
                } else {
                    VariableStatus::AtLowerBound
                });
                self.slack_or_surplus_mapping.push(col);
            }
        }
        for position in 0..self.primal_num_cols.to_usize() {
            let col = ColIndex::from_usize(position);
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            if upper != f64::INFINITY {
                self.dual_status_correspondence.push(if upper == lower {
                    VariableStatus::FixedValue
                } else {
                    VariableStatus::AtUpperBound
                });
                self.slack_or_surplus_mapping.push(col);
            }
        }
        let mut dual = LinearProgram::new();
        self.duplicated_rows = dual.populate_from_dual(lp);
        lp.swap(&mut dual);
        true
    }

    /// Reconstructs the original-side basis and solution from its dual.
    ///
    /// # Panics
    ///
    /// Panics if the dual solution dimensions do not match the transformed LP.
    #[allow(clippy::too_many_lines)]
    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        let mut new_primal_values = DenseRow::filled(self.primal_num_cols, 0.0);
        let mut new_variable_statuses = lp_data::lp_types::VariableStatusRow::filled(
            self.primal_num_cols,
            VariableStatus::Free,
        );
        for position in 0..self.primal_num_cols.to_usize() {
            let col = ColIndex::from_usize(position);
            let row = RowIndex::from_usize(position);
            let lower = self.variable_lower_bounds[col];
            let upper = self.variable_upper_bounds[col];
            let shift = min_in_magnitude_or_zero_if_infinite(lower, upper);
            new_primal_values[col] = solution.dual_values[row] + shift;
            new_variable_statuses[col] =
                if solution.constraint_statuses[row] == ConstraintStatus::Basic {
                    status_at_value(shift, lower, upper)
                } else {
                    VariableStatus::Basic
                };
        }
        let begin = self.primal_num_rows.to_usize();
        for index in begin..self.dual_status_correspondence.len() {
            let dual_col = ColIndex::from_usize(index);
            if solution.variable_statuses[dual_col] == VariableStatus::Basic {
                let col = self.slack_or_surplus_mapping[index - begin];
                let status = self.dual_status_correspondence[index];
                new_variable_statuses[col] = status;
                new_primal_values[col] = if status == VariableStatus::AtUpperBound
                    || status == VariableStatus::FixedValue
                {
                    self.variable_upper_bounds[col]
                } else {
                    self.variable_lower_bounds[col]
                };
            }
        }
        let mut new_dual_values = lp_data::lp_types::DenseColumn::filled(self.primal_num_rows, 0.0);
        let mut new_constraint_statuses = lp_data::lp_types::ConstraintStatusColumn::filled(
            self.primal_num_rows,
            ConstraintStatus::Free,
        );
        let sign = if self.primal_is_maximization_problem {
            -1.0
        } else {
            1.0
        };
        for position in 0..self.primal_num_rows.to_usize() {
            let row = RowIndex::from_usize(position);
            let col = ColIndex::from_usize(position);
            new_dual_values[row] = sign * solution.primal_values[col];
            if solution.variable_statuses[col] == VariableStatus::Basic {
                new_constraint_statuses[row] = self.dual_status_correspondence[position].into();
            } else {
                new_constraint_statuses[row] = ConstraintStatus::Basic;
                if self.duplicated_rows[row] != INVALID_COL
                    && solution.variable_statuses[self.duplicated_rows[row]]
                        == VariableStatus::Basic
                {
                    new_constraint_statuses[row] = ConstraintStatus::AtLowerBound;
                }
            }
            if self.duplicated_rows[row] != INVALID_COL {
                new_dual_values[row] = sign.mul_add(
                    solution.primal_values[self.duplicated_rows[row]],
                    new_dual_values[row],
                );
            }
        }
        solution.status = match solution.status {
            ProblemStatus::PrimalInfeasible => ProblemStatus::DualInfeasible,
            ProblemStatus::DualInfeasible => ProblemStatus::PrimalInfeasible,
            ProblemStatus::PrimalUnbounded => ProblemStatus::DualUnbounded,
            ProblemStatus::DualUnbounded => ProblemStatus::PrimalUnbounded,
            ProblemStatus::PrimalFeasible => ProblemStatus::DualFeasible,
            ProblemStatus::DualFeasible => ProblemStatus::PrimalFeasible,
            other => other,
        };
        solution.primal_values = new_primal_values;
        solution.dual_values = new_dual_values;
        solution.variable_statuses = new_variable_statuses;
        solution.constraint_statuses = new_constraint_statuses;
    }
}

#[derive(Debug)]
pub struct UnconstrainedVariablePreprocessor {
    dual_lower: Vec<f64>,
    dual_upper: Vec<f64>,
    may_have_participated_lower: Vec<bool>,
    may_have_participated_upper: Vec<bool>,
    column_deletion_helper: ColumnDeletionHelper,
    row_deletion_helper: RowDeletionHelper,
    rows_saver: ColumnsSaver,
    rhs: Vec<f64>,
    activity_sign_correction: Vec<f64>,
    is_unbounded: Vec<bool>,
    status: ProblemStatus,
}

impl Default for UnconstrainedVariablePreprocessor {
    fn default() -> Self {
        Self {
            dual_lower: Vec::new(),
            dual_upper: Vec::new(),
            may_have_participated_lower: Vec::new(),
            may_have_participated_upper: Vec::new(),
            column_deletion_helper: ColumnDeletionHelper::default(),
            row_deletion_helper: RowDeletionHelper::default(),
            rows_saver: ColumnsSaver::default(),
            rhs: Vec::new(),
            activity_sign_correction: Vec::new(),
            is_unbounded: Vec::new(),
            status: ProblemStatus::Init,
        }
    }
}

impl UnconstrainedVariablePreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn remove_zero_cost_unconstrained_variable(
        &mut self,
        col: ColIndex,
        target_bound: f64,
        lp: &LinearProgram,
    ) {
        if self.rhs.is_empty() {
            self.rhs.resize(lp.num_constraints().to_usize(), 0.0);
            self.activity_sign_correction
                .resize(lp.num_constraints().to_usize(), 1.0);
            self.is_unbounded
                .resize(lp.num_variables().to_usize(), false);
        }
        let is_unbounded_up = target_bound == f64::INFINITY;
        for entry in lp.sparse_column(col) {
            let row = entry.index();
            if !self.row_deletion_helper.is_marked(row) {
                self.row_deletion_helper.mark(row);
                let transpose = lp.transpose_sparse_matrix();
                self.rows_saver.save_column(
                    ColIndex::from_usize(row.to_usize()),
                    transpose.column(ColIndex::from_usize(row.to_usize())),
                );
            }
            let upper_relevant = if entry.coefficient() > 0.0 {
                !is_unbounded_up
            } else {
                is_unbounded_up
            };
            self.activity_sign_correction[row.to_usize()] = if upper_relevant { 1.0 } else { -1.0 };
            self.rhs[row.to_usize()] = if upper_relevant {
                lp.constraint_upper_bounds()[row]
            } else {
                lp.constraint_lower_bounds()[row]
            };
        }
        self.is_unbounded[col.to_usize()] = true;
        let lower = lp.variable_lower_bounds()[col];
        let upper = lp.variable_upper_bounds()[col];
        let initial = min_in_magnitude_or_zero_if_infinite(lower, upper);
        self.column_deletion_helper.mark_with_state(
            col,
            initial,
            status_at_value(initial, lower, upper),
        );
    }

    #[allow(clippy::too_many_lines)]
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        *self = Self::default();
        let low_tolerance = parameters.preprocessor_zero_tolerance;
        let high_tolerance = 1e-4;
        let rows = lp.num_constraints().to_usize();
        let cols = lp.num_variables().to_usize();
        self.dual_lower = vec![f64::NEG_INFINITY; rows];
        self.dual_upper = vec![f64::INFINITY; rows];
        for position in 0..rows {
            let row = RowIndex::from_usize(position);
            if lp.constraint_lower_bounds()[row] == f64::NEG_INFINITY {
                self.dual_upper[position] = 0.0;
            }
            if lp.constraint_upper_bounds()[row] == f64::INFINITY {
                self.dual_lower[position] = 0.0;
            }
        }
        self.may_have_participated_lower = vec![false; cols];
        self.may_have_participated_upper = vec![false; cols];
        let mut columns_to_process = (0..cols).map(ColIndex::from_usize).collect::<VecDeque<_>>();
        let mut in_columns_to_process = vec![true; cols];
        let mut changed_rows = Vec::new();
        for _ in 0..5 * cols {
            let Some(col) = columns_to_process.pop_front() else {
                break;
            };
            in_columns_to_process[col.to_usize()] = false;
            if self.column_deletion_helper.is_marked(col) {
                continue;
            }
            let column = lp.sparse_column(col);
            let cost = lp.objective_coefficient_for_minimization(col);
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            let mut rc_lower = SumWithNegativeInfiniteAndOneMissing::default();
            let mut rc_upper = SumWithPositiveInfiniteAndOneMissing::default();
            rc_lower.add(cost);
            rc_upper.add(cost);
            for entry in column {
                let row = entry.index().to_usize();
                if self.row_deletion_helper.is_marked(entry.index()) {
                    continue;
                }
                let coeff = entry.coefficient();
                if coeff > 0.0 {
                    rc_lower.add(-coeff * self.dual_upper[row]);
                    rc_upper.add(-coeff * self.dual_lower[row]);
                } else {
                    rc_lower.add(-coeff * self.dual_lower[row]);
                    rc_upper.add(-coeff * self.dual_upper[row]);
                }
            }
            let mut can_be_removed = false;
            let mut target_bound = 0.0;
            let mut rc_is_away_from_zero = false;
            if rc_upper.sum() <= low_tolerance {
                can_be_removed = !self.may_have_participated_upper[col.to_usize()];
                target_bound = upper;
                rc_is_away_from_zero = rc_upper.sum() <= -high_tolerance;
            }
            if rc_lower.sum() >= -low_tolerance && (!can_be_removed || !target_bound.is_finite()) {
                can_be_removed = !self.may_have_participated_lower[col.to_usize()];
                target_bound = lower;
                rc_is_away_from_zero = rc_lower.sum() >= high_tolerance;
            }
            if can_be_removed {
                if target_bound.is_finite() {
                    self.column_deletion_helper.mark_with_state(
                        col,
                        target_bound,
                        status_at_value(target_bound, lower, upper),
                    );
                    continue;
                }
                if rc_is_away_from_zero {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                if cost != 0.0 {
                    continue;
                }
                let direction = if target_bound == f64::INFINITY {
                    1.0
                } else {
                    -1.0
                };
                let mut skip = false;
                for entry in column {
                    let row = entry.index();
                    let blocked = if direction * entry.coefficient() > 0.0 {
                        lp.constraint_upper_bounds()[row] != f64::INFINITY
                    } else {
                        lp.constraint_lower_bounds()[row] != f64::NEG_INFINITY
                    };
                    if blocked {
                        skip = true;
                        break;
                    }
                }
                if !skip {
                    self.remove_zero_cost_unconstrained_variable(col, target_bound, lp);
                }
                continue;
            }
            if lower != f64::NEG_INFINITY && upper != f64::INFINITY {
                continue;
            }
            changed_rows.clear();
            for entry in column {
                let row = entry.index();
                if self.row_deletion_helper.is_marked(row) {
                    continue;
                }
                let c = entry.coefficient();
                let position = row.to_usize();
                if upper == f64::INFINITY {
                    if c > 0.0 {
                        let candidate = rc_upper.sum_without_ub(-c * self.dual_lower[position]) / c;
                        if candidate < self.dual_upper[position] {
                            self.dual_upper[position] = candidate;
                            self.may_have_participated_lower[col.to_usize()] = true;
                            changed_rows.push(row);
                        }
                    } else {
                        let candidate = rc_upper.sum_without_ub(-c * self.dual_upper[position]) / c;
                        if candidate > self.dual_lower[position] {
                            self.dual_lower[position] = candidate;
                            self.may_have_participated_lower[col.to_usize()] = true;
                            changed_rows.push(row);
                        }
                    }
                }
                if lower == f64::NEG_INFINITY {
                    if c > 0.0 {
                        let candidate = rc_lower.sum_without_lb(-c * self.dual_upper[position]) / c;
                        if candidate > self.dual_lower[position] {
                            self.dual_lower[position] = candidate;
                            self.may_have_participated_upper[col.to_usize()] = true;
                            changed_rows.push(row);
                        }
                    } else {
                        let candidate = rc_lower.sum_without_lb(-c * self.dual_lower[position]) / c;
                        if candidate < self.dual_upper[position] {
                            self.dual_upper[position] = candidate;
                            self.may_have_participated_upper[col.to_usize()] = true;
                            changed_rows.push(row);
                        }
                    }
                }
            }
            if !changed_rows.is_empty() {
                let transpose = lp.transpose_sparse_matrix();
                for &row in &changed_rows {
                    for entry in transpose.column(ColIndex::from_usize(row.to_usize())) {
                        let neighbor = entry.index().to_usize();
                        if !in_columns_to_process[neighbor] {
                            columns_to_process.push_back(ColIndex::from_usize(neighbor));
                            in_columns_to_process[neighbor] = true;
                        }
                    }
                }
            }
        }
        for position in 0..self.column_deletion_helper.deleted.len().to_usize() {
            let col = ColIndex::from_usize(position);
            if self.column_deletion_helper.is_marked(col) {
                let target = self.column_deletion_helper.stored_values[col];
                subtract_column_multiple_from_constraint_bounds(lp, col, target);
            }
        }
        lp.delete_columns(&self.column_deletion_helper.deleted);
        lp.delete_rows(&self.row_deletion_helper.deleted);
        !self.column_deletion_helper.is_empty() || !self.row_deletion_helper.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.column_deletion_helper.restore(solution);
        self.row_deletion_helper.restore(solution);
        let mut entries = Vec::new();
        for position in 0..solution.dual_values.len().to_usize() {
            let row = RowIndex::from_usize(position);
            if !self.row_deletion_helper.is_marked(row) {
                continue;
            }
            let mut last = None;
            for entry in self.rows_saver.saved_column(ColIndex::from_usize(position)) {
                let col = ColIndex::from_usize(entry.index().to_usize());
                if self.is_unbounded[col.to_usize()] {
                    last = Some((col, entry.coefficient()));
                }
            }
            if let Some((col, coefficient)) = last {
                entries.push((col, row, coefficient));
            }
        }
        entries.sort_unstable_by_key(|&(col, row, _)| (col, row));
        let mut i = 0;
        while i < entries.len() {
            let col = entries[i].0;
            let mut shift: f64 = 0.0;
            let mut row_at_bound = None;
            while i < entries.len() && entries[i].0 == col {
                let (_, row, coefficient) = entries[i];
                if self.rhs[row.to_usize()].is_finite() {
                    let mut activity = 0.0;
                    for entry in self
                        .rows_saver
                        .saved_column(ColIndex::from_usize(row.to_usize()))
                    {
                        let variable = ColIndex::from_usize(entry.index().to_usize());
                        activity = entry
                            .coefficient()
                            .mul_add(solution.primal_values[variable], activity);
                    }
                    let activity = self.rhs[row.to_usize()] - activity;
                    if activity * self.activity_sign_correction[row.to_usize()] < 0.0 {
                        let bound = activity / coefficient;
                        if bound.abs() > shift.abs() {
                            shift = bound;
                            row_at_bound = Some(row);
                        }
                    }
                }
                i += 1;
            }
            solution.primal_values[col] += shift;
            if let Some(row) = row_at_bound {
                solution.variable_statuses[col] = VariableStatus::Basic;
                solution.constraint_statuses[row] =
                    if self.activity_sign_correction[row.to_usize()] == 1.0 {
                        ConstraintStatus::AtUpperBound
                    } else {
                        ConstraintStatus::AtLowerBound
                    };
            }
        }
    }
}

/// Deletes free rows, restoring zero duals and BASIC statuses.
#[derive(Debug, Default)]
pub struct FreeConstraintPreprocessor {
    deleted: RowDeletionHelper,
}

impl FreeConstraintPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self, lp: &mut LinearProgram) -> bool {
        self.deleted = RowDeletionHelper::default();
        for position in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            if lp.constraint_lower_bounds()[row] == f64::NEG_INFINITY
                && lp.constraint_upper_bounds()[row] == f64::INFINITY
            {
                self.deleted.mark(row);
            }
        }
        lp.delete_rows(&self.deleted.deleted);
        !self.deleted.is_empty()
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.deleted.restore(solution);
    }
}

/// Removes empty rows or reports infeasibility when zero violates a row bound.
#[derive(Debug)]
pub struct EmptyConstraintPreprocessor {
    deleted: RowDeletionHelper,
    status: ProblemStatus,
}

impl Default for EmptyConstraintPreprocessor {
    fn default() -> Self {
        Self {
            deleted: RowDeletionHelper::default(),
            status: ProblemStatus::Init,
        }
    }
}

impl EmptyConstraintPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self, lp: &mut LinearProgram, feasibility_tolerance: f64) -> bool {
        self.deleted = RowDeletionHelper::default();
        self.status = ProblemStatus::Init;
        let mut degree = vec![0_usize; lp.num_constraints().to_usize()];
        for position in 0..lp.num_variables().to_usize() {
            for entry in lp.sparse_column(ColIndex::from_usize(position)) {
                degree[entry.index().to_usize()] += 1;
            }
        }
        for (position, &count) in degree.iter().enumerate() {
            if count == 0 {
                let row = RowIndex::from_usize(position);
                if lp.constraint_lower_bounds()[row] > feasibility_tolerance
                    || lp.constraint_upper_bounds()[row] < -feasibility_tolerance
                {
                    self.status = ProblemStatus::PrimalInfeasible;
                    return false;
                }
                self.deleted.mark(row);
            }
        }
        lp.delete_rows(&self.deleted.deleted);
        !self.deleted.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.deleted.restore(solution);
    }
}

fn status_at_value(value: f64, lower: f64, upper: f64) -> VariableStatus {
    if lower == upper {
        VariableStatus::FixedValue
    } else if value == lower {
        VariableStatus::AtLowerBound
    } else if value == upper {
        VariableStatus::AtUpperBound
    } else {
        VariableStatus::Free
    }
}

/// Deletes empty columns using upstream's bound and objective conventions.
#[derive(Debug)]
pub struct EmptyColumnPreprocessor {
    deleted: ColumnDeletionHelper,
    status: ProblemStatus,
}

impl Default for EmptyColumnPreprocessor {
    fn default() -> Self {
        Self {
            deleted: ColumnDeletionHelper::default(),
            status: ProblemStatus::Init,
        }
    }
}

impl EmptyColumnPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self, lp: &mut LinearProgram) -> bool {
        self.deleted = ColumnDeletionHelper::default();
        self.status = ProblemStatus::Init;
        for position in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(position);
            if !lp.sparse_column(col).is_empty() {
                continue;
            }
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            let cost = lp.objective_coefficient_for_minimization(col);
            let value = if cost == 0.0 {
                if upper.is_finite() {
                    upper
                } else if lower.is_finite() {
                    lower
                } else {
                    0.0
                }
            } else {
                let target = if cost > 0.0 { lower } else { upper };
                if !target.is_finite() {
                    self.status = ProblemStatus::InfeasibleOrUnbounded;
                    return false;
                }
                lp.set_objective_offset(
                    target.mul_add(lp.objective_coefficients()[col], lp.objective_offset()),
                );
                target
            };
            self.deleted
                .mark_with_state(col, value, status_at_value(value, lower, upper));
        }
        lp.delete_columns(&self.deleted.deleted);
        !self.deleted.is_empty()
    }

    #[must_use]
    #[inline]
    pub const fn status(&self) -> ProblemStatus {
        self.status
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.deleted.restore(solution);
    }
}

/// Makes every remaining singleton coefficient nonnegative after the other
/// presolve passes, preserving variable bounds and objective signs.
#[derive(Debug, Default)]
pub struct SingletonColumnSignPreprocessor {
    changed_columns: Vec<ColIndex>,
}

impl SingletonColumnSignPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self, lp: &mut LinearProgram) -> bool {
        self.changed_columns.clear();
        if lp.num_variables().to_usize() == 0 {
            return false;
        }
        for position in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(position);
            let column = lp.mutable_sparse_column(col);
            if column.num_entries() == 1 && column.first_coefficient() < 0.0 {
                column.multiply_by_constant(-1.0);
                let lower = lp.variable_lower_bounds()[col];
                let upper = lp.variable_upper_bounds()[col];
                let cost = lp.objective_coefficients()[col];
                lp.set_variable_bounds(col, -upper, -lower);
                lp.set_objective_coefficient(col, -cost);
                self.changed_columns.push(col);
            }
        }
        !self.changed_columns.is_empty()
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        for &col in &self.changed_columns {
            solution.primal_values[col] = -solution.primal_values[col];
            solution.variable_statuses[col] = match solution.variable_statuses[col] {
                VariableStatus::AtUpperBound => VariableStatus::AtLowerBound,
                VariableStatus::AtLowerBound => VariableStatus::AtUpperBound,
                other => other,
            };
        }
    }
}

// The recurrence used by upstream's `KahanSum` (`AccurateSum<double>`).
#[derive(Clone, Copy, Debug, Default)]
struct AccurateSum {
    sum: f64,
    error_sum: f64,
}

impl AccurateSum {
    fn add(&mut self, term: f64) {
        self.error_sum += term;
        let new_sum = self.sum + self.error_sum;
        self.error_sum += self.sum - new_sum;
        self.sum = new_sum;
    }

    const fn value(self) -> f64 {
        self.sum
    }
}

/// Translates each variable whose domain excludes zero toward zero, while
/// compensating row bounds and the objective offset in original column order.
#[derive(Debug, Default)]
pub struct ShiftVariableBoundsPreprocessor {
    offsets: DenseRow,
    variable_initial_lbs: DenseRow,
    variable_initial_ubs: DenseRow,
}

impl ShiftVariableBoundsPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether postsolve is needed, or `InvalidProblem` if a
    /// compensated row/objective shift overflows, as in pinned GLOP.
    ///
    /// # Errors
    ///
    /// Returns `InvalidProblem` on shift overflow.
    pub fn run(&mut self, lp: &mut LinearProgram) -> Result<bool, ProblemStatus> {
        self.variable_initial_lbs = lp.variable_lower_bounds().clone();
        self.variable_initial_ubs = lp.variable_upper_bounds().clone();
        let num_cols = lp.num_variables().to_usize();
        let needs_shift = (0..num_cols).any(|position| {
            let col = ColIndex::from_usize(position);
            0.0 < self.variable_initial_lbs[col] || 0.0 > self.variable_initial_ubs[col]
        });
        if !needs_shift {
            return Ok(false);
        }

        self.offsets = DenseRow::filled(lp.num_variables(), 0.0);
        let mut row_offsets = vec![AccurateSum::default(); lp.num_constraints().to_usize()];
        let mut objective_offset = AccurateSum::default();
        for position in 0..num_cols {
            let col = ColIndex::from_usize(position);
            let lower = self.variable_initial_lbs[col];
            let upper = self.variable_initial_ubs[col];
            if 0.0 < lower || 0.0 > upper {
                // Strict comparison preserves upstream's tie choice (upper).
                let nearest = if lower.abs() < upper.abs() {
                    lower
                } else {
                    upper
                };
                let offset = if nearest.is_finite() { nearest } else { 0.0 };
                self.offsets[col] = offset;
                lp.set_variable_bounds(col, lower - offset, upper - offset);
                for entry in lp.sparse_column(col) {
                    row_offsets[entry.index().to_usize()].add(entry.coefficient() * offset);
                }
                objective_offset.add(lp.objective_coefficients()[col] * offset);
            }
        }
        for (position, offset) in row_offsets.into_iter().enumerate() {
            let shift = offset.value();
            if !shift.is_finite() {
                return Err(ProblemStatus::InvalidProblem);
            }
            let row = RowIndex::from_usize(position);
            lp.set_constraint_bounds(
                row,
                lp.constraint_lower_bounds()[row] - shift,
                lp.constraint_upper_bounds()[row] - shift,
            );
        }
        let shift = objective_offset.value();
        if !shift.is_finite() {
            return Err(ProblemStatus::InvalidProblem);
        }
        lp.set_objective_offset(lp.objective_offset() + shift);
        Ok(true)
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        for position in 0..solution.variable_statuses.len().to_usize() {
            let col = ColIndex::from_usize(position);
            match solution.variable_statuses[col] {
                VariableStatus::FixedValue | VariableStatus::AtLowerBound => {
                    solution.primal_values[col] = self.variable_initial_lbs[col];
                }
                VariableStatus::AtUpperBound => {
                    solution.primal_values[col] = self.variable_initial_ubs[col];
                }
                VariableStatus::Basic => {
                    solution.primal_values[col] += self.offsets[col];
                }
                VariableStatus::Free => {}
            }
        }
    }

    #[must_use]
    pub const fn offsets(&self) -> &DenseRow {
        &self.offsets
    }
}

/// GLOP's matrix/objective/bound scaling and corresponding primal/dual postsolve.
#[derive(Debug, Default)]
pub struct ScalingPreprocessor {
    variable_lower_bounds: DenseRow,
    variable_upper_bounds: DenseRow,
    cost_scaling_factor: f64,
    bound_scaling_factor: f64,
    scaler: SparseMatrixScaler,
}

impl ScalingPreprocessor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether a postsolve step is required. The `LINEAR_PROGRAM`
    /// method solves upstream's logarithmic auxiliary LP, falling back to
    /// equilibration only if the auxiliary simplex reports an error.
    pub fn run(&mut self, lp: &mut LinearProgram, parameters: &GlopParameters) -> bool {
        if !parameters.use_scaling {
            return false;
        }
        self.variable_lower_bounds = lp.variable_lower_bounds().clone();
        self.variable_upper_bounds = lp.variable_upper_bounds().clone();
        parameters.scale_matrix(lp, &mut self.scaler);
        self.cost_scaling_factor = lp.scale_objective(parameters.cost_scaling_algorithm());
        self.bound_scaling_factor = lp.scale_bounds();
        true
    }

    pub fn recover_solution(&self, solution: &mut ProblemSolution) {
        self.scaler
            .scale_row_vector(false, &mut solution.primal_values);
        for value in solution.primal_values.as_mut_slice() {
            *value *= self.bound_scaling_factor;
        }
        self.scaler
            .scale_column_vector(false, &mut solution.dual_values);
        for value in solution.dual_values.as_mut_slice() {
            *value *= self.cost_scaling_factor;
        }
        // As upstream, snap nonbasic variables to their *original* exact
        // bounds after reversing scaling's floating-point multiplications.
        for position in 0..solution.primal_values.len().to_usize() {
            let col = ColIndex::from_usize(position);
            match solution.variable_statuses[col] {
                VariableStatus::AtUpperBound | VariableStatus::FixedValue => {
                    solution.primal_values[col] = self.variable_upper_bounds[col];
                }
                VariableStatus::AtLowerBound => {
                    solution.primal_values[col] = self.variable_lower_bounds[col];
                }
                VariableStatus::Basic | VariableStatus::Free => {}
            }
        }
    }
}

impl LpScalingParameters for GlopParameters {
    fn scale_matrix(&self, lp: &mut LinearProgram, scaler: &mut SparseMatrixScaler) {
        scale_matrix_with_method(lp, scaler, self.scaling_method);
    }

    fn cost_scaling_algorithm(&self) -> ModelCostScaling {
        match self.cost_scaling {
            CostScalingAlgorithm::NoCostScaling => ModelCostScaling::NoCostScaling,
            CostScalingAlgorithm::ContainOneCostScaling => ModelCostScaling::ContainOneCostScaling,
            CostScalingAlgorithm::MeanCostScaling => ModelCostScaling::MeanCostScaling,
            CostScalingAlgorithm::MedianCostScaling => ModelCostScaling::MedianCostScaling,
        }
    }
}

fn scale_matrix_with_method(
    lp: &mut LinearProgram,
    scaler: &mut SparseMatrixScaler,
    method: ScalingAlgorithm,
) {
    scale_matrix_with_method_using(lp, scaler, method, scale_using_auxiliary_lp);
}

fn scale_matrix_with_method_using(
    lp: &mut LinearProgram,
    scaler: &mut SparseMatrixScaler,
    method: ScalingAlgorithm,
    auxiliary_scale: impl FnOnce(&mut LinearProgram, &mut SparseMatrixScaler) -> bool,
) {
    if method != ScalingAlgorithm::LinearProgram || !auxiliary_scale(lp, scaler) {
        // Match SparseMatrixScaler::Scale(): failed LP scaling falls through to
        // the ordinary geometric/equilibration passes.
        lp.scale(scaler);
    }
}

fn scale_using_auxiliary_lp(lp: &mut LinearProgram, scaler: &mut SparseMatrixScaler) -> bool {
    if lp.matrix().min_and_max_magnitudes().0 == 0.0 {
        // Native `Scale()` returns before constructing the auxiliary LP.
        lp.scale(scaler);
        return true;
    }
    // Native `LPScale()` cleans the matrix before iterating its columns.
    lp.clean_up();
    let matrix = lp.matrix();
    let num_rows = matrix.num_rows().to_usize();
    let num_cols = matrix.num_cols().to_usize();

    let mut auxiliary = LinearProgram::new();
    let beta = auxiliary.create_new_variable();
    auxiliary.set_variable_bounds(beta, f64::NEG_INFINITY, f64::INFINITY);
    auxiliary.set_objective_coefficient(beta, 1.0);
    let mut row_variables = vec![None; num_rows];
    let mut column_variables = vec![None; num_cols];
    for (position, slot) in column_variables.iter_mut().enumerate() {
        let column_variable = auxiliary.create_new_variable();
        auxiliary.set_variable_bounds(column_variable, f64::NEG_INFINITY, f64::INFINITY);
        *slot = Some(column_variable);
        for entry in matrix.column(ColIndex::from_usize(position)) {
            let row = entry.index().to_usize();
            let row_variable = *row_variables[row].get_or_insert_with(|| {
                let variable = auxiliary.create_new_variable();
                auxiliary.set_variable_bounds(variable, f64::NEG_INFINITY, f64::INFINITY);
                variable
            });
            let log_magnitude = entry.coefficient().abs().log2();
            let positive = auxiliary.create_new_constraint();
            auxiliary.set_constraint_bounds(positive, -log_magnitude, f64::INFINITY);
            auxiliary.set_coefficient(positive, row_variable, 1.0);
            auxiliary.set_coefficient(positive, column_variable, 1.0);
            auxiliary.set_coefficient(positive, beta, 1.0);
            let negative = auxiliary.create_new_constraint();
            auxiliary.set_constraint_bounds(negative, f64::NEG_INFINITY, -log_magnitude);
            auxiliary.set_coefficient(negative, row_variable, 1.0);
            auxiliary.set_coefficient(negative, column_variable, 1.0);
            auxiliary.set_coefficient(negative, beta, -1.0);
        }
    }
    auxiliary.add_slack_variables_where_necessary(false);
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters::default());
    if simplex
        .solve(&auxiliary, &mut TimeLimit::default())
        .is_err()
    {
        return false;
    }
    let column_factors = column_variables
        .into_iter()
        .map(|variable| 2.0_f64.powf(-simplex.variable_value(variable.unwrap())));
    let mut row_factors = lp_data::lp_types::DenseColumn::filled(matrix.num_rows(), 0.0);
    for (position, variable) in row_variables.into_iter().enumerate() {
        let index = RowIndex::from_usize(position);
        row_factors[index] =
            variable.map_or(1.0, |column| 2.0_f64.powf(-simplex.variable_value(column)));
    }
    lp.scale_from_lp_factors(scaler, column_factors, &row_factors);
    true
}

#[cfg(test)]
mod scaling_dispatch_tests {
    use super::*;
    use lp_data::lp_data::CostScalingAlgorithm as ModelCostScaling;

    #[test]
    fn failed_auxiliary_solve_falls_back_to_equilibration() {
        let mut lp = LinearProgram::new();
        let col = lp.create_new_variable();
        let row = lp.create_new_constraint();
        lp.set_coefficient(row, col, 16.0);
        let mut expected = lp.clone();
        let mut expected_scaler = SparseMatrixScaler::new();
        expected.scale(&mut expected_scaler);

        let mut actual_scaler = SparseMatrixScaler::new();
        let mut attempted = false;
        scale_matrix_with_method_using(
            &mut lp,
            &mut actual_scaler,
            ScalingAlgorithm::LinearProgram,
            |_, _| {
                attempted = true;
                false
            },
        );
        assert!(attempted);
        assert_eq!(lp.matrix().column(col), expected.matrix().column(col));
        assert_eq!(actual_scaler.row_scales(), expected_scaler.row_scales());
        assert_eq!(actual_scaler.col_scales(), expected_scaler.col_scales());
    }

    // Retain the native adapter's printed decimals without rounding them for lint.
    #[allow(clippy::unreadable_literal, clippy::excessive_precision)]
    #[test]
    fn injected_failure_matches_native_equilibration_whole_model() {
        // Native `LpScalingHelper::Scale()` with EQUILIBRATION on this input
        // produces the values below. Its LINEAR_PROGRAM scaling is distinct,
        // so this fixture checks the fallback rather than a coincidental tie.
        let mut lp = LinearProgram::new();
        let x = lp.create_new_variable();
        let y = lp.create_new_variable();
        let first = lp.create_new_constraint();
        let second = lp.create_new_constraint();
        lp.set_coefficient(first, x, 1000.0);
        lp.set_coefficient(second, x, 0.1);
        lp.set_coefficient(first, y, 0.2);
        lp.set_coefficient(second, y, 10.0);
        lp.set_variable_bounds(x, 0.0, 10.0);
        lp.set_variable_bounds(y, -1.0, 5.0);
        lp.set_objective_coefficient(x, 1.0);
        lp.set_objective_coefficient(y, 2.0);
        lp.set_constraint_bounds(first, -2.0, 8.0);
        lp.set_constraint_bounds(second, 0.0, 3.0);
        lp.clean_up();

        let mut scaler = SparseMatrixScaler::new();
        scale_matrix_with_method_using(
            &mut lp,
            &mut scaler,
            ScalingAlgorithm::LinearProgram,
            |_, _| false,
        );
        assert_eq!(lp.scale_objective(ModelCostScaling::NoCostScaling), 1.0);
        assert_eq!(lp.scale_bounds(), 1.0);
        let check = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 4e-15 * expected.abs().max(1.0),
                "{actual} differs from native {expected}"
            );
        };
        for (actual, expected) in lp
            .objective_coefficients()
            .iter()
            .zip([0.37606030930863937, 5.3182958969449894])
        {
            check(*actual, expected);
        }
        for (actual, expected) in lp
            .variable_lower_bounds()
            .iter()
            .zip([0.0, -0.37606030930863932])
        {
            check(*actual, expected);
        }
        for (actual, expected) in lp
            .variable_upper_bounds()
            .iter()
            .zip([26.591479484724942, 1.8803015465431967])
        {
            check(*actual, expected);
        }
        for (actual, expected) in lp
            .constraint_lower_bounds()
            .iter()
            .zip([-0.0053182958969449877, 0.0])
        {
            check(*actual, expected);
        }
        for (actual, expected) in lp
            .constraint_upper_bounds()
            .iter()
            .zip([0.021273183587779951, 0.11281809279259179])
        {
            check(*actual, expected);
        }
        let expected = [
            [(0, 1.0), (1, 0.001414213562373095)],
            [(0, 0.001414213562373095), (1, 1.0)],
        ];
        for (column, entries) in expected.into_iter().enumerate() {
            for (entry, (row, value)) in lp
                .sparse_column(ColIndex::from_usize(column))
                .into_iter()
                .zip(entries)
            {
                assert_eq!(entry.index().to_usize(), row);
                check(entry.coefficient(), value);
            }
        }
    }
}
