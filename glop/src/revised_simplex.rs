//! Revised-simplex driver.
//!
//! This is the Rust counterpart of `ortools/glop/revised_simplex.{h,cc}` at
//! the commit pinned in `UPSTREAM.md`.  The implementation deliberately keeps
//! GLOP's equation-form representation, basis mapping, phase-I objective,
//! Harris ratio test, bound flips, and product-form basis updates visible in
//! the same layer as upstream.

#![allow(
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]

use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{
    ColIndex, ConstraintStatus, DenseColumn, DenseRow, INVALID_COL, ProblemStatus, RowIndex,
    RowToColMapping, VariableStatus, VectorIndex,
};
use lp_data::scattered_vector::{ScatteredColumn, ScatteredRow};
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};

use crate::basis_representation::BasisRepresentation;
use crate::initial_basis::InitialBasis;
use crate::lu_factorization::FactorizationError;
use crate::parameters::{GlopParameters, InitialBasisHeuristic};
use crate::primal_ratio_test::{LeavingChoice, choose_leaving_variable_row};
use crate::time_limit::TimeLimit;
use crate::variables_info::{BasisState, VariablesInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimplexPhase {
    Feasibility,
    Optimization,
}

/// A normalized event emitted only when tracing is enabled.
#[derive(Clone, Debug, PartialEq)]
pub struct IterationEvent {
    pub iteration: u64,
    pub phase: SimplexPhase,
    pub entering_column: Option<ColIndex>,
    pub leaving_row: Option<RowIndex>,
    pub step: f64,
    pub objective: f64,
}

#[derive(Debug)]
pub struct RevisedSimplex {
    parameters: GlopParameters,
    problem_status: ProblemStatus,
    num_rows: RowIndex,
    num_cols: ColIndex,
    first_slack_col: ColIndex,
    matrix: SparseMatrix,
    compact_matrix: CompactSparseMatrix,
    objective: DenseRow,
    objective_offset: f64,
    objective_scaling_factor: f64,
    basis: RowToColMapping,
    basis_factorization: Option<BasisRepresentation>,
    variables_info: Option<VariablesInfo>,
    variable_values: DenseRow,
    reduced_costs: DenseRow,
    dual_values: DenseColumn,
    primal_ray: DenseRow,
    dual_ray: DenseColumn,
    solution_state: BasisState,
    state_for_next_solve: Option<BasisState>,
    starting_values: DenseRow,
    num_iterations: u64,
    trace_enabled: bool,
    trace: Vec<IterationEvent>,
}

impl Default for RevisedSimplex {
    fn default() -> Self {
        Self::new()
    }
}

impl RevisedSimplex {
    #[must_use]
    pub fn new() -> Self {
        Self {
            parameters: GlopParameters::default(),
            problem_status: ProblemStatus::Init,
            num_rows: RowIndex::new(0),
            num_cols: ColIndex::new(0),
            first_slack_col: ColIndex::new(0),
            matrix: SparseMatrix::new(),
            compact_matrix: CompactSparseMatrix::default(),
            objective: DenseRow::new(),
            objective_offset: 0.0,
            objective_scaling_factor: 1.0,
            basis: RowToColMapping::new(),
            basis_factorization: None,
            variables_info: None,
            variable_values: DenseRow::new(),
            reduced_costs: DenseRow::new(),
            dual_values: DenseColumn::new(),
            primal_ray: DenseRow::new(),
            dual_ray: DenseColumn::new(),
            solution_state: BasisState::default(),
            state_for_next_solve: None,
            starting_values: DenseRow::new(),
            num_iterations: 0,
            trace_enabled: false,
            trace: Vec::new(),
        }
    }

    pub fn set_parameters(&mut self, parameters: &GlopParameters) {
        self.parameters = parameters.clone();
    }

    #[must_use]
    pub const fn parameters(&self) -> &GlopParameters {
        &self.parameters
    }

    pub fn set_trace_enabled(&mut self, enabled: bool) {
        self.trace_enabled = enabled;
        if !enabled {
            self.trace.clear();
        }
    }

    #[must_use]
    pub fn trace(&self) -> &[IterationEvent] {
        &self.trace
    }

    pub fn clear_state_for_next_solve(&mut self) {
        self.state_for_next_solve = Some(BasisState::default());
    }

    pub fn load_state_for_next_solve(&mut self, state: &BasisState) {
        self.state_for_next_solve = Some(state.clone());
    }

    pub fn set_starting_variable_values_for_next_solve(&mut self, values: &DenseRow) {
        self.starting_values.clone_from(values);
    }

    pub fn solve(
        &mut self,
        linear_program: &LinearProgram,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        self.initialize(linear_program)?;
        self.trace.clear();
        self.num_iterations = 0;

        self.run_primal_phase(SimplexPhase::Feasibility, time_limit)?;
        if self.problem_status == ProblemStatus::PrimalFeasible && !time_limit.limit_reached() {
            self.run_primal_phase(SimplexPhase::Optimization, time_limit)?;
        }
        self.finish_solution()?;
        self.starting_values.clear();
        Ok(())
    }

    fn initialize(&mut self, lp: &LinearProgram) -> Result<(), FactorizationError> {
        let mut equation_lp = LinearProgram::default();
        equation_lp.populate_from_linear_program(lp);
        if !equation_lp.is_in_equation_form() {
            equation_lp.add_slack_variables_where_necessary(false);
        }
        self.num_rows = equation_lp.num_constraints();
        self.num_cols = equation_lp.num_variables();
        self.first_slack_col = equation_lp
            .first_slack_variable()
            .unwrap_or(equation_lp.num_variables());
        self.matrix = equation_lp.matrix().clone();
        self.compact_matrix = CompactSparseMatrix::from_sparse(&self.matrix);
        self.objective = DenseRow::from_vec(
            (0..self.num_cols.to_usize())
                .map(|column| {
                    equation_lp.objective_coefficient_for_minimization(ColIndex::from_usize(column))
                })
                .collect(),
        );
        self.objective_offset = equation_lp.objective_offset();
        self.objective_scaling_factor = equation_lp.objective_scaling_factor();

        let mut info = VariablesInfo::new(&self.matrix);
        info.load_bounds_and_return_true_if_unchanged(
            equation_lp.variable_lower_bounds().as_slice(),
            equation_lp.variable_upper_bounds().as_slice(),
        );
        let has_external_basis = self
            .state_for_next_solve
            .as_ref()
            .is_some_and(|state| !state.is_empty());
        if let Some(state) = self
            .state_for_next_solve
            .take()
            .filter(|state| !state.is_empty())
        {
            info.initialize_from_basis_state(self.first_slack_col.to_usize(), 0, &state);
        } else {
            info.initialize_to_default_status();
        }

        let mut basis = vec![ColIndex::new(0); self.num_rows.to_usize()];
        let mut next_row = 0;
        for column in 0..self.num_cols.to_usize() {
            let column_index = ColIndex::from_usize(column);
            if info.variable_statuses()[column_index] == VariableStatus::Basic
                && next_row < basis.len()
            {
                basis[next_row] = column_index;
                next_row += 1;
            }
        }
        for (row, basis_column) in basis.iter_mut().enumerate().skip(next_row) {
            let slack = ColIndex::new(self.first_slack_col.value() + i32::try_from(row).unwrap());
            *basis_column = slack;
        }
        if !has_external_basis && self.parameters.initial_basis != InitialBasisHeuristic::None {
            let mut crashed = RowToColMapping::from_vec(basis.clone());
            if matches!(
                self.parameters.initial_basis,
                InitialBasisHeuristic::Bixby | InitialBasisHeuristic::Triangular
            ) {
                for row in 0..crashed.len().to_usize() {
                    let row = RowIndex::from_usize(row);
                    let column = crashed[row];
                    if info.lower_bounds()[column.to_usize()]
                        == info.upper_bounds()[column.to_usize()]
                    {
                        crashed[row] = INVALID_COL;
                    }
                }
            }
            let crash_lower = DenseRow::from_vec(info.lower_bounds().to_vec());
            let crash_upper = DenseRow::from_vec(info.upper_bounds().to_vec());
            let mut crash = InitialBasis::new(
                &self.compact_matrix,
                &self.objective,
                &crash_lower,
                &crash_upper,
                info.variable_types(),
            );
            match self.parameters.initial_basis {
                InitialBasisHeuristic::Bixby if self.parameters.use_scaling => {
                    crash.complete_bixby_basis(self.first_slack_col, &mut crashed);
                }
                InitialBasisHeuristic::None | InitialBasisHeuristic::Bixby => {}
                InitialBasisHeuristic::Triangular if self.parameters.use_dual_simplex => {
                    crash.complete_triangular_dual_basis(self.num_cols, &mut crashed);
                }
                InitialBasisHeuristic::Triangular => {
                    crash.complete_triangular_primal_basis(self.num_cols, &mut crashed);
                }
                InitialBasisHeuristic::Maros if self.parameters.use_dual_simplex => {
                    crash.get_dual_maros_basis(self.num_cols, &mut crashed);
                }
                InitialBasisHeuristic::Maros => {
                    crash.get_primal_maros_basis(self.num_cols, &mut crashed);
                }
            }
            for row in 0..crashed.len().to_usize() {
                let row = RowIndex::from_usize(row);
                if crashed[row] == INVALID_COL {
                    crashed[row] = ColIndex::new(self.first_slack_col.value() + row.value());
                }
            }
            basis = crashed.as_slice().to_vec();
        }
        self.basis = RowToColMapping::from_vec(basis);

        let mut basis_matrix = SparseMatrix::new();
        basis_matrix.populate_from_zero(
            self.num_rows,
            ColIndex::from_usize(self.num_rows.to_usize()),
        );
        for row in 0..self.num_rows.to_usize() {
            *basis_matrix.mutable_column(ColIndex::from_usize(row)) = self
                .matrix
                .column(self.basis[RowIndex::from_usize(row)])
                .clone();
        }
        let mut basis_factorization =
            BasisRepresentation::new_with_parameters(basis_matrix, &self.parameters)?;
        if basis_factorization.infinity_norm_condition_number_upper_bound()
            > self.parameters.initial_condition_number_threshold
        {
            self.basis = RowToColMapping::from_vec(
                (0..self.num_rows.to_usize())
                    .map(|row| {
                        ColIndex::new(self.first_slack_col.value() + i32::try_from(row).unwrap())
                    })
                    .collect(),
            );
            let mut slack_basis = SparseMatrix::new();
            slack_basis.populate_from_zero(
                self.num_rows,
                ColIndex::from_usize(self.num_rows.to_usize()),
            );
            for row in 0..self.num_rows.to_usize() {
                *slack_basis.mutable_column(ColIndex::from_usize(row)) = self
                    .matrix
                    .column(self.basis[RowIndex::from_usize(row)])
                    .clone();
            }
            basis_factorization =
                BasisRepresentation::new_with_parameters(slack_basis, &self.parameters)?;
        }
        if !basis_factorization.column_permutation().is_empty() {
            let mut permuted = self.basis.clone();
            for (source, &destination) in
                basis_factorization.column_permutation().iter().enumerate()
            {
                permuted[RowIndex::from_usize(destination)] =
                    self.basis[RowIndex::from_usize(source)];
            }
            self.basis = permuted;
            basis_factorization.set_column_permutation_to_identity();
        }
        info.change_unused_basic_variables_to_free(&self.basis);
        self.basis_factorization = Some(basis_factorization);
        self.variables_info = Some(info);
        self.variable_values = DenseRow::filled(self.num_cols, 0.0);
        self.initialize_values()?;
        self.reduced_costs = DenseRow::filled(self.num_cols, 0.0);
        self.dual_values = DenseColumn::filled(self.num_rows, 0.0);
        self.primal_ray = DenseRow::new();
        self.dual_ray = DenseColumn::new();
        self.problem_status = ProblemStatus::Init;
        Ok(())
    }

    fn initialize_values(&mut self) -> Result<(), FactorizationError> {
        let info = self.variables_info.as_ref().unwrap();
        let mut rhs = vec![0.0; self.num_rows.to_usize()];
        for column in 0..self.num_cols.to_usize() {
            let index = ColIndex::from_usize(column);
            if info.is_basic().contains(index) {
                continue;
            }
            let value = match info.variable_statuses()[index] {
                VariableStatus::AtLowerBound | VariableStatus::FixedValue => {
                    info.lower_bounds()[column]
                }
                VariableStatus::AtUpperBound => info.upper_bounds()[column],
                VariableStatus::Free => self
                    .starting_values
                    .as_slice()
                    .get(column)
                    .copied()
                    .unwrap_or(0.0),
                VariableStatus::Basic => unreachable!(),
            };
            self.variable_values[index] = value;
            for entry in self.matrix.column(index) {
                rhs[entry.index().to_usize()] -= entry.coefficient() * value;
            }
        }
        let basic = self.basis_factorization.as_ref().unwrap().solve(&rhs)?;
        for (row, &value) in basic.iter().enumerate() {
            self.variable_values[self.basis[RowIndex::from_usize(row)]] = value;
        }
        Ok(())
    }

    fn phase_objective(&self, phase: SimplexPhase) -> DenseRow {
        if phase == SimplexPhase::Optimization {
            return self.objective.clone();
        }
        let info = self.variables_info.as_ref().unwrap();
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let mut objective = DenseRow::filled(self.num_cols, 0.0);
        for row in 0..self.num_rows.to_usize() {
            let column = self.basis[RowIndex::from_usize(row)];
            let value = self.variable_values[column];
            objective[column] = if value - info.upper_bounds()[column.to_usize()] > tolerance {
                1.0
            } else if info.lower_bounds()[column.to_usize()] - value > tolerance {
                -1.0
            } else {
                0.0
            };
        }
        objective
    }

    fn compute_reduced_costs(&mut self, objective: &DenseRow) -> Result<(), FactorizationError> {
        let basic_objective: Vec<_> = (0..self.num_rows.to_usize())
            .map(|row| objective[self.basis[RowIndex::from_usize(row)]])
            .collect();
        let dual = self
            .basis_factorization
            .as_ref()
            .unwrap()
            .transpose_solve(&basic_objective)?;
        self.dual_values = DenseColumn::from_vec(dual.clone());
        for column in 0..self.num_cols.to_usize() {
            let index = ColIndex::from_usize(column);
            let mut value = objective[index];
            for entry in self.matrix.column(index) {
                value -= dual[entry.index().to_usize()] * entry.coefficient();
            }
            self.reduced_costs[index] = value;
        }
        Ok(())
    }

    fn choose_entering(&self) -> Option<ColIndex> {
        let info = self.variables_info.as_ref().unwrap();
        let tolerance = self.parameters.dual_feasibility_tolerance;
        let mut best = None;
        let mut best_price = tolerance;
        for column in info.relevance().iter_ones() {
            let reduced = self.reduced_costs[column];
            let price = if reduced < -tolerance && info.can_increase().contains(column) {
                -reduced
            } else if reduced > tolerance && info.can_decrease().contains(column) {
                reduced
            } else {
                continue;
            };
            if price > best_price {
                best_price = price;
                best = Some(column);
            }
        }
        best
    }

    fn direction(&self, entering: ColIndex) -> Result<ScatteredColumn, FactorizationError> {
        let mut direction = ScatteredColumn::new(self.num_rows);
        self.basis_factorization
            .as_ref()
            .unwrap()
            .right_solve_for_problem_column(
                entering.to_usize(),
                self.matrix.column(entering),
                &mut direction,
            )?;
        // An empty position list is GLOP's dense-vector sentinel.  The ratio
        // kernels consume an explicit sparse traversal, so materialize its
        // support here when the selected basis-update representation took the
        // dense solve path.
        if direction.non_zeros().is_empty() {
            let positions: Vec<_> = direction
                .values()
                .as_slice()
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .map(|(row, _)| RowIndex::from_usize(row))
                .collect();
            direction.non_zeros_mut().extend(positions);
        }
        #[cfg(debug_assertions)]
        {
            let mut residual = vec![0.0; self.num_rows.to_usize()];
            for row in 0..self.num_rows.to_usize() {
                let coefficient = direction.value(RowIndex::from_usize(row));
                for entry in self.matrix.column(self.basis[RowIndex::from_usize(row)]) {
                    residual[entry.index().to_usize()] += entry.coefficient() * coefficient;
                }
            }
            for entry in self.matrix.column(entering) {
                residual[entry.index().to_usize()] -= entry.coefficient();
            }
            let maximum = residual
                .into_iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            debug_assert!(
                maximum < 1e-8,
                "iteration {}, direction residual {maximum}",
                self.num_iterations
            );
        }
        Ok(direction)
    }

    fn run_primal_phase(
        &mut self,
        phase: SimplexPhase,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        loop {
            if time_limit.limit_reached()
                || self.num_iterations
                    >= u64::try_from(self.parameters.max_number_of_iterations).unwrap_or(u64::MAX)
            {
                self.problem_status = if phase == SimplexPhase::Feasibility {
                    ProblemStatus::Init
                } else {
                    ProblemStatus::PrimalFeasible
                };
                return Ok(());
            }
            let phase_objective = self.phase_objective(phase);
            self.compute_reduced_costs(&phase_objective)?;
            let Some(entering) = self.choose_entering() else {
                if phase == SimplexPhase::Feasibility {
                    let infeasibility = self.maximum_primal_infeasibility();
                    self.problem_status =
                        if infeasibility < self.parameters.primal_feasibility_tolerance {
                            ProblemStatus::PrimalFeasible
                        } else {
                            ProblemStatus::PrimalInfeasible
                        };
                } else {
                    self.problem_status = ProblemStatus::Optimal;
                }
                return Ok(());
            };
            let direction = self.direction(entering)?;
            let direction_norm = direction
                .values()
                .as_slice()
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            let reduced = self.reduced_costs[entering];
            let choice = if phase == SimplexPhase::Feasibility {
                self.phase_one_ratio_test(entering, reduced, &direction, direction_norm)
            } else {
                let info = self.variables_info.as_ref().unwrap();
                choose_leaving_variable_row(
                    entering,
                    reduced,
                    &direction,
                    direction_norm,
                    self.variable_values.as_slice(),
                    info.lower_bounds(),
                    info.upper_bounds(),
                    &self.basis,
                    self.basis_factorization.as_ref().unwrap().is_refactorized(),
                    &self.parameters,
                )
            };
            if choice == LeavingChoice::Refactorize {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                continue;
            }
            let (leaving_row, step_length, target_bound) = match choice {
                LeavingChoice::BoundFlip { step } => (None, step, 0.0),
                LeavingChoice::Pivot {
                    row,
                    step,
                    target_bound,
                } => (Some(row), step, target_bound),
                LeavingChoice::Refactorize => unreachable!(),
            };
            if step_length.is_infinite() {
                self.problem_status = if phase == SimplexPhase::Feasibility {
                    ProblemStatus::Abnormal
                } else {
                    ProblemStatus::PrimalUnbounded
                };
                if phase == SimplexPhase::Optimization {
                    self.primal_ray = DenseRow::filled(self.num_cols, 0.0);
                    let sign = if reduced > 0.0 { -1.0 } else { 1.0 };
                    self.primal_ray[entering] = sign;
                    for row in 0..self.num_rows.to_usize() {
                        self.primal_ray[self.basis[RowIndex::from_usize(row)]] =
                            -sign * direction.value(RowIndex::from_usize(row));
                    }
                }
                return Ok(());
            }
            let mut step = if reduced > 0.0 {
                -step_length
            } else {
                step_length
            };
            if phase == SimplexPhase::Feasibility
                && let Some(row) = leaving_row
            {
                let leaving = self.basis[row];
                step = (self.variable_values[leaving] - target_bound) / direction.value(row);
            }
            for entry in &direction {
                self.variable_values[self.basis[entry.row()]] -= entry.coefficient() * step;
            }
            self.variable_values[entering] += step;

            if let Some(row) = leaving_row {
                let leaving = self.basis[row];
                self.variable_values[leaving] = target_bound;
                let leaving_status = self.status_at_bound(leaving, target_bound);
                {
                    let info = self.variables_info.as_mut().unwrap();
                    info.update_to_nonbasic_status(leaving, leaving_status);
                    info.update_to_basic_status(entering);
                }
                self.basis[row] = entering;
                let mut unit_left_inverse =
                    ScatteredRow::new(ColIndex::from_usize(self.num_rows.to_usize()));
                self.basis_factorization
                    .as_ref()
                    .unwrap()
                    .left_solve_for_unit_row(row.to_usize(), &mut unit_left_inverse)?;
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .replace_column_after_solve(
                        entering.to_usize(),
                        row.to_usize(),
                        &direction,
                        self.matrix.column(entering).clone(),
                    )?;
                self.incorporate_basis_permutation();
            } else {
                let info = self.variables_info.as_mut().unwrap();
                if step > 0.0 {
                    info.update_to_nonbasic_status(entering, VariableStatus::AtUpperBound);
                    self.variable_values[entering] = info.upper_bounds()[entering.to_usize()];
                } else {
                    info.update_to_nonbasic_status(entering, VariableStatus::AtLowerBound);
                    self.variable_values[entering] = info.lower_bounds()[entering.to_usize()];
                }
            }
            self.num_iterations += 1;
            if self.trace_enabled {
                self.trace.push(IterationEvent {
                    iteration: self.num_iterations,
                    phase,
                    entering_column: Some(entering),
                    leaving_row,
                    step,
                    objective: self.internal_objective(),
                });
            }
        }
    }

    fn phase_one_ratio_test(
        &self,
        entering: ColIndex,
        reduced_cost: f64,
        direction: &ScatteredColumn,
        direction_norm: f64,
    ) -> LeavingChoice {
        #[derive(Clone, Copy)]
        struct BreakPoint {
            row: RowIndex,
            ratio: f64,
            magnitude: f64,
            target: f64,
        }
        let info = self.variables_info.as_ref().unwrap();
        let entering_value = self.variable_values[entering];
        let mut current_ratio = if reduced_cost > 0.0 {
            entering_value - info.lower_bounds()[entering.to_usize()]
        } else {
            info.upper_bounds()[entering.to_usize()] - entering_value
        };
        let mut points = Vec::new();
        let tolerance = self.parameters.primal_feasibility_tolerance;
        for entry in direction {
            let signed_direction = if reduced_cost > 0.0 {
                entry.coefficient()
            } else {
                -entry.coefficient()
            };
            let magnitude = signed_direction.abs();
            if magnitude < tolerance {
                continue;
            }
            let column = self.basis[entry.row()];
            let value = self.variable_values[column];
            let lower = info.lower_bounds()[column.to_usize()];
            let upper = info.upper_bounds()[column.to_usize()];
            let to_lower = (lower - tolerance - value) / signed_direction;
            let to_upper = (upper + tolerance - value) / signed_direction;
            if to_lower >= 0.0 && to_lower < current_ratio {
                points.push(BreakPoint {
                    row: entry.row(),
                    ratio: to_lower,
                    magnitude,
                    target: lower,
                });
            }
            if to_upper >= 0.0 && to_upper < current_ratio {
                points.push(BreakPoint {
                    row: entry.row(),
                    ratio: to_upper,
                    magnitude,
                    target: upper,
                });
            }
        }
        points.sort_by(|left, right| {
            left.ratio
                .total_cmp(&right.ratio)
                .then_with(|| right.magnitude.total_cmp(&left.magnitude))
                .then_with(|| left.row.value().cmp(&right.row.value()))
        });
        let mut improvement = reduced_cost.abs();
        let mut best = None;
        let mut best_magnitude = 0.0;
        for point in points {
            if point.magnitude > best_magnitude {
                current_ratio = point.ratio;
                best_magnitude = point.magnitude;
                best = Some(point);
            }
            improvement -= point.magnitude;
            if improvement <= 0.0 {
                break;
            }
        }
        if let Some(point) = best {
            if best_magnitude < self.parameters.small_pivot_threshold * direction_norm
                && !self.basis_factorization.as_ref().unwrap().is_refactorized()
            {
                LeavingChoice::Refactorize
            } else {
                LeavingChoice::Pivot {
                    row: point.row,
                    step: current_ratio,
                    target_bound: point.target,
                }
            }
        } else {
            LeavingChoice::BoundFlip {
                step: current_ratio,
            }
        }
    }

    fn status_at_bound(&self, column: ColIndex, target: f64) -> VariableStatus {
        let info = self.variables_info.as_ref().unwrap();
        let index = column.to_usize();
        if info.lower_bounds()[index] == info.upper_bounds()[index] {
            VariableStatus::FixedValue
        } else if target == info.lower_bounds()[index] {
            VariableStatus::AtLowerBound
        } else {
            VariableStatus::AtUpperBound
        }
    }

    fn incorporate_basis_permutation(&mut self) {
        let factorization = self.basis_factorization.as_mut().unwrap();
        if factorization.column_permutation().is_empty() {
            return;
        }
        let mut permuted = self.basis.clone();
        for (source, &destination) in factorization.column_permutation().iter().enumerate() {
            permuted[RowIndex::from_usize(destination)] = self.basis[RowIndex::from_usize(source)];
        }
        self.basis = permuted;
        factorization.set_column_permutation_to_identity();
    }

    fn maximum_primal_infeasibility(&self) -> f64 {
        let info = self.variables_info.as_ref().unwrap();
        (0..self.num_cols.to_usize()).fold(0.0_f64, |maximum, column| {
            maximum.max(
                (info.lower_bounds()[column] - self.variable_values.as_slice()[column])
                    .max(self.variable_values.as_slice()[column] - info.upper_bounds()[column])
                    .max(0.0),
            )
        })
    }

    fn internal_objective(&self) -> f64 {
        (0..self.num_cols.to_usize())
            .map(|column| {
                self.objective[ColIndex::from_usize(column)]
                    * self.variable_values.as_slice()[column]
            })
            .sum()
    }

    fn finish_solution(&mut self) -> Result<(), FactorizationError> {
        let objective = self.objective.clone();
        self.compute_reduced_costs(&objective)?;
        self.solution_state.statuses = self
            .variables_info
            .as_ref()
            .unwrap()
            .variable_statuses()
            .clone();
        Ok(())
    }

    #[must_use]
    pub const fn problem_status(&self) -> ProblemStatus {
        self.problem_status
    }
    #[must_use]
    pub const fn problem_num_rows(&self) -> RowIndex {
        self.num_rows
    }
    #[must_use]
    pub const fn problem_num_cols(&self) -> ColIndex {
        self.first_slack_col
    }
    #[must_use]
    pub const fn number_of_iterations(&self) -> u64 {
        self.num_iterations
    }
    #[must_use]
    pub fn objective_value(&self) -> f64 {
        self.objective_scaling_factor * (self.internal_objective() + self.objective_offset)
    }
    #[must_use]
    pub fn variable_value(&self, column: ColIndex) -> f64 {
        self.variable_values[column]
    }
    #[must_use]
    pub fn reduced_cost(&self, column: ColIndex) -> f64 {
        self.reduced_costs[column]
    }
    #[must_use]
    pub const fn reduced_costs(&self) -> &DenseRow {
        &self.reduced_costs
    }
    #[must_use]
    pub fn dual_value(&self, row: RowIndex) -> f64 {
        self.dual_values[row]
    }
    #[must_use]
    pub fn constraint_activity(&self, row: RowIndex) -> f64 {
        -self.variable_values[ColIndex::new(self.first_slack_col.value() + row.value())]
    }
    #[must_use]
    pub fn variable_status(&self, column: ColIndex) -> VariableStatus {
        self.variables_info.as_ref().unwrap().variable_statuses()[column]
    }
    #[must_use]
    pub fn constraint_status(&self, row: RowIndex) -> ConstraintStatus {
        match self.variable_status(ColIndex::new(self.first_slack_col.value() + row.value())) {
            VariableStatus::AtLowerBound => ConstraintStatus::AtUpperBound,
            VariableStatus::AtUpperBound => ConstraintStatus::AtLowerBound,
            VariableStatus::Basic => ConstraintStatus::Basic,
            VariableStatus::FixedValue => ConstraintStatus::FixedValue,
            VariableStatus::Free => ConstraintStatus::Free,
        }
    }
    #[must_use]
    pub const fn state(&self) -> &BasisState {
        &self.solution_state
    }
    #[must_use]
    pub const fn primal_ray(&self) -> &DenseRow {
        &self.primal_ray
    }
    #[must_use]
    pub const fn dual_ray(&self) -> &DenseColumn {
        &self.dual_ray
    }
    #[must_use]
    pub fn basis(&self, row: RowIndex) -> ColIndex {
        self.basis[row]
    }
    #[must_use]
    pub fn maximum_equation_residual(&self) -> f64 {
        let mut residual = vec![0.0_f64; self.num_rows.to_usize()];
        for column in 0..self.num_cols.to_usize() {
            let column = ColIndex::from_usize(column);
            for entry in self.matrix.column(column) {
                residual[entry.index().to_usize()] +=
                    entry.coefficient() * self.variable_values[column];
            }
        }
        residual
            .into_iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
    }
}
