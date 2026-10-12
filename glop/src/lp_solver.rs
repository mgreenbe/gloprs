//! Public LP solver orchestration and solution validation.
//!
//! This follows `ortools/glop/lp_solver.{h,cc}`. The Phase-5 presolve stack
//! includes the ordered presolve stack. The recovered solution is checked against the caller's
//! original model, not the transformed copy used by [`RevisedSimplex`].

#![allow(clippy::float_cmp, clippy::missing_errors_doc)]

use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{
    ColIndex, ConstraintStatus, ConstraintStatusColumn, DenseColumn, DenseRow, ProblemStatus,
    RowIndex, VariableStatus, VariableStatusRow, VectorIndex,
};
use lp_data::lp_utils::{accurate_sum, precise_scalar_product};

use crate::parameters::GlopParameters;
use crate::preprocessor::MainLpPreprocessor;
use crate::revised_simplex::RevisedSimplex;
use crate::time_limit::TimeLimit;
use crate::variables_info::BasisState;

#[derive(Debug)]
pub struct LPSolver {
    parameters: GlopParameters,
    revised_simplex: RevisedSimplex,
    current_linear_program: LinearProgram,
    num_revised_simplex_iterations: u64,
    primal_values: DenseRow,
    dual_values: DenseColumn,
    variable_statuses: VariableStatusRow,
    constraint_statuses: ConstraintStatusColumn,
    reduced_costs: DenseRow,
    constraint_activities: DenseColumn,
    primal_ray: DenseRow,
    constraints_dual_ray: DenseColumn,
    variable_bounds_dual_ray: DenseRow,
    problem_objective_value: f64,
    max_absolute_primal_infeasibility: f64,
    max_absolute_dual_infeasibility: f64,
    may_have_multiple_solutions: bool,
    last_error: Option<String>,
}

/// Borrowed, allocation-free snapshot of the public solve result.
#[derive(Debug)]
pub struct SolveResult<'a> {
    pub status: ProblemStatus,
    pub objective_value: f64,
    pub simplex_iterations: u64,
    pub maximum_primal_infeasibility: f64,
    pub maximum_dual_infeasibility: f64,
    pub deterministic_time: f64,
    pub may_have_multiple_optimal_solutions: bool,
    pub variable_values: &'a DenseRow,
    pub reduced_costs: &'a DenseRow,
    pub variable_statuses: &'a VariableStatusRow,
    pub dual_values: &'a DenseColumn,
    pub constraint_activities: &'a DenseColumn,
    pub constraint_statuses: &'a ConstraintStatusColumn,
    pub primal_ray: &'a DenseRow,
    pub constraints_dual_ray: &'a DenseColumn,
    pub variable_bounds_dual_ray: &'a DenseRow,
}

/// Public basis statuses in the model's variable and constraint coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SolverBasis {
    pub variable_statuses: VariableStatusRow,
    pub constraint_statuses: ConstraintStatusColumn,
}

impl Default for LPSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LPSolver {
    #[must_use]
    pub fn new() -> Self {
        Self {
            parameters: GlopParameters::default(),
            revised_simplex: RevisedSimplex::new(),
            current_linear_program: LinearProgram::new(),
            num_revised_simplex_iterations: 0,
            primal_values: DenseRow::new(),
            dual_values: DenseColumn::new(),
            variable_statuses: VariableStatusRow::new(),
            constraint_statuses: ConstraintStatusColumn::new(),
            reduced_costs: DenseRow::new(),
            constraint_activities: DenseColumn::new(),
            primal_ray: DenseRow::new(),
            constraints_dual_ray: DenseColumn::new(),
            variable_bounds_dual_ray: DenseRow::new(),
            problem_objective_value: 0.0,
            max_absolute_primal_infeasibility: 0.0,
            max_absolute_dual_infeasibility: 0.0,
            may_have_multiple_solutions: false,
            last_error: None,
        }
    }

    pub fn set_parameters(&mut self, parameters: &GlopParameters) {
        self.parameters = parameters.clone();
    }

    #[must_use]
    #[inline]
    pub const fn parameters(&self) -> &GlopParameters {
        &self.parameters
    }

    #[inline]
    pub fn parameters_mut(&mut self) -> &mut GlopParameters {
        &mut self.parameters
    }

    /// Diagnostic for an unsupported Phase-5 option or a pre-solve failure.
    #[must_use]
    #[inline]
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn solve(&mut self, lp: &LinearProgram) -> ProblemStatus {
        let mut time_limit = TimeLimit::from_parameters(&self.parameters);
        self.solve_with_time_limit(lp, &mut time_limit)
    }

    pub fn solve_with_time_limit(
        &mut self,
        lp: &LinearProgram,
        time_limit: &mut TimeLimit,
    ) -> ProblemStatus {
        self.last_error = None;
        self.num_revised_simplex_iterations = 0;
        if let Err(error) = self.parameters.validate_supported() {
            self.last_error = Some(error);
            self.resize_solution(lp.num_constraints(), lp.num_variables());
            return ProblemStatus::Abnormal;
        }
        if !lp.is_cleaned_up() || !lp.is_valid(self.parameters.max_valid_magnitude) {
            self.resize_solution(lp.num_constraints(), lp.num_variables());
            return ProblemStatus::InvalidProblem;
        }
        // Keep GLOP's reusable internal copy owned by the solver. Temporarily
        // move it out so postsolve can mutate the other solver fields.
        let mut current_lp = std::mem::take(&mut self.current_linear_program);
        current_lp.populate_from_linear_program(lp);
        let status = self.solve_copied_program(lp, time_limit, &mut current_lp);
        self.current_linear_program = current_lp;
        status
    }

    fn solve_copied_program(
        &mut self,
        lp: &LinearProgram,
        time_limit: &mut TimeLimit,
        current_lp: &mut LinearProgram,
    ) -> ProblemStatus {
        // Certificates are solver-owned, not stored in `ProblemSolution`.
        self.primal_ray.clear();
        self.constraints_dual_ray.clear();
        self.variable_bounds_dual_ray.clear();
        self.num_revised_simplex_iterations = 0;
        // GLOP makes an internal model copy and drops tiny entries before
        // running its ordered presolve stack, even when presolve is disabled.
        current_lp.remove_near_zero_entries(self.parameters.drop_magnitude);
        let mut preprocessor = MainLpPreprocessor::new();
        let postsolve_needed = preprocessor.run(current_lp, &self.parameters, time_limit);
        let mut solution =
            ProblemSolution::new(current_lp.num_constraints(), current_lp.num_variables());
        solution.status = preprocessor.status();
        if solution.status != ProblemStatus::Init {
            if postsolve_needed {
                preprocessor.destructive_recover_solution(&mut solution);
            }
            return self.load_and_verify_solution(lp, &solution);
        }
        if time_limit.limit_reached() {
            solution.status = ProblemStatus::Init;
            if postsolve_needed {
                preprocessor.destructive_recover_solution(&mut solution);
            }
            return self.load_and_verify_solution(lp, &solution);
        }
        current_lp.clear_transpose_matrix();
        self.revised_simplex.set_parameters(&self.parameters);
        if self.revised_simplex.solve(current_lp, time_limit).is_err() {
            self.resize_solution(lp.num_constraints(), lp.num_variables());
            return ProblemStatus::Abnormal;
        }
        self.num_revised_simplex_iterations = self.revised_simplex.number_of_iterations();
        solution.status = self.revised_simplex.problem_status();
        for column in 0..current_lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            solution.primal_values[column] = self.revised_simplex.variable_value(column);
            solution.variable_statuses[column] = self.revised_simplex.variable_status(column);
        }
        for row in 0..current_lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(row);
            solution.dual_values[row] = self.revised_simplex.dual_value(row);
            solution.constraint_statuses[row] = self.revised_simplex.constraint_status(row);
        }
        if !self.parameters.use_preprocessing && !self.parameters.use_scaling {
            match solution.status {
                ProblemStatus::PrimalUnbounded => {
                    self.primal_ray = self.revised_simplex.primal_ray().clone();
                    self.primal_ray.truncate(current_lp.num_variables());
                }
                ProblemStatus::DualUnbounded => {
                    self.constraints_dual_ray = self.revised_simplex.dual_ray().clone();
                    if !lp.is_maximization_problem() {
                        for value in self.constraints_dual_ray.as_mut_slice() {
                            *value = -*value;
                        }
                    }
                    self.variable_bounds_dual_ray =
                        DenseRow::filled(current_lp.num_variables(), 0.0);
                    for column in 0..current_lp.num_variables().to_usize() {
                        let column = ColIndex::from_usize(column);
                        self.variable_bounds_dual_ray[column] = -current_lp
                            .sparse_column(column)
                            .iter()
                            .map(|entry| {
                                self.constraints_dual_ray[entry.index()] * entry.coefficient()
                            })
                            .sum::<f64>();
                    }
                }
                _ => {}
            }
        }
        if postsolve_needed {
            preprocessor.destructive_recover_solution(&mut solution);
        }
        self.load_and_verify_solution(lp, &solution)
    }

    pub fn clear(&mut self) {
        let parameters = self.parameters.clone();
        *self = Self::new();
        self.parameters = parameters;
    }

    /// Solve and expose the same solver-owned values as the individual GLOP getters.
    pub fn solve_result(&mut self, lp: &LinearProgram) -> SolveResult<'_> {
        let status = self.solve(lp);
        self.result(status)
    }

    #[must_use]
    pub fn result(&self, status: ProblemStatus) -> SolveResult<'_> {
        SolveResult {
            status,
            objective_value: self.objective_value(),
            simplex_iterations: self.number_of_simplex_iterations(),
            maximum_primal_infeasibility: self.maximum_primal_infeasibility(),
            maximum_dual_infeasibility: self.maximum_dual_infeasibility(),
            deterministic_time: self.deterministic_time(),
            may_have_multiple_optimal_solutions: self.may_have_multiple_optimal_solutions(),
            variable_values: self.variable_values(),
            reduced_costs: self.reduced_costs(),
            variable_statuses: self.variable_statuses(),
            dual_values: self.dual_values(),
            constraint_activities: self.constraint_activities(),
            constraint_statuses: self.constraint_statuses(),
            primal_ray: self.primal_ray(),
            constraints_dual_ray: self.constraints_dual_ray(),
            variable_bounds_dual_ray: self.variable_bounds_dual_ray(),
        }
    }

    #[must_use]
    pub fn export_basis(&self) -> SolverBasis {
        SolverBasis {
            variable_statuses: self.variable_statuses.clone(),
            constraint_statuses: self.constraint_statuses.clone(),
        }
    }

    pub fn import_basis(&mut self, basis: &SolverBasis) {
        self.set_initial_basis(&basis.variable_statuses, &basis.constraint_statuses);
    }

    /// Cumulative deterministic work since construction or [`Self::clear`].
    #[must_use]
    #[inline]
    pub fn deterministic_time(&self) -> f64 {
        self.revised_simplex.deterministic_time()
    }

    pub fn set_initial_basis(
        &mut self,
        variable_statuses: &VariableStatusRow,
        constraint_statuses: &ConstraintStatusColumn,
    ) {
        let mut statuses = variable_statuses.clone();
        for &status in constraint_statuses {
            statuses.push(match status {
                ConstraintStatus::Free => VariableStatus::Free,
                ConstraintStatus::AtLowerBound => VariableStatus::AtUpperBound,
                ConstraintStatus::AtUpperBound => VariableStatus::AtLowerBound,
                ConstraintStatus::FixedValue => VariableStatus::FixedValue,
                ConstraintStatus::Basic => VariableStatus::Basic,
            });
        }
        self.revised_simplex
            .load_state_for_next_solve(&BasisState { statuses });
    }

    pub fn load_and_verify_solution(
        &mut self,
        lp: &LinearProgram,
        solution: &ProblemSolution,
    ) -> ProblemStatus {
        if !Self::is_problem_solution_consistent(lp, solution) {
            self.resize_solution(lp.num_constraints(), lp.num_variables());
            return ProblemStatus::Abnormal;
        }
        self.primal_values.clone_from(&solution.primal_values);
        self.dual_values.clone_from(&solution.dual_values);
        self.variable_statuses
            .clone_from(&solution.variable_statuses);
        self.constraint_statuses
            .clone_from(&solution.constraint_statuses);
        self.compute_reduced_costs(lp);
        let primal_objective_value = self.compute_objective(lp);
        let dual_objective_value = self.compute_dual_objective(lp);
        // GLOP's strong optimal guarantee snaps primal values into their
        // variable bounds and one-sided dual values to zero after recording
        // the pre-snap objective pair, then recomputes public residuals.
        if solution.status == ProblemStatus::Optimal
            && self.parameters.provide_strong_optimal_guarantee
        {
            self.move_primal_values_within_bounds(lp);
            self.move_dual_values_within_bounds(lp);
        }
        self.compute_reduced_costs(lp);
        self.compute_constraint_activities(lp);
        self.problem_objective_value =
            lp.objective_scaling_factor() * (self.compute_objective(lp) + lp.objective_offset());
        let rhs_perturbation_is_too_large = self.rhs_perturbation_is_too_large(lp);
        let cost_perturbation_is_too_large = self.cost_perturbation_is_too_large(lp);
        let (primal_infeasibility_is_too_large, dual_infeasibility_is_too_large) =
            self.compute_infeasibilities(lp);

        let mut status = solution.status;
        if status == ProblemStatus::Optimal
            && self.parameters.provide_strong_optimal_guarantee
            && self.parameters.change_status_to_imprecise
            && (rhs_perturbation_is_too_large || cost_perturbation_is_too_large)
        {
            status = ProblemStatus::Imprecise;
        }
        if matches!(
            status,
            ProblemStatus::Optimal | ProblemStatus::PrimalFeasible
        ) && primal_infeasibility_is_too_large
            && self.parameters.change_status_to_imprecise
        {
            status = ProblemStatus::Imprecise;
        }
        if matches!(status, ProblemStatus::Optimal | ProblemStatus::DualFeasible)
            && dual_infeasibility_is_too_large
            && self.parameters.change_status_to_imprecise
        {
            status = ProblemStatus::Imprecise;
        }
        if status == ProblemStatus::Optimal
            && (primal_objective_value - dual_objective_value).abs()
                > self.compute_max_expected_objective_error(lp)
            && self.parameters.change_status_to_imprecise
        {
            status = ProblemStatus::Imprecise;
        }
        self.may_have_multiple_solutions = status == ProblemStatus::Optimal
            && (0..lp.num_variables().to_usize()).any(|column| {
                let index = ColIndex::from_usize(column);
                self.variable_statuses[index] != VariableStatus::FixedValue
                    && self.reduced_costs[index].abs() <= 1e-9
                    && ((self.primal_values[index] - lp.variable_lower_bounds()[index]).abs()
                        <= 1e-7
                        || (self.primal_values[index] - lp.variable_upper_bounds()[index]).abs()
                            <= 1e-7)
            });
        if status == ProblemStatus::Optimal && !self.may_have_multiple_solutions {
            self.may_have_multiple_solutions = (0..lp.num_constraints().to_usize()).any(|row| {
                let index = RowIndex::from_usize(row);
                self.constraint_statuses[index] != ConstraintStatus::FixedValue
                    && self.dual_values[index].abs() <= 1e-9
                    && ((self.constraint_activities[index] - lp.constraint_lower_bounds()[index])
                        .abs()
                        <= 1e-7
                        || (self.constraint_activities[index]
                            - lp.constraint_upper_bounds()[index])
                            .abs()
                            <= 1e-7)
            });
        }
        status
    }

    fn is_problem_solution_consistent(lp: &LinearProgram, solution: &ProblemSolution) -> bool {
        if solution.primal_values.len() != lp.num_variables()
            || solution.dual_values.len() != lp.num_constraints()
            || solution.variable_statuses.len() != lp.num_variables()
            || solution.constraint_statuses.len() != lp.num_constraints()
        {
            return false;
        }
        if !matches!(
            solution.status,
            ProblemStatus::Optimal | ProblemStatus::PrimalFeasible | ProblemStatus::DualFeasible
        ) {
            return true;
        }

        let mut num_basic_variables = 0_usize;
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            let value = solution.primal_values[column];
            let lower = lp.variable_lower_bounds()[column];
            let upper = lp.variable_upper_bounds()[column];
            match solution.variable_statuses[column] {
                VariableStatus::Basic => num_basic_variables += 1,
                VariableStatus::FixedValue => {
                    if value != upper && value != lower {
                        return false;
                    }
                }
                VariableStatus::AtLowerBound => {
                    if value != lower || lower == upper {
                        return false;
                    }
                }
                VariableStatus::AtUpperBound => {
                    let error = (value - upper).abs();
                    if error.is_nan() || error > 1e-7 || lower == upper {
                        return false;
                    }
                }
                VariableStatus::Free => {
                    if lower != f64::NEG_INFINITY || upper != f64::INFINITY || value != 0.0 {
                        return false;
                    }
                }
            }
        }
        for row in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(row);
            let dual_value = solution.dual_values[row];
            let lower = lp.constraint_lower_bounds()[row];
            let upper = lp.constraint_upper_bounds()[row];
            match solution.constraint_statuses[row] {
                ConstraintStatus::Basic => {
                    if dual_value != 0.0 {
                        return false;
                    }
                    num_basic_variables += 1;
                }
                ConstraintStatus::FixedValue => {
                    if upper - lower > 1e-12 {
                        return false;
                    }
                }
                ConstraintStatus::AtLowerBound => {
                    if lower == f64::NEG_INFINITY {
                        return false;
                    }
                }
                ConstraintStatus::AtUpperBound => {
                    if upper == f64::INFINITY {
                        return false;
                    }
                }
                ConstraintStatus::Free => {
                    if dual_value != 0.0 || lower != f64::NEG_INFINITY || upper != f64::INFINITY {
                        return false;
                    }
                }
            }
        }
        num_basic_variables == lp.num_constraints().to_usize()
    }

    fn resize_solution(&mut self, rows: RowIndex, columns: ColIndex) {
        self.primal_values = DenseRow::filled(columns, 0.0);
        self.reduced_costs = DenseRow::filled(columns, 0.0);
        self.variable_statuses = VariableStatusRow::filled(columns, VariableStatus::Free);
        self.dual_values = DenseColumn::filled(rows, 0.0);
        self.constraint_activities = DenseColumn::filled(rows, 0.0);
        self.constraint_statuses = ConstraintStatusColumn::filled(rows, ConstraintStatus::Free);
    }

    fn compute_reduced_costs(&mut self, lp: &LinearProgram) {
        self.reduced_costs = DenseRow::from_vec(
            (0..lp.num_variables().to_usize())
                .map(|column| {
                    let column = ColIndex::from_usize(column);
                    let matrix_dual = lp.sparse_column(column).scalar_product(&self.dual_values);
                    lp.objective_coefficients()[column] - matrix_dual
                })
                .collect(),
        );
    }

    fn move_primal_values_within_bounds(&mut self, lp: &LinearProgram) {
        for position in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(position);
            self.primal_values[col] = self.primal_values[col].min(lp.variable_upper_bounds()[col]);
            self.primal_values[col] = self.primal_values[col].max(lp.variable_lower_bounds()[col]);
        }
    }

    fn move_dual_values_within_bounds(&mut self, lp: &LinearProgram) {
        let optimization_sign = if lp.is_maximization_problem() {
            -1.0
        } else {
            1.0
        };
        for position in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            let mut value = optimization_sign * self.dual_values[row];
            if lp.constraint_lower_bounds()[row] == f64::NEG_INFINITY && value > 0.0 {
                value = 0.0;
            }
            if lp.constraint_upper_bounds()[row] == f64::INFINITY && value < 0.0 {
                value = 0.0;
            }
            self.dual_values[row] = optimization_sign * value;
        }
    }

    fn compute_constraint_activities(&mut self, lp: &LinearProgram) {
        self.constraint_activities = DenseColumn::filled(lp.num_constraints(), 0.0);
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            for entry in lp.sparse_column(column) {
                let row = entry.index();
                self.constraint_activities[row] = self.primal_values[column]
                    .mul_add(entry.coefficient(), self.constraint_activities[row]);
            }
        }
    }

    fn compute_objective(&self, lp: &LinearProgram) -> f64 {
        precise_scalar_product(
            lp.objective_coefficients().as_slice(),
            self.primal_values.as_slice(),
        )
    }

    fn compute_dual_objective(&self, lp: &LinearProgram) -> f64 {
        let optimization_sign = if lp.is_maximization_problem() {
            -1.0
        } else {
            1.0
        };
        let row_terms = (0..lp.num_constraints().to_usize()).filter_map(|row| {
            let row = RowIndex::from_usize(row);
            let lower = lp.constraint_lower_bounds()[row];
            let upper = lp.constraint_upper_bounds()[row];
            let dual = self.dual_values[row];
            let corrected_dual = optimization_sign * dual;
            if corrected_dual > 0.0 && lower != f64::NEG_INFINITY {
                Some(dual * lower)
            } else if corrected_dual < 0.0 && upper != f64::INFINITY {
                Some(dual * upper)
            } else {
                None
            }
        });
        let column_terms = (0..lp.num_variables().to_usize()).map(|column| {
            let column = ColIndex::from_usize(column);
            let reduced = optimization_sign * self.reduced_costs[column];
            let correction = match self.variable_statuses[column] {
                VariableStatus::AtLowerBound if reduced > 0.0 => {
                    reduced * lp.variable_lower_bounds()[column]
                }
                VariableStatus::AtUpperBound if reduced < 0.0 => {
                    reduced * lp.variable_upper_bounds()[column]
                }
                VariableStatus::FixedValue => reduced * lp.variable_upper_bounds()[column],
                _ => 0.0,
            };
            optimization_sign * correction
        });
        accurate_sum(row_terms.chain(column_terms))
    }

    fn compute_max_expected_objective_error(&self, lp: &LinearProgram) -> f64 {
        let tolerance = self.parameters.solution_feasibility_tolerance;
        let mut error = 0.0;
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            error += lp.objective_coefficients()[column].abs()
                * (tolerance * self.primal_values[column].abs().max(1.0));
        }
        error
    }

    fn rhs_perturbation_is_too_large(&self, lp: &LinearProgram) -> bool {
        let tolerance = self.parameters.solution_feasibility_tolerance;
        for position in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(position);
            let activity = self.constraint_activities[row];
            let lower = lp.constraint_lower_bounds()[row];
            let upper = lp.constraint_upper_bounds()[row];
            let status = self.constraint_statuses[row];
            if status == ConstraintStatus::AtLowerBound || activity < lower {
                if (activity - lower).abs() > tolerance * lower.abs().max(1.0) {
                    return true;
                }
            } else if (status == ConstraintStatus::AtUpperBound || activity > upper)
                && (activity - upper).abs() > tolerance * upper.abs().max(1.0)
            {
                return true;
            }
        }
        false
    }

    fn cost_perturbation_is_too_large(&self, lp: &LinearProgram) -> bool {
        let tolerance = self.parameters.solution_feasibility_tolerance;
        let sign = if lp.is_maximization_problem() {
            -1.0
        } else {
            1.0
        };
        for position in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(position);
            let reduced = sign * self.reduced_costs[col];
            let status = self.variable_statuses[col];
            if (status == VariableStatus::Basic
                || status == VariableStatus::Free
                || (status == VariableStatus::AtUpperBound && reduced > 0.0)
                || (status == VariableStatus::AtLowerBound && reduced < 0.0))
                && reduced.abs() > tolerance * lp.objective_coefficients()[col].abs().max(1.0)
            {
                return true;
            }
        }
        false
    }

    fn compute_infeasibilities(&mut self, lp: &LinearProgram) -> (bool, bool) {
        let tolerance = self.parameters.solution_feasibility_tolerance;
        let allowed_error = |value: f64| tolerance * value.abs().max(1.0);
        let mut primal_is_too_large = false;
        self.max_absolute_primal_infeasibility = 0.0;
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            let lower = lp.variable_lower_bounds()[column];
            let upper = lp.variable_upper_bounds()[column];
            let value = self.primal_values[column];
            let (error, reference) = if lower == upper {
                ((value - upper).abs(), upper)
            } else if value > upper {
                (value - upper, upper)
            } else if value < lower {
                (lower - value, lower)
            } else {
                (0.0, 0.0)
            };
            self.max_absolute_primal_infeasibility =
                self.max_absolute_primal_infeasibility.max(error);
            primal_is_too_large |= error > allowed_error(reference);
        }
        for row in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(row);
            let lower = lp.constraint_lower_bounds()[row];
            let upper = lp.constraint_upper_bounds()[row];
            let activity = self.constraint_activities[row];
            let (error, reference) = if lower == upper {
                ((activity - upper).abs(), upper)
            } else if activity > upper {
                (activity - upper, upper)
            } else if activity < lower {
                (lower - activity, lower)
            } else {
                (0.0, 0.0)
            };
            self.max_absolute_primal_infeasibility =
                self.max_absolute_primal_infeasibility.max(error);
            primal_is_too_large |= error > allowed_error(reference);
        }
        self.max_absolute_dual_infeasibility = 0.0;
        let mut dual_is_too_large = false;
        let optimization_sign = if lp.is_maximization_problem() {
            -1.0
        } else {
            1.0
        };
        for row in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(row);
            let value = optimization_sign * self.dual_values[row];
            let mut error = 0.0_f64;
            if lp.constraint_lower_bounds()[row] == f64::NEG_INFINITY {
                error = error.max(value);
            }
            if lp.constraint_upper_bounds()[row] == f64::INFINITY {
                error = error.max(-value);
            }
            self.max_absolute_dual_infeasibility = self.max_absolute_dual_infeasibility.max(error);
            dual_is_too_large |= error > tolerance;
        }
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            let reduced = optimization_sign * self.reduced_costs[column];
            let mut violation = 0.0_f64;
            if lp.variable_lower_bounds()[column] == f64::NEG_INFINITY {
                violation = violation.max(reduced);
            }
            if lp.variable_upper_bounds()[column] == f64::INFINITY {
                violation = violation.max(-reduced);
            }
            self.max_absolute_dual_infeasibility =
                self.max_absolute_dual_infeasibility.max(violation);
            dual_is_too_large |= violation > allowed_error(lp.objective_coefficients()[column]);
        }
        (primal_is_too_large, dual_is_too_large)
    }

    #[must_use]
    #[inline]
    pub const fn objective_value(&self) -> f64 {
        self.problem_objective_value
    }
    #[must_use]
    #[inline]
    pub const fn variable_values(&self) -> &DenseRow {
        &self.primal_values
    }
    #[must_use]
    #[inline]
    pub const fn reduced_costs(&self) -> &DenseRow {
        &self.reduced_costs
    }
    #[must_use]
    #[inline]
    pub const fn variable_statuses(&self) -> &VariableStatusRow {
        &self.variable_statuses
    }
    #[must_use]
    #[inline]
    pub const fn dual_values(&self) -> &DenseColumn {
        &self.dual_values
    }
    #[must_use]
    #[inline]
    pub const fn constraint_activities(&self) -> &DenseColumn {
        &self.constraint_activities
    }
    #[must_use]
    #[inline]
    pub const fn constraint_statuses(&self) -> &ConstraintStatusColumn {
        &self.constraint_statuses
    }
    #[must_use]
    #[inline]
    pub const fn primal_ray(&self) -> &DenseRow {
        &self.primal_ray
    }
    #[must_use]
    pub const fn constraints_dual_ray(&self) -> &DenseColumn {
        &self.constraints_dual_ray
    }
    #[must_use]
    #[inline]
    pub const fn variable_bounds_dual_ray(&self) -> &DenseRow {
        &self.variable_bounds_dual_ray
    }
    #[must_use]
    #[inline]
    pub const fn maximum_primal_infeasibility(&self) -> f64 {
        self.max_absolute_primal_infeasibility
    }
    #[must_use]
    #[inline]
    pub const fn maximum_dual_infeasibility(&self) -> f64 {
        self.max_absolute_dual_infeasibility
    }
    #[must_use]
    pub const fn may_have_multiple_optimal_solutions(&self) -> bool {
        self.may_have_multiple_solutions
    }
    #[must_use]
    #[inline]
    pub const fn number_of_simplex_iterations(&self) -> u64 {
        self.num_revised_simplex_iterations
    }
    #[must_use]
    pub const fn revised_simplex(&self) -> &RevisedSimplex {
        &self.revised_simplex
    }
    pub fn revised_simplex_mut(&mut self) -> &mut RevisedSimplex {
        &mut self.revised_simplex
    }
}
