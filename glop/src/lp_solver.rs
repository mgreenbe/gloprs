//! Public LP solver orchestration and solution validation.
//!
//! This follows `ortools/glop/lp_solver.{h,cc}` with preprocessing and scaling
//! intentionally disabled until Phase 5.  The unscaled solution is checked
//! against the caller's original model, not the equation-form copy used by
//! [`RevisedSimplex`].

#![allow(clippy::float_cmp, clippy::missing_errors_doc)]

use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{
    ColIndex, ConstraintStatus, ConstraintStatusColumn, DenseColumn, DenseRow, ProblemStatus,
    RowIndex, VariableStatus, VariableStatusRow, VectorIndex,
};
use lp_data::lp_utils::{accurate_sum, precise_scalar_product};

use crate::parameters::GlopParameters;
use crate::revised_simplex::RevisedSimplex;
use crate::time_limit::TimeLimit;
use crate::variables_info::BasisState;

#[derive(Debug)]
pub struct LPSolver {
    parameters: GlopParameters,
    revised_simplex: RevisedSimplex,
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
        }
    }

    pub fn set_parameters(&mut self, parameters: &GlopParameters) {
        self.parameters = parameters.clone();
    }

    #[must_use]
    pub const fn parameters(&self) -> &GlopParameters {
        &self.parameters
    }

    pub fn parameters_mut(&mut self) -> &mut GlopParameters {
        &mut self.parameters
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
        if !lp.is_cleaned_up() || !lp.is_valid(self.parameters.max_valid_magnitude) {
            self.resize_solution(lp.num_constraints(), lp.num_variables());
            return ProblemStatus::InvalidProblem;
        }
        self.revised_simplex.set_parameters(&self.parameters);
        if self.revised_simplex.solve(lp, time_limit).is_err() {
            self.resize_solution(lp.num_constraints(), lp.num_variables());
            return ProblemStatus::Abnormal;
        }
        self.num_revised_simplex_iterations = self.revised_simplex.number_of_iterations();
        let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
        solution.status = self.revised_simplex.problem_status();
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            solution.primal_values[column] = self.revised_simplex.variable_value(column);
            solution.variable_statuses[column] = self.revised_simplex.variable_status(column);
        }
        for row in 0..lp.num_constraints().to_usize() {
            let row = RowIndex::from_usize(row);
            solution.dual_values[row] = self.revised_simplex.dual_value(row);
            solution.constraint_statuses[row] = self.revised_simplex.constraint_status(row);
        }
        self.primal_ray = self.revised_simplex.primal_ray().clone();
        self.constraints_dual_ray = self.revised_simplex.dual_ray().clone();
        self.load_and_verify_solution(lp, &solution)
    }

    pub fn clear(&mut self) {
        *self = Self::new();
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
        self.compute_constraint_activities(lp);
        self.problem_objective_value =
            lp.objective_scaling_factor() * (self.compute_objective(lp) + lp.objective_offset());
        let (primal_infeasibility_is_too_large, dual_infeasibility_is_too_large) =
            self.compute_infeasibilities(lp);

        let mut status = solution.status;
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
                    let matrix_dual = lp
                        .sparse_column(column)
                        .iter()
                        .map(|entry| self.dual_values[entry.index()] * entry.coefficient())
                        .sum::<f64>();
                    lp.objective_coefficients()[column] - matrix_dual
                })
                .collect(),
        );
    }

    fn compute_constraint_activities(&mut self, lp: &LinearProgram) {
        self.constraint_activities = DenseColumn::filled(lp.num_constraints(), 0.0);
        for column in 0..lp.num_variables().to_usize() {
            let column = ColIndex::from_usize(column);
            for entry in lp.sparse_column(column) {
                self.constraint_activities[entry.index()] +=
                    entry.coefficient() * self.primal_values[column];
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
                * tolerance
                * self.primal_values[column].abs().max(1.0);
        }
        error
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
    pub const fn objective_value(&self) -> f64 {
        self.problem_objective_value
    }
    #[must_use]
    pub const fn variable_values(&self) -> &DenseRow {
        &self.primal_values
    }
    #[must_use]
    pub const fn reduced_costs(&self) -> &DenseRow {
        &self.reduced_costs
    }
    #[must_use]
    pub const fn variable_statuses(&self) -> &VariableStatusRow {
        &self.variable_statuses
    }
    #[must_use]
    pub const fn dual_values(&self) -> &DenseColumn {
        &self.dual_values
    }
    #[must_use]
    pub const fn constraint_activities(&self) -> &DenseColumn {
        &self.constraint_activities
    }
    #[must_use]
    pub const fn constraint_statuses(&self) -> &ConstraintStatusColumn {
        &self.constraint_statuses
    }
    #[must_use]
    pub const fn primal_ray(&self) -> &DenseRow {
        &self.primal_ray
    }
    #[must_use]
    pub const fn constraints_dual_ray(&self) -> &DenseColumn {
        &self.constraints_dual_ray
    }
    #[must_use]
    pub const fn variable_bounds_dual_ray(&self) -> &DenseRow {
        &self.variable_bounds_dual_ray
    }
    #[must_use]
    pub const fn maximum_primal_infeasibility(&self) -> f64 {
        self.max_absolute_primal_infeasibility
    }
    #[must_use]
    pub const fn maximum_dual_infeasibility(&self) -> f64 {
        self.max_absolute_dual_infeasibility
    }
    #[must_use]
    pub const fn may_have_multiple_optimal_solutions(&self) -> bool {
        self.may_have_multiple_solutions
    }
    #[must_use]
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
