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
    clippy::struct_excessive_bools,
    clippy::too_many_lines
)]

use std::rc::Rc;

use lp_data::lp_data::LinearProgram;
use lp_data::lp_types::{
    ColIndex, ConstraintStatus, DenseColumn, DenseRow, INVALID_COL, ProblemStatus, RowIndex,
    RowToColMapping, VariableStatus, VariableType, VectorIndex,
};
use lp_data::lp_utils::precise_scalar_product;
use lp_data::permutation::ColumnPermutation;
use lp_data::scattered_vector::ScatteredColumn;
use lp_data::sparse::{CompactSparseMatrix, SparseMatrix};

use crate::basis_representation::BasisRepresentation;
use crate::dual_edge_norms::DualEdgeNorms;
use crate::entering_variable::EnteringVariable;
use crate::initial_basis::InitialBasis;
use crate::lu_factorization::FactorizationError;
use crate::parameters::{GlopParameters, InitialBasisHeuristic};
use crate::pricing::DynamicMaximum;
use crate::primal_edge_norms::{PricingRule as EdgePricingRule, PrimalEdgeNorms};
use crate::primal_ratio_test::{LeavingChoice, choose_leaving_variable_row};
use crate::random::SharedRandom;
use crate::reduced_costs::{
    PrimalPrices, perturb_costs_into, update_reduced_cost_values_before_basis_pivot,
};
use crate::time_limit::TimeLimit;
use crate::update_row::UpdateRow;
use crate::variables_info::{BasisState, VariablesInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimplexPhase {
    Feasibility,
    Optimization,
}

/// GLOP's `ScalarProduct(basic_objective_, direction_)` dispatches on the
/// scattered direction: sparse positions use their stored order, while dense
/// directions accumulate four products per block. The distinction matters
/// for reduced-cost precision tests near the refactorization threshold.
fn basic_objective_direction_scalar_product(
    objective: &DenseRow,
    basis: &RowToColMapping,
    direction: &ScatteredColumn,
) -> f64 {
    if !direction.should_use_dense_iteration(0.8) {
        let mut sum = 0.0;
        for entry in direction {
            sum += objective[basis[entry.index()]] * entry.coefficient();
        }
        return sum;
    }
    let values = direction.values().as_slice();
    let mut sum = 0.0;
    let blocks = values.len() / 4;
    for block in 0..blocks {
        let i = 4 * block;
        sum += objective[basis[RowIndex::from_usize(i)]] * values[i]
            + objective[basis[RowIndex::from_usize(i + 1)]] * values[i + 1]
            + objective[basis[RowIndex::from_usize(i + 2)]] * values[i + 2]
            + objective[basis[RowIndex::from_usize(i + 3)]] * values[i + 3];
    }
    for (i, &value) in values.iter().enumerate().skip(4 * blocks) {
        sum += objective[basis[RowIndex::from_usize(i)]] * value;
    }
    sum
}

/// A normalized event emitted only when tracing is enabled.
#[derive(Clone, Debug, PartialEq)]
pub struct IterationEvent {
    pub iteration: u64,
    pub phase: SimplexPhase,
    pub entering_column: Option<ColIndex>,
    pub leaving_row: Option<RowIndex>,
    pub leaving_column: Option<ColIndex>,
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
    matrix: Rc<SparseMatrix>,
    compact_matrix: CompactSparseMatrix,
    objective: DenseRow,
    objective_offset: f64,
    objective_scaling_factor: f64,
    solution_objective_value: f64,
    primal_objective_limit: f64,
    dual_objective_limit: f64,
    objective_limit_reached: bool,
    basis: RowToColMapping,
    basis_factorization: Option<BasisRepresentation>,
    variables_info: Option<VariablesInfo>,
    variable_values: DenseRow,
    reduced_costs: DenseRow,
    cost_perturbations: DenseRow,
    has_cost_shift: bool,
    dual_values: DenseColumn,
    solution_reduced_costs: DenseRow,
    solution_dual_values: DenseColumn,
    is_maximization_problem: bool,
    random: SharedRandom,
    primal_edge_norms: Option<PrimalEdgeNorms>,
    primal_prices: PrimalPrices,
    dual_edge_norms: DualEdgeNorms,
    dual_prices: DynamicMaximum,
    entering_variable: EnteringVariable,
    bound_flip_candidates: Vec<ColIndex>,
    dual_phase_one_improvement_direction: DenseRow,
    dual_phase_one_pricing_vector: DenseColumn,
    num_dual_infeasible_positions: usize,
    update_row: Option<UpdateRow>,
    primal_ray: DenseRow,
    dual_ray: DenseColumn,
    dual_ray_row_combination: DenseRow,
    solution_state: BasisState,
    state_for_next_solve: Option<BasisState>,
    external_state_for_next_solve: bool,
    starting_values: DenseRow,
    num_iterations: u64,
    num_update_price_operations: i64,
    reduced_costs_deterministic_time: f64,
    reduced_costs_dirty: bool,
    reduced_costs_precise: bool,
    previous_component_deterministic_time: f64,
    last_deterministic_time_update: f64,
    trace_enabled: bool,
    trace: Vec<IterationEvent>,
    phase4_events_enabled: bool,
    phase4_events: Vec<&'static str>,
    initial_basis_before_permutation: RowToColMapping,
    initial_column_permutation: Vec<usize>,
    recovered_warm_basis: Option<RowToColMapping>,
}

impl Default for RevisedSimplex {
    fn default() -> Self {
        Self::new()
    }
}

impl RevisedSimplex {
    fn component_deterministic_time(&self) -> f64 {
        self.basis_factorization
            .as_ref()
            .map_or(0.0, BasisRepresentation::deterministic_time)
            + self
                .update_row
                .as_ref()
                .map_or(0.0, UpdateRow::deterministic_time)
            + self.entering_variable.deterministic_time()
            + self
                .primal_edge_norms
                .as_ref()
                .map_or(0.0, PrimalEdgeNorms::deterministic_time)
    }

    #[must_use]
    pub fn deterministic_time(&self) -> f64 {
        lp_data::lp_types::deterministic_time_for_fp_operations(self.num_update_price_operations)
            + self.previous_component_deterministic_time
            + self
                .basis_factorization
                .as_ref()
                .map_or(0.0, BasisRepresentation::deterministic_time)
            + self
                .update_row
                .as_ref()
                .map_or(0.0, UpdateRow::deterministic_time)
            + self.entering_variable.deterministic_time()
            + self.reduced_costs_deterministic_time
            + self
                .primal_edge_norms
                .as_ref()
                .map_or(0.0, PrimalEdgeNorms::deterministic_time)
    }

    fn advance_deterministic_time(&mut self, time_limit: &mut TimeLimit) {
        let current = self.deterministic_time();
        let delta = current - self.last_deterministic_time_update;
        debug_assert!(delta >= -f64::EPSILON);
        time_limit.advance_deterministic_time(delta.max(0.0));
        self.last_deterministic_time_update = current;
    }

    #[must_use]
    pub fn new() -> Self {
        let random = SharedRandom::new(1);
        Self {
            parameters: GlopParameters::default(),
            problem_status: ProblemStatus::Init,
            num_rows: RowIndex::new(0),
            num_cols: ColIndex::new(0),
            first_slack_col: ColIndex::new(0),
            matrix: Rc::new(SparseMatrix::new()),
            compact_matrix: CompactSparseMatrix::default(),
            objective: DenseRow::new(),
            objective_offset: 0.0,
            objective_scaling_factor: 1.0,
            solution_objective_value: 0.0,
            primal_objective_limit: f64::NEG_INFINITY,
            dual_objective_limit: f64::INFINITY,
            objective_limit_reached: false,
            basis: RowToColMapping::new(),
            basis_factorization: None,
            variables_info: None,
            variable_values: DenseRow::new(),
            reduced_costs: DenseRow::new(),
            cost_perturbations: DenseRow::new(),
            has_cost_shift: false,
            dual_values: DenseColumn::new(),
            solution_reduced_costs: DenseRow::new(),
            solution_dual_values: DenseColumn::new(),
            is_maximization_problem: false,
            random: random.clone(),
            primal_edge_norms: None,
            primal_prices: PrimalPrices::new_with_random(random.clone()),
            dual_edge_norms: DualEdgeNorms::new(),
            dual_prices: DynamicMaximum::new_with_random(random.clone()),
            entering_variable: EnteringVariable::new_with_random(random),
            bound_flip_candidates: Vec::new(),
            dual_phase_one_improvement_direction: DenseRow::new(),
            dual_phase_one_pricing_vector: DenseColumn::new(),
            num_dual_infeasible_positions: 0,
            update_row: None,
            primal_ray: DenseRow::new(),
            dual_ray: DenseColumn::new(),
            dual_ray_row_combination: DenseRow::new(),
            solution_state: BasisState::default(),
            state_for_next_solve: None,
            external_state_for_next_solve: false,
            starting_values: DenseRow::new(),
            num_iterations: 0,
            num_update_price_operations: 0,
            reduced_costs_deterministic_time: 0.0,
            reduced_costs_dirty: false,
            reduced_costs_precise: false,
            previous_component_deterministic_time: 0.0,
            last_deterministic_time_update: 0.0,
            trace_enabled: false,
            trace: Vec::new(),
            phase4_events_enabled: false,
            phase4_events: Vec::new(),
            initial_basis_before_permutation: RowToColMapping::new(),
            initial_column_permutation: Vec::new(),
            recovered_warm_basis: None,
        }
    }

    pub fn set_parameters(&mut self, parameters: &GlopParameters) {
        #[allow(clippy::cast_sign_loss)]
        self.random.seed(parameters.random_seed as u64);
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

    /// Enable stable diagnostic tags for Phase-4 algorithmic branches.
    /// Disabled by default so ordinary solves do not allocate or emit events.
    pub fn set_phase4_events_enabled(&mut self, enabled: bool) {
        self.phase4_events_enabled = enabled;
        self.phase4_events.clear();
    }

    #[must_use]
    pub fn phase4_events(&self) -> &[&'static str] {
        &self.phase4_events
    }

    fn phase4_event(&mut self, event: &'static str) {
        if self.phase4_events_enabled {
            self.phase4_events.push(event);
        }
    }

    #[must_use]
    pub fn num_basis_updates(&self) -> usize {
        self.basis_factorization
            .as_ref()
            .map_or(0, BasisRepresentation::num_updates)
    }

    #[must_use]
    pub fn initial_basis_before_permutation(&self) -> &RowToColMapping {
        &self.initial_basis_before_permutation
    }

    #[must_use]
    pub fn initial_column_permutation(&self) -> &[usize] {
        &self.initial_column_permutation
    }

    #[must_use]
    pub fn dual_phase_one_pricing_vector(&self) -> &DenseColumn {
        &self.dual_phase_one_pricing_vector
    }

    pub fn dual_edge_squared_norms(&mut self) -> Result<&[f64], FactorizationError> {
        self.dual_edge_norms
            .edge_squared_norms(self.basis_factorization.as_ref().unwrap())
    }

    pub fn clear_state_for_next_solve(&mut self) {
        self.state_for_next_solve = Some(BasisState::default());
        self.external_state_for_next_solve = false;
        self.solution_state = BasisState::default();
        self.starting_values.clear();
    }

    pub fn load_state_for_next_solve(&mut self, state: &BasisState) {
        if self
            .state_for_next_solve
            .as_ref()
            .unwrap_or(&self.solution_state)
            == state
        {
            return;
        }
        self.state_for_next_solve = Some(state.clone());
        self.external_state_for_next_solve = true;
    }

    pub fn set_starting_variable_values_for_next_solve(&mut self, values: &DenseRow) {
        self.starting_values.clone_from(values);
    }

    pub fn solve(
        &mut self,
        linear_program: &LinearProgram,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        let result = self.solve_internal(linear_program, time_limit);
        self.advance_deterministic_time(time_limit);
        result
    }

    fn solve_internal(
        &mut self,
        linear_program: &LinearProgram,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        self.phase4_events.clear();
        // GLOP retains SaveState() for the next Solve() by default. An
        // explicit empty or external state overrides this one-shot choice.
        if self.state_for_next_solve.is_none() && !self.solution_state.is_empty() {
            self.state_for_next_solve = Some(self.solution_state.clone());
            self.external_state_for_next_solve = false;
        }
        if !self.starting_values.is_empty() {
            self.phase4_event("starting_variable_values");
        }
        self.initialize(linear_program)?;
        self.trace.clear();
        self.num_iterations = 0;

        let mut ran_dual = false;
        let mut optimization_started = false;
        if self.parameters.use_dual_simplex {
            if self.parameters.perturb_costs_in_dual_simplex {
                self.phase4_event("dual_cost_perturbation");
                self.perturb_costs();
            }
            (ran_dual, optimization_started) = self.prepare_and_run_dual_phase_two(time_limit)?;
        }
        if !ran_dual {
            self.run_primal_phase(SimplexPhase::Feasibility, time_limit)?;
            // The native driver restores the user objective and invalidates
            // reduced costs after Phase I, even if the basis is unchanged.
            if self.problem_status != ProblemStatus::PrimalInfeasible {
                // ResetForNewObjective() is lazy upstream. In particular,
                // hitting a Phase-I limit does not solve for the new reduced
                // costs until a later request (possibly the final snapshot).
                self.reduced_costs_dirty = true;
            }
            if self.problem_status == ProblemStatus::PrimalFeasible && !time_limit.limit_reached() {
                if self.num_iterations == 0
                    || self.num_iterations
                        < u64::try_from(self.parameters.max_number_of_iterations)
                            .unwrap_or(u64::MAX)
                {
                    self.phase4_event("primal_phase_two");
                    optimization_started = true;
                    self.run_primal_phase(SimplexPhase::Optimization, time_limit)?;
                } else {
                    self.phase4_event("primal_phase_two_skipped_at_limit");
                }
            }
        }
        if optimization_started {
            self.reoptimize_after_cleanup(time_limit)?;
        }
        self.validate_terminal_precision()?;
        if !self.starting_values.is_empty()
            && self.parameters.push_to_vertex
            && self.problem_status == ProblemStatus::Optimal
            && (0..self.num_cols.to_usize()).any(|column| {
                let index = ColIndex::from_usize(column);
                self.variables_info.as_ref().unwrap().variable_statuses()[index]
                    == VariableStatus::Free
                    && self.variable_values[index] != 0.0
            })
        {
            self.phase4_event("primal_push");
            self.run_primal_push(time_limit)?;
        }
        self.finish_solution()?;
        self.starting_values.clear();
        Ok(())
    }

    /// Removes temporary shifts and reoptimizes if the precise, refactorized
    /// solution no longer satisfies the requested primal or dual tolerance.
    ///
    /// This is the cleanup/reoptimization loop in GLOP's `Minimize()`.  It is
    /// deliberately outside the individual primal and dual drivers: removing
    /// a stabilization shift can make the opposite simplex algorithm the
    /// appropriate one for the next pass.
    fn reoptimize_after_cleanup(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        if self.objective_limit_reached && self.problem_status == ProblemStatus::PrimalFeasible {
            self.phase4_event("primal_objective_limit_cleanup");
        }
        if self.objective_limit_reached && self.problem_status == ProblemStatus::DualFeasible {
            self.phase4_event("dual_objective_limit_cleanup");
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let maximum_reoptimizations = self.parameters.max_number_of_reoptimizations as usize;
        for reoptimization in 0..=maximum_reoptimizations {
            if !matches!(
                self.problem_status,
                ProblemStatus::Optimal
                    | ProblemStatus::PrimalFeasible
                    | ProblemStatus::DualFeasible
                    | ProblemStatus::PrimalUnbounded
                    | ProblemStatus::DualUnbounded
            ) {
                break;
            }

            self.remove_cost_shifts();
            self.reduced_costs_dirty = true;
            self.basis_factorization.as_mut().unwrap().refactorize()?;
            self.incorporate_basis_permutation();
            self.initialize_values()?;

            // Upstream performs shift removal and this refactorization after
            // an optimization call even when that call found an unbounded
            // ray. Only then does it validate the ray and leave the loop.
            if matches!(
                self.problem_status,
                ProblemStatus::PrimalUnbounded | ProblemStatus::DualUnbounded
            ) {
                break;
            }

            if self.problem_status == ProblemStatus::Optimal {
                let objective = self.objective.clone();
                self.compute_reduced_costs(&objective)?;
                self.phase4_event("optimal_cleanup");
                let solution_tolerance = self.parameters.solution_feasibility_tolerance;
                let primal_residual = self.maximum_primal_residual();
                let dual_residual = self.maximum_dual_residual(&self.objective);
                if primal_residual > solution_tolerance || dual_residual > solution_tolerance {
                    self.phase4_event("optimal_cleanup_residual_rejected");
                    if self.parameters.change_status_to_imprecise {
                        self.problem_status = ProblemStatus::Imprecise;
                    }
                    break;
                }
                let primal_infeasibility = self.maximum_primal_infeasibility();
                let dual_infeasibility = self.maximum_dual_infeasibility();
                let primal_tolerance =
                    primal_residual.max(self.parameters.primal_feasibility_tolerance);
                let dual_tolerance = dual_residual.max(self.parameters.dual_feasibility_tolerance);
                if primal_infeasibility > primal_tolerance && dual_infeasibility > dual_tolerance {
                    if self.parameters.change_status_to_imprecise {
                        self.problem_status = ProblemStatus::Imprecise;
                    }
                    break;
                }
                if primal_infeasibility > primal_tolerance {
                    self.phase4_event("cleanup_primal_to_dual");
                    if reoptimization == maximum_reoptimizations {
                        self.phase4_event("cleanup_reoptimization_limit");
                        break;
                    }
                    self.problem_status = ProblemStatus::DualFeasible;
                } else if dual_infeasibility > dual_tolerance {
                    self.phase4_event("cleanup_dual_to_primal");
                    if reoptimization == maximum_reoptimizations {
                        self.phase4_event("cleanup_reoptimization_limit");
                        break;
                    }
                    self.problem_status = ProblemStatus::PrimalFeasible;
                } else {
                    break;
                }
            }

            // The first optimization call happened before this cleanup loop.
            // Native permits max_number_of_reoptimizations + 1 calls total,
            // but performs cleanup after the final call as well.
            if self.objective_limit_reached
                || reoptimization == maximum_reoptimizations
                || (time_limit.limit_reached() && {
                    self.phase4_event("cleanup_time_limit");
                    true
                })
                || (self.num_iterations != 0
                    && self.num_iterations
                        >= u64::try_from(self.parameters.max_number_of_iterations)
                            .unwrap_or(u64::MAX))
            {
                break;
            }
            if self.problem_status == ProblemStatus::PrimalFeasible {
                self.phase4_event("cleanup_primal_reoptimization");
                self.run_primal_phase(SimplexPhase::Optimization, time_limit)?;
            } else if self.problem_status == ProblemStatus::DualFeasible {
                self.phase4_event("cleanup_dual_reoptimization");
                self.run_dual_phase_two(time_limit)?;
            }
        }
        Ok(())
    }

    /// GLOP's post-optimal `PrimalPush()`: move each nonbasic super-basic
    /// variable toward zero or its nearest bound without pricing it by its
    /// (approximately zero) true reduced cost.
    fn run_primal_push(&mut self, time_limit: &mut TimeLimit) -> Result<(), FactorizationError> {
        let result = self.run_primal_push_internal(time_limit);
        self.advance_deterministic_time(time_limit);
        result
    }

    fn run_primal_push_internal(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        self.primal_edge_norms.as_mut().unwrap().clear();
        self.dual_edge_norms.clear();
        self.update_row.as_mut().unwrap().invalidate();
        self.remove_cost_shifts();
        self.reduced_costs_dirty = true;
        let mut super_basic_cols = Vec::new();
        for column in 0..self.num_cols.to_usize() {
            let index = ColIndex::from_usize(column);
            if self.variables_info.as_ref().unwrap().variable_statuses()[index]
                == VariableStatus::Free
                && self.variable_values[index] != 0.0
            {
                super_basic_cols.push(index);
            }
        }
        let mut direction = ScatteredColumn::new(self.num_rows);
        let mut refactorize = false;
        while let Some(&entering) = super_basic_cols.last() {
            self.advance_deterministic_time(time_limit);
            if time_limit.limit_reached() {
                self.phase4_event("primal_push_time_limit");
                break;
            }
            if refactorize && !self.basis_factorization.as_ref().unwrap().is_refactorized() {
                self.basis_factorization.as_mut().unwrap().refactorize()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
            }
            refactorize = false;
            if self.basis_factorization.as_ref().unwrap().is_refactorized() {
                self.correct_errors_on_variable_values()?;
            }
            let entering_value = self.variable_values[entering];
            let info = self.variables_info.as_ref().unwrap();
            let unbounded = info.variable_types()[entering] == VariableType::Unconstrained;
            let fake_reduced_cost = if unbounded {
                if entering_value > 0.0 { 1.0 } else { -1.0 }
            } else if entering_value - info.lower_bounds()[entering.to_usize()]
                <= info.upper_bounds()[entering.to_usize()] - entering_value
            {
                1.0
            } else {
                -1.0
            };
            self.direction(entering, &mut direction)?;
            let direction_norm = direction
                .values()
                .as_slice()
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            let choice = choose_leaving_variable_row(
                entering,
                fake_reduced_cost,
                &direction,
                direction_norm,
                self.variable_values.as_slice(),
                info.lower_bounds(),
                info.upper_bounds(),
                &self.basis,
                self.basis_factorization.as_ref().unwrap().is_refactorized(),
                &self.parameters,
                &self.random,
            );
            let (leaving_row, mut step_length, target_bound) = match choice {
                LeavingChoice::Refactorize => {
                    self.phase4_event("primal_push_refactorization_requested");
                    refactorize = true;
                    continue;
                }
                LeavingChoice::BoundFlip { step } => (None, step, 0.0),
                LeavingChoice::Pivot {
                    row,
                    step,
                    target_bound,
                    ..
                } => (Some(row), step, target_bound),
            };
            super_basic_cols.pop();
            if step_length.is_infinite() {
                if unbounded {
                    step_length = entering_value.abs();
                } else {
                    self.problem_status = ProblemStatus::Abnormal;
                    break;
                }
            }
            let step = if fake_reduced_cost > 0.0 {
                -step_length
            } else {
                step_length
            };
            let degenerate = leaving_row.is_some_and(|row| {
                let leaving = self.basis[row];
                let movement = -direction.value(row) * step;
                movement == 0.0
                    || (movement > 0.0 && self.variable_values[leaving] >= target_bound)
                    || (movement < 0.0 && self.variable_values[leaving] <= target_bound)
            });
            for entry in &direction {
                let basic = self.basis[entry.row()];
                self.variable_values[basic] =
                    (-entry.coefficient()).mul_add(step, self.variable_values[basic]);
            }
            self.variable_values[entering] += step;
            if let Some(row) = leaving_row {
                self.phase4_event("primal_push_pivot");
                let leaving = self.basis[row];
                if !degenerate {
                    self.variable_values[leaving] = target_bound;
                }
                let status = self.status_at_bound(leaving, target_bound);
                self.variables_info
                    .as_mut()
                    .unwrap()
                    .update_to_nonbasic_status(leaving, status);
                self.variables_info
                    .as_mut()
                    .unwrap()
                    .update_to_basic_status(entering);
                self.basis[row] = entering;
                self.update_row
                    .as_mut()
                    .unwrap()
                    .compute_unit_row_left_inverse(
                        self.basis_factorization.as_ref().unwrap(),
                        row.to_usize(),
                    )?;
                let pivot_from_update_row = self
                    .matrix
                    .column(entering)
                    .iter()
                    .map(|entry| {
                        self.update_row.as_ref().unwrap().unit_row_left_inverse()
                            [entry.index().to_usize()]
                            * entry.coefficient()
                    })
                    .sum::<f64>();
                let pivot_from_direction = direction.value(row);
                let pivot_difference = (pivot_from_update_row - pivot_from_direction).abs();
                let imprecise_pivot = pivot_difference
                    > self.parameters.refactorization_threshold
                        * (1.0 + pivot_from_update_row.abs().min(pivot_from_direction.abs()));
                if imprecise_pivot {
                    self.strengthen_lu_pivoting_after_early_imprecision()?;
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .update_and_refactorize(row.to_usize(), entering)?;
                } else {
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .update_after_solve(entering, row.to_usize(), &direction)?;
                }
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
            } else if unbounded {
                self.variable_values[entering] = 0.0;
            } else if step > 0.0 {
                self.variables_info
                    .as_mut()
                    .unwrap()
                    .update_to_nonbasic_status(entering, VariableStatus::AtUpperBound);
                self.variable_values[entering] =
                    self.variables_info.as_ref().unwrap().upper_bounds()[entering.to_usize()];
            } else if step < 0.0 {
                self.variables_info
                    .as_mut()
                    .unwrap()
                    .update_to_nonbasic_status(entering, VariableStatus::AtLowerBound);
                self.variable_values[entering] =
                    self.variables_info.as_ref().unwrap().lower_bounds()[entering.to_usize()];
            }
            self.num_iterations += 1;
        }
        Ok(())
    }

    /// GLOP's final status check in `SolveInternal()`. Do not treat a status
    /// reached at a time/iteration limit as precise merely because the loop
    /// could not run another cleanup pass.
    fn validate_terminal_precision(&mut self) -> Result<(), FactorizationError> {
        let change_status_to_imprecise = self.parameters.change_status_to_imprecise;
        if !change_status_to_imprecise {
            self.phase4_event("imprecise_status_disabled");
        }
        if self.problem_status == ProblemStatus::DualInfeasible {
            return Ok(());
        }
        let tolerance = self.parameters.solution_feasibility_tolerance;
        if change_status_to_imprecise {
            if self.reduced_costs_dirty
                && matches!(
                    self.problem_status,
                    ProblemStatus::PrimalUnbounded | ProblemStatus::DualUnbounded
                )
            {
                let objective = self.objective.clone();
                self.compute_reduced_costs(&objective)?;
            }
            // GLOP temporarily replaces objective_ with the Phase-I objective
            // when primal feasibility fails. This port keeps the user objective
            // separately and constructs that Phase-I objective on demand.
            let objective = if self.problem_status == ProblemStatus::PrimalInfeasible {
                self.phase_objective(SimplexPhase::Feasibility)
            } else {
                self.objective.clone()
            };
            let primal_residual_too_large = self.maximum_primal_residual() > tolerance;
            let dual_residual_too_large = if primal_residual_too_large {
                false
            } else {
                self.refresh_dual_values(&objective)?;
                self.maximum_dual_residual(&objective) > tolerance
            };
            if primal_residual_too_large
                || dual_residual_too_large
                || matches!(
                    self.problem_status,
                    ProblemStatus::DualFeasible
                        | ProblemStatus::DualUnbounded
                        | ProblemStatus::PrimalInfeasible
                ) && {
                    if self.reduced_costs_dirty {
                        self.compute_reduced_costs_from_dual(&objective);
                    }
                    self.maximum_dual_infeasibility() > tolerance
                }
                || matches!(
                    self.problem_status,
                    ProblemStatus::PrimalFeasible | ProblemStatus::PrimalUnbounded
                ) && self.maximum_primal_infeasibility() > tolerance
            {
                self.phase4_event("imprecise_final_status");
                self.problem_status = ProblemStatus::Imprecise;
                return Ok(());
            }
        }
        if self.problem_status == ProblemStatus::PrimalUnbounded {
            if !change_status_to_imprecise {
                // The unbounded-ray check in GLOP's optimization cleanup
                // computes the dual residual even when final imprecise-status
                // conversion is disabled.
                let objective = self.objective.clone();
                self.refresh_dual_values(&objective)?;
            }
            let info = self.variables_info.as_ref().unwrap();
            let mut min_distance = f64::INFINITY;
            let mut cost_delta = 0.0;
            for column in 0..self.num_cols.to_usize() {
                let ray = self.primal_ray[ColIndex::from_usize(column)];
                cost_delta += ray * self.objective[ColIndex::from_usize(column)];
                if ray > 0.0 && info.upper_bounds()[column] != f64::INFINITY {
                    let distance = (info.upper_bounds()[column]
                        - self.variable_values[ColIndex::from_usize(column)]
                        + tolerance)
                        / ray;
                    min_distance = min_distance.min(distance);
                }
                if ray < 0.0 && info.lower_bounds()[column] != f64::NEG_INFINITY {
                    let distance = (self.variable_values[ColIndex::from_usize(column)]
                        - info.lower_bounds()[column]
                        + tolerance)
                        / -ray;
                    min_distance = min_distance.min(distance);
                }
            }
            if min_distance * cost_delta.abs() < 1.0 {
                // Native ComputeMaximumDualInfeasibility() obtains reduced
                // costs lazily. Shift removal may have invalidated them even
                // when terminal imprecise-status checks are disabled.
                if self.reduced_costs_dirty {
                    let objective = self.objective.clone();
                    self.compute_reduced_costs_from_dual(&objective);
                }
                if self.maximum_dual_infeasibility() <= tolerance {
                    self.phase4_event("primal_ray_rejected");
                    self.problem_status = ProblemStatus::Optimal;
                } else {
                    self.phase4_event("primal_ray_validated");
                }
            } else {
                self.phase4_event("primal_ray_validated");
            }
        }
        if self.problem_status == ProblemStatus::DualUnbounded {
            let info = self.variables_info.as_ref().unwrap();
            let mut implied_lower_bound = 0.0;
            let mut error: f64 = 0.0;
            for column in 0..self.num_cols.to_usize() {
                let coefficient = self.dual_ray_row_combination[ColIndex::from_usize(column)];
                if coefficient > 0.0 {
                    if info.lower_bounds()[column] == f64::NEG_INFINITY {
                        error = error.max(coefficient);
                    } else {
                        implied_lower_bound += coefficient * info.lower_bounds()[column];
                    }
                } else if coefficient < 0.0 {
                    if info.upper_bounds()[column] == f64::INFINITY {
                        error = error.max(-coefficient);
                    } else {
                        implied_lower_bound += coefficient * info.upper_bounds()[column];
                    }
                }
            }
            if implied_lower_bound < tolerance || error > tolerance {
                self.phase4_event("dual_ray_rejected");
                if change_status_to_imprecise {
                    self.problem_status = ProblemStatus::Imprecise;
                }
            } else {
                self.phase4_event("dual_ray_validated");
            }
        }
        Ok(())
    }

    fn prepare_and_run_dual_phase_two(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(bool, bool), FactorizationError> {
        let mut transformed_phase_one = false;
        // Dedicated DualMinimize() first checks the time limit at its loop
        // head, then makes reduced costs precise. Do not move that solve
        // ahead of the check: an exact deterministic-time boundary can stop
        // before it, even when the final operation total is unchanged.
        if self.reduced_costs_dirty && !self.parameters.use_dedicated_dual_feasibility_algorithm {
            let objective = self.objective.clone();
            self.compute_reduced_costs(&objective)?;
        }
        if self.parameters.use_dedicated_dual_feasibility_algorithm {
            self.variables_info
                .as_mut()
                .unwrap()
                .make_boxed_variable_relevant(false);
        }

        let info = self.variables_info.as_ref().unwrap();
        let tolerance = self.parameters.dual_feasibility_tolerance;
        let nonboxed_dual_infeasible = !self.parameters.use_dedicated_dual_feasibility_algorithm
            && info.relevance().iter_ones().any(|column| {
                if info.non_basic_boxed_variables().contains(column) {
                    return false;
                }
                let reduced = self.reduced_costs[column];
                (info.can_increase().contains(column) && reduced < -tolerance)
                    || (info.can_decrease().contains(column) && reduced > tolerance)
            });
        // Native SolveInternal enters the dedicated dual feasibility solve
        // even when the initial reduced costs are already dual feasible. In
        // particular its time-limit check then leaves status INIT.
        if nonboxed_dual_infeasible || self.parameters.use_dedicated_dual_feasibility_algorithm {
            if self.parameters.use_dedicated_dual_feasibility_algorithm {
                self.phase4_event("dual_dedicated_phase_one");
                self.run_dedicated_dual_phase_one(time_limit)?;
                // GLOP restores precision, boxed-variable statuses, and
                // basic values after dedicated Phase I even if the dual
                // solve stopped at an iteration or time limit. Only proven
                // dual infeasibility skips this block.
                if self.problem_status != ProblemStatus::DualInfeasible {
                    self.phase4_event("dual_phase_one_post_cleanup");
                    self.basis_factorization.as_mut().unwrap().refactorize()?;
                    self.incorporate_basis_permutation();
                    let objective = self.objective.clone();
                    if self.reduced_costs_dirty || !self.reduced_costs_precise {
                        self.compute_reduced_costs(&objective)?;
                    }
                    let boxed: Vec<_> = self
                        .variables_info
                        .as_ref()
                        .unwrap()
                        .non_basic_boxed_variables()
                        .iter_ones()
                        .collect();
                    self.make_boxed_variables_dual_feasible(&boxed, false)?;
                    self.initialize_values()?;
                }
            } else {
                self.phase4_event("dual_transformed_phase_one");
                self.run_transformed_dual_phase_one(time_limit)?;
                transformed_phase_one = true;
            }
            if self.problem_status != ProblemStatus::DualFeasible {
                return Ok((true, false));
            }
            if !self.parameters.use_dedicated_dual_feasibility_algorithm {
                self.basis_factorization.as_mut().unwrap().refactorize()?;
                self.incorporate_basis_permutation();
            }
        }

        // Boxed nonbasic variables can be made dual feasible by moving them to
        // their opposite bound without changing the basis.
        let boxed: Vec<_> = self
            .variables_info
            .as_ref()
            .unwrap()
            .non_basic_boxed_variables()
            .iter_ones()
            .collect();
        if !self.parameters.use_dedicated_dual_feasibility_algorithm {
            self.make_boxed_variables_dual_feasible(&boxed, false)?;
        }
        self.variables_info
            .as_mut()
            .unwrap()
            .make_boxed_variable_relevant(true);
        if !self.parameters.use_dedicated_dual_feasibility_algorithm
            && (!transformed_phase_one || !boxed.is_empty())
        {
            // EndDualPhaseI() already recomputed the basic values. In the
            // transformed path there is no second unconditional solve before
            // the optimization loop (unless boxed variables were moved).
            self.initialize_values()?;
        }
        self.problem_status = ProblemStatus::DualFeasible;
        if time_limit.limit_reached()
            || (self.num_iterations != 0
                && self.num_iterations
                    >= u64::try_from(self.parameters.max_number_of_iterations).unwrap_or(u64::MAX))
        {
            return Ok((true, false));
        }
        self.phase4_event("dual_phase_two");
        self.run_dual_phase_two(time_limit)?;
        Ok((true, true))
    }

    /// GLOP's nondefault dual Phase I: temporarily box every nonfixed
    /// variable, optimize that auxiliary problem, then restore original bounds.
    fn run_transformed_dual_phase_one(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        let tolerance = self.parameters.dual_feasibility_tolerance;
        self.variables_info
            .as_mut()
            .unwrap()
            .transform_to_dual_phase_one_problem(tolerance, self.reduced_costs.as_slice());
        // The transformed bounds fix formerly free nonbasic variables at zero.
        self.initialize_values()?;
        self.problem_status = ProblemStatus::DualFeasible;
        self.run_dual_phase_two(time_limit)?;

        self.variables_info
            .as_mut()
            .unwrap()
            .end_dual_phase_one(tolerance, self.reduced_costs.as_slice());
        self.initialize_values()?;
        if self.problem_status == ProblemStatus::Optimal {
            self.problem_status = if self.maximum_dual_infeasibility() < tolerance + 1e-6 {
                ProblemStatus::DualFeasible
            } else {
                ProblemStatus::DualInfeasible
            };
        }
        Ok(())
    }

    fn recompute_dual_prices(&mut self) -> Result<(), FactorizationError> {
        if self.parameters.dual_price_prioritize_norm {
            self.phase4_event("dual_norm_prioritized_pricing");
        }
        let norms = self
            .dual_edge_norms
            .edge_squared_norms(self.basis_factorization.as_ref().unwrap())?;
        self.dual_prices.clear_and_resize(self.num_rows.to_usize());
        self.dual_prices.start_dense_updates();
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let info = self.variables_info.as_ref().unwrap();
        for (row, &squared_norm) in norms.iter().enumerate() {
            let row_index = RowIndex::from_usize(row);
            let column = self.basis[row_index];
            let infeasibility = (self.variable_values[column]
                - info.upper_bounds()[column.to_usize()])
            .max(info.lower_bounds()[column.to_usize()] - self.variable_values[column]);
            if infeasibility > tolerance {
                let price = if self.parameters.dual_price_prioritize_norm {
                    infeasibility.abs() / squared_norm
                } else {
                    infeasibility * infeasibility / squared_norm
                };
                self.dual_prices.dense_add_or_update(row, price);
            }
        }
        Ok(())
    }

    fn update_dual_prices(&mut self, rows: &[RowIndex]) -> Result<(), FactorizationError> {
        let norms = self
            .dual_edge_norms
            .edge_squared_norms(self.basis_factorization.as_ref().unwrap())?;
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let info = self.variables_info.as_ref().unwrap();
        for &row in rows {
            let position = row.to_usize();
            let column = self.basis[row];
            let infeasibility = (self.variable_values[column]
                - info.upper_bounds()[column.to_usize()])
            .max(info.lower_bounds()[column.to_usize()] - self.variable_values[column]);
            if infeasibility > tolerance {
                let price = if self.parameters.dual_price_prioritize_norm {
                    infeasibility.abs() / norms[position]
                } else {
                    infeasibility * infeasibility / norms[position]
                };
                self.dual_prices.add_or_update(position, price);
            } else {
                self.dual_prices.remove(position);
            }
        }
        Ok(())
    }

    fn make_boxed_variables_dual_feasible(
        &mut self,
        columns: &[ColIndex],
        update_basic_values: bool,
    ) -> Result<(), FactorizationError> {
        let tolerance = self.parameters.dual_feasibility_tolerance;
        let mut changed = Vec::new();
        for &column in columns {
            let reduced = self.reduced_costs[column];
            let status = self.variables_info.as_ref().unwrap().variable_statuses()[column];
            let new_status = if reduced > tolerance && status == VariableStatus::AtUpperBound {
                Some(VariableStatus::AtLowerBound)
            } else if reduced < -tolerance && status == VariableStatus::AtLowerBound {
                Some(VariableStatus::AtUpperBound)
            } else {
                None
            };
            if let Some(status) = new_status {
                let old_value = self.variable_values[column];
                let info = self.variables_info.as_mut().unwrap();
                info.update_to_nonbasic_status(column, status);
                self.variable_values[column] = match status {
                    VariableStatus::AtLowerBound => info.lower_bounds()[column.to_usize()],
                    VariableStatus::AtUpperBound => info.upper_bounds()[column.to_usize()],
                    _ => unreachable!(),
                };
                changed.push((column, self.variable_values[column] - old_value));
            }
        }
        if update_basic_values && !changed.is_empty() {
            let mut rhs = ScatteredColumn::new(self.num_rows);
            let mut use_dense = false;
            for (column, delta) in changed {
                if use_dense {
                    self.compact_matrix.column_add_multiple_to_dense_column(
                        column,
                        delta,
                        rhs.values_mut(),
                    );
                } else {
                    self.compact_matrix
                        .column_add_multiple_to_scattered_column(column, delta, &mut rhs);
                    use_dense = rhs.should_use_dense_iteration(0.8);
                }
            }
            rhs.clear_sparse_mask();
            rhs.clear_non_zeros_if_too_dense(0.8);
            self.basis_factorization
                .as_ref()
                .unwrap()
                .solve_with_nonzeros(&mut rhs)?;
            if rhs.non_zeros().is_empty() {
                for row in 0..self.num_rows.to_usize() {
                    let row_index = RowIndex::from_usize(row);
                    self.variable_values[self.basis[row_index]] -= rhs.value(row_index);
                }
                self.recompute_dual_prices()?;
            } else {
                let changed_rows = rhs.non_zeros().to_vec();
                for &row in &changed_rows {
                    self.variable_values[self.basis[row]] -= rhs.value(row);
                }
                self.update_dual_prices(&changed_rows)?;
            }
        }
        Ok(())
    }

    fn dual_phase_one_sign(&self, column: ColIndex) -> f64 {
        let info = self.variables_info.as_ref().unwrap();
        let reduced = self.reduced_costs[column];
        let tolerance = self.parameters.dual_feasibility_tolerance;
        if info.can_increase().contains(column) && reduced < -tolerance {
            1.0
        } else if info.can_decrease().contains(column) && reduced > tolerance {
            -1.0
        } else {
            0.0
        }
    }

    fn dual_phase_one_leaving_candidate(
        cost: f64,
        variable_type: VariableType,
        threshold: f64,
    ) -> bool {
        cost != 0.0
            && (matches!(
                variable_type,
                VariableType::UpperAndLowerBounded | VariableType::FixedVariable
            ) || variable_type == VariableType::UpperBounded && cost < -threshold
                || variable_type == VariableType::LowerBounded && cost > threshold)
    }

    fn update_dual_phase_one_price_at(
        dual_prices: &mut DynamicMaximum,
        row: RowIndex,
        price: f64,
        squared_norm: f64,
        variable_type: VariableType,
        threshold: f64,
        dense: bool,
    ) {
        if Self::dual_phase_one_leaving_candidate(price, variable_type, threshold) {
            let scaled = price * price / squared_norm;
            if dense {
                dual_prices.dense_add_or_update(row.to_usize(), scaled);
            } else {
                dual_prices.add_or_update(row.to_usize(), scaled);
            }
        } else {
            dual_prices.remove(row.to_usize());
        }
    }

    fn update_dual_phase_one_prices_for_columns(
        &mut self,
        columns: &[usize],
        from_scratch: bool,
    ) -> Result<(), FactorizationError> {
        if from_scratch {
            self.num_dual_infeasible_positions = 0;
            self.dual_phase_one_pricing_vector = DenseColumn::filled(self.num_rows, 0.0);
            self.dual_phase_one_improvement_direction = DenseRow::filled(self.num_cols, 0.0);
            self.dual_prices.clear_and_resize(self.num_rows.to_usize());
        }
        let mut rhs = ScatteredColumn::new(self.num_rows);
        for &position in columns {
            let column = ColIndex::from_usize(position);
            let sign = self.dual_phase_one_sign(column);
            let old_sign = self.dual_phase_one_improvement_direction[column];
            if sign == old_sign {
                continue;
            }
            self.num_update_price_operations +=
                10 * self.compact_matrix.column_num_entries(column).value();
            if sign == 0.0 {
                self.num_dual_infeasible_positions -= 1;
            } else if old_sign == 0.0 {
                self.num_dual_infeasible_positions += 1;
            }
            self.compact_matrix.column_add_multiple_to_scattered_column(
                column,
                sign - old_sign,
                &mut rhs,
            );
            self.dual_phase_one_improvement_direction[column] = sign;
        }
        if rhs.non_zeros().is_empty() {
            return Ok(());
        }
        rhs.clear_sparse_mask();
        rhs.clear_non_zeros_if_too_dense(0.8);
        self.basis_factorization
            .as_ref()
            .unwrap()
            .solve_with_nonzeros(&mut rhs)?;
        let norms = self
            .dual_edge_norms
            .edge_squared_norms(self.basis_factorization.as_ref().unwrap())?;
        let types = self.variables_info.as_ref().unwrap().variable_types();
        let threshold = self.parameters.ratio_test_zero_threshold;
        if rhs.non_zeros().is_empty() {
            self.dual_prices.start_dense_updates();
            for row in 0..self.num_rows.to_usize() {
                let row = RowIndex::from_usize(row);
                let delta = rhs.value(row);
                if delta == 0.0 {
                    continue;
                }
                self.dual_phase_one_pricing_vector[row] += delta;
                Self::update_dual_phase_one_price_at(
                    &mut self.dual_prices,
                    row,
                    self.dual_phase_one_pricing_vector[row],
                    norms[row.to_usize()],
                    types[self.basis[row]],
                    threshold,
                    true,
                );
            }
        } else {
            let rows = rhs.non_zeros().to_vec();
            for row in rows {
                self.dual_phase_one_pricing_vector[row] += rhs.value(row);
                Self::update_dual_phase_one_price_at(
                    &mut self.dual_prices,
                    row,
                    self.dual_phase_one_pricing_vector[row],
                    norms[row.to_usize()],
                    types[self.basis[row]],
                    threshold,
                    false,
                );
            }
        }
        Ok(())
    }

    fn update_dual_phase_one_prices_on_pivot(
        &mut self,
        leaving_row: RowIndex,
        entering: ColIndex,
        direction: &ScatteredColumn,
        reduced_costs_recomputed: bool,
    ) -> Result<(), FactorizationError> {
        // Keep this condition synchronized with the full-recomputation path at
        // the start of the next Phase-I leaving-row selection. GLOP skips this
        // eta update whenever the norms will force all prices to be rebuilt.
        if reduced_costs_recomputed
            || self.dual_edge_norms.needs_basis_refactorization()
            || self.dual_phase_one_pricing_vector.is_empty()
        {
            return Ok(());
        }
        let norms = self
            .dual_edge_norms
            .edge_squared_norms(self.basis_factorization.as_ref().unwrap())?;
        let types = self.variables_info.as_ref().unwrap().variable_types();
        let threshold = self.parameters.ratio_test_zero_threshold;
        let step = self.dual_phase_one_pricing_vector[leaving_row] / direction.value(leaving_row);
        for entry in direction {
            let row = entry.row();
            self.dual_phase_one_pricing_vector[row] =
                (-entry.coefficient()).mul_add(step, self.dual_phase_one_pricing_vector[row]);
            Self::update_dual_phase_one_price_at(
                &mut self.dual_prices,
                row,
                self.dual_phase_one_pricing_vector[row],
                norms[row.to_usize()],
                types[self.basis[row]],
                threshold,
                false,
            );
        }
        self.dual_phase_one_pricing_vector[leaving_row] = step;
        self.dual_phase_one_pricing_vector[leaving_row] -=
            self.dual_phase_one_improvement_direction[entering];
        if self.dual_phase_one_improvement_direction[entering] != 0.0 {
            self.num_dual_infeasible_positions -= 1;
        }
        self.dual_phase_one_improvement_direction[entering] = 0.0;
        self.dual_phase_one_improvement_direction[self.basis[leaving_row]] = 0.0;
        Self::update_dual_phase_one_price_at(
            &mut self.dual_prices,
            leaving_row,
            self.dual_phase_one_pricing_vector[leaving_row],
            norms[leaving_row.to_usize()],
            types[entering],
            threshold,
            false,
        );
        Ok(())
    }

    fn initialize(&mut self, lp: &LinearProgram) -> Result<(), FactorizationError> {
        self.recovered_warm_basis = None;
        let mut equation_lp = LinearProgram::default();
        equation_lp.populate_from_linear_program(lp);
        if !equation_lp.is_in_equation_form() {
            equation_lp.add_slack_variables_where_necessary(false);
        }
        if self.try_reuse_unchanged_matrix(&equation_lp)? {
            return Ok(());
        }
        if self.try_reuse_with_added_columns(&equation_lp) {
            return Ok(());
        }
        if self.try_reuse_with_added_rows(&equation_lp)? {
            return Ok(());
        }
        // The full initialization replaces these four owned components.
        // Preserve their cumulative operation time across Solve() calls.
        self.previous_component_deterministic_time += self.component_deterministic_time();
        // A saved basis places its slack statuses immediately after the old
        // structural columns. When columns are appended, those statuses must
        // move past the new columns, as in GLOP's incremental initialization.
        let new_first_slack = equation_lp
            .first_slack_variable()
            .unwrap_or(equation_lp.num_variables());
        let added_columns = if self.num_rows == equation_lp.num_constraints()
            && self.first_slack_col < new_first_slack
            && self.state_for_next_solve.as_ref() == Some(&self.solution_state)
            && !self.external_state_for_next_solve
            && self.solution_state.statuses.len() == self.num_cols
            && (0..self.first_slack_col.to_usize()).all(|column| {
                let column = ColIndex::from_usize(column);
                let old = self.matrix.column(column);
                let new = equation_lp.matrix().column(column);
                old.num_entries() == new.num_entries()
                    && old.iter().zip(new.iter()).all(|(left, right)| {
                        left.row() == right.row() && left.coefficient() == right.coefficient()
                    })
            }) {
            new_first_slack.to_usize() - self.first_slack_col.to_usize()
        } else {
            0
        };
        self.num_rows = equation_lp.num_constraints();
        self.num_cols = equation_lp.num_variables();
        self.first_slack_col = equation_lp
            .first_slack_variable()
            .unwrap_or(equation_lp.num_variables());
        self.is_maximization_problem = equation_lp.is_maximization_problem();
        self.matrix = Rc::new(equation_lp.matrix().clone());
        self.compact_matrix = CompactSparseMatrix::from_sparse(&self.matrix);
        self.objective = DenseRow::from_vec(
            (0..self.num_cols.to_usize())
                .map(|column| {
                    equation_lp.objective_coefficient_for_minimization(ColIndex::from_usize(column))
                })
                .collect(),
        );
        if equation_lp.is_maximization_problem() {
            self.objective_offset = -equation_lp.objective_offset();
            self.objective_scaling_factor = -equation_lp.objective_scaling_factor();
        } else {
            self.objective_offset = equation_lp.objective_offset();
            self.objective_scaling_factor = equation_lp.objective_scaling_factor();
        }
        let external_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_lower_limit
        } else {
            self.parameters.objective_upper_limit
        };
        self.primal_objective_limit =
            external_limit / self.objective_scaling_factor - self.objective_offset;
        let external_dual_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_upper_limit
        } else {
            self.parameters.objective_lower_limit
        };
        self.dual_objective_limit =
            external_dual_limit / self.objective_scaling_factor - self.objective_offset;
        self.objective_limit_reached = false;

        let mut info = VariablesInfo::new(&self.matrix);
        info.load_bounds_and_return_true_if_unchanged(
            equation_lp.variable_lower_bounds().as_slice(),
            equation_lp.variable_upper_bounds().as_slice(),
        );
        let has_external_basis = self
            .state_for_next_solve
            .as_ref()
            .is_some_and(|state| !state.is_empty());
        if has_external_basis {
            self.phase4_event("warm_start_basis");
        }
        if let Some(state) = self
            .state_for_next_solve
            .take()
            .filter(|state| !state.is_empty())
        {
            if added_columns != 0 {
                self.phase4_event("warm_added_column_status_remap");
            }
            info.initialize_from_basis_state(
                self.first_slack_col.to_usize(),
                added_columns,
                &state,
            );
        } else {
            info.initialize_to_default_status();
        }

        let number_of_basic_candidates = info.is_basic().iter_ones().count();
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
            match (
                self.parameters.initial_basis,
                self.parameters.use_dual_simplex,
            ) {
                (InitialBasisHeuristic::Triangular, true) => {
                    self.phase4_event("initial_triangular_dual");
                }
                (InitialBasisHeuristic::Triangular, false) => {
                    self.phase4_event("initial_triangular_primal");
                }
                (InitialBasisHeuristic::Maros, true) => self.phase4_event("initial_maros_dual"),
                (InitialBasisHeuristic::Maros, false) => self.phase4_event("initial_maros_primal"),
                (InitialBasisHeuristic::Bixby, _) if !self.parameters.use_scaling => {
                    self.phase4_event("initial_bixby_skipped_without_scaling");
                }
                _ => {}
            }
            for row in 0..crashed.len().to_usize() {
                let row = RowIndex::from_usize(row);
                if crashed[row] == INVALID_COL {
                    crashed[row] = ColIndex::new(self.first_slack_col.value() + row.value());
                }
            }
            basis = crashed.as_slice().to_vec();
        }
        let had_recovered_warm_basis = self.recovered_warm_basis.is_some();
        if let Some(recovered) = self.recovered_warm_basis.take() {
            self.phase4_event("warm_added_row_recovered_basis_used");
            basis = recovered.as_slice().to_vec();
        }
        // FinishInitialization() uses a saved state as candidate columns.
        // Unless it supplies exactly one BASIC candidate per row, upstream
        // first runs Markowitz's ComputeInitialBasis(): this discards surplus
        // candidates and fills missing rows with slacks before factorization.
        let mut candidate_factorization = None;
        if has_external_basis
            && !had_recovered_warm_basis
            && number_of_basic_candidates != basis.len()
        {
            let slack_basis = RowToColMapping::from_vec(
                (0..self.num_rows.to_usize())
                    .map(|row| ColIndex::from_usize(self.first_slack_col.to_usize() + row))
                    .collect(),
            );
            let mut factorization = BasisRepresentation::new_for_basis(
                Rc::clone(&self.matrix),
                &slack_basis,
                &self.parameters,
            )?;
            let candidates: Vec<_> = info.is_basic().iter_ones().collect();
            let recovered = factorization.compute_initial_basis(&candidates)?;
            basis = recovered.as_slice().to_vec();
            candidate_factorization = Some(factorization);
            self.phase4_event("warm_changed_matrix_compute_initial_basis");
        }
        self.basis = RowToColMapping::from_vec(basis);
        if self.trace_enabled {
            self.initial_basis_before_permutation = self.basis.clone();
        }
        let triangular_crash_can_fall_back = !has_external_basis
            && self.parameters.initial_basis == InitialBasisHeuristic::Triangular;
        let mut retried_rejected_saved_basis = false;
        let mut basis_factorization = if let Some(mut factorization) = candidate_factorization {
            factorization.reinitialize_for_basis(
                Rc::clone(&self.matrix),
                &self.basis,
                &self.parameters,
            )?;
            factorization
        } else {
            match BasisRepresentation::new_for_basis(
                Rc::clone(&self.matrix),
                &self.basis,
                &self.parameters,
            ) {
                Ok(factorization) => factorization,
                Err(_) if triangular_crash_can_fall_back => {
                    // CreateInitialBasis() immediately tests TRIANGULAR's proposed
                    // basis upstream and reverts to the all-slack basis when that
                    // advanced crash is not factorizable.
                    self.use_all_slack_basis();
                    self.phase4_event("initial_triangular_basis_fallback");
                    BasisRepresentation::new_for_basis(
                        Rc::clone(&self.matrix),
                        &self.basis,
                        &self.parameters,
                    )?
                }
                Err(_) if has_external_basis => {
                    // FinishInitialization() retries a rejected complete
                    // saved basis with its BASIC columns as Markowitz hints.
                    let slack_basis = RowToColMapping::from_vec(
                        (0..self.num_rows.to_usize())
                            .map(|row| ColIndex::from_usize(self.first_slack_col.to_usize() + row))
                            .collect(),
                    );
                    let mut factorization = BasisRepresentation::new_for_basis(
                        Rc::clone(&self.matrix),
                        &slack_basis,
                        &self.parameters,
                    )?;
                    let candidates: Vec<_> = info.is_basic().iter_ones().collect();
                    self.basis = factorization.compute_initial_basis(&candidates)?;
                    retried_rejected_saved_basis = true;
                    factorization.reinitialize_for_basis(
                        Rc::clone(&self.matrix),
                        &self.basis,
                        &self.parameters,
                    )?;
                    self.phase4_event("warm_rejected_basis_compute_initial_basis");
                    factorization
                }
                Err(error) => return Err(error),
            }
        };
        // InitializeFirstBasis() calls PermuteBasis() before checking the
        // condition number. The permuted mapping may itself be the identity
        // even when its pre-permutation LU was not.
        self.absorb_initial_column_permutation(&mut basis_factorization);
        let mut condition_rejected = basis_factorization
            .infinity_norm_condition_number_upper_bound()
            > self.parameters.initial_condition_number_threshold;
        if condition_rejected
            && has_external_basis
            && !had_recovered_warm_basis
            && !retried_rejected_saved_basis
            && number_of_basic_candidates == self.num_rows.to_usize()
        {
            // FinishInitialization() gives a complete but rejected saved
            // basis one more chance: Markowitz uses its BASIC columns as
            // candidates before CreateInitialBasis() starts from scratch.
            let candidates: Vec<_> = info.is_basic().iter_ones().collect();
            self.basis = basis_factorization.compute_initial_basis(&candidates)?;
            self.phase4_event("warm_saved_basis_condition_candidates");
            condition_rejected = if basis_factorization
                .reinitialize_for_basis(Rc::clone(&self.matrix), &self.basis, &self.parameters)
                .is_ok()
            {
                self.absorb_initial_column_permutation(&mut basis_factorization);
                basis_factorization.infinity_norm_condition_number_upper_bound()
                    > self.parameters.initial_condition_number_threshold
            } else {
                true
            };
        }
        if condition_rejected {
            if has_external_basis {
                self.phase4_event("warm_saved_basis_condition_fallback");
            }
            self.use_all_slack_basis();
            if triangular_crash_can_fall_back {
                self.phase4_event("initial_triangular_basis_fallback");
            }
            basis_factorization.reinitialize_for_basis(
                Rc::clone(&self.matrix),
                &self.basis,
                &self.parameters,
            )?;
            self.absorb_initial_column_permutation(&mut basis_factorization);
            let upper_bound = basis_factorization.infinity_norm_condition_number_upper_bound();
            if upper_bound > self.parameters.initial_condition_number_threshold {
                self.phase4_event("initial_all_slack_condition_failure");
                return Err(FactorizationError::IllConditioned { upper_bound });
            }
        }
        let info_before_advanced_basis = info.clone();
        let unused_basic_variables = info.change_unused_basic_variables_to_free(&self.basis);
        if unused_basic_variables != 0
            && info.snap_free_variables_to_bound(
                self.parameters.crossover_bound_snapping_distance,
                self.starting_values.as_slice(),
            ) != 0
        {
            self.phase4_event("warm_superbasic_snapped_to_bound");
        }
        let variable_values = match self.compute_initial_values(&info, &basis_factorization) {
            Ok(values) => values,
            Err(_) if triangular_crash_can_fall_back => {
                // InitializeFirstBasis() also recomputes the basic values. A
                // numerical failure in that solve rejects TRIANGULAR's crash
                // just like a factorization or condition-number failure.
                self.use_all_slack_basis();
                self.phase4_event("initial_triangular_basis_fallback");
                basis_factorization.reinitialize_for_basis(
                    Rc::clone(&self.matrix),
                    &self.basis,
                    &self.parameters,
                )?;
                info = info_before_advanced_basis;
                info.change_unused_basic_variables_to_free(&self.basis);
                self.compute_initial_values(&info, &basis_factorization)?
            }
            Err(error) => return Err(error),
        };
        self.basis_factorization = Some(basis_factorization);
        self.variables_info = Some(info);
        self.variable_values = variable_values;
        self.reduced_costs = DenseRow::filled(self.num_cols, 0.0);
        self.reduced_costs_dirty = true;
        self.reduced_costs_precise = false;
        self.cost_perturbations = DenseRow::filled(self.num_cols, 0.0);
        self.has_cost_shift = false;
        self.dual_values = DenseColumn::filled(self.num_rows, 0.0);
        let mut primal_edge_norms = PrimalEdgeNorms::new(&self.matrix);
        primal_edge_norms.set_glop_parameters(&self.parameters);
        self.primal_edge_norms = Some(primal_edge_norms);
        self.primal_prices = PrimalPrices::new_with_random(self.random.clone());
        self.dual_edge_norms.set_glop_parameters(&self.parameters);
        self.dual_edge_norms
            .resize_on_new_rows(self.num_rows.to_usize());
        self.dual_edge_norms.clear();
        self.dual_prices = DynamicMaximum::new_with_random(self.random.clone());
        self.entering_variable = EnteringVariable::new_with_random(self.random.clone());
        self.entering_variable.set_parameters(&self.parameters);
        self.bound_flip_candidates.clear();
        let mut update_row = UpdateRow::new(&self.matrix);
        update_row.set_glop_parameters(&self.parameters);
        self.update_row = Some(update_row);
        self.primal_ray = DenseRow::new();
        self.dual_ray = DenseColumn::new();
        self.dual_ray_row_combination = DenseRow::new();
        self.problem_status = ProblemStatus::Init;
        Ok(())
    }

    fn absorb_initial_column_permutation(&mut self, factorization: &mut BasisRepresentation) {
        if self.trace_enabled {
            self.initial_basis_before_permutation = self.basis.clone();
            self.initial_column_permutation = factorization.column_permutation().to_vec();
        }
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

    /// Native `RevisedSimplex::Initialize()` keeps its LU when a primal warm
    /// start only appends zero-bound structural variables.
    fn try_reuse_with_added_columns(&mut self, equation_lp: &LinearProgram) -> bool {
        let new_first_slack = equation_lp
            .first_slack_variable()
            .unwrap_or(equation_lp.num_variables());
        let old_first_slack = self.first_slack_col.to_usize();
        let new_first_slack = new_first_slack.to_usize();
        if self.parameters.use_dual_simplex
            || self.basis_factorization.is_none()
            || self.num_rows != equation_lp.num_constraints()
            || new_first_slack <= old_first_slack
            || self.is_maximization_problem != equation_lp.is_maximization_problem()
            || self.state_for_next_solve.as_ref() != Some(&self.solution_state)
            || self.external_state_for_next_solve
            || self.solution_state.is_empty()
            || !self.starting_values.is_empty()
        {
            return false;
        }
        let added = new_first_slack - old_first_slack;
        if self.num_cols.to_usize() + added != equation_lp.num_variables().to_usize() {
            return false;
        }
        let old_info = self.variables_info.as_ref().unwrap();
        let lower = equation_lp.variable_lower_bounds().as_slice();
        let upper = equation_lp.variable_upper_bounds().as_slice();
        if (0..old_first_slack).any(|col| {
            old_info.lower_bounds()[col] != lower[col] || old_info.upper_bounds()[col] != upper[col]
        }) || (old_first_slack..new_first_slack)
            .any(|col| lower[col] != 0.0 && upper[col] != 0.0)
            || (old_first_slack..self.num_cols.to_usize()).any(|col| {
                old_info.lower_bounds()[col] != lower[col + added]
                    || old_info.upper_bounds()[col] != upper[col + added]
            })
        {
            return false;
        }
        for col in 0..old_first_slack {
            let index = ColIndex::from_usize(col);
            let old = self.matrix.column(index);
            let new = equation_lp.matrix().column(index);
            if old.num_entries() != new.num_entries()
                || !old
                    .iter()
                    .zip(new.iter())
                    .all(|(a, b)| a.row() == b.row() && a.coefficient() == b.coefficient())
            {
                return false;
            }
        }
        let state = self.state_for_next_solve.take().unwrap();
        self.phase4_event("warm_start_basis");
        self.phase4_event("warm_added_column_status_remap");
        self.phase4_event("warm_added_column_reuse_factorization");
        self.phase4_event("warm_primal_reuse_factorization");
        self.dual_edge_norms.clear();
        self.dual_phase_one_pricing_vector.clear();
        self.num_cols = equation_lp.num_variables();
        self.first_slack_col = ColIndex::from_usize(new_first_slack);
        self.matrix = Rc::new(equation_lp.matrix().clone());
        self.compact_matrix = CompactSparseMatrix::from_sparse(&self.matrix);
        self.objective = DenseRow::from_vec(
            (0..self.num_cols.to_usize())
                .map(|col| {
                    equation_lp.objective_coefficient_for_minimization(ColIndex::from_usize(col))
                })
                .collect(),
        );
        if equation_lp.is_maximization_problem() {
            self.objective_offset = -equation_lp.objective_offset();
            self.objective_scaling_factor = -equation_lp.objective_scaling_factor();
        } else {
            self.objective_offset = equation_lp.objective_offset();
            self.objective_scaling_factor = equation_lp.objective_scaling_factor();
        }
        let primal_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_lower_limit
        } else {
            self.parameters.objective_upper_limit
        };
        self.primal_objective_limit =
            primal_limit / self.objective_scaling_factor - self.objective_offset;
        let dual_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_upper_limit
        } else {
            self.parameters.objective_lower_limit
        };
        self.dual_objective_limit =
            dual_limit / self.objective_scaling_factor - self.objective_offset;
        let mut info = VariablesInfo::new(&self.matrix);
        info.load_bounds_and_return_true_if_unchanged(lower, upper);
        info.initialize_from_basis_state(new_first_slack, added, &state);
        self.variables_info = Some(info);
        let old_values = self.variable_values.as_slice();
        let mut values = vec![0.0; self.num_cols.to_usize()];
        values[..old_first_slack].copy_from_slice(&old_values[..old_first_slack]);
        values[new_first_slack..].copy_from_slice(&old_values[old_first_slack..]);
        self.variable_values = DenseRow::from_vec(values);
        for row in 0..self.basis.len().to_usize() {
            let index = RowIndex::from_usize(row);
            if self.basis[index].to_usize() >= old_first_slack {
                self.basis[index] = ColIndex::from_usize(self.basis[index].to_usize() + added);
            }
        }
        self.basis_factorization
            .as_mut()
            .unwrap()
            .rebind_after_added_columns(Rc::clone(&self.matrix), old_first_slack, added);
        self.primal_edge_norms
            .as_mut()
            .unwrap()
            .rebind_matrix(&self.matrix);
        self.update_row
            .as_mut()
            .unwrap()
            .rebind_matrix(&self.matrix);
        self.reduced_costs = DenseRow::filled(self.num_cols, 0.0);
        self.reduced_costs_dirty = true;
        self.reduced_costs_precise = false;
        self.cost_perturbations = DenseRow::filled(self.num_cols, 0.0);
        self.has_cost_shift = false;
        self.primal_ray = DenseRow::new();
        self.dual_ray = DenseColumn::new();
        self.dual_ray_row_combination = DenseRow::new();
        self.problem_status = ProblemStatus::Init;
        self.objective_limit_reached = false;
        true
    }

    /// Dual warm start after appending constraints: retain the old basis
    /// columns, give each new row its slack, then refactorize that basis.
    fn try_reuse_with_added_rows(
        &mut self,
        equation_lp: &LinearProgram,
    ) -> Result<bool, FactorizationError> {
        let first_slack = equation_lp
            .first_slack_variable()
            .unwrap_or(equation_lp.num_variables());
        if !self.parameters.use_dual_simplex
            || self.basis_factorization.is_none()
            || equation_lp.num_constraints().to_usize() <= self.num_rows.to_usize()
            || first_slack != self.first_slack_col
            || self.is_maximization_problem != equation_lp.is_maximization_problem()
            || self.state_for_next_solve.as_ref() != Some(&self.solution_state)
            || self.external_state_for_next_solve
            || self.solution_state.is_empty()
            || !self.starting_values.is_empty()
            || self.objective_offset
                != if equation_lp.is_maximization_problem() {
                    -equation_lp.objective_offset()
                } else {
                    equation_lp.objective_offset()
                }
            || self.objective_scaling_factor
                != if equation_lp.is_maximization_problem() {
                    -equation_lp.objective_scaling_factor()
                } else {
                    equation_lp.objective_scaling_factor()
                }
        {
            return Ok(false);
        }
        let old_rows = self.num_rows.to_usize();
        for col in 0..self.first_slack_col.to_usize() {
            let index = ColIndex::from_usize(col);
            if self.objective[index] != equation_lp.objective_coefficient_for_minimization(index) {
                return Ok(false);
            }
            let old = self.matrix.column(index);
            let new = equation_lp.matrix().column(index);
            let mut new_old_rows = new.iter().filter(|entry| entry.row().to_usize() < old_rows);
            for entry in old {
                let Some(other) = new_old_rows.next() else {
                    return Ok(false);
                };
                if entry.row() != other.row() || entry.coefficient() != other.coefficient() {
                    return Ok(false);
                }
            }
            if new_old_rows.next().is_some() {
                return Ok(false);
            }
        }
        self.phase4_event("warm_start_basis");
        self.phase4_event("warm_added_row_reinitialize_basis");
        let state = self.state_for_next_solve.take().unwrap();
        self.num_rows = equation_lp.num_constraints();
        self.num_cols = equation_lp.num_variables();
        self.matrix = Rc::new(equation_lp.matrix().clone());
        self.compact_matrix = CompactSparseMatrix::from_sparse(&self.matrix);
        self.objective = DenseRow::from_vec(
            (0..self.num_cols.to_usize())
                .map(|col| {
                    equation_lp.objective_coefficient_for_minimization(ColIndex::from_usize(col))
                })
                .collect(),
        );
        let primal_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_lower_limit
        } else {
            self.parameters.objective_upper_limit
        };
        self.primal_objective_limit =
            primal_limit / self.objective_scaling_factor - self.objective_offset;
        let dual_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_upper_limit
        } else {
            self.parameters.objective_lower_limit
        };
        self.dual_objective_limit =
            dual_limit / self.objective_scaling_factor - self.objective_offset;
        let mut info = VariablesInfo::new(&self.matrix);
        info.load_bounds_and_return_true_if_unchanged(
            equation_lp.variable_lower_bounds().as_slice(),
            equation_lp.variable_upper_bounds().as_slice(),
        );
        info.initialize_from_basis_state(self.first_slack_col.to_usize(), 0, &state);
        let mut basis = self.basis.as_slice().to_vec();
        for row in old_rows..self.num_rows.to_usize() {
            basis.push(ColIndex::from_usize(self.first_slack_col.to_usize() + row));
        }
        self.basis = RowToColMapping::from_vec(basis);
        self.dual_edge_norms
            .resize_on_new_rows(self.num_rows.to_usize());
        self.reduced_costs = DenseRow::filled(self.num_cols, 0.0);
        self.reduced_costs_dirty = true;
        self.reduced_costs_precise = false;
        self.cost_perturbations = DenseRow::filled(self.num_cols, 0.0);
        self.has_cost_shift = false;
        self.dual_phase_one_pricing_vector.clear();
        if self
            .basis_factorization
            .as_mut()
            .unwrap()
            .reinitialize_for_basis(Rc::clone(&self.matrix), &self.basis, &self.parameters)
            .is_err()
        {
            self.phase4_event("warm_added_row_basis_fallback");
            self.basis_factorization.as_mut().unwrap().clear();
            let candidates: Vec<_> = info.is_basic().iter_ones().collect();
            let recovered = self
                .basis_factorization
                .as_mut()
                .unwrap()
                .compute_initial_basis(&candidates)?;
            self.recovered_warm_basis = Some(recovered);
            self.phase4_event("warm_added_row_compute_initial_basis");
            self.state_for_next_solve = Some(state);
            return Ok(false);
        }
        let factorization = self.basis_factorization.as_mut().unwrap();
        if !factorization.column_permutation().is_empty() {
            let mut permuted = self.basis.clone();
            for (source, &destination) in factorization.column_permutation().iter().enumerate() {
                permuted[RowIndex::from_usize(destination)] =
                    self.basis[RowIndex::from_usize(source)];
            }
            self.basis = permuted;
            factorization.set_column_permutation_to_identity();
        }
        if factorization.infinity_norm_condition_number_upper_bound()
            > self.parameters.initial_condition_number_threshold
        {
            factorization.clear();
            let candidates: Vec<_> = info.is_basic().iter_ones().collect();
            let recovered = factorization.compute_initial_basis(&candidates)?;
            self.recovered_warm_basis = Some(recovered);
            self.phase4_event("warm_added_row_basis_fallback");
            self.phase4_event("warm_added_row_compute_initial_basis");
            self.state_for_next_solve = Some(state);
            return Ok(false);
        }
        for &column in self.basis.as_slice() {
            info.update_to_basic_status(column);
        }
        self.variable_values =
            self.compute_initial_values(&info, self.basis_factorization.as_ref().unwrap())?;
        self.variables_info = Some(info);
        self.update_row
            .as_mut()
            .unwrap()
            .rebind_matrix(&self.matrix);
        self.primal_edge_norms
            .as_mut()
            .unwrap()
            .rebind_matrix(&self.matrix);
        self.dual_values = DenseColumn::filled(self.num_rows, 0.0);
        self.primal_ray = DenseRow::new();
        self.dual_ray = DenseColumn::new();
        self.dual_ray_row_combination = DenseRow::new();
        self.problem_status = ProblemStatus::Init;
        self.objective_limit_reached = false;
        Ok(true)
    }

    /// GLOP's quick warm starts for an unchanged matrix: primal simplex can
    /// reuse the factorization when bounds are unchanged (even if the
    /// objective changed), while dual simplex can do so when the objective is
    /// unchanged (even if bounds changed).
    fn try_reuse_unchanged_matrix(
        &mut self,
        equation_lp: &LinearProgram,
    ) -> Result<bool, FactorizationError> {
        if self.basis_factorization.is_none()
            || self.num_rows != equation_lp.num_constraints()
            || self.num_cols != equation_lp.num_variables()
            || self.is_maximization_problem != equation_lp.is_maximization_problem()
            || self.state_for_next_solve.as_ref() != Some(&self.solution_state)
            || self.external_state_for_next_solve
            || self.solution_state.is_empty()
            || !self.starting_values.is_empty()
        {
            return Ok(false);
        }
        let new_offset = if equation_lp.is_maximization_problem() {
            -equation_lp.objective_offset()
        } else {
            equation_lp.objective_offset()
        };
        let new_scale = if equation_lp.is_maximization_problem() {
            -equation_lp.objective_scaling_factor()
        } else {
            equation_lp.objective_scaling_factor()
        };
        if self.objective_offset != new_offset || self.objective_scaling_factor != new_scale {
            return Ok(false);
        }
        for column in 0..self.num_cols.to_usize() {
            let index = ColIndex::from_usize(column);
            if self.parameters.use_dual_simplex
                && self.objective[index]
                    != equation_lp.objective_coefficient_for_minimization(index)
            {
                return Ok(false);
            }
            let old = self.matrix.column(index);
            let new = equation_lp.matrix().column(index);
            if old.num_entries() != new.num_entries()
                || !old.iter().zip(new.iter()).all(|(left, right)| {
                    left.row() == right.row() && left.coefficient() == right.coefficient()
                })
            {
                return Ok(false);
            }
        }
        let info = self.variables_info.as_mut().unwrap();
        let bounds_unchanged = info.load_bounds_and_return_true_if_unchanged(
            equation_lp.variable_lower_bounds().as_slice(),
            equation_lp.variable_upper_bounds().as_slice(),
        );
        if !self.parameters.use_dual_simplex && !bounds_unchanged {
            return Ok(false);
        }
        if self.parameters.use_dual_simplex {
            self.phase4_event("warm_reuse_factorization");
        } else {
            self.phase4_event("warm_start_basis");
            self.phase4_event("warm_primal_reuse_factorization");
            self.dual_edge_norms.clear();
            self.dual_phase_one_pricing_vector.clear();
            for column in 0..self.num_cols.to_usize() {
                let index = ColIndex::from_usize(column);
                self.objective[index] = equation_lp.objective_coefficient_for_minimization(index);
            }
        }
        // InitializeObjectiveLimit() runs on every native solve, including a
        // quick warm start; a caller may change these parameters between solves.
        let external_primal_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_lower_limit
        } else {
            self.parameters.objective_upper_limit
        };
        self.primal_objective_limit =
            external_primal_limit / self.objective_scaling_factor - self.objective_offset;
        let external_dual_limit = if self.objective_scaling_factor >= 0.0 {
            self.parameters.objective_upper_limit
        } else {
            self.parameters.objective_lower_limit
        };
        self.dual_objective_limit =
            external_dual_limit / self.objective_scaling_factor - self.objective_offset;
        let state = self.state_for_next_solve.take().unwrap();
        if !bounds_unchanged {
            self.variables_info
                .as_mut()
                .unwrap()
                .initialize_from_basis_state(self.first_slack_col.to_usize(), 0, &state);
            self.variable_values = self.compute_initial_values(
                self.variables_info.as_ref().unwrap(),
                self.basis_factorization.as_ref().unwrap(),
            )?;
        }
        if self.parameters.use_dual_simplex {
            self.primal_edge_norms.as_mut().unwrap().clear();
        }
        self.update_row.as_mut().unwrap().invalidate();
        self.cost_perturbations.as_mut_slice().fill(0.0);
        self.has_cost_shift = false;
        self.primal_ray = DenseRow::new();
        self.dual_ray = DenseColumn::new();
        self.dual_ray_row_combination = DenseRow::new();
        self.problem_status = ProblemStatus::Init;
        self.objective_limit_reached = false;
        Ok(true)
    }

    fn use_all_slack_basis(&mut self) {
        self.basis = RowToColMapping::from_vec(
            (0..self.num_rows.to_usize())
                .map(|row| {
                    ColIndex::new(self.first_slack_col.value() + i32::try_from(row).unwrap())
                })
                .collect(),
        );
    }

    fn strengthen_lu_pivoting_after_early_imprecision(&mut self) -> Result<(), FactorizationError> {
        if self.basis_factorization.as_ref().unwrap().num_updates() < 10 {
            self.phase4_event("adaptive_lu_pivot_threshold");
            self.parameters.lu_factorization_pivot_threshold =
                (1.5 * self.parameters.lu_factorization_pivot_threshold).min(0.9);
            self.basis_factorization
                .as_mut()
                .unwrap()
                .set_parameters(&self.parameters)?;
        }
        Ok(())
    }

    fn initialize_values(&mut self) -> Result<(), FactorizationError> {
        let info = self.variables_info.as_ref().unwrap();
        self.variable_values =
            self.compute_initial_values(info, self.basis_factorization.as_ref().unwrap())?;
        Ok(())
    }

    fn compute_initial_values(
        &self,
        info: &VariablesInfo,
        basis_factorization: &BasisRepresentation,
    ) -> Result<DenseRow, FactorizationError> {
        let mut variable_values = DenseRow::filled(self.num_cols, 0.0);
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
            variable_values[index] = value;
            for entry in self.matrix.column(index) {
                let row = entry.index().to_usize();
                rhs[row] = (-entry.coefficient()).mul_add(value, rhs[row]);
            }
        }
        // VariableValues::RecomputeBasicVariableValues() passes a scattered
        // workspace to RightSolve(). Its positions start empty even though
        // the dense values contain the accumulated nonbasic contribution.
        // The solve's output density, not the matrix dimension, determines
        // GLOP's deterministic-time charge.
        let mut basic = ScatteredColumn::new(self.num_rows);
        basic.values_mut().as_mut_slice().copy_from_slice(&rhs);
        basis_factorization.solve_with_nonzeros(&mut basic)?;
        for row in 0..self.num_rows.to_usize() {
            let row = RowIndex::from_usize(row);
            variable_values[self.basis[row]] = basic.value(row);
        }
        Ok(variable_values)
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

    /// Update only the basic Phase-I costs in `rows`, as in
    /// `VariableValues::UpdatePrimalPhaseICosts()`. In particular, a cost on
    /// a variable that left the basis is not cleared here.
    fn update_primal_phase_one_costs(&self, rows: &[RowIndex], objective: &mut DenseRow) -> bool {
        let info = self.variables_info.as_ref().unwrap();
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let mut changed = false;
        for &row in rows {
            let column = self.basis[row];
            let value = self.variable_values[column];
            let cost = if value - info.upper_bounds()[column.to_usize()] > tolerance {
                1.0
            } else if info.lower_bounds()[column.to_usize()] - value > tolerance {
                -1.0
            } else {
                0.0
            };
            if cost != objective[column] {
                objective[column] = cost;
                changed = true;
            }
        }
        changed
    }

    fn compute_reduced_costs(&mut self, objective: &DenseRow) -> Result<(), FactorizationError> {
        let basic_objective: Vec<_> = (0..self.num_rows.to_usize())
            .map(|row| {
                let column = self.basis[RowIndex::from_usize(row)];
                objective[column] + self.cost_perturbations[column]
            })
            .collect();
        let mut dual = lp_data::scattered_vector::ScatteredRow::new(ColIndex::from_usize(
            self.num_rows.to_usize(),
        ));
        dual.values_mut()
            .as_mut_slice()
            .copy_from_slice(&basic_objective);
        self.basis_factorization
            .as_ref()
            .unwrap()
            .transpose_solve_with_nonzeros(&mut dual)?;
        self.dual_values = DenseColumn::from_vec(dual.values().as_slice().to_vec());
        self.compute_reduced_costs_from_dual(objective);
        Ok(())
    }

    /// Recompute reduced costs from the left inverse already held in
    /// `dual_values`, as native `GetReducedCosts()` does after `GetDualValues()`.
    fn compute_reduced_costs_from_dual(&mut self, objective: &DenseRow) {
        let dual_row = self.dual_values.as_slice();
        let first_slack = self.num_cols.to_usize() - self.num_rows.to_usize();
        for column in 0..first_slack {
            let index = ColIndex::from_usize(column);
            self.reduced_costs[index] = objective[index] + self.cost_perturbations[index]
                - self
                    .compact_matrix
                    .column_scalar_product_slice(index, dual_row);
        }
        for column in first_slack..self.num_cols.to_usize() {
            let index = ColIndex::from_usize(column);
            self.reduced_costs[index] =
                objective[index] + self.cost_perturbations[index] - dual_row[column - first_slack];
        }
        self.reduced_costs_deterministic_time +=
            lp_data::lp_types::deterministic_time_for_fp_operations(
                self.compact_matrix.num_entries().value(),
            );
        self.reduced_costs_dirty = false;
        self.reduced_costs_precise = self.basis_factorization.as_ref().unwrap().is_refactorized();
    }

    // ReducedCosts::GetDualValues() always performs a fresh BTRAN, even when
    // GetReducedCosts() can return its cached row without a matrix pass.
    fn refresh_dual_values(&mut self, objective: &DenseRow) -> Result<(), FactorizationError> {
        let basic_objective: Vec<_> = (0..self.num_rows.to_usize())
            .map(|row| {
                let column = self.basis[RowIndex::from_usize(row)];
                objective[column] + self.cost_perturbations[column]
            })
            .collect();
        let mut dual = lp_data::scattered_vector::ScatteredRow::new(ColIndex::from_usize(
            self.num_rows.to_usize(),
        ));
        dual.values_mut()
            .as_mut_slice()
            .copy_from_slice(&basic_objective);
        self.basis_factorization
            .as_ref()
            .unwrap()
            .transpose_solve_with_nonzeros(&mut dual)?;
        self.dual_values = DenseColumn::from_vec(dual.values().as_slice().to_vec());
        Ok(())
    }

    fn shift_cost_if_needed(&mut self, increasing_reduced_cost_needed: bool, column: ColIndex) {
        let minimum_delta =
            self.parameters.degenerate_ministep_factor * self.parameters.dual_feasibility_tolerance;
        let value = self.reduced_costs[column];
        if increasing_reduced_cost_needed && value <= -minimum_delta
            || !increasing_reduced_cost_needed && value >= minimum_delta
        {
            return;
        }
        let delta = if increasing_reduced_cost_needed {
            minimum_delta
        } else {
            -minimum_delta
        };
        self.cost_perturbations[column] -= value + delta;
        self.reduced_costs[column] = -delta;
        self.has_cost_shift = true;
        self.phase4_event("dual_cost_shift");
    }

    fn perturb_costs(&mut self) {
        perturb_costs_into(
            &self.objective,
            self.variables_info.as_ref().unwrap(),
            self.first_slack_col.to_usize(),
            &self.parameters,
            &self.random,
            self.cost_perturbations.as_mut_slice(),
        );
    }

    fn remove_cost_shifts(&mut self) {
        if self.has_cost_shift {
            self.phase4_event("dual_cost_shift_removed");
        } else if self.phase4_events_enabled
            && self
                .cost_perturbations
                .as_slice()
                .iter()
                .any(|value| *value != 0.0)
        {
            self.phase4_event("dual_cost_perturbation_removed");
        }
        self.cost_perturbations.as_mut_slice().fill(0.0);
        self.has_cost_shift = false;
    }

    /// GLOP's `PrimalPrices::RecomputePriceAt()` returns before requesting
    /// edge norms when a full price pass is pending. Preserve that laziness:
    /// exact norm recomputation requires a refactorized basis.
    fn recompute_primal_price_at(&mut self, column: ColIndex) -> Result<(), FactorizationError> {
        if self.primal_prices.recomputation_pending() {
            return Ok(());
        }
        let norms = self.primal_edge_norms.as_mut().unwrap().squared_norms(
            self.basis_factorization.as_ref().unwrap(),
            self.variables_info.as_ref().unwrap().relevance(),
        )?;
        self.primal_prices.recompute_price_at_from_values(
            column,
            self.variables_info.as_ref().unwrap(),
            self.reduced_costs.as_slice(),
            norms,
            self.parameters.dual_feasibility_tolerance,
        );
        Ok(())
    }

    fn choose_entering(&mut self) -> Result<Option<ColIndex>, FactorizationError> {
        let info = self.variables_info.as_ref().unwrap();
        let norms = self
            .primal_edge_norms
            .as_mut()
            .unwrap()
            .squared_norms(self.basis_factorization.as_ref().unwrap(), info.relevance())?;
        Ok(self.primal_prices.best_entering_column_from_values(
            info,
            self.reduced_costs.as_slice(),
            norms,
            self.parameters.dual_feasibility_tolerance,
        ))
    }

    fn direction(
        &self,
        entering: ColIndex,
        direction: &mut ScatteredColumn,
    ) -> Result<(), FactorizationError> {
        self.basis_factorization
            .as_ref()
            .unwrap()
            .right_solve_for_problem_column(
                entering.to_usize(),
                self.matrix.column(entering),
                direction,
            )?;
        // An empty position list is GLOP's dense-vector sentinel.  The ratio
        // kernels consume an explicit sparse traversal, so materialize its
        // support here when the selected basis-update representation took the
        // dense solve path.
        if direction.non_zeros().is_empty() {
            let (values, non_zeros) = direction.mutable_parts();
            for (row, &value) in values.iter().enumerate() {
                if value != 0.0 {
                    non_zeros.push(RowIndex::from_usize(row));
                }
            }
        }
        #[cfg(debug_assertions)]
        {
            let mut residual = vec![0.0; self.num_rows.to_usize()];
            for row in 0..self.num_rows.to_usize() {
                let coefficient = direction.value(RowIndex::from_usize(row));
                for entry in self.matrix.column(self.basis[RowIndex::from_usize(row)]) {
                    let residual_row = entry.index().to_usize();
                    residual[residual_row] = entry
                        .coefficient()
                        .mul_add(coefficient, residual[residual_row]);
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
        Ok(())
    }

    fn run_primal_phase(
        &mut self,
        phase: SimplexPhase,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        let result = self.run_primal_phase_internal(phase, time_limit);
        self.advance_deterministic_time(time_limit);
        result
    }

    fn run_primal_phase_internal(
        &mut self,
        phase: SimplexPhase,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        if phase == SimplexPhase::Feasibility {
            self.phase4_event("primal_phase_one");
        }
        let pricing_rule = match if phase == SimplexPhase::Feasibility {
            self.parameters.feasibility_rule
        } else {
            self.parameters.optimization_rule
        } {
            crate::parameters::PricingRule::Dantzig => EdgePricingRule::Dantzig,
            crate::parameters::PricingRule::SteepestEdge => EdgePricingRule::SteepestEdge,
            crate::parameters::PricingRule::Devex => EdgePricingRule::Devex,
        };
        if phase == SimplexPhase::Optimization {
            match pricing_rule {
                EdgePricingRule::SteepestEdge => self.phase4_event("primal_steepest_edge_pricing"),
                EdgePricingRule::Devex => self.phase4_event("primal_devex_pricing"),
                EdgePricingRule::Dantzig => {}
            }
        }
        self.primal_edge_norms
            .as_mut()
            .unwrap()
            .set_pricing_rule(pricing_rule);
        self.primal_prices.force_recomputation();
        let mut final_check_performed = false;
        let mut recompute_reduced_costs =
            phase == SimplexPhase::Feasibility || self.reduced_costs_dirty;
        let mut refactorize_for_precision = false;
        let mut direction = ScatteredColumn::new(self.num_rows);
        let mut phase_objective = self.phase_objective(phase);
        let all_rows: Vec<_> = (0..self.num_rows.to_usize())
            .map(RowIndex::from_usize)
            .collect();
        loop {
            self.advance_deterministic_time(time_limit);
            if time_limit.limit_reached() {
                self.phase4_event("primal_phase_time_limit");
                if phase == SimplexPhase::Feasibility {
                    self.phase4_event("primal_phase_one_limit");
                } else {
                    self.phase4_event("primal_phase_two_limit");
                }
                self.problem_status = if phase == SimplexPhase::Feasibility {
                    ProblemStatus::Init
                } else {
                    ProblemStatus::PrimalFeasible
                };
                return Ok(());
            }
            if refactorize_for_precision {
                self.phase4_event("primal_precision_refactorization");
                // RefactorizeBasisIfNeeded() leaves an already refactorized
                // basis, and its cached update row, untouched. A reduced-cost
                // precision retry can still request a fresh cost pass.
                if !self.basis_factorization.as_ref().unwrap().is_refactorized() {
                    self.basis_factorization.as_mut().unwrap().refactorize()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                }
                self.primal_prices.force_recomputation();
                recompute_reduced_costs = true;
                refactorize_for_precision = false;
            }
            if self
                .primal_edge_norms
                .as_ref()
                .unwrap()
                .needs_basis_refactorization()
                && !self.basis_factorization.as_ref().unwrap().is_refactorized()
            {
                self.phase4_event("primal_edge_norm_refactorization_requested");
                refactorize_for_precision = true;
                continue;
            }
            let refactorized = self.basis_factorization.as_ref().unwrap().is_refactorized();
            if refactorized {
                self.correct_errors_on_variable_values()?;
            }
            // PrimalMinimize() tests the objective limit before asking
            // PrimalPrices for a column. Pricing may lazily recompute the
            // reduced costs, so reversing this order charges an extra BTRAN
            // and can alter a close deterministic-limit decision.
            if phase == SimplexPhase::Optimization
                && refactorized
                && self.internal_objective() < self.primal_objective_limit
            {
                self.phase4_event("primal_objective_limit");
                self.problem_status = ProblemStatus::PrimalFeasible;
                self.objective_limit_reached = true;
                return Ok(());
            }
            if phase == SimplexPhase::Feasibility {
                let rows = if refactorized {
                    &all_rows
                } else {
                    direction.non_zeros()
                };
                if !refactorized && !rows.is_empty() {
                    self.phase4_event("primal_phase_one_sparse_cost_update");
                }
                if self.update_primal_phase_one_costs(rows, &mut phase_objective) {
                    recompute_reduced_costs = true;
                } else if !refactorized && !rows.is_empty() && !recompute_reduced_costs {
                    self.phase4_event("primal_phase_one_incremental_reduced_costs");
                }
            }
            if recompute_reduced_costs {
                self.compute_reduced_costs(&phase_objective)?;
                self.primal_prices.force_recomputation();
                recompute_reduced_costs = false;
            }
            let Some(entering) = self.choose_entering()? else {
                // GLOP accepts an empty pricing set only after its FINAL_CHECK
                // has both precise reduced costs and a refactorized basis.
                // Perform the check once explicitly and retain it across
                // rejected precise candidates. Only an actual pivot makes
                // the checked state obsolete.
                if !final_check_performed {
                    if !self.basis_factorization.as_ref().unwrap().is_refactorized() {
                        self.basis_factorization.as_mut().unwrap().refactorize()?;
                        self.incorporate_basis_permutation();
                        self.update_row.as_mut().unwrap().invalidate();
                        self.primal_prices.force_recomputation();
                        recompute_reduced_costs = true;
                        final_check_performed = true;
                        continue;
                    }
                    if !self.reduced_costs_precise {
                        recompute_reduced_costs = true;
                        final_check_performed = true;
                        continue;
                    }
                }
                if phase == SimplexPhase::Feasibility {
                    let infeasibility = self.maximum_primal_infeasibility();
                    self.problem_status =
                        if infeasibility < self.parameters.primal_feasibility_tolerance {
                            ProblemStatus::PrimalFeasible
                        } else {
                            ProblemStatus::PrimalInfeasible
                        };
                    if self.problem_status == ProblemStatus::PrimalInfeasible {
                        self.phase4_event("primal_infeasible_termination");
                    }
                } else {
                    self.problem_status = ProblemStatus::Optimal;
                }
                return Ok(());
            };
            self.direction(entering, &mut direction)?;
            let entering_edge_norm_is_precise = self
                .primal_edge_norms
                .as_mut()
                .unwrap()
                .test_entering_edge_norm_precision(
                    entering.to_usize(),
                    direction.values().as_slice(),
                );
            if self
                .primal_edge_norms
                .as_ref()
                .unwrap()
                .needs_basis_refactorization()
            {
                self.phase4_event("primal_edge_norm_recomputation_requested");
                // Native edge norms notify PrimalPrices' recomputation
                // watcher. Its RecomputePriceAt() and UpdateBeforeBasisPivot()
                // then return before requesting norms from an updated basis.
                self.primal_prices.force_recomputation();
            }
            if !entering_edge_norm_is_precise {
                self.recompute_primal_price_at(entering)?;
                continue;
            }
            let precise_reduced = phase_objective[entering]
                - basic_objective_direction_scalar_product(
                    &phase_objective,
                    &self.basis,
                    &direction,
                );
            let old_reduced = self.reduced_costs[entering];
            self.reduced_costs[entering] = precise_reduced;
            let scale = if precise_reduced.abs() <= 1.0 {
                1.0
            } else {
                precise_reduced
            };
            let reduced_cost_imprecise = (old_reduced - precise_reduced).abs() / scale
                > self.parameters.recompute_reduced_costs_threshold;
            if reduced_cost_imprecise {
                self.phase4_event("primal_reduced_cost_precision_retry");
                // MakeReducedCostsPrecise() marks the full row for lazy
                // recomputation. A pivot can still proceed in this
                // iteration, but its incremental row update is then skipped.
                self.reduced_costs_dirty = true;
                self.primal_prices.force_recomputation();
            }
            refactorize_for_precision |= reduced_cost_imprecise;
            self.recompute_primal_price_at(entering)?;
            {
                let info = self.variables_info.as_ref().unwrap();
                let valid_entering_candidate = (info.can_increase().contains(entering)
                    && precise_reduced < -self.parameters.dual_feasibility_tolerance)
                    || (info.can_decrease().contains(entering)
                        && precise_reduced > self.parameters.dual_feasibility_tolerance);
                if !valid_entering_candidate {
                    // GLOP's MakeReducedCostsPrecise() is a no-op if they are
                    // already precise. The selected column was just updated
                    // by the exact BTRAN/FTRAN check above.
                    if !self.basis_factorization.as_ref().unwrap().is_refactorized() {
                        refactorize_for_precision = true;
                    }
                    continue;
                }
            }
            // GLOP checks the iteration limit only after pricing and the
            // precise entering-cost check. At limit zero an already feasible
            // or optimal basis must still be recognized above.
            if self.num_iterations
                == u64::try_from(self.parameters.max_number_of_iterations).unwrap_or(u64::MAX)
            {
                if phase == SimplexPhase::Feasibility {
                    self.phase4_event("primal_phase_one_limit");
                } else {
                    self.phase4_event("primal_phase_two_limit");
                }
                self.problem_status = if phase == SimplexPhase::Feasibility {
                    ProblemStatus::Init
                } else {
                    ProblemStatus::PrimalFeasible
                };
                return Ok(());
            }
            let direction_norm = direction
                .values()
                .as_slice()
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            let reduced = precise_reduced;
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
                    &self.random,
                )
            };
            if choice == LeavingChoice::Refactorize {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                self.primal_prices.force_recomputation();
                recompute_reduced_costs = true;
                continue;
            }
            let (leaving_row, step_length, target_bound) = match choice {
                LeavingChoice::BoundFlip { step } => {
                    self.phase4_event("primal_bound_flip");
                    (None, step, 0.0)
                }
                LeavingChoice::Pivot {
                    row,
                    step,
                    target_bound,
                    exact_tie,
                } => {
                    if exact_tie {
                        self.phase4_event("primal_harris_exact_tie");
                    }
                    (Some(row), step, target_bound)
                }
                LeavingChoice::Refactorize => unreachable!(),
            };
            if step_length.is_infinite() {
                // As in PrimalMinimize(), confirm an infinite ratio-test step
                // against a freshly factorized basis before exposing a ray.
                if !self.basis_factorization.as_ref().unwrap().is_refactorized() {
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .force_refactorization()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    self.primal_prices.force_recomputation();
                    recompute_reduced_costs = true;
                    continue;
                }
                self.problem_status = if phase == SimplexPhase::Feasibility {
                    ProblemStatus::Abnormal
                } else {
                    ProblemStatus::PrimalUnbounded
                };
                if phase == SimplexPhase::Optimization {
                    self.phase4_event("primal_unbounded_termination");
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
            let trace_leaving_column = leaving_row.map(|row| self.basis[row]);
            if phase == SimplexPhase::Feasibility
                && let Some(row) = leaving_row
            {
                let leaving = self.basis[row];
                step = (self.variable_values[leaving] - target_bound) / direction.value(row);
            }
            let degenerate_pivot = leaving_row.is_some_and(|row| {
                let leaving = self.basis[row];
                let movement = -direction.value(row) * step;
                movement == 0.0
                    || (movement > 0.0 && self.variable_values[leaving] >= target_bound)
                    || (movement < 0.0 && self.variable_values[leaving] <= target_bound)
            });
            if degenerate_pivot {
                self.phase4_event("primal_degenerate_pivot");
            }
            for entry in &direction {
                let column = self.basis[entry.row()];
                self.variable_values[column] =
                    (-entry.coefficient()).mul_add(step, self.variable_values[column]);
            }
            self.variable_values[entering] += step;

            if let Some(row) = leaving_row {
                let leaving = self.basis[row];
                self.update_row
                    .as_mut()
                    .unwrap()
                    .compute_unit_row_left_inverse(
                        self.basis_factorization.as_ref().unwrap(),
                        row.to_usize(),
                    )?;
                self.primal_edge_norms
                    .as_mut()
                    .unwrap()
                    .update_before_basis_pivot(
                        self.basis_factorization.as_ref().unwrap(),
                        self.variables_info.as_ref().unwrap().relevance(),
                        self.variables_info
                            .as_ref()
                            .unwrap()
                            .num_entries_in_relevant_columns()
                            .value()
                            .try_into()
                            .expect("negative relevant entry count"),
                        entering.to_usize(),
                        leaving.to_usize(),
                        row.to_usize(),
                        direction.values().as_slice(),
                        self.update_row.as_mut().unwrap(),
                    )?;
                if !self.reduced_costs_dirty {
                    // Dantzig edge norms do not need the update row. Native
                    // ReducedCosts computes it only when maintaining its
                    // incremental row; a pending full recomputation skips it.
                    self.update_row.as_mut().unwrap().compute_update_row(
                        self.basis_factorization.as_ref().unwrap(),
                        &self.matrix,
                        self.variables_info.as_ref().unwrap().relevance(),
                        self.variables_info
                            .as_ref()
                            .unwrap()
                            .num_entries_in_relevant_columns()
                            .value()
                            .try_into()
                            .expect("negative relevant entry count"),
                        row.to_usize(),
                    )?;
                    update_reduced_cost_values_before_basis_pivot(
                        self.reduced_costs.as_mut_slice(),
                        entering,
                        leaving,
                        direction.value(row),
                        self.update_row.as_ref().unwrap(),
                    );
                }
                self.reduced_costs_precise = false;
                if !self.reduced_costs_dirty && !self.primal_prices.recomputation_pending() {
                    let norms = self.primal_edge_norms.as_mut().unwrap().squared_norms(
                        self.basis_factorization.as_ref().unwrap(),
                        self.variables_info.as_ref().unwrap().relevance(),
                    )?;
                    self.primal_prices.update_before_basis_pivot_from_values(
                        entering,
                        self.update_row.as_ref().unwrap(),
                        self.variables_info.as_ref().unwrap(),
                        self.reduced_costs.as_slice(),
                        norms,
                        self.parameters.dual_feasibility_tolerance,
                    );
                }
                if !degenerate_pivot {
                    self.variable_values[leaving] = target_bound;
                } else if self.variable_values[leaving] != target_bound {
                    self.phase4_event("primal_bound_shift");
                }
                let leaving_status = self.status_at_bound(leaving, target_bound);
                {
                    let info = self.variables_info.as_mut().unwrap();
                    info.update_to_nonbasic_status(leaving, leaving_status);
                    info.update_to_basic_status(entering);
                }
                self.basis[row] = entering;
                let pivot_from_update_row = self
                    .matrix
                    .column(entering)
                    .iter()
                    .map(|entry| {
                        self.update_row.as_ref().unwrap().unit_row_left_inverse()
                            [entry.index().to_usize()]
                            * entry.coefficient()
                    })
                    .sum::<f64>();
                let pivot_from_direction = direction.value(row);
                let pivot_difference = (pivot_from_update_row - pivot_from_direction).abs();
                let imprecise_pivot = pivot_difference
                    > self.parameters.refactorization_threshold
                        * (1.0 + pivot_from_update_row.abs().min(pivot_from_direction.abs()));
                if imprecise_pivot {
                    self.strengthen_lu_pivoting_after_early_imprecision()?;
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .update_and_refactorize(row.to_usize(), entering)?;
                } else {
                    if !self.parameters.use_middle_product_form_update {
                        self.phase4_event("primal_eta_basis_update");
                    }
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .update_after_solve(entering, row.to_usize(), &direction)?;
                }
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                if self.basis_factorization.as_ref().unwrap().is_refactorized() {
                    // GLOP retains the incremental reduced costs after an
                    // ordinary UpdateAndPivot() refactorization. Recomputing
                    // them here erases the error estimate that can request a
                    // separate reduced-cost precision refactorization later.
                    self.primal_prices.force_recomputation();
                }
                if phase == SimplexPhase::Feasibility {
                    self.phase4_event("primal_phase_one_leaving_cost_cleared");
                    // PrimalMinimize() puts the leaving nonbasic variable at
                    // its exact bound and removes its temporary Phase-I cost
                    // after UpdateAndPivot(). The reduced cost changes by
                    // precisely the removed cost, without a full solve.
                    self.variable_values[leaving] = target_bound;
                    self.reduced_costs[leaving] -= phase_objective[leaving];
                    phase_objective[leaving] = 0.0;
                    self.recompute_primal_price_at(leaving)?;
                }
            } else {
                let info = self.variables_info.as_mut().unwrap();
                if step > 0.0 {
                    info.update_to_nonbasic_status(entering, VariableStatus::AtUpperBound);
                    self.variable_values[entering] = info.upper_bounds()[entering.to_usize()];
                } else {
                    info.update_to_nonbasic_status(entering, VariableStatus::AtLowerBound);
                    self.variable_values[entering] = info.lower_bounds()[entering.to_usize()];
                }
                self.primal_prices
                    .set_and_debug_check_column_is_dual_feasible_from_values(
                        entering,
                        info,
                        self.reduced_costs.as_slice(),
                        self.parameters.dual_feasibility_tolerance,
                    );
            }
            self.num_iterations += 1;
            final_check_performed = false;
            if self.trace_enabled {
                self.trace.push(IterationEvent {
                    iteration: self.num_iterations,
                    phase,
                    entering_column: Some(entering),
                    leaving_row,
                    leaving_column: trace_leaving_column,
                    step,
                    objective: self.internal_objective(),
                });
            }
        }
    }

    fn run_dedicated_dual_phase_one(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        let result = self.run_dedicated_dual_phase_one_internal(time_limit);
        self.advance_deterministic_time(time_limit);
        result
    }

    fn run_dedicated_dual_phase_one_internal(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        let mut reduced_costs_precise =
            self.basis_factorization.as_ref().unwrap().is_refactorized();
        let mut reduced_costs_recomputed = true;
        let mut prices_initialized = false;
        let mut direction = ScatteredColumn::new(self.num_rows);
        loop {
            self.advance_deterministic_time(time_limit);
            if time_limit.limit_reached() {
                self.phase4_event("dual_phase_one_time_limit");
                self.problem_status = ProblemStatus::Init;
                return Ok(());
            }
            if self.dual_edge_norms.needs_basis_refactorization()
                && !self.basis_factorization.as_ref().unwrap().is_refactorized()
            {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                reduced_costs_precise = false;
                prices_initialized = false;
            }
            if self.reduced_costs_dirty
                || (self.basis_factorization.as_ref().unwrap().is_refactorized()
                    && !reduced_costs_precise)
            {
                let objective = self.objective.clone();
                self.compute_reduced_costs(&objective)?;
                reduced_costs_precise = true;
                reduced_costs_recomputed = true;
                prices_initialized = false;
            }
            let columns = if prices_initialized {
                Vec::new()
            } else {
                self.variables_info
                    .as_ref()
                    .unwrap()
                    .relevance()
                    .iter_ones()
                    .map(VectorIndex::to_usize)
                    .collect()
            };
            self.update_dual_phase_one_prices_for_columns(&columns, !prices_initialized)?;
            prices_initialized = true;
            if self.num_dual_infeasible_positions == 0 {
                // GLOP's final check clears perturbations and retries when
                // either the basis is not refactorized or a cost was shifted.
                if !self.basis_factorization.as_ref().unwrap().is_refactorized()
                    || self.has_cost_shift
                {
                    self.remove_cost_shifts();
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .force_refactorization()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    reduced_costs_precise = false;
                    prices_initialized = false;
                    continue;
                }
                self.problem_status = ProblemStatus::DualFeasible;
                return Ok(());
            }
            let Some(leaving_position) = self.dual_prices.get_maximum() else {
                self.phase4_event("dual_infeasible_termination");
                self.problem_status = ProblemStatus::DualInfeasible;
                return Ok(());
            };
            let leaving_row = RowIndex::from_usize(leaving_position);
            let cost_variation = self.dual_phase_one_pricing_vector[leaving_row];
            let leaving_column = self.basis[leaving_row];
            let target_bound = if cost_variation < 0.0 {
                self.variables_info.as_ref().unwrap().upper_bounds()[leaving_column.to_usize()]
            } else {
                self.variables_info.as_ref().unwrap().lower_bounds()[leaving_column.to_usize()]
            };

            self.update_row
                .as_mut()
                .unwrap()
                .compute_unit_row_left_inverse(
                    self.basis_factorization.as_ref().unwrap(),
                    leaving_position,
                )?;
            if !self.dual_edge_norms.test_precision(
                leaving_position,
                self.update_row
                    .as_ref()
                    .unwrap()
                    .unit_row_left_inverse_scattered(),
            ) {
                // TestPrecision() installs the exact norm for this row. As in
                // GLOP, update just this heap entry before choosing again; a
                // norm-wide precision failure is handled at the loop head.
                let norms = self
                    .dual_edge_norms
                    .edge_squared_norms(self.basis_factorization.as_ref().unwrap())?;
                let variable_type =
                    self.variables_info.as_ref().unwrap().variable_types()[leaving_column];
                Self::update_dual_phase_one_price_at(
                    &mut self.dual_prices,
                    leaving_row,
                    cost_variation,
                    norms[leaving_position],
                    variable_type,
                    self.parameters.ratio_test_zero_threshold,
                    false,
                );
                continue;
            }
            self.update_row.as_mut().unwrap().compute_update_row(
                self.basis_factorization.as_ref().unwrap(),
                &self.matrix,
                self.variables_info.as_ref().unwrap().relevance(),
                self.variables_info
                    .as_ref()
                    .unwrap()
                    .num_entries_in_relevant_columns()
                    .value()
                    .try_into()
                    .expect("negative relevant entry count"),
                leaving_position,
            )?;
            let entering = self
                .entering_variable
                .dual_phase_one_choose_entering_column_from_values(
                    reduced_costs_precise,
                    self.update_row.as_ref().unwrap(),
                    cost_variation,
                    self.variables_info.as_ref().unwrap(),
                    self.reduced_costs.as_slice(),
                    self.parameters.dual_feasibility_tolerance,
                );
            let Some(entering) = entering else {
                if !reduced_costs_precise {
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .force_refactorization()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    prices_initialized = false;
                    continue;
                }
                self.problem_status = ProblemStatus::Abnormal;
                return Ok(());
            };
            let entering_coefficient = self
                .update_row
                .as_ref()
                .unwrap()
                .coefficient(entering.to_usize());
            if entering_coefficient.abs() < self.parameters.dual_small_pivot_threshold
                && !reduced_costs_precise
            {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                prices_initialized = false;
                continue;
            }
            self.direction(entering, &mut direction)?;
            let pivot = direction.value(leaving_row);
            let direction_norm = direction
                .values()
                .as_slice()
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            if pivot.abs() < self.parameters.small_pivot_threshold * direction_norm
                && !reduced_costs_precise
            {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                prices_initialized = false;
                continue;
            }
            if pivot.abs() <= 1e-20 {
                return Err(FactorizationError::Singular {
                    step: leaving_position,
                });
            }
            // GLOP checks the iteration limit only after pricing, the ratio
            // test, and FTRAN. This permits a zero-iteration solve to report
            // feasibility/optimality while stopping immediately before the
            // first actual pivot.
            if self.num_iterations
                >= u64::try_from(self.parameters.max_number_of_iterations).unwrap_or(u64::MAX)
            {
                self.phase4_event("dual_phase_one_limit");
                self.problem_status = ProblemStatus::Init;
                return Ok(());
            }
            let increasing_reduced_cost_needed =
                (cost_variation > 0.0) == (entering_coefficient > 0.0);
            self.shift_cost_if_needed(increasing_reduced_cost_needed, entering);
            if update_reduced_cost_values_before_basis_pivot(
                self.reduced_costs.as_mut_slice(),
                entering,
                leaving_column,
                pivot,
                self.update_row.as_ref().unwrap(),
            ) {
                reduced_costs_recomputed = false;
            }
            self.dual_edge_norms.update_before_basis_pivot(
                self.basis_factorization.as_ref().unwrap(),
                leaving_position,
                direction.values().as_slice(),
                self.update_row
                    .as_ref()
                    .unwrap()
                    .unit_row_left_inverse_scattered(),
            )?;
            self.update_dual_phase_one_prices_on_pivot(
                leaving_row,
                entering,
                &direction,
                reduced_costs_recomputed,
            )?;
            let changed_columns = self
                .update_row
                .as_ref()
                .unwrap()
                .non_zero_positions()
                .to_vec();

            let leaving_status = self.status_at_bound(leaving_column, target_bound);
            {
                let info = self.variables_info.as_mut().unwrap();
                info.update_to_nonbasic_status(leaving_column, leaving_status);
                info.update_to_basic_status(entering);
            }
            self.basis[leaving_row] = entering;
            self.variable_values[leaving_column] = target_bound;
            let pivot_from_update_row = self
                .update_row
                .as_ref()
                .unwrap()
                .coefficient(entering.to_usize());
            let pivot_difference = (pivot_from_update_row - pivot).abs();
            let imprecise_pivot = pivot_difference
                > self.parameters.refactorization_threshold
                    * (1.0 + pivot_from_update_row.abs().min(pivot.abs()));
            if imprecise_pivot {
                self.strengthen_lu_pivoting_after_early_imprecision()?;
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .update_and_refactorize(leaving_position, entering)?;
            } else {
                if !self.parameters.use_middle_product_form_update {
                    self.phase4_event("dual_phase_one_eta_basis_update");
                }
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .update_after_solve(entering, leaving_position, &direction)?;
            }
            self.incorporate_basis_permutation();
            self.update_row.as_mut().unwrap().invalidate();
            reduced_costs_precise = false;
            self.reduced_costs_precise = false;
            if reduced_costs_recomputed
                || self.basis_factorization.as_ref().unwrap().is_refactorized()
                || self.dual_edge_norms.needs_basis_refactorization()
            {
                prices_initialized = false;
            } else {
                self.update_dual_phase_one_prices_for_columns(&changed_columns, false)?;
            }
            self.num_iterations += 1;
            if self.trace_enabled {
                self.trace.push(IterationEvent {
                    iteration: self.num_iterations,
                    phase: SimplexPhase::Feasibility,
                    entering_column: Some(entering),
                    leaving_row: Some(leaving_row),
                    leaving_column: Some(leaving_column),
                    step: 0.0,
                    objective: self.internal_objective(),
                });
            }
        }
    }

    fn run_dual_phase_two(&mut self, time_limit: &mut TimeLimit) -> Result<(), FactorizationError> {
        let result = self.run_dual_phase_two_internal(time_limit);
        self.advance_deterministic_time(time_limit);
        result
    }

    fn run_dual_phase_two_internal(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<(), FactorizationError> {
        if self.reduced_costs_dirty {
            let objective = self.objective.clone();
            self.compute_reduced_costs(&objective)?;
        }
        let mut reduced_costs_precise =
            self.basis_factorization.as_ref().unwrap().is_refactorized();
        let mut recompute_reduced_costs_after_refactorization = false;
        self.recompute_dual_prices()?;
        let mut pending_price_rows = Vec::new();
        let mut direction = ScatteredColumn::new(self.num_rows);
        loop {
            self.advance_deterministic_time(time_limit);
            if time_limit.limit_reached() {
                self.problem_status = ProblemStatus::DualFeasible;
                return Ok(());
            }
            if self.dual_edge_norms.needs_basis_refactorization()
                && !self.basis_factorization.as_ref().unwrap().is_refactorized()
            {
                self.phase4_event("dual_edge_norm_forced_refactorization");
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                reduced_costs_precise = false;
            }
            let basis_is_refactorized =
                self.basis_factorization.as_ref().unwrap().is_refactorized();
            if basis_is_refactorized {
                // GLOP deliberately preserves incrementally updated reduced
                // costs across a routine update-count refactorization. It
                // only makes them precise when refactorization was explicitly
                // requested by the iteration loop (old_refactorize_value).
                if !reduced_costs_precise && recompute_reduced_costs_after_refactorization {
                    let objective = self.objective.clone();
                    self.compute_reduced_costs(&objective)?;
                    reduced_costs_precise = true;
                    recompute_reduced_costs_after_refactorization = false;
                }
                // As in GLOP, move every dual-infeasible nonbasic boxed
                // variable to its opposite bound before recomputing the
                // basic values. The incremental branch only processes the
                // preceding ratio test's flip candidates, but a refactorized
                // basis must refresh the complete boxed set.
                let boxed: Vec<_> = self
                    .variables_info
                    .as_ref()
                    .unwrap()
                    .non_basic_boxed_variables()
                    .iter_ones()
                    .collect();
                self.make_boxed_variables_dual_feasible(&boxed, false)?;
                self.initialize_values()?;
                self.recompute_dual_prices()?;
                pending_price_rows.clear();
                if self.dual_objective_limit != f64::INFINITY
                    && self.internal_objective() > self.dual_objective_limit
                {
                    self.phase4_event("dual_objective_limit");
                    self.problem_status = ProblemStatus::DualFeasible;
                    self.objective_limit_reached = true;
                    return Ok(());
                }
            } else if !basis_is_refactorized {
                if !self.bound_flip_candidates.is_empty() {
                    self.phase4_event("dual_boxed_bound_flip");
                    let candidates = std::mem::take(&mut self.bound_flip_candidates);
                    self.make_boxed_variables_dual_feasible(&candidates, true)?;
                }
                if !pending_price_rows.is_empty() {
                    // Upstream retains `direction_.non_zeros` until a
                    // successful pivot replaces the direction. A retry
                    // therefore reprices the same rows again, including the
                    // duplicate heap entries and shared-RNG draws that this
                    // can deliberately create.
                    self.update_dual_prices(&pending_price_rows)?;
                }
            }
            let Some(leaving_position) = self.dual_prices.get_maximum() else {
                if !self.basis_factorization.as_ref().unwrap().is_refactorized()
                    || self.has_cost_shift
                {
                    self.remove_cost_shifts();
                    self.basis_factorization.as_mut().unwrap().refactorize()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    reduced_costs_precise = false;
                    recompute_reduced_costs_after_refactorization = true;
                    continue;
                }
                self.problem_status = ProblemStatus::Optimal;
                self.phase4_event("dual_phase_two_optimal");
                return Ok(());
            };
            let leaving_row = RowIndex::from_usize(leaving_position);
            let leaving_column = self.basis[leaving_row];
            let (cost_variation, target_bound) = {
                let info = self.variables_info.as_ref().unwrap();
                let value = self.variable_values[leaving_column];
                if value < info.lower_bounds()[leaving_column.to_usize()] {
                    let target = info.lower_bounds()[leaving_column.to_usize()];
                    (target - value, target)
                } else {
                    let target = info.upper_bounds()[leaving_column.to_usize()];
                    (target - value, target)
                }
            };

            self.update_row
                .as_mut()
                .unwrap()
                .compute_unit_row_left_inverse(
                    self.basis_factorization.as_ref().unwrap(),
                    leaving_position,
                )?;
            if !self.dual_edge_norms.test_precision(
                leaving_position,
                self.update_row
                    .as_ref()
                    .unwrap()
                    .unit_row_left_inverse_scattered(),
            ) {
                self.update_dual_prices(&[leaving_row])?;
                continue;
            }
            if self.dual_edge_norms.needs_basis_refactorization() {
                self.phase4_event("dual_edge_norm_recomputation_requested");
            }
            self.update_row.as_mut().unwrap().compute_update_row(
                self.basis_factorization.as_ref().unwrap(),
                &self.matrix,
                self.variables_info.as_ref().unwrap().relevance(),
                self.variables_info
                    .as_ref()
                    .unwrap()
                    .num_entries_in_relevant_columns()
                    .value()
                    .try_into()
                    .expect("negative relevant entry count"),
                leaving_position,
            )?;
            let entering = self
                .entering_variable
                .dual_choose_entering_column_from_values(
                    reduced_costs_precise,
                    self.update_row.as_ref().unwrap(),
                    cost_variation,
                    self.variables_info.as_ref().unwrap(),
                    self.reduced_costs.as_slice(),
                    self.parameters.dual_feasibility_tolerance,
                    &mut self.bound_flip_candidates,
                );
            let Some(entering) = entering else {
                if !reduced_costs_precise {
                    self.basis_factorization.as_mut().unwrap().refactorize()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    recompute_reduced_costs_after_refactorization = true;
                    continue;
                }
                self.problem_status = ProblemStatus::DualUnbounded;
                self.phase4_event("dual_unbounded_termination");
                self.dual_ray = DenseColumn::from_vec(
                    self.update_row
                        .as_ref()
                        .unwrap()
                        .unit_row_left_inverse()
                        .to_vec(),
                );
                self.dual_ray_row_combination = DenseRow::from_vec(
                    self.update_row.as_ref().unwrap().compute_full_update_row(
                        &self.matrix,
                        &self
                            .basis
                            .as_slice()
                            .iter()
                            .map(|column| column.to_usize())
                            .collect::<Vec<_>>(),
                        self.variables_info.as_ref().unwrap().not_basic(),
                        leaving_position,
                    )?,
                );
                if cost_variation < 0.0 {
                    for value in self.dual_ray.as_mut_slice() {
                        *value = -*value;
                    }
                    for value in self.dual_ray_row_combination.as_mut_slice() {
                        *value = -*value;
                    }
                }
                return Ok(());
            };

            // GLOP first rejects a small update-row coefficient before doing
            // FTRAN. This is distinct from the direction-relative pivot test
            // below and requests a precise, refactorized retry when the
            // incremental reduced costs are not yet precise.
            let entering_coefficient = self
                .update_row
                .as_ref()
                .unwrap()
                .coefficient(entering.to_usize());
            if entering_coefficient.abs() < self.parameters.dual_small_pivot_threshold
                && !reduced_costs_precise
            {
                self.basis_factorization.as_mut().unwrap().refactorize()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                recompute_reduced_costs_after_refactorization = true;
                continue;
            }

            self.direction(entering, &mut direction)?;
            let pivot = direction.value(leaving_row);
            let direction_norm = direction
                .values()
                .as_slice()
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
            if pivot.abs() < self.parameters.small_pivot_threshold * direction_norm
                && !reduced_costs_precise
            {
                self.basis_factorization.as_mut().unwrap().refactorize()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                recompute_reduced_costs_after_refactorization = true;
                continue;
            }
            if pivot.abs() <= 1e-20 {
                return Err(FactorizationError::Singular {
                    step: leaving_position,
                });
            }
            // Match DualMinimize(): test the iteration limit only after
            // pricing and pivot validation, so a basis that is already
            // optimal at the limit is still reported as optimal.
            if self.num_iterations
                == u64::try_from(self.parameters.max_number_of_iterations).unwrap_or(u64::MAX)
            {
                self.phase4_event("dual_iteration_limit");
                self.problem_status = ProblemStatus::DualFeasible;
                return Ok(());
            }
            let increasing_reduced_cost_needed =
                (cost_variation > 0.0) == (entering_coefficient > 0.0);
            self.shift_cost_if_needed(increasing_reduced_cost_needed, entering);
            update_reduced_cost_values_before_basis_pivot(
                self.reduced_costs.as_mut_slice(),
                entering,
                leaving_column,
                pivot,
                self.update_row.as_ref().unwrap(),
            );
            self.dual_edge_norms.update_before_basis_pivot(
                self.basis_factorization.as_ref().unwrap(),
                leaving_position,
                direction.values().as_slice(),
                self.update_row
                    .as_ref()
                    .unwrap()
                    .unit_row_left_inverse_scattered(),
            )?;
            let step = (self.variable_values[leaving_column] - target_bound) / pivot;
            for entry in &direction {
                let column = self.basis[entry.row()];
                self.variable_values[column] =
                    (-entry.coefficient()).mul_add(step, self.variable_values[column]);
            }
            self.variable_values[entering] += step;

            let leaving_status = self.status_at_bound(leaving_column, target_bound);
            {
                let info = self.variables_info.as_mut().unwrap();
                info.update_to_nonbasic_status(leaving_column, leaving_status);
                info.update_to_basic_status(entering);
            }
            self.basis[leaving_row] = entering;
            let pivot_from_update_row = self
                .update_row
                .as_ref()
                .unwrap()
                .coefficient(entering.to_usize());
            let pivot_difference = (pivot_from_update_row - pivot).abs();
            let imprecise_pivot = pivot_difference
                > self.parameters.refactorization_threshold
                    * (1.0 + pivot_from_update_row.abs().min(pivot.abs()));
            if imprecise_pivot {
                self.strengthen_lu_pivoting_after_early_imprecision()?;
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .update_and_refactorize(leaving_position, entering)?;
            } else {
                if !self.parameters.use_middle_product_form_update {
                    self.phase4_event("dual_eta_basis_update");
                }
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .update_after_solve(entering, leaving_position, &direction)?;
            }
            self.incorporate_basis_permutation();
            self.update_row.as_mut().unwrap().invalidate();
            self.variable_values[leaving_column] = target_bound;
            reduced_costs_precise = false;
            self.num_iterations += 1;

            if self.trace_enabled {
                self.trace.push(IterationEvent {
                    iteration: self.num_iterations,
                    phase: SimplexPhase::Optimization,
                    entering_column: Some(entering),
                    leaving_row: Some(leaving_row),
                    leaving_column: Some(leaving_column),
                    step,
                    objective: self.internal_objective(),
                });
            }
            if self.basis_factorization.as_ref().unwrap().is_refactorized()
                || self.dual_edge_norms.needs_basis_refactorization()
            {
                self.dual_prices.clear();
                pending_price_rows.clear();
            } else {
                pending_price_rows = if direction.non_zeros().is_empty() {
                    (0..self.num_rows.to_usize())
                        .map(RowIndex::from_usize)
                        .collect::<Vec<_>>()
                } else {
                    direction.non_zeros().to_vec()
                };
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
                    exact_tie: false,
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
        let permutation = self
            .basis_factorization
            .as_ref()
            .unwrap()
            .column_permutation()
            .to_vec();
        if permutation.is_empty() {
            return;
        }
        let mut permuted = self.basis.clone();
        for (source, &destination) in permutation.iter().enumerate() {
            permuted[RowIndex::from_usize(destination)] = self.basis[RowIndex::from_usize(source)];
        }
        self.basis = permuted;

        if !self.dual_phase_one_pricing_vector.is_empty() {
            let mut permuted = self.dual_phase_one_pricing_vector.clone();
            for (source, &destination) in permutation.iter().enumerate() {
                permuted[RowIndex::from_usize(destination)] =
                    self.dual_phase_one_pricing_vector[RowIndex::from_usize(source)];
            }
            self.dual_phase_one_pricing_vector = permuted;
        }
        self.dual_edge_norms
            .update_data_on_basis_permutation(&ColumnPermutation::from_vec(
                permutation
                    .iter()
                    .copied()
                    .map(ColIndex::from_usize)
                    .collect(),
            ));
        self.basis_factorization
            .as_mut()
            .unwrap()
            .set_column_permutation_to_identity();
    }

    fn maximum_primal_residual(&self) -> f64 {
        let mut residual = DenseColumn::filled(self.num_rows, 0.0);
        for column in 0..self.num_cols.to_usize() {
            self.compact_matrix.column_add_multiple_to_dense_column(
                ColIndex::from_usize(column),
                self.variable_values[ColIndex::from_usize(column)],
                &mut residual,
            );
        }
        residual
            .as_slice()
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
    }

    fn maximum_dual_residual(&self, objective: &DenseRow) -> f64 {
        let dual_row = DenseRow::from_vec(self.dual_values.as_slice().to_vec());
        (0..self.num_rows.to_usize()).fold(0.0_f64, |maximum, row| {
            let basic = self.basis[RowIndex::from_usize(row)];
            let residual = objective[basic] + self.cost_perturbations[basic]
                - self.compact_matrix.column_scalar_product(basic, &dual_row);
            maximum.max(residual.abs())
        })
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

    fn maximum_dual_infeasibility(&self) -> f64 {
        let info = self.variables_info.as_ref().unwrap();
        (0..self.num_cols.to_usize()).fold(0.0_f64, |maximum, column| {
            let column = ColIndex::from_usize(column);
            let reduced_cost = self.reduced_costs[column];
            let mut infeasibility = 0.0_f64;
            if info.can_increase().contains(column) {
                infeasibility = infeasibility.max(-reduced_cost);
            }
            if info.can_decrease().contains(column) {
                infeasibility = infeasibility.max(reduced_cost);
            }
            maximum.max(infeasibility)
        })
    }

    fn correct_errors_on_variable_values(&mut self) -> Result<(), FactorizationError> {
        let mut residual = vec![0.0; self.num_rows.to_usize()];
        for column in 0..self.num_cols.to_usize() {
            let column = ColIndex::from_usize(column);
            let value = self.variable_values[column];
            for entry in self.matrix.column(column) {
                let row = entry.index().to_usize();
                residual[row] = entry.coefficient().mul_add(value, residual[row]);
            }
        }
        let maximum = residual
            .into_iter()
            .fold(0.0_f64, |value, entry| value.max(entry.abs()));
        if maximum
            < self.parameters.harris_tolerance_ratio * self.parameters.primal_feasibility_tolerance
        {
            return Ok(());
        }

        let info = self.variables_info.as_ref().unwrap();
        let mut rhs = vec![0.0; self.num_rows.to_usize()];
        for column in info.not_basic().iter_ones() {
            let value = self.variable_values[column];
            for entry in self.matrix.column(column) {
                let row = entry.index().to_usize();
                rhs[row] = (-entry.coefficient()).mul_add(value, rhs[row]);
            }
        }
        let basic = self.basis_factorization.as_ref().unwrap().solve(&rhs)?;
        for (row, &value) in basic.iter().enumerate() {
            self.variable_values[self.basis[RowIndex::from_usize(row)]] = value;
        }
        Ok(())
    }

    fn internal_objective(&self) -> f64 {
        precise_scalar_product(self.objective.as_slice(), self.variable_values.as_slice())
    }

    fn finish_solution(&mut self) -> Result<(), FactorizationError> {
        // GLOP leaves its temporary Phase-I objective installed when primal
        // feasibility fails. Expose reduced costs/dual values for that same
        // objective rather than recomputing them for the unsolved user cost.
        let objective = if self.problem_status == ProblemStatus::PrimalInfeasible {
            self.phase_objective(SimplexPhase::Feasibility)
        } else {
            self.objective.clone()
        };
        self.solution_objective_value = self.objective_scaling_factor
            * (precise_scalar_product(objective.as_slice(), self.variable_values.as_slice())
                + self.objective_offset);
        if matches!(
            self.problem_status,
            ProblemStatus::PrimalUnbounded | ProblemStatus::DualUnbounded
        ) {
            let sign = if self.problem_status == ProblemStatus::DualUnbounded {
                1.0
            } else {
                -1.0
            };
            self.solution_objective_value = if self.is_maximization_problem {
                -sign * f64::INFINITY
            } else {
                sign * f64::INFINITY
            };
        }
        self.refresh_dual_values(&objective)?;
        if self.reduced_costs_dirty {
            // GetReducedCosts() reuses the left inverse from GetDualValues().
            self.compute_reduced_costs_from_dual(&objective);
        }
        self.solution_reduced_costs = self.reduced_costs.clone();
        self.solution_dual_values = self.dual_values.clone();
        if self.is_maximization_problem {
            for value in self.solution_reduced_costs.as_mut_slice() {
                *value = -*value;
            }
            for value in self.solution_dual_values.as_mut_slice() {
                *value = -*value;
            }
        }
        self.solution_state.statuses = self
            .variables_info
            .as_ref()
            .unwrap()
            .variable_statuses()
            .clone();
        self.external_state_for_next_solve = false;
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
        self.solution_objective_value
    }
    #[must_use]
    pub const fn objective_limit_reached(&self) -> bool {
        self.objective_limit_reached
    }
    #[must_use]
    pub fn variable_value(&self, column: ColIndex) -> f64 {
        self.variable_values[column]
    }
    #[must_use]
    pub fn reduced_cost(&self, column: ColIndex) -> f64 {
        self.solution_reduced_costs[column]
    }
    #[must_use]
    #[allow(clippy::misnamed_getters)] // Mirrors GLOP's final-solution getter.
    pub const fn reduced_costs(&self) -> &DenseRow {
        &self.solution_reduced_costs
    }
    #[must_use]
    pub fn dual_value(&self, row: RowIndex) -> f64 {
        self.solution_dual_values[row]
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
