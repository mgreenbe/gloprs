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
use crate::reduced_costs::{PrimalPrices, update_reduced_cost_values_before_basis_pivot};
use crate::time_limit::TimeLimit;
use crate::update_row::UpdateRow;
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
    matrix: SparseMatrix,
    compact_matrix: CompactSparseMatrix,
    objective: DenseRow,
    objective_offset: f64,
    objective_scaling_factor: f64,
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
    solution_state: BasisState,
    state_for_next_solve: Option<BasisState>,
    starting_values: DenseRow,
    num_iterations: u64,
    trace_enabled: bool,
    trace: Vec<IterationEvent>,
    initial_basis_before_permutation: RowToColMapping,
    initial_column_permutation: Vec<usize>,
}

impl Default for RevisedSimplex {
    fn default() -> Self {
        Self::new()
    }
}

impl RevisedSimplex {
    #[must_use]
    pub fn new() -> Self {
        let random = SharedRandom::new(1);
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
            solution_state: BasisState::default(),
            state_for_next_solve: None,
            starting_values: DenseRow::new(),
            num_iterations: 0,
            trace_enabled: false,
            trace: Vec::new(),
            initial_basis_before_permutation: RowToColMapping::new(),
            initial_column_permutation: Vec::new(),
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

        let mut ran_dual = false;
        if self.parameters.use_dual_simplex {
            ran_dual = self.prepare_and_run_dual_phase_two(time_limit)?;
        }
        if !ran_dual {
            self.run_primal_phase(SimplexPhase::Feasibility, time_limit)?;
            if self.problem_status == ProblemStatus::PrimalFeasible && !time_limit.limit_reached() {
                self.run_primal_phase(SimplexPhase::Optimization, time_limit)?;
            }
        }
        self.finish_solution()?;
        self.starting_values.clear();
        Ok(())
    }

    fn prepare_and_run_dual_phase_two(
        &mut self,
        time_limit: &mut TimeLimit,
    ) -> Result<bool, FactorizationError> {
        let objective = self.objective.clone();
        self.compute_reduced_costs(&objective)?;
        self.variables_info
            .as_mut()
            .unwrap()
            .make_boxed_variable_relevant(false);

        let info = self.variables_info.as_ref().unwrap();
        let tolerance = self.parameters.dual_feasibility_tolerance;
        let nonboxed_dual_infeasible = info.relevance().iter_ones().any(|column| {
            let reduced = self.reduced_costs[column];
            (info.can_increase().contains(column) && reduced < -tolerance)
                || (info.can_decrease().contains(column) && reduced > tolerance)
        });
        if nonboxed_dual_infeasible {
            if !self.parameters.use_dedicated_dual_feasibility_algorithm {
                // The transformed-problem alternative remains a separate
                // Phase-4 path; keep the primal fallback for this nondefault.
                return Ok(false);
            }
            self.run_dedicated_dual_phase_one(time_limit)?;
            if self.problem_status != ProblemStatus::DualFeasible {
                return Ok(true);
            }
            self.basis_factorization.as_mut().unwrap().refactorize()?;
            self.incorporate_basis_permutation();
            self.update_row.as_mut().unwrap().invalidate();
            let objective = self.objective.clone();
            self.compute_reduced_costs(&objective)?;
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
        self.make_boxed_variables_dual_feasible(&boxed, false)?;
        self.variables_info
            .as_mut()
            .unwrap()
            .make_boxed_variable_relevant(true);
        self.initialize_values()?;
        self.problem_status = ProblemStatus::DualFeasible;
        self.run_dual_phase_two(time_limit)?;
        Ok(true)
    }

    fn recompute_dual_prices(&mut self) -> Result<(), FactorizationError> {
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
            for (column, delta) in changed {
                self.compact_matrix
                    .column_add_multiple_to_scattered_column(column, delta, &mut rhs);
            }
            rhs.clear_sparse_mask();
            rhs.clear_non_zeros_if_too_dense(0.8);
            self.basis_factorization
                .as_ref()
                .unwrap()
                .solve_with_nonzeros(&mut rhs)?;
            let changed_rows: Vec<_> = if rhs.non_zeros().is_empty() {
                (0..self.num_rows.to_usize())
                    .map(RowIndex::from_usize)
                    .collect()
            } else {
                rhs.non_zeros().to_vec()
            };
            for &row in &changed_rows {
                self.variable_values[self.basis[row]] -= rhs.value(row);
            }
            self.update_dual_prices(&changed_rows)?;
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
        self.is_maximization_problem = equation_lp.is_maximization_problem();
        self.matrix = equation_lp.matrix().clone();
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
        if self.trace_enabled {
            self.initial_basis_before_permutation = self.basis.clone();
        }
        let triangular_crash_can_fall_back = !has_external_basis
            && self.parameters.initial_basis == InitialBasisHeuristic::Triangular;
        let mut basis_factorization = match BasisRepresentation::new_with_parameters(
            self.current_basis_matrix(),
            &self.parameters,
        ) {
            Ok(factorization) => factorization,
            Err(_) if triangular_crash_can_fall_back => {
                // CreateInitialBasis() immediately tests TRIANGULAR's proposed
                // basis upstream and reverts to the all-slack basis when that
                // advanced crash is not factorizable.
                self.use_all_slack_basis();
                BasisRepresentation::new_with_parameters(
                    self.current_basis_matrix(),
                    &self.parameters,
                )?
            }
            Err(error) => return Err(error),
        };
        if basis_factorization.infinity_norm_condition_number_upper_bound()
            > self.parameters.initial_condition_number_threshold
        {
            self.use_all_slack_basis();
            basis_factorization = BasisRepresentation::new_with_parameters(
                self.current_basis_matrix(),
                &self.parameters,
            )?;
            if self.trace_enabled {
                self.initial_basis_before_permutation = self.basis.clone();
            }
        }
        if self.trace_enabled {
            self.initial_column_permutation = basis_factorization.column_permutation().to_vec();
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
        let info_before_advanced_basis = info.clone();
        info.change_unused_basic_variables_to_free(&self.basis);
        let variable_values = match self.compute_initial_values(&info, &basis_factorization) {
            Ok(values) => values,
            Err(_) if triangular_crash_can_fall_back => {
                // InitializeFirstBasis() also recomputes the basic values. A
                // numerical failure in that solve rejects TRIANGULAR's crash
                // just like a factorization or condition-number failure.
                self.use_all_slack_basis();
                basis_factorization = BasisRepresentation::new_with_parameters(
                    self.current_basis_matrix(),
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
        self.problem_status = ProblemStatus::Init;
        Ok(())
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

    fn current_basis_matrix(&self) -> SparseMatrix {
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
        basis_matrix
    }

    fn strengthen_lu_pivoting_after_early_imprecision(&mut self) -> Result<(), FactorizationError> {
        if self.basis_factorization.as_ref().unwrap().num_updates() < 10 {
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
                rhs[entry.index().to_usize()] -= entry.coefficient() * value;
            }
        }
        let basic = basis_factorization.solve(&rhs)?;
        for (row, &value) in basic.iter().enumerate() {
            variable_values[self.basis[RowIndex::from_usize(row)]] = value;
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

    fn compute_reduced_costs(&mut self, objective: &DenseRow) -> Result<(), FactorizationError> {
        let basic_objective: Vec<_> = (0..self.num_rows.to_usize())
            .map(|row| {
                let column = self.basis[RowIndex::from_usize(row)];
                objective[column] + self.cost_perturbations[column]
            })
            .collect();
        let dual = self
            .basis_factorization
            .as_ref()
            .unwrap()
            .transpose_solve(&basic_objective)?;
        self.dual_values = DenseColumn::from_vec(dual.clone());
        for column in 0..self.num_cols.to_usize() {
            let index = ColIndex::from_usize(column);
            let mut value = objective[index] + self.cost_perturbations[index];
            for entry in self.matrix.column(index) {
                value -= dual[entry.index().to_usize()] * entry.coefficient();
            }
            self.reduced_costs[index] = value;
        }
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
    }

    fn remove_cost_shifts(&mut self) {
        self.cost_perturbations.as_mut_slice().fill(0.0);
        self.has_cost_shift = false;
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
        let pricing_rule = match if phase == SimplexPhase::Feasibility {
            self.parameters.feasibility_rule
        } else {
            self.parameters.optimization_rule
        } {
            crate::parameters::PricingRule::Dantzig => EdgePricingRule::Dantzig,
            crate::parameters::PricingRule::SteepestEdge => EdgePricingRule::SteepestEdge,
            crate::parameters::PricingRule::Devex => EdgePricingRule::Devex,
        };
        self.primal_edge_norms
            .as_mut()
            .unwrap()
            .set_pricing_rule(pricing_rule);
        self.primal_prices.force_recomputation();
        let mut final_check_performed = false;
        let incremental_reduced_costs = phase == SimplexPhase::Optimization;
        let mut recompute_reduced_costs = true;
        let mut refactorize_for_precision = false;
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
            if refactorize_for_precision {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .force_refactorization()?;
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
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
                refactorize_for_precision = true;
                continue;
            }
            if self.basis_factorization.as_ref().unwrap().is_refactorized() {
                self.correct_errors_on_variable_values()?;
            }
            let phase_objective = self.phase_objective(phase);
            if !incremental_reduced_costs || recompute_reduced_costs {
                self.compute_reduced_costs(&phase_objective)?;
                self.primal_prices.force_recomputation();
                recompute_reduced_costs = false;
            }
            if phase == SimplexPhase::Optimization
                && self.basis_factorization.as_ref().unwrap().is_refactorized()
                && self.internal_objective() < self.primal_objective_limit
            {
                self.problem_status = ProblemStatus::PrimalFeasible;
                self.objective_limit_reached = true;
                return Ok(());
            }
            let Some(entering) = self.choose_entering()? else {
                // GLOP accepts an empty pricing set only after its FINAL_CHECK
                // has both precise reduced costs and a refactorized basis.
                // This provisional driver does not yet retain that precision
                // state, so perform the check once explicitly and price again.
                if !final_check_performed {
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .force_refactorization()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    self.primal_prices.force_recomputation();
                    recompute_reduced_costs = true;
                    final_check_performed = true;
                    continue;
                }
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
            final_check_performed = false;
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
                self.primal_prices.force_recomputation();
                refactorize_for_precision = true;
                continue;
            }
            if !entering_edge_norm_is_precise {
                let norms = self.primal_edge_norms.as_mut().unwrap().squared_norms(
                    self.basis_factorization.as_ref().unwrap(),
                    self.variables_info.as_ref().unwrap().relevance(),
                )?;
                self.primal_prices.recompute_price_at_from_values(
                    entering,
                    self.variables_info.as_ref().unwrap(),
                    self.reduced_costs.as_slice(),
                    norms,
                    self.parameters.dual_feasibility_tolerance,
                );
                continue;
            }
            let precise_reduced = phase_objective[entering]
                - (0..self.num_rows.to_usize())
                    .map(|row| {
                        phase_objective[self.basis[RowIndex::from_usize(row)]]
                            * direction.value(RowIndex::from_usize(row))
                    })
                    .sum::<f64>();
            let old_reduced = self.reduced_costs[entering];
            self.reduced_costs[entering] = precise_reduced;
            if incremental_reduced_costs {
                let scale = if precise_reduced.abs() <= 1.0 {
                    1.0
                } else {
                    precise_reduced
                };
                refactorize_for_precision |= ((old_reduced - precise_reduced) / scale).abs()
                    > self.parameters.recompute_reduced_costs_threshold;
            }
            {
                let norms = self.primal_edge_norms.as_mut().unwrap().squared_norms(
                    self.basis_factorization.as_ref().unwrap(),
                    self.variables_info.as_ref().unwrap().relevance(),
                )?;
                self.primal_prices.recompute_price_at_from_values(
                    entering,
                    self.variables_info.as_ref().unwrap(),
                    self.reduced_costs.as_slice(),
                    norms,
                    self.parameters.dual_feasibility_tolerance,
                );
            }
            {
                let info = self.variables_info.as_ref().unwrap();
                let valid_entering_candidate = (info.can_increase().contains(entering)
                    && precise_reduced < -self.parameters.dual_feasibility_tolerance)
                    || (info.can_decrease().contains(entering)
                        && precise_reduced > self.parameters.dual_feasibility_tolerance);
                if !valid_entering_candidate {
                    // Matches ReducedCosts::MakeReducedCostsPrecise() after
                    // TestEnteringReducedCostPrecision() changes the sign or
                    // feasibility of the selected column.
                    refactorize_for_precision = true;
                    continue;
                }
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
                LeavingChoice::BoundFlip { step } => (None, step, 0.0),
                LeavingChoice::Pivot {
                    row,
                    step,
                    target_bound,
                } => (Some(row), step, target_bound),
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
            for entry in &direction {
                self.variable_values[self.basis[entry.row()]] -= entry.coefficient() * step;
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
                        entering.to_usize(),
                        leaving.to_usize(),
                        row.to_usize(),
                        direction.values().as_slice(),
                        self.update_row.as_mut().unwrap(),
                    )?;
                if incremental_reduced_costs {
                    update_reduced_cost_values_before_basis_pivot(
                        self.reduced_costs.as_mut_slice(),
                        entering,
                        leaving,
                        direction.value(row),
                        self.update_row.as_ref().unwrap(),
                    );
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
                self.variable_values[leaving] = target_bound;
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
                        .replace_column_and_refactorize(
                            row.to_usize(),
                            self.matrix.column(entering).clone(),
                        )?;
                } else {
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .replace_column_after_solve(
                            entering.to_usize(),
                            row.to_usize(),
                            &direction,
                            self.matrix.column(entering).clone(),
                        )?;
                }
                self.incorporate_basis_permutation();
                self.update_row.as_mut().unwrap().invalidate();
                if self.basis_factorization.as_ref().unwrap().is_refactorized() {
                    recompute_reduced_costs = true;
                    self.primal_prices.force_recomputation();
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
        let mut reduced_costs_precise =
            self.basis_factorization.as_ref().unwrap().is_refactorized();
        let mut reduced_costs_recomputed = true;
        let mut prices_initialized = false;
        loop {
            if time_limit.limit_reached() {
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
            if self.basis_factorization.as_ref().unwrap().is_refactorized()
                && !reduced_costs_precise
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
                if self.has_cost_shift {
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
            let direction = self.direction(entering)?;
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
                    .replace_column_and_refactorize(
                        leaving_position,
                        self.matrix.column(entering).clone(),
                    )?;
            } else {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .replace_column_after_solve(
                        entering.to_usize(),
                        leaving_position,
                        &direction,
                        self.matrix.column(entering).clone(),
                    )?;
            }
            self.incorporate_basis_permutation();
            self.update_row.as_mut().unwrap().invalidate();
            reduced_costs_precise = false;
            if reduced_costs_recomputed
                || self.basis_factorization.as_ref().unwrap().is_refactorized()
                || self.dual_edge_norms.needs_basis_refactorization()
            {
                if self.basis_factorization.as_ref().unwrap().is_refactorized() {
                    self.dual_edge_norms.clear();
                }
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
        let mut reduced_costs_precise =
            self.basis_factorization.as_ref().unwrap().is_refactorized();
        self.recompute_dual_prices()?;
        loop {
            if time_limit.limit_reached()
                || self.num_iterations
                    >= u64::try_from(self.parameters.max_number_of_iterations).unwrap_or(u64::MAX)
            {
                self.problem_status = ProblemStatus::DualFeasible;
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
            }
            if self.basis_factorization.as_ref().unwrap().is_refactorized()
                && !reduced_costs_precise
            {
                let objective = self.objective.clone();
                self.compute_reduced_costs(&objective)?;
                self.initialize_values()?;
                reduced_costs_precise = true;
                self.recompute_dual_prices()?;
                if self.dual_objective_limit != f64::INFINITY
                    && self.internal_objective() > self.dual_objective_limit
                {
                    self.problem_status = ProblemStatus::DualFeasible;
                    self.objective_limit_reached = true;
                    return Ok(());
                }
            }
            if !self.bound_flip_candidates.is_empty() {
                let candidates = std::mem::take(&mut self.bound_flip_candidates);
                self.make_boxed_variables_dual_feasible(&candidates, true)?;
            }

            let Some(leaving_position) = self.dual_prices.get_maximum() else {
                if !self.basis_factorization.as_ref().unwrap().is_refactorized()
                    || !reduced_costs_precise
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
                    continue;
                }
                self.problem_status = ProblemStatus::Optimal;
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
            self.update_row.as_mut().unwrap().compute_update_row(
                self.basis_factorization.as_ref().unwrap(),
                &self.matrix,
                self.variables_info.as_ref().unwrap().relevance(),
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
                    self.basis_factorization
                        .as_mut()
                        .unwrap()
                        .force_refactorization()?;
                    self.incorporate_basis_permutation();
                    self.update_row.as_mut().unwrap().invalidate();
                    continue;
                }
                self.problem_status = ProblemStatus::DualUnbounded;
                self.dual_ray = DenseColumn::from_vec(
                    self.update_row
                        .as_ref()
                        .unwrap()
                        .unit_row_left_inverse()
                        .to_vec(),
                );
                if cost_variation < 0.0 {
                    for value in self.dual_ray.as_mut_slice() {
                        *value = -*value;
                    }
                }
                return Ok(());
            };

            let direction = self.direction(entering)?;
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
                continue;
            }
            if pivot.abs() <= 1e-20 {
                return Err(FactorizationError::Singular {
                    step: leaving_position,
                });
            }

            let entering_coefficient = self
                .update_row
                .as_ref()
                .unwrap()
                .coefficient(entering.to_usize());
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
                self.variable_values[self.basis[entry.row()]] -= entry.coefficient() * step;
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
                    .replace_column_and_refactorize(
                        leaving_position,
                        self.matrix.column(entering).clone(),
                    )?;
            } else {
                self.basis_factorization
                    .as_mut()
                    .unwrap()
                    .replace_column_after_solve(
                        entering.to_usize(),
                        leaving_position,
                        &direction,
                        self.matrix.column(entering).clone(),
                    )?;
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
                if self.basis_factorization.as_ref().unwrap().is_refactorized() {
                    self.dual_edge_norms.clear();
                }
                self.dual_prices.clear();
            } else {
                let changed_rows = if direction.non_zeros().is_empty() {
                    (0..self.num_rows.to_usize())
                        .map(RowIndex::from_usize)
                        .collect::<Vec<_>>()
                } else {
                    direction.non_zeros().to_vec()
                };
                self.update_dual_prices(&changed_rows)?;
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

    fn correct_errors_on_variable_values(&mut self) -> Result<(), FactorizationError> {
        let mut residual = vec![0.0; self.num_rows.to_usize()];
        for column in 0..self.num_cols.to_usize() {
            let column = ColIndex::from_usize(column);
            let value = self.variable_values[column];
            for entry in self.matrix.column(column) {
                residual[entry.index().to_usize()] += entry.coefficient() * value;
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
                rhs[entry.index().to_usize()] -= entry.coefficient() * value;
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
        let objective = self.objective.clone();
        self.compute_reduced_costs(&objective)?;
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
