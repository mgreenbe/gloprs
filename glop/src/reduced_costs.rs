//! Reduced costs, dual values, and primal pricing.
//!
//! This follows `ortools/glop/reduced_costs.{h,cc}`. Objective storage is
//! owned here so Rust can safely apply GLOP's perturbations and cost shifts;
//! all formulas and update ordering match the upstream implementation.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::struct_excessive_bools,
    clippy::struct_field_names
)]

use lp_data::lp_types::{ColIndex, DenseRow, RowIndex, RowToColMapping, VariableType, VectorIndex};
use lp_data::scattered_vector::ScatteredColumn;
use lp_data::sparse::CompactSparseMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;
use crate::parameters::GlopParameters;
use crate::pricing::DynamicMaximum;
use crate::primal_edge_norms::PrimalEdgeNorms;
use crate::update_row::UpdateRow;
use crate::variables_info::VariablesInfo;

#[derive(Debug)]
pub struct ReducedCosts<'a> {
    matrix: &'a CompactSparseMatrix,
    objective: DenseRow,
    basis: &'a RowToColMapping,
    variables_info: &'a VariablesInfo,
    basis_factorization: &'a BasisRepresentation,
    parameters: GlopParameters,
    must_refactorize_basis: bool,
    recompute_basic_objective_left_inverse: bool,
    recompute_basic_objective: bool,
    recompute_reduced_costs: bool,
    are_reduced_costs_precise: bool,
    are_reduced_costs_recomputed: bool,
    has_cost_shift: bool,
    basic_objective: Vec<f64>,
    cost_perturbations: Vec<f64>,
    reduced_costs: Vec<f64>,
    basic_objective_left_inverse: Vec<f64>,
    dual_feasibility_tolerance: f64,
    random: StdRng,
    deterministic_time: f64,
}

impl<'a> ReducedCosts<'a> {
    #[must_use]
    pub fn new(
        matrix: &'a CompactSparseMatrix,
        objective: &DenseRow,
        basis: &'a RowToColMapping,
        variables_info: &'a VariablesInfo,
        basis_factorization: &'a BasisRepresentation,
        seed: u64,
    ) -> Self {
        Self {
            matrix,
            objective: objective.clone(),
            basis,
            variables_info,
            basis_factorization,
            parameters: GlopParameters::default(),
            must_refactorize_basis: false,
            recompute_basic_objective_left_inverse: true,
            recompute_basic_objective: true,
            recompute_reduced_costs: true,
            are_reduced_costs_precise: false,
            are_reduced_costs_recomputed: false,
            has_cost_shift: false,
            basic_objective: Vec::new(),
            cost_perturbations: Vec::new(),
            reduced_costs: Vec::new(),
            basic_objective_left_inverse: Vec::new(),
            dual_feasibility_tolerance: 0.0,
            random: StdRng::seed_from_u64(seed),
            deterministic_time: 0.0,
        }
    }

    pub fn set_parameters(&mut self, parameters: &GlopParameters) {
        self.parameters = parameters.clone();
    }

    #[must_use]
    pub const fn needs_basis_refactorization(&self) -> bool {
        self.must_refactorize_basis
    }

    #[must_use]
    pub const fn are_reduced_costs_precise(&self) -> bool {
        self.are_reduced_costs_precise
    }

    #[must_use]
    pub const fn are_reduced_costs_recomputed(&self) -> bool {
        self.recompute_reduced_costs || self.are_reduced_costs_recomputed
    }

    pub fn reduced_costs(&mut self) -> Result<&[f64], FactorizationError> {
        if self.basis_factorization.is_refactorized() {
            self.must_refactorize_basis = false;
        }
        if self.recompute_reduced_costs {
            self.compute_reduced_costs()?;
        }
        Ok(&self.reduced_costs)
    }

    pub fn full_reduced_costs(&mut self) -> Result<&[f64], FactorizationError> {
        if !self.are_reduced_costs_recomputed {
            self.recompute_reduced_costs = true;
        }
        self.reduced_costs()
    }

    pub fn dual_values(&mut self) -> Result<&[f64], FactorizationError> {
        self.compute_basic_objective_left_inverse()?;
        Ok(&self.basic_objective_left_inverse)
    }

    pub fn test_entering_reduced_cost_precision(
        &mut self,
        entering_column: ColIndex,
        direction: &ScatteredColumn,
    ) -> Result<f64, FactorizationError> {
        if self.recompute_basic_objective {
            self.compute_basic_objective();
        }
        let precise = self.objective[entering_column]
            + self.cost_perturbations[entering_column.to_usize()]
            - self
                .basic_objective
                .iter()
                .enumerate()
                .map(|(row, value)| *value * direction.value(RowIndex::from_usize(row)))
                .sum::<f64>();
        let old = self.reduced_costs[entering_column.to_usize()];
        self.reduced_costs[entering_column.to_usize()] = precise;
        if !self.recompute_reduced_costs {
            let scale = if precise.abs() <= 1.0 { 1.0 } else { precise };
            if ((old - precise) / scale).abs() > self.parameters.recompute_reduced_costs_threshold {
                self.make_reduced_costs_precise();
            }
        }
        Ok(precise)
    }

    pub fn compute_maximum_dual_residual(&mut self) -> Result<f64, FactorizationError> {
        let dual = self.dual_values()?.to_vec();
        let mut maximum = 0.0_f64;
        for row in 0..self.matrix.num_rows().to_usize() {
            let column = self.basis[RowIndex::from_usize(row)];
            let value = self.objective[column] + self.cost_perturbations[column.to_usize()]
                - self
                    .matrix
                    .column(column)
                    .iter()
                    .map(|(entry_row, coefficient)| dual[entry_row.to_usize()] * coefficient)
                    .sum::<f64>();
            maximum = maximum.max(value.abs());
        }
        Ok(maximum)
    }

    pub fn compute_maximum_dual_infeasibility(&mut self) -> Result<f64, FactorizationError> {
        self.compute_dual_infeasibility(false, false)
    }

    pub fn compute_maximum_dual_infeasibility_on_non_boxed_variables(
        &mut self,
    ) -> Result<f64, FactorizationError> {
        self.compute_dual_infeasibility(true, false)
    }

    pub fn compute_sum_of_dual_infeasibilities(&mut self) -> Result<f64, FactorizationError> {
        self.compute_dual_infeasibility(false, true)
    }

    fn compute_dual_infeasibility(
        &mut self,
        exclude_boxed: bool,
        sum: bool,
    ) -> Result<f64, FactorizationError> {
        let reduced = self.reduced_costs()?.to_vec();
        let mut result = 0.0_f64;
        for column in self.variables_info.relevance().iter_ones() {
            if exclude_boxed
                && self
                    .variables_info
                    .non_basic_boxed_variables()
                    .contains(column)
            {
                continue;
            }
            let value = reduced[column.to_usize()];
            if (self.variables_info.can_increase().contains(column) && value < 0.0)
                || (self.variables_info.can_decrease().contains(column) && value > 0.0)
            {
                if sum {
                    result += value.abs();
                } else {
                    result = result.max(value.abs());
                }
            }
        }
        Ok(result)
    }

    pub fn update_before_basis_pivot(
        &mut self,
        entering_column: ColIndex,
        leaving_row: RowIndex,
        direction: &ScatteredColumn,
        update_row: &UpdateRow,
    ) {
        let leaving_column = self.basis[leaving_row];
        if !self.recompute_reduced_costs {
            self.update_reduced_costs(
                entering_column,
                leaving_column,
                direction.value(leaving_row),
                update_row,
            );
        }
        self.update_basic_objective(entering_column, leaving_row);
    }

    pub fn set_non_basic_variable_cost_to_zero(&mut self, column: ColIndex) {
        self.reduced_costs[column.to_usize()] -= self.objective[column];
        self.objective[column] = 0.0;
    }

    pub fn make_reduced_costs_precise(&mut self) {
        if self.are_reduced_costs_precise {
            return;
        }
        self.must_refactorize_basis = true;
        self.recompute_basic_objective_left_inverse = true;
        self.recompute_reduced_costs = true;
    }

    pub fn perturb_costs(&mut self) {
        let structural_size = self.matrix.num_cols().to_usize() - self.matrix.num_rows().to_usize();
        let maximum = (0..structural_size)
            .map(|column| self.objective[ColIndex::from_usize(column)].abs())
            .fold(0.0, f64::max);
        self.cost_perturbations = vec![0.0; self.matrix.num_cols().to_usize()];
        for column in 0..structural_size {
            let index = ColIndex::from_usize(column);
            let objective = self.objective[index];
            let magnitude = (1.0 + self.random.random::<f64>())
                * (self.parameters.relative_cost_perturbation * objective.abs()
                    + self.parameters.relative_max_cost_perturbation * maximum);
            self.cost_perturbations[column] = match self.variables_info.variable_types()[index] {
                VariableType::LowerBounded => magnitude,
                VariableType::UpperBounded => -magnitude,
                VariableType::UpperAndLowerBounded if objective > 0.0 => magnitude,
                VariableType::UpperAndLowerBounded if objective < 0.0 => -magnitude,
                _ => 0.0,
            };
        }
    }

    pub fn shift_cost_if_needed(&mut self, increasing_reduced_cost_needed: bool, column: ColIndex) {
        let minimum_delta =
            self.parameters.degenerate_ministep_factor * self.dual_feasibility_tolerance;
        let value = self.reduced_costs[column.to_usize()];
        if increasing_reduced_cost_needed && value <= -minimum_delta {
            return;
        }
        if !increasing_reduced_cost_needed && value >= minimum_delta {
            return;
        }
        let delta = if increasing_reduced_cost_needed {
            minimum_delta
        } else {
            -minimum_delta
        };
        self.cost_perturbations[column.to_usize()] -= value + delta;
        self.reduced_costs[column.to_usize()] = -delta;
        self.has_cost_shift = true;
    }

    #[must_use]
    pub fn step_is_dual_degenerate(
        &self,
        increasing_reduced_cost_needed: bool,
        column: ColIndex,
    ) -> bool {
        let value = self.reduced_costs[column.to_usize()];
        (increasing_reduced_cost_needed && value >= 0.0)
            || (!increasing_reduced_cost_needed && value <= 0.0)
    }

    #[must_use]
    pub const fn has_cost_shift(&self) -> bool {
        self.has_cost_shift
    }

    pub fn clear_and_remove_cost_shifts(&mut self) {
        self.has_cost_shift = false;
        self.cost_perturbations = vec![0.0; self.matrix.num_cols().to_usize()];
        self.recompute_basic_objective = true;
        self.recompute_basic_objective_left_inverse = true;
        self.are_reduced_costs_precise = false;
        self.recompute_reduced_costs = true;
    }

    pub fn reset_for_new_objective(&mut self, objective: &DenseRow) {
        self.objective.clone_from(objective);
        self.recompute_basic_objective = true;
        self.recompute_basic_objective_left_inverse = true;
        self.are_reduced_costs_precise = false;
        self.recompute_reduced_costs = true;
    }

    pub fn update_data_on_basis_permutation(&mut self) {
        self.recompute_basic_objective = true;
        self.recompute_basic_objective_left_inverse = true;
    }

    #[must_use]
    pub const fn dual_feasibility_tolerance(&self) -> f64 {
        self.dual_feasibility_tolerance
    }

    #[must_use]
    pub fn is_valid_primal_entering_candidate(&self, column: ColIndex) -> bool {
        let value = self.reduced_costs[column.to_usize()];
        (self.variables_info.can_increase().contains(column)
            && value < -self.dual_feasibility_tolerance)
            || (self.variables_info.can_decrease().contains(column)
                && value > self.dual_feasibility_tolerance)
    }

    #[must_use]
    pub fn cost_perturbations(&self) -> &[f64] {
        &self.cost_perturbations
    }

    #[must_use]
    pub const fn deterministic_time(&self) -> f64 {
        self.deterministic_time
    }

    fn compute_basic_objective(&mut self) {
        let rows = self.matrix.num_rows().to_usize();
        self.cost_perturbations
            .resize(self.matrix.num_cols().to_usize(), 0.0);
        self.basic_objective.resize(rows, 0.0);
        for row in 0..rows {
            let column = self.basis[RowIndex::from_usize(row)];
            self.basic_objective[row] =
                self.objective[column] + self.cost_perturbations[column.to_usize()];
        }
        self.recompute_basic_objective = false;
        self.recompute_basic_objective_left_inverse = true;
    }

    fn compute_basic_objective_left_inverse(&mut self) -> Result<(), FactorizationError> {
        if !self.recompute_basic_objective_left_inverse {
            return Ok(());
        }
        if self.recompute_basic_objective {
            self.compute_basic_objective();
        }
        self.basic_objective_left_inverse = self
            .basis_factorization
            .transpose_solve(&self.basic_objective)?;
        self.recompute_basic_objective_left_inverse = false;
        Ok(())
    }

    fn compute_reduced_costs(&mut self) -> Result<(), FactorizationError> {
        self.compute_basic_objective_left_inverse()?;
        let columns = self.matrix.num_cols().to_usize();
        let first_slack = columns - self.matrix.num_rows().to_usize();
        self.reduced_costs.resize(columns, 0.0);
        let mut residual = 0.0_f64;
        for column in 0..first_slack {
            let index = ColIndex::from_usize(column);
            self.reduced_costs[column] = self.objective[index] + self.cost_perturbations[column]
                - self
                    .matrix
                    .column(index)
                    .iter()
                    .map(|(row, value)| self.basic_objective_left_inverse[row.to_usize()] * value)
                    .sum::<f64>();
            if self.variables_info.is_basic().contains(index) {
                residual = residual.max(self.reduced_costs[column].abs());
            }
        }
        for column in first_slack..columns {
            let index = ColIndex::from_usize(column);
            self.reduced_costs[column] = self.objective[index] + self.cost_perturbations[column]
                - self.basic_objective_left_inverse[column - first_slack];
            if self.variables_info.is_basic().contains(index) {
                residual = residual.max(self.reduced_costs[column].abs());
            }
        }
        self.deterministic_time += lp_data::lp_types::deterministic_time_for_fp_operations(
            self.matrix.num_entries().value(),
        );
        self.recompute_reduced_costs = false;
        self.are_reduced_costs_recomputed = true;
        self.are_reduced_costs_precise = self.basis_factorization.is_refactorized();
        self.dual_feasibility_tolerance = self.parameters.dual_feasibility_tolerance.max(residual);
        Ok(())
    }

    fn update_reduced_costs(
        &mut self,
        entering_column: ColIndex,
        leaving_column: ColIndex,
        pivot: f64,
        update_row: &UpdateRow,
    ) {
        debug_assert_ne!(pivot, 0.0);
        let entering_cost = self.reduced_costs[entering_column.to_usize()];
        if entering_cost == 0.0 {
            self.are_reduced_costs_precise = false;
            return;
        }
        self.are_reduced_costs_recomputed = false;
        self.are_reduced_costs_precise = false;
        let new_leaving_cost = entering_cost / -pivot;
        for &column in update_row.non_zero_positions() {
            self.reduced_costs[column] += new_leaving_cost * update_row.coefficient(column);
        }
        self.reduced_costs[leaving_column.to_usize()] = new_leaving_cost;
        self.reduced_costs[entering_column.to_usize()] = 0.0;
    }

    fn update_basic_objective(&mut self, entering_column: ColIndex, leaving_row: RowIndex) {
        self.basic_objective[leaving_row.to_usize()] =
            self.objective[entering_column] + self.cost_perturbations[entering_column.to_usize()];
        self.recompute_basic_objective_left_inverse = true;
    }
}

#[derive(Debug)]
pub struct PrimalPrices {
    recompute: bool,
    prices: DynamicMaximum,
}

impl PrimalPrices {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            recompute: true,
            prices: DynamicMaximum::new(seed),
        }
    }

    pub fn force_recomputation(&mut self) {
        self.recompute = true;
    }

    pub fn best_entering_column(
        &mut self,
        variables_info: &VariablesInfo,
        basis: &BasisRepresentation,
        primal_edge_norms: &mut PrimalEdgeNorms,
        reduced_costs: &mut ReducedCosts<'_>,
    ) -> Result<Option<ColIndex>, FactorizationError> {
        if self.recompute {
            let reduced = reduced_costs.reduced_costs()?.to_vec();
            let norms = primal_edge_norms.squared_norms(basis, variables_info.relevance())?;
            self.prices.clear_and_resize(reduced.len());
            let tolerance = reduced_costs.dual_feasibility_tolerance();
            for column in variables_info.relevance().iter_ones() {
                let value = reduced[column.to_usize()];
                let infeasible = (variables_info.can_decrease().contains(column)
                    && value > tolerance)
                    || (variables_info.can_increase().contains(column) && value < -tolerance);
                if infeasible {
                    self.prices
                        .add_or_update(column.to_usize(), value * value / norms[column.to_usize()]);
                }
            }
            self.recompute = false;
        }
        Ok(self.prices.get_maximum().map(ColIndex::from_usize))
    }

    pub fn recompute_price_at(
        &mut self,
        column: ColIndex,
        variables_info: &VariablesInfo,
        basis: &BasisRepresentation,
        primal_edge_norms: &mut PrimalEdgeNorms,
        reduced_costs: &mut ReducedCosts<'_>,
    ) -> Result<(), FactorizationError> {
        if self.recompute {
            return Ok(());
        }
        if reduced_costs.is_valid_primal_entering_candidate(column) {
            let reduced = reduced_costs.reduced_costs()?[column.to_usize()];
            let norms = primal_edge_norms.squared_norms(basis, variables_info.relevance())?;
            self.prices.add_or_update(
                column.to_usize(),
                reduced * reduced / norms[column.to_usize()],
            );
        } else {
            self.prices.remove(column.to_usize());
        }
        Ok(())
    }

    pub fn update_before_basis_pivot(
        &mut self,
        entering_column: ColIndex,
        update_row: &UpdateRow,
        variables_info: &VariablesInfo,
        basis: &BasisRepresentation,
        primal_edge_norms: &mut PrimalEdgeNorms,
        reduced_costs: &mut ReducedCosts<'_>,
    ) -> Result<(), FactorizationError> {
        if self.recompute {
            return Ok(());
        }
        for &column in update_row.non_zero_positions() {
            self.recompute_price_at(
                ColIndex::from_usize(column),
                variables_info,
                basis,
                primal_edge_norms,
                reduced_costs,
            )?;
        }
        self.prices.remove(entering_column.to_usize());
        Ok(())
    }
}
