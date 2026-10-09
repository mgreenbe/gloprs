//! Primal variable values and infeasibility prices.
//!
//! This follows `ortools/glop/variable_values.{h,cc}`. Rust passes the two
//! mutable collaborators to the methods that update them; the numerical loops
//! and their ordering otherwise correspond directly to upstream.

#![allow(
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::needless_range_loop,
    clippy::struct_field_names
)]

use lp_data::lp_types::{
    ColIndex, DenseRow, RowIndex, RowToColMapping, VariableStatus, VectorIndex,
};
use lp_data::scattered_vector::ScatteredColumn;
use lp_data::sparse::CompactSparseMatrix;

use crate::basis_representation::BasisRepresentation;
use crate::dual_edge_norms::DualEdgeNorms;
use crate::lu_factorization::FactorizationError;
use crate::parameters::GlopParameters;
use crate::pricing::DynamicMaximum;
use crate::variables_info::VariablesInfo;

#[derive(Debug)]
pub struct VariableValues<'a> {
    parameters: GlopParameters,
    matrix: &'a CompactSparseMatrix,
    basis: &'a RowToColMapping,
    variables_info: &'a VariablesInfo,
    basis_factorization: &'a BasisRepresentation,
    put_more_importance_on_norm: bool,
    variable_values: DenseRow,
}

impl<'a> VariableValues<'a> {
    #[must_use]
    pub fn new(
        parameters: &GlopParameters,
        matrix: &'a CompactSparseMatrix,
        basis: &'a RowToColMapping,
        variables_info: &'a VariablesInfo,
        basis_factorization: &'a BasisRepresentation,
    ) -> Self {
        Self {
            parameters: parameters.clone(),
            matrix,
            basis,
            variables_info,
            basis_factorization,
            put_more_importance_on_norm: false,
            variable_values: DenseRow::new(),
        }
    }

    #[must_use]
    pub fn get(&self, column: ColIndex) -> f64 {
        self.variable_values[column]
    }

    #[must_use]
    pub const fn dense_row(&self) -> &DenseRow {
        &self.variable_values
    }

    pub fn set(&mut self, column: ColIndex, value: f64) {
        self.variable_values.resize(self.matrix.num_cols(), 0.0);
        self.variable_values[column] = value;
    }

    pub fn set_non_basic_variable_value_from_status(&mut self, column: ColIndex) {
        self.variable_values.resize(self.matrix.num_cols(), 0.0);
        self.variable_values[column] = match self.variables_info.variable_statuses()[column] {
            VariableStatus::FixedValue | VariableStatus::AtLowerBound => {
                self.variables_info.lower_bounds()[column.to_usize()]
            }
            VariableStatus::AtUpperBound => self.variables_info.upper_bounds()[column.to_usize()],
            VariableStatus::Free => {
                debug_assert!(false, "must not reset a FREE variable individually");
                self.variable_values[column]
            }
            VariableStatus::Basic => {
                debug_assert!(false, "must not reset a BASIC variable");
                self.variable_values[column]
            }
        };
    }

    pub fn reset_all_non_basic_variable_values(&mut self, free_initial_values: &DenseRow) {
        self.variable_values.resize(self.matrix.num_cols(), 0.0);
        for index in 0..self.matrix.num_cols().to_usize() {
            let column = ColIndex::from_usize(index);
            self.variable_values[column] = match self.variables_info.variable_statuses()[column] {
                VariableStatus::FixedValue | VariableStatus::AtLowerBound => {
                    self.variables_info.lower_bounds()[index]
                }
                VariableStatus::AtUpperBound => self.variables_info.upper_bounds()[index],
                VariableStatus::Free => free_initial_values
                    .as_slice()
                    .get(index)
                    .copied()
                    .unwrap_or(0.0),
                VariableStatus::Basic => self.variable_values[column],
            };
        }
    }

    /// Recomputes `x_B = -B^-1 A_N x_N`, in upstream column order.
    pub fn recompute_basic_variable_values(
        &mut self,
        dual_prices: &mut DynamicMaximum,
    ) -> Result<(), FactorizationError> {
        let mut rhs = lp_data::lp_types::DenseColumn::filled(self.matrix.num_rows(), 0.0);
        for column in self.variables_info.not_basic().iter_ones() {
            let value = self.variable_values[column];
            self.matrix
                .column_add_multiple_to_dense_column(column, -value, &mut rhs);
        }
        let basic = self.basis_factorization.solve(rhs.as_slice())?;
        for (row, &value) in basic.iter().enumerate() {
            self.variable_values[self.basis[RowIndex::from_usize(row)]] = value;
        }
        dual_prices.clear();
        Ok(())
    }

    #[must_use]
    pub fn compute_maximum_primal_residual(&self) -> f64 {
        let mut residual = lp_data::lp_types::DenseColumn::filled(self.matrix.num_rows(), 0.0);
        for index in 0..self.matrix.num_cols().to_usize() {
            let column = ColIndex::from_usize(index);
            self.matrix.column_add_multiple_to_dense_column(
                column,
                self.variable_values[column],
                &mut residual,
            );
        }
        residual
            .as_slice()
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
    }

    #[must_use]
    pub fn compute_maximum_primal_infeasibility(&self) -> f64 {
        (0..self.matrix.num_cols().to_usize())
            .map(|index| self.column_infeasibility(index))
            .fold(0.0, f64::max)
    }

    #[must_use]
    pub fn compute_sum_of_primal_infeasibilities(&self) -> f64 {
        (0..self.matrix.num_cols().to_usize())
            .map(|index| self.column_infeasibility(index).max(0.0))
            .sum()
    }

    pub fn update_on_pivoting(
        &mut self,
        direction: &ScatteredColumn,
        entering_column: ColIndex,
        step: f64,
    ) {
        debug_assert!(step.is_finite());
        for entry in direction {
            let column = self.basis[entry.row()];
            self.variable_values[column] =
                (-entry.coefficient()).mul_add(step, self.variable_values[column]);
        }
        self.variable_values[entering_column] += step;
    }

    pub fn update_given_non_basic_variables(
        &mut self,
        columns_to_update: &[ColIndex],
        update_basic_variables: bool,
        dual_edge_norms: &mut DualEdgeNorms,
        dual_prices: &mut DynamicMaximum,
    ) -> Result<(), FactorizationError> {
        if !update_basic_variables {
            for &column in columns_to_update {
                self.set_non_basic_variable_value_from_status(column);
            }
            return Ok(());
        }
        let mut rhs = ScatteredColumn::new(self.matrix.num_rows());
        let mut use_dense = false;
        for &column in columns_to_update {
            let old_value = self.variable_values[column];
            self.set_non_basic_variable_value_from_status(column);
            let multiplier = self.variable_values[column] - old_value;
            if use_dense {
                self.matrix.column_add_multiple_to_dense_column(
                    column,
                    multiplier,
                    rhs.values_mut(),
                );
            } else {
                self.matrix
                    .column_add_multiple_to_scattered_column(column, multiplier, &mut rhs);
                use_dense = rhs.should_use_dense_iteration(0.8);
            }
        }
        rhs.clear_sparse_mask();
        rhs.clear_non_zeros_if_too_dense(0.8);
        self.basis_factorization.solve_with_nonzeros(&mut rhs)?;
        if rhs.non_zeros().is_empty() {
            for row in 0..self.matrix.num_rows().to_usize() {
                let row = RowIndex::from_usize(row);
                self.variable_values[self.basis[row]] -= rhs.value(row);
            }
            self.recompute_dual_prices(dual_edge_norms, dual_prices, false)?;
        } else {
            let changed_rows = rhs.non_zeros().to_vec();
            for &row in &changed_rows {
                self.variable_values[self.basis[row]] -= rhs.value(row);
            }
            self.update_dual_prices(&changed_rows, dual_edge_norms, dual_prices)?;
        }
        Ok(())
    }

    pub fn update_primal_phase_one_costs(
        &self,
        rows: impl IntoIterator<Item = RowIndex>,
        objective: &mut DenseRow,
    ) -> bool {
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let mut changed = false;
        for row in rows {
            let column = self.basis[row];
            let value = self.variable_values[column];
            let index = column.to_usize();
            let new_cost = if value - self.variables_info.upper_bounds()[index] > tolerance {
                1.0
            } else if self.variables_info.lower_bounds()[index] - value > tolerance {
                -1.0
            } else {
                0.0
            };
            if new_cost != objective[column] {
                objective[column] = new_cost;
                changed = true;
            }
        }
        changed
    }

    pub fn recompute_dual_prices(
        &mut self,
        dual_edge_norms: &mut DualEdgeNorms,
        dual_prices: &mut DynamicMaximum,
        put_more_importance_on_norm: bool,
    ) -> Result<(), FactorizationError> {
        dual_prices.clear_and_resize(self.matrix.num_rows().to_usize());
        dual_prices.start_dense_updates();
        self.put_more_importance_on_norm = put_more_importance_on_norm;
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let norms = dual_edge_norms.edge_squared_norms(self.basis_factorization)?;
        for row_index in 0..self.matrix.num_rows().to_usize() {
            let column = self.basis[RowIndex::from_usize(row_index)];
            let infeasibility = self.column_infeasibility(column.to_usize());
            if infeasibility > tolerance {
                let price = if put_more_importance_on_norm {
                    infeasibility.abs() / norms[row_index]
                } else {
                    infeasibility * infeasibility / norms[row_index]
                };
                dual_prices.dense_add_or_update(row_index, price);
            }
        }
        Ok(())
    }

    pub fn update_dual_prices(
        &mut self,
        rows: &[RowIndex],
        dual_edge_norms: &mut DualEdgeNorms,
        dual_prices: &mut DynamicMaximum,
    ) -> Result<(), FactorizationError> {
        if dual_prices.size() != self.matrix.num_rows().to_usize() {
            return self.recompute_dual_prices(
                dual_edge_norms,
                dual_prices,
                self.put_more_importance_on_norm,
            );
        }
        let tolerance = self.parameters.primal_feasibility_tolerance;
        let norms = dual_edge_norms.edge_squared_norms(self.basis_factorization)?;
        for &row in rows {
            let column = self.basis[row];
            let infeasibility = self.column_infeasibility(column.to_usize());
            if infeasibility > tolerance {
                let price = if self.put_more_importance_on_norm {
                    infeasibility.abs() / norms[row.to_usize()]
                } else {
                    infeasibility * infeasibility / norms[row.to_usize()]
                };
                dual_prices.add_or_update(row.to_usize(), price);
            } else {
                dual_prices.remove(row.to_usize());
            }
        }
        Ok(())
    }

    fn column_infeasibility(&self, index: usize) -> f64 {
        let column = ColIndex::from_usize(index);
        (self.variable_values[column] - self.variables_info.upper_bounds()[index])
            .max(self.variables_info.lower_bounds()[index] - self.variable_values[column])
    }
}
