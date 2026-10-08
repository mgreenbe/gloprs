//! Scaling helpers from `ortools/lp_data/lp_data_utils.{h,cc}`.

use crate::lp_data::{CostScalingAlgorithm, LinearProgram};
use crate::lp_types::{ColIndex, DenseRow, Fractional, INFINITY, RowIndex, VectorIndex};
use crate::matrix_scaler::SparseMatrixScaler;
use crate::scattered_vector::{ScatteredColumn, ScatteredRow};

#[derive(Clone, Debug)]
pub struct LpScalingHelper {
    matrix_is_scaled: bool,
    row_unscaling_factors: Vec<Fractional>,
    col_unscaling_factors: Vec<Fractional>,
    bound_scaling_factor: Fractional,
    objective_scaling_factor: Fractional,
}

impl Default for LpScalingHelper {
    fn default() -> Self {
        Self {
            matrix_is_scaled: false,
            row_unscaling_factors: Vec::new(),
            col_unscaling_factors: Vec::new(),
            bound_scaling_factor: 1.0,
            objective_scaling_factor: 1.0,
        }
    }
}

#[allow(clippy::float_cmp)] // GLOP tests exact zero, infinity, and identity factors.
impl LpScalingHelper {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.matrix_is_scaled = false;
        self.bound_scaling_factor = 1.0;
        self.objective_scaling_factor = 1.0;
    }

    /// Scales a model with GLOP's default matrix, bound, and cost algorithms.
    pub fn scale(&mut self, lp: &mut LinearProgram) {
        let mut scaler = SparseMatrixScaler::new();
        lp.scale(&mut scaler);
        self.bound_scaling_factor = 1.0 / lp.scale_bounds();
        self.objective_scaling_factor =
            1.0 / lp.scale_objective(CostScalingAlgorithm::ContainOneCostScaling);
        self.matrix_is_scaled = true;
        self.row_unscaling_factors = scaler.row_scales().as_slice().to_vec();
        self.col_unscaling_factors = scaler.col_scales().as_slice().to_vec();
    }

    pub fn configure_from_factors(&mut self, row_factors: &[f64], col_factors: &[f64]) {
        self.matrix_is_scaled = true;
        self.row_unscaling_factors = row_factors.iter().map(|factor| 1.0 / factor).collect();
        self.col_unscaling_factors = col_factors.iter().map(|factor| 1.0 / factor).collect();
    }

    #[must_use]
    pub fn scale_variable_value(&self, col: ColIndex, value: Fractional) -> Fractional {
        value * self.col_unscaling_factor(col) * self.bound_scaling_factor
    }

    #[must_use]
    pub fn scale_reduced_cost(&self, col: ColIndex, value: Fractional) -> Fractional {
        value / self.col_unscaling_factor(col) * self.objective_scaling_factor
    }

    #[must_use]
    pub fn scale_dual_value(&self, row: RowIndex, value: Fractional) -> Fractional {
        value * (self.row_unscaling_factor(row) * self.objective_scaling_factor)
    }

    #[must_use]
    pub fn scale_constraint_activity(&self, row: RowIndex, value: Fractional) -> Fractional {
        value / self.row_unscaling_factor(row) * self.bound_scaling_factor
    }

    #[must_use]
    pub fn unscale_variable_value(&self, col: ColIndex, value: Fractional) -> Fractional {
        value / (self.col_unscaling_factor(col) * self.bound_scaling_factor)
    }

    #[must_use]
    pub fn unscale_reduced_cost(&self, col: ColIndex, value: Fractional) -> Fractional {
        value * self.col_unscaling_factor(col) / self.objective_scaling_factor
    }

    #[must_use]
    pub fn unscale_dual_value(&self, row: RowIndex, value: Fractional) -> Fractional {
        value / (self.row_unscaling_factor(row) * self.objective_scaling_factor)
    }

    #[must_use]
    pub fn unscale_left_solve_value(&self, row: RowIndex, value: Fractional) -> Fractional {
        value / self.row_unscaling_factor(row)
    }

    #[must_use]
    pub fn unscale_constraint_activity(&self, row: RowIndex, value: Fractional) -> Fractional {
        value * self.row_unscaling_factor(row) / self.bound_scaling_factor
    }

    pub fn unscale_unit_row_left_solve(
        &self,
        basis_col: ColIndex,
        left_inverse: &mut ScatteredRow,
    ) {
        let global_factor = self.col_unscaling_factor(basis_col);
        if left_inverse.non_zeros().is_empty() {
            for index in 0..left_inverse.len().to_usize() {
                let col = ColIndex::from_usize(index);
                let divisor =
                    self.row_unscaling_factor(RowIndex::from_usize(index)) * global_factor;
                left_inverse.values_mut()[col] /= divisor;
            }
        } else {
            let positions = left_inverse.non_zeros().to_vec();
            for col in positions {
                let divisor =
                    self.row_unscaling_factor(RowIndex::from_usize(col.to_usize())) * global_factor;
                left_inverse.values_mut()[col] /= divisor;
            }
        }
    }

    pub fn unscale_column_right_solve(
        &self,
        basis: &[ColIndex],
        col: ColIndex,
        right_inverse: &mut ScatteredColumn,
    ) {
        let global_factor = 1.0 / self.col_unscaling_factor(col);
        if right_inverse.non_zeros().is_empty() {
            for (index, &basis_col) in basis
                .iter()
                .enumerate()
                .take(right_inverse.len().to_usize())
            {
                let row = RowIndex::from_usize(index);
                right_inverse.values_mut()[row] /=
                    self.col_unscaling_factor(basis_col) * global_factor;
            }
        } else {
            let positions = right_inverse.non_zeros().to_vec();
            for row in positions {
                right_inverse.values_mut()[row] /=
                    self.col_unscaling_factor(basis[row.to_usize()]) * global_factor;
            }
        }
    }

    #[must_use]
    pub fn variable_scaling_factor(&self, col: ColIndex) -> Fractional {
        self.col_unscaling_factor(col) * self.bound_scaling_factor
    }

    #[must_use]
    pub fn variable_scaling_factor_with_slack(&self, col: ColIndex) -> Fractional {
        if !self.matrix_is_scaled {
            return self.bound_scaling_factor;
        }
        if col.to_usize() < self.col_unscaling_factors.len() {
            return self.col_unscaling_factors[col.to_usize()] * self.bound_scaling_factor;
        }
        self.row_unscaling_factors[col.to_usize() - self.col_unscaling_factors.len()]
            * self.bound_scaling_factor
    }

    pub fn average_cost_scaling(&mut self, objective: &mut DenseRow) {
        let mut sum = 0.0;
        let mut count = 0;
        for &value in objective.as_slice() {
            if value != 0.0 {
                count += 1;
                sum += value.abs();
            }
        }
        if count == 0 {
            self.objective_scaling_factor = 1.0;
            return;
        }
        self.objective_scaling_factor = 1.0 / (sum / f64::from(count));
        for value in objective.as_mut_slice() {
            *value *= self.objective_scaling_factor;
        }
    }

    pub fn contain_one_bound_scaling(
        &mut self,
        upper_bounds: &mut DenseRow,
        lower_bounds: &mut DenseRow,
    ) {
        let mut minimum = INFINITY;
        let mut maximum: Fractional = 0.0;
        for &value in lower_bounds
            .as_slice()
            .iter()
            .chain(upper_bounds.as_slice())
        {
            let magnitude = value.abs();
            if magnitude != 0.0 && magnitude != INFINITY {
                minimum = minimum.min(magnitude);
                maximum = maximum.max(magnitude);
            }
        }
        self.bound_scaling_factor = 1.0;
        if minimum != INFINITY {
            if minimum > 1.0 {
                self.bound_scaling_factor = 1.0 / minimum;
            } else if maximum < 1.0 {
                self.bound_scaling_factor = 1.0 / maximum;
            }
        }
        if self.bound_scaling_factor != 1.0 {
            for value in lower_bounds
                .as_mut_slice()
                .iter_mut()
                .chain(upper_bounds.as_mut_slice())
            {
                *value *= self.bound_scaling_factor;
            }
        }
    }

    #[must_use]
    pub const fn bounds_scaling_factor(&self) -> Fractional {
        self.bound_scaling_factor
    }

    #[must_use]
    pub const fn objective_scaling_factor(&self) -> Fractional {
        self.objective_scaling_factor
    }

    fn row_unscaling_factor(&self, row: RowIndex) -> Fractional {
        if self.matrix_is_scaled {
            self.row_unscaling_factors[row.to_usize()]
        } else {
            1.0
        }
    }

    fn col_unscaling_factor(&self, col: ColIndex) -> Fractional {
        if self.matrix_is_scaled {
            self.col_unscaling_factors[col.to_usize()]
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn configured_scaling_round_trips_scalar_domains() {
        let mut helper = LpScalingHelper::new();
        helper.configure_from_factors(&[2.0], &[4.0]);
        let col = ColIndex::new(0);
        let row = RowIndex::new(0);
        assert_eq!(
            helper.unscale_variable_value(col, helper.scale_variable_value(col, 3.0)),
            3.0
        );
        assert_eq!(
            helper.unscale_reduced_cost(col, helper.scale_reduced_cost(col, 3.0)),
            3.0
        );
        assert_eq!(
            helper.unscale_dual_value(row, helper.scale_dual_value(row, 3.0)),
            3.0
        );
    }
}
