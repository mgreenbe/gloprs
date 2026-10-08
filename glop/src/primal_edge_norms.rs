//! Primal steepest-edge and Devex norm maintenance.

use lp_data::lp_types::{ColBitVec, ColIndex, VectorIndex, deterministic_time_for_fp_operations};
use lp_data::lp_utils::{precise_squared_norm, squared_norm};
use lp_data::sparse::SparseMatrix;

use crate::basis_representation::BasisRepresentation;
use crate::lu_factorization::FactorizationError;
use crate::parameters::GlopParameters;
use crate::stats::{DistributionKind, StatsGroup};
use crate::time_limit::TimeLimit;
use crate::update_row::UpdateRow;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PricingRule {
    Dantzig,
    SteepestEdge,
    Devex,
}

#[derive(Clone, Debug)]
pub struct PrimalEdgeNorms {
    matrix: SparseMatrix,
    pricing_rule: PricingRule,
    recompute_edge_squared_norms: bool,
    reset_devex_weights: bool,
    edge_squared_norms: Vec<f64>,
    matrix_column_norms: Vec<f64>,
    devex_weights: Vec<f64>,
    num_devex_updates_since_reset: i32,
    direction_left_inverse: Vec<f64>,
    recompute_edges_norm_threshold: f64,
    devex_weights_reset_period: i32,
    initialize_devex_with_column_norms: bool,
    num_operations: i64,
    time_limit: Option<Rc<RefCell<TimeLimit>>>,
    watchers: Vec<Rc<Cell<bool>>>,
    stats: StatsGroup,
}

impl PrimalEdgeNorms {
    #[must_use]
    pub fn new(matrix: &SparseMatrix) -> Self {
        Self {
            matrix: matrix.clone(),
            pricing_rule: PricingRule::Dantzig,
            recompute_edge_squared_norms: true,
            reset_devex_weights: true,
            edge_squared_norms: Vec::new(),
            matrix_column_norms: Vec::new(),
            devex_weights: Vec::new(),
            num_devex_updates_since_reset: 0,
            direction_left_inverse: Vec::new(),
            recompute_edges_norm_threshold: 100.0,
            devex_weights_reset_period: 150,
            initialize_devex_with_column_norms: true,
            num_operations: 0,
            time_limit: None,
            watchers: Vec::new(),
            stats: StatsGroup::new("PrimalEdgeNorms"),
        }
    }

    pub fn clear(&mut self) {
        self.matrix_column_norms.clear();
        self.recompute_edge_squared_norms = true;
        self.reset_devex_weights = true;
        for watcher in &self.watchers {
            watcher.set(true);
        }
    }

    #[must_use]
    pub fn needs_basis_refactorization(&self) -> bool {
        self.pricing_rule == PricingRule::SteepestEdge && self.recompute_edge_squared_norms
    }

    pub fn set_pricing_rule(&mut self, pricing_rule: PricingRule) {
        self.pricing_rule = pricing_rule;
    }

    pub fn set_parameters(
        &mut self,
        recompute_edges_norm_threshold: f64,
        devex_weights_reset_period: i32,
        initialize_devex_with_column_norms: bool,
    ) {
        self.recompute_edges_norm_threshold = recompute_edges_norm_threshold;
        self.devex_weights_reset_period = devex_weights_reset_period;
        self.initialize_devex_with_column_norms = initialize_devex_with_column_norms;
    }

    pub fn set_glop_parameters(&mut self, parameters: &GlopParameters) {
        self.set_parameters(
            parameters.recompute_edges_norm_threshold,
            parameters.devex_weights_reset_period,
            parameters.initialize_devex_with_column_norms,
        );
    }

    pub fn set_time_limit(&mut self, time_limit: Option<Rc<RefCell<TimeLimit>>>) {
        self.time_limit = time_limit;
    }

    pub fn add_recomputation_watcher(&mut self, watcher: Rc<Cell<bool>>) {
        self.watchers.push(watcher);
    }

    /// Returns the vector selected by the active pricing rule.
    ///
    /// # Errors
    ///
    /// Propagates basis solve failures during exact recomputation.
    pub fn squared_norms<'a>(
        &'a mut self,
        basis: &BasisRepresentation,
        relevant: &ColBitVec,
    ) -> Result<&'a [f64], FactorizationError> {
        match self.pricing_rule {
            PricingRule::Dantzig => Ok(self.matrix_column_norms()),
            PricingRule::SteepestEdge => self.edge_squared_norms(basis, relevant),
            PricingRule::Devex => Ok(self.devex_weights()),
        }
    }

    /// Returns exact primal edge norms, recomputing when requested.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates a basis solve failure.
    pub fn edge_squared_norms<'a>(
        &'a mut self,
        basis: &BasisRepresentation,
        relevant: &ColBitVec,
    ) -> Result<&'a [f64], FactorizationError> {
        if relevant.len().to_usize() != self.matrix.num_cols().to_usize() {
            return Err(FactorizationError::DimensionMismatch);
        }
        if self.recompute_edge_squared_norms {
            self.compute_edge_squared_norms(basis, relevant)?;
        }
        Ok(&self.edge_squared_norms)
    }

    #[must_use]
    pub fn matrix_column_norms(&mut self) -> &[f64] {
        if self.matrix_column_norms.is_empty() {
            self.matrix_column_norms = (0..self.matrix.num_cols().to_usize())
                .map(|column| {
                    let entries = self.matrix.column(ColIndex::from_usize(column));
                    self.num_operations += i64::try_from(entries.num_entries()).unwrap_or(i64::MAX);
                    entries
                        .into_iter()
                        .map(|entry| entry.coefficient() * entry.coefficient())
                        .sum()
                })
                .collect();
        }
        &self.matrix_column_norms
    }

    #[must_use]
    pub fn devex_weights(&mut self) -> &[f64] {
        if self.reset_devex_weights {
            self.devex_weights = if self.initialize_devex_with_column_norms {
                self.matrix_column_norms().to_vec()
            } else {
                vec![1.0; self.matrix.num_cols().to_usize()]
            };
            self.num_devex_updates_since_reset = 0;
            self.reset_devex_weights = false;
        }
        &self.devex_weights
    }

    /// Replaces the entering norm by its precise value and checks its drift.
    #[must_use]
    pub fn test_entering_edge_norm_precision(
        &mut self,
        entering_column: usize,
        direction: &[f64],
    ) -> bool {
        if self.recompute_edge_squared_norms {
            return true;
        }
        let old_squared_norm = self.edge_squared_norms[entering_column];
        let precise_squared_norm = 1.0 + squared_norm(direction);
        self.edge_squared_norms[entering_column] = precise_squared_norm;
        let precise_norm = precise_squared_norm.sqrt();
        let accuracy = (precise_norm - old_squared_norm.sqrt()) / precise_norm;
        self.stats
            .add("edges_norm_accuracy", DistributionKind::Double, accuracy);
        if accuracy.abs() > self.recompute_edges_norm_threshold {
            self.recompute_edge_squared_norms = true;
            for watcher in &self.watchers {
                watcher.set(true);
            }
        }
        old_squared_norm >= 0.25 * precise_squared_norm
    }

    /// Updates steepest-edge and/or Devex data before applying a basis pivot.
    ///
    /// # Errors
    ///
    /// Returns a dimension error or propagates update-row/basis solve failures.
    #[allow(clippy::too_many_arguments)]
    pub fn update_before_basis_pivot(
        &mut self,
        basis: &BasisRepresentation,
        relevant: &ColBitVec,
        entering_column: usize,
        leaving_column: usize,
        leaving_row: usize,
        direction: &[f64],
        update_row: &mut UpdateRow,
    ) -> Result<(), FactorizationError> {
        if direction.len() != basis.dimension()
            || entering_column >= self.matrix.num_cols().to_usize()
            || leaving_column >= self.matrix.num_cols().to_usize()
        {
            return Err(FactorizationError::DimensionMismatch);
        }
        if !self.recompute_edge_squared_norms {
            update_row.compute_update_row(basis, &self.matrix, relevant, leaving_row)?;
            self.direction_left_inverse = basis.transpose_solve(direction)?;
            self.update_edge_squared_norms(
                entering_column,
                leaving_column,
                leaving_row,
                direction,
                update_row,
            );
        }
        if !self.reset_devex_weights {
            self.num_devex_updates_since_reset += 1;
            if self.num_devex_updates_since_reset > self.devex_weights_reset_period {
                self.reset_devex_weights = true;
            } else {
                update_row.compute_update_row(basis, &self.matrix, relevant, leaving_row)?;
                self.update_devex_weights(leaving_column, leaving_row, direction, update_row);
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn num_operations(&self) -> i64 {
        self.num_operations
    }

    #[must_use]
    pub fn deterministic_time(&self) -> f64 {
        deterministic_time_for_fp_operations(self.num_operations)
    }

    #[must_use]
    pub fn stat_string(&self) -> String {
        self.stats.stat_string()
    }

    fn compute_edge_squared_norms(
        &mut self,
        basis: &BasisRepresentation,
        relevant: &ColBitVec,
    ) -> Result<(), FactorizationError> {
        self.edge_squared_norms = vec![1.0; self.matrix.num_cols().to_usize()];
        let test_limit = self.time_limit.is_some() && basis.number_of_entries_in_lu() > 10_000;
        for typed_column in relevant.iter_ones() {
            let column = typed_column.to_usize();
            let mut dense = vec![0.0; basis.dimension()];
            for entry in self.matrix.column(ColIndex::from_usize(column)) {
                dense[entry.index().to_usize()] = entry.coefficient();
            }
            self.edge_squared_norms[column] = 1.0 + squared_norm(&basis.solve(&dense)?);
            if test_limit
                && self
                    .time_limit
                    .as_ref()
                    .is_some_and(|limit| limit.borrow_mut().limit_reached())
            {
                break;
            }
        }
        self.recompute_edge_squared_norms = false;
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn update_edge_squared_norms(
        &mut self,
        entering_column: usize,
        leaving_column: usize,
        leaving_row: usize,
        direction: &[f64],
        update_row: &UpdateRow,
    ) {
        let pivot = -direction[leaving_row];
        let entering_squared_norm = self.edge_squared_norms[entering_column];
        let leaving_squared_norm = (entering_squared_norm / (pivot * pivot)).max(1.0);
        let factor = 2.0 / pivot;
        let first_slack = self
            .matrix
            .num_cols()
            .to_usize()
            .saturating_sub(self.matrix.num_rows().to_usize());
        let mut lower_bounded_norms = 0_i64;
        for &column in update_row.non_zero_positions() {
            let coefficient = update_row.coefficient(column);
            let scalar_product = if column >= first_slack {
                self.direction_left_inverse[column - first_slack]
            } else {
                self.matrix
                    .column(ColIndex::from_usize(column))
                    .into_iter()
                    .fold(0.0, |sum, entry| {
                        self.direction_left_inverse[entry.index().to_usize()]
                            .mul_add(entry.coefficient(), sum)
                    })
            };
            self.num_operations += i64::try_from(
                self.matrix
                    .column(ColIndex::from_usize(column))
                    .num_entries(),
            )
            .unwrap_or(i64::MAX);
            let inner = coefficient.mul_add(leaving_squared_norm, factor * scalar_product);
            self.edge_squared_norms[column] =
                coefficient.mul_add(inner, self.edge_squared_norms[column]);
            let scaled_coefficient = coefficient / pivot;
            let lower_bound = 1.0 + scaled_coefficient * scaled_coefficient;
            if self.edge_squared_norms[column] < lower_bound {
                self.edge_squared_norms[column] = lower_bound;
                lower_bounded_norms += 1;
            }
        }
        self.edge_squared_norms[leaving_column] = leaving_squared_norm;
        self.stats.add(
            "lower_bounded_norms",
            DistributionKind::Integer,
            lower_bounded_norms as f64,
        );
    }

    fn update_devex_weights(
        &mut self,
        leaving_column: usize,
        leaving_row: usize,
        direction: &[f64],
        update_row: &UpdateRow,
    ) {
        let pivot_magnitude = direction[leaving_row].abs();
        let entering_norm = precise_squared_norm(direction).sqrt();
        let leaving_norm = (entering_norm / pivot_magnitude).max(1.0);
        for &column in update_row.non_zero_positions() {
            let update_norm = update_row.coefficient(column).abs() * leaving_norm;
            self.devex_weights[column] = self.devex_weights[column].max(update_norm * update_norm);
        }
        self.devex_weights[leaving_column] = leaving_norm * leaving_norm;
    }
}

/// Returns `1 + ||B^-1 a_j||_2^2` for every structural column.
///
/// # Errors
///
/// Returns a basis solve or dimension error.
pub fn compute_primal_edge_squared_norms(
    basis: &BasisRepresentation,
    matrix: &SparseMatrix,
) -> Result<Vec<f64>, FactorizationError> {
    if matrix.num_rows().to_usize() != basis.dimension() {
        return Err(FactorizationError::DimensionMismatch);
    }
    let mut norms = Vec::with_capacity(matrix.num_cols().to_usize());
    for column in 0..matrix.num_cols().to_usize() {
        let mut dense = vec![0.0; basis.dimension()];
        for entry in matrix.column(ColIndex::from_usize(column)) {
            dense[entry.index().to_usize()] = entry.coefficient();
        }
        let direction = basis.solve(&dense)?;
        norms.push(1.0 + direction.iter().map(|value| value * value).sum::<f64>());
    }
    Ok(norms)
}
use std::cell::{Cell, RefCell};
use std::rc::Rc;
