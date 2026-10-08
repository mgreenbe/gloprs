//! Dual phase-I and phase-II entering-variable ratio tests.
//!
//! Direct port of `ortools/glop/entering_variable.{h,cc}`, including Harris
//! tolerance, bound flipping, stable-pivot preference, and breakpoint order.

#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::too_many_lines
)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use lp_data::lp_types::{ColIndex, VariableType, VectorIndex};

use crate::lu_factorization::FactorizationError;
use crate::parameters::GlopParameters;
use crate::random::SharedRandom;
use crate::reduced_costs::ReducedCosts;
use crate::update_row::UpdateRow;
use crate::variables_info::VariablesInfo;

#[derive(Clone, Copy, Debug)]
struct ColWithRatio {
    column: usize,
    ratio: f64,
    coefficient_magnitude: f64,
}

impl PartialEq for ColWithRatio {
    fn eq(&self, other: &Self) -> bool {
        self.column == other.column
            && self.ratio == other.ratio
            && self.coefficient_magnitude == other.coefficient_magnitude
    }
}

impl Eq for ColWithRatio {}

impl Ord for ColWithRatio {
    fn cmp(&self, other: &Self) -> Ordering {
        // std::make_heap uses upstream operator< so the root is the smallest
        // ratio, then greatest coefficient, then smallest column.
        other
            .ratio
            .total_cmp(&self.ratio)
            .then_with(|| {
                self.coefficient_magnitude
                    .total_cmp(&other.coefficient_magnitude)
            })
            .then_with(|| other.column.cmp(&self.column))
    }
}

impl PartialOrd for ColWithRatio {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct EnteringVariable {
    parameters: GlopParameters,
    equivalent_entering_choices: Vec<usize>,
    random: SharedRandom,
    num_operations: i64,
}

impl EnteringVariable {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::new_with_random(SharedRandom::new(seed))
    }

    #[must_use]
    pub fn new_with_random(random: SharedRandom) -> Self {
        Self {
            parameters: GlopParameters::default(),
            equivalent_entering_choices: Vec::new(),
            random,
            num_operations: 0,
        }
    }

    pub fn set_parameters(&mut self, parameters: &GlopParameters) {
        self.parameters = parameters.clone();
    }

    pub fn dual_choose_entering_column(
        &mut self,
        nothing_to_recompute: bool,
        update_row: &UpdateRow,
        cost_variation: f64,
        variables_info: &VariablesInfo,
        reduced_costs_state: &mut ReducedCosts<'_>,
        bound_flip_candidates: &mut Vec<ColIndex>,
    ) -> Result<Option<ColIndex>, FactorizationError> {
        let reduced_costs = reduced_costs_state.reduced_costs()?.to_vec();
        let dual_tolerance = reduced_costs_state.dual_feasibility_tolerance();
        Ok(self.dual_choose_entering_column_from_values(
            nothing_to_recompute,
            update_row,
            cost_variation,
            variables_info,
            &reduced_costs,
            dual_tolerance,
            bound_flip_candidates,
        ))
    }

    /// Explicit-collaborator form of GLOP's dual Phase-II ratio test.
    #[allow(clippy::too_many_arguments)]
    pub fn dual_choose_entering_column_from_values(
        &mut self,
        nothing_to_recompute: bool,
        update_row: &UpdateRow,
        cost_variation: f64,
        variables_info: &VariablesInfo,
        reduced_costs: &[f64],
        dual_tolerance: f64,
        bound_flip_candidates: &mut Vec<ColIndex>,
    ) -> Option<ColIndex> {
        let threshold = if nothing_to_recompute {
            self.parameters.minimum_acceptable_pivot
        } else {
            self.parameters.ratio_test_zero_threshold
        };
        let mut variation_magnitude = cost_variation.abs() - threshold;
        let harris_tolerance = self.parameters.harris_tolerance_ratio * dual_tolerance;
        let minimum_delta = self.parameters.degenerate_ministep_factor * dual_tolerance;
        let mut harris_ratio = f64::MAX;
        let mut breakpoints = BinaryHeap::new();
        self.num_operations +=
            10 * i64::try_from(update_row.non_zero_positions().len()).unwrap_or(i64::MAX);
        for &column in update_row.non_zero_positions() {
            let index = ColIndex::from_usize(column);
            let coefficient = if cost_variation > 0.0 {
                update_row.coefficient(column)
            } else {
                -update_row.coefficient(column)
            };
            let entry = if variables_info.can_decrease().contains(index) && coefficient > threshold
            {
                if -reduced_costs[column] > harris_ratio * coefficient {
                    continue;
                }
                ColWithRatio {
                    column,
                    ratio: -reduced_costs[column] / coefficient,
                    coefficient_magnitude: coefficient,
                }
            } else if variables_info.can_increase().contains(index) && coefficient < -threshold {
                if reduced_costs[column] > harris_ratio * -coefficient {
                    continue;
                }
                ColWithRatio {
                    column,
                    ratio: reduced_costs[column] / -coefficient,
                    coefficient_magnitude: -coefficient,
                }
            } else {
                continue;
            };
            let ratio = (minimum_delta / entry.coefficient_magnitude)
                .max(entry.ratio + harris_tolerance / entry.coefficient_magnitude);
            if ratio < harris_ratio {
                if variables_info.non_basic_boxed_variables().contains(index) {
                    let delta =
                        variables_info.bound_difference(index) * entry.coefficient_magnitude;
                    if delta >= variation_magnitude {
                        harris_ratio = ratio;
                    }
                } else {
                    harris_ratio = ratio;
                }
            }
            breakpoints.push(entry);
        }
        harris_ratio = f64::MAX;
        bound_flip_candidates.clear();
        let mut entering = None;
        let mut step = 0.0;
        let mut best_coefficient = -1.0;
        self.equivalent_entering_choices.clear();
        while let Some(top) = breakpoints.peek().copied() {
            if top.ratio > harris_ratio {
                break;
            }
            if variation_magnitude > 0.0
                && variables_info
                    .non_basic_boxed_variables()
                    .contains(ColIndex::from_usize(top.column))
            {
                variation_magnitude -= variables_info
                    .bound_difference(ColIndex::from_usize(top.column))
                    * top.coefficient_magnitude;
                if variation_magnitude > 0.0 {
                    bound_flip_candidates.push(ColIndex::from_usize(top.column));
                    breakpoints.pop();
                    continue;
                }
            }
            if top.coefficient_magnitude >= best_coefficient {
                harris_ratio = harris_ratio.min(
                    (minimum_delta / top.coefficient_magnitude)
                        .max(top.ratio + harris_tolerance / top.coefficient_magnitude),
                );
                if top.coefficient_magnitude == best_coefficient && top.ratio == step {
                    self.equivalent_entering_choices.push(top.column);
                } else {
                    self.equivalent_entering_choices.clear();
                    best_coefficient = top.coefficient_magnitude;
                    entering = Some(top.column);
                    step = top.ratio;
                }
            }
            breakpoints.pop();
        }
        if !self.equivalent_entering_choices.is_empty() {
            if let Some(first) = entering {
                self.equivalent_entering_choices.push(first);
            }
            let choice = self
                .random
                .uniform_index(self.equivalent_entering_choices.len());
            entering = Some(self.equivalent_entering_choices[choice]);
        }
        if best_coefficient < self.parameters.minimum_acceptable_pivot
            && !bound_flip_candidates.is_empty()
        {
            for &candidate in bound_flip_candidates.iter().rev() {
                if update_row.coefficient(candidate.to_usize()).abs()
                    >= self.parameters.minimum_acceptable_pivot
                {
                    entering = Some(candidate.to_usize());
                    break;
                }
            }
        }
        entering.map(ColIndex::from_usize)
    }

    pub fn dual_phase_one_choose_entering_column(
        &mut self,
        nothing_to_recompute: bool,
        update_row: &UpdateRow,
        cost_variation: f64,
        variables_info: &VariablesInfo,
        reduced_costs_state: &mut ReducedCosts<'_>,
    ) -> Result<Option<ColIndex>, FactorizationError> {
        let reduced_costs = reduced_costs_state.reduced_costs()?.to_vec();
        let dual_tolerance = reduced_costs_state.dual_feasibility_tolerance();
        Ok(self.dual_phase_one_choose_entering_column_from_values(
            nothing_to_recompute,
            update_row,
            cost_variation,
            variables_info,
            &reduced_costs,
            dual_tolerance,
        ))
    }

    pub fn dual_phase_one_choose_entering_column_from_values(
        &mut self,
        nothing_to_recompute: bool,
        update_row: &UpdateRow,
        cost_variation: f64,
        variables_info: &VariablesInfo,
        reduced_costs: &[f64],
        dual_tolerance: f64,
    ) -> Option<ColIndex> {
        let threshold = if nothing_to_recompute {
            self.parameters.minimum_acceptable_pivot
        } else {
            self.parameters.ratio_test_zero_threshold
        };
        let harris_tolerance = self.parameters.harris_tolerance_ratio * dual_tolerance;
        let minimum_delta = self.parameters.degenerate_ministep_factor * dual_tolerance;
        let mut breakpoints = BinaryHeap::new();
        self.num_operations +=
            10 * i64::try_from(update_row.non_zero_positions().len()).unwrap_or(i64::MAX);
        for &column in update_row.non_zero_positions() {
            let index = ColIndex::from_usize(column);
            debug_assert!(!matches!(
                variables_info.variable_types()[index],
                VariableType::UpperAndLowerBounded | VariableType::FixedVariable
            ));
            let update_coefficient = update_row.coefficient(column);
            if update_coefficient.abs() < threshold {
                continue;
            }
            let coefficient = if cost_variation > 0.0 {
                update_coefficient
            } else {
                -update_coefficient
            };
            if reduced_costs[column].abs() <= dual_tolerance {
                if coefficient > 0.0 && !variables_info.can_decrease().contains(index) {
                    continue;
                }
                if coefficient < 0.0 && !variables_info.can_increase().contains(index) {
                    continue;
                }
                if coefficient * reduced_costs[column] > 0.0 {
                    let numerator =
                        minimum_delta.max(harris_tolerance - reduced_costs[column].abs());
                    breakpoints.push(ColWithRatio {
                        column,
                        ratio: numerator / coefficient.abs(),
                        coefficient_magnitude: coefficient.abs(),
                    });
                    continue;
                }
            } else if coefficient * reduced_costs[column] > 0.0 {
                continue;
            }
            breakpoints.push(ColWithRatio {
                column,
                ratio: (reduced_costs[column].abs() + harris_tolerance) / coefficient.abs(),
                coefficient_magnitude: coefficient.abs(),
            });
        }
        let mut pivot_magnitude = 0.0;
        let mut entering = None;
        let mut step = -1.0;
        let mut improvement = cost_variation.abs();
        while let Some(top) = breakpoints.peek().copied() {
            if top.ratio > step && top.coefficient_magnitude >= pivot_magnitude {
                entering = Some(top.column);
                step = top.ratio;
                pivot_magnitude = top.coefficient_magnitude;
            }
            improvement -= top.coefficient_magnitude;
            let index = ColIndex::from_usize(top.column);
            if variables_info.can_decrease().contains(index)
                && variables_info.can_increase().contains(index)
                && reduced_costs[top.column].abs() > threshold
            {
                improvement -= top.coefficient_magnitude;
            }
            if improvement <= 0.0 {
                break;
            }
            breakpoints.pop();
        }
        entering.map(ColIndex::from_usize)
    }

    #[must_use]
    pub fn deterministic_time(&self) -> f64 {
        lp_data::lp_types::deterministic_time_for_fp_operations(self.num_operations)
    }
}
