//! Primal Harris ratio test extracted from `revised_simplex.cc`.
//!
//! Phase 3 validates this pivot mechanic independently; the Phase 4 driver
//! will call the same function rather than introducing another implementation.

#![allow(clippy::float_cmp, clippy::missing_panics_doc)]

use lp_data::lp_types::{ColIndex, RowIndex, RowToColMapping, VectorIndex};
use lp_data::scattered_vector::ScatteredColumn;

use crate::parameters::GlopParameters;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeavingChoice {
    BoundFlip {
        step: f64,
    },
    Pivot {
        row: RowIndex,
        step: f64,
        target_bound: f64,
    },
    Refactorize,
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn choose_leaving_variable_row(
    entering_column: ColIndex,
    reduced_cost: f64,
    direction: &ScatteredColumn,
    direction_infinity_norm: f64,
    variable_values: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    basis: &RowToColMapping,
    basis_is_refactorized: bool,
    parameters: &GlopParameters,
) -> LeavingChoice {
    debug_assert_ne!(reduced_cost, 0.0);
    let entering = entering_column.to_usize();
    let mut current_ratio = if reduced_cost > 0.0 {
        variable_values[entering] - lower_bounds[entering]
    } else {
        upper_bounds[entering] - variable_values[entering]
    };
    debug_assert!(current_ratio > 0.0);
    let harris_tolerance =
        parameters.harris_tolerance_ratio * parameters.primal_feasibility_tolerance;
    let minimum_delta =
        parameters.degenerate_ministep_factor * parameters.primal_feasibility_tolerance;
    let threshold = if basis_is_refactorized {
        parameters.minimum_acceptable_pivot
    } else {
        parameters.ratio_test_zero_threshold
    };
    let mut harris_ratio = current_ratio;
    let mut candidates = Vec::new();
    for entry in direction {
        let magnitude = entry.coefficient().abs();
        if magnitude <= threshold {
            continue;
        }
        let row = entry.row();
        let column = basis[row].to_usize();
        let value = variable_values[column];
        let ratio = if reduced_cost > 0.0 {
            if entry.coefficient() > 0.0 {
                (upper_bounds[column] - value) / entry.coefficient()
            } else {
                (lower_bounds[column] - value) / entry.coefficient()
            }
        } else if entry.coefficient() > 0.0 {
            (value - lower_bounds[column]) / entry.coefficient()
        } else {
            (value - upper_bounds[column]) / entry.coefficient()
        };
        if ratio <= harris_ratio {
            candidates.push((row, ratio));
            harris_ratio = harris_ratio
                .min((minimum_delta / magnitude).max(ratio + harris_tolerance / magnitude));
        }
    }
    if current_ratio <= harris_ratio {
        return LeavingChoice::BoundFlip {
            step: current_ratio,
        };
    }
    let mut leaving_row = None;
    let mut pivot_magnitude = 0.0_f64;
    for (row, ratio) in candidates {
        if ratio > harris_ratio {
            continue;
        }
        let candidate_magnitude = direction.value(row).abs();
        if candidate_magnitude < pivot_magnitude {
            continue;
        }
        if candidate_magnitude == pivot_magnitude
            && !ratio_more_or_equally_stable(ratio, current_ratio)
        {
            continue;
        }
        current_ratio = ratio;
        pivot_magnitude = candidate_magnitude;
        leaving_row = Some(row);
    }
    let row = leaving_row.expect("a Harris candidate exists when bound flip is rejected");
    let step = if current_ratio <= 0.0 {
        minimum_delta / pivot_magnitude
    } else {
        current_ratio
    };
    if pivot_magnitude < parameters.small_pivot_threshold * direction_infinity_norm
        && !basis_is_refactorized
    {
        return LeavingChoice::Refactorize;
    }
    let coefficient_positive = direction.value(row) > 0.0;
    let target = if (reduced_cost > 0.0) == coefficient_positive {
        upper_bounds[basis[row].to_usize()]
    } else {
        lower_bounds[basis[row].to_usize()]
    };
    LeavingChoice::Pivot {
        row,
        step,
        target_bound: target,
    }
}

fn ratio_more_or_equally_stable(candidate: f64, current: f64) -> bool {
    if current >= 0.0 {
        candidate >= 0.0 && candidate <= current
    } else {
        candidate >= current
    }
}
