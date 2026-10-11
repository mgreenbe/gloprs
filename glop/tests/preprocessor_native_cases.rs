//! Pinned-native transformation and postsolve traces for the first Phase-5 components.

use std::collections::BTreeMap;

use glop::parameters::{CostScalingAlgorithm, GlopParameters, ScalingAlgorithm};
use glop::preprocessor::{ScalingPreprocessor, ShiftVariableBoundsPreprocessor};
use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{ColIndex, ProblemStatus, RowIndex, VariableStatus, VectorIndex};
use serde::Deserialize;
use sha2::Digest;

const CASES: &str = include_str!("../../baselines/preprocessor-cases.json");
const EXPECTED: &str = include_str!("../../baselines/preprocessor-native.json");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

#[derive(Deserialize)]
struct Specification {
    upstream_commit: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    mode: String,
    cost_scaling: u8,
    #[serde(default)]
    scaling_method: u8,
    matrix: Vec<(usize, usize, String)>,
    variables: Vec<(String, String, String, String, String)>,
    constraints: Vec<(String, String, String)>,
}

#[derive(Deserialize)]
struct Fixture {
    upstream_commit: String,
    cases_sha256: String,
    results: Vec<NativeResult>,
}

#[derive(Deserialize)]
struct NativeResult {
    name: String,
    trace: BTreeMap<String, Vec<u64>>,
}

fn number(value: &str) -> f64 {
    match value {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => value.parse().unwrap(),
    }
}

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

#[allow(clippy::too_many_lines)] // Keep native's input/run/trace sequence together.
fn rust_trace(case: &Case) -> BTreeMap<String, Vec<u64>> {
    let mut lp = LinearProgram::default();
    for _ in &case.variables {
        lp.create_new_variable();
    }
    for _ in &case.constraints {
        lp.create_new_constraint();
    }
    for (row, col, value) in &case.matrix {
        lp.set_coefficient(
            RowIndex::from_usize(*row),
            ColIndex::from_usize(*col),
            number(value),
        );
    }
    for (position, (objective, lower, upper, _, _)) in case.variables.iter().enumerate() {
        let col = ColIndex::from_usize(position);
        lp.set_objective_coefficient(col, number(objective));
        lp.set_variable_bounds(col, number(lower), number(upper));
    }
    for (position, (lower, upper, _)) in case.constraints.iter().enumerate() {
        lp.set_constraint_bounds(RowIndex::from_usize(position), number(lower), number(upper));
    }
    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    for (position, (_, _, _, status, value)) in case.variables.iter().enumerate() {
        let col = ColIndex::from_usize(position);
        solution.variable_statuses[col] = match status.as_str() {
            "basic" => VariableStatus::Basic,
            "lower" => VariableStatus::AtLowerBound,
            "upper" => VariableStatus::AtUpperBound,
            "fixed" => VariableStatus::FixedValue,
            "free" => VariableStatus::Free,
            _ => panic!("unknown variable status {status}"),
        };
        solution.primal_values[col] = number(value);
    }
    for (position, (_, _, value)) in case.constraints.iter().enumerate() {
        solution.dual_values[RowIndex::from_usize(position)] = number(value);
    }
    lp.clean_up();

    let parameters = GlopParameters {
        scaling_method: match case.scaling_method {
            0 => ScalingAlgorithm::Equilibration,
            1 => ScalingAlgorithm::LinearProgram,
            _ => panic!("unknown scaling method"),
        },
        cost_scaling: match case.cost_scaling {
            0 => CostScalingAlgorithm::NoCostScaling,
            1 => CostScalingAlgorithm::ContainOneCostScaling,
            2 => CostScalingAlgorithm::MeanCostScaling,
            3 => CostScalingAlgorithm::MedianCostScaling,
            _ => panic!("unknown cost scaling"),
        },
        ..GlopParameters::default()
    };
    let mut shift = ShiftVariableBoundsPreprocessor::new();
    let mut scaling = ScalingPreprocessor::new();
    let do_shift = case.mode == "shift" || case.mode == "both";
    let do_scale = case.mode == "scale" || case.mode == "both";
    let shift_result = if do_shift {
        shift.run(&mut lp)
    } else {
        Ok(false)
    };
    let shifted = shift_result == Ok(true);
    let invalid = shift_result == Err(ProblemStatus::InvalidProblem);
    let scaled = do_scale && !invalid && scaling.run(&mut lp, &parameters);

    let mut trace = BTreeMap::new();
    trace.insert(
        "flags".to_owned(),
        vec![u64::from(shifted), u64::from(scaled), u64::from(invalid)],
    );
    trace.insert(
        "objective".to_owned(),
        bits(lp.objective_coefficients().iter().copied()),
    );
    trace.insert(
        "objective_offset".to_owned(),
        vec![lp.objective_offset().to_bits()],
    );
    trace.insert(
        "variable_lower".to_owned(),
        bits(lp.variable_lower_bounds().iter().copied()),
    );
    trace.insert(
        "variable_upper".to_owned(),
        bits(lp.variable_upper_bounds().iter().copied()),
    );
    trace.insert(
        "constraint_lower".to_owned(),
        bits(lp.constraint_lower_bounds().iter().copied()),
    );
    trace.insert(
        "constraint_upper".to_owned(),
        bits(lp.constraint_upper_bounds().iter().copied()),
    );
    let mut matrix = Vec::new();
    for position in 0..case.variables.len() {
        for entry in lp.sparse_column(ColIndex::from_usize(position)) {
            matrix.extend([
                position as u64,
                entry.index().to_usize() as u64,
                entry.coefficient().to_bits(),
            ]);
        }
    }
    trace.insert("matrix".to_owned(), matrix);
    if shifted {
        trace.insert("offsets".to_owned(), bits(shift.offsets().iter().copied()));
    }
    if scaled {
        scaling.recover_solution(&mut solution);
    }
    if shifted {
        shift.recover_solution(&mut solution);
    }
    trace.insert(
        "recovered_primal".to_owned(),
        bits(solution.primal_values.iter().copied()),
    );
    trace.insert(
        "recovered_dual".to_owned(),
        bits(solution.dual_values.iter().copied()),
    );
    trace
}

#[test]
fn phase5_shift_and_scaling_match_pinned_native_glop() {
    let specification: Specification = serde_json::from_str(CASES).unwrap();
    let fixture: Fixture = serde_json::from_str(EXPECTED).unwrap();
    assert_eq!(specification.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(
        fixture.cases_sha256,
        format!("{:x}", sha2::Sha256::digest(CASES.as_bytes()))
    );
    assert_eq!(specification.cases.len(), fixture.results.len());
    for (case, native) in specification.cases.iter().zip(&fixture.results) {
        assert_eq!(case.name, native.name);
        let actual = rust_trace(case);
        assert_eq!(actual, native.trace, "{}", case.name);
    }
}
