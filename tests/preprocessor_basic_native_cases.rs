//! Native-pinned transformation and postsolve checks for simple presolve passes.

use gloprs::parameters::{GlopParameters, SolverBehavior};
use gloprs::preprocessor::{
    DoubletonEqualityRowPreprocessor, DoubletonFreeColumnPreprocessor, DualizerPreprocessor,
    EmptyColumnPreprocessor, EmptyConstraintPreprocessor, FixedVariablePreprocessor,
    ForcingAndImpliedFreeConstraintPreprocessor, FreeConstraintPreprocessor,
    ImpliedFreePreprocessor, MainLpPreprocessor, ProportionalColumnPreprocessor,
    ProportionalRowPreprocessor, SingletonColumnSignPreprocessor, SingletonPreprocessor,
    UnconstrainedVariablePreprocessor,
};
use gloprs::time_limit::TimeLimit;
use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{
    ColIndex, ConstraintStatus, ProblemStatus, RowIndex, VariableStatus, VectorIndex,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::Digest;

const CASES: &str = include_str!("../baselines/preprocessor-basic-cases.json");
const EXPECTED: &str = include_str!("../baselines/preprocessor-basic-native.json");
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
    matrix: Vec<(usize, usize, String)>,
    variables: Vec<(String, String, String)>,
    constraints: Vec<(String, String)>,
    post_variables: Vec<(String, String)>,
    post_constraints: Vec<(String, String)>,
    post_status: Option<String>,
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
    trace: Value,
}

fn number(value: &str) -> f64 {
    match value {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => value.parse().unwrap(),
    }
}

fn variable_status(value: &str) -> VariableStatus {
    match value {
        "basic" => VariableStatus::Basic,
        "lower" => VariableStatus::AtLowerBound,
        "upper" => VariableStatus::AtUpperBound,
        "fixed" => VariableStatus::FixedValue,
        "free" => VariableStatus::Free,
        _ => panic!("unknown status {value}"),
    }
}

fn constraint_status(value: &str) -> ConstraintStatus {
    match value {
        "basic" => ConstraintStatus::Basic,
        "lower" => ConstraintStatus::AtLowerBound,
        "upper" => ConstraintStatus::AtUpperBound,
        "free" => ConstraintStatus::Free,
        _ => panic!("unknown status {value}"),
    }
}

#[allow(clippy::too_many_lines)] // Keep each native transform and its reverse map adjacent.
fn rust_trace(case: &Case) -> Value {
    let mut lp = LinearProgram::new();
    for _ in &case.variables {
        lp.create_new_variable();
    }
    for _ in &case.constraints {
        lp.create_new_constraint();
    }
    for (row, col, coefficient) in &case.matrix {
        lp.set_coefficient(
            RowIndex::from_usize(*row),
            ColIndex::from_usize(*col),
            number(coefficient),
        );
    }
    for (position, (cost, lower, upper)) in case.variables.iter().enumerate() {
        let col = ColIndex::from_usize(position);
        lp.set_objective_coefficient(col, number(cost));
        lp.set_variable_bounds(col, number(lower), number(upper));
    }
    for (position, (lower, upper)) in case.constraints.iter().enumerate() {
        lp.set_constraint_bounds(RowIndex::from_usize(position), number(lower), number(upper));
    }
    lp.clean_up();
    let mut fixed = FixedVariablePreprocessor::new();
    let mut free_row = FreeConstraintPreprocessor::new();
    let mut empty_row = EmptyConstraintPreprocessor::new();
    let mut empty_column = EmptyColumnPreprocessor::new();
    let mut singleton_sign = SingletonColumnSignPreprocessor::new();
    let mut singleton = SingletonPreprocessor::new();
    let mut forcing = ForcingAndImpliedFreeConstraintPreprocessor::new();
    let mut implied_free = ImpliedFreePreprocessor::new();
    let mut doubleton_free = DoubletonFreeColumnPreprocessor::new();
    let mut doubleton_equality = DoubletonEqualityRowPreprocessor::new();
    let mut proportional_row = ProportionalRowPreprocessor::new();
    let mut proportional_column = ProportionalColumnPreprocessor::new();
    let mut dualizer = DualizerPreprocessor::new();
    let mut unconstrained = UnconstrainedVariablePreprocessor::new();
    let mut main_preprocessor = MainLpPreprocessor::new();
    let changed = match case.mode.as_str() {
        "fixed" => fixed.run(&mut lp),
        "free_row" => free_row.run(&mut lp),
        "empty_row" => empty_row.run(&mut lp, 1e-6),
        "empty_column" => empty_column.run(&mut lp),
        "singleton_sign" => singleton_sign.run(&mut lp),
        "forcing" => forcing.run(&mut lp, &GlopParameters::default()),
        "forcing_exact" => {
            let parameters = GlopParameters {
                solution_feasibility_tolerance: 0.0,
                preprocessor_zero_tolerance: 0.0,
                ..GlopParameters::default()
            };
            forcing.run(&mut lp, &parameters)
        }
        "implied_free" => implied_free.run(&mut lp, &GlopParameters::default()),
        "doubleton_free" => doubleton_free.run(&mut lp, &GlopParameters::default()),
        "doubleton_equality" => doubleton_equality.run(&mut lp, &GlopParameters::default()),
        "proportional_row" => proportional_row.run(&mut lp, &GlopParameters::default()),
        "proportional_column" => proportional_column.run(&mut lp, &GlopParameters::default()),
        "dualizer" => dualizer.run(
            &mut lp,
            &GlopParameters {
                solve_dual_problem: SolverBehavior::AlwaysDo,
                ..GlopParameters::default()
            },
        ),
        "unconstrained" => unconstrained.run(&mut lp, &GlopParameters::default()),
        "singleton" => singleton.run(&mut lp, &GlopParameters::default()),
        "main" => main_preprocessor.run(
            &mut lp,
            &GlopParameters {
                use_scaling: false,
                ..GlopParameters::default()
            },
            &mut TimeLimit::default(),
        ),
        other => panic!("unknown mode {other}"),
    };
    let status = match case.mode.as_str() {
        "empty_row" => empty_row.status(),
        "empty_column" => empty_column.status(),
        "forcing" | "forcing_exact" => forcing.status(),
        "doubleton_equality" => doubleton_equality.status(),
        "proportional_row" => proportional_row.status(),
        "proportional_column" => proportional_column.status(),
        "implied_free" => implied_free.status(),
        "unconstrained" => unconstrained.status(),
        "singleton" => singleton.status(),
        "main" => main_preprocessor.status(),
        _ => lp_data::lp_types::ProblemStatus::Init,
    };
    let mut matrix = Vec::new();
    for position in 0..lp.num_variables().to_usize() {
        for entry in lp.sparse_column(ColIndex::from_usize(position)) {
            matrix.push(json!(position));
            matrix.push(json!(entry.index().to_usize()));
            matrix.push(json!(entry.coefficient().to_bits()));
        }
    }
    assert_eq!(case.post_variables.len(), lp.num_variables().to_usize());
    assert_eq!(case.post_constraints.len(), lp.num_constraints().to_usize());
    let mut solution = ProblemSolution::new(lp.num_constraints(), lp.num_variables());
    solution.status = match case.post_status.as_deref().unwrap_or("INIT") {
        "INIT" => ProblemStatus::Init,
        "PRIMAL_INFEASIBLE" => ProblemStatus::PrimalInfeasible,
        "DUAL_FEASIBLE" => ProblemStatus::DualFeasible,
        other => panic!("unknown post status {other}"),
    };
    for (position, (status, value)) in case.post_variables.iter().enumerate() {
        let col = ColIndex::from_usize(position);
        solution.variable_statuses[col] = variable_status(status);
        solution.primal_values[col] = number(value);
    }
    for (position, (status, value)) in case.post_constraints.iter().enumerate() {
        let row = RowIndex::from_usize(position);
        solution.constraint_statuses[row] = constraint_status(status);
        solution.dual_values[row] = number(value);
    }
    if changed && case.mode == "main" {
        let mut reusable = solution.clone();
        main_preprocessor.recover_solution(&mut reusable);
        let mut reused = solution.clone();
        main_preprocessor.recover_solution(&mut reused);
        assert_eq!(reusable, reused, "{} reusable postsolve", case.name);
        let mut destructive = solution.clone();
        main_preprocessor.destructive_recover_solution(&mut destructive);
        assert_eq!(reusable, destructive, "{} destructive postsolve", case.name);
        solution = reusable;
    } else if changed {
        match case.mode.as_str() {
            "fixed" => fixed.recover_solution(&mut solution),
            "free_row" => free_row.recover_solution(&mut solution),
            "empty_row" => empty_row.recover_solution(&mut solution),
            "empty_column" => empty_column.recover_solution(&mut solution),
            "singleton_sign" => singleton_sign.recover_solution(&mut solution),
            "forcing" | "forcing_exact" => forcing.recover_solution(&mut solution),
            "implied_free" => implied_free.recover_solution(&mut solution),
            "doubleton_free" => doubleton_free.recover_solution(&mut solution),
            "doubleton_equality" => doubleton_equality.recover_solution(&mut solution),
            "proportional_row" => proportional_row.recover_solution(&mut solution),
            "proportional_column" => proportional_column.recover_solution(&mut solution),
            "dualizer" => dualizer.recover_solution(&mut solution),
            "unconstrained" => unconstrained.recover_solution(&mut solution),
            "singleton" => singleton.recover_solution(&mut solution),
            "main" => main_preprocessor.destructive_recover_solution(&mut solution),
            _ => unreachable!(),
        }
    }
    json!({
        "changed": [u64::from(changed)],
        "status": [status.to_string()],
        "objective": lp.objective_coefficients().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "objective_offset": [lp.objective_offset().to_bits()],
        "variable_lower": lp.variable_lower_bounds().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "variable_upper": lp.variable_upper_bounds().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "constraint_lower": lp.constraint_lower_bounds().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "constraint_upper": lp.constraint_upper_bounds().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "matrix": matrix,
        "recovered_primal": solution.primal_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "recovered_dual": solution.dual_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "recovered_status": [solution.status.to_string()],
        "variable_status": solution.variable_statuses.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "constraint_status": solution.constraint_statuses.iter().map(ToString::to_string).collect::<Vec<_>>(),
    })
}

#[test]
fn basic_presolve_passes_match_native() {
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
        assert_eq!(rust_trace(case), native.trace, "{}", case.name);
    }
}
