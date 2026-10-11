//! End-to-end native differential solves through the main Phase-5 presolve stack.

use glop::lp_solver::LPSolver;
use glop::parameters::{GlopParameters, InitialBasisHeuristic, ScalingAlgorithm};
use glop::preprocessor::{ScalingPreprocessor, ShiftVariableBoundsPreprocessor};
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_data::{LinearProgram, ProblemSolution};
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use serde::Deserialize;
use sha2::Digest;

const CASES: &str = include_str!("../../baselines/phase5-solve-cases.json");
const EXPECTED: &str = include_str!("../../baselines/phase5-solve-native.json");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

#[derive(Deserialize)]
struct Specification {
    upstream_commit: String,
    matched_solver_parameters: MatchedSolverParameters,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct MatchedSolverParameters {
    initial_basis_is_per_case: bool,
    singleton_shortcut_is_per_case: bool,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    mode: String,
    #[serde(default)]
    initial_basis: Option<String>,
    #[serde(default)]
    singleton_shortcut: bool,
    #[serde(default)]
    scaling_method: u8,
    matrix: Vec<(usize, usize, String)>,
    variables: Vec<(String, String, String)>,
    constraints: Vec<(String, String)>,
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
    status: String,
    iterations: u64,
    primal_bits: Vec<u64>,
    dual_bits: Vec<u64>,
    variable_status: Vec<String>,
    constraint_status: Vec<String>,
    basis: Vec<usize>,
    capped_trajectory: Vec<CappedState>,
}

#[derive(Deserialize)]
struct CappedState {
    limit: u64,
    status: String,
    basis: Vec<usize>,
    primal_bits: Vec<u64>,
    dual_bits: Vec<u64>,
}

fn number(value: &str) -> f64 {
    match value {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => value.parse().unwrap(),
    }
}

#[test]
fn full_solves_with_main_preprocessor_match_native() {
    let specification: Specification = serde_json::from_str(CASES).unwrap();
    let fixture: Fixture = serde_json::from_str(EXPECTED).unwrap();
    assert_eq!(specification.upstream_commit, UPSTREAM_COMMIT);
    assert!(
        specification
            .matched_solver_parameters
            .initial_basis_is_per_case
    );
    assert!(
        specification
            .matched_solver_parameters
            .singleton_shortcut_is_per_case
    );
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(
        fixture.cases_sha256,
        format!("{:x}", sha2::Sha256::digest(CASES.as_bytes()))
    );
    assert_eq!(specification.cases.len(), fixture.results.len());
    for (case, native) in specification.cases.iter().zip(&fixture.results) {
        assert_eq!(case.name, native.name);
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
        for (position, (objective, lower, upper)) in case.variables.iter().enumerate() {
            let col = ColIndex::from_usize(position);
            lp.set_objective_coefficient(col, number(objective));
            lp.set_variable_bounds(col, number(lower), number(upper));
        }
        for (position, (lower, upper)) in case.constraints.iter().enumerate() {
            lp.set_constraint_bounds(RowIndex::from_usize(position), number(lower), number(upper));
        }
        lp.clean_up();
        let mut solver = LPSolver::new();
        solver.set_parameters(&GlopParameters {
            initial_basis: match case.initial_basis.as_deref().unwrap_or("none") {
                "none" => InitialBasisHeuristic::None,
                "triangular" => InitialBasisHeuristic::Triangular,
                other => panic!("unknown initial basis {other}"),
            },
            exploit_singleton_column_in_initial_basis: case.singleton_shortcut,
            scaling_method: match case.scaling_method {
                0 => ScalingAlgorithm::Equilibration,
                1 => ScalingAlgorithm::LinearProgram,
                _ => panic!("unknown scaling method"),
            },
            use_preprocessing: case.mode == "shift" || case.mode == "both",
            use_scaling: case.mode == "scale" || case.mode == "both",
            ..GlopParameters::default()
        });
        let status = solver.solve(&lp);
        assert_eq!(status.to_string(), native.status, "{} status", case.name);
        assert_eq!(
            solver.number_of_simplex_iterations(),
            native.iterations,
            "{} iterations",
            case.name
        );
        let primal_bits: Vec<_> = solver
            .variable_values()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let dual_bits: Vec<_> = solver.dual_values().iter().map(|v| v.to_bits()).collect();
        assert_eq!(primal_bits, native.primal_bits, "{} primal", case.name);
        assert_eq!(dual_bits, native.dual_bits, "{} dual", case.name);
        let variable_status: Vec<_> = solver
            .variable_statuses()
            .iter()
            .map(ToString::to_string)
            .collect();
        let constraint_status: Vec<_> = solver
            .constraint_statuses()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            variable_status, native.variable_status,
            "{} variable status",
            case.name
        );
        assert_eq!(
            constraint_status, native.constraint_status,
            "{} constraint status",
            case.name
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Native input, stage order, and capped-state checks stay together.
fn capped_presolve_trajectories_match_native() {
    let specification: Specification = serde_json::from_str(CASES).unwrap();
    let fixture: Fixture = serde_json::from_str(EXPECTED).unwrap();
    let mut covered = 0;
    for (case, native) in specification.cases.iter().zip(&fixture.results) {
        if case.mode == "none" {
            continue;
        }
        covered += 1;
        assert_eq!(
            native.capped_trajectory.len(),
            usize::try_from(native.iterations).unwrap() + 1
        );
        for expected in &native.capped_trajectory {
            let mut lp = LinearProgram::new();
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
            for (position, (objective, lower, upper)) in case.variables.iter().enumerate() {
                let col = ColIndex::from_usize(position);
                lp.set_objective_coefficient(col, number(objective));
                lp.set_variable_bounds(col, number(lower), number(upper));
            }
            for (position, (lower, upper)) in case.constraints.iter().enumerate() {
                lp.set_constraint_bounds(
                    RowIndex::from_usize(position),
                    number(lower),
                    number(upper),
                );
            }
            lp.clean_up();
            let mut work = LinearProgram::new();
            work.populate_from_linear_program(&lp);
            let parameters = GlopParameters {
                initial_basis: match case.initial_basis.as_deref().unwrap_or("none") {
                    "none" => InitialBasisHeuristic::None,
                    "triangular" => InitialBasisHeuristic::Triangular,
                    other => panic!("unknown initial basis {other}"),
                },
                exploit_singleton_column_in_initial_basis: case.singleton_shortcut,
                use_preprocessing: case.mode == "shift" || case.mode == "both",
                use_scaling: case.mode == "scale" || case.mode == "both",
                scaling_method: if case.scaling_method == 1 {
                    ScalingAlgorithm::LinearProgram
                } else {
                    ScalingAlgorithm::Equilibration
                },
                max_number_of_iterations: i64::try_from(expected.limit).unwrap(),
                ..GlopParameters::default()
            };
            work.remove_near_zero_entries(parameters.drop_magnitude);
            let mut shift = ShiftVariableBoundsPreprocessor::new();
            let mut scale = ScalingPreprocessor::new();
            let shifted = parameters.use_preprocessing && shift.run(&mut work).unwrap();
            let scaled = parameters.use_scaling && scale.run(&mut work, &parameters);
            let mut simplex = RevisedSimplex::new();
            simplex.set_parameters(&parameters);
            simplex.solve(&work, &mut TimeLimit::default()).unwrap();
            assert_eq!(
                simplex.number_of_iterations(),
                expected.limit,
                "{} cap {}",
                case.name,
                expected.limit
            );
            assert_eq!(
                simplex.problem_status().to_string(),
                expected.status,
                "{} cap {} status",
                case.name,
                expected.limit
            );
            let basis: Vec<_> = (0..case.constraints.len())
                .map(|row| simplex.basis(RowIndex::from_usize(row)).to_usize())
                .collect();
            assert_eq!(
                basis, expected.basis,
                "{} cap {} basis",
                case.name, expected.limit
            );
            let mut solution = ProblemSolution::new(work.num_constraints(), work.num_variables());
            for position in 0..case.variables.len() {
                let col = ColIndex::from_usize(position);
                solution.primal_values[col] = simplex.variable_value(col);
                solution.variable_statuses[col] = simplex.variable_status(col);
            }
            for position in 0..case.constraints.len() {
                let row = RowIndex::from_usize(position);
                solution.dual_values[row] = simplex.dual_value(row);
                solution.constraint_statuses[row] = simplex.constraint_status(row);
            }
            if scaled {
                scale.recover_solution(&mut solution);
            }
            if shifted {
                shift.recover_solution(&mut solution);
            }
            let primal_bits: Vec<_> = solution
                .primal_values
                .iter()
                .map(|value| value.to_bits())
                .collect();
            let dual_bits: Vec<_> = solution
                .dual_values
                .iter()
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(
                primal_bits, expected.primal_bits,
                "{} cap {} primal",
                case.name, expected.limit
            );
            assert_eq!(
                dual_bits, expected.dual_bits,
                "{} cap {} dual",
                case.name, expected.limit
            );
        }
        if native.iterations > 0 {
            let terminal = native.capped_trajectory.last().unwrap();
            assert_eq!(terminal.basis, native.basis, "{} terminal basis", case.name);
        }
    }
    assert!(covered >= 5);
}
