//! Consumer-level coverage: no implementation modules or direct `lp_data` import.
#![allow(clippy::float_cmp)]
use gloprs::{
    ColIndex, GlopParameters, LPSolver, LinearProgram, MpsError, MpsFormat, ProblemStatus,
    RowIndex, SolveResult, SolverBasis, VariableStatus, parse_mps, parse_mps_with_format,
};

#[test]
fn construct_solve_and_warm_start_using_only_root_exports() {
    let mut model = LinearProgram::new();
    let x: ColIndex = model.create_new_variable();
    let row: RowIndex = model.create_new_constraint();
    model.set_variable_bounds(x, 0.0, 10.0);
    model.set_constraint_bounds(row, 2.0, f64::INFINITY);
    model.set_coefficient(row, x, 1.0);
    model.set_objective_coefficient(x, 1.0);
    model.clean_up();

    let parameters = GlopParameters {
        use_preprocessing: false,
        use_scaling: false,
        ..GlopParameters::default()
    };
    let mut solver = LPSolver::new();
    solver.set_parameters(&parameters);
    let result: SolveResult<'_> = solver.solve(&model);
    assert_eq!(result.status, ProblemStatus::Optimal);
    assert_eq!(result.objective_value, 2.0);
    assert_eq!(result.variable_values[x], 2.0);
    assert_eq!(result.variable_statuses[x], VariableStatus::Basic);
    assert_eq!(result.constraint_activities[row], 2.0);
    assert!(solver.last_error().is_none());

    let basis: SolverBasis = solver.export_basis();
    let mut warm = LPSolver::new();
    warm.set_parameters(&parameters);
    warm.import_basis(&basis);
    let result = warm.solve(&model);
    assert_eq!(result.status, ProblemStatus::Optimal);
    assert_eq!(result.objective_value, 2.0);
    assert_eq!(result.simplex_iterations, 0);
}

#[test]
fn parse_and_report_errors_using_only_root_exports() {
    let input = include_str!("fixtures/cli/model.mps");
    let model = parse_mps(input).unwrap();
    let mut solver = LPSolver::new();
    assert_eq!(solver.solve(&model).objective_value, -11.0);
    let (_model, format): (LinearProgram, MpsFormat) =
        parse_mps_with_format(input, MpsFormat::AutoDetect).unwrap();
    assert_eq!(format, MpsFormat::Free);
    let error: Result<LinearProgram, MpsError> = parse_mps("not an MPS model");
    assert!(error.is_err());
}

#[test]
fn status_only_api_matches_structured_result() {
    let model = parse_mps(include_str!("fixtures/cli/model.mps")).unwrap();
    let mut rich = LPSolver::new();
    let result: SolveResult<'_> = rich.solve(&model);
    let mut plain = LPSolver::new();
    let status: ProblemStatus = plain.solve_status(&model);
    assert_eq!(status, result.status);
    assert_eq!(plain.objective_value(), result.objective_value);
    assert_eq!(plain.variable_values(), result.variable_values);
    assert_eq!(plain.dual_values(), result.dual_values);
    assert_eq!(plain.variable_statuses(), result.variable_statuses);
    assert_eq!(
        plain.number_of_simplex_iterations(),
        result.simplex_iterations
    );
}
