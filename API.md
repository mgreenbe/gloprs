# Rust API

The application API is exported directly from `gloprs`. No direct dependency
on the supporting `gloprs-lp-data` package is required.

| Task | Root exports |
|---|---|
| Build and inspect a model | `LinearProgram`, `ModelSummary`, `ModelVariableType` |
| Solve and read results | `LPSolver`, `SolveResult`, `ProblemStatus` |
| Configure solving | `GlopParameters`, `ScalingAlgorithm`, `CostScalingAlgorithm`, `PricingRule`, `InitialBasisHeuristic`, `SolverBehavior`, `TimeLimit` |
| Index variables and constraints | `ColIndex`, `RowIndex`, `VectorIndex` |
| Read solution vectors and statuses | `DenseRow`, `DenseColumn`, `VariableStatus`, `ConstraintStatus`, `VariableStatusRow`, `ConstraintStatusColumn` |
| Save or restore a basis | `SolverBasis` |
| Load MPS models | `parse_mps_file`, `parse_mps`, `MpsError`, `MpsFormat`, `parse_mps_file_with_format`, `parse_mps_with_format` |
| Supply a solution for verification | `ProblemSolution` |

Existing module paths remain available for compatibility and advanced numerical
work. Ordinary applications should use the root exports; the implementation
module tree is not the guide to the application API.

## Models and indices

Create variables and constraints with `LinearProgram::create_new_variable()`
and `create_new_constraint()`. Their returned `ColIndex` and `RowIndex` types
are distinct. Use them to set bounds, objective coefficients and matrix entries.
Call `clean_up()` before solving a model constructed or edited by hand.
`parse_mps_file()` and `parse_mps()` return cleaned models or `MpsError`.
The explicit-format readers return `(LinearProgram, MpsFormat)`, including the
format actually detected.

## Solving and errors

Create an `LPSolver` and call `solve(&model)` for a `SolveResult`.
`solve_status(&model)` instead returns just `ProblemStatus`, with solution data
available through solver getters. Both use the same default preprocessing,
scaling and numerical checks. Set parameters with `parameters_mut()` or
`set_parameters(&GlopParameters)`; unsupported settings are rejected.

Check the result status before interpreting numerical values. Only
`ProblemStatus::Optimal` certifies an optimum; feasible, infeasible, unbounded,
imprecise and invalid outcomes have different meanings. A solve does not return
`Result`: inspect `solver.last_error()` for unsupported parameters or solver
errors. Parsing errors are returned separately as `Result<_, MpsError>`.

`SolveResult` borrows solution storage from the solver: primal values,
constraint activities, dual values, reduced costs, basis statuses and available
primal/dual certificates. It also provides objective, iterations, feasibility
errors and deterministic time. End that borrow before mutating or solving
again; copy any values needed afterward. It is not an owned solution snapshot.
`ProblemSolution` is a separate owned representation for solution verification.

## Warm starts and time limits

`export_basis()` returns an owned `SolverBasis`, reusable with
`import_basis(&basis)`. Import only a basis for a compatible model, and disable
preprocessing when using a warm start, as native GLOP recommends. The
[public_lp example](examples/public_lp.rs) demonstrates construction, solving,
and a basis warm start using only root imports.

Use `GlopParameters::max_time_in_seconds` and `max_deterministic_time` for normal
limits. `solve_with_time_limit(&model, &mut TimeLimit)` exposes explicit limit
management and returns `ProblemStatus`; read the solution through solver getters.
A limit may stop before an optimal solution exists—always check status.

See [public_mps](examples/public_mps.rs) for loading and solving an MPS file,
and run `cargo doc --no-deps --open` for complete type and method documentation.
