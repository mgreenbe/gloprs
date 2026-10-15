//! Linear programming with the GLOP simplex solver.
//!
//! Import the application API directly from `gloprs`:
//!
//! ```
//! use gloprs::{LinearProgram, LPSolver, ProblemStatus};
//!
//! let mut model = LinearProgram::new();
//! let x = model.create_new_variable();
//! model.set_variable_bounds(x, 0.0, 5.0);
//! model.set_objective_coefficient(x, -1.0);
//! model.clean_up();
//!
//! let mut solver = LPSolver::new();
//! let result = solver.solve(&model);
//! assert_eq!(result.status, ProblemStatus::Optimal);
//! assert_eq!(result.variable_values[x], 5.0);
//! ```
//!
//! The root exports are the application-facing API. Models and [`SolverBasis`]
//! are owned; [`SolveResult`] borrows the solver's solution storage. Check
//! [`ProblemStatus`] before using a solution, and [`LPSolver::last_error`] for
//! unsupported parameters or solver errors. [`GlopParameters`] controls settings;
//! [`parse_mps_file`] and [`parse_mps`] load models.
//!
//! Public implementation modules remain available for advanced use and
//! compatibility. They are not required for ordinary model construction,
//! solving, result inspection, or warm starts.

pub use lp_data;

pub mod basis_representation;
pub mod dual_edge_norms;
pub mod entering_variable;
pub mod initial_basis;
mod libcpp_sort;
pub mod lp_solver;
pub mod lu_factorization;
pub mod markowitz;
pub mod numerical;
pub mod parameters;
pub mod preprocessor;
pub mod pricing;
pub mod primal_edge_norms;
pub mod primal_ratio_test;
pub mod random;
pub mod rank_one_update;
pub mod reduced_costs;
pub mod revised_simplex;
pub mod simplex_probe;
pub mod stats;
pub mod status;
pub mod time_limit;
pub mod update_row;
pub mod variable_values;
pub mod variables_info;

// Application-facing model, solution, and configuration types.
#[doc(inline)]
pub use lp_data::lp_data::{LinearProgram, ModelSummary, ModelVariableType, ProblemSolution};
#[doc(inline)]
pub use lp_data::lp_types::{
    ColIndex, ConstraintStatus, ConstraintStatusColumn, DenseColumn, DenseRow, ProblemStatus,
    RowIndex, VariableStatus, VariableStatusRow, VectorIndex,
};
#[doc(inline)]
pub use lp_data::mps_reader::{
    MpsError, MpsFormat, parse_mps, parse_mps_file, parse_mps_file_with_format,
    parse_mps_with_format,
};
#[doc(inline)]
pub use lp_solver::{LPSolver, SolveResult, SolverBasis};
#[doc(inline)]
pub use parameters::{
    CostScalingAlgorithm, GlopParameters, InitialBasisHeuristic, PricingRule, ScalingAlgorithm,
    SolverBehavior,
};
#[doc(inline)]
pub use time_limit::TimeLimit;
