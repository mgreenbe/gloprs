//! Bit-exact default-presolve states from pinned GLOP on 96 Netlib models.
//! Run with `cargo test -p gloprs-glop --test main_preprocessor_netlib -- --ignored`.

use std::fmt::Write;
use std::path::PathBuf;

#[cfg(not(debug_assertions))]
use glop::lp_solver::LPSolver;
use glop::parameters::GlopParameters;
use glop::preprocessor::MainLpPreprocessor;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const NATIVE: &str = include_str!("../../baselines/main-preprocessor-netlib-native.json");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

#[derive(Deserialize)]
struct Fixture {
    upstream_commit: String,
    results: Vec<ModelFixture>,
}

#[derive(Deserialize)]
struct ModelFixture {
    name: String,
    presolved_sha256: String,
    presolved_summary: String,
    #[cfg(not(debug_assertions))]
    solve_status: String,
    #[cfg(not(debug_assertions))]
    solve_iterations: u64,
}

#[cfg(not(debug_assertions))]
#[test]
#[ignore = "requires ../datasets/netlib/mps; default pipeline on 96 models in release mode"]
fn fast_96_default_pipeline_matches_native_iterations() {
    let fixture: Fixture = serde_json::from_str(NATIVE).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps");
    for case in &fixture.results {
        let lp = parse_mps_file(root.join(format!("{}.mps", case.name))).unwrap();
        let mut solver = LPSolver::new();
        let mut limit = TimeLimit::new(20.0, f64::INFINITY);
        let status = solver.solve_with_time_limit(&lp, &mut limit);
        assert_eq!(
            status.to_string(),
            case.solve_status,
            "{} status",
            case.name
        );
        assert_eq!(
            solver.number_of_simplex_iterations(),
            case.solve_iterations,
            "{} iterations",
            case.name
        );
    }
}

#[test]
#[ignore = "requires ../datasets/netlib/mps; run to validate default presolve on 96 models"]
fn fast_96_presolved_models_match_native_bit_for_bit() {
    let fixture: Fixture = serde_json::from_str(NATIVE).unwrap();
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(fixture.results.len(), 96);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps");
    for case in &fixture.results {
        let mut lp = parse_mps_file(root.join(format!("{}.mps", case.name))).unwrap();
        let parameters = GlopParameters::default();
        lp.remove_near_zero_entries(parameters.drop_magnitude);
        let mut preprocessor = MainLpPreprocessor::new();
        let mut limit = TimeLimit::new(20.0, f64::INFINITY);
        preprocessor.run(&mut lp, &parameters, &mut limit);
        let mut state = format!(
            "presolved {} {} {} {}\n",
            preprocessor.status(),
            lp.num_constraints().value(),
            lp.num_variables().value(),
            lp.num_entries().value()
        );
        assert_eq!(
            state.trim_end(),
            case.presolved_summary,
            "{} dimensions",
            case.name
        );
        for column in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(column);
            writeln!(
                state,
                "col {column} {} {} {}",
                lp.objective_coefficients()[col].to_bits(),
                lp.variable_lower_bounds()[col].to_bits(),
                lp.variable_upper_bounds()[col].to_bits()
            )
            .unwrap();
            for entry in lp.sparse_column(col) {
                writeln!(
                    state,
                    "entry {column} {} {}",
                    entry.index().value(),
                    entry.coefficient().to_bits()
                )
                .unwrap();
            }
        }
        for row in 0..lp.num_constraints().to_usize() {
            let r = RowIndex::from_usize(row);
            writeln!(
                state,
                "row {row} {} {}",
                lp.constraint_lower_bounds()[r].to_bits(),
                lp.constraint_upper_bounds()[r].to_bits()
            )
            .unwrap();
        }
        writeln!(state, "offset {}", lp.objective_offset().to_bits()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(state.as_bytes())),
            case.presolved_sha256,
            "{} presolved LP",
            case.name
        );
    }
}
