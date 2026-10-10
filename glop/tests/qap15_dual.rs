//! Pinned-native regression for direct, unscaled qap15 dual simplex prefixes.
//!
//! Run with `cargo test --release -p gloprs-glop --test qap15_dual -- --ignored`.

use std::path::PathBuf;

use glop::parameters::GlopParameters;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE: &str = include_str!("../../baselines/qap15-dual.json");
const CLOCK_TRANSITION_FIXTURE: &str = include_str!("../../baselines/qap15-clock-transition.json");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

#[derive(Deserialize)]
struct Fixture {
    upstream_commit: String,
    mps_sha256: String,
    parameters: Parameters,
    snapshots: Vec<Snapshot>,
}

#[derive(Deserialize)]
struct Parameters {
    use_dual_simplex: bool,
    use_scaling: bool,
}

#[derive(Deserialize)]
struct Snapshot {
    limit: i64,
    status: String,
    iterations: u64,
    deterministic_time_bits: u64,
    basis_sha256: String,
    reduced_bits_sha256: String,
    norm_bits_sha256: String,
}

fn digest(values: impl IntoIterator<Item = String>) -> String {
    format!(
        "{:x}",
        Sha256::digest(values.into_iter().collect::<Vec<_>>().join(" ").as_bytes())
    )
}

fn check_snapshots(fixture_text: &str, filter: Option<i64>) {
    let fixture: Fixture = serde_json::from_str(fixture_text).unwrap();
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert!(fixture.parameters.use_dual_simplex);
    assert!(!fixture.parameters.use_scaling);
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps/qap15.mps");
    let bytes = std::fs::read(&path).expect("missing qap15 Netlib input");
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), fixture.mps_sha256);
    let model = parse_mps_file(path).unwrap();
    let rows = model.num_constraints().to_usize();
    let columns = model.num_variables().to_usize() + rows;

    let mut checked = 0;
    for expected in fixture.snapshots {
        if filter.is_some_and(|limit| limit != expected.limit) {
            continue;
        }
        checked += 1;
        let parameters = GlopParameters {
            use_dual_simplex: true,
            use_scaling: false,
            max_number_of_iterations: expected.limit,
            ..GlopParameters::default()
        };
        let mut simplex = RevisedSimplex::new();
        simplex.set_parameters(&parameters);
        simplex
            .solve(&model, &mut TimeLimit::new(600.0, f64::INFINITY))
            .unwrap();
        if std::env::var_os("GLOPRS_QAP15_DEBUG").is_some() {
            eprintln!(
                "limit {}: clock {}, basis {}",
                expected.limit,
                simplex.deterministic_time(),
                simplex.basis_deterministic_time()
            );
        }
        assert_eq!(simplex.problem_status().to_string(), expected.status);
        assert_eq!(simplex.number_of_iterations(), expected.iterations);
        assert_eq!(
            simplex.deterministic_time().to_bits(),
            expected.deterministic_time_bits,
            "operation clock at limit {}",
            expected.limit
        );
        assert_eq!(
            digest((0..rows).map(|row| {
                simplex
                    .basis(RowIndex::from_usize(row))
                    .to_usize()
                    .to_string()
            })),
            expected.basis_sha256,
            "ordered basis at limit {}",
            expected.limit
        );
        assert_eq!(
            digest((0..columns).map(|column| {
                simplex
                    .reduced_cost(ColIndex::from_usize(column))
                    .to_bits()
                    .to_string()
            })),
            expected.reduced_bits_sha256,
            "reduced costs at limit {}",
            expected.limit
        );
        assert_eq!(
            digest(
                simplex
                    .dual_edge_squared_norms()
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits().to_string())
            ),
            expected.norm_bits_sha256,
            "dual norms at limit {}",
            expected.limit
        );
    }
    assert!(checked > 0, "no qap15 snapshots selected");
}

#[test]
#[ignore = "requires ../datasets/netlib; opt-in qap15 direct-solve snapshots"]
fn qap15_direct_dual_matches_native_snapshots() {
    let filter = std::env::var("GLOPRS_QAP15_LIMIT")
        .ok()
        .map(|value| value.parse::<i64>().expect("invalid GLOPRS_QAP15_LIMIT"));
    check_snapshots(FIXTURE, filter);
}

#[test]
#[ignore = "requires ../datasets/netlib; opt-in qap15 former-clock-split snapshots"]
fn qap15_former_clock_split_matches_native_state() {
    check_snapshots(CLOCK_TRANSITION_FIXTURE, None);
}

#[test]
#[ignore = "diagnostic qap15 operation-clock probe; set GLOPRS_QAP15_PROBE_LIMIT"]
fn qap15_operation_clock_probe() {
    let limit: i64 = std::env::var("GLOPRS_QAP15_PROBE_LIMIT")
        .expect("missing GLOPRS_QAP15_PROBE_LIMIT")
        .parse()
        .expect("invalid GLOPRS_QAP15_PROBE_LIMIT");
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps/qap15.mps");
    let model = parse_mps_file(path).unwrap();
    let parameters = GlopParameters {
        use_dual_simplex: true,
        use_scaling: false,
        max_number_of_iterations: limit,
        change_status_to_imprecise: std::env::var_os("GLOPRS_QAP15_NO_IMPRECISE").is_none(),
        ..GlopParameters::default()
    };
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&parameters);
    simplex
        .solve(&model, &mut TimeLimit::new(600.0, f64::INFINITY))
        .unwrap();
    eprintln!(
        "limit {limit}: {} iterations, status {}, clock {:.17}, basis {:.17}, LU entries {}",
        simplex.number_of_iterations(),
        simplex.problem_status(),
        simplex.deterministic_time(),
        simplex.basis_deterministic_time(),
        simplex.basis_num_entries_in_lu()
    );
}
