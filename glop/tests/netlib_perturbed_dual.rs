//! Native-GLOP perturbed-dual regression over the 96 fast Netlib instances.
//!
//! Run with:
//! `cargo test --release -p gloprs-glop --test netlib_perturbed_dual -- --ignored --nocapture`

use std::fmt::Debug;
use std::io::Read;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use glop::parameters::GlopParameters;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;
use serde::Deserialize;
use sha2::Digest;

const PERTURBED_FIXTURE_BYTES: &[u8] =
    include_bytes!("../../baselines/netlib-perturbed-dual.json.gz");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";
const INSTANCE_TIME_LIMIT_SECONDS: f64 = 20.0;

#[derive(Deserialize)]
struct Parameters {
    use_dual_simplex: bool,
    use_scaling: bool,
    perturb_costs_in_dual_simplex: bool,
}

#[derive(Deserialize)]
struct PerturbedFixture {
    upstream_commit: String,
    parameters: Parameters,
    omitted_models: Vec<String>,
    results: Vec<PerturbedModelFixture>,
}

#[derive(Deserialize)]
struct PerturbedModelFixture {
    name: String,
    input: String,
    mps_sha256: String,
    status: String,
    iterations: u64,
    updates: usize,
    basis: Vec<usize>,
    value_bits: Vec<u64>,
    reduced_bits: Vec<u64>,
    norm_bits: Vec<u64>,
}

fn read_perturbed_fixture() -> PerturbedFixture {
    let mut decoder = GzDecoder::new(PERTURBED_FIXTURE_BYTES);
    let mut contents = String::new();
    decoder
        .read_to_string(&mut contents)
        .expect("failed to decompress perturbed native-GLOP fixture");
    serde_json::from_str(&contents).expect("failed to parse perturbed native-GLOP fixture")
}

fn normalized_float_bits(bits: u64) -> u64 {
    if bits.trailing_zeros() >= 63 { 0 } else { bits }
}

#[test]
#[ignore = "requires ../datasets/netlib"]
fn perturbed_afiro_matches_native_pivot_path() {
    let input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps/afiro.mps");
    let model = parse_mps_file(input).expect("failed to parse afiro.mps");
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_scaling: false,
        use_dual_simplex: true,
        perturb_costs_in_dual_simplex: true,
        max_number_of_iterations: 1_000_000,
        ..GlopParameters::default()
    });
    simplex
        .solve(
            &model,
            &mut TimeLimit::new(INSTANCE_TIME_LIMIT_SECONDS, f64::INFINITY),
        )
        .expect("perturbed afiro solve failed");
    assert_eq!(simplex.problem_status().to_string(), "OPTIMAL");
    assert_eq!(simplex.number_of_iterations(), 14);
    let basis: Vec<_> = (0..model.num_constraints().to_usize())
        .map(|row| simplex.basis(RowIndex::from_usize(row)).to_usize())
        .collect();
    assert_eq!(
        basis,
        [
            38, 48, 49, 50, 51, 54, 56, 57, 58, 14, 30, 29, 3, 19, 28, 16, 17, 4, 12, 1, 0, 2, 15,
            7, 6, 5, 9,
        ]
    );
}

#[test]
#[ignore = "requires ../datasets/netlib"]
fn perturbed_scsd6_matches_native_final_tie_order() {
    let input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps/scsd6.mps");
    let model = parse_mps_file(input).expect("failed to parse scsd6.mps");
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&GlopParameters {
        use_scaling: false,
        use_dual_simplex: true,
        perturb_costs_in_dual_simplex: true,
        max_number_of_iterations: 1_000_000,
        ..GlopParameters::default()
    });
    simplex
        .solve(
            &model,
            &mut TimeLimit::new(INSTANCE_TIME_LIMIT_SECONDS, f64::INFINITY),
        )
        .expect("perturbed scsd6 solve failed");
    assert_eq!(simplex.problem_status().to_string(), "OPTIMAL");
    assert_eq!(simplex.number_of_iterations(), 283);
    // Pinned GLOP's ordered basis after the final tied leaving choice.
    let ordered_basis_hash = (0..model.num_constraints().to_usize()).fold(
        14_695_981_039_346_656_037_u64,
        |hash, row| {
            (hash ^ simplex.basis(RowIndex::from_usize(row)).to_usize() as u64)
                .wrapping_mul(1_099_511_628_211)
        },
    );
    assert_eq!(ordered_basis_hash, 3_111_341_882_033_243_066);
}

fn assert_exact<T: Debug + PartialEq>(model: &str, field: &str, actual: &[T], expected: &[T]) {
    if actual == expected {
        return;
    }
    let first_difference = actual
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right);
    let detail = first_difference.map_or_else(
        || "one sequence is a prefix of the other".to_owned(),
        |index| {
            format!(
                "first difference at index {index}: gloprs={:?}, native={:?}",
                actual[index], expected[index]
            )
        },
    );
    panic!(
        "{model}: {field} differs ({detail}); gloprs length={}, native length={}",
        actual.len(),
        expected.len()
    );
}

#[test]
#[ignore = "requires ../datasets/netlib and takes about a minute in release mode"]
#[allow(clippy::too_many_lines)]
fn perturbed_dual_matches_native_glop_on_fast_netlib_instances() {
    let fixture = read_perturbed_fixture();
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert!(fixture.parameters.use_dual_simplex);
    assert!(!fixture.parameters.use_scaling);
    assert!(fixture.parameters.perturb_costs_in_dual_simplex);
    assert_eq!(fixture.omitted_models, ["qap12", "qap15"]);
    assert_eq!(fixture.results.len(), 96);

    let dataset_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib");
    let parameters = GlopParameters {
        use_scaling: false,
        use_dual_simplex: true,
        perturb_costs_in_dual_simplex: true,
        max_number_of_iterations: 1_000_000,
        ..GlopParameters::default()
    };

    for expected in fixture.results {
        let input = dataset_root.join(&expected.input);
        let digest = sha2::Sha256::digest(std::fs::read(&input).unwrap_or_else(|error| {
            panic!("{}: failed to read Netlib input: {error}", expected.name)
        }));
        assert_eq!(
            format!("{digest:x}"),
            expected.mps_sha256,
            "{}: input checksum",
            expected.name
        );
        let model = parse_mps_file(input).unwrap_or_else(|error| {
            panic!("{}: failed to parse Netlib input: {error}", expected.name)
        });
        let mut simplex = RevisedSimplex::new();
        simplex.set_parameters(&parameters);
        simplex
            .solve(
                &model,
                &mut TimeLimit::new(INSTANCE_TIME_LIMIT_SECONDS, f64::INFINITY),
            )
            .unwrap_or_else(|error| panic!("{}: solve failed: {error}", expected.name));

        assert_eq!(
            simplex.problem_status().to_string(),
            expected.status,
            "{}: status",
            expected.name
        );
        assert_eq!(
            simplex.number_of_iterations(),
            expected.iterations,
            "{}: iterations",
            expected.name
        );
        assert_eq!(
            simplex.num_basis_updates(),
            expected.updates,
            "{}: basis updates",
            expected.name
        );

        let basis: Vec<_> = (0..model.num_constraints().to_usize())
            .map(|row| simplex.basis(RowIndex::from_usize(row)).to_usize())
            .collect();
        assert_exact(&expected.name, "ordered basis", &basis, &expected.basis);

        let total_columns = model.num_variables().to_usize() + model.num_constraints().to_usize();
        let value_bits: Vec<_> = (0..total_columns)
            .map(|column| {
                normalized_float_bits(
                    simplex
                        .variable_value(ColIndex::from_usize(column))
                        .to_bits(),
                )
            })
            .collect();
        let expected_values: Vec<_> = expected
            .value_bits
            .iter()
            .copied()
            .map(normalized_float_bits)
            .collect();
        assert_exact(
            &expected.name,
            "primal value bits",
            &value_bits,
            &expected_values,
        );

        let reduced_bits: Vec<_> = (0..total_columns)
            .map(|column| simplex.reduced_cost(ColIndex::from_usize(column)).to_bits())
            .collect();
        assert_exact(
            &expected.name,
            "reduced-cost bits",
            &reduced_bits,
            &expected.reduced_bits,
        );

        let norm_bits: Vec<_> = simplex
            .dual_edge_squared_norms()
            .unwrap_or_else(|error| {
                panic!("{}: failed to obtain dual norms: {error}", expected.name)
            })
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_exact(
            &expected.name,
            "dual-edge norm bits",
            &norm_bits,
            &expected.norm_bits,
        );
        eprintln!("{}: perturbed native result matched", expected.name);
    }
}
