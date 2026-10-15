//! Direct dual-simplex pivot paths for extra-corpus performance targets.
//!
//! Run with `cargo test --release -p gloprs-glop --test phase8_extra_trajectories -- --ignored`.

use std::io::Read;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use glop::parameters::GlopParameters;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::lp_types::{RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE: &[u8] = include_bytes!("../../baselines/phase8-extra-trajectories.json.gz");
const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

#[derive(Deserialize)]
struct Fixture {
    upstream_commit: String,
    parameters: Parameters,
    results: Vec<ModelFixture>,
}

#[derive(Deserialize)]
struct Parameters {
    use_scaling: bool,
    use_dual_simplex: bool,
}

#[derive(Deserialize)]
struct ModelFixture {
    name: String,
    input: String,
    mps_sha256: String,
    status: String,
    iterations: u64,
    basis: Vec<usize>,
    pivots: Vec<[usize; 4]>,
}

#[test]
#[ignore = "requires ../datasets/nmm0-50k and nmm50k-100k; run in release mode"]
fn selected_extra_models_match_native_pivot_paths() {
    let mut decoder = GzDecoder::new(FIXTURE);
    let mut contents = String::new();
    decoder.read_to_string(&mut contents).unwrap();
    let fixture: Fixture = serde_json::from_str(&contents).unwrap();
    assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
    assert!(!fixture.parameters.use_scaling);
    assert!(fixture.parameters.use_dual_simplex);
    assert_eq!(fixture.results.len(), 6);

    let dataset_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets");
    let parameters = GlopParameters {
        use_scaling: false,
        use_dual_simplex: true,
        max_number_of_iterations: 1_000_000,
        ..GlopParameters::default()
    };
    for expected in fixture.results {
        let path = dataset_root.join(&expected.input);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected.mps_sha256,
            "{} input checksum",
            expected.name
        );
        let model = parse_mps_file(path).unwrap();
        let mut simplex = RevisedSimplex::new();
        simplex.set_parameters(&parameters);
        simplex.set_trace_enabled(true);
        simplex
            .solve(&model, &mut TimeLimit::new(20.0, f64::INFINITY))
            .unwrap();
        assert_eq!(
            simplex.problem_status().to_string(),
            expected.status,
            "{}",
            expected.name
        );
        assert_eq!(
            simplex.number_of_iterations(),
            expected.iterations,
            "{}",
            expected.name
        );
        let basis: Vec<_> = (0..model.num_constraints().to_usize())
            .map(|row| simplex.basis(RowIndex::from_usize(row)).to_usize())
            .collect();
        assert_eq!(basis, expected.basis, "{} basis", expected.name);
        let pivots: Vec<_> = simplex
            .trace()
            .iter()
            .filter_map(|event| {
                Some([
                    event.entering_column?.to_usize(),
                    event.leaving_column?.to_usize(),
                    event.leaving_row?.to_usize(),
                    usize::try_from(event.iteration).expect("negative iteration"),
                ])
            })
            .collect();
        if pivots != expected.pivots {
            let first = pivots
                .iter()
                .zip(&expected.pivots)
                .position(|(rust, native)| rust != native);
            panic!("{} pivot path differs at {first:?}", expected.name);
        }
    }
}
