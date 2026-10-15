//! Pinned-native checkpoints around the former presolved qap15 path split.
//!
//! Run with `cargo test --release -p gloprs --test qap15_presolved_trajectory -- --ignored`.

#![cfg(all(not(debug_assertions), target_arch = "aarch64", target_os = "macos"))]

use std::path::PathBuf;

use gloprs::parameters::GlopParameters;
use gloprs::preprocessor::MainLpPreprocessor;
use gloprs::revised_simplex::RevisedSimplex;
use gloprs::time_limit::TimeLimit;
use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
use lp_data::mps_reader::parse_mps_file;
use sha2::{Digest, Sha256};

const MPS_SHA256: &str = "872a7cfa2ea3d11b67d932c34ffa90a2ada9dc400558151a2b719889c0c595b2";
const OFFSET: u64 = 14_695_981_039_346_656_037;
const PRIME: u64 = 1_099_511_628_211;

fn hash_words(words: impl IntoIterator<Item = u64>) -> u64 {
    words
        .into_iter()
        .fold(OFFSET, |hash, word| (hash ^ word).wrapping_mul(PRIME))
}

#[test]
#[ignore = "requires ../datasets/netlib/mps/qap15.mps; several minute release-mode solve"]
fn presolved_qap15_matches_native_pricing_rng_and_terminal_state() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../datasets/netlib/mps/qap15.mps");
    let bytes = std::fs::read(&path).expect("missing qap15 Netlib input");
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), MPS_SHA256);
    let mut lp = parse_mps_file(path).expect("valid qap15 MPS");
    let parameters = GlopParameters::default();
    lp.remove_near_zero_entries(parameters.drop_magnitude);
    let mut preprocessor = MainLpPreprocessor::new();
    let mut presolve_limit = TimeLimit::new(300.0, f64::INFINITY);
    preprocessor.run(&mut lp, &parameters, &mut presolve_limit);
    assert_eq!(preprocessor.status().to_string(), "INIT");
    assert_eq!(lp.num_constraints().to_usize(), 6330);
    assert_eq!(lp.num_variables().to_usize(), 22275);

    // These values come from the pinned GLOP reference adapter at commit
    // 100f66e6242ab8bf8d32feb8f3bf086db66ae2b5.
    for cap in [7506, 65430, 1_000_000] {
        let mut direct_parameters = parameters.clone();
        direct_parameters.max_number_of_iterations = cap;
        let mut simplex = RevisedSimplex::new();
        simplex.set_parameters(&direct_parameters);
        let mut limit = TimeLimit::new(300.0, f64::INFINITY);
        simplex
            .solve(&lp, &mut limit)
            .expect("presolved qap15 solve");
        if cap == 7506 {
            assert_eq!(simplex.number_of_iterations(), 7506);
            assert_eq!(simplex.diagnostic_random_next(), 14_566_507_788_786_802_277);
            let tops = simplex.diagnostic_primal_price_tops();
            assert_eq!(tops.len(), 31);
            assert_eq!(
                hash_words(
                    tops.into_iter()
                        .flat_map(|(column, bits)| [column as u64, bits])
                ),
                2_313_578_205_268_415_744,
            );
        } else if cap == 65430 {
            assert_eq!(simplex.number_of_iterations(), 65430);
            assert_eq!(simplex.diagnostic_random_next(), 14_817_094_865_727_719_610);
        } else {
            assert_eq!(simplex.problem_status().to_string(), "OPTIMAL");
            assert_eq!(simplex.number_of_iterations(), 74317);
            assert_eq!(simplex.num_basis_refactorizations(), 527);
            assert_eq!(
                hash_words((0..lp.num_constraints().to_usize()).map(|row| {
                    u64::try_from(simplex.basis(RowIndex::from_usize(row)).value())
                        .expect("nonnegative basis index")
                })),
                17_339_455_275_875_652_566,
            );
            let columns = lp.num_variables().to_usize() + lp.num_constraints().to_usize();
            assert_eq!(
                hash_words((0..columns).map(|column| {
                    let value = simplex.reduced_cost(ColIndex::from_usize(column));
                    if value == 0.0 { 0 } else { value.to_bits() }
                })),
                9_153_896_259_488_864_112,
            );
        }
    }
}
