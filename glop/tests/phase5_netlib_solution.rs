//! Native-pinned public-solver postsolve states on the 96 fast Netlib models.

#[cfg(not(debug_assertions))]
mod release_gate {
    use std::io::Read;
    use std::path::PathBuf;

    use flate2::read::GzDecoder;
    use glop::lp_solver::LPSolver;
    use glop::parameters::GlopParameters;
    use glop::revised_simplex::RevisedSimplex;
    use glop::time_limit::TimeLimit;
    use lp_data::lp_data::LinearProgram;
    use lp_data::lp_types::{ColIndex, RowIndex, VectorIndex};
    use lp_data::mps_reader::parse_mps_file;
    use serde::Deserialize;

    const NATIVE: &[u8] = include_bytes!("../../baselines/phase5-netlib-solution-native.json.gz");
    const QAP15_NATIVE: &[u8] =
        include_bytes!("../../baselines/phase7-qap15-terminal-native.json.gz");
    const UPSTREAM_COMMIT: &str = "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5";

    #[derive(Deserialize)]
    struct Fixture {
        upstream_commit: String,
        time_limit_seconds: f64,
        results: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        lines: Vec<String>,
    }

    #[derive(Deserialize)]
    struct Qap15Fixture {
        upstream_commit: String,
        lines: Vec<String>,
    }

    fn close(expected: f64, actual: f64) -> bool {
        expected == actual
            || (expected - actual).abs() <= 1e-9 * expected.abs().max(actual.abs()).max(1.0)
    }

    fn check_values(name: &str, label: &str, expected: &str, actual: &[f64]) {
        let tokens: Vec<_> = expected.split_whitespace().collect();
        assert_eq!(tokens[0], label, "{name} {label} label");
        assert_eq!(tokens.len() - 1, actual.len(), "{name} {label} length");
        for (position, (bits, &value)) in tokens[1..].iter().zip(actual).enumerate() {
            let native = f64::from_bits(bits.parse().unwrap());
            assert!(
                close(native, value),
                "{name} {label}[{position}]: {native} != {value}"
            );
        }
    }

    /// Recompute original-model equations, stationarity, and complementary
    /// signs independently of the solver's residual and activity workspaces.
    fn check_independent_kkt(name: &str, lp: &LinearProgram, solver: &LPSolver) {
        let mut activities = vec![0.0; lp.num_constraints().to_usize()];
        let mut scales = vec![0.0; activities.len()];
        let mut objective = lp.objective_offset();
        for column in 0..lp.num_variables().to_usize() {
            let col = ColIndex::from_usize(column);
            let x = solver.variable_values()[col];
            let lower = lp.variable_lower_bounds()[col];
            let upper = lp.variable_upper_bounds()[col];
            let tolerance = 1e-6 * (1.0 + x.abs());
            assert!(
                x + tolerance >= lower && x - tolerance <= upper,
                "{name}: variable {column} bound"
            );
            objective += lp.objective_coefficients()[col] * x;
            let mut dual_dot = 0.0;
            let mut dual_scale = 1.0 + lp.objective_coefficients()[col].abs();
            for entry in lp.sparse_column(col) {
                let row = entry.index().to_usize();
                let coefficient = entry.coefficient();
                activities[row] += coefficient * x;
                scales[row] += (coefficient * x).abs();
                let term = coefficient * solver.dual_values()[entry.index()];
                dual_dot += term;
                dual_scale += term.abs();
            }
            let reduced = lp.objective_coefficients()[col] - dual_dot;
            assert!(
                (reduced - solver.reduced_costs()[col]).abs() <= 1e-7 * dual_scale,
                "{name}: variable {column} stationarity"
            );
            if !(upper.is_finite() && x >= upper - tolerance) {
                assert!(
                    reduced >= -1e-6 * dual_scale,
                    "{name}: variable {column} lower-side sign"
                );
            }
            if !(lower.is_finite() && x <= lower + tolerance) {
                assert!(
                    reduced <= 1e-6 * dual_scale,
                    "{name}: variable {column} upper-side sign"
                );
            }
        }
        for row in 0..activities.len() {
            let index = RowIndex::from_usize(row);
            let activity = activities[row];
            let tolerance = 1e-6 * (1.0 + scales[row]);
            assert!(
                (activity - solver.constraint_activities()[index]).abs() <= tolerance,
                "{name}: row {row} equation"
            );
            let lower = lp.constraint_lower_bounds()[index];
            let upper = lp.constraint_upper_bounds()[index];
            assert!(
                activity + tolerance >= lower && activity - tolerance <= upper,
                "{name}: row {row} bound"
            );
            let dual = solver.dual_values()[index];
            if !(upper.is_finite() && activity >= upper - tolerance) {
                assert!(dual >= -1e-6, "{name}: row {row} lower-side sign");
            }
            if !(lower.is_finite() && activity <= lower + tolerance) {
                assert!(dual <= 1e-6, "{name}: row {row} upper-side sign");
            }
        }
        let expected_objective = lp.objective_scaling_factor() * objective;
        assert!(
            (expected_objective - solver.objective_value()).abs()
                <= 1e-7 * (1.0 + expected_objective.abs()),
            "{name}: objective recomputation"
        );
    }

    #[test]
    #[ignore = "requires ../datasets/netlib/mps; run in release mode"]
    fn fast_96_public_postsolve_matches_native() {
        let mut decompressed = String::new();
        GzDecoder::new(NATIVE)
            .read_to_string(&mut decompressed)
            .unwrap();
        let fixture: Fixture = serde_json::from_str(&decompressed).unwrap();
        assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
        assert_eq!(fixture.results.len(), 96);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps");
        for case in fixture.results {
            let lp = parse_mps_file(root.join(format!("{}.mps", case.name))).unwrap();
            let mut solver = LPSolver::new();
            let mut limit = TimeLimit::new(fixture.time_limit_seconds, f64::INFINITY);
            let status = solver.solve_with_time_limit(&lp, &mut limit);
            if status.to_string() == "OPTIMAL" {
                check_independent_kkt(&case.name, &lp, &solver);
            }
            assert_eq!(case.lines.len(), 8);
            let header: Vec<_> = case.lines[0].split_whitespace().collect();
            assert_eq!(status.to_string(), header[0], "{} status", case.name);
            assert_eq!(
                solver.number_of_simplex_iterations(),
                header[1].parse::<u64>().unwrap(),
                "{} iterations",
                case.name
            );
            for (label, bits, actual) in [
                ("objective", header[2], solver.objective_value()),
                (
                    "primal infeasibility",
                    header[3],
                    solver.maximum_primal_infeasibility(),
                ),
                (
                    "dual infeasibility",
                    header[4],
                    solver.maximum_dual_infeasibility(),
                ),
            ] {
                let native = f64::from_bits(bits.parse().unwrap());
                assert!(
                    close(native, actual),
                    "{} {label}: {native} != {actual}",
                    case.name
                );
            }
            let dimensions: Vec<_> = case.lines[1].split_whitespace().collect();
            assert_eq!(
                lp.num_variables().value(),
                dimensions[0].parse::<i32>().unwrap()
            );
            assert_eq!(
                lp.num_constraints().value(),
                dimensions[1].parse::<i32>().unwrap()
            );
            check_values(
                &case.name,
                "primal",
                &case.lines[2],
                solver.variable_values().as_slice(),
            );
            check_values(
                &case.name,
                "reduced",
                &case.lines[3],
                solver.reduced_costs().as_slice(),
            );
            check_values(
                &case.name,
                "dual",
                &case.lines[4],
                solver.dual_values().as_slice(),
            );
            check_values(
                &case.name,
                "activity",
                &case.lines[5],
                solver.constraint_activities().as_slice(),
            );
            let variables: Vec<_> = case.lines[6].split_whitespace().collect();
            assert_eq!(variables[0], "variable_status");
            assert_eq!(
                variables.len() - 1,
                solver.variable_statuses().len().to_usize()
            );
            for (native, rust) in variables[1..].iter().zip(solver.variable_statuses()) {
                assert_eq!(*native, rust.to_string(), "{} variable status", case.name);
            }
            let constraints: Vec<_> = case.lines[7].split_whitespace().collect();
            assert_eq!(constraints[0], "constraint_status");
            assert_eq!(
                constraints.len() - 1,
                solver.constraint_statuses().len().to_usize()
            );
            for (native, rust) in constraints[1..].iter().zip(solver.constraint_statuses()) {
                assert_eq!(*native, rust.to_string(), "{} constraint status", case.name);
            }
        }
    }

    #[test]
    #[ignore = "requires ../datasets/netlib/mps/qap15.mps; about two minutes in release mode"]
    fn qap15_terminal_solution_satisfies_original_model_kkt() {
        let mut decompressed = String::new();
        GzDecoder::new(QAP15_NATIVE)
            .read_to_string(&mut decompressed)
            .unwrap();
        let fixture: Qap15Fixture = serde_json::from_str(&decompressed).unwrap();
        assert_eq!(fixture.upstream_commit, UPSTREAM_COMMIT);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps");
        let lp = parse_mps_file(root.join("qap15.mps")).unwrap();
        let mut solver = LPSolver::new();
        let mut limit = TimeLimit::new(180.0, f64::INFINITY);
        let status = solver.solve_with_time_limit(&lp, &mut limit);
        assert_eq!(status.to_string(), "OPTIMAL");
        let native: Vec<_> = fixture.lines[0].split_whitespace().collect();
        assert_eq!(native[0], "OPTIMAL");
        assert!(close(
            f64::from_bits(native[2].parse().unwrap()),
            solver.objective_value()
        ));
        check_independent_kkt("qap15", &lp, &solver);
    }

    #[test]
    #[ignore = "requires ../datasets/netlib/mps/pilot.mps; run in release mode"]
    fn pilot_without_presolve_or_scaling_matches_native() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps");
        let lp = parse_mps_file(root.join("pilot.mps")).unwrap();
        let mut solver = LPSolver::new();
        solver.set_parameters(&GlopParameters {
            use_preprocessing: false,
            use_scaling: false,
            ..GlopParameters::default()
        });
        let mut limit = TimeLimit::new(20.0, f64::INFINITY);
        assert_eq!(
            solver.solve_with_time_limit(&lp, &mut limit).to_string(),
            "OPTIMAL"
        );
        assert_eq!(solver.number_of_simplex_iterations(), 4_427);
        assert_eq!(
            solver.objective_value().to_bits(),
            13_871_486_983_988_092_121
        );
        assert_eq!(
            solver.maximum_primal_infeasibility().to_bits(),
            4_465_016_664_398_335_355
        );
        assert_eq!(
            solver.maximum_dual_infeasibility().to_bits(),
            4_486_291_244_646_182_827
        );
    }

    #[test]
    #[ignore = "requires ../datasets/netlib/mps/pilot.mps; run in release mode"]
    fn pilot_primal_prefixes_match_native() {
        // These hashes were captured from the pinned GLOP direct simplex on
        // the public solver's copied, near-zero-cleaned model.
        let prefixes = [
            (
                0,
                7_991_495_002_123_035_064_u64,
                15_170_392_146_534_748_314_u64,
            ),
            (1, 16_312_023_428_594_291_821, 15_170_392_146_534_748_314),
            (50, 2_952_965_549_467_288_915, 15_170_392_146_534_748_314),
            (500, 6_816_574_859_943_369_995, 675_440_610_316_951_005),
            (2000, 15_737_156_557_865_847_102, 4_475_385_536_542_171_291),
            (4427, 4_128_710_273_636_839_663, 5_356_145_131_740_024_392),
        ];
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib/mps");
        let input = parse_mps_file(root.join("pilot.mps")).unwrap();
        let mut model = LinearProgram::new();
        model.populate_from_linear_program(&input);
        model.remove_near_zero_entries(GlopParameters::default().drop_magnitude);
        model.clear_transpose_matrix();
        let total_columns = model.num_variables().to_usize() + model.num_constraints().to_usize();
        for (cap, native_basis, native_reduced) in prefixes {
            let parameters = GlopParameters {
                use_preprocessing: false,
                use_scaling: false,
                max_number_of_iterations: cap,
                ..GlopParameters::default()
            };
            let mut simplex = RevisedSimplex::new();
            simplex.set_parameters(&parameters);
            let mut limit = TimeLimit::new(20.0, f64::INFINITY);
            simplex.solve(&model, &mut limit).unwrap();
            assert_eq!(simplex.number_of_iterations(), cap as u64);
            let basis = (0..model.num_constraints().to_usize())
                .map(|row| simplex.basis(RowIndex::from_usize(row)).to_usize() as u64)
                .fold(14_695_981_039_346_656_037_u64, |hash, value| {
                    (hash ^ value).wrapping_mul(1_099_511_628_211)
                });
            let reduced = (0..total_columns)
                .map(|col| simplex.reduced_cost(ColIndex::from_usize(col)).to_bits())
                .fold(14_695_981_039_346_656_037_u64, |hash, bits| {
                    (hash ^ bits).wrapping_mul(1_099_511_628_211)
                });
            assert_eq!(basis, native_basis, "pilot cap {cap} basis");
            assert_eq!(reduced, native_reduced, "pilot cap {cap} reduced costs");
        }
    }
}
