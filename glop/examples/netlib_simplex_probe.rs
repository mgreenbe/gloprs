//! Write opt-in Matrix Market basis/FTRAN/BTRAN probes for one Netlib model.

use std::env;
use std::path::{Path, PathBuf};

use glop::parameters::GlopParameters;
use glop::preprocessor::MainLpPreprocessor;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::mps_reader::parse_mps_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let model_path = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: netlib_simplex_probe MODEL.mps [WALL_SECONDS] [OUTPUT_ROOT]")?,
    );
    let wall_seconds = arguments
        .next()
        .map_or(Ok(120.0), |value| value.to_string_lossy().parse::<f64>())?;
    let root = arguments.next().map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/netlib-simplex-probes"),
        PathBuf::from,
    );
    if arguments.next().is_some() || !wall_seconds.is_finite() || wall_seconds <= 0.0 {
        return Err("invalid arguments".into());
    }
    let instance = model_path
        .file_stem()
        .ok_or("model has no filename stem")?
        .to_str()
        .ok_or("model filename is not UTF-8")?;
    let mut model = parse_mps_file(&model_path)?;
    let parameters = GlopParameters::default();
    model.remove_near_zero_entries(parameters.drop_magnitude);
    let mut preprocessor = MainLpPreprocessor::new();
    let mut presolve_limit = TimeLimit::new(wall_seconds, f64::INFINITY);
    preprocessor.run(&mut model, &parameters, &mut presolve_limit);
    if preprocessor.status().to_string() != "INIT" {
        return Err(format!("presolve ended with {}", preprocessor.status()).into());
    }
    let mut simplex = RevisedSimplex::new();
    simplex.set_parameters(&parameters);
    simplex.enable_simplex_probes(&root, instance)?;
    let mut solve_limit = TimeLimit::new(wall_seconds, f64::INFINITY);
    simplex.solve(&model, &mut solve_limit)?;
    println!(
        "{} status={} iterations={} refactorizations={} root={}",
        instance,
        simplex.problem_status(),
        simplex.number_of_iterations(),
        simplex.num_basis_refactorizations(),
        root.display()
    );
    Ok(())
}
