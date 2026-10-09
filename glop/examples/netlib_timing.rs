use std::env;
use std::time::Instant;

use glop::parameters::GlopParameters;
use glop::revised_simplex::RevisedSimplex;
use glop::time_limit::TimeLimit;
use lp_data::mps_reader::parse_mps_file;

fn main() {
    let mut arguments = env::args_os().skip(1);
    let path = arguments.next().expect("missing MPS path");
    let repetitions: usize = arguments.next().map_or(1, |value| {
        value
            .to_string_lossy()
            .parse()
            .expect("invalid repetitions")
    });
    let mode = arguments.next();
    assert!(mode.as_ref().is_none_or(|value| value == "primal"));
    let primal = mode.is_some();
    assert!(arguments.next().is_none());
    assert!(repetitions > 0);

    let model = parse_mps_file(path).expect("failed to parse MPS");
    let parameters = GlopParameters {
        use_scaling: false,
        use_dual_simplex: !primal,
        max_number_of_iterations: 1_000_000,
        ..GlopParameters::default()
    };
    let start = Instant::now();
    let mut simplex = RevisedSimplex::new();
    for _ in 0..repetitions {
        simplex = RevisedSimplex::new();
        simplex.set_parameters(&parameters);
        let mut limit = TimeLimit::new(20.0, f64::INFINITY);
        simplex.solve(&model, &mut limit).expect("solve failed");
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{elapsed:.17} {} {} {:.17}",
        simplex.problem_status(),
        simplex.number_of_iterations(),
        simplex.objective_value()
    );
}
