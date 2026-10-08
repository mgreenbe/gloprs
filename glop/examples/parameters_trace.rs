use std::io::{self, Read};

use glop::parameters::GlopParameters;

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut fields = input.split_whitespace();
    let field = fields.next().unwrap();
    let value: f64 = fields.next().unwrap().parse().unwrap();
    let mut p = GlopParameters::default();
    macro_rules! set_double {
        ($($name:ident),+ $(,)?) => { match field { $(stringify!($name) => p.$name = value,)+ _ => set_integer(&mut p, field, value), } };
    }
    set_double!(
        degenerate_ministep_factor,
        drop_tolerance,
        dual_feasibility_tolerance,
        dual_small_pivot_threshold,
        dualizer_threshold,
        harris_tolerance_ratio,
        lu_factorization_pivot_threshold,
        markowitz_singularity_threshold,
        max_number_of_reoptimizations,
        minimum_acceptable_pivot,
        preprocessor_zero_tolerance,
        primal_feasibility_tolerance,
        ratio_test_zero_threshold,
        recompute_edges_norm_threshold,
        recompute_reduced_costs_threshold,
        refactorization_threshold,
        relative_cost_perturbation,
        relative_max_cost_perturbation,
        small_pivot_threshold,
        solution_feasibility_tolerance,
        objective_lower_limit,
        objective_upper_limit,
        crossover_bound_snapping_distance,
        initial_condition_number_threshold,
        max_deterministic_time,
        max_time_in_seconds,
        max_valid_magnitude,
        drop_magnitude,
    );
    println!("{}", p.validate().err().unwrap_or_else(|| "OK".into()));
}

fn set_integer(p: &mut GlopParameters, field: &str, value: f64) {
    #[allow(clippy::cast_possible_truncation)]
    let value = value as i32;
    match field {
        "basis_refactorization_period" => p.basis_refactorization_period = value,
        "devex_weights_reset_period" => p.devex_weights_reset_period = value,
        "num_omp_threads" => p.num_omp_threads = value,
        "random_seed" => p.random_seed = value,
        "markowitz_zlatev_parameter" => p.markowitz_zlatev_parameter = value,
        _ => panic!("unknown field"),
    }
}
