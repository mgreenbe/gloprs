//! Direct Rust representation of pinned `ortools/glop/parameters.proto`.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScalingAlgorithm {
    Default,
    #[default]
    Equilibration,
    LinearProgram,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SolverBehavior {
    AlwaysDo,
    NeverDo,
    #[default]
    LetSolverDecide,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PricingRule {
    Dantzig,
    #[default]
    SteepestEdge,
    Devex,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InitialBasisHeuristic {
    None,
    Bixby,
    #[default]
    Triangular,
    Maros,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CostScalingAlgorithm {
    NoCostScaling,
    #[default]
    ContainOneCostScaling,
    MeanCostScaling,
    MedianCostScaling,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct GlopParameters {
    pub scaling_method: ScalingAlgorithm,
    pub feasibility_rule: PricingRule,
    pub optimization_rule: PricingRule,
    pub refactorization_threshold: f64,
    pub recompute_reduced_costs_threshold: f64,
    pub recompute_edges_norm_threshold: f64,
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
    pub ratio_test_zero_threshold: f64,
    pub harris_tolerance_ratio: f64,
    pub small_pivot_threshold: f64,
    pub minimum_acceptable_pivot: f64,
    pub drop_tolerance: f64,
    pub use_scaling: bool,
    pub cost_scaling: CostScalingAlgorithm,
    pub initial_basis: InitialBasisHeuristic,
    pub use_transposed_matrix: bool,
    pub basis_refactorization_period: i32,
    pub dynamically_adjust_refactorization_period: bool,
    pub solve_dual_problem: SolverBehavior,
    pub dualizer_threshold: f64,
    pub solution_feasibility_tolerance: f64,
    pub provide_strong_optimal_guarantee: bool,
    pub change_status_to_imprecise: bool,
    pub max_number_of_reoptimizations: f64,
    pub lu_factorization_pivot_threshold: f64,
    pub max_time_in_seconds: f64,
    pub max_deterministic_time: f64,
    pub max_number_of_iterations: i64,
    pub markowitz_zlatev_parameter: i32,
    pub markowitz_singularity_threshold: f64,
    pub use_dual_simplex: bool,
    pub allow_simplex_algorithm_change: bool,
    pub devex_weights_reset_period: i32,
    pub use_preprocessing: bool,
    pub use_middle_product_form_update: bool,
    pub initialize_devex_with_column_norms: bool,
    pub exploit_singleton_column_in_initial_basis: bool,
    pub dual_small_pivot_threshold: f64,
    pub preprocessor_zero_tolerance: f64,
    pub objective_lower_limit: f64,
    pub objective_upper_limit: f64,
    pub degenerate_ministep_factor: f64,
    pub random_seed: i32,
    pub use_absl_random: bool,
    pub num_omp_threads: i32,
    pub perturb_costs_in_dual_simplex: bool,
    pub use_dedicated_dual_feasibility_algorithm: bool,
    pub relative_cost_perturbation: f64,
    pub relative_max_cost_perturbation: f64,
    pub initial_condition_number_threshold: f64,
    pub log_search_progress: bool,
    pub log_to_stdout: bool,
    pub crossover_bound_snapping_distance: f64,
    pub push_to_vertex: bool,
    pub use_implied_free_preprocessor: bool,
    pub max_valid_magnitude: f64,
    pub drop_magnitude: f64,
    pub dual_price_prioritize_norm: bool,
}

impl Default for GlopParameters {
    fn default() -> Self {
        Self {
            scaling_method: ScalingAlgorithm::Equilibration,
            feasibility_rule: PricingRule::SteepestEdge,
            optimization_rule: PricingRule::SteepestEdge,
            refactorization_threshold: 1e-9,
            recompute_reduced_costs_threshold: 1e-8,
            recompute_edges_norm_threshold: 100.0,
            primal_feasibility_tolerance: 1e-8,
            dual_feasibility_tolerance: 1e-8,
            ratio_test_zero_threshold: 1e-9,
            harris_tolerance_ratio: 0.5,
            small_pivot_threshold: 1e-6,
            minimum_acceptable_pivot: 1e-6,
            drop_tolerance: 1e-14,
            use_scaling: true,
            cost_scaling: CostScalingAlgorithm::ContainOneCostScaling,
            initial_basis: InitialBasisHeuristic::Triangular,
            use_transposed_matrix: true,
            basis_refactorization_period: 64,
            dynamically_adjust_refactorization_period: true,
            solve_dual_problem: SolverBehavior::LetSolverDecide,
            dualizer_threshold: 1.5,
            solution_feasibility_tolerance: 1e-6,
            provide_strong_optimal_guarantee: true,
            change_status_to_imprecise: true,
            max_number_of_reoptimizations: 40.0,
            lu_factorization_pivot_threshold: 0.01,
            max_time_in_seconds: f64::INFINITY,
            max_deterministic_time: f64::INFINITY,
            max_number_of_iterations: -1,
            markowitz_zlatev_parameter: 3,
            markowitz_singularity_threshold: 1e-15,
            use_dual_simplex: false,
            allow_simplex_algorithm_change: false,
            devex_weights_reset_period: 150,
            use_preprocessing: true,
            use_middle_product_form_update: true,
            initialize_devex_with_column_norms: true,
            exploit_singleton_column_in_initial_basis: true,
            dual_small_pivot_threshold: 1e-4,
            preprocessor_zero_tolerance: 1e-9,
            objective_lower_limit: f64::NEG_INFINITY,
            objective_upper_limit: f64::INFINITY,
            degenerate_ministep_factor: 0.01,
            random_seed: 1,
            use_absl_random: false,
            num_omp_threads: 1,
            perturb_costs_in_dual_simplex: false,
            use_dedicated_dual_feasibility_algorithm: true,
            relative_cost_perturbation: 1e-5,
            relative_max_cost_perturbation: 1e-7,
            initial_condition_number_threshold: 1e50,
            log_search_progress: false,
            log_to_stdout: true,
            crossover_bound_snapping_distance: f64::INFINITY,
            push_to_vertex: true,
            use_implied_free_preprocessor: true,
            max_valid_magnitude: 1e30,
            drop_magnitude: 1e-30,
            dual_price_prioritize_norm: false,
        }
    }
}

impl GlopParameters {
    /// Applies pinned GLOP's validation rules in source order.
    ///
    /// # Errors
    /// Returns the same first diagnostic as `ValidateParameters()`.
    pub fn validate(&self) -> Result<(), String> {
        macro_rules! finite_nonnegative {
            ($field:ident) => {{
                let value = self.$field;
                if !value.is_finite() {
                    return Err(format!(
                        "parameter '{}' is NaN or not finite",
                        stringify!($field)
                    ));
                }
                if value < 0.0 {
                    return Err(format!(
                        "Parameters '{}' must be non-negative",
                        stringify!($field)
                    ));
                }
            }};
        }
        macro_rules! nonnegative {
            ($field:ident) => {{
                let value = self.$field;
                if value.is_nan() {
                    return Err(format!("parameter '{}' is NaN", stringify!($field)));
                }
                if value < 0.0 {
                    return Err(format!(
                        "Parameters '{}' must be non-negative",
                        stringify!($field)
                    ));
                }
            }};
        }
        finite_nonnegative!(degenerate_ministep_factor);
        finite_nonnegative!(drop_tolerance);
        finite_nonnegative!(dual_feasibility_tolerance);
        finite_nonnegative!(dual_small_pivot_threshold);
        finite_nonnegative!(dualizer_threshold);
        finite_nonnegative!(harris_tolerance_ratio);
        finite_nonnegative!(lu_factorization_pivot_threshold);
        finite_nonnegative!(markowitz_singularity_threshold);
        finite_nonnegative!(max_number_of_reoptimizations);
        finite_nonnegative!(minimum_acceptable_pivot);
        finite_nonnegative!(preprocessor_zero_tolerance);
        finite_nonnegative!(primal_feasibility_tolerance);
        finite_nonnegative!(ratio_test_zero_threshold);
        finite_nonnegative!(recompute_edges_norm_threshold);
        finite_nonnegative!(recompute_reduced_costs_threshold);
        finite_nonnegative!(refactorization_threshold);
        finite_nonnegative!(relative_cost_perturbation);
        finite_nonnegative!(relative_max_cost_perturbation);
        finite_nonnegative!(small_pivot_threshold);
        finite_nonnegative!(solution_feasibility_tolerance);
        if self.objective_lower_limit.is_nan() {
            return Err("parameter 'objective_lower_limit' is NaN".into());
        }
        if self.objective_upper_limit.is_nan() {
            return Err("parameter 'objective_upper_limit' is NaN".into());
        }
        nonnegative!(crossover_bound_snapping_distance);
        nonnegative!(initial_condition_number_threshold);
        nonnegative!(max_deterministic_time);
        nonnegative!(max_time_in_seconds);
        finite_nonnegative!(max_valid_magnitude);
        if self.max_valid_magnitude > 1e100 {
            return Err("max_valid_magnitude must be <= 1e100".into());
        }
        finite_nonnegative!(drop_magnitude);
        if self.drop_magnitude < 1e-100 {
            return Err("drop magnitude must be finite and >= 1e-100".into());
        }
        for (name, value) in [
            (
                "basis_refactorization_period",
                self.basis_refactorization_period,
            ),
            (
                "devex_weights_reset_period",
                self.devex_weights_reset_period,
            ),
            ("num_omp_threads", self.num_omp_threads),
            ("random_seed", self.random_seed),
        ] {
            if value < 0 {
                return Err(format!("Parameters '{name}' must be non-negative"));
            }
        }
        if self.markowitz_zlatev_parameter < 1 {
            return Err("markowitz_zlatev_parameter must be >= 1".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::GlopParameters;
    #[test]
    fn defaults_validate() {
        assert_eq!(GlopParameters::default().validate(), Ok(()));
    }
    #[test]
    fn diagnostics_match_upstream() {
        let mut p = GlopParameters {
            lu_factorization_pivot_threshold: f64::NAN,
            ..GlopParameters::default()
        };
        assert_eq!(
            p.validate(),
            Err("parameter 'lu_factorization_pivot_threshold' is NaN or not finite".into())
        );
        p.lu_factorization_pivot_threshold = 0.0;
        p.markowitz_zlatev_parameter = 0;
        assert_eq!(
            p.validate(),
            Err("markowitz_zlatev_parameter must be >= 1".into())
        );
    }
}
