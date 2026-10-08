// Copyright 2026 gloprs contributors
#include <iostream>
#include <string>

#include "ortools/glop/parameters.pb.h"
#include "ortools/glop/parameters_validation.h"

int main() {
  using operations_research::glop::GlopParameters;
  using operations_research::glop::ValidateParameters;
  std::string field;
  std::string encoded_value;
  if (!(std::cin >> field >> encoded_value)) return 2;
  const double value = std::stod(encoded_value);
  GlopParameters p;
#define SET_DOUBLE(name) if (field == #name) p.set_##name(value); else
  SET_DOUBLE(degenerate_ministep_factor)
  SET_DOUBLE(drop_tolerance)
  SET_DOUBLE(dual_feasibility_tolerance)
  SET_DOUBLE(dual_small_pivot_threshold)
  SET_DOUBLE(dualizer_threshold)
  SET_DOUBLE(harris_tolerance_ratio)
  SET_DOUBLE(lu_factorization_pivot_threshold)
  SET_DOUBLE(markowitz_singularity_threshold)
  SET_DOUBLE(max_number_of_reoptimizations)
  SET_DOUBLE(minimum_acceptable_pivot)
  SET_DOUBLE(preprocessor_zero_tolerance)
  SET_DOUBLE(primal_feasibility_tolerance)
  SET_DOUBLE(ratio_test_zero_threshold)
  SET_DOUBLE(recompute_edges_norm_threshold)
  SET_DOUBLE(recompute_reduced_costs_threshold)
  SET_DOUBLE(refactorization_threshold)
  SET_DOUBLE(relative_cost_perturbation)
  SET_DOUBLE(relative_max_cost_perturbation)
  SET_DOUBLE(small_pivot_threshold)
  SET_DOUBLE(solution_feasibility_tolerance)
  SET_DOUBLE(objective_lower_limit)
  SET_DOUBLE(objective_upper_limit)
  SET_DOUBLE(crossover_bound_snapping_distance)
  SET_DOUBLE(initial_condition_number_threshold)
  SET_DOUBLE(max_deterministic_time)
  SET_DOUBLE(max_time_in_seconds)
  SET_DOUBLE(max_valid_magnitude)
  SET_DOUBLE(drop_magnitude)
#undef SET_DOUBLE
  if (field == "basis_refactorization_period") p.set_basis_refactorization_period(value);
  else if (field == "devex_weights_reset_period") p.set_devex_weights_reset_period(value);
  else if (field == "num_omp_threads") p.set_num_omp_threads(value);
  else if (field == "random_seed") p.set_random_seed(value);
  else if (field == "markowitz_zlatev_parameter") p.set_markowitz_zlatev_parameter(value);
  else if (field != "drop_magnitude") return 3;
  const std::string error = ValidateParameters(p);
  std::cout << (error.empty() ? "OK" : error) << '\n';
}
