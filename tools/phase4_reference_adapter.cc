// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Small no-presolve LP outcomes from the pinned native RevisedSimplex.

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "ortools/glop/revised_simplex.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/util/time_limit.h"

namespace {
double ReadDouble() {
  std::string token;
  std::cin >> token;
  if (token == "inf") return std::numeric_limits<double>::infinity();
  if (token == "-inf") return -std::numeric_limits<double>::infinity();
  return std::stod(token);
}

uint64_t Bits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
}  // namespace

int main() {
  namespace glop = operations_research::glop;
  std::string mode;
  int rows, columns, entries, iterations;
  if (!(std::cin >> mode >> rows >> columns >> entries >> iterations)) return 2;
  const double deterministic_limit = ReadDouble();
  glop::LinearProgram lp;
  for (int column = 0; column < columns; ++column) lp.CreateNewVariable();
  for (int row = 0; row < rows; ++row) lp.CreateNewConstraint();
  for (int entry = 0; entry < entries; ++entry) {
    int row, column;
    std::cin >> row >> column;
    const double value = ReadDouble();
    lp.SetCoefficient(glop::RowIndex(row), glop::ColIndex(column), value);
  }
  for (int column = 0; column < columns; ++column) {
    const double objective = ReadDouble();
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    lp.SetObjectiveCoefficient(glop::ColIndex(column), objective);
    lp.SetVariableBounds(glop::ColIndex(column), lower, upper);
  }
  for (int row = 0; row < rows; ++row) {
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    lp.SetConstraintBounds(glop::RowIndex(row), lower, upper);
  }
  glop::GlopParameters parameters;
  parameters.set_use_scaling(false);
  parameters.set_initial_basis(glop::GlopParameters::NONE);
  parameters.set_exploit_singleton_column_in_initial_basis(false);
  parameters.set_use_dual_simplex(mode != "primal" && mode != "primal_limit" &&
                                   mode != "primal_no_imprecise" &&
                                   mode != "primal_time_zero" &&
                                   mode != "tight_internal_primal" &&
                                   mode != "relaxed_internal_primal" &&
                                   mode != "warm_primal" && mode != "warm_bound_change" &&
                                   mode != "warm_auto_primal_repeat" &&
                                   mode != "warm_clear_state" &&
                                   mode != "warm_external_then_restore" &&
                                   mode != "warm_auto_primal_objective_change" &&
                                   mode != "warm_primal_limit_change" &&
                                   mode != "warm_objective_change" &&
                                   mode != "warm_multiple_bound_changes" &&
                                   mode != "external_basis" &&
                                   mode != "warm_added_column" &&
                                   mode != "starting_values" &&
                                   mode != "starting_values_push" &&
                                   mode != "starting_values_push_boxed" &&
                                   mode != "starting_values_push_boxed_no_snap" &&
                                   mode != "starting_values_push_boxed_upper_no_snap" &&
                                   mode != "starting_values_push_boxed_snap_upper_boundary" &&
                                   mode != "starting_values_push_with_row" &&
                                   mode != "starting_values_push_refactorize" &&
                                   mode != "starting_values_push_control" &&
                                   mode != "starting_values_two_no_push" &&
                                   mode != "zero_tolerance" && mode != "zero_tolerance_no_imprecise");
  if (mode == "tight_tolerance" || mode == "loose_tolerance")
    parameters.set_use_dual_simplex(false);
  if (mode == "precision_refactorization") {
    parameters.set_use_dual_simplex(false);
    parameters.set_recompute_reduced_costs_threshold(0.0);
  }
  if (mode == "adaptive_pivot") {
    parameters.set_use_dual_simplex(false);
    parameters.set_refactorization_threshold(0.0);
  }
  parameters.set_perturb_costs_in_dual_simplex(mode == "dual_perturbed");
  parameters.set_use_dedicated_dual_feasibility_algorithm(mode != "dual_transformed");
  parameters.set_feasibility_rule(glop::GlopParameters::DANTZIG);
  parameters.set_optimization_rule(glop::GlopParameters::DANTZIG);
  parameters.set_max_number_of_iterations(iterations);
  if (mode == "starting_values_push_refactorize")
    parameters.set_small_pivot_threshold(0.1);
  if (mode == "starting_values_push_boxed_no_snap" ||
      mode == "starting_values_push_boxed_upper_no_snap")
    parameters.set_crossover_bound_snapping_distance(0.0);
  if (mode == "starting_values_push_boxed_snap_upper_boundary")
    parameters.set_crossover_bound_snapping_distance(0.25);
  if (mode == "primal_limit") parameters.set_objective_lower_limit(-0.5);
  if (mode == "dual_limit") parameters.set_objective_upper_limit(0.5);
  if (mode == "zero_tolerance" || mode == "zero_tolerance_no_imprecise") {
    parameters.set_solution_feasibility_tolerance(0.0);
    if (mode == "zero_tolerance_no_imprecise")
      parameters.set_change_status_to_imprecise(false);
  }
  if (mode == "tight_tolerance")
    parameters.set_solution_feasibility_tolerance(1e-18);
  if (mode == "loose_tolerance")
    parameters.set_solution_feasibility_tolerance(1e-16);
  if (mode == "primal_no_imprecise" || mode == "dual_no_imprecise")
    parameters.set_change_status_to_imprecise(false);
  if (mode == "tight_internal_dual" || mode == "tight_internal_primal") {
    parameters.set_primal_feasibility_tolerance(1e-16);
    parameters.set_dual_feasibility_tolerance(1e-16);
  }
  if (mode == "relaxed_internal_primal" || mode == "relaxed_internal_dual") {
    parameters.set_primal_feasibility_tolerance(1e-15);
    parameters.set_dual_feasibility_tolerance(1e-15);
  }
  if (mode == "starting_values" || mode == "starting_values_two_no_push")
    parameters.set_push_to_vertex(false);
  glop::RevisedSimplex simplex;
  simplex.SetParameters(parameters);
  if (mode == "external_basis") {
    glop::BasisState state;
    state.statuses.resize(glop::ColIndex(columns + rows), glop::VariableStatus::AT_LOWER_BOUND);
    state.statuses[glop::ColIndex(0)] = glop::VariableStatus::BASIC;
    state.statuses[glop::ColIndex(columns)] = glop::VariableStatus::AT_UPPER_BOUND;
    simplex.LoadStateForNextSolve(state);
  }
  if (mode == "starting_values_push_boxed" ||
      mode == "starting_values_push_boxed_no_snap" ||
      mode == "starting_values_push_boxed_upper_no_snap" ||
      mode == "starting_values_push_boxed_snap_upper_boundary") {
    glop::BasisState state;
    state.statuses.resize(glop::ColIndex(columns), glop::VariableStatus::BASIC);
    simplex.LoadStateForNextSolve(state);
  }
  if (mode == "starting_values" || mode == "starting_values_push" ||
      mode == "starting_values_push_boxed" ||
      mode == "starting_values_push_boxed_no_snap" ||
      mode == "starting_values_push_boxed_upper_no_snap" ||
      mode == "starting_values_push_boxed_snap_upper_boundary" ||
      mode == "starting_values_push_with_row" ||
      mode == "starting_values_push_refactorize" ||
      mode == "starting_values_push_control" ||
      mode == "starting_values_two_no_push") {
    glop::DenseRow values;
    values.resize(glop::ColIndex(columns), 0.0);
    values[glop::ColIndex(0)] =
        mode == "starting_values_push_boxed_upper_no_snap" ||
                mode == "starting_values_push_boxed_snap_upper_boundary"
            ? 0.75
            : 0.5;
    if (mode == "starting_values_push_refactorize" ||
        mode == "starting_values_push_control" ||
        mode == "starting_values_two_no_push")
      values[glop::ColIndex(1)] = 0.5;
    simplex.SetStartingVariableValuesForNextSolve(values);
  }
  operations_research::TimeLimit limit(
      std::numeric_limits<double>::infinity(),
      (mode == "primal_time_zero" || mode == "dual_time_zero")
          ? 0.0
      : deterministic_limit);
  if (mode == "warm_primal" || mode == "warm_bound_change" ||
      mode == "warm_auto_primal_repeat" ||
      mode == "warm_clear_state" ||
      mode == "warm_external_then_restore" ||
      mode == "warm_auto_primal_objective_change" ||
      mode == "warm_auto_dual_repeat" ||
      mode == "warm_auto_dual_bound_change" ||
      mode == "warm_primal_limit_change" ||
      mode == "warm_dual_bound_change" ||
      mode == "warm_dual_repeat" ||
      mode == "warm_dual_limit_change" ||
      mode == "warm_objective_change" || mode == "warm_multiple_bound_changes" ||
      mode == "warm_dual_multiple_bound_changes" ||
      mode == "warm_added_column" || mode == "warm_added_row" ||
      mode == "warm_added_row_low_condition_threshold" ||
      mode == "warm_added_row_slack") {
    const auto first = simplex.Solve(lp, &limit);
    if (!first.ok()) return 4;
    if (mode == "warm_clear_state")
      simplex.ClearStateForNextSolve();
    else if (mode == "warm_external_then_restore") {
      const auto saved = simplex.GetState();
      glop::BasisState other;
      other.statuses.resize(glop::ColIndex(columns + rows),
                            glop::VariableStatus::AT_LOWER_BOUND);
      other.statuses[glop::ColIndex(columns)] = glop::VariableStatus::BASIC;
      simplex.LoadStateForNextSolve(other);
      simplex.LoadStateForNextSolve(saved);
    }
    else if (mode.rfind("warm_auto_", 0) != 0)
      simplex.LoadStateForNextSolve(simplex.GetState());
    if (mode == "warm_primal_limit_change") {
      parameters.set_objective_lower_limit(-0.5);
      simplex.SetParameters(parameters);
    }
    if (mode == "warm_dual_limit_change") {
      parameters.set_objective_upper_limit(0.5);
      simplex.SetParameters(parameters);
    }
    if (mode == "warm_added_row_low_condition_threshold") {
      parameters.set_initial_condition_number_threshold(1.0);
      simplex.SetParameters(parameters);
    }
    if (mode == "warm_bound_change" || mode == "warm_dual_bound_change" ||
        mode == "warm_auto_dual_bound_change") {
      lp.SetConstraintBounds(glop::RowIndex(0), -std::numeric_limits<double>::infinity(), 0.5);
    }
    if (mode == "warm_objective_change" ||
        mode == "warm_auto_primal_objective_change") {
      lp.SetObjectiveCoefficient(glop::ColIndex(0), 1.0);
    }
    if (mode == "warm_multiple_bound_changes" ||
        mode == "warm_dual_multiple_bound_changes") {
      lp.SetConstraintBounds(glop::RowIndex(0), -std::numeric_limits<double>::infinity(), 0.5);
      lp.SetConstraintBounds(glop::RowIndex(1), -std::numeric_limits<double>::infinity(), 0.5);
    }
    if (mode == "warm_added_column") {
      const auto added = lp.CreateNewVariable();
      ++columns;
      lp.SetVariableBounds(added, 0.0, std::numeric_limits<double>::infinity());
      lp.SetObjectiveCoefficient(added, -2.0);
      lp.SetCoefficient(glop::RowIndex(0), added, 1.0);
    }
    if (mode == "warm_added_row" || mode == "warm_added_row_slack" ||
        mode == "warm_added_row_low_condition_threshold") {
      const auto added = lp.CreateNewConstraint();
      ++rows;
      lp.SetConstraintBounds(added, mode == "warm_added_row_slack" ? 1.0 : 2.0,
                             std::numeric_limits<double>::infinity());
      lp.SetCoefficient(added, glop::ColIndex(0), 1.0);
    }
  }
  const auto result = simplex.Solve(lp, &limit);
  if (!result.ok()) {
    std::cout << "error " << result.error_message() << '\n';
    return 0;
  }
  std::cout << "status " << glop::GetProblemStatusString(simplex.GetProblemStatus())
            << "\niterations " << simplex.GetNumberOfIterations()
            << "\nobjective_bits " << Bits(simplex.GetObjectiveValue())
            << "\ndeterministic_time_bits " << Bits(simplex.DeterministicTime())
            << "\nlu_pivot_threshold_bits "
            << Bits(simplex.GetParameters().lu_factorization_pivot_threshold())
            << "\nbasis";
  for (int row = 0; row < rows; ++row) {
    std::cout << ' ' << simplex.GetBasis(glop::RowIndex(row)).value();
  }
  std::cout << "\nvalue_bits";
  for (int column = 0; column < columns + rows; ++column) {
    std::cout << ' ' << Bits(simplex.GetVariableValue(glop::ColIndex(column)));
  }
  std::cout << "\nreduced_bits";
  for (int column = 0; column < columns + rows; ++column) {
    std::cout << ' ' << Bits(simplex.GetReducedCost(glop::ColIndex(column)));
  }
  std::cout << "\nprimal_ray_bits";
  for (double value : simplex.GetPrimalRay()) std::cout << ' ' << Bits(value);
  std::cout << "\ndual_ray_bits";
  for (double value : simplex.GetDualRay()) std::cout << ' ' << Bits(value);
  std::cout << '\n';
  return 0;
}
