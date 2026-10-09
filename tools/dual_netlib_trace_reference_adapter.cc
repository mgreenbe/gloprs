// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Iteration-prefix snapshots from pinned GLOP's dual revised simplex.

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "ortools/glop/revised_simplex.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"
#include "ortools/util/time_limit.h"

int main(int argc, char** argv) {
  namespace glop = operations_research::glop;
  if (argc != 3) return EXIT_FAILURE;
  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) return EXIT_FAILURE;
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);

  glop::GlopParameters parameters;
  parameters.set_use_scaling(false);
  parameters.set_use_dual_simplex(true);
  parameters.set_max_number_of_iterations(std::stoll(argv[2]));
  glop::RevisedSimplex simplex;
  simplex.SetParameters(parameters);
  operations_research::TimeLimit limit;
  const auto status = simplex.Solve(lp, &limit);
  if (!status.ok()) {
    std::cout << "error " << status.error_message() << '\n';
    return EXIT_SUCCESS;
  }
  // RevisedSimplex adds these same slack columns to its internal copy. Add
  // them to the adapter's LP only after Solve(), so the diagnostic coordinate
  // system matches without changing GLOP's initialization path.
  lp.AddSlackVariablesWhereNecessary(false);

  std::cout << std::setprecision(17) << "iterations "
            << simplex.GetNumberOfIterations() << "\nstatus "
            << glop::GetProblemStatusString(simplex.GetProblemStatus())
            << "\nupdates "
            << simplex.GetBasisFactorization().NumUpdates()
            << "\nbasis";
  for (int row = 0; row < lp.num_constraints().value(); ++row) {
    std::cout << ' ' << simplex.GetBasis(glop::RowIndex(row)).value();
  }
  std::cout << "\nvalues";
  const int total_columns = lp.num_variables().value();
  for (int column = 0; column < total_columns; ++column) {
    std::cout << ' ' << simplex.GetVariableValue(glop::ColIndex(column));
  }
  std::cout << "\nvalue_bits";
  for (int column = 0; column < total_columns; ++column) {
    const double value = simplex.GetVariableValue(glop::ColIndex(column));
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    std::cout << ' ' << bits;
  }
  std::cout << "\nreduced";
  for (int column = 0; column < total_columns; ++column) {
    std::cout << ' ' << simplex.GetReducedCost(glop::ColIndex(column));
  }
  std::cout << "\nreduced_bits";
  for (int column = 0; column < total_columns; ++column) {
    const double value = simplex.GetReducedCost(glop::ColIndex(column));
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    std::cout << ' ' << bits;
  }
  glop::ScatteredColumn phase_one_rhs;
  phase_one_rhs.values.AssignToZero(lp.num_constraints());
  const double tolerance = parameters.dual_feasibility_tolerance();
  for (int column = 0; column < total_columns; ++column) {
    const glop::ColIndex col(column);
    const auto status = simplex.GetVariableStatus(col);
    const double lower = lp.variable_lower_bounds()[col];
    const double upper = lp.variable_upper_bounds()[col];
    const bool boxed = std::isfinite(lower) && std::isfinite(upper) && lower != upper;
    const bool fixed = lower == upper;
    const bool can_increase = status == glop::VariableStatus::AT_LOWER_BOUND ||
                              status == glop::VariableStatus::FREE;
    const bool can_decrease = status == glop::VariableStatus::AT_UPPER_BOUND ||
                              status == glop::VariableStatus::FREE;
    const double reduced = simplex.GetReducedCost(col);
    const double sign = boxed || fixed ? 0.0
        : can_increase && reduced < -tolerance ? 1.0
        : can_decrease && reduced > tolerance ? -1.0
                                                : 0.0;
    if (sign != 0.0) {
      for (const auto entry : lp.GetSparseColumn(col)) {
        phase_one_rhs[entry.row()] += sign * entry.coefficient();
      }
    }
  }
  simplex.GetBasisFactorization().RightSolve(&phase_one_rhs);
  const auto norms = simplex.GetDualSquaredNorms();
  std::cout << "\nnorms";
  for (const double norm : norms) std::cout << ' ' << norm;
  std::cout << "\nnorm_bits";
  for (const double norm : norms) {
    uint64_t bits;
    std::memcpy(&bits, &norm, sizeof(bits));
    std::cout << ' ' << bits;
  }
  std::cout << '\n';
  std::vector<std::tuple<double, int, int, double, double>> prices;
  for (int row = 0; row < lp.num_constraints().value(); ++row) {
    const glop::RowIndex row_index(row);
    const glop::ColIndex basic = simplex.GetBasis(row_index);
    const double lower = lp.variable_lower_bounds()[basic];
    const double upper = lp.variable_upper_bounds()[basic];
    const double price = phase_one_rhs[row_index];
    const bool candidate = price != 0.0 &&
        ((std::isfinite(lower) && std::isfinite(upper)) ||
         (!std::isfinite(lower) && price < -parameters.ratio_test_zero_threshold()) ||
         (!std::isfinite(upper) && price > parameters.ratio_test_zero_threshold()));
    if (candidate) prices.emplace_back(price * price / norms[row_index], basic.value(), row,
                                       price, norms[row_index]);
  }
  std::sort(prices.begin(), prices.end(), std::greater<>());
  std::cout << "\nphase1_top";
  for (int i = 0; i < std::min<int>(10, prices.size()); ++i) {
    const auto [scaled, basic, row, price, norm] = prices[i];
    std::cout << ' ' << basic << ':' << row << ':' << price << ':' << norm << ':' << scaled;
  }
  std::cout << '\n';
  return EXIT_SUCCESS;
}
