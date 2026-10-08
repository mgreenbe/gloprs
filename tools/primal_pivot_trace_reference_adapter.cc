// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Controlled no-presolve primal-simplex traces from pinned GLOP.

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
}  // namespace

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries;
  int iterations;
  if (!(std::cin >> rows >> columns >> entries >> iterations)) return 2;
  LinearProgram lp;
  for (int column = 0; column < columns; ++column) lp.CreateNewVariable();
  for (int row = 0; row < rows; ++row) lp.CreateNewConstraint();
  for (int entry = 0; entry < entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    lp.SetCoefficient(RowIndex(row), ColIndex(column), value);
  }
  for (int column = 0; column < columns; ++column) {
    const double objective = ReadDouble();
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    lp.SetObjectiveCoefficient(ColIndex(column), objective);
    lp.SetVariableBounds(ColIndex(column), lower, upper);
  }
  for (int row = 0; row < rows; ++row) {
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    lp.SetConstraintBounds(RowIndex(row), lower, upper);
  }
  GlopParameters parameters;
  parameters.set_use_scaling(false);
  parameters.set_initial_basis(GlopParameters::NONE);
  parameters.set_exploit_singleton_column_in_initial_basis(false);
  parameters.set_use_dual_simplex(false);
  parameters.set_feasibility_rule(GlopParameters::DANTZIG);
  parameters.set_optimization_rule(GlopParameters::DANTZIG);
  parameters.set_max_number_of_iterations(iterations);
  RevisedSimplex simplex;
  simplex.SetParameters(parameters);
  operations_research::TimeLimit limit;
  if (!simplex.Solve(lp, &limit).ok()) return 3;
  std::cout << std::setprecision(17) << "iterations "
            << simplex.GetNumberOfIterations() << "\nbasis";
  for (int row = 0; row < rows; ++row) {
    std::cout << ' ' << simplex.GetBasis(RowIndex(row)).value();
  }
  std::cout << "\nvalues";
  for (int column = 0; column < columns + rows; ++column) {
    std::cout << ' ' << simplex.GetVariableValue(ColIndex(column));
  }
  std::cout << "\nreduced";
  for (int column = 0; column < columns + rows; ++column) {
    std::cout << ' ' << simplex.GetReducedCost(ColIndex(column));
  }
  std::cout << '\n';
  return 0;
}
