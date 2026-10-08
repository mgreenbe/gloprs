// Copyright 2026 Matthew Greenberg
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/lp_data_utils.h"
#include "ortools/glop/parameters.pb.h"

template <typename Vector>
void Print(const char* name, const Vector& values) {
  std::cout << name;
  for (double value : values) std::cout << ' ' << value;
  std::cout << '\n';
}

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries;
  if (!(std::cin >> rows >> columns >> entries)) return 2;
  LinearProgram lp;
  for (int col = 0; col < columns; ++col) lp.CreateNewVariable();
  for (int row = 0; row < rows; ++row) lp.CreateNewConstraint();
  for (int i = 0; i < entries; ++i) {
    int row;
    int col;
    double value;
    std::cin >> row >> col >> value;
    lp.SetCoefficient(RowIndex(row), ColIndex(col), value);
  }
  for (int col = 0; col < columns; ++col) {
    double objective;
    double lower;
    double upper;
    std::cin >> objective >> lower >> upper;
    lp.SetObjectiveCoefficient(ColIndex(col), objective);
    lp.SetVariableBounds(ColIndex(col), lower, upper);
  }
  for (int row = 0; row < rows; ++row) {
    double lower;
    double upper;
    std::cin >> lower >> upper;
    lp.SetConstraintBounds(RowIndex(row), lower, upper);
  }
  int cost_scaling;
  std::cin >> cost_scaling;
  lp.CleanUp();
  LpScalingHelper helper;
  GlopParameters params;
  params.set_cost_scaling(
      static_cast<GlopParameters::CostScalingAlgorithm>(cost_scaling));
  helper.Scale(params, &lp);
  std::cout << std::setprecision(17);
  Print("objective", lp.objective_coefficients());
  Print("variable_lower", lp.variable_lower_bounds());
  Print("variable_upper", lp.variable_upper_bounds());
  Print("constraint_lower", lp.constraint_lower_bounds());
  Print("constraint_upper", lp.constraint_upper_bounds());
  std::cout << "matrix";
  for (ColIndex col(0); col < lp.num_variables(); ++col) {
    for (const auto entry : lp.GetSparseColumn(col)) {
      std::cout << ' ' << col.value() << ' ' << entry.row().value() << ' '
                << entry.coefficient();
    }
  }
  std::cout << "\nfactors " << helper.BoundsScalingFactor() << ' '
            << helper.ObjectiveScalingFactor() << '\n';
}
