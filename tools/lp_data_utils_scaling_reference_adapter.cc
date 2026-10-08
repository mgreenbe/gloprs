// Copyright 2026 Matthew Greenberg
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "ortools/lp_data/lp_data_utils.h"

template <typename Vector>
void Print(const char* name, const Vector& values) {
  std::cout << name;
  for (double value : values) std::cout << ' ' << value;
  std::cout << '\n';
}

double ReadDouble() {
  std::string token;
  std::cin >> token;
  if (token == "inf") return std::numeric_limits<double>::infinity();
  if (token == "-inf") return -std::numeric_limits<double>::infinity();
  return std::stod(token);
}

int main() {
  using namespace operations_research::glop;
  int n;
  if (!(std::cin >> n)) return 2;
  std::vector<double> row_factors(n), col_factors(n);
  for (double& value : row_factors) std::cin >> value;
  for (double& value : col_factors) std::cin >> value;
  DenseRow objective(ColIndex(n), 0.0), lower(ColIndex(n), 0.0),
      upper(ColIndex(n), 0.0);
  for (double& value : objective) std::cin >> value;
  for (double& value : lower) value = ReadDouble();
  for (double& value : upper) value = ReadDouble();
  DenseColumn solve_values(RowIndex(n), 0.0);
  for (double& value : solve_values) std::cin >> value;
  int pattern_size;
  std::cin >> pattern_size;
  std::vector<int> pattern(pattern_size);
  for (int& value : pattern) std::cin >> value;
  RowToColMapping basis(RowIndex(n), ColIndex(0));
  for (ColIndex& value : basis) {
    int input;
    std::cin >> input;
    value = ColIndex(input);
  }
  int selected;
  std::cin >> selected;

  LpScalingHelper helper;
  helper.ConfigureFromFactors(row_factors, col_factors);
  std::cout << std::setprecision(17);
  std::cout << "scalar";
  for (int i = 0; i < n; ++i) {
    const double value = solve_values[RowIndex(i)];
    std::cout << ' ' << helper.ScaleVariableValue(ColIndex(i), value)
              << ' ' << helper.UnscaleVariableValue(ColIndex(i), value)
              << ' ' << helper.ScaleReducedCost(ColIndex(i), value)
              << ' ' << helper.UnscaleReducedCost(ColIndex(i), value)
              << ' ' << helper.ScaleDualValue(RowIndex(i), value)
              << ' ' << helper.UnscaleDualValue(RowIndex(i), value)
              << ' ' << helper.ScaleConstraintActivity(RowIndex(i), value)
              << ' ' << helper.UnscaleConstraintActivity(RowIndex(i), value)
              << ' ' << helper.UnscaleLeftSolveValue(RowIndex(i), value)
              << ' ' << helper.VariableScalingFactor(ColIndex(i))
              << ' '
              << helper.VariableScalingFactorWithSlack(ColIndex(n + i));
  }
  std::cout << '\n';

  for (const bool sparse : {false, true}) {
    ScatteredRow left;
    left.values.resize(ColIndex(n), 0.0);
    for (int i = 0; i < n; ++i) {
      left.values[ColIndex(i)] = solve_values[RowIndex(i)];
    }
    left.is_non_zero.Resize(ColIndex(n));
    if (sparse) {
      for (int position : pattern) left.non_zeros.push_back(ColIndex(position));
    }
    helper.UnscaleUnitRowLeftSolve(ColIndex(selected), &left);
    Print(sparse ? "left_sparse" : "left_dense", left.values);
  }

  for (const bool sparse : {false, true}) {
    ScatteredColumn right;
    right.values = solve_values;
    right.is_non_zero.Resize(RowIndex(n));
    if (sparse) {
      for (int position : pattern) right.non_zeros.push_back(RowIndex(position));
    }
    helper.UnscaleColumnRightSolve(basis, ColIndex(selected), &right);
    Print(sparse ? "right_sparse" : "right_dense", right.values);
  }

  helper.AverageCostScaling(&objective);
  Print("objective", objective);
  std::cout << "objective_factor " << helper.ObjectiveScalingFactor() << '\n';
  helper.ContainOneBoundScaling(&upper, &lower);
  Print("lower", lower);
  Print("upper", upper);
  std::cout << "bound_factor " << helper.BoundsScalingFactor() << '\n';
  helper.Clear();
  std::cout << "cleared " << helper.BoundsScalingFactor() << ' '
            << helper.ObjectiveScalingFactor() << ' '
            << helper.VariableScalingFactor(ColIndex(selected)) << ' '
            << helper.VariableScalingFactorWithSlack(ColIndex(n + selected))
            << ' ' << helper.ScaleVariableValue(ColIndex(selected), 3.0) << ' '
            << helper.ScaleReducedCost(ColIndex(selected), 3.0) << ' '
            << helper.ScaleDualValue(RowIndex(selected), 3.0) << ' '
            << helper.ScaleConstraintActivity(RowIndex(selected), 3.0) << '\n';
}
