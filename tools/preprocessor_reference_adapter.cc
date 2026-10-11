// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Pinned-native bound-shift/scaling transformation and postsolve trace.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "ortools/glop/preprocessor.h"

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

template <typename Vector>
void PrintBits(const char* label, const Vector& values) {
  std::cout << label;
  for (double value : values) std::cout << ' ' << Bits(value);
  std::cout << '\n';
}
}  // namespace

int main() {
  namespace glop = operations_research::glop;
  std::string mode;
  int rows, columns, entries, cost_scaling, scaling_method;
  if (!(std::cin >> mode >> rows >> columns >> entries >> cost_scaling >> scaling_method)) return 2;
  glop::LinearProgram lp;
  for (int col = 0; col < columns; ++col) lp.CreateNewVariable();
  for (int row = 0; row < rows; ++row) lp.CreateNewConstraint();
  for (int index = 0; index < entries; ++index) {
    int row, col;
    std::cin >> row >> col;
    lp.SetCoefficient(glop::RowIndex(row), glop::ColIndex(col), ReadDouble());
  }
  for (int col = 0; col < columns; ++col) {
    const double objective = ReadDouble();
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    lp.SetObjectiveCoefficient(glop::ColIndex(col), objective);
    lp.SetVariableBounds(glop::ColIndex(col), lower, upper);
  }
  for (int row = 0; row < rows; ++row) {
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    lp.SetConstraintBounds(glop::RowIndex(row), lower, upper);
  }
  glop::ProblemSolution solution{glop::RowIndex(rows), glop::ColIndex(columns)};
  for (int col = 0; col < columns; ++col) {
    std::string status;
    std::cin >> status;
    const glop::ColIndex index(col);
    if (status == "basic") solution.variable_statuses[index] = glop::VariableStatus::BASIC;
    if (status == "lower") solution.variable_statuses[index] = glop::VariableStatus::AT_LOWER_BOUND;
    if (status == "upper") solution.variable_statuses[index] = glop::VariableStatus::AT_UPPER_BOUND;
    if (status == "fixed") solution.variable_statuses[index] = glop::VariableStatus::FIXED_VALUE;
    solution.primal_values[index] = ReadDouble();
  }
  for (int row = 0; row < rows; ++row) {
    solution.dual_values[glop::RowIndex(row)] = ReadDouble();
  }
  lp.CleanUp();
  glop::GlopParameters parameters;
  parameters.set_cost_scaling(
      static_cast<glop::GlopParameters::CostScalingAlgorithm>(cost_scaling));
  parameters.set_scaling_method(scaling_method == 1
                                    ? glop::GlopParameters::LINEAR_PROGRAM
                                    : glop::GlopParameters::EQUILIBRATION);
  glop::ShiftVariableBoundsPreprocessor shift(&parameters);
  glop::ScalingPreprocessor scaling(&parameters);
  const bool do_shift = mode == "shift" || mode == "both";
  const bool do_scale = mode == "scale" || mode == "both";
  const bool shifted = do_shift && shift.Run(&lp);
  const bool invalid = do_shift && shift.status() == glop::ProblemStatus::INVALID_PROBLEM;
  const bool scaled = do_scale && !invalid && scaling.Run(&lp);
  std::cout << "flags " << shifted << ' ' << scaled << ' ' << invalid << '\n';
  PrintBits("objective", lp.objective_coefficients());
  std::cout << "objective_offset " << Bits(lp.objective_offset()) << '\n';
  PrintBits("variable_lower", lp.variable_lower_bounds());
  PrintBits("variable_upper", lp.variable_upper_bounds());
  PrintBits("constraint_lower", lp.constraint_lower_bounds());
  PrintBits("constraint_upper", lp.constraint_upper_bounds());
  std::cout << "matrix";
  for (int col = 0; col < columns; ++col) {
    for (const auto entry : lp.GetSparseColumn(glop::ColIndex(col))) {
      std::cout << ' ' << col << ' ' << entry.row().value() << ' '
                << Bits(entry.coefficient());
    }
  }
  std::cout << '\n';
  if (shifted) PrintBits("offsets", shift.offsets());
  if (scaled) scaling.RecoverSolution(&solution);
  if (shifted) shift.RecoverSolution(&solution);
  PrintBits("recovered_primal", solution.primal_values);
  PrintBits("recovered_dual", solution.dual_values);
}
