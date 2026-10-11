// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Native full presolve solve and isolated capped shift/scale states.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "ortools/glop/preprocessor.h"
#include "ortools/glop/revised_simplex.h"
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
  std::string initial_basis;
  int rows, columns, entries, singleton_shortcut, scaling_method, iteration_limit;
  if (!(std::cin >> mode >> initial_basis >> rows >> columns >> entries >> singleton_shortcut >> scaling_method >> iteration_limit)) return 2;
  glop::LinearProgram original;
  for (int col = 0; col < columns; ++col) original.CreateNewVariable();
  for (int row = 0; row < rows; ++row) original.CreateNewConstraint();
  for (int entry = 0; entry < entries; ++entry) {
    int row, col;
    std::cin >> row >> col;
    original.SetCoefficient(glop::RowIndex(row), glop::ColIndex(col), ReadDouble());
  }
  for (int col = 0; col < columns; ++col) {
    const double objective = ReadDouble();
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    original.SetObjectiveCoefficient(glop::ColIndex(col), objective);
    original.SetVariableBounds(glop::ColIndex(col), lower, upper);
  }
  for (int row = 0; row < rows; ++row) {
    const double lower = ReadDouble();
    const double upper = ReadDouble();
    original.SetConstraintBounds(glop::RowIndex(row), lower, upper);
  }
  original.CleanUp();
  glop::LinearProgram work;
  work.PopulateFromLinearProgram(original);
  glop::GlopParameters parameters;
  parameters.set_initial_basis(initial_basis == "triangular"
                                   ? glop::GlopParameters::TRIANGULAR
                                   : glop::GlopParameters::NONE);
  parameters.set_exploit_singleton_column_in_initial_basis(singleton_shortcut != 0);
  parameters.set_scaling_method(scaling_method == 1
                                    ? glop::GlopParameters::LINEAR_PROGRAM
                                    : glop::GlopParameters::EQUILIBRATION);
  parameters.set_use_preprocessing(mode == "shift" || mode == "both");
  parameters.set_use_scaling(mode == "scale" || mode == "both");
  if (iteration_limit >= 0) parameters.set_max_number_of_iterations(iteration_limit);
  work.RemoveNearZeroEntries(parameters.drop_magnitude());
  if (iteration_limit < 0) {
    glop::MainLpPreprocessor main_preprocessor(&parameters);
    auto limit = operations_research::TimeLimit::Infinite();
    main_preprocessor.SetTimeLimit(limit.get());
    const bool postsolve_needed = main_preprocessor.Run(&work);
    glop::ProblemSolution solution(work.num_constraints(), work.num_variables());
    solution.status = main_preprocessor.status();
    glop::RevisedSimplex simplex;
    simplex.SetParameters(parameters);
    if (solution.status == glop::ProblemStatus::INIT && !limit->LimitReached()) {
      if (!simplex.Solve(work, limit.get()).ok()) return 3;
      solution.status = simplex.GetProblemStatus();
      for (int col = 0; col < work.num_variables().value(); ++col) {
        const glop::ColIndex index(col);
        solution.primal_values[index] = simplex.GetVariableValue(index);
        solution.variable_statuses[index] = simplex.GetVariableStatus(index);
      }
      for (int row = 0; row < work.num_constraints().value(); ++row) {
        const glop::RowIndex index(row);
        solution.dual_values[index] = simplex.GetDualValue(index);
        solution.constraint_statuses[index] = simplex.GetConstraintStatus(index);
      }
    }
    if (postsolve_needed) main_preprocessor.DestructiveRecoverSolution(&solution);
    std::cout << "status " << glop::GetProblemStatusString(solution.status) << '\n';
    std::cout << "iterations " << simplex.GetNumberOfIterations() << '\n';
    std::cout << "basis";
    for (const auto col : simplex.GetBasisVector()) std::cout << ' ' << col.value();
    std::cout << '\n';
    PrintBits("primal", solution.primal_values);
    PrintBits("dual", solution.dual_values);
    std::cout << "variable_status";
    for (const auto status : solution.variable_statuses)
      std::cout << ' ' << glop::GetVariableStatusString(status);
    std::cout << "\nconstraint_status";
    for (const auto status : solution.constraint_statuses)
      std::cout << ' ' << glop::GetConstraintStatusString(status);
    std::cout << '\n';
    return 0;
  }
  glop::ShiftVariableBoundsPreprocessor shift(&parameters);
  glop::ScalingPreprocessor scaling(&parameters);
  const bool shifted = parameters.use_preprocessing() && shift.Run(&work);
  const bool scaled = parameters.use_scaling() && scaling.Run(&work);
  glop::RevisedSimplex simplex;
  simplex.SetParameters(parameters);
  auto limit = operations_research::TimeLimit::Infinite();
  if (!simplex.Solve(work, limit.get()).ok()) return 3;
  glop::ProblemSolution solution{glop::RowIndex(rows), glop::ColIndex(columns)};
  solution.status = simplex.GetProblemStatus();
  for (int col = 0; col < columns; ++col) {
    const glop::ColIndex index(col);
    solution.primal_values[index] = simplex.GetVariableValue(index);
    solution.variable_statuses[index] = simplex.GetVariableStatus(index);
  }
  for (int row = 0; row < rows; ++row) {
    const glop::RowIndex index(row);
    solution.dual_values[index] = simplex.GetDualValue(index);
    solution.constraint_statuses[index] = simplex.GetConstraintStatus(index);
  }
  if (scaled) scaling.RecoverSolution(&solution);
  if (shifted) shift.RecoverSolution(&solution);

  std::cout << "status " << glop::GetProblemStatusString(solution.status) << '\n';
  std::cout << "iterations " << simplex.GetNumberOfIterations() << '\n';
  std::cout << "basis";
  for (const auto col : simplex.GetBasisVector()) std::cout << ' ' << col.value();
  std::cout << '\n';
  PrintBits("primal", solution.primal_values);
  PrintBits("dual", solution.dual_values);
  std::cout << "variable_status";
  for (const auto status : solution.variable_statuses)
    std::cout << ' ' << glop::GetVariableStatusString(status);
  std::cout << "\nconstraint_status";
  for (const auto status : solution.constraint_statuses)
    std::cout << ' ' << glop::GetConstraintStatusString(status);
  std::cout << '\n';
}
