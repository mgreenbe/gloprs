// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Pinned-native small presolve transformations and their postsolve maps.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "ortools/glop/preprocessor.h"
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
  int rows, columns, entries;
  if (!(std::cin >> mode >> rows >> columns >> entries)) return 2;
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
  lp.CleanUp();
  glop::GlopParameters parameters;
  if (mode == "forcing_exact") {
    parameters.set_solution_feasibility_tolerance(0.0);
    parameters.set_preprocessor_zero_tolerance(0.0);
  }
  if (mode == "dualizer") {
    parameters.set_solve_dual_problem(glop::GlopParameters::ALWAYS_DO);
  }
  if (mode == "main") parameters.set_use_scaling(false);
  glop::FixedVariablePreprocessor fixed(&parameters);
  glop::FreeConstraintPreprocessor free_row(&parameters);
  glop::EmptyConstraintPreprocessor empty_row(&parameters);
  glop::EmptyColumnPreprocessor empty_column(&parameters);
  glop::SingletonColumnSignPreprocessor singleton_sign(&parameters);
  glop::SingletonPreprocessor singleton(&parameters);
  glop::ForcingAndImpliedFreeConstraintPreprocessor forcing(&parameters);
  glop::ImpliedFreePreprocessor implied_free(&parameters);
  glop::DoubletonFreeColumnPreprocessor doubleton_free(&parameters);
  glop::DoubletonEqualityRowPreprocessor doubleton_equality(&parameters);
  glop::ProportionalRowPreprocessor proportional_row(&parameters);
  glop::ProportionalColumnPreprocessor proportional_column(&parameters);
  glop::DualizerPreprocessor dualizer(&parameters);
  glop::UnconstrainedVariablePreprocessor unconstrained(&parameters);
  glop::MainLpPreprocessor main_preprocessor(&parameters);
  auto main_time_limit = operations_research::TimeLimit::Infinite();
  main_preprocessor.SetTimeLimit(main_time_limit.get());
  bool changed = false;
  if (mode == "fixed") changed = fixed.Run(&lp);
  if (mode == "free_row") changed = free_row.Run(&lp);
  if (mode == "empty_row") changed = empty_row.Run(&lp);
  if (mode == "empty_column") changed = empty_column.Run(&lp);
  if (mode == "singleton_sign") changed = singleton_sign.Run(&lp);
  if (mode == "singleton") changed = singleton.Run(&lp);
  if (mode == "forcing") changed = forcing.Run(&lp);
  if (mode == "forcing_exact") changed = forcing.Run(&lp);
  if (mode == "implied_free") changed = implied_free.Run(&lp);
  if (mode == "doubleton_free") changed = doubleton_free.Run(&lp);
  if (mode == "doubleton_equality") changed = doubleton_equality.Run(&lp);
  if (mode == "proportional_row") changed = proportional_row.Run(&lp);
  if (mode == "proportional_column") changed = proportional_column.Run(&lp);
  if (mode == "dualizer") changed = dualizer.Run(&lp);
  if (mode == "unconstrained") changed = unconstrained.Run(&lp);
  if (mode == "main") changed = main_preprocessor.Run(&lp);
  std::cout << "changed " << changed << '\n';
  const auto status = mode == "main" ? main_preprocessor.status()
                      : mode == "singleton" ? singleton.status()
                      : mode == "empty_row" ? empty_row.status()
                      : mode == "empty_column" ? empty_column.status()
                      : (mode == "forcing" || mode == "forcing_exact") ? forcing.status()
                      : mode == "implied_free" ? implied_free.status()
                      : mode == "doubleton_equality" ? doubleton_equality.status()
                      : mode == "proportional_row" ? proportional_row.status()
                      : mode == "proportional_column" ? proportional_column.status()
                      : mode == "unconstrained" ? unconstrained.status()
                      : glop::ProblemStatus::INIT;
  std::cout << "status " << glop::GetProblemStatusString(status) << '\n';
  PrintBits("objective", lp.objective_coefficients());
  std::cout << "objective_offset " << Bits(lp.objective_offset()) << '\n';
  PrintBits("variable_lower", lp.variable_lower_bounds());
  PrintBits("variable_upper", lp.variable_upper_bounds());
  PrintBits("constraint_lower", lp.constraint_lower_bounds());
  PrintBits("constraint_upper", lp.constraint_upper_bounds());
  std::cout << "matrix";
  for (int col = 0; col < lp.num_variables().value(); ++col) {
    for (const auto entry : lp.GetSparseColumn(glop::ColIndex(col))) {
      std::cout << ' ' << col << ' ' << entry.row().value() << ' '
                << Bits(entry.coefficient());
    }
  }
  std::cout << '\n';

  glop::ProblemSolution solution(lp.num_constraints(), lp.num_variables());
  std::string post_status;
  std::cin >> post_status;
  if (post_status == "INIT") solution.status = glop::ProblemStatus::INIT;
  if (post_status == "PRIMAL_INFEASIBLE") solution.status = glop::ProblemStatus::PRIMAL_INFEASIBLE;
  if (post_status == "DUAL_FEASIBLE") solution.status = glop::ProblemStatus::DUAL_FEASIBLE;
  for (int col = 0; col < lp.num_variables().value(); ++col) {
    std::string status;
    std::cin >> status;
    const auto index = glop::ColIndex(col);
    if (status == "basic") solution.variable_statuses[index] = glop::VariableStatus::BASIC;
    if (status == "lower") solution.variable_statuses[index] = glop::VariableStatus::AT_LOWER_BOUND;
    if (status == "upper") solution.variable_statuses[index] = glop::VariableStatus::AT_UPPER_BOUND;
    if (status == "fixed") solution.variable_statuses[index] = glop::VariableStatus::FIXED_VALUE;
    if (status == "free") solution.variable_statuses[index] = glop::VariableStatus::FREE;
    solution.primal_values[index] = ReadDouble();
  }
  for (int row = 0; row < lp.num_constraints().value(); ++row) {
    std::string status;
    std::cin >> status;
    const auto index = glop::RowIndex(row);
    if (status == "basic") solution.constraint_statuses[index] = glop::ConstraintStatus::BASIC;
    if (status == "lower") solution.constraint_statuses[index] = glop::ConstraintStatus::AT_LOWER_BOUND;
    if (status == "upper") solution.constraint_statuses[index] = glop::ConstraintStatus::AT_UPPER_BOUND;
    if (status == "free") solution.constraint_statuses[index] = glop::ConstraintStatus::FREE;
    solution.dual_values[index] = ReadDouble();
  }
  if (changed) {
    if (mode == "fixed") fixed.RecoverSolution(&solution);
    if (mode == "free_row") free_row.RecoverSolution(&solution);
    if (mode == "empty_row") empty_row.RecoverSolution(&solution);
    if (mode == "empty_column") empty_column.RecoverSolution(&solution);
    if (mode == "singleton_sign") singleton_sign.RecoverSolution(&solution);
    if (mode == "singleton") singleton.RecoverSolution(&solution);
    if (mode == "forcing" || mode == "forcing_exact") forcing.RecoverSolution(&solution);
    if (mode == "implied_free") implied_free.RecoverSolution(&solution);
    if (mode == "doubleton_free") doubleton_free.RecoverSolution(&solution);
    if (mode == "doubleton_equality") doubleton_equality.RecoverSolution(&solution);
    if (mode == "proportional_row") proportional_row.RecoverSolution(&solution);
    if (mode == "proportional_column") proportional_column.RecoverSolution(&solution);
    if (mode == "dualizer") dualizer.RecoverSolution(&solution);
    if (mode == "unconstrained") unconstrained.RecoverSolution(&solution);
    if (mode == "main") main_preprocessor.DestructiveRecoverSolution(&solution);
  }
  PrintBits("recovered_primal", solution.primal_values);
  PrintBits("recovered_dual", solution.dual_values);
  std::cout << "recovered_status " << glop::GetProblemStatusString(solution.status) << '\n';
  std::cout << "variable_status";
  for (const auto status : solution.variable_statuses)
    std::cout << ' ' << glop::GetVariableStatusString(status);
  std::cout << "\nconstraint_status";
  for (const auto status : solution.constraint_statuses)
    std::cout << ' ' << glop::GetConstraintStatusString(status);
  std::cout << '\n';
}
