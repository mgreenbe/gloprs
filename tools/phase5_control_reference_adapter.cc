// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Native public-solver controls for empty LP and pre-expired time limit.

#include <iostream>

#include "ortools/glop/lp_solver.h"
#include "ortools/util/time_limit.h"

int main() {
  namespace glop = operations_research::glop;
  glop::GlopParameters parameters;
  parameters.set_use_preprocessing(false);
  parameters.set_use_scaling(false);
  glop::LinearProgram empty;
  glop::LPSolver solver;
  solver.SetParameters(parameters);
  std::cout << "empty " << glop::GetProblemStatusString(solver.Solve(empty)) << '\n';

  glop::LinearProgram one_row;
  const auto column = one_row.CreateNewVariable();
  const auto row = one_row.CreateNewConstraint();
  one_row.SetVariableBounds(column, 0.0, 10.0);
  one_row.SetConstraintBounds(row, 5.0, glop::kInfinity);
  one_row.SetCoefficient(row, column, 1.0);
  one_row.CleanUp();
  auto limit = operations_research::TimeLimit::FromDeterministicTime(0.0);
  std::cout << "expired "
            << glop::GetProblemStatusString(solver.SolveWithTimeLimit(one_row, limit.get()))
            << '\n';
}
