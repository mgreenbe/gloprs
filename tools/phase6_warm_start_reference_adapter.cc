// Native public-LPSolver basis export/import reference for the pinned GLOP.
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

#include "ortools/glop/lp_solver.h"

namespace {
using namespace operations_research::glop;

void MakeProgram(LinearProgram& lp) {
  const ColIndex x = lp.CreateNewVariable();
  const ColIndex y = lp.CreateNewVariable();
  const RowIndex r = lp.CreateNewConstraint();
  lp.SetVariableBounds(x, 0.0, 10.0);
  lp.SetVariableBounds(y, 0.0, 10.0);
  lp.SetConstraintBounds(r, 5.0, std::numeric_limits<double>::infinity());
  lp.SetCoefficient(r, x, 1.0);
  lp.SetCoefficient(r, y, 1.0);
  lp.SetObjectiveCoefficient(x, 1.0);
  lp.SetObjectiveCoefficient(y, 2.0);
  lp.CleanUp();
}

void Print(const char* label, LPSolver& solver, const LinearProgram& lp) {
  const ProblemStatus status = solver.Solve(lp);
  std::uint64_t bits;
  const double objective = solver.GetObjectiveValue();
  std::memcpy(&bits, &objective, sizeof(bits));
  std::cout << label << ' ' << GetProblemStatusString(status) << ' '
            << solver.GetNumberOfSimplexIterations() << ' ' << bits;
  for (const auto value : solver.variable_values()) std::cout << ' ' << value;
  for (const auto value : solver.variable_statuses())
    std::cout << ' ' << GetVariableStatusString(value);
  for (const auto value : solver.constraint_statuses())
    std::cout << ' ' << GetConstraintStatusString(value);
  std::cout << '\n';
}
}  // namespace

int main() {
  LinearProgram lp;
  MakeProgram(lp);
  GlopParameters parameters;
  parameters.set_use_preprocessing(false);
  parameters.set_use_scaling(false);
  parameters.set_initial_basis(GlopParameters::NONE);
  parameters.set_exploit_singleton_column_in_initial_basis(false);
  LPSolver cold;
  cold.SetParameters(parameters);
  Print("cold", cold, lp);
  LPSolver warm;
  warm.SetParameters(parameters);
  warm.SetInitialBasis(cold.variable_statuses(), cold.constraint_statuses());
  Print("warm", warm, lp);
}
