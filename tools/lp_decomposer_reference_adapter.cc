#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ortools/lp_data/lp_decomposer.h"

int main() {
  using namespace operations_research::glop;
  int n, m, count, maximize;
  if (!(std::cin >> n >> m >> count >> maximize)) return 2;
  LinearProgram lp;
  lp.SetMaximizationProblem(maximize);
  for (int col = 0; col < n; ++col) {
    std::string name; int integer; double lower, upper, objective;
    std::cin >> name >> integer >> lower >> upper >> objective;
    const ColIndex c = lp.CreateNewVariable();
    lp.SetVariableName(c, name);
    if (integer) lp.SetVariableType(c, LinearProgram::VariableType::INTEGER);
    lp.SetVariableBounds(c, lower, upper);
    lp.SetObjectiveCoefficient(c, objective);
  }
  for (int row = 0; row < m; ++row) {
    std::string name; double lower, upper;
    std::cin >> name >> lower >> upper;
    const RowIndex r = lp.CreateNewConstraint();
    lp.SetConstraintName(r, name);
    lp.SetConstraintBounds(r, lower, upper);
  }
  for (int i = 0; i < count; ++i) {
    int row, col; double value; std::cin >> row >> col >> value;
    lp.SetCoefficient(RowIndex(row), ColIndex(col), value);
  }
  LPDecomposer d; d.Decompose(&lp);
  std::cout << "problems " << d.GetNumberOfProblems() << '\n';
  DenseRow global(ColIndex(n), 0.0);
  for (int col = 0; col < n; ++col) global[ColIndex(col)] = col + 0.5;
  std::vector<DenseRow> locals;
  for (int p = 0; p < d.GetNumberOfProblems(); ++p) {
    LinearProgram local; d.ExtractLocalProblem(p, &local);
    std::cout << "P " << p << ' ' << local.IsMaximizationProblem() << ' '
              << local.num_variables().value() << ' ' << local.num_constraints().value()
              << ' ' << local.num_entries().value() << '\n';
    for (ColIndex c(0); c < local.num_variables(); ++c) {
      std::cout << "V " << local.GetVariableName(c) << ' ' << static_cast<int>(local.GetVariableType(c))
                << ' ' << std::hex << std::bit_cast<uint64_t>(local.variable_lower_bounds()[c])
                << ' ' << std::bit_cast<uint64_t>(local.variable_upper_bounds()[c])
                << ' ' << std::bit_cast<uint64_t>(local.objective_coefficients()[c]) << std::dec << '\n';
      for (const auto e : local.GetSparseColumn(c))
        std::cout << "E " << c.value() << ' ' << e.row().value() << ' ' << std::hex
                  << std::bit_cast<uint64_t>(e.coefficient()) << std::dec << '\n';
    }
    for (RowIndex r(0); r < local.num_constraints(); ++r)
      std::cout << "R " << local.GetConstraintName(r) << ' ' << std::hex
                << std::bit_cast<uint64_t>(local.constraint_lower_bounds()[r]) << ' '
                << std::bit_cast<uint64_t>(local.constraint_upper_bounds()[r]) << std::dec << '\n';
    locals.push_back(d.ExtractLocalAssignment(p, global));
    std::cout << "A";
    for (double value : locals.back()) std::cout << ' ' << std::hex << std::bit_cast<uint64_t>(value) << std::dec;
    std::cout << '\n';
  }
  const DenseRow aggregate = d.AggregateAssignments(locals);
  std::cout << "G";
  for (double value : aggregate) std::cout << ' ' << std::hex << std::bit_cast<uint64_t>(value) << std::dec;
  std::cout << '\n';
}
