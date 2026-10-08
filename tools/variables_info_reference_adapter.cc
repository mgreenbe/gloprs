// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Differential adapter for pinned GLOP VariablesInfo.

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "ortools/glop/variables_info.h"
#include "ortools/lp_data/sparse.h"

namespace {
double ReadDouble() {
  std::string token;
  std::cin >> token;
  if (token == "inf") return std::numeric_limits<double>::infinity();
  if (token == "-inf") return -std::numeric_limits<double>::infinity();
  return std::stod(token);
}

void Emit(const std::string& label,
          const operations_research::glop::VariablesInfo& info) {
  using namespace operations_research::glop;
  const ColIndex n = info.GetNumberOfColumns();
  std::cout << label << "_types";
  for (ColIndex col(0); col < n; ++col) {
    std::cout << ' ' << GetVariableTypeString(info.GetTypeRow()[col]);
  }
  std::cout << '\n' << label << "_statuses";
  for (ColIndex col(0); col < n; ++col) {
    std::cout << ' ' << GetVariableStatusString(info.GetStatusRow()[col]);
  }
  for (const auto& [name, bits] :
       std::vector<std::pair<std::string, const DenseBitRow*>>{
           {"increase", &info.GetCanIncreaseBitRow()},
           {"decrease", &info.GetCanDecreaseBitRow()},
           {"relevant", &info.GetIsRelevantBitRow()},
           {"basic", &info.GetIsBasicBitRow()},
           {"not_basic", &info.GetNotBasicBitRow()},
           {"boxed", &info.GetNonBasicBoxedVariables()}}) {
    std::cout << '\n' << label << '_' << name;
    for (ColIndex col(0); col < n; ++col) std::cout << ' ' << ((*bits)[col] ? 1 : 0);
  }
  std::cout << '\n' << label << "_entries "
            << info.GetNumEntriesInRelevantColumns().value();
  std::cout << '\n' << label << "_lower";
  for (ColIndex col(0); col < n; ++col) {
    std::cout << ' ' << info.GetVariableLowerBounds()[col];
  }
  std::cout << '\n' << label << "_upper";
  for (ColIndex col(0); col < n; ++col) {
    std::cout << ' ' << info.GetVariableUpperBounds()[col];
  }
  std::cout << '\n';
}
}  // namespace

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int num_entries;
  if (!(std::cin >> rows >> columns >> num_entries)) return 2;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int i = 0; i < num_entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);

  DenseRow lower(ColIndex(columns), 0.0);
  DenseRow upper(ColIndex(columns), 0.0);
  for (int column = 0; column < columns; ++column) {
    lower[ColIndex(column)] = ReadDouble();
    upper[ColIndex(column)] = ReadDouble();
  }
  DenseRow reduced_costs(ColIndex(columns), 0.0);
  for (int column = 0; column < columns; ++column) {
    reduced_costs[ColIndex(column)] = ReadDouble();
  }
  int num_basic;
  std::cin >> num_basic;
  std::vector<int> basic(num_basic);
  for (int& column : basic) std::cin >> column;

  VariablesInfo info(compact);
  info.LoadBoundsAndReturnTrueIfUnchanged(lower, upper);
  info.InitializeToDefaultStatus();
  std::cout << std::setprecision(17);
  Emit("default", info);
  info.MakeBoxedVariableRelevant(false);
  Emit("unboxed", info);
  info.MakeBoxedVariableRelevant(true);
  for (const int column : basic) info.UpdateToBasicStatus(ColIndex(column));
  Emit("basic", info);
  info.TransformToDualPhaseIProblem(1e-7, reduced_costs.const_view());
  Emit("phase1", info);
  info.EndDualPhaseI(1e-7, reduced_costs.const_view());
  Emit("restored", info);

  // Exercise GLOP's zero-copy incremental bound API. InitializeFromMutatedState
  // intentionally updates types only; callers initialize statuses separately.
  for (int column = 0; column < columns; ++column) {
    const ColIndex col(column);
    switch (column % 5) {
      case 0:
        (*info.MutableLowerBounds())[col] = -std::numeric_limits<double>::infinity();
        (*info.MutableUpperBounds())[col] = std::numeric_limits<double>::infinity();
        break;
      case 1:
        (*info.MutableLowerBounds())[col] = -2.0;
        (*info.MutableUpperBounds())[col] = std::numeric_limits<double>::infinity();
        break;
      case 2:
        (*info.MutableLowerBounds())[col] = -std::numeric_limits<double>::infinity();
        (*info.MutableUpperBounds())[col] = 3.0;
        break;
      case 3:
        (*info.MutableLowerBounds())[col] = -4.0;
        (*info.MutableUpperBounds())[col] = 5.0;
        break;
      default:
        (*info.MutableLowerBounds())[col] = 6.0;
        (*info.MutableUpperBounds())[col] = 6.0;
    }
  }
  info.InitializeFromMutatedState();
  info.InitializeToDefaultStatus();
  Emit("mutated", info);

  // Exercise the allocation-free structural/slack overload and its unchanged
  // fast path. The last `rows` columns are interpreted as slacks.
  const int variables = columns - rows;
  DenseRow variable_lower(ColIndex(variables), 0.0);
  DenseRow variable_upper(ColIndex(variables), 0.0);
  DenseColumn constraint_lower(RowIndex(rows), 0.0);
  DenseColumn constraint_upper(RowIndex(rows), 0.0);
  for (int column = 0; column < variables; ++column) {
    variable_lower[ColIndex(column)] = lower[ColIndex(column)];
    variable_upper[ColIndex(column)] = upper[ColIndex(column)];
  }
  for (int row = 0; row < rows; ++row) {
    const ColIndex slack(variables + row);
    constraint_lower[RowIndex(row)] = -upper[slack];
    constraint_upper[RowIndex(row)] = -lower[slack];
  }
  VariablesInfo structural(compact);
  const bool first_unchanged = structural.LoadBoundsAndReturnTrueIfUnchanged(
      variable_lower, variable_upper, constraint_lower, constraint_upper);
  const bool second_unchanged = structural.LoadBoundsAndReturnTrueIfUnchanged(
      variable_lower, variable_upper, constraint_lower, constraint_upper);
  std::cout << "structural_unchanged " << first_unchanged << ' '
            << second_unchanged << '\n';
  structural.InitializeToDefaultStatus();
  Emit("structural", structural);
  return 0;
}
