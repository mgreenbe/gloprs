// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Differential adapter for GLOP's dual ratio tests.

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/random/random.h"
#include "ortools/glop/basis_representation.h"
#include "ortools/glop/entering_variable.h"
#include "ortools/glop/reduced_costs.h"
#include "ortools/glop/update_row.h"
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
}  // namespace

int main() {
  using namespace operations_research::glop;
  int rows;
  int structural;
  int entries;
  int leaving_row;
  double cost_variation;
  if (!(std::cin >> rows >> structural >> entries >> leaving_row >>
        cost_variation)) {
    return 2;
  }
  const int columns = structural + rows;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int entry = 0; entry < entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row),
                                                            value);
  }
  for (int row = 0; row < rows; ++row) {
    matrix.mutable_column(ColIndex(structural + row))
        ->SetCoefficient(RowIndex(row), 1.0);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  CompactSparseMatrix transpose;
  transpose.PopulateFromTranspose(compact);
  DenseRow objective(ColIndex(columns), 0.0);
  DenseRow lower(ColIndex(columns), 0.0);
  DenseRow upper(ColIndex(columns), 0.0);
  for (int column = 0; column < columns; ++column) {
    objective[ColIndex(column)] = ReadDouble();
    lower[ColIndex(column)] = ReadDouble();
    upper[ColIndex(column)] = ReadDouble();
  }
  RowToColMapping basis(RowIndex(rows), ColIndex(0));
  for (int row = 0; row < rows; ++row) {
    basis[RowIndex(row)] = ColIndex(structural + row);
  }
  VariablesInfo variables(compact);
  variables.LoadBoundsAndReturnTrueIfUnchanged(lower, upper);
  variables.InitializeToDefaultStatus();
  for (int row = 0; row < rows; ++row) {
    variables.UpdateToBasicStatus(basis[RowIndex(row)]);
  }
  BasisFactorization factorization(&compact, &basis);
  if (!factorization.Initialize().ok()) return 3;
  UpdateRow update(compact, transpose, variables, basis, factorization);
  update.ComputeUpdateRow(RowIndex(leaving_row));
  absl::BitGen random;
  ReducedCosts reduced(compact, objective, basis, variables, factorization,
                       random);
  GlopParameters parameters;
  reduced.SetParameters(parameters);
  EnteringVariable entering(variables, random, &reduced);
  entering.SetParameters(parameters);
  ColIndex phase_two = kInvalidCol;
  std::vector<ColIndex> flips;
  if (!entering
           .DualChooseEnteringColumn(true, update, cost_variation, &flips,
                                     &phase_two)
           .ok()) {
    return 4;
  }
  ColIndex phase_one = kInvalidCol;
  if (!entering
           .DualPhaseIChooseEnteringColumn(true, update, cost_variation,
                                           &phase_one)
           .ok()) {
    return 5;
  }
  std::cout << std::setprecision(17) << "positions";
  for (const ColIndex column : update.GetNonZeroPositions()) {
    std::cout << ' ' << column.value() << ':'
              << update.GetCoefficient(column);
  }
  std::cout << "\nphase_two " << phase_two.value() << "\nflips";
  for (const ColIndex column : flips) std::cout << ' ' << column.value();
  std::cout << "\nphase_one " << phase_one.value() << '\n';
  return 0;
}
