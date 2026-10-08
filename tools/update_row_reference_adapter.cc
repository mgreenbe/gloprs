// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>
#include <string>

#include "ortools/glop/basis_representation.h"
#include "ortools/glop/update_row.h"
#include "ortools/glop/variables_info.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int rows;
  int structural_columns;
  int entries;
  if (!(std::cin >> rows >> structural_columns >> entries)) return 2;
  const int columns = structural_columns + rows;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int i = 0; i < entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  for (int row = 0; row < rows; ++row) {
    matrix.mutable_column(ColIndex(structural_columns + row))
        ->SetCoefficient(RowIndex(row), 1.0);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  CompactSparseMatrix transpose;
  transpose.PopulateFromTranspose(compact);
  DenseRow lower(ColIndex(columns), -kInfinity);
  DenseRow upper(ColIndex(columns), kInfinity);
  VariablesInfo variables(compact);
  variables.LoadBoundsAndReturnTrueIfUnchanged(lower, upper);
  variables.InitializeToDefaultStatus();
  RowToColMapping basis(RowIndex(rows), ColIndex(0));
  for (int row = 0; row < rows; ++row) {
    basis[RowIndex(row)] = ColIndex(structural_columns + row);
  }
  BasisFactorization factorization(&compact, &basis);
  UpdateRow update(compact, transpose, variables, basis, factorization);
  DenseRow lhs(ColIndex(rows), 0.0);
  for (int row = 0; row < rows; ++row) std::cin >> lhs[ColIndex(row)];

  std::cout << std::setprecision(17);
  for (const std::string algorithm : {"column", "row", "row_hypersparse"}) {
    update.ComputeUpdateRowForBenchmark(lhs, algorithm);
    std::cout << algorithm << "_positions";
    for (const ColIndex column : update.GetNonZeroPositions()) {
      std::cout << ' ' << column.value();
    }
    std::cout << '\n' << algorithm << "_coefficients";
    for (int column = 0; column < columns; ++column) {
      std::cout << ' ' << update.GetCoefficient(ColIndex(column));
    }
    std::cout << '\n';
  }
  std::cout << "deterministic_time " << update.DeterministicTime() << '\n';
  std::cout << "stats_size " << update.StatString().size() << '\n';
  return 0;
}
