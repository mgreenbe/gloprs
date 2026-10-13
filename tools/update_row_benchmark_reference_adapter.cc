// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <iomanip>
#include <iostream>

#include "ortools/glop/basis_representation.h"
#include "ortools/glop/update_row.h"
#include "ortools/glop/variables_info.h"
#include "ortools/lp_data/sparse.h"

int main(int argc, char** argv) {
  using namespace operations_research::glop;
  if (argc != 5) return 2;
  const int rows = std::stoi(argv[1]);
  const int structural_columns = std::stoi(argv[2]);
  const int entries = std::stoi(argv[3]);
  const int repetitions = std::stoi(argv[4]);
  if (rows < 1 || structural_columns < 1 || entries < 0 ||
      entries * 13 >= rows || repetitions < 1) {
    return 2;
  }
  const int columns = structural_columns + rows;
  SparseMatrix sparse;
  sparse.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int column = 0; column < structural_columns; ++column) {
    for (int entry = 0; entry < entries; ++entry) {
      sparse.mutable_column(ColIndex(column))
          ->SetCoefficient(RowIndex((column * 97 + entry * 13) % rows),
                           (entry % 7 + 1) * 0.125);
    }
  }
  for (int row = 0; row < rows; ++row) {
    sparse.mutable_column(ColIndex(structural_columns + row))
        ->SetCoefficient(RowIndex(row), 1.0);
  }
  sparse.CleanUp();
  CompactSparseMatrix matrix(sparse);
  CompactSparseMatrix transpose;
  transpose.PopulateFromTranspose(matrix);
  DenseRow lower(ColIndex(columns), -kInfinity);
  DenseRow upper(ColIndex(columns), kInfinity);
  VariablesInfo variables(matrix);
  variables.LoadBoundsAndReturnTrueIfUnchanged(lower, upper);
  variables.InitializeToDefaultStatus();
  RowToColMapping basis(RowIndex(rows), ColIndex(0));
  for (int row = 0; row < rows; ++row) {
    basis[RowIndex(row)] = ColIndex(structural_columns + row);
  }
  BasisFactorization factorization(&matrix, &basis);
  UpdateRow update(matrix, transpose, variables, basis, factorization);
  DenseRow lhs(ColIndex(rows), 0.0);
  for (int row = 0; row < rows; ++row) {
    if (row % 4 == 0) lhs[ColIndex(row)] = (row % 9 + 1) * 0.125;
  }

  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    update.ComputeUpdateRowForBenchmark(lhs, "column");
    checksum += update.GetCoefficient(ColIndex(repetition % columns));
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  std::cout << std::setprecision(17)
            << std::chrono::duration<double, std::micro>(elapsed).count() /
                   repetitions
            << ' ' << checksum << '\n';
}
