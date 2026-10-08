// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ortools/glop/lu_factorization.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int num_entries;
  int repetitions;
  if (!(std::cin >> n >> num_entries >> repetitions)) return 2;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(n), ColIndex(n));
  for (int entry = 0; entry < num_entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row),
                                                            value);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  std::vector<ColIndex> columns;
  columns.reserve(n);
  for (int column = 0; column < n; ++column) columns.push_back(ColIndex(column));
  CompactSparseMatrixView view(&compact, &columns);
  GlopParameters parameters;

  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    LuFactorization factorization;
    factorization.SetParameters(parameters);
    if (!factorization.ComputeFactorization(view).ok()) return 3;
    checksum += factorization.ComputeDeterminant();
  }
  const auto stop = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = stop - start;
  std::cout << std::setprecision(17) << elapsed.count() << ' ' << checksum
            << '\n';
  return 0;
}
