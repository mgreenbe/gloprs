// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int count;
  int repetitions;
  if (!(std::cin >> n >> count >> repetitions)) return 2;
  SparseMatrix input;
  input.PopulateFromZero(RowIndex(n), ColIndex(n));
  for (int position = 0; position < count; ++position) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    input.mutable_column(ColIndex(column))
        ->SetCoefficient(RowIndex(row), value);
  }
  input.CleanUp();
  TriangularMatrix lower;
  lower.PopulateFromTriangularSparseMatrix(input);
  DenseColumn base(RowIndex(n), 0.0);
  DenseColumn rhs(RowIndex(n), 0.0);
  for (int row = 0; row < n; ++row) {
    base[RowIndex(row)] = 1.0 + (row % 17) * 0.125;
  }

  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    for (int row = 0; row < n; ++row) {
      rhs[RowIndex(row)] = base[RowIndex(row)];
    }
    lower.TransposeLowerSolve(&rhs);
    checksum += rhs[RowIndex(repetition % n)];
  }
  const auto end = std::chrono::steady_clock::now();
  const double nanos =
      std::chrono::duration<double, std::nano>(end - start).count();
  std::cout << std::setprecision(17) << nanos / repetitions << ' '
            << checksum << '\n';
}
