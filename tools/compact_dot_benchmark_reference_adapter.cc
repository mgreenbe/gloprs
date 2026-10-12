// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries_per_column;
  int repetitions;
  if (!(std::cin >> rows >> columns >> entries_per_column >> repetitions)) {
    return 2;
  }
  CompactSparseMatrix matrix;
  matrix.Reset(RowIndex(rows));
  for (int column = 0; column < columns; ++column) {
    for (int entry = 0; entry < entries_per_column; ++entry) {
      int row;
      double coefficient;
      std::cin >> row >> coefficient;
      matrix.AddEntryToCurrentColumn(RowIndex(row), coefficient);
    }
    matrix.CloseCurrentColumn();
  }
  DenseRow vector(ColIndex(rows), 0.0);
  for (int row = 0; row < rows; ++row) {
    vector[ColIndex(row)] = 1.0 + (row % 17) * 0.125;
  }
  const auto view = matrix.view();
  const auto vector_view = vector.const_view();
  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    for (int column = 0; column < columns; ++column) {
      checksum += view.ColumnScalarProduct(ColIndex(column), vector_view);
    }
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double nanos =
      std::chrono::duration<double, std::nano>(elapsed).count();
  std::cout << std::setprecision(17)
            << nanos / (static_cast<double>(columns) * repetitions) << ' '
            << checksum << '\n';
}
