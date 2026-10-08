// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iostream>

#include "ortools/lp_data/matrix_utils.h"

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries;
  double tolerance;
  if (!(std::cin >> rows >> columns >> entries >> tolerance)) return 2;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int entry = 0; entry < entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  matrix.CleanUp();
  for (const auto& [name, mapping] :
       {std::pair{"fast", FindProportionalColumns(matrix, tolerance)},
        std::pair{"simple", FindProportionalColumnsUsingSimpleAlgorithm(
                                matrix, tolerance)}}) {
    std::cout << name;
    for (ColIndex column(0); column < matrix.num_cols(); ++column) {
      std::cout << ' ' << mapping[column].value();
    }
    std::cout << '\n';
  }
  std::cout << "identity " << IsRightMostSquareMatrixIdentity(matrix) << '\n';
  CompactSparseMatrix compact(matrix);
  std::cout << "equal "
            << AreFirstColumnsAndRowsExactlyEquals(
                   matrix.num_rows(), matrix.num_cols(), matrix, compact)
            << '\n';
  return 0;
}
