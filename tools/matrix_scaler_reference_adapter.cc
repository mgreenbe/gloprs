// Copyright 2026 Matthew Greenberg
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/glop/parameters.pb.h"
#include "ortools/lp_data/matrix_scaler.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries;
  if (!(std::cin >> rows >> columns >> entries)) return 2;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int i = 0; i < entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  matrix.CleanUp();
  SparseMatrixScaler scaler;
  scaler.Init(&matrix);
  scaler.Scale(GlopParameters::EQUILIBRATION);
  std::cout << std::setprecision(17);
  std::cout << "rows";
  for (double value : scaler.row_scales()) std::cout << ' ' << value;
  std::cout << "\ncolumns";
  for (double value : scaler.col_scales()) std::cout << ' ' << value;
  std::cout << "\nmatrix";
  for (ColIndex col(0); col < matrix.num_cols(); ++col) {
    for (const auto entry : matrix.column(col)) {
      std::cout << ' ' << col.value() << ' ' << entry.row().value() << ' '
                << entry.coefficient();
    }
  }
  std::cout << '\n';
}
