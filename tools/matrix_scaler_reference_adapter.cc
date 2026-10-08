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
  std::cout << "\nrow_factors";
  for (int row = 0; row <= rows + 1; ++row) {
    std::cout << ' ' << scaler.RowScalingFactor(RowIndex(row)) << ' '
              << scaler.RowUnscalingFactor(RowIndex(row));
  }
  std::cout << "\ncolumn_factors";
  for (int col = 0; col <= columns + 1; ++col) {
    std::cout << ' ' << scaler.ColScalingFactor(ColIndex(col)) << ' '
              << scaler.ColUnscalingFactor(ColIndex(col));
  }
  DenseRow row_vector(ColIndex(columns + 2), 0.0);
  for (int col = 0; col < columns + 2; ++col) {
    row_vector[ColIndex(col)] = col + 0.25;
  }
  scaler.ScaleRowVector(true, &row_vector);
  std::cout << "\nrow_up";
  for (double value : row_vector) std::cout << ' ' << value;
  scaler.ScaleRowVector(false, &row_vector);
  std::cout << "\nrow_roundtrip";
  for (double value : row_vector) std::cout << ' ' << value;
  DenseColumn column_vector(RowIndex(rows + 2), 0.0);
  for (int row = 0; row < rows + 2; ++row) {
    column_vector[RowIndex(row)] = row - 0.75;
  }
  scaler.ScaleColumnVector(true, &column_vector);
  std::cout << "\ncolumn_up";
  for (double value : column_vector) std::cout << ' ' << value;
  scaler.ScaleColumnVector(false, &column_vector);
  std::cout << "\ncolumn_roundtrip";
  for (double value : column_vector) std::cout << ' ' << value;

  // The pinned Init() implementation uses resize(), whose actual behavior is
  // to retain factors when the dimensions are unchanged.
  scaler.Init(&matrix);
  std::cout << "\nreinit_rows";
  for (double value : scaler.row_scales()) std::cout << ' ' << value;
  std::cout << "\nreinit_columns";
  for (double value : scaler.col_scales()) std::cout << ' ' << value;
  scaler.Clear();
  std::cout << "\nclear " << scaler.row_scales().size().value() << ' '
            << scaler.col_scales().size().value() << ' '
            << scaler.RowUnscalingFactor(RowIndex(rows + 1)) << ' '
            << scaler.ColUnscalingFactor(ColIndex(columns + 1));
  std::cout << '\n';
}
