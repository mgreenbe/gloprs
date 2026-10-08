// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse.h"

namespace {
using namespace operations_research::glop;

void Print(const char* name, const CompactSparseMatrix& matrix) {
  std::cout << name << ' ' << matrix.num_rows().value() << ' '
            << matrix.num_cols().value() << ' ' << matrix.num_entries().value();
  for (ColIndex column(0); column < matrix.num_cols(); ++column) {
    for (const auto entry : matrix.column(column)) {
      std::cout << ' ' << column.value() << ' ' << entry.row().value() << ' '
                << entry.coefficient();
    }
  }
  std::cout << '\n';
}
}  // namespace

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries;
  if (!(std::cin >> rows >> columns >> entries)) return 2;
  SparseMatrix sparse;
  sparse.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int entry = 0; entry < entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    sparse.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  sparse.CleanUp();
  CompactSparseMatrix compact(sparse);
  CompactSparseMatrix transpose;
  transpose.PopulateFromTranspose(compact);
  CompactSparseMatrix slacks;
  slacks.PopulateFromSparseMatrixAndAddSlacks(sparse);
  std::cout << std::setprecision(17);
  Print("compact", compact);
  Print("transpose", transpose);
  Print("slacks", slacks);
  std::vector<ColIndex> selected;
  for (int column = columns - 1; column >= 0; column -= 2) {
    selected.push_back(ColIndex(column));
  }
  CompactSparseMatrixView compact_view(&compact, &selected);
  std::cout << "compact_view " << compact_view.num_rows().value() << ' '
            << compact_view.num_cols().value() << ' '
            << compact_view.num_entries().value() << ' '
            << compact_view.ComputeOneNorm() << ' '
            << compact_view.ComputeInfinityNorm();
  for (ColIndex column(0); column < compact_view.num_cols(); ++column) {
    for (const auto entry : compact_view.column(column)) {
      std::cout << ' ' << column.value() << ' ' << entry.row().value() << ' '
                << entry.coefficient();
    }
  }
  std::cout << '\n';
  MatrixView full_view(sparse);
  RowToColMapping basis(RowIndex(selected.size()), ColIndex(0));
  for (int position = 0; position < selected.size(); ++position) {
    basis[RowIndex(position)] = selected[position];
  }
  MatrixView sparse_view;
  sparse_view.PopulateFromBasis(full_view, basis);
  std::cout << "sparse_view " << sparse_view.num_rows().value() << ' '
            << sparse_view.num_cols().value() << ' '
            << sparse_view.num_entries().value() << ' '
            << sparse_view.ComputeOneNorm() << ' '
            << sparse_view.ComputeInfinityNorm() << '\n';
  DenseRow vector(ColIndex(rows), 0.0);
  for (int row = 0; row < rows; ++row) vector[ColIndex(row)] = row + 0.25;
  std::cout << "products";
  for (int column = 0; column < columns; ++column) {
    std::cout << ' ' << compact.ColumnScalarProduct(ColIndex(column), vector);
  }
  std::cout << '\n';
  return 0;
}
