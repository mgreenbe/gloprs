// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Focused differential adapter for GLOP's LU initial-basis construction.

#include <iostream>
#include <vector>

#include "ortools/glop/lu_factorization.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int num_rows;
  int num_columns;
  int num_entries;
  int num_candidates;
  if (!(std::cin >> num_rows >> num_columns >> num_entries >> num_candidates)) {
    return 2;
  }
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(num_rows), ColIndex(num_columns));
  for (int i = 0; i < num_entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row),
                                                            value);
  }
  matrix.CleanUp();
  std::vector<ColIndex> candidates;
  candidates.reserve(num_candidates);
  for (int i = 0; i < num_candidates; ++i) {
    int column;
    std::cin >> column;
    candidates.push_back(ColIndex(column));
  }
  CompactSparseMatrix compact(matrix);
  LuFactorization factorization;
  const RowToColMapping basis =
      factorization.ComputeInitialBasis(compact, candidates);
  std::cout << "basis";
  for (const ColIndex column : basis) std::cout << ' ' << column.value();
  std::cout << "\npivots";
  const RowPermutation& row_perm = factorization.row_perm();
  const ColumnPermutation& col_perm = factorization.GetColumnPermutation();
  for (int step = 0; step < basis.size().value(); ++step) {
    int pivot_row = -1;
    int pivot_column = -1;
    for (int row = 0; row < row_perm.size().value(); ++row) {
      if (row_perm[RowIndex(row)].value() == step) pivot_row = row;
    }
    for (int column = 0; column < col_perm.size().value(); ++column) {
      if (col_perm[ColIndex(column)].value() == step) {
        pivot_column = candidates[column].value();
      }
    }
    if (pivot_row < 0 || pivot_column < 0) break;
    std::cout << ' ' << pivot_row << ':' << pivot_column;
  }
  std::cout << '\n';
  return 0;
}
