// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <bit>
#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int count;
  int root_value;
  if (!(std::cin >> n >> count >> root_value)) return 2;
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
  std::cout << std::setprecision(17);
  std::cout << "metadata " << lower.num_rows().value() << ' '
            << lower.num_cols().value() << ' ' << lower.num_entries().value()
            << ' ' << lower.GetFirstNonIdentityColumn().value();
  for (int column = 0; column < n; ++column) {
    std::cout << ' '
              << lower.GetDiagonalCoefficient(ColIndex(column)) << ' '
              << lower.ColumnIsDiagonalOnly(ColIndex(column));
  }
  std::cout << ' ' << lower.IsLowerTriangular() << ' '
            << lower.IsUpperTriangular() << ' '
            << lower.ComputeInverseInfinityNormUpperBound() << ' '
            << lower.ComputeInverseInfinityNorm();
  std::cout << '\n';

  SparseMatrix copied;
  lower.CopyToSparseMatrix(&copied);
  std::cout << "copied";
  for (int column = 0; column < n; ++column) {
    for (const auto entry : copied.column(ColIndex(column))) {
      std::cout << ' ' << column << ' ' << entry.row().value() << ' '
                << entry.coefficient();
    }
  }
  std::cout << '\n';

  DenseColumn right(RowIndex(n), 0.0);
  DenseColumn left(RowIndex(n), 0.0);
  for (int row = 0; row < n; ++row) {
    right[RowIndex(row)] = row + 0.375;
    left[RowIndex(row)] = row + 0.375;
  }
  lower.LowerSolve(&right);
  lower.TransposeLowerSolve(&left);
  DenseColumn starting(RowIndex(n), 0.0);
  const int start = n / 3;
  for (int row = start; row < n; ++row) {
    starting[RowIndex(row)] = row + 0.625;
  }
  lower.LowerSolveStartingAt(ColIndex(start), &starting);
  DenseColumn signed_zero_right(RowIndex(n), -0.0);
  DenseColumn signed_zero_left(RowIndex(n), -0.0);
  lower.LowerSolve(&signed_zero_right);
  lower.TransposeLowerSolve(&signed_zero_left);
  std::cout << "right";
  for (const double value : right) std::cout << ' ' << value;
  std::cout << "\nleft";
  for (const double value : left) std::cout << ' ' << value;
  std::cout << "\nstarting";
  for (const double value : starting) std::cout << ' ' << value;
  std::cout << "\nsigned_zero";
  for (const double value : signed_zero_right) {
    std::cout << " x" << std::hex << std::bit_cast<uint64_t>(value) << std::dec;
  }
  for (const double value : signed_zero_left) {
    std::cout << " x" << std::hex << std::bit_cast<uint64_t>(value) << std::dec;
  }
  std::cout << '\n';

  const RowIndex root(root_value);
  RowIndexVector sorted_rows;
  if (n != 0) sorted_rows.push_back(root);
  lower.ComputeRowsToConsiderInSortedOrder(&sorted_rows);
  RowIndexVector dfs_rows;
  if (n != 0) dfs_rows.push_back(root);
  lower.ComputeRowsToConsiderWithDfs(&dfs_rows);
  std::cout << "sorted " << sorted_rows.size();
  for (const RowIndex row : sorted_rows) std::cout << ' ' << row.value();
  std::cout << "\ndfs " << dfs_rows.size();
  for (const RowIndex row : dfs_rows) std::cout << ' ' << row.value();

  DenseColumn hyper_sorted(RowIndex(n), 0.0);
  if (n != 0) hyper_sorted[root] = 1.25;
  RowIndexVector hyper_sorted_rows = sorted_rows;
  lower.HyperSparseSolve(&hyper_sorted, &hyper_sorted_rows);
  std::cout << "\nhyper_sorted " << hyper_sorted_rows.size();
  for (const RowIndex row : hyper_sorted_rows) std::cout << ' ' << row.value();
  for (const double value : hyper_sorted) std::cout << ' ' << value;

  DenseColumn hyper_dfs(RowIndex(n), 0.0);
  if (n != 0) hyper_dfs[root] = 1.25;
  RowIndexVector hyper_dfs_rows = dfs_rows;
  lower.HyperSparseSolveWithReversedNonZeros(&hyper_dfs, &hyper_dfs_rows);
  std::cout << "\nhyper_dfs " << hyper_dfs_rows.size();
  for (const RowIndex row : hyper_dfs_rows) std::cout << ' ' << row.value();
  for (const double value : hyper_dfs) std::cout << ' ' << value;

  SparseMatrix transposed_input;
  transposed_input.PopulateFromTranspose(input);
  TriangularMatrix upper;
  upper.PopulateFromTriangularSparseMatrix(transposed_input);
  RowIndexVector transpose_sorted_rows;
  if (n != 0) transpose_sorted_rows.push_back(RowIndex(n - 1));
  upper.ComputeRowsToConsiderInSortedOrder(&transpose_sorted_rows);
  DenseColumn transpose_hyper(RowIndex(n), 0.0);
  if (n != 0) transpose_hyper[RowIndex(n - 1)] = -0.75;
  lower.TransposeHyperSparseSolveWithReversedNonZeros(
      &transpose_hyper, &transpose_sorted_rows);
  std::cout << "\ntranspose_hyper " << transpose_sorted_rows.size();
  for (const RowIndex row : transpose_sorted_rows) {
    std::cout << ' ' << row.value();
  }
  for (const double value : transpose_hyper) std::cout << ' ' << value;
  std::cout << '\n';

  TriangularMatrix normalized;
  normalized.Reset(RowIndex(n), ColIndex(n));
  for (int column = 0; column < n; ++column) {
    const double diagonal = input.LookUpValue(RowIndex(column), ColIndex(column));
    normalized.AddAndNormalizeTriangularColumn(
        input.column(ColIndex(column)), RowIndex(column), diagonal);
  }
  SparseMatrix normalized_copy;
  normalized.CopyToSparseMatrix(&normalized_copy);
  std::cout << "normalized " << normalized.GetFirstNonIdentityColumn().value();
  for (int column = 0; column < n; ++column) {
    for (const auto entry : normalized_copy.column(ColIndex(column))) {
      std::cout << ' ' << column << ' ' << entry.row().value() << ' '
                << entry.coefficient();
    }
  }
  std::cout << '\n';
  return 0;
}
