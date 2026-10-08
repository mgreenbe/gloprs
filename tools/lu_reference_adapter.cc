// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Focused differential adapter for the pinned GLOP LU implementation.

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <vector>

#include "ortools/glop/lu_factorization.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int num_entries;
  if (!(std::cin >> n >> num_entries)) return 2;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(n), ColIndex(n));
  for (int i = 0; i < num_entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row),
                                                            value);
  }
  matrix.CleanUp();
  double pivot_threshold = 0.01;
  int zlatev_parameter = 3;
  double singularity_threshold = 1e-15;
  (void)(std::cin >> pivot_threshold >> zlatev_parameter >>
         singularity_threshold);
  CompactSparseMatrix compact(matrix);
  std::vector<ColIndex> columns;
  for (int column = 0; column < n; ++column) columns.push_back(ColIndex(column));
  CompactSparseMatrixView view(&compact, &columns);

  LuFactorization factorization;
  GlopParameters parameters;
  parameters.set_lu_factorization_pivot_threshold(pivot_threshold);
  parameters.set_markowitz_zlatev_parameter(zlatev_parameter);
  parameters.set_markowitz_singularity_threshold(singularity_threshold);
  factorization.SetParameters(parameters);
  const Status status = factorization.ComputeFactorization(view);
  if (!status.ok()) {
    std::cout << "singular\n";
    return 0;
  }
  std::cout << std::setprecision(17);
  std::cout << "row_perm";
  for (int row = 0; row < n; ++row) {
    std::cout << ' ' << factorization.row_perm()[RowIndex(row)].value();
  }
  std::cout << "\ninverse_col_perm";
  for (int column = 0; column < n; ++column) {
    std::cout << ' '
              << factorization.inverse_col_perm()[ColIndex(column)].value();
  }
  std::cout << "\ndeterminant " << factorization.ComputeDeterminant();
  std::cout << "\ndeterministic_time "
            << factorization.DeterministicTimeOfLastFactorization();
  std::cout << "\nentries " << factorization.NumberOfEntries().value();
  int upper_entries = 0;
  std::cout << "\nupper";
  for (int column = 0; column < n; ++column) {
    const SparseColumn& upper = factorization.GetColumnOfU(ColIndex(column));
    upper_entries += upper.num_entries().value();
    for (const SparseColumn::Entry entry : upper) {
      std::cout << ' ' << column << ':' << entry.row().value() << ':'
                << entry.coefficient();
    }
  }
  std::cout << "\nupper_entries " << upper_entries;

  DenseColumn right(RowIndex(n), 0.0);
  DenseRow left(ColIndex(n), 0.0);
  for (int i = 0; i < n; ++i) {
    right[RowIndex(i)] = i + 1;
    left[ColIndex(i)] = i + 1;
  }
  factorization.RightSolve(&right);
  factorization.LeftSolve(&left);
  std::cout << "\nright";
  for (int i = 0; i < n; ++i) std::cout << ' ' << right[RowIndex(i)];
  std::cout << "\nleft";
  for (int i = 0; i < n; ++i) std::cout << ' ' << left[ColIndex(i)];
  std::vector<int> inverse_column_permutation(n);
  std::vector<int> column_permutation(n);
  for (int i = 0; i < n; ++i) {
    inverse_column_permutation[i] =
        factorization.inverse_col_perm()[ColIndex(i)].value();
    column_permutation[i] =
        factorization.GetColumnPermutation()[ColIndex(i)].value();
  }
  // GLOP's hypersparse U kernels intentionally require Q to have been folded
  // into the basis mapping, as BasisFactorization does in production.
  factorization.SetColumnPermutationToIdentity();
  ScatteredColumn sparse_right;
  sparse_right.values.AssignToZero(RowIndex(n));
  sparse_right[RowIndex(n / 2)] = 1.0;
  sparse_right.non_zeros.push_back(RowIndex(n / 2));
  sparse_right.non_zeros_are_sorted = false;
  factorization.RightSolveLWithNonZeros(&sparse_right);
  factorization.RightSolveUWithNonZeros(&sparse_right);
  sparse_right.SortNonZerosIfNeeded();
  std::vector<double> sparse_right_values(n, 0.0);
  std::vector<int> sparse_right_positions;
  if (sparse_right.non_zeros.empty()) {
    for (int row = 0; row < n; ++row) {
      sparse_right_values[inverse_column_permutation[row]] =
          sparse_right[RowIndex(row)];
    }
  } else {
    for (const RowIndex row : sparse_right.non_zeros) {
      const int destination = inverse_column_permutation[row.value()];
      sparse_right_values[destination] = sparse_right[row];
      sparse_right_positions.push_back(destination);
    }
  }
  std::sort(sparse_right_positions.begin(), sparse_right_positions.end());
  std::cout << "\nsparse_right_positions";
  for (const int row : sparse_right_positions) std::cout << ' ' << row;
  std::cout << "\nsparse_right";
  for (const double value : sparse_right_values) std::cout << ' ' << value;
  ScatteredRow sparse_left;
  sparse_left.values.AssignToZero(ColIndex(n));
  const ColIndex sparse_left_input(column_permutation[n / 2]);
  sparse_left[sparse_left_input] = 1.0;
  sparse_left.non_zeros.push_back(sparse_left_input);
  sparse_left.non_zeros_are_sorted = false;
  factorization.LeftSolveUWithNonZeros(&sparse_left);
  factorization.LeftSolveLWithNonZeros(&sparse_left);
  sparse_left.SortNonZerosIfNeeded();
  std::cout << "\nsparse_left_positions";
  for (const ColIndex column : sparse_left.non_zeros) {
    std::cout << ' ' << column.value();
  }
  std::cout << "\nsparse_left";
  for (int i = 0; i < n; ++i) std::cout << ' ' << sparse_left[ColIndex(i)];
  std::cout << "\nstats_hex ";
  for (const unsigned char byte : factorization.StatString()) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(byte);
  }
  std::cout << '\n';
  return 0;
}
