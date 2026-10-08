// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "ortools/glop/rank_one_update.h"

namespace {
using namespace operations_research::glop;

template <typename Vector>
void PrintDense(const char* name, const Vector& values) {
  std::cout << name;
  for (const double value : values) std::cout << ' ' << value;
  std::cout << '\n';
}

template <typename Scattered>
void PrintScattered(const char* name, const Scattered& values) {
  std::cout << name << " positions";
  for (const auto position : values.non_zeros) {
    std::cout << ' ' << position.value();
  }
  std::cout << " values";
  for (const double value : values.values) std::cout << ' ' << value;
  std::cout << '\n';
}
}  // namespace

int main() {
  using namespace operations_research::glop;
  int dimension;
  int num_updates;
  double ratio;
  if (!(std::cin >> dimension >> num_updates >> ratio)) return 2;

  CompactSparseMatrix storage;
  storage.Reset(RowIndex(dimension));
  std::vector<std::tuple<ColIndex, ColIndex, double>> updates;
  for (int update = 0; update < num_updates; ++update) {
    int u_count;
    int v_count;
    double u_dot_v;
    std::cin >> u_count >> v_count >> u_dot_v;
    const ColIndex u_index = storage.num_cols();
    for (int entry = 0; entry < u_count; ++entry) {
      int row;
      double value;
      std::cin >> row >> value;
      storage.AddEntryToCurrentColumn(RowIndex(row), value);
    }
    storage.CloseCurrentColumn();
    const ColIndex v_index = storage.num_cols();
    for (int entry = 0; entry < v_count; ++entry) {
      int row;
      double value;
      std::cin >> row >> value;
      storage.AddEntryToCurrentColumn(RowIndex(row), value);
    }
    storage.CloseCurrentColumn();
    updates.emplace_back(u_index, v_index, u_dot_v);
  }

  DenseColumn rhs{RowIndex(dimension)};
  for (RowIndex row(0); row < rhs.size(); ++row) std::cin >> rhs[row];
  int pattern_size;
  std::cin >> pattern_size;
  std::vector<int> pattern(pattern_size);
  for (int& position : pattern) std::cin >> position;

  std::cout << std::scientific << std::setprecision(17);
  if (!updates.empty()) {
    const auto [u, v, dot] = updates.front();
    RankOneUpdateElementaryMatrix elementary(&storage, u, v, dot);
    DenseColumn right = rhs;
    elementary.RightMultiply(&right);
    PrintDense("elementary_right_multiply", right);
    elementary.RightSolve(&right);
    PrintDense("elementary_right_restore", right);
    DenseRow left(rhs.begin(), rhs.end());
    elementary.LeftMultiply(&left);
    PrintDense("elementary_left_multiply", left);
    elementary.LeftSolve(&left);
    PrintDense("elementary_left_restore", left);
    std::cout << "elementary " << elementary.IsSingular() << ' '
              << elementary.num_entries().value() << '\n';
  }

  RankOneUpdateFactorization factorization;
  factorization.set_hypersparse_ratio(ratio);
  for (const auto [u, v, dot] : updates) {
    factorization.Update(RankOneUpdateElementaryMatrix(&storage, u, v, dot));
  }
  DenseColumn right = rhs;
  factorization.RightSolve(&right);
  PrintDense("dense_right", right);
  DenseRow left(rhs.begin(), rhs.end());
  factorization.LeftSolve(&left);
  PrintDense("dense_left", left);

  ScatteredColumn sparse_right;
  sparse_right.values = rhs;
  sparse_right.is_non_zero.Resize(RowIndex(dimension));
  for (const int position : pattern) {
    sparse_right.non_zeros.push_back(RowIndex(position));
  }
  factorization.RightSolveWithNonZeros(&sparse_right);
  PrintScattered("sparse_right", sparse_right);

  ScatteredRow sparse_left;
  sparse_left.values.resize(ColIndex(dimension), 0.0);
  for (int position = 0; position < dimension; ++position) {
    sparse_left.values[ColIndex(position)] = rhs[RowIndex(position)];
  }
  sparse_left.is_non_zero.Resize(ColIndex(dimension));
  for (const int position : pattern) {
    sparse_left.non_zeros.push_back(ColIndex(position));
  }
  factorization.LeftSolveWithNonZeros(&sparse_left);
  PrintScattered("sparse_left", sparse_left);
  std::cout << "factor " << factorization.num_entries().value() << ' '
            << std::bit_cast<uint64_t>(factorization.DeterministicTimeSinceLastReset())
            << '\n';
  factorization.Clear();
  std::cout << "cleared " << factorization.num_entries().value() << '\n';
  factorization.ResetDeterministicTime();
  std::cout << "reset "
            << std::bit_cast<uint64_t>(factorization.DeterministicTimeSinceLastReset())
            << '\n';
  return 0;
}
