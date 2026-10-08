// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Differential adapter for pinned GLOP SparseRow.

#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse_row.h"

namespace {
using namespace operations_research::glop;

void Emit(const char* label, const SparseRow& row) {
  std::cout << label << ' ' << row.num_entries().value();
  for (EntryIndex i(0); i < row.num_entries(); ++i) {
    std::cout << ' ' << row.EntryCol(i).value() << ' ' << std::hex
              << std::bit_cast<uint64_t>(row.EntryCoefficient(i)) << std::dec;
  }
  std::cout << " iter";
  for (const SparseRowEntry entry : row) {
    std::cout << ' ' << entry.col().value() << ' ' << std::hex
              << std::bit_cast<uint64_t>(entry.coefficient()) << std::dec;
  }
  std::cout << " ends " << row.GetFirstCol().value() << ' '
            << row.GetLastCol().value() << '\n';
}
}  // namespace

int main() {
  using namespace operations_research::glop;
  int n;
  int count;
  if (!(std::cin >> n >> count)) return 2;
  SparseRow input;
  for (int i = 0; i < count; ++i) {
    int column;
    double coefficient;
    std::cin >> column >> coefficient;
    input.SetCoefficient(ColIndex(column), coefficient);
  }
  ColumnPermutation complete{ColIndex(n)};
  ColumnPermutation partial{ColIndex(n)};
  for (int i = 0; i < n; ++i) {
    int destination;
    std::cin >> destination;
    complete[ColIndex(i)] = ColIndex(destination);
  }
  for (int i = 0; i < n; ++i) {
    int destination;
    std::cin >> destination;
    partial[ColIndex(i)] = ColIndex(destination);
  }

  std::cout << std::setprecision(17);
  Emit("input", input);
  SparseRow permuted = input;
  permuted.ApplyColPermutation(complete);
  Emit("complete", permuted);
  SparseRow retained = input;
  retained.ApplyPartialColPermutation(partial);
  // The generator always retains at least one input entry.
  Emit("partial", retained);

  RowMajorSparseMatrix matrix{RowIndex(2)};
  matrix[RowIndex(0)] = input;
  matrix[RowIndex(1)] = permuted;
  std::cout << "matrix " << matrix.size();
  for (size_t i = 0; i < matrix.size(); ++i) {
    const SparseRow& row = matrix[RowIndex(i)];
    std::cout << ' ' << row.num_entries().value();
    for (const SparseRowEntry entry : row) {
      std::cout << ' ' << entry.col().value() << ' ' << std::hex
                << std::bit_cast<uint64_t>(entry.coefficient()) << std::dec;
    }
  }
  std::cout << '\n';
  return 0;
}
