// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse_column.h"

namespace {

using operations_research::glop::DenseColumn;
using operations_research::glop::ColumnView;
using operations_research::glop::RandomAccessSparseColumn;
using operations_research::glop::RowIndex;
using operations_research::glop::RowPermutation;
using operations_research::glop::SparseColumn;

void Print(const char* name, const SparseColumn& vector) {
  std::cout << name << ' ' << vector.num_entries().value();
  for (const auto entry : vector) {
    std::cout << ' ' << entry.row().value() << ' ' << entry.coefficient();
  }
  std::cout << '\n';
}

void PrintView(const char* name, const ColumnView view) {
  std::cout << name << ' ' << view.num_entries().value();
  for (const auto entry : view) {
    std::cout << ' ' << entry.row().value() << ' ' << entry.coefficient();
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  int n;
  int count;
  if (!(std::cin >> n >> count)) return 2;
  SparseColumn input;
  for (int entry = 0; entry < count; ++entry) {
    int row;
    double value;
    std::cin >> row >> value;
    input.SetCoefficient(RowIndex(row), value);
  }
  DenseColumn weights(RowIndex(n), 0.0);
  for (int row = 0; row < n; ++row) std::cin >> weights[RowIndex(row)];
  double threshold;
  std::cin >> threshold;
  RowPermutation partial{RowIndex(n)};
  RowPermutation tags{RowIndex(n)};
  for (int row = 0; row < n; ++row) {
    int destination;
    std::cin >> destination;
    partial[RowIndex(row)] = RowIndex(destination);
  }
  for (int row = 0; row < n; ++row) {
    int tag;
    std::cin >> tag;
    tags[RowIndex(row)] = RowIndex(tag);
  }

  std::cout << std::setprecision(17);
  SparseColumn clean;
  clean.PopulateFromSparseVector(input);
  clean.CleanUp();
  Print("clean", clean);
  PrintView("view", ColumnView(clean));

  RandomAccessSparseColumn random_access{RowIndex(n)};
  random_access.PopulateFromSparseColumn(clean);
  random_access.AddToCoefficient(RowIndex(0), 1.25);
  random_access.SetCoefficient(RowIndex(n - 1), -0.5);
  SparseColumn random_access_output;
  random_access.PopulateSparseColumn(&random_access_output);
  Print("random", random_access_output);

  SparseColumn near = clean;
  near.RemoveNearZeroEntries(threshold);
  Print("near", near);

  SparseColumn weighted = clean;
  weighted.RemoveNearZeroEntriesWithWeights(threshold, weights);
  Print("weighted", weighted);

  SparseColumn permuted = clean;
  permuted.ApplyPartialIndexPermutation(partial);
  Print("partial", permuted);

  SparseColumn retained = clean;
  SparseColumn moved;
  retained.MoveTaggedEntriesTo(tags, &moved);
  Print("retained", retained);
  Print("moved", moved);
  return 0;
}
