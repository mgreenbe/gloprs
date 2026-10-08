// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/scattered_vector.h"

namespace {
using operations_research::glop::RowIndex;
using operations_research::glop::ScatteredColumn;
using operations_research::glop::TransposedView;

void Print(const char* name, const ScatteredColumn& vector) {
  std::cout << name << " positions";
  for (const RowIndex row : vector.non_zeros) std::cout << ' ' << row.value();
  std::cout << " values";
  for (const double value : vector.values) std::cout << ' ' << value;
  std::cout << '\n';
}
}  // namespace

int main() {
  using namespace operations_research::glop;
  int size;
  int count;
  double ratio;
  if (!(std::cin >> size >> count >> ratio)) return 2;
  ScatteredColumn vector;
  vector.values.assign(RowIndex(size), 0.0);
  vector.is_non_zero.Resize(RowIndex(size));
  for (int i = 0; i < count; ++i) {
    int row;
    double value;
    std::cin >> row >> value;
    vector.Add(RowIndex(row), value);
  }
  std::cout << std::scientific << std::setprecision(17);
  Print("added", vector);
  std::cout << "metrics " << vector.ShouldUseDenseIteration(ratio) << ' '
            << vector.ShouldUseDenseIteration() << ' '
            << vector.NumNonZerosEstimate() << '\n';
  vector.SortNonZerosIfNeeded();
  Print("sorted", vector);

  if (size > 0) {
    vector.ClearSparseMask();
    vector.Add(RowIndex(0), 0.75);
    Print("mask_cleared_add", vector);
    vector.RepopulateSparseMask();
    vector.Add(RowIndex(0), 0.5);
    Print("mask_repopulated_add", vector);
  }

  const ScatteredRow& transpose = TransposedView(vector);
  std::cout << "transpose";
  for (const ColIndex column : transpose.non_zeros) {
    std::cout << ' ' << column.value() << ' ' << transpose.values[column];
  }
  std::cout << '\n';

  vector.ClearNonZerosIfTooDense(ratio);
  Print("density_switch", vector);
  std::cout << "estimate " << vector.NumNonZerosEstimate() << '\n';
  return 0;
}
