// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/sparse.h"

namespace {
using namespace operations_research::glop;

void ReadMatrix(int rows, int columns, int entries, SparseMatrix* matrix) {
  matrix->PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int entry = 0; entry < entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix->mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  matrix->CleanUp();
}

void Print(const char* name, const SparseMatrix& matrix) {
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
  int m;
  int k;
  int n;
  int a_entries;
  int b_entries;
  double alpha;
  double beta;
  if (!(std::cin >> m >> k >> n >> a_entries >> b_entries >> alpha >> beta)) return 2;
  SparseMatrix a;
  SparseMatrix b;
  ReadMatrix(m, k, a_entries, &a);
  ReadMatrix(k, n, b_entries, &b);
  std::cout << std::setprecision(17);
  SparseMatrix transpose;
  transpose.PopulateFromTranspose(a);
  Print("transpose", transpose);
  SparseMatrix product;
  product.PopulateFromProduct(a, b);
  Print("product", product);
  SparseMatrix combination;
  combination.PopulateFromLinearCombination(alpha, a, beta, a);
  Print("combination", combination);
  Fractional minimum;
  Fractional maximum;
  a.ComputeMinAndMaxMagnitudes(&minimum, &maximum);
  std::cout << "magnitudes " << minimum << ' ' << maximum << '\n';
  return 0;
}
