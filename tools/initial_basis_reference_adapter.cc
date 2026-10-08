// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Differential adapter for GLOP's high-level initial-basis crash procedures.

#include <iostream>

#include "ortools/glop/initial_basis.h"
#include "ortools/lp_data/lp_types.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int mode;
  int rows;
  int columns;
  int entries;
  int candidate_columns;
  if (!(std::cin >> mode >> rows >> columns >> entries >> candidate_columns)) {
    return 2;
  }
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(rows), ColIndex(columns));
  for (int entry = 0; entry < entries; ++entry) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row),
                                                            value);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  DenseRow objective(ColIndex(columns), 0.0);
  DenseRow lower(ColIndex(columns), 0.0);
  DenseRow upper(ColIndex(columns), 0.0);
  VariableTypeRow types(ColIndex(columns), VariableType::UNCONSTRAINED);
  for (int column = 0; column < columns; ++column) {
    int type;
    std::cin >> objective[ColIndex(column)] >> lower[ColIndex(column)] >>
        upper[ColIndex(column)] >> type;
    types[ColIndex(column)] = static_cast<VariableType>(type);
  }
  RowToColMapping basis(RowIndex(rows), kInvalidCol);
  for (int row = 0; row < rows; ++row) {
    int column;
    std::cin >> column;
    basis[RowIndex(row)] = ColIndex(column);
  }
  InitialBasis crash(compact, objective, lower, upper, types);
  switch (mode) {
    case 0:
      crash.CompleteBixbyBasis(ColIndex(candidate_columns), &basis);
      break;
    case 1:
      crash.CompleteTriangularPrimalBasis(ColIndex(candidate_columns), &basis);
      break;
    case 2:
      crash.CompleteTriangularDualBasis(ColIndex(candidate_columns), &basis);
      break;
    case 3:
      crash.GetPrimalMarosBasis(ColIndex(candidate_columns), &basis);
      break;
    case 4:
      crash.GetDualMarosBasis(ColIndex(candidate_columns), &basis);
      break;
    default:
      return 3;
  }
  std::cout << "basis";
  for (RowIndex row(0); row < basis.size(); ++row) {
    std::cout << ' ' << basis[row].value();
  }
  std::cout << '\n';
  return 0;
}
