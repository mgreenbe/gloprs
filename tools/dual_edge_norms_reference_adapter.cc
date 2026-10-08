// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "ortools/glop/basis_representation.h"
#include "ortools/glop/dual_edge_norms.h"
#include "ortools/glop/parameters.pb.h"
#include "ortools/lp_data/scattered_vector.h"
#include "ortools/lp_data/sparse.h"
#include "ortools/util/time_limit.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int leaving;
  int off_diagonal_entries;
  double deterministic_limit;
  if (!(std::cin >> n >> leaving >> off_diagonal_entries >> deterministic_limit)) return 2;

  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(n), ColIndex(n + 1));
  for (int row = 0; row < n; ++row) {
    double value;
    std::cin >> value;
    matrix.mutable_column(ColIndex(row))->SetCoefficient(RowIndex(row), value);
  }
  for (int i = 0; i < off_diagonal_entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  for (int row = 0; row < n; ++row) {
    double value;
    std::cin >> value;
    if (value != 0.0) {
      matrix.mutable_column(ColIndex(n))->SetCoefficient(RowIndex(row), value);
    }
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  RowToColMapping basis(RowIndex(n), ColIndex(0));
  for (int row = 0; row < n; ++row) basis[RowIndex(row)] = ColIndex(row);

  BasisFactorization factorization(&compact, &basis);
  GlopParameters parameters;
  parameters.set_use_middle_product_form_update(true);
  parameters.set_basis_refactorization_period(10000);
  parameters.set_dynamically_adjust_refactorization_period(false);
  factorization.SetParameters(parameters);
  if (!factorization.Initialize().ok()) return 3;
  factorization.SetColumnPermutationToIdentity();
  std::unique_ptr<operations_research::TimeLimit> time_limit;
  DualEdgeNorms norms(factorization);
  norms.SetParameters(parameters);
  if (deterministic_limit >= 0.0) {
    time_limit = std::make_unique<operations_research::TimeLimit>(
        std::numeric_limits<double>::infinity(), deterministic_limit);
    norms.SetTimeLimit(time_limit.get());
  }
  const auto initial_view = norms.GetEdgeSquaredNorms();
  std::vector<double> initial(n);
  for (int row = 0; row < n; ++row) initial[row] = initial_view[RowIndex(row)];

  // Prime GLOP's adaptive tau cache. The following measured unit-row solve
  // must retain its pre-permutation intermediate for RightSolveForTau().
  ScatteredRow warm_left_inverse;
  factorization.LeftSolveForUnitRow(ColIndex((leaving + 1) % n),
                                    &warm_left_inverse);
  factorization.RightSolveForTau(TransposedView(warm_left_inverse));

  ScatteredColumn direction;
  factorization.RightSolveForProblemColumn(ColIndex(n), &direction);
  if (direction.non_zeros.empty()) {
    for (int row = 0; row < n; ++row) {
      if (direction[RowIndex(row)] != 0.0) {
        direction.non_zeros.push_back(RowIndex(row));
      }
    }
  }
  ScatteredRow left_inverse;
  factorization.LeftSolveForUnitRow(ColIndex(leaving), &left_inverse);
  const bool precise = norms.TestPrecision(RowIndex(leaving), left_inverse);
  norms.UpdateBeforeBasisPivot(ColIndex(n), RowIndex(leaving), direction,
                               left_inverse);
  const auto updated = norms.GetEdgeSquaredNorms();

  std::cout << std::setprecision(17);
  std::cout << "entries " << factorization.NumberOfEntriesInLU() << '\n';
  std::cout << "initial";
  for (int row = 0; row < n; ++row) std::cout << ' ' << initial[row];
  std::cout << "\ndirection";
  for (int row = 0; row < n; ++row) std::cout << ' ' << direction[RowIndex(row)];
  std::cout << "\nleft";
  for (int column = 0; column < n; ++column) {
    std::cout << ' ' << left_inverse[ColIndex(column)];
  }
  std::cout << "\nprecise " << precise;
  std::cout << "\nupdated";
  for (int row = 0; row < n; ++row) std::cout << ' ' << updated[RowIndex(row)];
  std::cout << "\nstats_hex ";
  for (const unsigned char byte : norms.StatString()) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(byte);
  }
  std::cout << '\n';
  return 0;
}
