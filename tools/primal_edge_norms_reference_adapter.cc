// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "ortools/glop/basis_representation.h"
#include "ortools/glop/parameters.pb.h"
#include "ortools/glop/primal_edge_norms.h"
#include "ortools/glop/update_row.h"
#include "ortools/glop/variables_info.h"
#include "ortools/lp_data/sparse.h"
#include "ortools/util/time_limit.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int nonbasic_columns;
  int leaving;
  int off_diagonal_entries;
  double deterministic_limit;
  if (!(std::cin >> n >> nonbasic_columns >> leaving >> off_diagonal_entries >>
        deterministic_limit))
    return 2;
  const int columns = n + nonbasic_columns;

  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(n), ColIndex(columns));
  std::vector<double> diagonal(n);
  for (double& value : diagonal) std::cin >> value;
  std::vector<int> off_rows(off_diagonal_entries);
  std::vector<int> off_columns(off_diagonal_entries);
  std::vector<double> off_values(off_diagonal_entries);
  for (int entry = 0; entry < off_diagonal_entries; ++entry) {
    std::cin >> off_rows[entry] >> off_columns[entry] >> off_values[entry];
  }
  for (int column = 0; column < nonbasic_columns; ++column) {
    for (int row = 0; row < n; ++row) {
      double value;
      std::cin >> value;
      if (value != 0.0) {
        matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row),
                                                                value);
      }
    }
  }
  for (int row = 0; row < n; ++row) {
    matrix.mutable_column(ColIndex(nonbasic_columns + row))
        ->SetCoefficient(RowIndex(row), diagonal[row]);
  }
  for (int entry = 0; entry < off_diagonal_entries; ++entry) {
    matrix.mutable_column(ColIndex(nonbasic_columns + off_columns[entry]))
        ->SetCoefficient(RowIndex(off_rows[entry]), off_values[entry]);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  CompactSparseMatrix transpose;
  transpose.PopulateFromTranspose(compact);

  RowToColMapping basis(RowIndex(n), ColIndex(0));
  for (int row = 0; row < n; ++row) {
    basis[RowIndex(row)] = ColIndex(nonbasic_columns + row);
  }
  BasisFactorization factorization(&compact, &basis);
  GlopParameters parameters;
  parameters.set_use_middle_product_form_update(true);
  parameters.set_basis_refactorization_period(10000);
  parameters.set_dynamically_adjust_refactorization_period(false);
  factorization.SetParameters(parameters);
  if (!factorization.Initialize().ok()) return 3;
  RowToColMapping permuted_basis;
  ApplyColumnPermutationToRowIndexedVector(
      factorization.GetColumnPermutation().const_view(), &basis,
      &permuted_basis);
  factorization.SetColumnPermutationToIdentity();

  DenseRow lower(ColIndex(columns), 0.0);
  DenseRow upper(ColIndex(columns), kInfinity);
  VariablesInfo variables(compact);
  variables.LoadBoundsAndReturnTrueIfUnchanged(lower, upper);
  variables.InitializeToDefaultStatus();
  variables.ChangeUnusedBasicVariablesToFree(basis);
  UpdateRow update(compact, transpose, variables, basis, factorization);

  std::unique_ptr<operations_research::TimeLimit> time_limit;
  PrimalEdgeNorms norms(compact, variables, factorization);
  norms.SetParameters(parameters);
  if (deterministic_limit >= 0.0) {
    time_limit = std::make_unique<operations_research::TimeLimit>(
        std::numeric_limits<double>::infinity(), deterministic_limit);
    norms.SetTimeLimit(time_limit.get());
  }
  bool watcher = false;
  norms.AddRecomputationWatcher(&watcher);
  const DenseRow& matrix_norms = norms.GetMatrixColumnNorms();
  const DenseRow& edge_view = norms.GetEdgeSquaredNorms();
  const DenseRow& devex_view = norms.GetDevexWeights();
  std::vector<double> edges(columns);
  std::vector<double> devex(columns);
  for (int column = 0; column < columns; ++column) {
    edges[column] = edge_view[ColIndex(column)];
    devex[column] = devex_view[ColIndex(column)];
  }

  const ColIndex entering(0);
  ScatteredColumn direction;
  factorization.RightSolveForProblemColumn(entering, &direction);
  const bool precise = norms.TestEnteringEdgeNormPrecision(entering, direction);
  norms.UpdateBeforeBasisPivot(entering,
                               basis[RowIndex(leaving)],
                               RowIndex(leaving), direction, &update);

  const auto updated_edges = norms.GetEdgeSquaredNorms();
  const auto updated_devex = norms.GetDevexWeights();
  watcher = false;
  parameters.set_recompute_edges_norm_threshold(-1.0);
  norms.SetParameters(parameters);
  norms.TestEnteringEdgeNormPrecision(entering, direction);
  const bool precision_watcher = watcher;
  watcher = false;
  norms.Clear();
  const bool clear_watcher = watcher;

  std::cout << std::setprecision(17);
  std::cout << "entries " << factorization.NumberOfEntriesInLU() << '\n';
  std::cout << "matrix";
  for (int column = 0; column < columns; ++column) {
    std::cout << ' ' << matrix_norms[ColIndex(column)];
  }
  std::cout << "\nedges";
  for (double value : edges) std::cout << ' ' << value;
  std::cout << "\ndevex";
  for (double value : devex) std::cout << ' ' << value;
  std::cout << "\ndirection";
  for (int row = 0; row < n; ++row) std::cout << ' ' << direction[RowIndex(row)];
  std::cout << "\nprecise " << precise;
  std::cout << "\nupdated_edges";
  for (double value : updated_edges) std::cout << ' ' << value;
  std::cout << "\nupdated_devex";
  for (double value : updated_devex) std::cout << ' ' << value;
  std::cout << "\nprecision_watcher " << precision_watcher;
  std::cout << "\nclear_watcher " << clear_watcher;
  std::cout << "\ndeterministic_time " << norms.DeterministicTime();
  std::cout << "\nstats_hex ";
  for (const unsigned char byte : norms.StatString()) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(byte);
  }
  std::cout << '\n';
  return 0;
}
