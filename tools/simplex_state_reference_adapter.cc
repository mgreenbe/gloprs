// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Differential adapter for reduced costs and primal variable values.

#include <iomanip>
#include <iostream>

#include "absl/random/random.h"
#include "ortools/glop/basis_representation.h"
#include "ortools/glop/dual_edge_norms.h"
#include "ortools/glop/pricing.h"
#include "ortools/glop/reduced_costs.h"
#include "ortools/glop/variable_values.h"
#include "ortools/glop/variables_info.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int rows;
  int structural;
  int entries;
  if (!(std::cin >> rows >> structural >> entries)) return 2;
  const int columns = structural + rows;
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
  for (int row = 0; row < rows; ++row) {
    matrix.mutable_column(ColIndex(structural + row))
        ->SetCoefficient(RowIndex(row), 1.0);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  DenseRow objective(ColIndex(columns), 0.0);
  DenseRow lower(ColIndex(columns), 0.0);
  DenseRow upper(ColIndex(columns), 0.0);
  for (int column = 0; column < columns; ++column) {
    std::cin >> objective[ColIndex(column)] >> lower[ColIndex(column)] >>
        upper[ColIndex(column)];
  }
  RowToColMapping basis(RowIndex(rows), ColIndex(0));
  for (int row = 0; row < rows; ++row) {
    basis[RowIndex(row)] = ColIndex(structural + row);
  }
  VariablesInfo variables(compact);
  variables.LoadBoundsAndReturnTrueIfUnchanged(lower, upper);
  variables.InitializeToDefaultStatus();
  for (int row = 0; row < rows; ++row) {
    variables.UpdateToBasicStatus(basis[RowIndex(row)]);
  }
  BasisFactorization factorization(&compact, &basis);
  if (!factorization.Initialize().ok()) return 3;
  absl::BitGen random;
  ReducedCosts reduced(compact, objective, basis, variables, factorization,
                       random);
  GlopParameters parameters;
  reduced.SetParameters(parameters);
  const auto reduced_values = reduced.GetReducedCosts();
  const auto dual_values = reduced.GetDualValues();

  DualEdgeNorms dual_norms(factorization);
  DynamicMaximum<RowIndex> dual_prices(random);
  VariableValues values(parameters, compact, basis, variables, factorization,
                        &dual_norms, &dual_prices);
  DenseRow empty;
  values.ResetAllNonBasicVariableValues(empty);
  values.RecomputeBasicVariableValues();

  std::cout << std::setprecision(17) << "reduced";
  for (int column = 0; column < columns; ++column) {
    std::cout << ' ' << reduced_values[ColIndex(column)];
  }
  std::cout << "\ndual";
  for (int row = 0; row < rows; ++row) {
    std::cout << ' ' << dual_values[RowIndex(row)];
  }
  std::cout << "\nvalues";
  for (int column = 0; column < columns; ++column) {
    std::cout << ' ' << values.Get(ColIndex(column));
  }
  std::cout << "\nresidual " << values.ComputeMaximumPrimalResidual();
  std::cout << "\ninfeasibility "
            << values.ComputeMaximumPrimalInfeasibility();
  std::cout << "\nsum_infeasibility "
            << values.ComputeSumOfPrimalInfeasibilities() << '\n';
  return 0;
}
