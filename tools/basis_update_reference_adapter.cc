// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>
#include <ranges>
#include <vector>

#define private public
#include "ortools/glop/basis_representation.h"
#undef private
#include "ortools/glop/parameters.pb.h"
#include "ortools/lp_data/permutation.h"
#include "ortools/lp_data/scattered_vector.h"
#include "ortools/lp_data/sparse.h"

int main() {
  using namespace operations_research::glop;
  int n;
  int structural_columns;
  int entries;
  int updates;
  int refactorization_period;
  int dynamically_adjust;
  int use_middle_product;
  if (!(std::cin >> n >> structural_columns >> entries >> updates >>
        refactorization_period >> dynamically_adjust >> use_middle_product))
    return 2;
  SparseMatrix matrix;
  matrix.PopulateFromZero(RowIndex(n), ColIndex(structural_columns + n));
  for (int i = 0; i < entries; ++i) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    matrix.mutable_column(ColIndex(column))->SetCoefficient(RowIndex(row), value);
  }
  for (int row = 0; row < n; ++row) {
    matrix.mutable_column(ColIndex(structural_columns + row))
        ->SetCoefficient(RowIndex(row), 1.0);
  }
  matrix.CleanUp();
  CompactSparseMatrix compact(matrix);
  RowToColMapping basis(RowIndex(n), ColIndex(0));
  RowToColMapping temporary_basis;
  for (int row = 0; row < n; ++row) {
    basis[RowIndex(row)] = ColIndex(structural_columns + row);
  }
  BasisFactorization factorization(&compact, &basis);
  GlopParameters parameters;
  parameters.set_use_middle_product_form_update(use_middle_product != 0);
  parameters.set_basis_refactorization_period(refactorization_period);
  parameters.set_dynamically_adjust_refactorization_period(dynamically_adjust != 0);
  factorization.SetParameters(parameters);
  if (!factorization.Initialize().ok()) return 3;

  std::cout << std::setprecision(17);
  for (int stage = 0; stage < updates; ++stage) {
    int entering;
    int leaving;
    std::cin >> entering >> leaving;
    ScatteredColumn direction;
    factorization.RightSolveForProblemColumn(ColIndex(entering), &direction);
    ScatteredRow unit_left_inverse;
    factorization.LeftSolveForUnitRow(ColIndex(leaving), &unit_left_inverse);
    // RevisedSimplex updates the externally owned basis mapping before it
    // calls BasisFactorization::Update(). A scheduled refactorization must
    // therefore see the newly installed basis column.
    basis[RowIndex(leaving)] = ColIndex(entering);
    const Status status =
        factorization.Update(ColIndex(entering), RowIndex(leaving), direction);
    if (!status.ok()) {
      std::cout << "stage" << stage << "_error\n";
      return 0;
    }
    if (factorization.IsRefactorized()) {
      ApplyColumnPermutationToRowIndexedVector(
          factorization.GetColumnPermutation().const_view(), &basis,
          &temporary_basis);
      factorization.SetColumnPermutationToIdentity();
    }
    ScatteredColumn right;
    right.values.AssignToZero(RowIndex(n));
    ScatteredRow left;
    left.values.AssignToZero(ColIndex(n));
    for (int i = 0; i < n; ++i) {
      right[RowIndex(i)] = i + 1;
      left[ColIndex(i)] = i + 1;
    }
    factorization.RightSolve(&right);
    factorization.LeftSolve(&left);
    std::cout << "stage" << stage << "_right";
    for (int i = 0; i < n; ++i) std::cout << ' ' << right[RowIndex(i)];
    std::cout << "\nstage" << stage << "_left";
    for (int i = 0; i < n; ++i) std::cout << ' ' << left[ColIndex(i)];
    std::cout << "\nstage" << stage << "_updates " << factorization.NumUpdates();
    std::cout << "\nstage" << stage << "_refactorized "
              << factorization.IsRefactorized() << '\n';
    std::cout << "stage" << stage << "_deterministic_time "
              << factorization.DeterministicTime() << '\n';
    std::cout << "stage" << stage << "_update_entries "
              << factorization.rank_one_factorization_.num_entries().value()
              << '\n';
    if (!factorization.rank_one_factorization_.elementary_matrices_.empty()) {
      const auto& last = factorization.rank_one_factorization_
                             .elementary_matrices_.back();
      const auto& u = last.storage_->column(last.u_index_);
      std::cout << "stage" << stage << "_last_entries "
                << u.num_entries().value() << ' '
                << last.storage_->column(last.v_index_).num_entries().value()
                << '\n';
    }
  }
  if (!factorization.ForceRefactorization().ok()) return 4;
  std::cout << "after_force_time " << factorization.DeterministicTime() << '\n';
  if (!factorization.Refactorize().ok()) return 5;
  std::cout << "after_noop_refactorize_time "
            << factorization.DeterministicTime() << '\n';
  std::cout << "special_right_norms";
  for (int column = 0; column < structural_columns; ++column) {
    std::cout << ' ' << factorization.RightSolveSquaredNorm(
                              compact.column(ColIndex(column)));
  }
  std::cout << "\nspecial_dual_norms";
  for (int row = 0; row < n; ++row) {
    std::cout << ' '
              << factorization.DualEdgeSquaredNorm(RowIndex(row));
  }
  std::cout << "\ntemporary_unit_rows";
  std::vector<int> temporary_unit_nnz;
  for (int row = 0; row < n; ++row) {
    ScatteredRow temporary;
    factorization.TemporaryLeftSolveForUnitRow(ColIndex(row), &temporary);
    temporary_unit_nnz.push_back(temporary.NumNonZerosEstimate());
    for (int column = 0; column < n; ++column) {
      std::cout << ' ' << temporary[ColIndex(column)];
    }
  }
  std::cout << '\n';
  std::cout << "temporary_unit_nnz";
  for (const int count : temporary_unit_nnz) std::cout << ' ' << count;
  std::cout << '\n';
  ApplyColumnPermutationToRowIndexedVector(
      factorization.GetColumnPermutation().const_view(), &basis,
      &temporary_basis);
  factorization.SetColumnPermutationToIdentity();
  std::cout << "one_norm " << factorization.ComputeOneNorm();
  std::cout << "\ninfinity_norm " << factorization.ComputeInfinityNorm();
  std::cout << "\ninverse_one_norm " << factorization.ComputeInverseOneNorm();
  std::cout << "\ninverse_infinity_norm "
            << factorization.ComputeInverseInfinityNorm();
  std::cout << "\none_condition "
            << factorization.ComputeOneNormConditionNumber();
  std::cout << "\ninfinity_condition "
            << factorization.ComputeInfinityNormConditionNumber();
  std::cout << "\ninfinity_condition_bound "
            << factorization.ComputeInfinityNormConditionNumberUpperBound();
  std::cout << "\nlu_entries " << factorization.NumberOfEntriesInLU().value();
  std::cout << "\nfinal_deterministic_time "
            << factorization.DeterministicTime() << '\n';
  std::cout << "stats_hex ";
  for (const unsigned char byte : factorization.StatString()) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(byte);
  }
  std::cout << std::dec << std::setfill(' ') << '\n';
  const double before_clear_time = factorization.DeterministicTime();
  factorization.Clear();
  ScatteredColumn cleared_right;
  cleared_right.values.AssignToZero(RowIndex(n));
  for (int i = 0; i < n; ++i) cleared_right[RowIndex(i)] = i + 1;
  factorization.RightSolve(&cleared_right);
  std::cout << "cleared_right";
  for (int i = 0; i < n; ++i) {
    std::cout << ' ' << cleared_right[RowIndex(i)];
  }
  std::cout << "\nclear_preserved_time " << before_clear_time << ' '
            << factorization.DeterministicTime() << '\n';
  return 0;
}
