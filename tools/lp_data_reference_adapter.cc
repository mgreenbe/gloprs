// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/lp_data.h"

int main() {
  using namespace operations_research::glop;
  int rows;
  int columns;
  int entries;
  int maximize;
  double offset;
  double scale;
  double tolerance;
  if (!(std::cin >> rows >> columns >> entries >> maximize >> offset >> scale >>
        tolerance)) {
    return 2;
  }
  LinearProgram lp;
  for (int column = 0; column < columns; ++column) lp.CreateNewVariable();
  for (int row = 0; row < rows; ++row) lp.CreateNewConstraint();
  lp.SetMaximizationProblem(maximize != 0);
  lp.SetObjectiveOffset(offset);
  lp.SetObjectiveScalingFactor(scale);
  for (int column = 0; column < columns; ++column) {
    int type;
    double lower;
    double upper;
    double objective;
    std::cin >> type >> lower >> upper >> objective;
    lp.SetVariableType(ColIndex(column),
                       static_cast<LinearProgram::VariableType>(type));
    lp.SetVariableBounds(ColIndex(column), lower, upper);
    lp.SetObjectiveCoefficient(ColIndex(column), objective);
  }
  for (int row = 0; row < rows; ++row) {
    double lower;
    double upper;
    std::cin >> lower >> upper;
    lp.SetConstraintBounds(RowIndex(row), lower, upper);
  }
  for (int position = 0; position < entries; ++position) {
    int row;
    int column;
    double value;
    std::cin >> row >> column >> value;
    lp.SetCoefficient(RowIndex(row), ColIndex(column), value);
  }
  DenseRow solution(ColIndex(columns), 0.0);
  for (int column = 0; column < columns; ++column) {
    std::cin >> solution[ColIndex(column)];
  }
  lp.CleanUp();
  std::cout << std::setprecision(17);
  std::cout << "types";
  for (int column = 0; column < columns; ++column) {
    std::cout << ' ' << lp.IsVariableInteger(ColIndex(column)) << ' '
              << lp.IsVariableBinary(ColIndex(column));
  }
  std::cout << "\nfeasibility "
            << lp.SolutionIsWithinVariableBounds(solution, tolerance) << ' '
            << lp.SolutionIsLPFeasible(solution, tolerance) << ' '
            << lp.SolutionIsInteger(solution, tolerance) << ' '
            << lp.SolutionIsMIPFeasible(solution, tolerance);
  std::cout << "\nnames " << lp.GetVariableName(ColIndex(0)) << ' '
            << lp.GetVariableName(ColIndex(columns + 3)) << ' '
            << lp.GetConstraintName(RowIndex(0)) << ' '
            << lp.GetConstraintName(RowIndex(rows + 3));
  std::cout << "\nobjective " << lp.ApplyObjectiveScalingAndOffset(2.25) << ' '
            << lp.RemoveObjectiveScalingAndOffset(
                   lp.ApplyObjectiveScalingAndOffset(2.25));
  for (int column = 0; column < columns; ++column) {
    std::cout << ' ' << lp.GetObjectiveCoefficientForMinimizationVersion(
                             ColIndex(column));
  }
  std::cout << "\nintegral_bounds "
            << lp.BoundsOfIntegerVariablesAreInteger(tolerance) << ' '
            << lp.BoundsOfIntegerConstraintsAreInteger(tolerance) << '\n';

  lp.AddSlackVariablesWhereNecessary(true);
  solution.resize(lp.num_variables(), 0.0);
  lp.ComputeSlackVariableValues(&solution);
  std::cout << "slacks " << lp.GetFirstSlackVariable().value() << ' '
            << lp.num_variables().value() << ' ' << lp.num_entries().value()
            << ' ' << lp.IsInEquationForm();
  for (int row = 0; row < rows; ++row) {
    const ColIndex slack = lp.GetSlackVariable(RowIndex(row));
    std::cout << ' ' << slack.value() << ' ' << solution[slack] << ' '
              << lp.variable_lower_bounds()[slack] << ' '
              << lp.variable_upper_bounds()[slack] << ' '
              << lp.IsVariableInteger(slack);
  }
  std::cout << '\n';
  lp.DeleteSlackVariables();
  std::cout << "restored " << lp.num_variables().value() << ' '
            << lp.num_constraints().value() << ' ' << lp.num_entries().value();
  for (int row = 0; row < rows; ++row) {
    std::cout << ' ' << lp.constraint_lower_bounds()[RowIndex(row)] << ' '
              << lp.constraint_upper_bounds()[RowIndex(row)];
  }
  std::cout << '\n';

  SparseMatrix appended;
  appended.PopulateFromZero(RowIndex(2), lp.num_variables());
  for (int column = 0; column < columns; ++column) {
    if (column % 2 == 0) {
      appended.mutable_column(ColIndex(column))
          ->SetCoefficient(RowIndex(0), column + 0.5);
    }
    if (column % 3 == 1) {
      appended.mutable_column(ColIndex(column))
          ->SetCoefficient(RowIndex(1), 1.25 - column);
    }
  }
  DenseColumn appended_lower(RowIndex(2), 0.0);
  DenseColumn appended_upper(RowIndex(2), 0.0);
  appended_lower[RowIndex(0)] = -7.0;
  appended_lower[RowIndex(1)] = -3.0;
  appended_upper[RowIndex(0)] = 4.0;
  appended_upper[RowIndex(1)] = 9.0;
  StrictITIVector<RowIndex, std::string> appended_names(RowIndex(2));
  appended_names[RowIndex(0)] = "appended0";
  appended_names[RowIndex(1)] = "appended1";
  lp.AddConstraints(appended, appended_lower, appended_upper, appended_names);
  std::cout << "appended " << lp.num_constraints().value() << ' '
            << lp.num_entries().value() << ' ' << lp.IsCleanedUp();
  const SparseMatrix& appended_transpose = lp.GetTransposeSparseMatrix();
  for (int row = rows; row < rows + 2; ++row) {
    std::cout << ' ' << lp.constraint_lower_bounds()[RowIndex(row)] << ' '
              << lp.constraint_upper_bounds()[RowIndex(row)];
    for (const auto entry :
         appended_transpose.column(ColIndex(row))) {
      std::cout << ' ' << row << ':' << entry.row().value() << ':'
                << entry.coefficient();
    }
  }
  std::cout << '\n';

  LinearProgram dual;
  RowToColMapping duplicated_rows;
  dual.PopulateFromDual(lp, &duplicated_rows);
  std::cout << "dual " << dual.num_constraints().value() << ' '
            << dual.num_variables().value() << ' ' << dual.num_entries().value()
            << ' ' << dual.IsMaximizationProblem() << ' '
            << dual.objective_offset() << ' '
            << dual.objective_scaling_factor();
  for (int row = 0; row < duplicated_rows.size().value(); ++row) {
    std::cout << ' ' << duplicated_rows[RowIndex(row)].value();
  }
  for (int row = 0; row < dual.num_constraints().value(); ++row) {
    std::cout << ' ' << dual.constraint_lower_bounds()[RowIndex(row)] << ' '
              << dual.constraint_upper_bounds()[RowIndex(row)];
  }
  for (int column = 0; column < dual.num_variables().value(); ++column) {
    const ColIndex col(column);
    std::cout << ' ' << dual.variable_lower_bounds()[col] << ' '
              << dual.variable_upper_bounds()[col] << ' '
              << dual.objective_coefficients()[col];
    for (const auto entry : dual.GetSparseColumn(col)) {
      std::cout << ' ' << column << ':' << entry.row().value() << ':'
                << entry.coefficient();
    }
  }
  std::cout << '\n';

  std::cout << "scaling";
  for (const auto method : {
           GlopParameters::NO_COST_SCALING,
           GlopParameters::CONTAIN_ONE_COST_SCALING,
           GlopParameters::MEAN_COST_SCALING,
           GlopParameters::MEDIAN_COST_SCALING}) {
    LinearProgram scaled;
    scaled.PopulateFromLinearProgram(lp);
    for (int column = 0; column < scaled.num_variables().value(); ++column) {
      const ColIndex col(column);
      scaled.SetObjectiveCoefficient(
          col, 100.0 * scaled.objective_coefficients()[col]);
    }
    const double factor = scaled.ScaleObjective(method);
    std::cout << ' ' << factor << ' ' << scaled.objective_scaling_factor()
              << ' ' << scaled.objective_offset();
    for (int column = 0; column < scaled.num_variables().value(); ++column) {
      std::cout << ' ' << scaled.objective_coefficients()[ColIndex(column)];
    }
  }
  LinearProgram bound_scaled;
  bound_scaled.PopulateFromLinearProgram(lp);
  for (int column = 0; column < bound_scaled.num_variables().value(); ++column) {
    const ColIndex col(column);
    bound_scaled.SetVariableBounds(
        col, 100.0 * bound_scaled.variable_lower_bounds()[col],
        100.0 * bound_scaled.variable_upper_bounds()[col]);
  }
  for (int row = 0; row < bound_scaled.num_constraints().value(); ++row) {
    const RowIndex index(row);
    bound_scaled.SetConstraintBounds(
        index, 100.0 * bound_scaled.constraint_lower_bounds()[index],
        100.0 * bound_scaled.constraint_upper_bounds()[index]);
  }
  const double bound_factor = bound_scaled.ScaleBounds();
  std::cout << ' ' << bound_factor << ' '
            << bound_scaled.objective_scaling_factor() << ' '
            << bound_scaled.objective_offset() << ' '
            << bound_scaled.IsValid(1e100) << ' '
            << bound_scaled.IsValid(1.0) << '\n';

  RowPermutation row_permutation(lp.num_constraints());
  for (int row = 0; row < lp.num_constraints().value(); ++row) {
    row_permutation[RowIndex(row)] =
        RowIndex(lp.num_constraints().value() - 1 - row);
  }
  ColumnPermutation column_permutation(lp.num_variables());
  for (int column = 0; column < lp.num_variables().value(); ++column) {
    column_permutation[ColIndex(column)] =
        ColIndex((column + 1) % lp.num_variables().value());
  }
  LinearProgram permuted;
  permuted.PopulateFromPermutedLinearProgram(
      lp, row_permutation, column_permutation);
  std::cout << "permuted " << permuted.num_constraints().value() << ' '
            << permuted.num_variables().value() << ' '
            << permuted.num_entries().value();
  for (int column = 0; column < permuted.num_variables().value(); ++column) {
    const ColIndex col(column);
    std::cout << ' ' << permuted.variable_lower_bounds()[col] << ' '
              << permuted.variable_upper_bounds()[col] << ' '
              << permuted.objective_coefficients()[col] << ' '
              << static_cast<int>(permuted.GetVariableType(col));
    for (const auto entry : permuted.GetSparseColumn(col)) {
      std::cout << ' ' << column << ':' << entry.row().value() << ':'
                << entry.coefficient();
    }
  }
  std::cout << '\n';
  LinearProgram variables_only;
  variables_only.PopulateFromLinearProgramVariables(permuted);
  std::cout << "variables_only " << variables_only.num_constraints().value()
            << ' ' << variables_only.num_variables().value() << ' '
            << variables_only.num_entries().value() << ' '
            << variables_only.objective_scaling_factor() << '\n';

  DenseBooleanRow deleted_columns(ColIndex(columns), false);
  for (int column = 1; column < columns; column += 3) {
    deleted_columns[ColIndex(column)] = true;
  }
  DenseBooleanColumn deleted_rows(RowIndex(rows), false);
  for (int row = 2; row < rows; row += 4) deleted_rows[RowIndex(row)] = true;
  lp.DeleteColumns(deleted_columns);
  lp.DeleteRows(deleted_rows);
  std::cout << "deleted " << lp.num_variables().value() << ' '
            << lp.num_constraints().value() << ' ' << lp.num_entries().value();
  for (int column = 0; column < lp.num_variables(); ++column) {
    std::cout << ' ' << lp.variable_lower_bounds()[ColIndex(column)] << ' '
              << lp.variable_upper_bounds()[ColIndex(column)] << ' '
              << lp.objective_coefficients()[ColIndex(column)] << ' '
              << static_cast<int>(lp.GetVariableType(ColIndex(column)));
    for (const auto entry : lp.GetSparseColumn(ColIndex(column))) {
      std::cout << ' ' << column << ':' << entry.row().value() << ':'
                << entry.coefficient();
    }
  }
  std::cout << '\n';
  return 0;
}
