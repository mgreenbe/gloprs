// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ortools/lp_data/lp_types.h"
#include "ortools/lp_data/sparse.h"

int main(int argc, char** argv) {
  using namespace operations_research::glop;
  if (argc != 4) return 2;
  const std::string mode = argv[2];
  const int repetitions = std::stoi(argv[3]);
  if (repetitions <= 0) return 2;
  std::ifstream input(argv[1]);
  int rows;
  int columns;
  if (!(input >> rows >> columns)) return 2;
  CompactSparseMatrix matrix;
  matrix.Reset(RowIndex(rows));
  for (int column = 0; column < columns; ++column) {
    int count;
    input >> count;
    for (int entry = 0; entry < count; ++entry) {
      int row;
      double coefficient;
      input >> row >> coefficient;
      matrix.AddEntryToCurrentColumn(RowIndex(row), coefficient);
    }
    matrix.CloseCurrentColumn();
  }
  DenseRow lhs(ColIndex(rows), 0.0);
  for (int row = 0; row < rows; ++row) input >> lhs[ColIndex(row)];
  int relevant_count;
  input >> relevant_count;
  DenseBitRow relevant{ColIndex(columns)};
  std::vector<ColIndex> relevant_positions;
  relevant_positions.reserve(relevant_count);
  for (int i = 0; i < relevant_count; ++i) {
    int column;
    input >> column;
    relevant.Set(ColIndex(column));
    relevant_positions.push_back(ColIndex(column));
  }
  if (!input || (mode != "dot" && mode != "output" && mode != "count" &&
                 mode != "coeffs" && mode != "positions" && mode != "full"))
    return 2;
  const auto view = matrix.view();
  const auto lhs_view = lhs.const_view();
  std::vector<double> precomputed(columns, 0.0);
  if (mode == "output") {
    for (const ColIndex column : relevant) {
      precomputed[column.value()] = view.ColumnScalarProduct(column, lhs_view);
    }
  }
  std::vector<double> coefficients(columns, 0.0);
  std::vector<ColIndex> positions;
  positions.reserve(columns);
  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  if (mode == "dot") {
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      for (const ColIndex column : relevant) {
        coefficients[column.value()] = view.ColumnScalarProduct(column, lhs_view);
      }
      std::atomic_signal_fence(std::memory_order_seq_cst);
      if (!relevant_positions.empty()) {
        checksum += coefficients[relevant_positions[repetition % relevant_count].value()];
      }
    }
  } else if (mode == "output") {
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      positions.clear();
      for (const ColIndex column : relevant) {
        const double coefficient = precomputed[column.value()];
        if (std::abs(coefficient) > 1e-14) {
          coefficients[column.value()] = coefficient;
          positions.push_back(column);
        }
      }
      checksum += positions.size();
      if (!positions.empty()) {
        checksum += coefficients[positions[repetition % positions.size()].value()];
      }
    }
  } else if (mode == "count") {
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      int count = 0;
      for (const ColIndex column : relevant) {
        const double coefficient = view.ColumnScalarProduct(column, lhs_view);
        count += std::abs(coefficient) > 1e-14;
      }
      checksum += count;
    }
  } else if (mode == "coeffs") {
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      int count = 0;
      for (const ColIndex column : relevant) {
        const double coefficient = view.ColumnScalarProduct(column, lhs_view);
        if (std::abs(coefficient) > 1e-14) {
          coefficients[column.value()] = coefficient;
          ++count;
        }
      }
      checksum += count;
      if (!relevant_positions.empty()) {
        checksum += coefficients[relevant_positions[repetition % relevant_count].value()];
      }
    }
  } else if (mode == "positions") {
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      positions.clear();
      for (const ColIndex column : relevant) {
        const double coefficient = view.ColumnScalarProduct(column, lhs_view);
        if (std::abs(coefficient) > 1e-14) positions.push_back(column);
      }
      checksum += positions.size();
    }
  } else {
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      positions.resize(columns);
      auto* nonzeros = positions.data();
      for (const ColIndex column : relevant) {
        const double coefficient = view.ColumnScalarProduct(column, lhs_view);
        if (std::abs(coefficient) > 1e-14) {
          coefficients[column.value()] = coefficient;
          *nonzeros++ = column;
        }
      }
      positions.resize(nonzeros - positions.data());
      checksum += positions.size();
      if (!positions.empty()) {
        checksum += coefficients[positions[repetition % positions.size()].value()];
      }
    }
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  std::cout << std::setprecision(17)
            << std::chrono::duration<double, std::micro>(elapsed).count() /
                   repetitions
            << ' ' << checksum << ' ' << relevant_count << ' '
            << matrix.num_entries().value() << '\n';
}
