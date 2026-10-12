// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

// Time GLOP sparse FTRAN/BTRAN on preloaded Matrix Market basis/RHS probes.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ortools/glop/lu_factorization.h"
#include "ortools/lp_data/sparse.h"

namespace {
using namespace operations_research::glop;

struct MatrixMarket {
  int rows;
  int columns;
  std::vector<std::vector<std::pair<int, double>>> grouped;
  size_t entries;
};

MatrixMarket ReadMatrixMarket(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error(path + ": cannot open file");
  std::string line;
  if (!std::getline(input, line) ||
      line != "%%MatrixMarket matrix coordinate real general") {
    throw std::runtime_error(path + ": unsupported Matrix Market header");
  }
  do {
    if (!std::getline(input, line)) {
      throw std::runtime_error(path + ": missing dimensions");
    }
  } while (line.empty() || line[0] == '%');
  int rows;
  int columns;
  size_t entries;
  std::istringstream dimensions(line);
  std::string extra;
  if (!(dimensions >> rows >> columns >> entries) || dimensions >> extra ||
      rows <= 0 || columns <= 0) {
    throw std::runtime_error(path + ": invalid dimensions");
  }
  MatrixMarket result{rows, columns,
                      std::vector<std::vector<std::pair<int, double>>>(columns),
                      entries};
  std::vector<std::unordered_set<int>> seen(columns);
  size_t count = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '%') continue;
    std::istringstream entry(line);
    int row;
    int column;
    double value;
    if (!(entry >> row >> column >> value) || entry >> extra ||
        row <= 0 || row > rows || column <= 0 || column > columns ||
        !std::isfinite(value) || count == entries ||
        !seen[column - 1].insert(row - 1).second) {
      throw std::runtime_error(path + ": invalid or duplicate coordinate");
    }
    result.grouped[column - 1].emplace_back(row - 1, value);
    ++count;
  }
  if (count != entries) throw std::runtime_error(path + ": entry count mismatch");
  return result;
}

template <typename Vector, typename Index>
Vector Prepare(const std::vector<std::pair<int, double>>& entries, int n) {
  Vector work;
  work.values.resize(Index(n), 0.0);
  for (const auto& [row, value] : entries) {
    work.values[Index(row)] = value;
    work.non_zeros.push_back(Index(row));
  }
  return work;
}

template <typename Vector, typename Index>
void Checksums(const Vector& work, int n, double* l1, double* linf,
               double* weighted) {
  *l1 = *linf = *weighted = 0.0;
  for (int row = 0; row < n; ++row) {
    const double value = work.values[Index(row)];
    *l1 += std::abs(value);
    *linf = std::max(*linf, std::abs(value));
    *weighted += value * static_cast<double>(row % 17 + 1);
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6) {
      throw std::runtime_error(
          "usage: netlib_solve_probe_reference_adapter REPETITIONS WARMUPS FACTORIZATION.mtx BASIS.mtx RHS.mtx...");
    }
    const int repetitions = std::stoi(argv[1]);
    const int warmups = std::stoi(argv[2]);
    if (repetitions <= 0 || warmups < 0) {
      throw std::runtime_error("repetitions must be positive and warmups nonnegative");
    }
    const MatrixMarket matrix = ReadMatrixMarket(argv[3]);
    const MatrixMarket basis = ReadMatrixMarket(argv[4]);
    if (matrix.rows != matrix.columns || basis.rows != matrix.rows ||
        basis.columns != matrix.columns) {
      throw std::runtime_error("basis/factorization dimension mismatch");
    }
    CompactSparseMatrix compact;
    compact.Reset(RowIndex(matrix.rows));
    for (const auto& column : matrix.grouped) {
      for (const auto& [row, value] : column) {
        compact.AddEntryToCurrentColumn(RowIndex(row), value);
      }
      compact.CloseCurrentColumn();
    }
    std::vector<ColIndex> columns;
    for (int column = 0; column < matrix.rows; ++column) {
      columns.push_back(ColIndex(column));
    }
    CompactSparseMatrixView view(&compact, &columns);
    LuFactorization factorization;
    factorization.SetParameters(GlopParameters());
    if (!factorization.ComputeFactorization(view).ok()) {
      throw std::runtime_error("basis factorization failed");
    }
    const auto& permutation = factorization.GetColumnPermutation();
    for (int source = 0; source < matrix.columns; ++source) {
      const int target = permutation.empty() ? source : permutation[ColIndex(source)].value();
      if (basis.grouped[target] != matrix.grouped[source]) {
        throw std::runtime_error(std::string(argv[4]) +
                                 ": not the factorization's permuted basis");
      }
    }
    factorization.SetColumnPermutationToIdentity();
    std::vector<std::string> rhs_paths;
    if (argc == 6 && std::string(argv[5]) == "-") {
      std::string path;
      while (std::getline(std::cin, path)) {
        if (!path.empty()) rhs_paths.push_back(path);
      }
    } else {
      for (int file = 5; file < argc; ++file) rhs_paths.emplace_back(argv[file]);
    }
    for (const std::string& path : rhs_paths) {
      const std::string name = std::filesystem::path(path).filename().string();
      const bool ftran = name.starts_with("ftran-");
      if (!ftran && !name.starts_with("btran-")) {
        throw std::runtime_error(path + ": expected ftran- or btran- filename");
      }
      const MatrixMarket rhs = ReadMatrixMarket(path);
      if (rhs.rows != basis.rows || rhs.columns != 1) {
        throw std::runtime_error(path + ": RHS dimensions do not match basis");
      }
      std::chrono::nanoseconds total(0);
      double l1 = 0.0;
      double linf = 0.0;
      double weighted = 0.0;
      for (int repetition = 0; repetition < warmups + repetitions; ++repetition) {
        if (ftran) {
          auto work = Prepare<ScatteredColumn, RowIndex>(rhs.grouped[0], rhs.rows);
          const auto start = std::chrono::steady_clock::now();
          factorization.RightSolveLWithNonZeros(&work);
          factorization.RightSolveUWithNonZeros(&work);
          const auto elapsed = std::chrono::steady_clock::now() - start;
          if (repetition >= warmups) {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
          }
          if (repetition + 1 == warmups + repetitions) {
            Checksums<ScatteredColumn, RowIndex>(work, rhs.rows, &l1, &linf,
                                                   &weighted);
          }
        } else {
          auto work = Prepare<ScatteredRow, ColIndex>(rhs.grouped[0], rhs.rows);
          const auto start = std::chrono::steady_clock::now();
          factorization.LeftSolveUWithNonZeros(&work);
          factorization.LeftSolveLWithNonZeros(&work);
          const auto elapsed = std::chrono::steady_clock::now() - start;
          if (repetition >= warmups) {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
          }
          if (repetition + 1 == warmups + repetitions) {
            Checksums<ScatteredRow, ColIndex>(work, rhs.rows, &l1, &linf,
                                               &weighted);
          }
        }
      }
      std::cout << path << '\t' << (ftran ? "ftran" : "btran") << '\t'
                << rhs.rows << '\t' << rhs.entries << '\t' << repetitions
                << '\t' << total.count() << '\t' << std::scientific
                << std::setprecision(17) << l1 << '\t' << linf << '\t'
                << weighted << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
