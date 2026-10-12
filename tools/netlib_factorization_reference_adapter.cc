// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

// Time GLOP LU construction on already-loaded Matrix Market probe matrices.
#include <chrono>
#include <cmath>
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

using operations_research::glop::ColIndex;
using operations_research::glop::CompactSparseMatrix;
using operations_research::glop::RowIndex;

struct LoadedMatrix {
  CompactSparseMatrix matrix;
  int n;
  size_t entries;
};

void ReadMatrixMarket(const std::string& path, LoadedMatrix* result) {
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
  if (rows != columns) throw std::runtime_error(path + ": not square");
  std::vector<std::vector<std::pair<int, double>>> grouped(columns);
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
    grouped[column - 1].emplace_back(row - 1, value);
    ++count;
  }
  if (count != entries) throw std::runtime_error(path + ": entry count mismatch");
  result->n = rows;
  result->entries = entries;
  result->matrix.Reset(RowIndex(rows));
  for (const auto& column : grouped) {
    for (const auto& [row, value] : column) {
      result->matrix.AddEntryToCurrentColumn(RowIndex(row), value);
    }
    result->matrix.CloseCurrentColumn();
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace operations_research::glop;
  try {
    if (argc < 4) {
      throw std::runtime_error(
          "usage: netlib_factorization_reference_adapter REPETITIONS WARMUPS FILE.mtx...");
    }
    const int repetitions = std::stoi(argv[1]);
    const int warmups = std::stoi(argv[2]);
    if (repetitions <= 0 || warmups < 0) {
      throw std::runtime_error("repetitions must be positive and warmups nonnegative");
    }
    GlopParameters parameters;
    for (int file = 3; file < argc; ++file) {
      LoadedMatrix loaded;
      ReadMatrixMarket(argv[file], &loaded);
      std::vector<ColIndex> columns;
      columns.reserve(loaded.n);
      for (int column = 0; column < loaded.n; ++column) {
        columns.push_back(ColIndex(column));
      }
      CompactSparseMatrixView view(&loaded.matrix, &columns);
      LuFactorization factorization;
      factorization.SetParameters(parameters);
      for (int warmup = 0; warmup < warmups; ++warmup) {
        if (!factorization.ComputeFactorization(view).ok()) {
          throw std::runtime_error(std::string(argv[file]) + ": LU warmup failed");
        }
      }
      const auto start = std::chrono::steady_clock::now();
      for (int repetition = 0; repetition < repetitions; ++repetition) {
        if (!factorization.ComputeFactorization(view).ok()) {
          throw std::runtime_error(std::string(argv[file]) + ": LU failed");
        }
      }
      const auto stop = std::chrono::steady_clock::now();
      const std::chrono::duration<double> seconds = stop - start;
      std::cout << argv[file] << '\t' << loaded.n << '\t' << loaded.entries
                << '\t' << repetitions << '\t' << std::fixed
                << std::setprecision(9) << seconds.count() << '\t'
                << std::scientific << std::setprecision(17)
                << factorization.ComputeDeterminant() << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
