// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

// Time GLOP LU construction on already-loaded Matrix Market probe matrices.
#include <chrono>
#include <cmath>
#include <cstdint>
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
    int first_file = 3;
    int evict_mib = 0;
    bool retain_factorization = false;
    while (first_file < argc) {
      if (std::string(argv[first_file]) == "--evict-mib") {
        if (first_file + 1 >= argc) throw std::runtime_error("missing cache eviction size");
        evict_mib = std::stoi(argv[first_file + 1]);
        if (evict_mib < 0 || evict_mib > 1024) {
          throw std::runtime_error("invalid cache eviction size");
        }
        first_file += 2;
      } else if (std::string(argv[first_file]) == "--retain-factorization") {
        retain_factorization = true;
        ++first_file;
      } else if (std::string(argv[first_file]) == "--pivot-threshold") {
        if (first_file + 1 >= argc) throw std::runtime_error("missing pivot threshold");
        const double threshold = std::stod(argv[first_file + 1]);
        if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
          throw std::runtime_error("invalid pivot threshold");
        }
        parameters.set_lu_factorization_pivot_threshold(threshold);
        first_file += 2;
      } else {
        break;
      }
    }
    std::vector<std::uint8_t> eviction(
        static_cast<size_t>(evict_mib) * 1024 * 1024, 1);
    LoadedMatrix source;
    const bool selected = argc >= first_file + 3 &&
                          std::string(argv[first_file]) == "--selected-source";
    if (selected) {
      ReadMatrixMarket(argv[first_file + 1], &source);
      first_file += 2;
    }
    LuFactorization retained_factorization;
    retained_factorization.SetParameters(parameters);
    for (int file = first_file; file < argc; ++file) {
      LoadedMatrix loaded;
      if (!selected) ReadMatrixMarket(argv[file], &loaded);
      std::vector<ColIndex> columns;
      const CompactSparseMatrix* matrix = selected ? &source.matrix : &loaded.matrix;
      if (selected) {
        std::ifstream input(argv[file]);
        int dimension;
        if (!(input >> dimension) || dimension != source.n) {
          throw std::runtime_error(std::string(argv[file]) + ": invalid selection dimension");
        }
        columns.reserve(dimension);
        for (int position = 0; position < dimension; ++position) {
          int column;
          if (!(input >> column) || column < 0 ||
              column >= matrix->num_cols().value()) {
            throw std::runtime_error(std::string(argv[file]) + ": invalid selected column");
          }
          columns.push_back(ColIndex(column));
        }
        int extra;
        if (input >> extra) {
          throw std::runtime_error(std::string(argv[file]) + ": extra selected column");
        }
      } else {
        if (loaded.n != matrix->num_cols().value()) {
          throw std::runtime_error(std::string(argv[file]) + ": not square");
        }
        columns.reserve(loaded.n);
        for (int column = 0; column < loaded.n; ++column) {
          columns.push_back(ColIndex(column));
        }
      }
      size_t selected_entries = 0;
      for (const ColIndex column : columns) {
        selected_entries += matrix->ColumnNumEntries(column).value();
      }
      CompactSparseMatrixView view(matrix, &columns);
      LuFactorization local_factorization;
      local_factorization.SetParameters(parameters);
      LuFactorization* factorization = retain_factorization
                                           ? &retained_factorization
                                           : &local_factorization;
      for (int warmup = 0; warmup < warmups; ++warmup) {
        if (!factorization->ComputeFactorization(view).ok()) {
          throw std::runtime_error(std::string(argv[file]) + ": LU warmup failed");
        }
      }
      double seconds = 0.0;
      if (eviction.empty()) {
        const auto start = std::chrono::steady_clock::now();
        for (int repetition = 0; repetition < repetitions; ++repetition) {
          if (!factorization->ComputeFactorization(view).ok()) {
            throw std::runtime_error(std::string(argv[file]) + ": LU failed");
          }
        }
        seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
      } else {
        for (int repetition = 0; repetition < repetitions; ++repetition) {
          // Cold-cache diagnostic: intervening simplex work touches other
          // arrays. Keep that traffic outside the numerical LU timer.
          volatile const std::uint8_t* bytes = eviction.data();
          std::uint64_t checksum = 0;
          for (size_t offset = 0; offset < eviction.size(); offset += 64) {
            checksum += bytes[offset];
          }
          if (checksum == 0) throw std::runtime_error("invalid eviction buffer");
          const auto start = std::chrono::steady_clock::now();
          if (!factorization->ComputeFactorization(view).ok()) {
            throw std::runtime_error(std::string(argv[file]) + ": LU failed");
          }
          seconds += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start).count();
        }
      }
      std::cout << argv[file] << '\t' << columns.size() << '\t' << selected_entries
                << '\t' << repetitions << '\t' << std::fixed
                << std::setprecision(9) << seconds << '\t'
                << std::scientific << std::setprecision(17)
                << factorization->ComputeDeterminant() << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
