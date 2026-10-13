// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ortools/glop/rank_one_update.h"

int main(int argc, char** argv) {
  using namespace operations_research::glop;
  if (argc != 6) return 2;
  const int dimension = std::stoi(argv[1]);
  const int updates = std::stoi(argv[2]);
  const int entries = std::stoi(argv[3]);
  const int repetitions = std::stoi(argv[4]);
  const bool left = std::stoi(argv[5]) != 0;
  if (dimension < 4 || updates < 1 || entries < 1 ||
      entries * 13 >= dimension / 2 || repetitions < 1) {
    return 2;
  }

  CompactSparseMatrix storage;
  storage.Reset(RowIndex(dimension));
  std::vector<std::pair<ColIndex, ColIndex>> columns;
  for (int k = 0; k < updates; ++k) {
    const ColIndex u = storage.num_cols();
    for (int e = 0; e < entries; ++e) {
      storage.AddEntryToCurrentColumn(RowIndex((k * 97 + e * 13) % (dimension / 2)),
                                      (e % 7 + 1) * 0.125);
    }
    storage.CloseCurrentColumn();
    const ColIndex v = storage.num_cols();
    for (int e = 0; e < entries; ++e) {
      storage.AddEntryToCurrentColumn(
          RowIndex(dimension / 2 + (k * 97 + e * 13) % (dimension / 2)),
          (e % 5 + 1) * 0.125);
    }
    storage.CloseCurrentColumn();
    columns.emplace_back(u, v);
  }
  RankOneUpdateFactorization factorization;
  for (const auto [u, v] : columns) {
    factorization.Update(RankOneUpdateElementaryMatrix(&storage, u, v, 0.0));
  }

  DenseColumn right(RowIndex(dimension), 0.0);
  DenseRow left_values(ColIndex(dimension), 0.0);
  double checksum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    if (left) {
      for (ColIndex i(0); i < left_values.size(); ++i) left_values[i] = 1.0;
      factorization.LeftSolve(&left_values);
      checksum += left_values[ColIndex(repetition % dimension)];
    } else {
      for (RowIndex i(0); i < right.size(); ++i) right[i] = 1.0;
      factorization.RightSolve(&right);
      checksum += right[RowIndex(repetition % dimension)];
    }
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  std::cout << std::setprecision(17)
            << std::chrono::duration<double, std::nano>(elapsed).count() /
                   repetitions
            << ' ' << checksum << '\n';
}
