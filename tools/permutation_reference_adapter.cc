// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <cstdint>
#include <iostream>

#include "ortools/lp_data/permutation.h"

int main() {
  using namespace operations_research::glop;
  int size;
  if (!(std::cin >> size)) return 2;
  ColumnPermutation permutation{ColIndex(size)};
  for (int i = 0; i < size; ++i) {
    int destination;
    std::cin >> destination;
    permutation[ColIndex(i)] = ColIndex(destination);
  }
  DenseColumn values{RowIndex(size)};
  for (int i = 0; i < size; ++i) std::cin >> values[RowIndex(i)];

  std::cout << "check " << permutation.Check() << '\n';
  if (!permutation.Check()) return 0;
  std::cout << "signature " << permutation.ComputeSignature() << '\n';

  ColumnPermutation inverse;
  inverse.PopulateFromInverse(permutation);
  std::cout << "inverse";
  for (int i = 0; i < size; ++i) {
    std::cout << ' ' << inverse[ColIndex(i)].value();
  }
  std::cout << '\n';

  DenseColumn result;
  ApplyPermutation(permutation, values, &result);
  std::cout << "apply";
  for (double value : result) std::cout << ' ' << static_cast<int64_t>(value);
  std::cout << '\n';
  DenseColumn restored;
  ApplyInversePermutation(permutation, result, &restored);
  std::cout << "restore";
  for (double value : restored) std::cout << ' ' << static_cast<int64_t>(value);
  std::cout << '\n';

  RowPermutation identity{RowIndex(size)};
  identity.PopulateFromIdentity();
  std::cout << "identity";
  for (int i = 0; i < size; ++i) {
    std::cout << ' ' << identity[RowIndex(i)].value();
  }
  std::cout << '\n';
  return 0;
}
