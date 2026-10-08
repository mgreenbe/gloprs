// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iomanip>
#include <iostream>

#include "ortools/lp_data/lp_utils.h"

int main() {
  using namespace operations_research::glop;
  int n;
  if (!(std::cin >> n)) return 2;
  DenseColumn left(RowIndex(n), 0.0);
  DenseColumn right(RowIndex(n), 0.0);
  for (int row = 0; row < n; ++row) std::cin >> left[RowIndex(row)];
  for (int row = 0; row < n; ++row) std::cin >> right[RowIndex(row)];
  DenseColumn reset = left;
  std::vector<RowIndex> non_zeros;
  ComputeNonZeros(left, &non_zeros);
  RowPermutation permutation{RowIndex(n)};
  for (int row = 0; row < n; ++row) {
    permutation[RowIndex(row)] = RowIndex(n - row - 1);
  }
  DenseColumn permuted = left;
  DenseColumn scratch(RowIndex(n), 0.0);
  PermuteWithScratchpad(permutation, &scratch, &permuted);
  DenseColumn known_permuted = left;
  PermuteWithKnownNonZeros(permutation, &scratch, &known_permuted,
                           &non_zeros);
  ChangeSign(&known_permuted);
  SumWithPositiveInfiniteAndOneMissing positive_sum;
  SumWithNegativeInfiniteAndOneMissing negative_sum;
  for (int row = 0; row < n; ++row) {
    positive_sum.Add(left[RowIndex(row)]);
    negative_sum.Add(left[RowIndex(row)]);
  }
  const int num_infinities = n % 3;
  for (int i = 0; i < num_infinities; ++i) {
    positive_sum.Add(kInfinity);
    negative_sum.Add(-kInfinity);
  }
  ScatteredColumn cleared;
  cleared.values.assign(RowIndex(n), 0.0);
  cleared.is_non_zero.ClearAndResize(RowIndex(n));
  for (int row = 0; row < n; ++row) {
    if (left[RowIndex(row)] != 0.0) {
      cleared.Add(RowIndex(row), left[RowIndex(row)]);
    }
  }
  cleared.SortNonZerosIfNeeded();
  ClearAndResizeVectorWithNonZeros(RowIndex(n), &cleared);
  if (n != 0) cleared.Add(RowIndex(0), 1.0);
  std::cout << std::setprecision(17);
  std::cout << "scalar " << ScalarProduct(left, right);
  std::cout << "\nprecise_scalar " << PreciseScalarProduct(left, right);
  std::cout << "\nsquared " << SquaredNorm(left);
  std::cout << "\nprecise_squared " << PreciseSquaredNorm(left);
  std::cout << "\ninfinity " << InfinityNorm(left);
  std::cout << "\nreset " << SquaredNormAndResetToZero(
      absl::MakeSpan(reset.data(), reset.size().value()));
  std::cout << "\nreset_values";
  for (int row = 0; row < n; ++row) std::cout << ' ' << reset[RowIndex(row)];
  std::cout << "\nsupport";
  for (const RowIndex row : non_zeros) std::cout << ' ' << row.value();
  std::cout << "\npermuted";
  for (int row = 0; row < n; ++row) std::cout << ' ' << permuted[RowIndex(row)];
  std::cout << "\nknown_negated";
  for (int row = 0; row < n; ++row) {
    std::cout << ' ' << known_permuted[RowIndex(row)];
  }
  const Fractional omitted = n == 0 ? 0.0 : left[RowIndex(0)];
  std::cout << "\npositive_sum " << positive_sum.Sum() << ' '
            << positive_sum.SumWithout(omitted) << ' '
            << positive_sum.SumWithoutLb(omitted) << ' '
            << positive_sum.SumWithoutUb(omitted) << ' '
            << positive_sum.SumWithout(kInfinity);
  std::cout << "\nnegative_sum " << negative_sum.Sum() << ' '
            << negative_sum.SumWithout(omitted) << ' '
            << negative_sum.SumWithoutLb(omitted) << ' '
            << negative_sum.SumWithoutUb(omitted) << ' '
            << negative_sum.SumWithout(-kInfinity);
  std::cout << "\nclear_protocol " << cleared.non_zeros.size();
  if (n != 0) std::cout << ' ' << cleared[RowIndex(0)];
  std::cout << '\n';
  return 0;
}
