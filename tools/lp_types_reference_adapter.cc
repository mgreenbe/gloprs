// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <bit>
#include <cstdint>
#include <iostream>

#include "ortools/lp_data/lp_types.h"

namespace {
using namespace operations_research::glop;

template <typename Index>
void PrintBits(const char* name,
               const operations_research::Bitset64<Index>& bits) {
  std::cout << name << ' ' << bits.size().value();
  for (Index i(0); i < bits.size(); ++i) std::cout << (bits[i] ? '1' : '0');
  std::cout << " set";
  for (const Index i : bits) std::cout << ' ' << i.value();
  std::cout << '\n';
}
}  // namespace

int main() {
  using namespace operations_research::glop;
  int size;
  int count;
  int other_size;
  int other_count;
  int query;
  if (!(std::cin >> size >> count >> other_size >> other_count >> query)) return 2;
  DenseBitColumn bits{RowIndex(size)};
  for (int i = 0; i < count; ++i) {
    int position;
    std::cin >> position;
    bits.Set(RowIndex(position));
  }
  DenseBitColumn other{RowIndex(other_size)};
  for (int i = 0; i < other_count; ++i) {
    int position;
    std::cin >> position;
    other.Set(RowIndex(position));
  }

  PrintBits("bits", bits);
  std::cout << "pair " << bits.AreOneOfTwoBitsSet(RowIndex(query)) << '\n';
  DenseBitColumn cleared = bits;
  cleared.ClearTwoBits(RowIndex(query));
  PrintBits("clear_pair", cleared);

  DenseBitColumn content{RowIndex(size)};
  for (int i = 1; i < size; i += 2) content.Set(RowIndex(i));
  content.SetContentFromBitset(other);
  PrintBits("content", content);

  DenseBitColumn intersection = bits;
  intersection.Intersection(other);
  PrintBits("intersection", intersection);
  DenseBitColumn united = bits;
  united.Union(other);
  PrintBits("union", united);

  DenseBitColumn resized = bits;
  resized.Resize(RowIndex(size / 2));
  resized.Resize(RowIndex(size + 5));
  PrintBits("resized", resized);

  for (const ProblemStatus status : {
           ProblemStatus::OPTIMAL, ProblemStatus::PRIMAL_INFEASIBLE,
           ProblemStatus::DUAL_INFEASIBLE, ProblemStatus::INFEASIBLE_OR_UNBOUNDED,
           ProblemStatus::PRIMAL_UNBOUNDED, ProblemStatus::DUAL_UNBOUNDED,
           ProblemStatus::INIT, ProblemStatus::PRIMAL_FEASIBLE,
           ProblemStatus::DUAL_FEASIBLE, ProblemStatus::ABNORMAL,
           ProblemStatus::INVALID_PROBLEM, ProblemStatus::IMPRECISE}) {
    std::cout << "problem " << GetProblemStatusString(status) << '\n';
  }
  for (const VariableType type : {
           VariableType::UNCONSTRAINED, VariableType::LOWER_BOUNDED,
           VariableType::UPPER_BOUNDED, VariableType::UPPER_AND_LOWER_BOUNDED,
           VariableType::FIXED_VARIABLE}) {
    std::cout << "type " << GetVariableTypeString(type) << '\n';
  }
  for (const VariableStatus status : {
           VariableStatus::BASIC, VariableStatus::FIXED_VALUE,
           VariableStatus::AT_LOWER_BOUND, VariableStatus::AT_UPPER_BOUND,
           VariableStatus::FREE}) {
    std::cout << "status " << GetVariableStatusString(status) << ' '
              << GetConstraintStatusString(VariableToConstraintStatus(status)) << '\n';
  }
  std::cout << "scalars " << std::bit_cast<uint64_t>(kRangeMax) << ' '
            << std::bit_cast<uint64_t>(kInfinity) << ' '
            << std::bit_cast<uint64_t>(kEpsilon) << ' '
            << std::bit_cast<uint64_t>(DeterministicTimeForFpOperations(123456789))
            << '\n';
  return 0;
}
