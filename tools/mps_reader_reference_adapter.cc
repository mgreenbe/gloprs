// Copyright 2026 Matthew Greenberg
// Licensed under the Apache License, Version 2.0.

#include <bit>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>

#include "ortools/lp_data/mps_reader.h"

namespace glop = operations_research::glop;

class Fingerprint {
 public:
  void Byte(uint8_t value) {
    hash_ ^= value;
    hash_ *= 1099511628211ULL;
  }
  void U64(uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) Byte(value >> shift);
  }
  void Double(double value) { U64(std::bit_cast<uint64_t>(value)); }
  void String(const std::string& value) {
    U64(value.size());
    for (const unsigned char byte : value) Byte(byte);
  }
  uint64_t value() const { return hash_; }

 private:
  uint64_t hash_ = 14695981039346656037ULL;
};

int main(int argc, char** argv) {
  const std::string source((std::istreambuf_iterator<char>(std::cin)),
                           std::istreambuf_iterator<char>());
  glop::LinearProgram lp;
  glop::MPSReader reader;
  glop::MPSReader::Form form = glop::MPSReader::AUTO_DETECT;
  if (argc == 2 && std::string(argv[1]) == "free") {
    form = glop::MPSReader::FREE;
  } else if (argc == 2 && std::string(argv[1]) == "fixed") {
    form = glop::MPSReader::FIXED;
  } else if (argc != 1 && !(argc == 2 && std::string(argv[1]) == "auto")) {
    return 2;
  }
  const auto status = reader.ParseProblemFromString(source, &lp, form);
  if (!status.ok()) {
    std::cout << "error\n";
    return 0;
  }

  Fingerprint hash;
  hash.String(lp.name());
  hash.Byte(lp.IsMaximizationProblem());
  hash.Double(lp.objective_offset());
  hash.U64(lp.num_constraints().value());
  hash.U64(lp.num_variables().value());
  for (glop::RowIndex row(0); row < lp.num_constraints(); ++row) {
    hash.String(lp.GetConstraintName(row));
    hash.Double(lp.constraint_lower_bounds()[row]);
    hash.Double(lp.constraint_upper_bounds()[row]);
  }
  for (glop::ColIndex col(0); col < lp.num_variables(); ++col) {
    hash.String(lp.GetVariableName(col));
    hash.Double(lp.objective_coefficients()[col]);
    hash.Double(lp.variable_lower_bounds()[col]);
    hash.Double(lp.variable_upper_bounds()[col]);
    hash.Byte(lp.IsVariableInteger(col));
    for (const auto entry : lp.GetSparseColumn(col)) {
      hash.U64(entry.row().value());
      hash.Double(entry.coefficient());
    }
  }
  std::cout << "ok " << std::hex << hash.value() << '\n';
  std::cout << lp.GetDimensionString() << '\n';
  std::cout << lp.GetObjectiveStatsString() << '\n';
  std::cout << lp.GetBoundsStatsString() << '\n';
  std::cout << lp.GetProblemStats() << '\n';
  std::cout << "pretty-problem-begin\n" << lp.GetPrettyProblemStats()
            << "pretty-problem-end\n";
  std::cout << lp.GetNonZeroStats() << '\n';
  std::cout << "pretty-nonzero-begin\n" << lp.GetPrettyNonZeroStats()
            << "pretty-nonzero-end\n";
  std::cout << "dump-begin\n" << lp.Dump() << "dump-end\n";
  std::cout << "solution " << lp.DumpSolution(lp.objective_coefficients())
            << '\n';
}
