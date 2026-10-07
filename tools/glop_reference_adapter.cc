// Copyright 2026 Matthew Greenberg
// Licensed under the Apache License, Version 2.0.
//
// A deliberately small adapter around the pinned native GLOP API.  The
// generic OR-Tools `solve` binary drops basis statuses from its response; this
// program preserves them for differential tests of gloprs.

#include <cstdlib>
#include <chrono>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

#include "ortools/glop/lp_solver.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/lp_types.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"

namespace glop = operations_research::glop;

uint64_t ModelDataFingerprint(const glop::LinearProgram& lp) {
  uint64_t hash = 14695981039346656037ULL;
  const auto add_byte = [&hash](uint8_t byte) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  };
  add_byte(lp.IsMaximizationProblem() ? 1 : 0);
  const auto add_double = [&add_byte](double value) {
    const uint64_t bits = std::bit_cast<uint64_t>(value);
    for (int shift = 0; shift < 64; shift += 8) {
      add_byte(static_cast<uint8_t>(bits >> shift));
    }
  };
  add_double(lp.objective_offset());
  add_double(lp.objective_scaling_factor());
  for (const double value : lp.objective_coefficients()) add_double(value);
  for (const double value : lp.variable_lower_bounds()) add_double(value);
  for (const double value : lp.variable_upper_bounds()) add_double(value);
  for (const double value : lp.constraint_lower_bounds()) add_double(value);
  for (const double value : lp.constraint_upper_bounds()) add_double(value);
  return hash;
}

template <typename Vector, typename Convert>
void PrintStringArray(const Vector& values, Convert convert) {
  std::cout << '[';
  bool first = true;
  for (const auto value : values) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << '"' << convert(value) << '"';
  }
  std::cout << ']';
}

template <typename Vector>
void PrintNumberArray(const Vector& values) {
  std::cout << '[';
  bool first = true;
  for (const auto value : values) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << value;
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: glop_reference_adapter MODEL.mps\n";
    return EXIT_FAILURE;
  }

  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) {
    std::cerr << model.status() << '\n';
    return EXIT_FAILURE;
  }
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);

  glop::LPSolver solver;
  const auto solve_start = std::chrono::steady_clock::now();
  const glop::ProblemStatus status = solver.Solve(lp);
  const std::chrono::duration<double> solve_time =
      std::chrono::steady_clock::now() - solve_start;

  std::cout << std::setprecision(17);
  std::cout << "{\"status\":\"" << glop::GetProblemStatusString(status)
            << "\",\"model\":{\"rows\":" << lp.num_constraints().value()
            << ",\"columns\":" << lp.num_variables().value()
            << ",\"nonzeros\":" << lp.num_entries().value()
            << ",\"data_fingerprint\":\"" << std::hex
            << std::setw(16) << std::setfill('0') << ModelDataFingerprint(lp)
            << std::dec << "\"}"
            << ",\"objective\":" << solver.GetObjectiveValue()
            << ",\"iterations\":" << solver.GetNumberOfSimplexIterations()
            << ",\"solve_time_seconds\":" << solve_time.count()
            << ",\"deterministic_time\":" << solver.DeterministicTime()
            << ",\"maximum_primal_infeasibility\":"
            << solver.GetMaximumPrimalInfeasibility()
            << ",\"maximum_dual_infeasibility\":"
            << solver.GetMaximumDualInfeasibility()
            << ",\"variable_values\":";
  PrintNumberArray(solver.variable_values());
  std::cout << ",\"dual_values\":";
  PrintNumberArray(solver.dual_values());
  std::cout << ",\"reduced_costs\":";
  PrintNumberArray(solver.reduced_costs());
  std::cout << ",\"basis\":{\"variables\":";
  PrintStringArray(solver.variable_statuses(), [](glop::VariableStatus value) {
    return glop::GetVariableStatusString(value);
  });
  std::cout << ",\"constraints\":";
  PrintStringArray(
      solver.constraint_statuses(), [](glop::ConstraintStatus value) {
        return glop::GetConstraintStatusString(value);
      });
  std::cout << "}}\n";
  return EXIT_SUCCESS;
}
