// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Compact public-solver result trace for the pinned GLOP default pipeline.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>

#include "ortools/glop/lp_solver.h"
#include "ortools/glop/revised_simplex.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"
#include "ortools/util/time_limit.h"

template <class Tag, auto Member>
struct PrivateMemberAccess {
  friend auto Access(Tag) { return Member; }
};
struct SolverSimplexTag { friend auto Access(SolverSimplexTag); };
template struct PrivateMemberAccess<SolverSimplexTag,
    &operations_research::glop::LPSolver::revised_simplex_>;
struct BasisStatsTag { friend auto Access(BasisStatsTag); };
template struct PrivateMemberAccess<BasisStatsTag,
    &operations_research::glop::BasisFactorization::stats_>;

namespace {
uint64_t Bits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename Vector>
uint64_t HashValues(const Vector& values) {
  uint64_t hash = 14695981039346656037ULL;
  for (double value : values) hash = (hash ^ Bits(value)) * 1099511628211ULL;
  return hash;
}

template <typename Vector>
void PrintValues(const char* label, const Vector& values) {
  std::cout << label;
  for (double value : values) std::cout << ' ' << Bits(value);
  std::cout << '\n';
}

template <typename Vector, typename Name>
void PrintStatuses(const char* label, const Vector& values, Name name) {
  std::cout << label;
  for (const auto status : values) std::cout << ' ' << name(status);
  std::cout << '\n';
}

template <typename Vector, typename Name>
uint64_t HashStatuses(const Vector& values, Name name) {
  uint64_t hash = 14695981039346656037ULL;
  for (const auto status : values) {
    for (const char byte : name(status)) {
      hash = (hash ^ static_cast<unsigned char>(byte)) * 1099511628211ULL;
    }
    hash = (hash ^ 255ULL) * 1099511628211ULL;
  }
  return hash;
}
}  // namespace

int main(int argc, char** argv) {
  namespace glop = operations_research::glop;
  if (argc != 2 && argc != 3 && argc != 4) return 2;
  if (argc == 3 && std::strcmp(argv[2], "raw") != 0 &&
      std::strcmp(argv[2], "phase7") != 0) return 4;
  if (argc == 4 && std::strcmp(argv[2], "phase7") != 0) return 4;
  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) return 3;
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);
  glop::LPSolver solver;
  if (argc == 3 && std::strcmp(argv[2], "raw") == 0) {
    glop::GlopParameters parameters;
    parameters.set_use_preprocessing(false);
    parameters.set_use_scaling(false);
    solver.SetParameters(parameters);
  }
  const double wall_seconds = argc == 4 ? std::atof(argv[3]) : 20.0;
  if (!(wall_seconds > 0.0)) return 5;
  operations_research::TimeLimit limit(wall_seconds);
  const auto start = std::chrono::steady_clock::now();
  const auto status = solver.SolveWithTimeLimit(lp, &limit);
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  std::cout << glop::GetProblemStatusString(status) << ' '
            << solver.GetNumberOfSimplexIterations() << ' '
            << Bits(solver.GetObjectiveValue()) << ' '
            << Bits(solver.GetMaximumPrimalInfeasibility()) << ' '
            << Bits(solver.GetMaximumDualInfeasibility()) << '\n';
  std::cout << lp.num_variables().value() << ' ' << lp.num_constraints().value()
            << ' ' << HashValues(solver.variable_values())
            << ' ' << HashValues(solver.reduced_costs())
            << ' ' << HashValues(solver.dual_values())
            << ' ' << HashValues(solver.constraint_activities())
            << ' ' << HashStatuses(solver.variable_statuses(), glop::GetVariableStatusString)
            << ' ' << HashStatuses(solver.constraint_statuses(), glop::GetConstraintStatusString)
            << '\n';
  PrintValues("primal", solver.variable_values());
  PrintValues("reduced", solver.reduced_costs());
  PrintValues("dual", solver.dual_values());
  PrintValues("activity", solver.constraint_activities());
  PrintStatuses("variable_status", solver.variable_statuses(), glop::GetVariableStatusString);
  PrintStatuses("constraint_status", solver.constraint_statuses(), glop::GetConstraintStatusString);
  std::cout << "elapsed " << elapsed << '\n';
  if (argc >= 3 && std::strcmp(argv[2], "phase7") == 0) {
    std::cout << "deterministic " << Bits(solver.DeterministicTime()) << '\n';
    const auto& simplex = solver.*Access(SolverSimplexTag{});
    const auto count = simplex == nullptr ? 0 :
        (simplex->GetBasisFactorization().*Access(BasisStatsTag{})).refactorization_interval.Num();
    std::cout << "refactorizations " << count << '\n';
  }
}
