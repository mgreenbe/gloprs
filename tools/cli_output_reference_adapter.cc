// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// The label width and scientific precision follow ortools/linear_solver/solve.cc.
// Values and inspect formatting come directly from the pinned GLOP API.
#include <cstdio>
#include <iostream>
#include <string>
#include "ortools/glop/lp_solver.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  namespace glop = operations_research::glop;
  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) return 3;
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);
  if (std::string(argv[2]) == "inspect") {
    std::cout << lp.GetPrettyProblemStats();
    return 0;
  }
  glop::LPSolver solver;
  const auto status = solver.Solve(lp);
  std::printf("%-12s: '%s'\n", "File", argv[1]);
  std::printf("%-12s: %d x %d\n", "Dimension", lp.num_constraints().value(), lp.num_variables().value());
  std::printf("%-12s: %s\n", "Status", glop::GetProblemStatusString(status).c_str());
  std::printf("%-12s: %15.15e\n", "Objective", solver.GetObjectiveValue());
  std::printf("%-12s: %lld\n", "Iterations", static_cast<long long>(solver.GetNumberOfSimplexIterations()));
  std::printf("%-12s: %15.15e\n", "Primal error", solver.GetMaximumPrimalInfeasibility());
  std::printf("%-12s: %15.15e\n", "Dual error", solver.GetMaximumDualInfeasibility());
}
