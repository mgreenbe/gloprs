// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Times the pinned native GLOP dual-simplex solve without parsing or tracing.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

#include "ortools/glop/revised_simplex.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"
#include "ortools/util/time_limit.h"

int main(int argc, char** argv) {
  namespace glop = operations_research::glop;
  if (argc != 2 && argc != 3) return EXIT_FAILURE;
  const int repetitions = argc == 3 ? std::atoi(argv[2]) : 1;
  if (repetitions <= 0) return EXIT_FAILURE;

  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) return EXIT_FAILURE;
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);

  glop::GlopParameters parameters;
  parameters.set_use_scaling(false);
  parameters.set_use_dual_simplex(true);
  parameters.set_max_number_of_iterations(1000000);
  const auto start = std::chrono::steady_clock::now();
  std::unique_ptr<glop::RevisedSimplex> simplex;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    simplex = std::make_unique<glop::RevisedSimplex>();
    simplex->SetParameters(parameters);
    operations_research::TimeLimit limit(20.0);
    const auto result = simplex->Solve(lp, &limit);
    if (!result.ok()) return EXIT_FAILURE;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;

  std::cout << std::setprecision(17)
            << std::chrono::duration<double>(elapsed).count() << ' '
            << glop::GetProblemStatusString(simplex->GetProblemStatus()) << ' '
            << simplex->GetNumberOfIterations() << ' '
            << simplex->GetObjectiveValue() << '\n';
  return EXIT_SUCCESS;
}
