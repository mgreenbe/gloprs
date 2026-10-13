// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Times the pinned native GLOP dual-simplex solve without parsing or tracing.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

#include "ortools/glop/revised_simplex.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"
#include "ortools/util/time_limit.h"

int main(int argc, char** argv) {
  namespace glop = operations_research::glop;
  if (argc < 2 || argc > 6) return EXIT_FAILURE;
  const std::string_view mode = argc >= 4 ? std::string_view(argv[3]) : "dual";
  if (mode != "dual" && mode != "primal" && mode != "advanced")
    return EXIT_FAILURE;
  const bool primal = mode == "primal";
  const int repetitions = argc >= 3 ? std::atoi(argv[2]) : 1;
  if (repetitions <= 0) return EXIT_FAILURE;
  const int iteration_limit = argc >= 5 ? std::atoi(argv[4]) : 1000000;
  if (iteration_limit < 0) return EXIT_FAILURE;
  const double wall_seconds = argc == 6 ? std::atof(argv[5]) : 20.0;
  if (wall_seconds <= 0.0) return EXIT_FAILURE;

  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) return EXIT_FAILURE;
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);

  glop::GlopParameters parameters;
  parameters.set_use_scaling(false);
  parameters.set_use_dual_simplex(!primal);
  parameters.set_max_number_of_iterations(iteration_limit);
  const auto start = std::chrono::steady_clock::now();
  std::unique_ptr<glop::RevisedSimplex> simplex;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    simplex = std::make_unique<glop::RevisedSimplex>();
    simplex->SetParameters(parameters);
    operations_research::TimeLimit limit(wall_seconds);
    if (mode == "advanced") {
      glop::CompactSparseMatrix with_slack;
      with_slack.PopulateFromSparseMatrixAndAddSlacks(lp.GetSparseMatrix());
      simplex->MutableTransposedMatrixWithSlack()->PopulateFromTranspose(
          with_slack);
      const int columns = lp.num_variables().value();
      const int rows = lp.num_constraints().value();
      auto* lower = simplex->MutableLowerBounds();
      auto* upper = simplex->MutableUpperBounds();
      lower->resize(glop::ColIndex(columns + rows), 0.0);
      upper->resize(glop::ColIndex(columns + rows), 0.0);
      glop::DenseRow objective;
      objective.resize(glop::ColIndex(columns + rows), 0.0);
      for (int column = 0; column < columns; ++column) {
        const glop::ColIndex index(column);
        (*lower)[index] = lp.variable_lower_bounds()[index];
        (*upper)[index] = lp.variable_upper_bounds()[index];
        objective[index] = lp.objective_coefficients()[index];
      }
      for (int row = 0; row < rows; ++row) {
        const glop::ColIndex index(columns + row);
        const glop::RowIndex row_index(row);
        (*lower)[index] = -lp.constraint_upper_bounds()[row_index];
        (*upper)[index] = -lp.constraint_lower_bounds()[row_index];
      }
      const auto result = simplex->MinimizeFromTransposedMatrixWithSlack(
          objective, 1.0, 0.0, &limit);
      if (!result.ok()) return EXIT_FAILURE;
      continue;
    }
    const auto result = simplex->Solve(lp, &limit);
    if (!result.ok()) return EXIT_FAILURE;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;

  std::cout << std::setprecision(17)
            << std::chrono::duration<double>(elapsed).count() << ' '
            << glop::GetProblemStatusString(simplex->GetProblemStatus()) << ' '
            << simplex->GetNumberOfIterations() << ' '
            << simplex->GetObjectiveValue() << '\n';
  if (std::getenv("GLOPRS_SHOW_LU_THRESHOLD") != nullptr) {
    std::cerr << std::setprecision(17) << "lu_pivot_threshold "
              << simplex->GetParameters().lu_factorization_pivot_threshold() << '\n';
  }
  if (std::getenv("GLOPRS_SHOW_NATIVE_STATS") != nullptr) {
    std::cerr << simplex->StatString();
  }
  return EXIT_SUCCESS;
}
