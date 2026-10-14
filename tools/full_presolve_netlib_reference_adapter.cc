// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Pinned GLOP public-solver result with its default presolve/scaling pipeline.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ortools/glop/lp_solver.h"
#include "ortools/glop/preprocessor.h"
#include "ortools/glop/revised_simplex.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/lp_utils.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/lp_data/proto_utils.h"
#include "ortools/util/time_limit.h"

// Diagnostic-only access to the exact cached norm object used by native
// pricing. This explicit-instantiation technique does not alter upstream.
template <class Tag, typename Tag::type Member>
struct PrivateMemberAccess {
  friend typename Tag::type Access(Tag) { return Member; }
};
struct PrimalNormsTag {
  using type = operations_research::glop::PrimalEdgeNorms operations_research::glop::RevisedSimplex::*;
  friend type Access(PrimalNormsTag);
};
template struct PrivateMemberAccess<PrimalNormsTag,
    &operations_research::glop::RevisedSimplex::primal_edge_norms_>;
struct DirectionInverseTag {
  using type = operations_research::glop::ScatteredRow operations_research::glop::PrimalEdgeNorms::*;
  friend type Access(DirectionInverseTag);
};
template struct PrivateMemberAccess<DirectionInverseTag,
    &operations_research::glop::PrimalEdgeNorms::direction_left_inverse_>;
struct UpdateRowTag {
  using type = operations_research::glop::UpdateRow operations_research::glop::RevisedSimplex::*;
  friend type Access(UpdateRowTag);
};
struct PrimalPricesTag {
  using type = operations_research::glop::PrimalPrices operations_research::glop::RevisedSimplex::*;
  friend type Access(PrimalPricesTag);
};
template struct PrivateMemberAccess<PrimalPricesTag,
    &operations_research::glop::RevisedSimplex::primal_prices_>;
struct PriceHeapTag {
  using type = operations_research::glop::DynamicMaximum<operations_research::glop::ColIndex> operations_research::glop::PrimalPrices::*;
  friend type Access(PriceHeapTag);
};
template struct PrivateMemberAccess<PriceHeapTag,
    &operations_research::glop::PrimalPrices::prices_>;
template struct PrivateMemberAccess<UpdateRowTag,
    &operations_research::glop::RevisedSimplex::update_row_>;
struct DirectionTag {
  using type = operations_research::glop::ScatteredColumn operations_research::glop::RevisedSimplex::*;
  friend type Access(DirectionTag);
};
struct EquivalentChoicesTag {
  using type = std::vector<operations_research::glop::RowIndex> operations_research::glop::RevisedSimplex::*;
  friend type Access(EquivalentChoicesTag);
};
template struct PrivateMemberAccess<EquivalentChoicesTag,
    &operations_research::glop::RevisedSimplex::equivalent_leaving_choices_>;
template struct PrivateMemberAccess<DirectionTag,
    &operations_research::glop::RevisedSimplex::direction_>;
template <class Tag, auto Member>
struct PrivateMemberAccessAuto {
  friend auto Access(Tag) { return Member; }
};
struct PriceTopsTag { friend auto Access(PriceTopsTag); };
template struct PrivateMemberAccessAuto<PriceTopsTag,
    &operations_research::glop::DynamicMaximum<operations_research::glop::ColIndex>::tops_>;
struct PhaseTag { friend auto Access(PhaseTag); };
template struct PrivateMemberAccessAuto<PhaseTag,
    &operations_research::glop::RevisedSimplex::phase_>;
struct RefactorReasonTag { friend auto Access(RefactorReasonTag); };
template struct PrivateMemberAccessAuto<RefactorReasonTag,
    &operations_research::glop::RevisedSimplex::last_refactorization_reason_>;
struct RightPoolMappingTag { friend auto Access(RightPoolMappingTag); };
template struct PrivateMemberAccessAuto<RightPoolMappingTag,
    &operations_research::glop::BasisFactorization::right_pool_mapping_>;
struct LeftPoolMappingTag { friend auto Access(LeftPoolMappingTag); };
template struct PrivateMemberAccessAuto<LeftPoolMappingTag,
    &operations_research::glop::BasisFactorization::left_pool_mapping_>;
struct ReducedCostsTag { friend auto Access(ReducedCostsTag); };
template struct PrivateMemberAccessAuto<ReducedCostsTag,
    &operations_research::glop::RevisedSimplex::reduced_costs_>;
struct RcValuesTag { friend auto Access(RcValuesTag); };
template struct PrivateMemberAccessAuto<RcValuesTag,
    &operations_research::glop::ReducedCosts::reduced_costs_>;
struct BasicObjectiveTag { friend auto Access(BasicObjectiveTag); };
template struct PrivateMemberAccessAuto<BasicObjectiveTag,
    &operations_research::glop::ReducedCosts::basic_objective_>;
struct ObjectiveTag { friend auto Access(ObjectiveTag); };
template struct PrivateMemberAccessAuto<ObjectiveTag,
    &operations_research::glop::RevisedSimplex::objective_>;
struct CostPerturbationsTag { friend auto Access(CostPerturbationsTag); };
template struct PrivateMemberAccessAuto<CostPerturbationsTag,
    &operations_research::glop::ReducedCosts::cost_perturbations_>;
struct RcPreciseTag { friend auto Access(RcPreciseTag); };
template struct PrivateMemberAccessAuto<RcPreciseTag,
    &operations_research::glop::ReducedCosts::are_reduced_costs_precise_>;
struct RcRecomputeTag { friend auto Access(RcRecomputeTag); };
template struct PrivateMemberAccessAuto<RcRecomputeTag,
    &operations_research::glop::ReducedCosts::recompute_reduced_costs_>;
struct RandomEngineTag { friend auto Access(RandomEngineTag); };
template struct PrivateMemberAccessAuto<RandomEngineTag,
    &operations_research::glop::RevisedSimplex::deterministic_random_>;
struct BasisLuTag {
  using type = operations_research::glop::LuFactorization operations_research::glop::BasisFactorization::*;
  friend type Access(BasisLuTag);
};
struct CompactMatrixTag { friend auto Access(CompactMatrixTag); };
template struct PrivateMemberAccessAuto<CompactMatrixTag,
    &operations_research::glop::RevisedSimplex::compact_matrix_>;
template struct PrivateMemberAccess<BasisLuTag,
    &operations_research::glop::BasisFactorization::lu_factorization_>;
struct LuNonZerosTag {
  using type = std::vector<operations_research::glop::RowIndex> operations_research::glop::LuFactorization::*;
  friend type Access(LuNonZerosTag);
};
struct LuRowPermutationTag { friend auto Access(LuRowPermutationTag); };
template struct PrivateMemberAccessAuto<LuRowPermutationTag,
    &operations_research::glop::LuFactorization::row_perm_>;
struct LuInverseRowPermutationTag { friend auto Access(LuInverseRowPermutationTag); };
template struct PrivateMemberAccessAuto<LuInverseRowPermutationTag,
    &operations_research::glop::LuFactorization::inverse_row_perm_>;
template struct PrivateMemberAccess<LuNonZerosTag,
    &operations_research::glop::LuFactorization::non_zero_rows_>;
struct MaxUpdatesTag {
  using type = int operations_research::glop::BasisFactorization::*;
  friend type Access(MaxUpdatesTag);
};
template struct PrivateMemberAccess<MaxUpdatesTag,
    &operations_research::glop::BasisFactorization::max_num_updates_>;
struct LastFactorizationTimeTag {
  using type = double operations_research::glop::BasisFactorization::*;
  friend type Access(LastFactorizationTimeTag);
};
template struct PrivateMemberAccess<LastFactorizationTimeTag,
    &operations_research::glop::BasisFactorization::last_factorization_deterministic_time_>;
struct RankOneTag {
  using type = operations_research::glop::RankOneUpdateFactorization operations_research::glop::BasisFactorization::*;
  friend type Access(RankOneTag);
};
template struct PrivateMemberAccess<RankOneTag,
    &operations_research::glop::BasisFactorization::rank_one_factorization_>;
struct BasisStatsTag { friend auto Access(BasisStatsTag); };
template struct PrivateMemberAccessAuto<BasisStatsTag,
    &operations_research::glop::BasisFactorization::stats_>;

operations_research::glop::RevisedSimplex* g_diagnostic_direct = nullptr;
extern "C" __attribute__((used, noinline)) int diag_basis(int row) {
  return g_diagnostic_direct->GetBasis(operations_research::glop::RowIndex(row)).value();
}
extern "C" __attribute__((used, noinline)) int diag_iteration() {
  return g_diagnostic_direct->GetNumberOfIterations();
}
extern "C" __attribute__((used, noinline)) uint64_t diag_random_next() {
  auto copy = g_diagnostic_direct->*Access(RandomEngineTag{});
  return copy();
}
extern "C" __attribute__((used, noinline)) uint64_t diag_direction_bit(int row) {
  const auto value = (g_diagnostic_direct->*Access(DirectionTag{})).values[
      operations_research::glop::RowIndex(row)];
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_direction_hash() {
  uint64_t hash = 14695981039346656037ULL;
  for (const auto value : (g_diagnostic_direct->*Access(DirectionTag{})).values) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ULL;
  }
  return hash;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_direction_order_hash() {
  uint64_t hash = 14695981039346656037ULL;
  for (const auto row : (g_diagnostic_direct->*Access(DirectionTag{})).non_zeros)
    hash = (hash ^ static_cast<uint64_t>(row.value())) * 1099511628211ULL;
  return hash;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_basic_objective_hash() {
  const auto& values = (g_diagnostic_direct->*Access(ReducedCostsTag{})).*Access(BasicObjectiveTag{});
  uint64_t hash = 14695981039346656037ULL;
  for (const auto value : values) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ULL;
  }
  return hash;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_precise_dot_bit() {
  const auto& values = (g_diagnostic_direct->*Access(ReducedCostsTag{})).*Access(BasicObjectiveTag{});
  const auto value = operations_research::glop::ScalarProduct(values,
      g_diagnostic_direct->*Access(DirectionTag{}));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_objective_sum_bit(int col) {
  const auto index = operations_research::glop::ColIndex(col);
  const auto& objective = g_diagnostic_direct->*Access(ObjectiveTag{});
  const auto& perturbations = (g_diagnostic_direct->*Access(ReducedCostsTag{})).*Access(CostPerturbationsTag{});
  const auto value = objective[index] + perturbations[index];
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_update_coeff_bit(int col) {
  const auto value = (g_diagnostic_direct->*Access(UpdateRowTag{})).GetCoefficient(
      operations_research::glop::ColIndex(col));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) int diag_direction_size() {
  return (g_diagnostic_direct->*Access(DirectionTag{})).non_zeros.size();
}
extern "C" __attribute__((used, noinline)) int diag_direction_order_index(int row) {
  const auto& positions = (g_diagnostic_direct->*Access(DirectionTag{})).non_zeros;
  for (int i = 0; i < positions.size(); ++i) if (positions[i].value() == row) return i;
  return -1;
}
extern "C" __attribute__((used, noinline)) int diag_equiv_size() {
  return (g_diagnostic_direct->*Access(EquivalentChoicesTag{})).size();
}
extern "C" __attribute__((used, noinline)) int diag_equiv_at(int i) {
  return (g_diagnostic_direct->*Access(EquivalentChoicesTag{}))[i].value();
}
extern "C" __attribute__((used, noinline)) int diag_price_root_index() {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  const auto& tops = heap.*Access(PriceTopsTag{});
  return tops.empty() ? -1 : tops[0].index.value();
}
extern "C" __attribute__((used, noinline)) int diag_price_top_size() {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  return (heap.*Access(PriceTopsTag{})).size();
}
extern "C" __attribute__((used, noinline)) int diag_price_top_at(int i) {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  return (heap.*Access(PriceTopsTag{}))[i].index.value();
}
extern "C" __attribute__((used, noinline)) int diag_price_top_position(int col) {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  const auto& tops = heap.*Access(PriceTopsTag{});
  for (int i = 0; i < tops.size(); ++i) if (tops[i].index.value() == col) return i;
  return -1;
}
extern "C" __attribute__((used, noinline)) int diag_print_price_tops() {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  const auto& tops = heap.*Access(PriceTopsTag{});
  std::cerr << "NATIVE_PRICE_TOPS";
  for (const auto& top : tops) std::cerr << ' ' << top.index.value();
  std::cerr << '\n';
  return tops.size();
}
extern "C" __attribute__((used, noinline)) int diag_print_price_top_bits() {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  const auto& tops = heap.*Access(PriceTopsTag{});
  std::cerr << "NATIVE_PRICE_TOP_BITS";
  for (const auto& top : tops) {
    uint64_t bits;
    std::memcpy(&bits, &top.value, sizeof(bits));
    std::cerr << ' ' << top.index.value() << ':' << bits;
  }
  std::cerr << '\n';
  return tops.size();
}
extern "C" __attribute__((used, noinline)) int diag_lu_row_perm_at(int row) {
  const auto& lu = g_diagnostic_direct->GetBasisFactorization().*Access(BasisLuTag{});
  return (lu.*Access(LuRowPermutationTag{}))[operations_research::glop::RowIndex(row)].value();
}
extern "C" __attribute__((used, noinline)) uint64_t diag_solve_norm_bit(int col) {
  const auto& compact = g_diagnostic_direct->*Access(CompactMatrixTag{});
  const auto norm = g_diagnostic_direct->GetBasisFactorization().RightSolveSquaredNorm(
      compact.column(operations_research::glop::ColIndex(col)));
  uint64_t bits;
  std::memcpy(&bits, &norm, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_norm_bit(int col) {
  const auto value = (g_diagnostic_direct->*Access(PrimalNormsTag{})).GetSquaredNorms()[
      operations_research::glop::ColIndex(col)];
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) uint64_t diag_price_root_bit() {
  const auto& heap = (g_diagnostic_direct->*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
  const auto& tops = heap.*Access(PriceTopsTag{});
  if (tops.empty()) return 0;
  const auto value = tops[0].value;
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) int diag_update_contains(int col) {
  for (const auto index : (g_diagnostic_direct->*Access(UpdateRowTag{})).GetNonZeroPositions())
    if (index.value() == col) return 1;
  return 0;
}
extern "C" __attribute__((used, noinline)) int diag_update_position(int col) {
  const auto& positions = (g_diagnostic_direct->*Access(UpdateRowTag{})).GetNonZeroPositions();
  for (int i = 0; i < positions.size(); ++i) if (positions[i].value() == col) return i;
  return -1;
}
extern "C" __attribute__((used, noinline)) int diag_phase() {
  return static_cast<int>(g_diagnostic_direct->*Access(PhaseTag{}));
}
extern "C" __attribute__((used, noinline)) int diag_refact() {
  return g_diagnostic_direct->GetBasisFactorization().IsRefactorized();
}
extern "C" __attribute__((used, noinline)) int diag_refact_reason() {
  return static_cast<int>(g_diagnostic_direct->*Access(RefactorReasonTag{}));
}
extern "C" __attribute__((used, noinline)) int diag_update_count() {
  return g_diagnostic_direct->GetBasisFactorization().NumUpdates();
}
extern "C" __attribute__((used, noinline)) int diag_max_updates() {
  return g_diagnostic_direct->GetBasisFactorization().*Access(MaxUpdatesTag{});
}
extern "C" __attribute__((used, noinline)) double diag_last_factorization_time() {
  return g_diagnostic_direct->GetBasisFactorization().*Access(LastFactorizationTimeTag{});
}
extern "C" __attribute__((used, noinline)) double diag_rank_one_time() {
  const auto& factorization = g_diagnostic_direct->GetBasisFactorization();
  return (factorization.*Access(RankOneTag{})).DeterministicTimeSinceLastReset();
}
extern "C" __attribute__((used, noinline)) int diag_right_pool(int col) {
  const auto& f = g_diagnostic_direct->GetBasisFactorization();
  const auto& mapping = f.*Access(RightPoolMappingTag{});
  return col < mapping.size().value() ? mapping[operations_research::glop::ColIndex(col)].value() : -1;
}
extern "C" __attribute__((used, noinline)) int diag_left_pool(int row) {
  const auto& f = g_diagnostic_direct->GetBasisFactorization();
  const auto& mapping = f.*Access(LeftPoolMappingTag{});
  return row < mapping.size().value() ? mapping[operations_research::glop::ColIndex(row)].value() : -1;
}
extern "C" __attribute__((used, noinline)) int diag_rc_refact() {
  return (g_diagnostic_direct->*Access(ReducedCostsTag{})).NeedsBasisRefactorization();
}
extern "C" __attribute__((used, noinline)) uint64_t diag_rc_value_bit(int col) {
  const auto& values = (g_diagnostic_direct->*Access(ReducedCostsTag{})).*Access(RcValuesTag{});
  const double value = values[operations_research::glop::ColIndex(col)];
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
extern "C" __attribute__((used, noinline)) int diag_rc_precise() {
  return (g_diagnostic_direct->*Access(ReducedCostsTag{})).*Access(RcPreciseTag{});
}
extern "C" __attribute__((used, noinline)) int diag_rc_recompute() {
  return (g_diagnostic_direct->*Access(ReducedCostsTag{})).*Access(RcRecomputeTag{});
}

int main(int argc, char** argv) {
  namespace glop = operations_research::glop;
  if (argc < 2 || argc > 6) return EXIT_FAILURE;
  const auto parse_start = std::chrono::steady_clock::now();
  const auto model = glop::MpsFileToMPModelProto(argv[1]);
  if (!model.ok()) return EXIT_FAILURE;
  glop::LinearProgram lp;
  glop::MPModelProtoToLinearProgram(*model, &lp);
  if (argc >= 3 && std::strcmp(argv[2], "parse_timing") == 0) {
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - parse_start).count();
    std::cout << std::setprecision(17) << "parse_seconds " << seconds
              << " rows " << lp.num_constraints().value()
              << " columns " << lp.num_variables().value()
              << " nonzeros " << lp.num_entries().value() << '\n';
    return EXIT_SUCCESS;
  }
  if (argc >= 3 && std::strcmp(argv[2], "presolve_timing") == 0) {
    const int repetitions = argc >= 4 ? std::atoi(argv[3]) : 1;
    if (repetitions <= 0) return EXIT_FAILURE;
    glop::GlopParameters parameters;
    const auto start = std::chrono::steady_clock::now();
    glop::ProblemStatus status = glop::ProblemStatus::INIT;
    int rows = 0, columns = 0, nonzeros = 0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      glop::LinearProgram presolved;
      presolved.PopulateFromLinearProgram(lp);
      presolved.RemoveNearZeroEntries(parameters.drop_magnitude());
      glop::MainLpPreprocessor preprocessor(&parameters);
      operations_research::TimeLimit limit(20.0);
      preprocessor.SetTimeLimit(&limit);
      preprocessor.Run(&presolved);
      status = preprocessor.status();
      rows = presolved.num_constraints().value();
      columns = presolved.num_variables().value();
      nonzeros = presolved.num_entries().value();
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count() / repetitions;
    std::cout << std::setprecision(17) << "presolve_seconds " << seconds
              << " status " << glop::GetProblemStatusString(status)
              << " rows " << rows << " columns " << columns
              << " nonzeros " << nonzeros << '\n';
    return EXIT_SUCCESS;
  }
  glop::LinearProgram presolved;
  presolved.PopulateFromLinearProgram(lp);
  glop::GlopParameters parameters;
  presolved.RemoveNearZeroEntries(parameters.drop_magnitude());
  glop::MainLpPreprocessor preprocessor(&parameters);
  operations_research::TimeLimit presolve_limit(20.0);
  preprocessor.SetTimeLimit(&presolve_limit);
  preprocessor.Run(&presolved);
  if (argc >= 3 && std::strcmp(argv[2], "presolved_solve_timing") == 0) {
    const int repetitions = argc >= 4 ? std::atoi(argv[3]) : 1;
    const double wall_seconds = argc >= 5 ? std::atof(argv[4]) : 20.0;
    glop::GlopParameters solve_parameters = parameters;
    if (argc >= 6) solve_parameters.set_max_number_of_iterations(std::atoi(argv[5]));
    if (repetitions <= 0 || !(wall_seconds > 0.0) ||
        (argc >= 6 && solve_parameters.max_number_of_iterations() < 0) ||
        preprocessor.status() != glop::ProblemStatus::INIT)
      return EXIT_FAILURE;
    glop::ProblemStatus status = glop::ProblemStatus::INIT;
    int iterations = 0;
    double objective = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      glop::RevisedSimplex simplex;
      simplex.SetParameters(solve_parameters);
      operations_research::TimeLimit limit(wall_seconds);
      const auto solve_status = simplex.Solve(presolved, &limit);
      if (!solve_status.ok() &&
          !(argc >= 6 && simplex.GetNumberOfIterations() >=
                              solve_parameters.max_number_of_iterations()))
        return EXIT_FAILURE;
      status = simplex.GetProblemStatus();
      iterations = simplex.GetNumberOfIterations();
      objective = simplex.GetObjectiveValue();
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count() / repetitions;
    uint64_t objective_bits;
    std::memcpy(&objective_bits, &objective, sizeof(objective_bits));
    std::cout << std::setprecision(17) << "solve_seconds " << seconds
              << " status " << glop::GetProblemStatusString(status)
              << " iterations " << iterations
              << " objective_bits " << objective_bits << '\n';
    return EXIT_SUCCESS;
  }
  const auto bits = [](double value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  if (argc == 2 || (argc == 3 && std::strcmp(argv[2], "presolve_only") == 0)) {
  std::cerr << "presolved " << glop::GetProblemStatusString(preprocessor.status())
            << ' ' << presolved.num_constraints().value()
            << ' ' << presolved.num_variables().value()
            << ' ' << presolved.num_entries().value() << '\n';
  for (int col = 0; col < presolved.num_variables().value(); ++col) {
    const glop::ColIndex c(col);
    std::cerr << "col " << col << ' ' << bits(presolved.objective_coefficients()[c])
              << ' ' << bits(presolved.variable_lower_bounds()[c])
              << ' ' << bits(presolved.variable_upper_bounds()[c]) << '\n';
    for (const auto e : presolved.GetSparseColumn(c)) {
      std::cerr << "entry " << col << ' ' << e.row().value() << ' '
                << bits(e.coefficient()) << '\n';
    }
  }
  for (int row = 0; row < presolved.num_constraints().value(); ++row) {
    const glop::RowIndex r(row);
    std::cerr << "row " << row << ' ' << bits(presolved.constraint_lower_bounds()[r])
              << ' ' << bits(presolved.constraint_upper_bounds()[r]) << '\n';
  }
  std::cerr << "offset " << bits(presolved.objective_offset()) << '\n';
  }
  if (argc == 3 && std::strcmp(argv[2], "presolve_only") == 0)
    return EXIT_SUCCESS;
  if (argc >= 3 && preprocessor.status() == glop::ProblemStatus::INIT) {
    parameters.set_max_number_of_iterations(std::atoi(argv[2]));
    glop::RevisedSimplex direct;
    g_diagnostic_direct = &direct;
    direct.SetParameters(parameters);
    operations_research::TimeLimit direct_limit(argc >= 5 ? std::atof(argv[4]) : 20.0);
    if (!direct.Solve(presolved, &direct_limit).ok()) return EXIT_FAILURE;
    if (argc >= 4 && std::strcmp(argv[3], "rng") == 0) {
      std::cout << "random_next " << diag_random_next() << '\n';
      return EXIT_SUCCESS;
    }
    if (argc >= 4 && std::strcmp(argv[3], "heap") == 0) {
      std::cout << "random_next " << diag_random_next() << "\nprice_tops";
      const auto& heap = (direct.*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
      for (const auto& top : heap.*Access(PriceTopsTag{}))
        std::cout << ' ' << top.index.value() << ':' << bits(top.value);
      std::cout << '\n';
      return EXIT_SUCCESS;
    }
    if (argc >= 4 && std::strcmp(argv[3], "clock") == 0) {
      std::cout << "clock " << bits(direct.DeterministicTime())
                << " updates " << direct.GetBasisFactorization().NumUpdates()
                << " basis_time " << bits(direct.GetBasisFactorization().DeterministicTime())
                << " refactorizations " << (direct.GetBasisFactorization().*Access(BasisStatsTag{})).refactorization_interval.Num()
                << '\n';
      return EXIT_SUCCESS;
    }
    if (argc >= 4 && std::strcmp(argv[3], "fingerprint") == 0) {
      uint64_t basis = 14695981039346656037ULL;
      uint64_t values = basis;
      uint64_t reduced = basis;
      for (int row = 0; row < presolved.num_constraints().value(); ++row)
        basis = (basis ^ static_cast<uint64_t>(direct.GetBasis(glop::RowIndex(row)).value())) * 1099511628211ULL;
      const int total = presolved.num_variables().value() + presolved.num_constraints().value();
      for (int col = 0; col < total; ++col) {
        const double value = direct.GetVariableValue(glop::ColIndex(col));
        const double cost = direct.GetReducedCost(glop::ColIndex(col));
        values = (values ^ bits(value == 0.0 ? 0.0 : value)) * 1099511628211ULL;
        reduced = (reduced ^ bits(cost == 0.0 ? 0.0 : cost)) * 1099511628211ULL;
      }
      std::cout << "fingerprint " << glop::GetProblemStatusString(direct.GetProblemStatus())
                << ' ' << direct.GetNumberOfIterations() << ' ' << basis << ' '
                << values << ' ' << reduced << ' ' << bits(direct.DeterministicTime())
                << ' ' << bits(direct.GetBasisFactorization().DeterministicTime())
                << ' ' << (direct.GetBasisFactorization().*Access(BasisStatsTag{})).refactorization_interval.Num()
                << '\n';
      return EXIT_SUCCESS;
    }
    std::cout << "direct " << glop::GetProblemStatusString(direct.GetProblemStatus())
              << ' ' << direct.GetNumberOfIterations() << '\n';
    std::cout << "basis";
    for (int row = 0; row < presolved.num_constraints().value(); ++row)
      std::cout << ' ' << direct.GetBasis(glop::RowIndex(row)).value();
    std::cout << "\nreduced";
    for (int col = 0; col < presolved.num_variables().value() + presolved.num_constraints().value(); ++col)
      std::cout << ' ' << bits(direct.GetReducedCost(glop::ColIndex(col)));
    std::cout << "\ncached_reduced";
    for (int col = 0; col < presolved.num_variables().value() + presolved.num_constraints().value(); ++col)
      std::cout << ' ' << diag_rc_value_bit(col);
    std::cout << "\nvalues";
    for (int col = 0; col < presolved.num_variables().value() + presolved.num_constraints().value(); ++col)
      std::cout << ' ' << bits(direct.GetVariableValue(glop::ColIndex(col)));
    std::cout << "\nstatuses";
    for (int col = 0; col < presolved.num_variables().value() + presolved.num_constraints().value(); ++col)
      std::cout << ' ' << glop::GetVariableStatusString(direct.GetVariableStatus(glop::ColIndex(col)));
    std::cout << "\nnorms";
    for (double norm : (direct.*Access(PrimalNormsTag{})).GetSquaredNorms())
      std::cout << ' ' << bits(norm);
    const auto& inverse = (direct.*Access(PrimalNormsTag{})).*Access(DirectionInverseTag{});
    std::cout << "\ninverse";
    for (double value : inverse.values) std::cout << ' ' << bits(value);
    std::cout << "\ninverse_order";
    for (const auto col : inverse.non_zeros) std::cout << ' ' << col.value();
    const auto& inverse_lu = direct.GetBasisFactorization().*Access(BasisLuTag{});
    std::cout << "\nlu_row_perm";
    const auto& lu_row_perm = inverse_lu.*Access(LuInverseRowPermutationTag{});
    for (int index = 0; index < lu_row_perm.size().value(); ++index)
      std::cout << ' ' << lu_row_perm[glop::RowIndex(index)].value();
    std::cout << "\nrandom_next " << diag_random_next() << "\nprice_tops";
    const auto& price_heap = (direct.*Access(PrimalPricesTag{})).*Access(PriceHeapTag{});
    for (const auto& top : price_heap.*Access(PriceTopsTag{}))
      std::cout << ' ' << top.index.value() << ':' << bits(top.value);
    if (argc >= 4 && std::strcmp(argv[3], "brief") == 0) {
      std::cout << '\n';
      return EXIT_SUCCESS;
    }
    const auto& update_row = direct.*Access(UpdateRowTag{});
    std::cout << "\nupdate_row";
    for (const auto col : update_row.GetNonZeroPositions())
      std::cout << ' ' << col.value() << ':' << bits(update_row.GetCoefficient(col));
    std::cout << "\ndirection_order";
    for (const auto row : (direct.*Access(DirectionTag{})).non_zeros)
      std::cout << ' ' << row.value();
    std::cout << "\ndirection_values";
    for (double value : (direct.*Access(DirectionTag{})).values)
      std::cout << ' ' << bits(value);
    if (argc >= 4 && std::strcmp(argv[3], "stages") == 0) {
      glop::ScatteredRow stage;
      stage.values = glop::Transpose((direct.*Access(DirectionTag{})).values);
      for (const auto row : (direct.*Access(DirectionTag{})).non_zeros)
        stage.non_zeros.push_back(glop::RowToColIndex(row));
      const auto& factorization = direct.GetBasisFactorization();
      const auto& lu = factorization.*Access(BasisLuTag{});
      const auto& updates = factorization.*Access(RankOneTag{});
      lu.LeftSolveUWithNonZeros(&stage);
      std::cout << "\nstage_upper";
      for (double value : stage.values) std::cout << ' ' << bits(value);
      std::cout << "\nstage_upper_order";
      for (const auto index : stage.non_zeros) std::cout << ' ' << index.value();
      updates.LeftSolveWithNonZeros(&stage);
      std::cout << "\nstage_updates";
      for (double value : stage.values) std::cout << ' ' << bits(value);
      std::cout << "\nstage_updates_order";
      for (const auto index : stage.non_zeros) std::cout << ' ' << index.value();
      lu.LeftSolveLWithNonZeros(&stage);
      std::cout << "\nstage_lower";
      for (double value : stage.values) std::cout << ' ' << bits(value);
      std::cout << "\nstage_lower_order";
      for (const auto index : stage.non_zeros) std::cout << ' ' << index.value();
      std::cout << '\n';
      return EXIT_SUCCESS;
    }
    std::cout << "\ndirection_precise_norm "
              << bits(1.0 + glop::SquaredNorm(direct.*Access(DirectionTag{})));
    glop::ScatteredColumn direction;
    direction.values.AssignToZero(presolved.num_constraints());
    for (const auto e : presolved.GetSparseColumn(glop::ColIndex(20))) {
      direction.values[e.row()] = e.coefficient();
      direction.non_zeros.push_back(e.row());
    }
    direct.GetBasisFactorization().RightSolve(&direction);
    std::cout << "\nsolve20";
    for (double value : direction.values) std::cout << ' ' << bits(value);
    if (presolved.num_variables().value() > 63) {
      glop::ScatteredColumn direction63;
      direction63.values.AssignToZero(presolved.num_constraints());
      for (const auto e : presolved.GetSparseColumn(glop::ColIndex(63))) {
        direction63.values[e.row()] = e.coefficient();
        direction63.non_zeros.push_back(e.row());
      }
      direct.GetBasisFactorization().RightSolve(&direction63);
      std::cout << "\nsolve63";
      for (double value : direction63.values) std::cout << ' ' << bits(value);
      std::cout << "\nsolve63_order";
      for (const auto row : direction63.non_zeros) std::cout << ' ' << row.value();
    }
    if (presolved.num_variables().value() + presolved.num_constraints().value() > 603) {
      glop::ScatteredColumn entering_direction;
      entering_direction.values.AssignToZero(presolved.num_constraints());
      if (presolved.num_variables().value() > 603) {
        for (const auto e : presolved.GetSparseColumn(glop::ColIndex(603))) {
          entering_direction.values[e.row()] = e.coefficient();
          entering_direction.non_zeros.push_back(e.row());
        }
      } else {
        const glop::RowIndex row(603 - presolved.num_variables().value());
        entering_direction.values[row] = 1.0;
        entering_direction.non_zeros.push_back(row);
      }
      direct.GetBasisFactorization().RightSolve(&entering_direction);
      std::cout << "\nsolve603";
      for (double value : entering_direction.values) std::cout << ' ' << bits(value);
      std::cout << "\nsolve603_order";
      for (const auto row : entering_direction.non_zeros) std::cout << ' ' << row.value();
    }
    glop::CompactSparseMatrix compact(presolved.GetSparseMatrix());
    const auto& native_lu = direct.GetBasisFactorization().*Access(BasisLuTag{});
    std::cout << "\nrenorm20 " << bits(native_lu.RightSolveSquaredNorm(compact.column(glop::ColIndex(20))));
    std::cout << "\nnonzeros20";
    for (const auto row : native_lu.*Access(LuNonZerosTag{})) std::cout << ' ' << row.value();
    auto& native_norms = direct.*Access(PrimalNormsTag{});
    native_norms.Clear();
    std::cout << "\nrenorms";
    for (double norm : native_norms.GetSquaredNorms()) std::cout << ' ' << bits(norm);
    std::cout << '\n';
    return EXIT_SUCCESS;
  }
  glop::LPSolver solver;
  operations_research::TimeLimit limit(20.0);
  const auto status = solver.SolveWithTimeLimit(lp, &limit);
  std::cout << glop::GetProblemStatusString(status) << ' '
            << solver.GetNumberOfSimplexIterations() << ' '
            << std::setprecision(17) << solver.GetObjectiveValue() << ' '
            << solver.GetMaximumPrimalInfeasibility() << ' '
            << solver.GetMaximumDualInfeasibility() << '\n';
}
