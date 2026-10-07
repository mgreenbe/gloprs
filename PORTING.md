# Porting inventory

Upstream commit: `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`

Status values are `not started`, `in progress`, `ported`, and `validated`.
“Validated” requires relevant unit tests plus differential evidence where the
component affects solver behavior.

## `ortools/glop`

| Upstream source | Rust destination | Status | Notes |
|---|---|---|---|
| `basis_representation.{h,cc}` | `glop/src/basis_representation.rs` | not started | Basis factorization and updates |
| `dual_edge_norms.{h,cc}` | `glop/src/dual_edge_norms.rs` | not started | Dual pricing norms |
| `entering_variable.{h,cc}` | `glop/src/entering_variable.rs` | not started | Entering-variable selection |
| `initial_basis.{h,cc}` | `glop/src/initial_basis.rs` | not started | Crash/initial basis |
| `lp_solver.{h,cc}` | `glop/src/lp_solver.rs` | not started | Public solver orchestration |
| `lu_factorization.{h,cc}` | `glop/src/lu_factorization.rs` | not started | Sparse LU kernel |
| `markowitz.{h,cc}` | `glop/src/markowitz.rs` | not started | Markowitz pivoting |
| `parameters.proto` | `glop/src/parameters.rs` | not started | Replace protobuf with Rust types and identical defaults |
| `parameters_validation.{h,cc}` | `glop/src/parameters_validation.rs` | not started | Parameter validation |
| `preprocessor.{h,cc}` | `glop/src/preprocessor.rs` | not started | Presolve and postsolve stack |
| `pricing.h` | `glop/src/pricing.rs` | not started | Pricing rules/templates |
| `primal_edge_norms.{h,cc}` | `glop/src/primal_edge_norms.rs` | not started | Primal pricing norms |
| `rank_one_update.h` | `glop/src/rank_one_update.rs` | not started | Rank-one update helper |
| `reduced_costs.{h,cc}` | `glop/src/reduced_costs.rs` | not started | Reduced costs and dual feasibility |
| `revised_simplex.{h,cc}` | `glop/src/revised_simplex.rs` | not started | Core primal/dual simplex loop |
| `status.{h,cc}` | `glop/src/status.rs` | not started | Problem and variable statuses |
| `update_row.{h,cc}` | `glop/src/update_row.rs` | not started | Sparse update-row computation |
| `variable_values.{h,cc}` | `glop/src/variable_values.rs` | not started | Primal values and feasibility |
| `variables_info.{h,cc}` | `glop/src/variables_info.rs` | not started | Bounds and variable state |

Build metadata (`BUILD.bazel`, `CMakeLists.txt`) is deliberately replaced by
Cargo. The upstream README remains a behavioral/source guide rather than a file
to translate.

## `ortools/lp_data`

| Upstream source | Rust destination | Status | Notes |
|---|---|---|---|
| `lp_types.{h,cc}` | `lp_data/src/lp_types.rs` | validated | Strong indices, typed dense vectors and bit vectors, statuses, scalar helpers, and idiomatic sparse-entry iteration; unit-tested |
| `sparse_vector.h` | `lp_data/src/sparse_vector.rs` | validated | Insertion, cleanup, lookup, dense conversion/accumulation, component-wise operations, mutation, and typed permutation; unit-tested |
| `sparse_column.{h,cc}` | `lp_data/src/sparse_vector.rs` | validated | Column/view aliases and random-access sparse column with touched-row clearing; unit-tested |
| `sparse_row.h` | `lp_data/src/sparse_row.rs` | ported | Thin row specialization over the shared sparse-vector representation |
| `sparse.{h,cc}` | `lp_data/src/sparse.rs` | validated | Phase-1 column matrix, transpose, norms, deletion, and permutation operations; unit-tested |
| `scattered_vector.h` | `lp_data/src/scattered_vector.rs` | validated | Reusable dense values plus touched-index pattern; unit-tested |
| `permutation.h` | `lp_data/src/permutation.rs` | validated | Typed permutations, inverses, signatures, and vector application; unit-tested |
| `lp_data.{h,cc}` | `lp_data/src/lp_data.rs` | validated | Model, bounds, objective metadata, basis-bearing `ProblemSolution`, validation, and deterministic summaries; all Netlib parser fingerprints agree with native GLOP |
| `lp_data_utils.{h,cc}` | integrated into `lp_data/src/lp_data.rs` | ported | Phase-1 validation and summary subset; remaining transformation utilities stay deferred until consumers enter scope |
| `lp_utils.{h,cc}` | `lp_data/src/lp_utils.rs` | not started | Solver/model utilities |
| `matrix_utils.{h,cc}` | `lp_data/src/matrix_utils.rs` | not started | Matrix transformations/checks |
| `matrix_scaler.{h,cc}` | `lp_data/src/matrix_scaler.rs` | not started | Scaling algorithms |
| `mps_reader_template.{h,cc}` | `lp_data/src/mps_reader.rs` | validated | Fixed/free scanner, rows, columns, RHS, ranges, bounds, objective sense, markers, duplicate entries, and numerical errors; 98/98 Netlib models agree with native GLOP dimensions, nonzeros, bounds, and objectives |
| `mps_reader.{h,cc}` | `lp_data/src/mps_reader.rs` | validated | File and string adapters; unit and full-Netlib differential tests |
| `lp_parser.{h,cc}` | `lp_data/src/lp_parser.rs` | not started | LP text parser |
| `sol_reader.{h,cc}` | `lp_data/src/sol_reader.rs` | not started | Solution files |
| `lp_print_utils.{h,cc}` | integrated into `lp_data/src/lp_data.rs` and `cli/src/main.rs` | ported | Deterministic Phase-1 model summary and data fingerprint; full upstream LP emission remains deferred |
| `lp_decomposer.{h,cc}` | `lp_data/src/lp_decomposer.rs` | not started | Independent component decomposition |
| `proto_utils.{h,cc}` | focused model conversion, if needed | not started | Do not reproduce protobuf infrastructure |

## Direct supporting dependencies to assess

The initial include inventory found these non-GLOP/non-`lp_data` dependencies:

- `ortools/algorithms/dynamic_partition.h`
- `ortools/base/accurate_sum.h`
- `ortools/base/hash.h`
- `ortools/base/iterator_adaptors.h`
- `ortools/base/numbers.h`
- `ortools/base/strong_vector.h`
- `ortools/base/timer.h`
- `ortools/graph/iterators.h`
- `ortools/util/bitset.h`
- `ortools/util/fp_utils.h`
- `ortools/util/random_engine.h`
- `ortools/util/rational_approximation.h`
- `ortools/util/stats.h`
- `ortools/util/strong_integers.h`
- `ortools/util/time_limit.h`

Logging, status macros, file helpers, protobuf helpers, and generated protobuf
headers are infrastructure replacements, not translation targets. Each source
module must still be checked for subtle semantics supplied by them.

## Test inventory caveat

The pinned public tree does not contain the historical file-local unit tests for
most `ortools/glop` and `ortools/lp_data` components. Available integration tests
and samples must be supplemented by differential tests against the native GLOP
binary and by Rust invariant/randomized tests. If corresponding upstream tests
are found in another public revision, record their exact provenance before
porting them.
