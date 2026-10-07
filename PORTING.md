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
| `lp_types.{h,cc}` | `lp_data/src/lp_types.rs` | in progress | Strong indices, typed dense vectors, statuses, and scalar helpers ported and unit-tested; bit vectors and sparse entry iterator remain |
| `sparse_vector.h` | `lp_data/src/sparse_vector.rs` | not started | Generic sparse vector |
| `sparse_column.{h,cc}` | `lp_data/src/sparse_column.rs` | not started | Column specialization |
| `sparse_row.h` | `lp_data/src/sparse_row.rs` | not started | Row specialization |
| `sparse.{h,cc}` | `lp_data/src/sparse.rs` | not started | Sparse matrix and transpose |
| `scattered_vector.h` | `lp_data/src/scattered_vector.rs` | not started | Reusable scattered workspace |
| `permutation.h` | `lp_data/src/permutation.rs` | not started | Typed permutations |
| `lp_data.{h,cc}` | `lp_data/src/lp_data.rs` | not started | Linear program model |
| `lp_data_utils.{h,cc}` | `lp_data/src/lp_data_utils.rs` | not started | Model utilities |
| `lp_utils.{h,cc}` | `lp_data/src/lp_utils.rs` | not started | Solver/model utilities |
| `matrix_utils.{h,cc}` | `lp_data/src/matrix_utils.rs` | not started | Matrix transformations/checks |
| `matrix_scaler.{h,cc}` | `lp_data/src/matrix_scaler.rs` | not started | Scaling algorithms |
| `mps_reader_template.{h,cc}` | `lp_data/src/mps_reader.rs` | not started | Core MPS parser; merge wrapper/template where appropriate |
| `mps_reader.{h,cc}` | `lp_data/src/mps_reader.rs` | not started | File/model adapters |
| `lp_parser.{h,cc}` | `lp_data/src/lp_parser.rs` | not started | LP text parser |
| `sol_reader.{h,cc}` | `lp_data/src/sol_reader.rs` | not started | Solution files |
| `lp_print_utils.{h,cc}` | `lp_data/src/lp_print_utils.rs` | not started | Deterministic formatting |
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
