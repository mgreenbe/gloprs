# Phase 8 porting ledger: remaining fidelity and performance work

Pinned upstream: `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`.
Completed file-by-file correspondence and validation history is retained in
Git history. This ledger tracks only live Phase 8 follow-ups; a performance
gap alone does not revoke existing behavioral validation.

| Upstream source | Rust destination | Remaining question / evidence needed |
|---|---|---|
| `revised_simplex.{h,cc}` | `glop/src/revised_simplex.rs` | Initial basic-value recomputation now follows upstream's nonbasic-bitset traversal, fills the scattered solve workspace directly, and skips zero-multiplier columns via the compact add helper. Fast-96 strict trajectories and selected native model outcomes pass. The focused `neos-1445765` solve ratio is still 1.27x; its early-solve/pricing excess and `graph20-20-1rand`'s 1.10x accumulated pivot-loop excess need matched-stage investigation. |
| `primal_edge_norms.{h,cc}` | `glop/src/primal_edge_norms.rs` | Separate edge-norm BTRAN and update cost from the rest of the `fhnw` primal-phase loop; compare matching native/Rust inputs and workspace reuse. |
| `update_row.{h,cc}` | `glop/src/update_row.rs` | The compact column-wise path now recognizes the exact trailing equation-form identity block once on binding and reads its coefficient directly, matching upstream's slack path. Focused unit, fast-96 trajectory, and 213-model public status/iteration checks pass; `cvs16r128-89` public timing improved 1.093x to 1.059x. Revisit row-wise/single-row paths only where matched profiles implicate them. |
| `basis_representation.{h,cc}`, `lu_factorization.{h,cc}`, `markowitz.{h,cc}` | `glop/src/{basis_representation,lu_factorization,markowitz}.rs` | Selected-view `fhnw` LU is near parity; investigate these paths again only where whole-solve or new selected-view probes show a material gap. Keep source-column entry order, pivot ties, and retained workspace semantics. |
| `lp_solver.{h,cc}`, `preprocessor.{h,cc}` | `glop/src/{lp_solver,preprocessor}.rs` | Recheck parse/presolve/postsolve shares for remaining public-pipeline outliers. Keep native presolved-model and public KKT gates; do not attribute public time to simplex without stage evidence. |
| `sparse.{h,cc}`, `sparse_vector.{h,cc}` and compact transpose | `lp_data/src/{sparse,sparse_vector}.rs` | Retained column and transpose-start allocations now match upstream. Maintain allocation-identity and ordered-entry tests; revisit only if new profiles show a remaining copy or storage-lifecycle mismatch. |

The fast-96 Netlib strict trajectory and presolve fixtures pass. The
164-model MIPLIB/Mittelmann gate currently checks completed-model outcomes,
status, and iteration count, but lacks strict pivot-by-pivot fixtures. Add
those for selected outliers as described in `PLAN.md`. Exact latest timings
and raw-artifact references are in `baselines/phase8-extra-performance.md`;
older experiments and completed port rows are available through Git history.
