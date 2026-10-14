# Phase 8 porting ledger: remaining fidelity and performance work

Pinned upstream: `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`.
Completed file-by-file correspondence and validation history is retained in
Git history. This ledger tracks only live Phase 8 follow-ups; a performance
gap alone does not revoke existing behavioral validation.

| Upstream source | Rust destination | Remaining question / evidence needed |
|---|---|---|
| `revised_simplex.{h,cc}`, `pricing.h`, `reduced_costs.{h,cc}` | `glop/src/{revised_simplex,pricing,reduced_costs}.rs`, `lp_data/src/lp_types.rs` | Initial basic-value recomputation follows upstream's nonbasic-bitset traversal and scattered workspace. The two-mask pricing membership check reads validated buckets directly like upstream. The incremental candidate-update loop remains out of line like pinned GLOP. Structural LP initialization borrows the model and augments only bounds and compact-matrix slacks, matching upstream's direct construction rather than cloning/cleaning the full LP. Initial-basis bounds are borrowed views and crash/recovered basis buffers are transferred in place; `VariablesInfo` is backed up only for TRIANGULAR retry. Virtual-slack and state tests, fast-96 strict trajectories, and native presolve fixtures pass. The structural-view repair reduced the cap-1 `neos-1445765` ratio from 2.657x to 1.401x and its public-pipeline ratio to 1.037x; remaining solve excess is downstream. |
| `primal_edge_norms.{h,cc}` | `glop/src/primal_edge_norms.rs` | The sparse direction-left-inverse reset uses upstream's thresholded clear-and-resize helper. Edge-norm operation accounting now uses the direct compact-column entry count rather than constructing a column view, matching GLOP's `ColumnNumEntries()` call. Two alternating matched A/B runs show a small consistent solve gain on `neos-1445765` and `graph20-20-1rand`; strict fast-96 trajectories pass. Still separate edge-norm BTRAN and update cost from the rest of the `fhnw` primal-phase loop using matched inputs. |
| `update_row.{h,cc}` | `glop/src/update_row.rs` | The compact column-wise path now recognizes the exact trailing equation-form identity block once on binding and reads its coefficient directly, matching upstream's slack path. Focused unit, fast-96 trajectory, and 213-model public status/iteration checks pass; `cvs16r128-89` public timing improved 1.093x to 1.059x. Revisit row-wise/single-row paths only where matched profiles implicate them. |
| `basis_representation.{h,cc}`, `lu_factorization.{h,cc}`, `markowitz.{h,cc}` | `glop/src/{basis_representation,lu_factorization,markowitz}.rs` | Selected-view `fhnw` LU is near parity; investigate these paths again only where whole-solve or new selected-view probes show a material gap. Keep source-column entry order, pivot ties, and retained workspace semantics. |
| `lp_solver.{h,cc}`, `preprocessor.{h,cc}` | `glop/src/{lp_solver,preprocessor}.rs` | Recheck parse/presolve/postsolve shares for remaining public-pipeline outliers. Keep native presolved-model and public KKT gates; do not attribute public time to simplex without stage evidence. |
| `sparse.{h,cc}`, `sparse_vector.{h,cc}`, typed vectors and compact transpose | `lp_data/src/{sparse,sparse_vector,lp_types}.rs` | Explicit `Clone::clone_from` on sparse columns and typed metadata vectors retains destination buffers, matching upstream assignment. The structural compact-matrix/slack builder pre-sizes and fills its three arrays directly like GLOP, rather than growing them entry-by-entry; repeated cap-1 A/B runs show a small startup gain. Keeping the sparse-column merge out of line like pinned GLOP brings `neos-1445765` presolve to 0.997x native in serial matched trials, with bit-exact fast-96 presolve fixtures. Remaining solve gaps are separate from matrix copying and presolve. |
| `lp_data/sparse.{h,cc}` triangular solves | `lp_data/src/triangular_matrix.rs` | The sorted symbolic closure uses a retained 64-bit row bitset and clears touched buckets like upstream. Dense lower/upper solves now match GLOP's template-specialized unit-diagonal dispatch and direct traversal of contiguous entry arrays. Checked square/RHS invariants justify the explicitly commented unchecked inner accesses. Paired Rust A/B solves improve `milo` about 2% and `cvs` about 0.6%, leaving `graph` neutral; four-model native pivot and fast-96 strict trajectory gates pass. |

The fast-96 Netlib strict trajectory and presolve fixtures pass. The
direct-dual Phase 8 extra-corpus pivot fixture also passes for
`neos-1445765` and `graph20-20-1rand` (6,825 pivots combined). The
164-model MIPLIB/Mittelmann gate currently checks completed-model outcomes,
status, and iteration count, but lacks strict pivot-by-pivot fixtures. Add
those for selected outliers as described in `PLAN.md`. Exact latest timings
and raw-artifact references are in `baselines/phase8-extra-performance.md`;
older experiments and completed port rows are available through Git history.
