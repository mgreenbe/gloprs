# Phase 8 allocation audit (14 October 2026)

The initial audit searched `glop/src` and `lp_data/src` for `Vec::new()`, `vec!`,
`Vec::with_capacity()`, `to_vec()`, and vector `collect()` sites. Each
constructor on the revised-simplex loop, basis-update, sparse-solve, pricing,
and Markowitz paths was checked against the corresponding pinned GLOP source
(`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`). Constructor syntax alone
does not imply an allocation: many hits initialize retained fields, occur
outside the pivot loop, or are tests and diagnostics.

A follow-up scan covered `::new()`, `::default()`, `::with_capacity()`,
`vec!`, `.clone()`, `.to_vec()`, `.to_owned()`, and `.collect()` syntax in
`glop/src` and `lp_data/src`. The hits were grouped by module and the
numerical hot-path candidates were inspected in context; this did not
individually review every parser, test, diagnostic, or preprocessor call.
It found that basis permutation at refactorization
cloned the basis, dual Phase-I prices, and factorization permutation, then
built another typed permutation for edge norms. Pinned GLOP uses retained
`tmp_basis_` and `tmp_dual_pricing_vector_` and borrows the permutation. Rust
now uses retained swap buffers and a borrowed slice. This is a source-level
audit, not proof that allocation profiling has ruled out every hidden heap
operation. The edited path passes `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, the workspace tests, and the release fast-96
native trajectory fixture.

| Path | Pinned GLOP | Rust audit result |
|---|---|---|
| Dual Phase-I reduced-cost price update | `RevisedSimplex::initially_all_zero_scratchpad_` survives calls; only changed-price updates prepare it; dense or touched sparse values are reset afterward. | Replaced a new `ScatteredColumn` per call with a solver-owned workspace, including GLOP's lazy preparation and clear-on-consume protocol. The full price rebuild also uses retained dense-row `AssignToZero` storage. |
| Dual Phase-I pivot price update | `UpdateRow::GetNonZeroPositions()` is passed as a span. | Removed the per-pivot `to_vec()` by temporarily borrowing the self-contained update-row cache; it is restored even if repricing fails. |
| Primal price recomputation | `PrimalPrices::GetBestEnteringColumn()` reads `ReducedCosts::GetReducedCosts()` by view. | Removed a full reduced-cost-row clone at every price recomputation. |
| Primal and dual phase loops | Direction, Harris candidates, breakpoint, and pending-price workspaces are created outside the iteration loop and reused. | Matching retained loop-local workspaces; their constructor calls are not per-pivot allocations. |
| Markowitz LU | Queue, residual pattern, candidate and factor buffers survive refactorizations. | Matching retained `MarkowitzWorkspace`; local boolean/candidate vectors in initial-basis construction are not allocated for each numerical refactorization. |
| `GetColumnOfU()` | LU-owned mutable sparse-column cache. | Aligned in the preceding upper-column change; no tuple-vector allocation per basis update. |
| Sparse-vector cleanup | A temporary array-of-pairs is allocated for stable sorting in `SparseVector::CleanUp()`. | Rust's temporary entry vector mirrors this upstream allocation. |
| Update-row rebinding | The retained sparse-position bitset is cleared and resized. | Rebinding now calls `clear_and_resize()` on the existing `ColBitVec` rather than allocating a replacement. |
| Diagnostic and probe hooks | Additional output storage is conditional on tracing/probing. | `probe_entries()`, residual debug assertions, trace snapshots, and public dense-returning helpers do not allocate on ordinary release solves. |
| Presolve passes and MPS/Matrix Market loading | Local pass-specific arrays are materialized and discarded; parsers have separate storage lifetimes. | Constructor hits in these paths are not simplex per-pivot work. Their individual reuse policies should be checked when a presolve or parse profile, rather than a numerical-solve profile, implicates them. |

The scan also identified a full relevant-column `Vec` on a *from-scratch*
dual Phase-I price rebuild and boxed-variable lists at phase transitions.
These are not per-pivot on the common path, but GLOP traverses bitset views
instead. Eliminating those remaining copies requires splitting the Rust
borrow of `VariablesInfo` from the mutable pricing state; it is a follow-up
fidelity opportunity, not a claimed completed alignment.

Two serial seven-trial Rust A/B comparisons against the pre-audit binary on
matched presolved solves gave ratios of 1.000x/0.999x for `milo`,
0.999x/0.998x for `graph`, 0.990x/0.998x for `cvs`, and 0.986x/0.981x for
`neos-1445765`. A subsequent uncontended run after GLOP-style sparse
scratchpad cleanup gave 1.002x, 1.001x, 0.990x, and 1.009x respectively;
the `neos` gain is therefore not yet repeatable. A final run after retained
update-row bitset rebinding gave 1.003x, 0.998x, 0.991x, and 0.998x. These
small measurements do
not establish parity with native GLOP. Strict fast-96
trajectories, the selected extra-model pivot fixture, and 14 presolved-state
fingerprints pass. A 213-model public check has matching statuses and
iteration counts with no timeouts (one-bit `capri` objective difference as
before); the two single-pass aggregates, before and after the bitset-rebind
edit, are 0.987x and 0.992x Rust/native. Neither is a reliable isolated
measure of the allocation edits.
