# Phase 8: performance parity

The goal is a faithful Rust port of pinned GLOP
(`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`) with comparable performance.
Preserve algorithms, sparse representations, operation order, numerical
safeguards, and workspace lifecycles. Keep changes that improve fidelity even
when timing-neutral; do not trade trajectory or KKT agreement for speed.

## Current baseline

- The M2-targeted, serial `nmm0-50k` run covers all 213 deduplicated models
  with matching status and iteration count: 1.010x aggregate and 1.023x
  sqrt(nnz)-weighted Rust/native public-solve time. The largest repeatable
  matched-solve excesses are `cvs16r128-89` (+130 ms), `dfl001` (+92 ms),
  `d6cube` (+34 ms), `rmatr100-p10` (+31 ms), and `graph20-20-1rand`
  (+16 ms). `neos-1445765` has separate presolve and solve gaps. See
  `baselines/phase8-nmm0-50k-performance.md` and its raw JSON artifacts.
- After restoring GLOP's direct trailing-slack update-row lookup, a second
  uncontended 213-model run matched all statuses and iteration counts. Its
  aggregate ratio is 1.005x (median 1.011x; sqrt(nnz)-weighted 1.024x).
  `cvs16r128-89` improved from 1.093x to 1.059x in the public run, while
  `d6cube`, `graph20-20-1rand`, and `neos-1445765` remain material targets.
  See the addendum in `baselines/phase8-nmm0-50k-performance.md`.
- Focused seven-trial matched-stage rechecks put `neos-1445765` at about
  1.21x presolve and 1.28x solve after correcting the diagnostic adapter's
  Rust model copy to `PopulateFromLinearProgram`. `graph20-20-1rand` is about
  0.94x presolve and 1.11x solve. Pivot-capped runs localize most `neos` solve
  excess before pivot 100, whereas `graph` excess grows through pivot 3092.
  Direct scattered-RHS/nonbasic-bitset initialization and GLOP's zero-
  multiplier column skip improve upstream correspondence. The final focused
  recheck is 1.21x `neos` presolve, 1.27x `neos` solve, and 1.10x `graph`
  solve. Both remain open performance targets.
- A matched repeated-copy probe found a Rust ownership/lifecycle gap in the
  LP metadata wrappers: derived `Clone::clone_from` discarded destination
  `TypedVec` and sparse-column allocations. Explicit `clone_from` now reuses
  them, mirroring upstream vector assignment. On `neos-1445765`, repeated
  whole-LP copy fell from about 1.03 ms to 0.25 ms per copy (native 0.58 ms);
  on `graph20-20-1rand`, from 0.245 ms to 0.064 ms (native 0.084 ms).
  Matrix-only copies were already close. Isolated matched-presolve and solve
  timings remain about 1.20x/1.27x on `neos` and 0.94x/1.10x on `graph`;
  their dominant gaps are therefore still open.
- A follow-up edge-norm audit found one exact primitive mismatch: Rust used a
  general scattered-vector clear where GLOP calls its thresholded
  `ClearAndResizeVectorWithNonZeros()`. Rust now calls the existing faithful
  helper. The fast-96 trajectory gate passes; serial seven-trial solve ratios
  remain 1.267x on `neos-1445765` and 1.113x on `graph20-20-1rand`, so the
  change is not a meaningful timing repair. The `neos` gap is concentrated in
  initial solve/pricing work; `graph` remains distributed across repeated
  pivot kernels, rather than this single reset.
- The primal-pricing two-mask test now uses direct bucket reads after a safe
  release-mode length assertion, matching upstream's bucket access without
  redundant physical-vector checks. Two 15-trial alternating A/B runs against
  the saved pre-edit Rust binary put `graph20-20-1rand` at 0.989x and
  `neos-1445765` at 0.991x/0.995x. The earlier debug-assert-only variant was
  discarded because a public safe Rust API must not permit unchecked access
  on invalid inputs. The final safe variant passes the 67 `lp_data` tests,
  strict fast-96 trajectory gate, and Clippy. This is a small repair, not
  resolution of either model's gap.
- After these edits, a serial 15-trial native-relative public-pipeline check
  is 1.204x for `neos-1445765` and 1.105x for `graph20-20-1rand`, with
  matching statuses and iteration counts. Two alternating A/B runs against a
  saved pre-edit Rust public binary show only about 0.3-0.4% whole-pipeline
  improvement. Reusing the model-copy destination substantially helps repeated
  copies, but a fresh one-shot public solve does not realize that full gain.
  A direct statistic-slot experiment and a Harris-ratio slice-binding
  experiment did not help `graph` and were reverted.
- A further compiled-boundary audit found that Rust inlined the sparse-column
  merge into the doubleton preprocessor, while pinned GLOP keeps that merge
  out of line. Keeping `add_multiple_to_sparse_vector_with_common_index`
  out of line reduces `neos-1445765` presolve to 0.997x native in 15 serial
  matched trials (0.822x the saved pre-edit Rust binary). The corresponding
  public-pipeline ratios are 1.085x for `neos` and 1.119x for `graph`, with
  matching statuses and iterations. Presolved solves remain 1.268x and
  1.105x respectively. The fast-96 bit-exact presolve and strict trajectory
  gates pass. An analogous Harris function-boundary experiment slowed the
  selected solves slightly and was reverted. The `neos` presolve gap is
  resolved; its initial solve/pricing gap remains.
- The latest serial, alternating 164-model MIPLIB/Mittelmann public-solver run
  is 0.992x Rust/native in aggregate and 1.013x by median model ratio; all
  completed models match status and iteration count. Two wall-limited models
  are excluded. See `baselines/phase8-extra-performance.md` and the raw
  `target/phase8-extra-public-top10-repairs.json` when available.
- Strict fast-96 Netlib trajectories and the native presolve fixtures pass.
  Isolated selected-view LU on `fhnw-binpack4-48` is about 1.025x native, but
  its matched whole solve remains about 1.12x. The remaining cost is outside
  isolated factorization.
- `qap15` has a pinned 40,000-pivot state/clock check and late-basis probes.
  Its long direct solve is not a deterministic Phase 8 wall-time gate; keep
  investigating measured regressions without making Phase 8 completion depend
  on a full `qap15` solve.

## Remaining work

1. Localize the largest repeatable *absolute* extra-corpus excesses with
   paired, matched-input stage measurements before changing kernels:
   `cvs16r128-89`, `neos-3216931-puriri`, `neos-1171737`, and
   `graph20-20-1rand`. Recheck high-ratio small models such as `neos-950242`
   with enough repetitions to separate signal from timer noise. The current
   samples distribute time across triangular solves, update rows, ratio tests,
   pricing, and edge norms; they do not identify a single universal bottleneck.
   For `nmm0-50k`, start with matched native/Rust stage accounting on
   `cvs16r128-89`, then `dfl001`, `d6cube`, `rmatr100-p10`, and
   `graph20-20-1rand`; profile the remaining `neos-1445765` solve gap.
2. Account for the remaining `fhnw-binpack4-48` whole-solve gap with matched
   primal-phase and BTRAN/edge-norm stage timings. Do not infer a need to
   rewrite LU from whole-solve samples: selected-view LU is already close to
   parity. For every concrete discrepancy, compare the pinned C++ path and
   Rust path, then repair representation or lifecycle differences first.
3. Add a strict pivot-by-pivot native trajectory fixture for the selected
   extra-corpus models. The existing 164-model gate checks public outcomes,
   status, and iteration count, not every pivot. Preserve the fast-96 strict
   trajectory, public postsolve/KKT, presolve, and focused numerical-kernel
   gates after each performance edit.
4. Repeat serial paired stage and public-pipeline benchmarks after repairs.
   Report per-model timings, absolute excesses, median and size-weighted
   ratios, trial dispersion, build fingerprints, and timeouts. Keep benchmark
   inputs and settings equivalent to pinned upstream GLOP.

## Completion criterion

Phase 8 is complete when the comparison is reproducible, material regressions
are fixed or localized and explained with evidence, and differential
trajectory/numerical-quality gates still pass. Historical implementation and
measurement detail remains in Git history and `baselines/`.
