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
   `graph20-20-1rand`; separately profile `neos-1445765` presolve and solve.
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
