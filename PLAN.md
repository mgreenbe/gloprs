# Phase 8: performance parity

**Status: complete (15 October 2026).** The port remains pinned to GLOP
`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`. Preserve algorithm,
operation order, numerical safeguards, sparse representations, and workspace
lifecycle even when a fidelity repair is performance-neutral. Full historical
plans and experiments remain in Git history and `baselines/`.

## Completion evidence

- Two clean, serial 213-model `nmm0-50k` public-pipeline passes with freshly
  rebuilt adapters, alternating three timed trials per side, and no wall
  limits give 0.960x and 0.962x Rust/native summed time. All statuses,
  iteration counts, and objectives agree. Median model ratios are 0.936x and
  0.931x; sqrt(nnz)-weighted geometric ratios are 0.948x and 0.946x.
- The current extra-corpus pass completed 162/166 models under a 20-second
  cap at 0.956x summed time. Two more completed under a 60-second cap, giving
  the same 164 completed models as the earlier full baseline; the other two
  remain known long-limit cases. All completed outcomes match pinned native.
  Background macOS analysis/indexing affected the broad run, so its small
  per-model timing differences are diagnostic, not precise. An independent
  164-model native-outcome validator passes.
- The only repeatable excess above 1 ms in both clean 213-model passes is
  `graph20-20-1rand`, +3.65/+2.96 ms (1.024x/1.020x). A 15-pair stage check
  localizes it to the presolved simplex solve (1.021x), not parsing or
  presolve. A quiet 15-pair extra-corpus recheck finds `graph` at 1.023x
  (+3.54 ms) and `neos-1171737` at 1.022x (+6.10 ms); the other leading
  apparent outliers, `momentum1` and `neos-1171448`, are both 1.002x.
- Workspace tests, Clippy, bit-exact fast-96 presolve, strict fast-96 pivot
  trajectories, six selected extra-model pivot paths, and public postsolve/
  KKT checks pass. The MPS parser also matches pinned native on 1,000 free
  and 1,000 fixed randomized cards in all three parser modes.

The benchmark commands, per-model trials and dispersion, binary SHA-256s,
timing caveats, and earlier repairs are documented in
`baselines/phase8-nmm0-50k-performance.md` and
`baselines/phase8-extra-performance.md`.

## Follow-ups outside Phase 8

- Keep the differential trajectory, presolve, numerical-quality, and KKT
  gates for subsequent changes.
- Revisit the residual roughly 2% solve costs on `graph20-20-1rand` and
  `neos-1171737` if profiling identifies a concrete upstream operation
  mismatch. These are small, localized costs, not material Phase 8 blockers.
- For a future precise per-model extra-corpus timing claim, rerun the broad
  campaign while media analysis and indexing stay idle. The current pass is
  sound for outcomes and broad triage, supplemented by the quiet focused
  recheck.


## Follow-up: iterator and counted-loop audit (15 October 2026)

Completed the core iterator audit and two full paired 213-model nmm0-50k
public-pipeline campaigns (five timed pairs per model), followed by isolated
builds and 11/15-pair outlier rechecks. Both candidate production rewrites were
reverted: counted packed rank-one products regressed CMS750_4 and greenbea;
counted pricing-heap construction regressed cvs16r128-89. Small aggregate gains
(~0.33--0.34%) did not justify these repeatable per-model costs.

Production code is unchanged from `77b83ac`. Preserve the audit, raw timings,
binary fingerprints, and reusable benchmark harnesses in
`baselines/iterator-audit-2026-10-15.md`. Ordinary exclusive ranges and compact
zip iteration were not generally slower; do not replace iterators wholesale.
Phase 8 remains complete with its existing validated implementation.

## Targeted graph/neos investigation (15 October 2026)

Benchmark mode was already on and verified against live worker/index/sleep
state before measurements and again before the full corpus. Quiet-mode
native-relative solve baselines remain 1.0192x graph / 1.0270x neos.
A retained SparseColumn for Harris candidates improved focused solves, but
its full 213-model A/B gate exposed a repeatable CVS slowdown (+1.85%,
+31.0 ms on an independent eleven-pair recheck). A boxed form also regressed
CVS. Both were reverted; the signed-row-pointer and 32-bit filtered-row
prototypes were rejected earlier for neos regressions.

The full campaign completed all 213 models with five timed pairs apiece,
no timeouts, and all 1,065 outcomes bit-exact against the baseline. A small
0.23% aggregate gain did not override the per-model regression. Production
code remains unchanged from `2f493f2`; results, fingerprints, profiles, and
correctness gates are recorded in `baselines/gaps-2026-10-15.md`. Benchmark
mode remains on as it was when this investigation began.

## NMM 50k–100k baseline (15 October 2026)

Completed all 25 models against pinned native GLOP under verified benchmark
mode, five timed pairs per model, one warmup, 60-second solve limits. All 125
status/iteration/objective-bit pairs matched exactly; no timeouts. Rust/native
aggregate time is 0.95058x, median model 0.96124x. A separate 15-pair recheck
confirmed baseline gaps on piperout-27 (1.02016x, +3.86 ms) and
neos-1171737 (1.02972x, +8.16 ms), with all 30 outcome pairs exact. No solver
changes were made. See `baselines/nmm50k-100k-2026-10-15.md` and raw trials.

## NMM 100k–150k baseline (15 October 2026)

Processed all 20 models: 19 completed, qap15 limited at 60 seconds on both
solvers. All 95 completed timed pairs were exact on status/iterations/objective
bits. Rust/native aggregate 0.95186x, median model 0.96418x, excluding qap15.
A 15-pair recheck confirmed six slower cases; leading gaps are neos-950242
(1.15330x, +10.76 ms) and fit2d (1.02580x, +28.17 ms). All 90 recheck pairs
were exact. The harness now skips completed Netlib fixture assertions for
wall-limited cases; the full run was restarted after that correction. No
solver changes. See `baselines/nmm100k-150k-2026-10-15.md`.

## neos-950242 parity repair (15 October 2026)

Retained the upstream doubleton-equality transpose-cache lifetime: changed
passes now return stale storage for lazy rebuilding instead of dropping its
sparse-column allocations. Target final native/Rust public medians are
71.680/68.041 ms (0.94922x), closing the earlier 15.3% deficit. Two full
213-model five-pair nmm0-50k campaigns completed without limits, all 1,065
outcomes exact each; final aggregate 0.99529x. Small per-model costs remain,
notably CMS750_4 around 1%; retained for concrete upstream storage fidelity,
large target gain, and full-corpus aggregate benefit. Workspace, Clippy,
fast-96 public postsolve/KKT, target presolve bits, and 21 pinned checkpoints
pass. See `baselines/neos950242-parity-2026-10-15.md`.
