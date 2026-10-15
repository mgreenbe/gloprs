# Phase 8: performance parity

The goal is a faithful Rust port of pinned GLOP
(`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`) with comparable performance.
Preserve algorithms, sparse representations, operation order, numerical
safeguards, and workspace lifecycles. Keep changes that improve fidelity even
when timing-neutral; do not trade trajectory or KKT agreement for speed.

## Current baseline

- A fresh M2-targeted 213-model `nmm0-50k` pass after the indexed-access
  audit matches every completed status and iteration count at 0.976x total
  Rust/native time, but a second pass was invalidated by a nonterminal native
  `pilot` result under visible machine load. Seven-trial matched-stage runs
  still put `cvs` at 1.039x, `graph` at 1.070x, and `traininstance2` at
  1.083x native solve time. An unchecked read of `UpdateRow`'s own reported
  coefficients aligns the primal edge/Devex loop with upstream; its paired
  A/B effect is neutral except a ~0.6% `cvs` gain. Strict fast-96 and
  extra-model pivot fixtures pass. See the latest addendum in
  `baselines/phase8-nmm0-50k-performance.md`; `graph` and the other matched
  solve gaps remain open, and the complete timing pass needs an uncontended
  repeat.

- An indexed-access audit compared hot numerical loops with pinned GLOP.
  Primal/dual edge-norm updates and column-wise update-row output now use
  documented unchecked access only after their dimensions are established;
  caller-controlled public inputs remain checked. Two paired old/new Rust
  runs are neutral to about 1% slower, so this is a fidelity alignment rather
  than a performance fix. See `baselines/phase8-bounds-audit.md`.

- A profiled LU scattered-permutation access change removed redundant Rust
  bounds checks while preserving a release-mode check for caller-mutable
  sparse positions. Two paired matched-solve A/B runs improve `cvs`, `rmatr`,
  and `graph`; the final 213-model native-relative ratio is 0.9756x with
  matching status/iterations. `rmatr` is now 0.949x native in seven trials;
  `cvs` (1.023x, +38 ms) and `graph` (1.067x, +10 ms) remain the principal
  per-instance targets. See `baselines/phase8-nmm0-50k-performance.md`.

- A broader constructor/copy-site scan found that basis permutation on
  refactorization cloned the basis, dual Phase-I prices, and permutation.
  It now uses GLOP-style retained temporary vectors and a borrowed permutation
  for edge norms. See the allocation audit for scope and validation.

- The allocation audit found and repaired three hot-path lifetime gaps:
  GLOP's retained dual Phase-I scattered RHS, its borrowed update-row
  positions on dual pivots, and direct reduced-cost views during primal
  price recomputation. Update-row rebinding also retains its bitset buckets.
  Paired Rust A/B runs show neutral `milo`/`graph`, small noisy gains on
  `cvs`, and an inconsistent `neos-1445765` effect (0.981–1.009x). Strict
  trajectories and presolved checkpoints pass. Remaining bitset-view copies
  are catalogued in
  `baselines/phase8-allocation-audit.md`.
- A pivot-loop clock-fidelity repair replaces Rust's 100-element scan per
  limit check with GLOP's `RunningMax` ring and reuses the same clock sample
  for the deadline check. Paired Rust A/B solves improve `graph` about
  0.4–0.9% and `milo` about 0.5–0.7%; `cvs` is neutral. Fourteen capped
  presolved-state fingerprints for those three models are bit-exact against
  native, including basis, values, reduced costs, deterministic clocks, and
  refactorization counts. Remaining gaps are pivot-time performance, not
  observed path divergence at those checkpoints.
- Solver-owned middle-product updates now append and consume the right
  scratchpad's touched rows in upstream order, including duplicates, and
  clear values in the compact-column append. This removes the Rust-only
  sparse-mask rebuild. Paired matched solves are now about 1.057x native for
  `milo`, 1.086x for `graph`, and 1.042x for `cvs`; the remaining pivot-time
  gaps still need localization.
- `GetColumnOfU()` now reuses an LU-owned `SparseColumn`, as upstream does,
  instead of allocating a new tuple vector at each basis update. Its cleanup
  and entry order match the native path; direct pivot and presolved-state
  fixtures pass. Two matched Rust A/B runs show no material whole-solve gain
  on `milo` or `graph`, and about 0.5–0.6% slower `cvs`; retain the change
  for ownership/workspace fidelity while investigating the remaining gap.
- The newest triangular-solve alignment mirrors GLOP's compile-time
  unit-diagonal dispatch and contiguous indexed traversal. `milo` matched
  solve improved about 2% but remains 1.066x native; `graph` remains 1.097x
  and `cvs` 1.052x. Presolved selected-view LU is 1.084x on `milo` but only
  about 1 ms of its excess, 1.021x on `graph`, and 0.957x on `cvs`; next
  investigate their pivot-time solves, update row, ratio test, and pricing
  with matched native/Rust inputs. The four-model native pivot fixture,
  fast-96 trajectories, and 213-model status/iteration recheck pass. Details
  are in `baselines/phase8-nmm0-50k-performance.md`.
- Two fresh serial, alternating M2-targeted `nmm0-50k` public runs at
  `8d51e03` cover all 213 models with identical native/Rust statuses and
  iteration counts and no timeouts. Aggregate Rust/native ratios are 0.9937x
  and 0.9938x; sqrt(nnz)-weighted geometric ratios are 0.9824x and 0.9834x.
  Excluding `qap12`, aggregate ratios are 1.0039x and 1.0020x. Five-trial
  matched-stage triage confirms that the leading repeatable regressions are
  simplex-solve costs, not parse or presolve: `cvs16r128-89` (~80 ms extra),
  `rmatr100-p10` (~23 ms), `d6cube` (~16 ms), `graph20-20-1rand` (~14 ms), and
  `milo-v12-6-r2-40-1` (~9 ms). The raw runs, build hashes, and intervention
  ranking are in `baselines/phase8-nmm0-50k-performance.md`.
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
- The next pricing-boundary check kept the hot incremental candidate update
  out of line, as in pinned GLOP. Two alternating 21-trial A/B runs against
  the saved Rust binary measured 0.994-0.995x previous solve time on `neos`
  and 0.988x on `graph`. Strict fast-96 trajectories pass. Native-relative
  15-trial presolved-solve ratios are still 1.265x and 1.104x respectively;
  this small consistent gain does not explain the broader kernel gap.
- A matched edge-norm audit found Rust constructing a compact-column view
  solely to count its entries inside each pivot update; GLOP uses the direct
  `ColumnNumEntries()` accessor. Using the existing Rust direct-count accessor
  gives 0.986-0.992x previous Rust solve time on `neos` and 0.991-0.992x on
  `graph` in two alternating 21-trial A/B runs. Clippy and strict fast-96
  trajectories pass. This is another small fidelity and timing repair, not
  resolution of the remaining whole-solve gap.
- A new `--iteration-cap` mode in the serial matched-stage harness isolates
  the early `neos` and `graph` discrepancy. At cap 1, 21-trial median solve
  time is 0.924/2.456 ms native/Rust on `neos` and 1.121/2.058 ms on
  `graph`; at cap 100 it is 1.168/2.670 ms and 2.335/3.427 ms.
  Cap 0 takes a different exit path (6.284/7.968 ms and 5.326/5.898 ms),
  so do not treat cap-0-to-1 differences as pivot cost. Repeated cap-1
  sampling shows Rust malloc/free and equation-form initialization prominent,
  while native's profile is dominated by compact column products and transpose
  construction. Next audit the structural LP-to-equation-form path and
  initialization ownership/lifecycle before optimizing steady-state pivots.
- The structural initialization path now borrows the input LP and presents
  virtual slack columns to the equation-form initializer. It allocates only
  the augmented bound arrays and builds the compact matrix directly with
  slacks, as pinned GLOP does, instead of cloning/cleaning a complete LP.
  Partial-slack inputs retain the old repair path. A unit test compares the
  virtual columns and bounds with materialized equation form; 72 library
  tests, Clippy, fast-96 trajectories, and native presolve fixtures pass.
  At cap 1 the 21-trial native-relative solve ratios fall from 2.657x to
  1.401x on `neos` and from 1.835x to 1.209x on `graph`. A direct 21-trial
  old/new Rust full-solve comparison is 0.900x on `neos` and 1.001x on
  `graph`. Fresh 15-trial public-pipeline ratios are 1.037x and 1.088x;
  full presolved-solve ratios are 1.131x and 1.093x. Initialization fidelity
  fixes most of `neos`'s fixed early excess, but steady-state gaps remain.
- Follow-up ownership audit: `InitialBasis` now borrows the existing bound
  slices instead of materializing two dense copies, and the mutable basis
  buffer moves into and out of crash construction rather than being cloned
  twice. These changes mirror GLOP's view and in-place basis APIs. Separate
  31-trial cap-1 A/B runs measured 0.986x previous Rust time on `neos` for
  bound borrowing and 0.984x on `graph` for basis transfer; the other model
  was near neutral in each comparison. Clippy, library/state tests, and the
  strict fast-96 trajectory gate pass. A fresh 15-trial presolved-solve check
  after both changes is 1.123x native on `neos` and 1.096x on `graph`.
- A further initialization audit found an unconditional `VariablesInfo` clone
  used only if a TRIANGULAR crash must be retried. It is now conditional;
  recovered warm-start basis buffers also move into place instead of being
  copied. The one-pivot timing of the selected default-path models is
  essentially unchanged, as expected, but ownership matches the native path
  more closely. Library tests, Clippy, strict fast-96 trajectories, and native
  presolve fixtures pass.
- The compact structural-matrix builder now follows pinned GLOP's
  `PopulateFromSparseMatrixAndAddSlacks()` storage lifecycle: pre-size starts,
  coefficients, and rows once and fill final offsets in source entry order.
  The previous Rust implementation pushed each entry and closed each column,
  growing three buffers during every initialization. Two alternating 31-trial
  cap-1 A/B runs measured 0.978x/0.981x previous Rust time on `graph` and
  0.979x/0.994x on `neos`. Unit tests, Clippy, strict fast-96 trajectories,
  and presolve fixtures pass. Fresh 21-trial full presolved-solve ratios remain
  1.135x on `neos` and 1.093x on `graph`; the builder correction primarily
  affects fixed startup cost.
- The sorted symbolic closure in triangular solves now uses a retained
  `RowBitVec` and native-style bucket clearing, instead of a byte-per-row
  mask plus touched-index vector. DFS uses its existing separate workspace.
  Two alternating 21-trial old/new Rust full-solve runs measured 0.979x and
  0.971x on `neos`; `graph` was neutral at 1.000x. All 67 `lp_data` tests,
  Clippy, strict fast-96 trajectories, and native presolve fixtures pass.
- Fresh three-second `graph20-20-1rand` solve samples after the bitset change
  show distributed cost: dense transpose substitution, Harris ratio testing,
  row-wise update-row multiplication, incremental pricing, and edge-norm
  updates. A direct trailing-zero scan in transpose substitution was about
  0.3% slower in paired timing and reverted. A safe validated three-vector
  gather for Harris removed redundant bound checks but slowed both selected
  solves about 0.5%; it too was reverted. Unsafe numerical accesses are now
  permitted when justified and accompanied by a safety-invariant comment;
  speculative hot-loop changes still require matched validation and timing.
  A focused symbolic-closure regression now covers repeated sparse reaches,
  dense fallback, and separate bitset buckets; it passes with Clippy.
- The full Phase 4 native-case gate exposed a fixture setup mismatch hidden by
  the former structural LP copy: Rust did not clean modified sparse columns
  before a warm solve, while the native adapter does. The fixture now calls
  `clean_up()` at the same point. It also revealed that an advanced transposed
  solve may have nonzero slack objective coefficients; only virtual slacks
  have implicit zero cost. The corrected objective view passes Phase 4 cases,
  full workspace tests, strict fast-96 trajectories, and native presolve and
  public-iteration fixtures.
- A pinned-native strict pivot-path fixture now covers direct dual solves of
  `neos-1445765` (5,688 pivots) and `graph20-20-1rand` (1,137 pivots), including
  status, iteration count, terminal basis, and every entering/leaving/row
  choice. The opt-in release regression passes. A fresh 21-trial matched
  presolved-solve rerun measures 1.105x and 1.086x native respectively; the
  remaining cost is not a pivot-path divergence.
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

1. Profile matched native/Rust simplex kernels for `cvs16r128-89` first,
   followed by `rmatr100-p10`, `d6cube`, `graph20-20-1rand`, and
   `milo-v12-6-r2-40-1`. Their parse/presolve stages are near or below parity,
   and their public and matched-solve excesses recur in both full runs. Check
   the implicated C++ and Rust control flow, sparse representation, and
   workspace lifecycle before changing a kernel. `dfl001` has a smaller,
   variable 0.4–1.5% gap; remeasure it before prioritizing. The remaining
   `neos-1445765` solve excess is ~1 ms in the public path and is secondary.
2. Recheck other extra-corpus outliers outside `nmm0-50k`, including
   `neos-3216931-puriri` and `neos-1171737`, under the same paired-stage
   protocol. The latest `nmm0-50k` public runs put `fhnw-binpack4-48` below
   native time, so it is not a current intervention target absent fresh
   contradictory measurements.
3. Extend the new strict pivot-by-pivot native fixture beyond `neos-1445765`
   and `graph20-20-1rand` to the other selected extra-corpus outliers. The
   existing 164-model gate checks public outcomes, status, and iteration
   count, not every pivot. Preserve the fast-96 strict trajectory, public
   postsolve/KKT, presolve, and focused numerical-kernel gates after each
   performance edit.
4. Repeat serial paired stage and public-pipeline benchmarks after repairs.
   Report per-model timings, absolute excesses, median and size-weighted
   ratios, trial dispersion, build fingerprints, and timeouts. Keep benchmark
   inputs and settings equivalent to pinned upstream GLOP.

## Completion criterion

Phase 8 is complete when the comparison is reproducible, material regressions
are fixed or localized and explained with evidence, and differential
trajectory/numerical-quality gates still pass. Historical implementation and
measurement detail remains in Git history and `baselines/`.
