# Phase 8 performance comparison: fast Netlib 96

This is a serial, optimized comparison against pinned native GLOP. The corpus
is the 96-model Netlib trajectory fixture, omitting `qap12` and `qap15`.
Every timing case agreed on status and iteration count. The strict pivot-path
regression and the 96-model public-postsolve KKT gate also pass after the
changes described below. Numerical and path fidelity is a gate, not an
optimization variable.

## Environment and method

- Apple M2, macOS 15.6.1 arm64; `rustc 1.95.0`, Apple Clang 17.
- Native OR-Tools commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`;
  initial Rust revision `8f84b1418f6f39eee5f73f916374f094ae853bf6`.
- Release builds, one warm-up and three alternating, serial timed trials per
  solver and model. Each reported model time is the median of its trials.
  Aggregate totals sum those medians. The `final` artifacts were measured
  from the uncommitted working-tree edits described here; their JSON revision
  field identifies the parent commit, not a clean source tree.
- Direct-dual and presolved-solve timers exclude MPS parsing and output.
  Parsing and presolve are timed separately. Public-solver timings include its
  default pipeline. The memory timer instead measures the entire process,
  including parsing and runtime/library startup.

## Workspace-audit rerun (12 October 2026)

### Validated compact gathers in the hot update-row call

The earlier attempt to pass the validated compact-row token through
`RevisedSimplex` missed the hot call: primal steepest-edge maintenance calls
`UpdateRow::compute_update_row_compact()` first, and the solver's later call
usually returns its cached result. The validated token already held by
`PrimalEdgeNorms` is now passed at its two actual update-row call sites.
Public update-row calls and malformed matrices still use checked gathers;
the token proves the immutable matrix's row indices and RHS dimension before
unchecked per-entry access. The compact-dot benchmark was extended to time
both modes. At 4, 16, and 64 entries per column, validated/checked medians
were 0.767, 0.732, and 0.785, respectively, while checksums matched native.

Five alternating, serial old/new Rust solves with identical status,
iterations, and objective bits measured `fit2d` new/old 0.929 (tight trials)
and `dfl001` 0.997. The strict fast-96 trajectory, public postsolve/KKT, full
workspace tests, formatting, and Clippy pass. The three-trial, one-warmup
serial fast-96 rerun measured matched presolved solve at 9.831 s native versus
10.851 s Rust (1.104x, previously 1.113x), and the public pipeline at
9.958 s native versus 10.856 s Rust (1.090x, previously 1.109x). `fit2d`
was 1.098x in matched solve and 1.093x in the public pipeline. The raw
trials are ignored artifacts
`target/phase8-{presolved-solve,public}-edge-update-row-validated-fast96.json`.
Cross-run total times varied, so the paired model A/B is stronger causal
evidence than subtraction of aggregate totals.

An experimental literal port of native `PrimalPrices::UpdateEnteringCandidates`
used its branchless `ConditionalXorOfTwoBits` expression in all three Rust
candidate-update sites. The strict fast-96 trajectory gate passed, but an
uncontended alternating five-trial presolved-solve A/B measured new/old 1.004
on `dfl001` and 1.009 on `fit2d`; the change was dropped. The Boolean
decisions are equivalent because the two reduced-cost threshold conditions
are mutually exclusive. `tools/benchmark_phase8_ab.py` now accepts
`--mode presolved` for these paired timing checks.

GLOP's column-wise update row also reads equation-form trailing slack
coefficients directly from its left-inverse vector. An experimental Rust
version validated those trailing identity columns once at matrix binding and
used the same lookup. It was exact on a focused test, but paired `fit2d`
solves were 1.028× old with per-column validation and 1.010× old after
hoisting validation out of the loop; `dfl001` was 1.002× old in the first
form. The shortcut was removed. The existing one-entry compact-column dot
computes the same value, and no useful speedup was established.

The primal Harris first pass was then split into positive- and negative-
reduced-cost const-generic instantiations, mirroring upstream's templated
`ComputeHarrisRatioAndLeavingCandidates`. This removes the entering-sign
branch from each direction entry without changing its arithmetic. Alternating
five-trial `dfl001` and `fit2d` presolved solves both measured 1.000× old;
the edit is kept for source-level correspondence, not claimed as a speedup.

The primal edge-norm update also built a fresh converted nonzero-position
`Vec` for sparse directions; upstream assigns into its retained vector.
Rust now clears and extends that retained buffer. A paired five-trial A/B
measured 1.006× old on `fit2d` and 1.003× on `dfl001`, so this is an
allocation/fidelity correction, not a timing win. Exact edge-norm
recomputation then switched from `vec![1.0; n]` to the upstream-equivalent
`resize(n, 1.0)`, preserving the existing storage and entries until relevant
columns are overwritten. Paired new/old medians were 0.997× on each model;
again, the small difference should not be read as a durable speedup.
The serial fast-96 matched-presolve solve rerun (three timed trials after one
warm-up per model) matched all native statuses and iteration counts, with
9.652 s native versus 10.596 s Rust (1.098×); the previous cross-run ratio
was 1.104×. The median model ratio was 1.134× and sqrt-nnz-weighted
geometric ratio 1.121×. Raw per-model trials are in ignored
`target/phase8-presolved-solve-edge-norm-reuse-fast96.json`.

### Bound compact-column dot view

`ValidatedCompactRows::bind()` now checks matrix identity and RHS length once
for a borrowed matrix/vector pair; its short-lived `ValidatedCompactDot`
performs the same four-accumulator fused product without repeating those
checks for each column. The bound view is used in the column-wise update row
and incremental primal edge-norm loop. Public/invalid-input products remain
checked, and the token still owns the immutable matrix that was validated.
Seven alternating old/new Rust presolved solves measured new/old 0.976 on
`fit2d` and 0.990 on `dfl001`, with unchanged statuses and iterations.

The isolated 4096-row, 1024-column benchmark (seven trials, 1000 in-process
repetitions) measured bound/per-call-validated 0.933 at four entries per
column, 1.001 at 16 entries, and 1.003 at 64 entries. Bound/native ratios
were 1.420, 1.230, and 1.069, respectively; checksums agreed. The serial
fast-96 matched-presolve rerun measured 9.631 s native and 10.481 s Rust
(1.088×), down from the prior cross-run 1.098×. All 96 statuses and
iteration counts matched; strict trajectories, public postsolve/KKT, and
workspace tests pass. Raw per-model trials are in ignored
`target/phase8-presolved-solve-bound-compact-dot-fast96.json`.
The corresponding serial public default-pipeline run measured 9.917 s native
versus 10.813 s Rust (1.090×), with all 96 status, iteration, and path-length
outcomes matched. This is effectively unchanged from the prior cross-run
public ratio of 1.090×; parsing and presolve dilute the solve-only gain.
Its raw trials are in ignored
`target/phase8-public-bound-compact-dot-fast96.json`.

### Packed middle-product update and reusable scratchpad

The rank-one update chain now uses parallel 32-bit row and `f64` coefficient
arrays like GLOP's `CompactSparseMatrix`. The simplex path reuses its right
scattered scratchpad and appends the stored compact left column directly,
avoiding temporary u/v vectors. The four-accumulator dot retains GLOP's
arithmetic order. Six 300-case native basis-update suites, 1,000 isolated
rank-one cases, strict fast-96 trajectories, public fast-96 postsolve/KKT,
and the release workspace suite pass.

Paired seven-trial old/new presolved solves versus the pre-audit Rust binary
gave new/old 0.996 for `fit2d`, 0.972 for `fit2p`, 0.972 for `stocfor3`, and
0.995 for `dfl001`, whose samples varied considerably. Direct append alone
versus the scratch-reuse build gave 0.982 on `fit2d` and 0.987 on `dfl001` in
five paired trials. The serial fast-96 run measured 9.395 s native versus
9.989 s Rust (1.063×) for matched presolved solves and 9.811 s versus
10.401 s (1.060×) for the public pipeline. All 96 status/iteration outcomes
matched. These cross-run ratios should not be interpreted as an isolated
effect of the edit. Raw per-model reports are ignored artifacts at
`target/phase8-{presolved,public}-rank-one-direct-fast96.json`.

An isolated native/Rust packed-update benchmark excludes factor construction
and times repeated dense right/left solves with the same deterministic chain
and checksums. Seven alternating trials, 1,000 repetitions, dimension 4096:
at 32 updates, Rust/native ranged from 0.892× to 0.982× across 4, 16, and
64 entries per update; at 96 updates it ranged from 0.827× to 0.996×.
Consequently the retained rank-one dense solve kernel does not explain the
remaining solver-level excess on these patterns. Reproduce with
`tools/benchmark_rank_one_solve.py` after building the matching release
example and pinned native adapter.

The native `ComputeUpdateRowForBenchmark()` hook and the Rust equivalent were
then measured on matched compact matrices with 2,048 rows, 2,048 structural
columns, and 2,048 slack columns. The first Rust hook scattered the dense lhs
each repetition, unlike native's dense assignment; this added roughly 3 µs
per call and was corrected. Five hundred differential cases still agree.
Seven alternating trials of 1,000 column-wise products excluded matrix
construction. After hook alignment, at 0/4/16/64 entries per structural
column the validated Rust/native ratios were 1.201×/1.231×/1.023×/0.851×.
Without the row-validation token they were 1.202×/1.344×/1.236×/1.029×.
Subtracting each implementation's empty-column baseline leaves about 2.30 µs
native versus 3.03 µs Rust for the four-entry structural products. The
short-column product, not packed rank-one dense solve, is the clearer residual
kernel gap. An exact-order branch for columns of at most four entries was
tested before the hook correction and reverted because the four-entry timing
increased. Reproduce with `tools/benchmark_update_row.py` after building its
matching release example and pinned native adapter.

Hoisting matrix representation and validation dispatch out of each column
iteration reduced the corrected validated whole-hook ratios to 1.142× at four
entries, 0.962× at 16, and 0.829× at 64. In a serial alternating old/new
presolved full-solve comparison over all fast-96 models (three trials plus
one untimed warm-up), the summed per-model medians were 10.241 s old and
10.191 s new (0.995×); median model ratio was 0.993×. All status/iteration
outcomes matched. Focused seven-trial paired solves measured `fit2d` 0.976×,
`dfl001` 1.009×, `fit2p` 1.000×, and `stocfor3` 0.996×. The strict fast-96
pivot trajectory and public postsolve/KKT gates pass, so the small aggregate
gain does not trade off fidelity.

### Update-row reset fidelity and kernel attribution

Paired probe timings on all 271 `dfl001` and 89 `fit2d` saved basis matrices
(three alternating trials, five factorizations per timing) measured LU at
1.070x native for `dfl001` and 0.963x for `fit2d`. The paired sparse solve
probe run sampled at most five RHSs of each kind per basis, with three trials
and 50 in-process repetitions. `dfl001` FTRAN was 1.101x over 275 RHSs,
BTRAN 0.992x over 563; `fit2d` FTRAN was 1.054x over 93 and BTRAN 1.202x
over 181. These isolated costs are too small in absolute time to explain the
full-solve excess. Raw files are
`target/phase8-dfl001-fit2d-{factorizations,solve-probes}.csv`.

Ten-second serial `sample` profiles of presolved `fit2d` solves on both
binaries put the bulk of execution in primal edge-norm updates, column-wise
update-row computation, and primal-price updates. Rust's column-wise path
had a fidelity mismatch: it filled the entire coefficient vector with zero
every pivot, whereas upstream resizes and retains its existing entries,
writing only active coefficients. That full clear was removed. A five-trial
alternating old/new Rust A/B gave new/old 0.997 for both `fit2d` and `dfl001`,
with identical status, iterations, and objective bits; this is too small to
claim a reliable speedup. The strict fast-96 trajectory gate passes.

The serial full fast-96 rerun after this edit (three trials, one warm-up)
measured matched presolved solve at 9.607 s native versus 10.838 s Rust
(1.128x), and public default pipeline at 9.785 s versus 10.998 s (1.124x).
All 96 cases matched status and iteration count. Raw trials are in
`target/phase8-{presolved-solve,public}-update-row-retain-fast96.json`.
The residual gap is concentrated in per-pivot update-row, edge-norm, and
pricing work; the present samples do not isolate which implementation detail
accounts for it.

### Validated compact gathers in primal edge norms

The hot primal edge-norm update repeatedly dots immutable compact columns
against a row-sized dense vector. A new validation token scans the compact
row indices once at binding, owns an `Rc` to prevent subsequent mutation,
and is tied to the exact matrix identity. Each product checks token identity
and RHS length once, then uses unchecked per-entry gathers. Matrices with an
invalid row cannot obtain a token and continue through the checked path;
ordinary public compact products remain checked. Tests cover exact result
bits, invalid rows, wrong matrix, and short RHS. This is a localized
optimization of GLOP's unchecked compact-column traversal, not a change to
the floating-point operation order.

Five alternating old/new Rust trials of matched presolved solves measured
new/old 0.947 on `fit2d` and 0.994 on `dfl001`, with identical status,
iterations, and objective bits. The strict fast-96 trajectory and public
postsolve/KKT gates, workspace tests, formatting, and Clippy pass. A clean
serial fast-96 rerun (three timed trials and one warm-up per solver/model)
measured matched presolved solve at 9.590 s native versus 10.676 s Rust
(1.113x, previously 1.128x), and public default pipeline at 9.863 s versus
10.937 s (1.109x, previously 1.124x). The `fit2d` matched-solve ratio fell
from 1.257x to 1.193x. Raw trials are in ignored
`target/phase8-{presolved-solve,public}-validated-compact-dot-fast96.json`.

The same token was experimentally threaded through the solver's column-wise
`UpdateRow` path. A paired five-trial A/B was noisy (`fit2d` new/old 0.988,
`dfl001` 1.002), and a serial fast-96 rerun measured 1.113x native versus
1.113x without that extension. Strict trajectories still passed, but no
reliable aggregate benefit was established, so the additional `UpdateRow` and
`RevisedSimplex` wiring was removed. Its ignored diagnostic run is
`target/phase8-presolved-solve-validated-update-row-fast96.json`.

### Follow-up ownership-scan rerun

After removing the transient sparse-index and retired adjacency-list clones
and reusing the reduced-cost left-inverse buffer, the same serial fast-96
protocol was rerun on the uncommitted working tree (one warm-up, three
alternating timed trials per model). Release binaries were rebuilt; each
stage run checked matching outcomes, and the public and direct-dual harnesses
checked their corresponding solver outcomes. The JSON revision field is the
parent commit `ffbaca5`, not the working-tree content.

| Timed scope | Native | Rust | Rust/native | Sqrt-nnz weighted geometric ratio |
| --- | ---: | ---: | ---: | ---: |
| Parse | 0.218 s | 0.201 s | 0.926× | 0.913× |
| Presolve | 0.262 s | 0.263 s | 1.005× | 0.999× |
| Solve after matched presolve | 9.652 s | 10.882 s | 1.127× | 1.149× |
| Public default pipeline | 9.935 s | 11.198 s | 1.127× | 1.117× |
| Direct unscaled dual | 23.778 s | 22.825 s | 0.960× | 1.175× |

The preceding documented full run had 1.134× matched presolved solve and
1.121× public-pipeline ratios. The new differences are small and do not by
themselves establish a gain from the ownership edits. In the matched
presolved solve, `dfl001` (+0.461 s), `fit2d` (+0.280 s), and `d6cube`
(+0.101 s) remain the largest absolute excesses. Direct dual is again
dominated by `pilot87` (9.274 s native, 7.101 s Rust); its below-one total
ratio does not represent most models. Raw per-model timings and dispersion
are in ignored `target/phase8-{stages,public,direct-dual}-ownership-scan-fast96.json`.

After retaining the dual-breakpoint heap, primal ratio-test candidate lists,
scattered dual workspace, and several reset buffers, the same fast-96 protocol
was rerun serially on the Apple M2: one warm-up and three alternating timed
trials per model, with native/Rust status and iteration checks. Results are
summed per-model medians; smaller is better for Rust/native ratios.

| Timed scope | Native | Rust | Rust/native | Sqrt-nnz weighted geometric ratio |
| --- | ---: | ---: | ---: | ---: |
| Parse | 0.217 s | 0.200 s | 0.923× | 0.910× |
| Presolve | 0.262 s | 0.264 s | 1.011× | 1.006× |
| Solve after matched presolve | 9.660 s | 11.165 s | 1.156× | 1.174× |
| Public default pipeline | 9.945 s | 11.466 s | 1.153× | 1.144× |
| Direct unscaled dual | 23.794 s | 22.998 s | 0.967× | 1.185× |

The direct-dual total is dominated by `pilot87`: 9.267 seconds native versus
7.147 Rust, with tight three-trial dispersion. Excluding only that model gives
a 1.091× direct-dual aggregate; Rust remains slower on 90 of 96 direct-dual
models. The largest remaining absolute excesses are `dfl001` (+0.610 s),
`truss` (+0.275 s), and `stocfor3` (+0.256 s). In the matched presolved solve,
the leading excesses are `dfl001` (+0.504 s), `fit2d` (+0.329 s), and `d6cube`
(+0.146 s). Thus the 0.967× direct total does not establish general parity.

The closest prior full stage sample, taken before the parser/presolve fixes and
this workspace audit, had a 1.164× presolved-solve ratio (9.616/11.192 s);
the new 1.156× is a modest change and not an isolated measurement of the
workspace edits. The preceding documented public and direct-dual comparisons
are older still and include intervening solver changes. Per-model raw trials,
dispersion, and metadata are in ignored `target/phase8-*-workspace-audit-fast96.json`.
Those artifacts identify the parent commit because the measured Rust binary
contains the current uncommitted workspace-audit edits.

## Compact-dot bounds-check follow-up

### Bounded numerical-kernel access follow-up

A second source audit replaced redundant repeated indexing in compact-matrix
column add/copy, triangular direct and hypersparse solves, rank-one updates,
sparse-vector dense updates, edge-norm updates, update rows, and residual
accumulation. Paired storage is traversed through bounded column views or
four-entry chunks; read/modify/write operations borrow one destination slot.
The generic dense scalar product likewise uses paired four-entry chunks.
These are safe rewrites, with no new `unsafe` and no change to numerical
operation order. Index-dependent gathers from public sparse vectors or
caller-supplied slices remain checked: their APIs do not establish a dimension
bound for every stored index. Removing those checks requires an API/invariant
change, not merely replacing `[]` with `get_unchecked`.

One warm-up and three alternating serial trials on the fast-96 matched
presolved solves measured 9.668 s native versus 10.938 s Rust, or 1.131×;
the preceding paired-slice-only run was 1.125×. To distinguish a regression
from machine variation, a separate five-trial alternating old/new Rust A/B
used the parent-commit release binary on four large contributors. New/old
median ratios were 0.997 (`dfl001`), 1.002 (`fit2d`), 0.998 (`truss`), and
0.995 (`stocfor3`), with matching statuses, iterations, and objective bits.
This does not show a material solve regression or a material aggregate gain.
The full per-model serial trials are in ignored
`target/phase8-solve-bounded-kernels-fast96.json`. The 1,000-case native
triangular, compact-matrix, and rank-one traces, full workspace suite, strict
fast-96 trajectory fixture, and public postsolve/KKT gate pass.

### Update-row sparse-column dot

An ARM64 release disassembly of `UpdateRow`'s four-accumulator sparse-column
dot showed checks for positional reads of its parallel row/coefficient arrays
in addition to the necessary caller-supplied RHS gather checks. The production
kernel now traverses paired four-entry slices, preserving the exact fused
accumulation and tail order; it keeps the RHS checks and uses no `unsafe`.
The in-repo `update_row_dot_benchmark` compares the old indexed and new paired
forms with identical generated columns, 1,024 columns, 4,096 rows, 500
repetitions, one warm-up, and seven alternating trials. Median nanoseconds per
dot were 3.565/3.542 (old/new) at four entries, 10.090/7.789 at 16, and
39.083/25.796 at 64. Thus the long-column speedups are about 23% and 34%,
while the four-entry case is tied. Checksums matched. The 500-case native
update-row trace and strict fast-96 pivot trajectory fixture pass.

A subsequent full serial fast-96 rerun used one warm-up and three alternating
trials per model, with the release Rust binaries rebuilt from the uncommitted
paired-slice source. Results are summed per-model medians:

| Timed scope | Native | Rust | Rust/native | Prior ratio |
| --- | ---: | ---: | ---: | ---: |
| Parse | 0.217 s | 0.202 s | 0.929× | 0.926× |
| Presolve | 0.262 s | 0.264 s | 1.009× | 1.005× |
| Solve after matched presolve | 9.699 s | 10.910 s | 1.125× | 1.127× |
| Public default pipeline | 9.973 s | 11.195 s | 1.122× | 1.127× |
| Direct unscaled dual | 23.774 s | 22.793 s | 0.959× | 0.960× |

Every timed case retained matching outcomes; the public run matched all 96
iteration counts. The direct-dual sqrt-nnz-weighted ratio is 1.176×, nearly
unchanged from 1.175×. The small full-solver movements are within the range
of run-to-run variation and do not establish an end-to-end gain from this
specific kernel edit. Per-model trials and dispersion are in ignored
`target/phase8-{stages,public,direct-dual}-update-row-slices-fast96.json`;
their revision field records parent commit `c7201e2`, not the uncommitted edit.

The compact-column dot product remained 1.7–1.8× native in matched isolated
measurements. A scoped unchecked read of only the matrix-owned coefficient and
row arrays removes repeated entry-array bounds checks. The compact builder
always appends one row and coefficient together and closes each column at
their common length; the RHS vector remains checked, including for malformed
row coordinates. The four-accumulator fused arithmetic and traversal order
are unchanged. A seven-trial alternating A/B of old/new Rust binaries measured
new/old medians of 0.821, 0.759, and 0.721 at 4, 16, and 64 entries per
column; corresponding new/native ratios are 1.429, 1.375, and 1.290.

Five-trial alternating presolved-solve A/Bs, timing inside the solver, measured
`fit2d` at 1.430 versus 1.384 seconds (new/old 0.968) and `dfl001` at
3.549 versus 3.526 seconds (0.993), with matching status, iterations, and
objective bits. A fresh serial fast-96 comparison (one warm-up, three timed
trials per model) measures the matched presolved solve at 9.623 seconds native
versus 10.866 Rust (1.129×), down from the preceding 1.156× run. The public
default pipeline measures 9.891 versus 11.100 seconds (1.122×), down from
1.153×. The strict 96-model native pivot-path and public postsolve/KKT gates
pass; the qap15 terminal KKT gate passes too. Raw per-model trials are in
ignored `target/phase8-{presolved-solve,public}-compact-bounds-fast96.json`.

Separately, reduced-cost refresh now temporarily moves the existing objective
allocation into a read-only local reference and restores it, instead of
cloning the full vector. A focused three-trial A/B on `fit2d`, `dfl001`, and
`stocfor3` showed no reliable timing gain; this is retained for ownership
fidelity, not credited with the preceding performance improvement.
The full serial fast-96 rerun on the final combined source measures the
presolved solve at 9.693 seconds native versus 10.996 Rust (1.134×) and the
public pipeline at 10.007 versus 11.216 seconds (1.121×). These small
differences from the compact-dot-only 1.129×/1.122× runs are not a reliable
effect of the objective move. Final-source trials are in ignored
`target/phase8-{presolved-solve,public}-compact-bounds-objective-view-fast96.json`.

The actual `dfl001` refactorization probes give a more specific transpose
answer than the synthetic lower-transpose benchmark. Across all 271 saved
basis groups, sampling up to five FTRAN and five BTRAN RHS per group, three
serial trials and 50 in-process repetitions per RHS, BTRAN is at 0.991×
native (563 RHS) with no checksum disagreements. FTRAN remains 1.107× (275
RHS). This does not support treating BTRAN/lower-transpose as the main source
of `dfl001`'s remaining full-solver excess. The per-RHS data are in ignored
`target/phase8-dfl001-solve-probes-all.csv`. The probe harness's `--rhs-limit`
now applies separately to FTRAN and BTRAN so a capped sample cannot omit
the latter.

| Timed scope | Initial ratio | Workspace edits | Direct row conversion | Inline dot product |
| --- | ---: | ---: | ---: | ---: |
| Direct unscaled dual solve | 1.086 (23.80 / 25.84 s) | 1.104 (23.74 / 26.22 s) | 1.172 (24.04 / 28.17 s) | not rerun |
| Public default pipeline | 1.347 (9.94 / 13.38 s) | 1.346 (9.68 / 13.04 s) | 1.301 (10.01 / 13.02 s) | 1.264 (10.32 / 13.05 s) |
| Solve after matched presolve | 1.356 (9.51 / 12.90 s) | 1.340 (9.62 / 12.89 s) | 1.291 (9.84 / 12.70 s) | 1.254 (9.81 / 12.31 s) |
| Parse to `LinearProgram` | 1.753 (0.216 / 0.379 s) | 1.756 (0.210 / 0.370 s) | not rerun | not rerun |
| Copy, cleanup, and presolve | 1.467 (0.261 / 0.383 s) | 1.461 (0.255 / 0.372 s) | not rerun | not rerun |

The public gap is overwhelmingly in the presolved simplex solve. Before the
compact-dot edit, its largest absolute per-model excesses were `dfl001`
(+0.88 s), `fit2d` (+0.77 s), and `d6cube` (+0.23 s). Small-model ratios should
not dominate an aggregate: after inlining, the presolved-solve median model
ratio is 1.250, and its sqrt-nonzero-weighted geometric mean is 1.264. The
corresponding public figures are 1.271 and 1.289. The direct-dual total is
strongly influenced by a few large models, so its 1.172 total ratio does not
describe the typical case; its sqrt-nonzero-weighted geometric mean is 1.313.
The direct-dual total also varied much more than the other two scopes across
these runs, warranting a controlled check before attributing its movement to
an implementation change.

## October 2026 parsing and presolve follow-up

Using the same serial, release-build fast-96 stage harness (one warm-up and
three alternating trials per model), an in-place partial-index permutation
removed a temporary sparse-entry allocation in presolve. Fixed MPS fields now
borrow the input text, and fixed-format data cards are no longer redundantly
whitespace-tokenized. Row and column deletion also now compact model metadata
in place, as upstream does, instead of rebuilding bounds, types, and cloned
names. The parser still uses the same fixed-first/free-fallback selection and
field boundaries. Summed per-model median times were:

| Stage | Native | Rust before | Rust after | Final ratio |
| --- | ---: | ---: | ---: | ---: |
| Parse | 0.218 s | 0.380 s | 0.199 s | 0.912× |
| Presolve | 0.263 s | 0.382 s | 0.267 s | 1.015× |

The stage harness checked parsed dimensions and presolved status/dimensions
for all 96 models. The final parser and presolve raw trials and dispersion are
in ignored `target/phase8-parse-presolve-final.json` and
`target/phase8-presolve-inplace-metadata.json`; intermediate runs are in
similarly named `target/phase8-*.json` files. Native parsing
includes an `MPModelProto` conversion whereas Rust builds `LinearProgram`
directly, so the final parse ratio is pipeline parity, not identical parser
algorithm cost. Presolve's residual aggregate excess is 0.004 s, within 1.5%.
After the metadata change, the native MPS reader's 1,000 free and 1,000 fixed
generated traces, the 500-case native `lp_data` trace, the strict fast-96
pivot-path regression, and the fast-96 public postsolve gate all pass.

The 96-model whole-process memory run used three alternating launches each.
Rust had lower peak RSS in all 96; median per-model Rust/native RSS ratio was
0.194, and the largest observed peaks were 33.3 MiB Rust versus 41.5 MiB
native. The difference includes executable and shared-library/runtime
footprints; it does **not** establish that Rust uses one-fifth as much solver
heap on matched instances.

## Findings and changes

Sampling `stocfor3` in the presolved primal path exposed frequent LU scattered
permutations. GLOP swaps the solved vector with a persistent zero scratchpad;
Rust previously copied the entire dense array. Rust now swaps the underlying
`Vec<f64>` storage, zeros only the old sparse positions when applicable, and
preserves the dense-sentinel behavior. The two-output cached left solve
likewise swaps its dense values instead of copying them. Strongly typed
row/column position lists still need a linear conversion. Separately,
`RightSolveForTau()` now returns a borrow of its persistent workspace instead
of cloning it every dual pivot, matching GLOP's reference-return API.
Focused tests check storage identity and values; six 300-case native
basis-update suites and the fast-96 strict trajectory gate pass.

The workspace changes help the focused `stocfor3` presolved solve, but alone
did not show a decisive full-corpus gain. A safe slice/chunk rewrite of the
compact-column dot product was tried, timed, and reverted because it did not
improve the focused direct-dual benchmark. A fresh presolved `fit2d` profile
put roughly 73% of Rust samples in that scalar product, called from update-row
and primal-edge-norm work. The compact matrix already stores 32-bit row
indices like upstream. In an isolated matched benchmark, Rust was 2.25–2.38×
native at 4–64 entries per column. ARM64 disassembly showed a redundant
signedness check on each stored row in addition to the checked vector gather.
Converting that row directly, as GLOP's `RowToColIndex()` does, retains the
vector bounds check and the exact four-accumulator arithmetic. The isolated
ratio fell to 1.81–2.11×, and the full presolved and public ratios improved as
shown above. The strict 96-model trajectory test and an invalid-negative-row
panic test pass. GLOP declares this kernel inline in `sparse.h`; marking the
Rust counterpart for inlining reduced the isolated ratio further to
1.73–1.83× and produced the final full-corpus presolved/public results in the
table. The final strict trajectory and public KKT gates still pass.

The after-edit direct-dual run instead showed a large `dfl001` slowdown.
Alternating five trials of the old/new Rust binaries on the same machine
measured the new binary faster on `dfl001` (0.955× old), `fit2d` (0.905×), and
`stocfor3` (0.965×). This points to machine variation in the full direct run;
it does not explain or erase that recorded result. More direct-dual sampling is
needed before claiming a whole-corpus gain. The remaining dot-product gap
likely involves checked vector gathers and code generation. The
lower-transpose triangular-solve gap documented in `PLAN.md` remains open.
No performance-parity claim is made yet.

A further layout experiment changed `TriangularMatrix` non-diagonal row storage
from 64-bit `usize` to 32-bit indices, matching native GLOP's storage width.
In the isolated lower-transpose benchmark at dimension 1024, the Rust/native
ratio fell from 1.51 to 1.45 with the same 3% density and 5,000 repetitions.
The full 96-model serial presolved-solve timing, however, is 9.527 seconds
native and 11.941 seconds Rust (1.253×), effectively unchanged from the
preceding 1.254×. The layout improves one kernel but does not account for the
solver-level gap. All 1,000 native triangular and LU traces, strict 96-model pivot
trajectories, and 96-model public postsolve/KKT comparisons pass. Its raw
per-model timing is `phase8-presolved-solve-u32-fast96.json`.

The A/B harness is `tools/benchmark_phase8_ab.py`. For that check, `A` was a
release binary with the checked signed conversion, and `B` differed only by
the direct integer conversion in the compact dot-product kernel. Its printed
per-trial times are reproducible with `--a`, `--b`, and `--models`; the binary
copies are local ignored `target/` artifacts, not source-controlled fixtures.
A second five-trial A/B comparison of the direct-conversion binary with the
inlined binary found the latter 2.5% faster on `dfl001` and 3.6% faster on
`fit2d`, but 1.3% slower on `stocfor3`; therefore the inline change is supported
more strongly by the complete presolved/public suites than by these three
direct-dual cases alone.

## Reproduction and raw results

Build the Rust release examples and the native adapters as described in
`PLAN.md`, then run these serially on the same idle machine:

```sh
python3 tools/benchmark_netlib_solve.py --trials 3 --warmups 1 --output baselines/phase8-direct-dual-dot-fast96.json
python3 tools/benchmark_phase5_netlib.py --mode default --trials 3 --warmups 1 --output baselines/phase8-public-dot-fast96.json
python3 tools/benchmark_phase8_stages.py --stage both --trials 3 --warmups 1 --output baselines/phase8-stages-final-fast96.json
python3 tools/benchmark_phase8_stages.py --stage solve --trials 3 --warmups 1 --output baselines/phase8-presolved-solve-dot-fast96.json
python3 tools/benchmark_phase8_stages.py --stage solve --trials 3 --warmups 1 --output baselines/phase8-presolved-solve-inline-fast96.json
python3 tools/benchmark_phase8_stages.py --stage solve --trials 3 --warmups 1 --output baselines/phase8-presolved-solve-u32-fast96.json
python3 tools/benchmark_phase5_netlib.py --mode default --trials 3 --warmups 1 --output baselines/phase8-public-inline-fast96.json
python3 tools/benchmark_phase8_memory.py --trials 3 --output baselines/phase8-memory-fast96.json
python3 tools/benchmark_compact_dot.py
```

The raw artifacts are `phase8-direct-dual-fast96.json`,
`phase8-public-fast96.json`, `phase8-presolved-solve-fast96.json`, and
`phase8-stages-fast96.json` for initial measurements; the corresponding
`*-final-fast96.json` files for after-workspace-edit runs;
`*-dot-fast96.json` files for after-row-conversion runs;
`*-inline-fast96.json` files for after-inlining runs; and
`phase8-memory-fast96.json`. Each timing JSON includes per-model trial times
and median absolute deviations. The memory JSON includes all three peak-RSS
and footprint trials per model.
### Residual `dfl001` sampling after validated update-row dispatch

Serial 1 ms samples of ten repeated matched-presolve solves are saved in
`target/phase8-dfl001-{rust,native}-after-update-row.sample.txt` (ignored
local artifacts). Both samples captured about 8,300 solver samples. Rust's
largest top-of-stack entries were transpose solve (1,563), column-wise
update-row (1,068), primal edge norms (732), ratio test (614), and primal
pricing (424). Native's three transpose-solve instantiations total 1,768
samples, while column-wise update-row (1,248), edge norms (720), and pricing
(520) are also prominent. Sampling symbols and inlining differ across the
two builds; these counts do not establish a transpose-solve regression.
The residual cost is distributed enough to require paired, matched-input
kernel timing before another code change.

### Production-input update-row replay

An opt-in `phase8_update_row_probe` build captured exact compact matrices,
dense left inverses, and relevant-column masks from presolved `dfl001` and
`fit2d` update-row calls under ignored
`target/phase8-update-row-production/`. The replay adapters load these
operands before timing and compare native/Rust checksums. In `dfl001`, 12,494
of 13,818 columns contain at most four entries, versus only 25 of 10,388 in
`fit2d`. This justifies analyzing the short-column and longer-column cases
separately. The adapters split the dot, threshold/count, coefficient write,
position write, and full work; mode dispatch is outside the measured loops.

For `dfl001` call 500 (5,702 relevant columns), seven alternating 100-replay
trials gave native/Rust medians of 12.37/12.21 us for dot plus nonzero count,
14.30/22.93 us for dot plus position appends, and 15.31/27.88 us for the
full append-based replay. The dot-only mode gave 12.49/19.30 us but has a
different anti-elision barrier in the two languages; its ratio should not be
interpreted as an isolated product cost. The source comparison identified
GLOP's pre-sized cursor writes versus Rust's `Vec::push`. Matching GLOP's
cursor pattern in both replay adapters reduced the Rust full replay to about
21.5 us at call 500 and 21.1 us at call 2000. This is diagnostic, not a
solver-level result.

The corresponding production edit was then tested with immediate before/after
release solver binaries, serial alternating trials, one warm-up, and seven
timed trials per model, using
`tools/benchmark_phase8_ab.py --mode presolved`. New/old medians were
`dfl001` 1.004x (3.3703/3.3846 s), `fit2d` 0.992x (1.1523/1.1428 s),
`fit2p` 0.997x, and `stocfor3` 1.013x; aggregate 1.002x. The edit was
reverted. It is not evidence that output append explains the remaining
full-solve discrepancy; its hot-loop share and interaction with dot/pricing
work require further profiling.

The follow-up implementation retains an `n`-slot position array and a
separate active count, rather than truncating/refilling the array each call.
It is the direct counterpart of GLOP's `non_zero_position_list_` and
`num_non_zeros_` and allocates at construction, with reallocation only if a
later rebind increases column count beyond capacity. A release unit test
checks pointer stability over ten repeated column-wise calls, a same-size
rebind, and a rebuild of nonzero positions. A fresh five-trial serial A/B
against the append binary measured new/old medians of `dfl001` 1.003x,
`fit2d` 0.988x, `fit2p` 0.998x, and `stocfor3` 1.009x; aggregate
5.183947/5.186412 s (1.000x). This is a fidelity/allocation correction,
not a material full-solve speedup.

### Harris ratio-test bound sharing

Fresh eight-second, 1-ms serial samples of matched presolved `fit2d` and
`dfl001` solves are in ignored
`target/phase8-{fit2d,dfl001}-current-{rust,native}.sample.txt`. The current
`fit2d` sample placed edge-norm update at 2,496 Rust versus 2,485 native
samples, column-wise update row at 2,174 versus 2,167, and incremental
pricing at 1,198 versus 1,398. These counts do not support the prior guess
that those three kernels explain the remaining gap. In `dfl001`, Rust's
primal Harris ratio test took 473 top-of-stack samples versus 270 in native's
Harris first pass; update row, edge norms, and pricing were near parity.

The Harris first pass uses a variable value and one bound from parallel dense
arrays at each candidate. Adding two constant-time equal-length assertions
before the scan lets optimized Rust reuse the checked variable-value index
bound for the chosen bound array. Indexing remains safe, arithmetic and
candidate order unchanged, and no per-pivot workspace is added. The follow-up
Rust sample (`target/phase8-dfl001-harris-length-rust.sample.txt`) had 243
ratio-test samples. ARM64 release disassembly of the before/after solver
binary shows 18 versus 12 out-of-line bounds-panic sites in the enclosing
ratio-test function (plus two one-time length assertions in the new version),
consistent with fewer repeated bound checks. Five serial alternating timed solve trials after one
warm-up measured `dfl001` old/new 3.383835/3.240146 s (new/old 0.958x,
tight ranges) and `fit2d` 1.138637/1.139753 s (1.001x). The three-trial
fast-96 A/B (`target/phase8-harris-length-fast96-ab.txt`) matched status and
iterations on every model and reduced summed model medians from 10.131284
to 9.914437 s (0.979x; median model 0.994x).

A serial native/Rust full fast-96 solve rerun (three timed trials, one warm-up,
matched presolved models) recorded 9.631649 s native versus 9.980340 s Rust:
1.036x aggregate, 1.070x median model, and 1.066x sqrt-nnz weighted
geometric ratio. The largest absolute residuals are `dfl001` (+0.210 s),
`fit2d` (+0.049 s), `stocfor3` (+0.049 s), `fit2p` (+0.046 s), and `d6cube`
(+0.045 s). Raw per-model trials and outcomes are in ignored
`target/phase8-harris-length-native-rust-fast96.json`. The strict fast-96
trajectory fixture, public original-model postsolve/KKT gate, release
workspace suite, formatting, and all-features Clippy pass.

Six-second, 1-ms `stocfor3` samples after this change, on 20 repeated
presolved solves per implementation, are in ignored
`target/phase8-stocfor3-current-{rust,native}.sample.txt`. Rust's largest
top-of-stack entries are transpose triangular solve (1,069), scattered
permutation (324), and scattered clear (298). Native's identified transpose
lower/upper variants total at least 1,153 samples, with 413 additional
samples under a deduplicated symbol. The symbol ambiguity prevents a claim
that transpose solve is the `stocfor3` regression; a matched BTRAN and
permutation probe is the next useful localization step.

### Validated triangular-solve bounds-check pass

Three sparse RHS updates in `TriangularMatrix::solve` and
`lower_solve_starting_at` now use unchecked mutable access. The invariant is
established by triangular builders (each stored row is below the dimension)
and by the solve's RHS-length check. The dense eta left/right solves use paired
slice iteration after a length assertion, avoiding positional indexing.
Caller-controlled sparse gathers in the other audited kernels remain checked:
their public APIs do not guarantee that every index fits the supplied slice.
The release strict fast-96 trajectory test, workspace tests, and Clippy pass.

A serial three-trial, one-warmup presolved-solve sample on `dfl001`, `fit2d`,
`truss`, and `stocfor3` measured Rust/native 1.187x in aggregate (artifact
`target/phase8-solve-unchecked-triangular-sample.json`). Against the earlier
bounded-kernel run on the same four instances, Rust medians changed by -0.5%,
-1.4%, -0.4%, and -0.8%, respectively. This is a cross-run comparison, not a
paired A/B; the small differences are not evidence of a reliable speedup.
