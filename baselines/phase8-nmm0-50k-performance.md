# NMM 0–50k: M2 performance triage

## Non-unit lower-transpose solve: descending-loop fidelity

The earlier focused lower-transpose benchmark exercised only unit diagonal
factors, but the live direction path often uses the non-unit kernel. The
benchmark now preserves input diagonal coefficients in its Rust triangular
matrix and accepts `--non-unit-diagonal`. At 5,174 rows and density 0.0005,
the old Rust kernel was 1.15x native across 15 alternating trials. Replacing
its reversed range with GLOP's explicit descending-column loop, without
changing entry order or arithmetic, reduced that ratio to 1.00x. At density
0.003 the new ratio is also 1.00x; the unit-diagonal check remains faster in
Rust. The non-unit benchmark command was:

```text
python3 tools/benchmark_lower_transpose.py --sizes 5174 --density 0.0005 --repetitions 200 --trials 15 --non-unit-diagonal
```

On an Apple M2 with release builds and 21 serial alternating trials per
model, the [paired old/new Rust solve](phase8-transpose-loop-ab.json) gives
new/old ratios of 0.968x (`traininstance2`), 0.975x (`graph20-20-1rand`),
and 0.985x (`cvs16r128-89`). The
[fresh native-relative solve](phase8-transpose-loop-native.json) gives 1.024x,
1.030x, and 0.990x, respectively; combined elapsed time is 0.995x. The
saved JSON includes per-trial dispersion, compiler versions, pinned upstream
commit, and machine details. Both runs used
`benchmark_phase8_stages.py --stage solve --corpus nmm0-50k` with the three
named models and 21 trials. Status, iteration, and objective checks passed;
the strict fast-96 and extra-model pivot gates passed separately. This closes
most of the previously observed live solve gap without changing numerical
trajectory. The remaining `traininstance2` and `graph` excesses need further
localization.

## `traininstance2` direction and selected-LU follow-up

GLOP's `ComputeDirection()` computes the infinity norm during its one support
traversal. Rust previously materialized dense support, then rescanned it (or
even all rows on two call paths). The port now returns the norm from that
traversal, preserving dense row order and sparse support order. The
[21-trial paired Rust A/B](phase8-traininstance2-direction-norm-ab.json)
measures new/old at 0.986x on `traininstance2`, 0.994x on `graph20-20-1rand`,
and 0.976x on `cvs16r128-89`.

Twenty-one selected compact basis views captured during an ordinary presolved
`traininstance2` solve exposed an increasing LU ratio at later pivots. In
GLOP, `Markowitz::ComputeColumn()` mutates a candidate in its reusable
physical column. Rust was moving that column out and restoring it on every
visit; the port now borrows it in place. The
[nine-trial paired Rust LU A/B](phase8-traininstance2-inplace-lu-ab.csv)
improved 5.6% across these bases, while a
[21-trial whole-solve A/B](phase8-traininstance2-inplace-column-ab.json)
was neutral on `traininstance2` and `cvs`, and 0.997x on `graph`. The final
[nine-trial native-relative selected-view LU results](phase8-traininstance2-selected-lu.csv)
sum to **1.122x** Rust/native, down from
[1.189x before the in-place change](phase8-traininstance2-selected-lu-before.csv).
The captured probes live under `target/phase8-train2-factorization-probes/`
and can be regenerated with `netlib_simplex_probe` in
`--presolved-factorizations` mode.

In the Harris candidate scan, one checked variable-value lookup now bounds
unchecked lower/upper bound reads of the same index, after the equal lengths
are asserted. An invalid-basis-column panic test covers the safety boundary.
The 21-trial Rust A/B was timing-neutral on all three models; the edit is
retained for port fidelity. Full workspace tests, strict Clippy, fast-96 and
extra-model pivot fixtures, and 21 presolved-state fingerprints against native
GLOP pass.

The fresh [31-trial matched solve](phase8-traininstance2-current-solve.json)
is 67.30 ms native versus 71.16 ms Rust on `traininstance2`, **1.057x**;
`graph` is 1.054x. The [21-trial public pipeline](phase8-traininstance2-current-public.json)
is 101.11 versus 103.67 ms, **1.025x**. Status, iteration count, and
objective agree. Saved pre-next-pivot RHS probes put isolated BTRAN at about
1.00x on 46 sampled solves and FTRAN at 1.17x on 70; all 10,472 startup
FTRAN RHS sum to a 0.25 ms Rust excess despite a 1.61x ratio for these very
cheap solves. These probes omit subsequent product-form updates and do not
explain the remaining several milliseconds of live solve time. Parity remains
open, especially in the live direction/transpose and edge-norm paths. A fresh
seven-second paired stack sample still places dense transpose substitution
among the largest costs on both sides; Rust spends a somewhat larger share
there, but inlining and sampling prevent attributing the whole residual gap
to that kernel without a matched live-stage timer.

## `traininstance2` rank-one follow-up

A fresh 21-trial serial matched-stage [baseline](phase8-traininstance2-baseline-stages.json)
put `traininstance2` at 1.087x native presolved-solve time (70.43 vs
64.82 ms); parse was 1.135x and presolve 0.952x. Paired 8-second native/Rust
stack samples showed dense transpose triangular solves as the largest shared
cost, followed by permutation, direction, and Harris work. The samples alone
do not establish whether every numerical branch matches on this model.

One concrete port difference was the packed rank-one scalar product: Rust
checked every RHS gather and dense update write, while upstream's compact
columns read and write directly. The Rust factor now records a required RHS
length when public updates are appended, or the known basis dimension once
for solver-owned compact updates. Each solve checks this length before its
documented unchecked inner accesses. A regression test verifies that short
caller-supplied vectors still panic safely. Operation order and storage order
are unchanged.

The final [21-trial old/new Rust A/B](phase8-traininstance2-rank-one-ab.json)
measures `traininstance2` at 0.994x, `graph20-20-1rand` at 0.991x, and
`cvs16r128-89` at 0.999x. Tracking a maximum on *every* solver-owned compact
entry instead of recording the basis dimension once had slowed `cvs` by
about 0.9% and was replaced. A function-boundary experiment on dense
transpose kernels and a four-row block-load experiment were timing-neutral
or slower and were reverted.

The fresh native-relative [31-trial solve](phase8-traininstance2-final-solve.json)
is 67.26 ms native versus 72.18 ms Rust, **1.073x**. The
[21-trial public pipeline](phase8-traininstance2-final-public.json) is
101.21 versus 104.79 ms, **1.035x**. Status, iteration count, and objective
match. New native/Rust presolved-state checkpoints at pivots 0, 1, 100, 500,
1000, 2000, and 2682 agree on basis, values, reduced costs, clocks, and
refactorization count; this is not an every-pivot trace. Full workspace tests,
strict Clippy, fast-96 trajectories, and the selected extra-model pivot
fixture pass. `traininstance2` is **not yet at
parity**. The remaining solve cost needs more precise localization within
the dense transpose and per-pivot paths; no speculative numerical rewrite
was retained.

An 11-trial matched iteration-cap sweep put the Rust/native solve ratio at
1.177x after one pivot, 1.085x after 500, and 1.073x at termination. The
remaining excess includes about 0.39 ms of fixed startup cost and persists
throughout the pivot sequence. As a build-only experiment, fat LTO improved
the matched `traininstance2` solve to 1.068x native in 31 trials, but it
regressed `d6cube` by about 1.5% in the paired Rust A/B; the release profile
therefore remains on thin LTO.

## Fresh indexed-access follow-up (14 October 2026)

At `1a9c753` (the bounds-access audit), the release public adapters were
rebuilt for Apple M2 and run serially with one warmup and three alternating
native/Rust trials per model. All 213 models completed with matching status
and iteration count and no wall limit. The summed Rust/native ratio was
**0.9757x**, the median model ratio **0.9580x**, and the sqrt(nnz)-weighted
geometric ratio **0.9672x**. See the [full raw run](phase8-nmm0-50k-public-2026-10-14-bounds.json).
This is one completed pass, not a new stable parity claim: a second pass
encountered a nonterminal native `pilot` result against its pinned fixture
and was discarded. Concurrent WindowServer/iTerm load was visible during
these measurements.
The pinned upstream checkout was `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`;
the native public adapter SHA-256 was
`399a6767b9e6822e37c00dcb62f2b8c4c9fba03e4dfd2abd44` and the Rust public adapter
SHA-256 was `1aa0b00a72e27e0184be8396621856722d7642c921bb3921e7dc95a89ae4905e`.

Seven-trial matched-stage [triage](phase8-nmm0-50k-stage-2026-10-14-bounds.json)
identified solve, not presolve, as the main remaining cost on selected models:

| Model | Native solve | Rust solve | Ratio | Excess |
|---|---:|---:|---:|---:|
| `cvs16r128-89` | 1.687 s | 1.752 s | 1.039x | 65 ms |
| `graph20-20-1rand` | 147.3 ms | 157.6 ms | 1.070x | 10.3 ms |
| `traininstance2` | 65.7 ms | 71.2 ms | 1.083x | 5.5 ms |
| `milo-v12-6-r2-40-1` | 101.9 ms | 106.5 ms | 1.045x | 4.6 ms |
| `d6cube` | 586.8 ms | 591.4 ms | 1.008x | 4.6 ms |
| `uct-subprob` | 62.2 ms | 65.5 ms | 1.052x | 3.3 ms |
| `neos-5107597-kakapo` | 55.3 ms | 58.4 ms | 1.056x | 3.1 ms |

Paired `sample` profiles of `cvs` and `graph` put substantial time on both
sides in triangular transposed solves, Harris ratio testing, update-row work,
and pricing. No single dominant Rust-only stage emerged. One exact access
difference was repaired: primal edge/Devex updates now read coefficients for
`UpdateRow`'s own reported positions without a per-read bounds check, as
upstream does. The [15-trial old/new Rust A/B](phase8-nmm0-50k-update-row-coeff-ab.json)
was 0.994x on `cvs` but 1.005x on `graph` and 1.004x on `milo`. A fresh
[nine-trial native-relative focus](phase8-nmm0-50k-update-row-coeff-focus.json)
still measures 1.018x, 1.070x, and 1.037x respectively. The access change
is retained for fidelity, not credited as a broad timing repair. Fast-96
strict trajectories and the selected extra-model pivot fixture pass.

The material `graph` gap remains open. A controlled, uncontended repeat of
the full corpus and kernel-level matched probes are needed before declaring
all selected instances at parity.

## LU permutation access follow-up (14 October 2026)

Three matched-stage profiles of `cvs16r128-89`, `rmatr100-p10`, and
`graph20-20-1rand` located the remaining excess in the presolved simplex
solve, not presolve. `rmatr` spent a substantial share of Rust samples in
LU's scattered permutation. Pinned GLOP traverses a validated permutation
and n-dimensional scratch vectors directly. Rust now uses narrowly scoped,
commented unchecked accesses there; it retains an explicit release-mode
check on each caller-mutable sparse position, so malformed safe-API input
still panics rather than causing undefined behavior.

Two 11-trial alternating old/new Rust matched-solve comparisons measured
0.988x/0.990x on `cvs`, 0.976x/0.977x on `rmatr`, and 0.989x/0.988x on
`graph` before the explicit sparse-position check was added. A seven-trial
public old/new check of the final checked variant measured 0.984x, 0.919x,
and 0.989x respectively; the unusually large `rmatr` gain should be read
alongside native-relative timings, not assumed to be solely the bounds-check
effect. Raw A/B diagnostics are under `target/phase8-permutation-access-*`.

After the final edit, a serial three-trial 213-model native-relative run
matched all statuses and iteration counts with no timeouts. Its aggregate
Rust/native ratio is **0.9756x** (23.680 / 24.273 s); excluding `qap12`,
0.9840x. Median model ratio is 0.9521x and sqrt(nnz)-weighted geometric
ratio is 0.9651x. Raw results are in
[the post-edit public fixture](phase8-nmm0-50k-public-permutation-access.json).
Seven-trial rechecks, in
[the final focus fixture](phase8-nmm0-50k-focus-permutation-access-final.json),
give:

| Model | Native → Rust | Rust/native | Excess |
|---|---:|---:|---:|
| `cvs16r128-89` | 1.690 → 1.729 s | 1.023x | +38.4 ms |
| `graph20-20-1rand` | 0.151 → 0.162 s | 1.067x | +10.2 ms |
| `milo-v12-6-r2-40-1` | 0.107 → 0.111 s | 1.035x | +3.7 ms |
| `traininstance2` | 0.099 → 0.103 s | 1.039x | +3.9 ms |
| `neos-5107597-kakapo` | 0.059 → 0.062 s | 1.039x | +2.3 ms |
| `d6cube` | 0.593 → 0.598 s | 1.009x | +5.1 ms |
| `rmatr100-p10` | 0.840 → 0.798 s | 0.949x | -42.6 ms |

Thus `rmatr` no longer needs intervention on this build; `cvs` and `graph`
remain the major repeatable gaps. `dfl001` is near parity in the three-trial
post-edit corpus (0.995x). The fast-96 strict trajectory fixture, extra-model
strict pivot fixture, full workspace tests, and Clippy pass.

## Fresh native-CPU rebuild at `a63f578` (14 October 2026)

The release Rust public adapter was rebuilt after adding
`-C target-cpu=native` to `.cargo/config.toml`; the paired pinned-GLOP
adapter is unchanged. On Apple M2, the established serial public-pipeline
harness ran one warmup and three alternating trials per side for all 213
manifest models, with a 20-second per-solve limit. Parsing and output are
excluded, while presolve, simplex, and postsolve are included. Every model
matched native status and iteration count; objectives passed the harness's
`1e-8` relative/absolute tolerance. No model hit the wall limit.

The sum of per-model median solve times was 24.148 s native and 23.859 s
Rust: **0.9880x Rust/native**. The median model ratio was 0.9568x and the
sqrt(nnz)-weighted geometric ratio 0.9679x. Excluding the 9.6-second
`qap12`, the aggregate ratio was 0.9957x. Raw results are in
[the 213-model fixture](phase8-nmm0-50k-public-a63f578.json). The dataset
manifest SHA-256 is
`f7e397964e5d26fa6897dab1d2fc65ae9c065c5783fabe4c1c7ce954522861f2`;
the Rust/native adapter hashes are
`426f58910ee2b8edb5f89df7ddf739459683bead870849cd4f43577875fbfa5a` /
`399a6767b9e6822e37c00dcb62f2f8b6d46a09fa7bf8c4c9fba03e4dfd2abd44`.

Eleven candidates received seven additional alternating trials each. Their
native and Rust trial ranges do not overlap except for `seymour`; this is a
repeatability screen, not a confidence interval. Full trials are in
[the focus fixture](phase8-nmm0-50k-focus-a63f578.json).

| Priority | Model | nnz | Native → Rust median | Ratio | Excess |
|---|---|---:|---:|---:|---:|
| 1 | `cvs16r128-89` | 14,576 | 1.699 → 1.761 s | 1.036x | 61.7 ms |
| 2 | `rmatr100-p10` | 29,236 | 0.844 → 0.875 s | 1.036x | 30.7 ms |
| 3 | `graph20-20-1rand` | 19,314 | 0.151 → 0.163 s | 1.076x | 11.5 ms |
| 4 | `dfl001` | 41,873 | 3.089 → 3.106 s | 1.006x | 17.6 ms |
| 5 | `d6cube` | 43,888 | 0.593 → 0.601 s | 1.014x | 8.1 ms |
| 6 | `milo-v12-6-r2-40-1` | 14,814 | 0.107 → 0.112 s | 1.047x | 5.1 ms |
| 7 | `traininstance2` | 41,532 | 0.099 → 0.103 s | 1.042x | 4.1 ms |
| 8 | `neos-5107597-kakapo` | 19,416 | 0.059 → 0.062 s | 1.055x | 3.3 ms |

`dfl001` is still reproducibly slower but now only 0.6%; investigate after
the larger percentage gaps unless a shared kernel explains it. `seymour1`
(1.022x, +3.7 ms) and `seymour` (1.019x, +3.2 ms; overlapping ranges) are
secondary. `neos-1445765` is 1.033x but only +0.9 ms in this public run;
its older much larger gap should not be extrapolated to this build.
Aggregate parity has been reached on this corpus, but these repeatable
individual regressions remain before per-instance parity can be declared.

## Focused follow-up after the fresh rerun

GLOP dispatches unit/nonunit triangular solves to separate template kernels
and traverses contiguous factor entries by index. The Rust dense solve now
does the same, with row and storage invariants documented at its unchecked
reads. Two alternating Rust A/B runs show `milo-v12-6-r2-40-1` at 0.985x
and 0.984x its prior solve time; `graph20-20-1rand` is neutral. A second,
direct-index RHS edit is neutral on `milo` and `graph` but places
`cvs16r128-89` at 0.993x and 0.994x in independent Rust A/B runs. The
native-relative matched-presolved-solve medians after these edits are:

| Model | Trials | Native | Rust | Rust/native |
|---|---:|---:|---:|---:|
| `milo-v12-6-r2-40-1` | 15 | 0.097 s | 0.104 s | 1.066x |
| `graph20-20-1rand` | 21 | 0.145 s | 0.159 s | 1.097x |
| `cvs16r128-89` | 7 | 1.694 s | 1.781 s | 1.052x |

Selected-view LU on the *presolved* basis sequence uses the original compact
matrix and retained factorization objects on both sides. Across 63 `milo`
bases, Rust/native is 1.084x, accounting for only about 1 ms of the full
solve excess. Across 49 `graph` bases it is 1.021x; across 286 `cvs` bases
it is 0.957x. Thus repeated simplex-pivot work, not isolated LU, remains the
primary target for `graph` and `cvs`. Sampling identifies a distributed mix
of transpose solves, update-row work, ratio tests, and pricing; no single
kernel yet explains the full gap. Bounds-check experiments in Harris ratios
and sparse permutation were timing-neutral and were reverted.

The native direct-dual pivot fixture now covers all four selected models,
including 2,111 `milo` and 8,793 `cvs` pivots. It and the fast-96 trajectory
gate pass. A single-trial public run over all 213 NMM 0–50k models still
matches every native status and iteration count, with no wall limits; its
aggregate ratio is 0.9943x (not a substitute for the paired timing runs
above). The local raw A/B, stage, and selected-view LU measurements are under
`target/phase8-*`; commands use `benchmark_phase8_stages.py` and
`benchmark_netlib_factorizations.py` with `--selected-source`.

## Clock and presolved-state follow-up

The Rust wall-limit check was scanning all 100 saved intervals on every call.
Pinned GLOP uses `RunningMax`, which recomputes only when an evicted maximum
must be replaced. Porting its ring update gives 0.991x `graph` and 0.993x
`milo` Rust A/B solve time in an 11-trial run (0.996x and 0.995x in a
21-trial combined recheck); `cvs` is timing-neutral. Using the same clock
sample for interval and deadline comparison, as upstream does, is also
timing-neutral. Native/Rust time-limit traces and deterministic clock-prefix
tests pass.

The opt-in `tools/validate_phase8_nmm_presolved_checkpoints.py` gate pins
14 native presolved states across these models. Rust matches each exactly for
status, iteration, basis hash, variable-value hash, reduced-cost hash,
deterministic-clock bits, and refactorization count. These checkpoints do not
prove every intermediate pivot, but they rule out a different state at the
sampled caps. Fixed-basis RHS probes on `graph` show some large *ratios* for
very short FTRAN calls; the summed absolute difference is under a millisecond
and does not account for the whole-solve excess. Accumulated basis updates
between refactorizations remain a better localization target.

The solver-owned middle-product update was then aligned with GLOP's compact
column append-and-clear path: duplicate touched rows are retained in traversal
order, their final values are appended once, and the scratchpad is cleared on
consume. This removes a Rust-only sparse-mask rebuild. Two serial Rust A/B
rechecks gave `graph` 0.989x and 0.992x, and `cvs` 0.996x twice; `milo` was
0.994x in the first recheck. Matched native-relative median presolved-solve
times after both changes are 1.057x for `milo` (15 trials), 1.086x for
`graph` (15), and 1.042x for `cvs` (7). These remain real gaps, most notably
`graph`; selected-view LU and fixed-basis RHS probes explain only a minority
of its whole-solve excess. The duplicate-row unit test, direct pivot fixture,
14 presolved-state checkpoints, and fast-96 trajectory gate pass.
The final one-pass 213-model public recheck after this update had matching
statuses and iteration counts throughout, no timeouts, and a 0.995x aggregate
Rust/native time ratio. `capri` alone differed by one objective bit, as in
the preexisting numerical tolerance checks. This one-pass aggregate is not a
substitute for the repeated matched timings above.

## Reusable upper-column follow-up

GLOP's `GetColumnOfU()` returns a reference to an LU-owned reusable
`SparseColumn`. Rust had been allocating a tuple vector on each basis update.
It now uses the same retained sparse-column buffer and `CleanUp()` path;
the buffer-reuse unit, direct extra-model pivot, 14 presolved-state, and
fast-96 trajectory gates pass. In two seven-trial Rust A/B runs against the
previous binary, `milo` was 1.000x and 0.996x, `graph` 1.002x and 1.001x,
and `cvs` 1.006x and 1.005x. Thus the ownership correction is not a timing
repair; the small `cvs` cost is retained for fidelity and remains open for
profiling.
The subsequent one-pass 213-model public check again matched every status
and iteration count, had no timeouts, and produced a 0.988x aggregate ratio.
Only `capri` differed by one objective bit, as before. This aggregate is
not evidence that the three focused presolved solves have reached parity.

## Fresh paired rerun at `8d51e03` (14 October 2026)

Two independent serial passes covered all 213 deduplicated models. Each pass
used one warmup per solver and three alternating native/Rust timed trials per
model, taking each model's median before aggregation. All statuses and
iteration counts match, the harness's objective tolerance passes, and no
model reached the 20-second wall limit. MPS loading is excluded; presolve,
simplex, and postsolve are included. The summary is:

| Pass | Native sum | Rust sum | Rust/native | Median model | sqrt(nnz)-weighted geometric |
|---|---:|---:|---:|---:|---:|
| First | 24.156 s | 24.005 s | 0.9937x | 0.9743x | 0.9824x |
| Repeat | 24.077 s | 23.927 s | 0.9938x | 0.9785x | 0.9834x |

`qap12` takes roughly 9.6 s native and 9.4 s Rust. Excluding it, the
aggregate is 1.0039x and 1.0020x in the two passes: close to parity, but
not evidence that every smaller model is at parity. Raw trials and outcomes
are in [first public run](phase8-nmm0-50k-public-2026-10-14.json) and
[repeat public run](phase8-nmm0-50k-public-2026-10-14-repeat.json).

The following public excesses recur in both passes; milliseconds are per
solve, computed from model medians. An eight-model, five-trial
[matched-stage run](phase8-nmm0-50k-stage-triage-2026-10-14.json) independently
places these costs in the presolved simplex solve rather than parsing or
presolve.

| Priority | Model | nnz | Public ratio, two passes | Extra time, two passes | Matched-solve ratio | Intervention |
|---|---|---:|---:|---:|---:|---|
| 1 | `cvs16r128-89` | 14,576 | 1.049x / 1.046x | 82 / 79 ms | 1.043x | Largest stable absolute solver gap; profile simplex stages first. |
| 2 | `rmatr100-p10` | 29,236 | 1.028x / 1.025x | 24 / 22 ms | 1.035x | Compare repeated numerical and pricing stages. |
| 3 | `d6cube` | 43,888 | 1.027x / 1.028x | 16 / 17 ms | 1.023x | Compare long-solve numerical stage accounting. |
| 4 | `graph20-20-1rand` | 19,314 | 1.094x / 1.096x | 14 / 15 ms | 1.095x | Highest substantial ratio; inspect distributed pivot kernels. |
| 5 | `milo-v12-6-r2-40-1` | 14,814 | 1.086x / 1.086x | 9 / 9 ms | 1.087x | Profile for a shared cause with `graph`. |

`dfl001` has a larger but less stable absolute excess (31 / 12 ms, public
ratios 1.010x / 1.004x); five matched-stage trials give 1.015x, so measure
more before attributing it to a specific kernel. `seymour`, `uct-subprob`,
and `neos-5107597-kakapo` show smaller recurring gaps. `neos-1445765` is
1.037x / 1.032x publicly, about 1 ms excess; its presolve is now at parity
(0.995x matched), though its isolated presolved solve remains 1.118x. Very
short cases with large ratios are lower priority until repeated timing can
separate fixed overhead from timer noise.

The pinned OR-Tools checkout was `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`
and the gloprs source was `8d51e0390d7a598b9f444094002b7a1b9131b1cc`.
The native library was current in its Release CMake build (`-O3 -DNDEBUG
-mcpu=apple-m2`); both adapters were rebuilt at `-O2 -DNDEBUG
-mcpu=apple-m2`. Rust examples were rebuilt in Release with
`RUSTFLAGS='-C target-cpu=native'` on Apple M2 (Rust 1.95.0), one codegen unit,
and ThinLTO. SHA-256 fingerprints: native library
`1a2f5b7426bc01abb87ee6104095dc7bd4fb75dd43001637004bf2c601ac41f3`,
public native/Rust adapters
`399a6767b9e6822e37c00dcb62f2f8b6d46a09fa7bf8c4c9fba03e4dfd2abd44` /
`40cb44f8a256c3f7e96108ab7e583f6c9d6cdabc2b4a260fc7fbc2fc2083e621`.
Dataset manifest SHA-256:
`f7e397964e5d26fa6897dab1d2fc65ae9c065c5783fabe4c1c7ce954522861f2`.

The older sections below record earlier builds and measurements; their
outlier rankings are superseded by this fresh run.

The fresh serial public-solver comparison completed all 213 deduplicated
instances. Native GLOP and gloprs agree on status, iteration count, and the
harness's objective tolerance for every model. The sum of per-model median
solve times is 24.124 s native and 24.358 s Rust (1.010x); the median model
ratio is 1.010x and the sqrt(nnz)-weighted geometric ratio is 1.023x. There
were no wall-limited or nonterminal results. `qap12` alone takes about 9.6 s
native and 9.4 s Rust, masking some shorter-model regressions in the total;
excluding it gives 1.027x over the other 212 models.

## Method and artifacts

- Apple M2; Rust 1.95.0 Release with `-C target-cpu=native` (Apple M2), one
  codegen unit, ThinLTO. Pinned OR-Tools
  `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5` was rebuilt Release with
  `-O3 -DNDEBUG -mcpu=apple-m2`; the native adapters used `-O2 -DNDEBUG
  -mcpu=apple-m2`. Rust source HEAD was `caeb0ccc0f4248ad6d53266b4d92d3374613f322`
  plus documentation and benchmark-harness edits; no solver code changed.
- Selection: `../datasets/nmm0-50k/manifest.json` SHA-256
  `f7e397964e5d26fa6897dab1d2fc65ae9c065c5783fabe4c1c7ce954522861f2`.
  The 213 canonical entries comprise 91 Netlib, 91 MIPLIB, and 31 Mittelmann
  models. Source MPS files are symlinked, not copied.
- Public pipeline: one warmup per side, then three serial alternating
  native/Rust trials per model. The 20-second per-solve wall limit was not hit.
  Timing starts from the parsed model and includes presolve, simplex, and
  postsolve; file loading and trace printing are excluded. Full trial times,
  statuses, iterations, and objective bits: [public JSON](phase8-nmm0-50k-public.json).
- Ten leading candidates received the same warmup/trial pattern for isolated
  parse, presolve (five repetitions), and matched-presolved solve stages.
  Raw data: [stage JSON](phase8-nmm0-50k-stage-triage.json). The previously
  interrupted one-trial screen was discarded.
- Native library SHA-256: `1a2f5b7426bc01abb87ee6104095dc7bd4fb75dd43001637004bf2c601ac41f3`.
  Native/Rust public adapter hashes: `399a6767b9e6822e37c00dcb62f2f8b6d46a09fa7bf8c4c9fba03e4dfd2abd44`
  / `5d9f7b8daa1aaf4f69dac10935c04bd609247b0ca25a8fe778faeeb32d1a4e7a`.

## Repeatable gaps to investigate

The following matched-solve native and Rust trial ranges do not overlap;
their ratios therefore exceed the observed run-to-run variation in this
sample. This is a triage criterion, not a formal confidence interval.

| Model | nnz | Matched solve native → Rust | Rust/native | Excess per solve | First target |
|---|---:|---:|---:|---:|---|
| `cvs16r128-89` | 14,576 | 1.694 → 1.824 s | 1.077x | 130 ms | Simplex stages; parse/presolve do not explain the gap |
| `dfl001` | 41,873 | 3.082 → 3.175 s | 1.030x | 92 ms | FTRAN/basis-update/pricing shares in the long solve |
| `d6cube` | 43,888 | 0.589 → 0.623 s | 1.057x | 34 ms | Simplex numerical-stage accounting |
| `rmatr100-p10` | 29,236 | 0.838 → 0.869 s | 1.037x | 31 ms | Simplex stages; parse overhead is smaller |
| `graph20-20-1rand` | 19,314 | 0.148 → 0.164 s | 1.112x | 16 ms | Triangular/update-row/ratio/pricing mix |
| `seymour` | 34,921 | 0.164 → 0.175 s | 1.070x | 11 ms | Simplex stages; `seymour1` shows a similar public gap |
| `milo-v12-6-r2-40-1` | 14,814 | 0.102 → 0.112 s | 1.098x | 10 ms | Simplex stages |
| `uct-subprob` | 11,492 | 0.063 → 0.069 s | 1.093x | 6 ms | Simplex stages |

`neos-1445765` has the highest credible public ratio among nontrivial short
models (1.220x, +6.2 ms). Its stage gaps are split: presolve 17.8 → 22.3 ms
(1.253x) and matched solve 9.5 → 12.2 ms (1.284x), each with nonoverlapping
trial ranges. Investigate both paths; a simplex-only profile would miss much
of its public gap. `app1-1` is 1.130x in the public solve but only +3.7 ms;
matched solve is 1.139x, with an additional roughly 1 ms parse gap. It is a
secondary target after the larger absolute excesses.

The Mittelmann `n370*` family mostly measures 8–9 ms native per model. Ratios
around 1.04–1.11x may indicate a shared small-model cost, but their sub-ms
individual excesses are lower priority and warrant repeated family-level
measurement before kernel changes. Likewise, very short models should not be
ranked by ratio alone.

## Next step

Profile matched native and Rust simplex stages on `cvs16r128-89` first, then
`dfl001`, `d6cube`, and `graph20-20-1rand`, using the same presolved inputs.
Compare the implicated C++ and Rust control flow, arithmetic, and workspace
lifecycle before editing. Re-run strict trajectory/KKT gates and this paired
corpus benchmark after any numerical-kernel change.

## Trailing-slack update-row follow-up

Matched profiling found a specific porting difference: native GLOP reads the
unit-row-left-inverse coefficient directly for an equation-form slack column,
whereas Rust traversed that compact sparse column and formed a dot product.
Rust now recognizes the exact trailing identity block once when binding the
compact matrix, and uses the direct lookup only for that block. The focused
identity test, fast-96 strict trajectory fixture, workspace tests, and selected
native/Rust solution comparisons passed.

The serial 213-model public rerun used the same one-warmup/three-alternating-
trial protocol and 20-second wall limit as above. Its raw data is in
[the post-slack public fixture](phase8-nmm0-50k-public-slack-fastpath.json). All
213 statuses and iteration counts match; no model was wall-limited. The only
objective-bit difference is one ULP on `capri`, which also occurred before
this change. Aggregate Rust/native time is 1.005x (24.232/24.122 s), median
model ratio 1.011x, and sqrt(nnz)-weighted geometric ratio 1.024x. The last
two metrics are essentially unchanged; the aggregate improves partly because
the largest case, `qap12`, remains slightly faster in Rust.

| Model | Previous public ratio | New public ratio | New excess |
|---|---:|---:|---:|
| `cvs16r128-89` | 1.093x | 1.059x | 101 ms |
| `dfl001` | 1.021x | 1.018x | 56 ms |
| `d6cube` | 1.054x | 1.058x | 35 ms |
| `rmatr100-p10` | 1.050x | 1.021x | 18 ms |
| `graph20-20-1rand` | 1.110x | 1.113x | 17 ms |
| `neos-1445765` | 1.220x | 1.224x | 6 ms |

Cross-run changes are directional evidence, not a randomized A/B estimate.
The repeatable remaining excesses on `d6cube` and `graph20-20-1rand`, plus
the separate presolve/solve costs on `neos-1445765`, need matched-stage
localization before another kernel edit.

## Focused `neos` and `graph` investigation

Seven alternating trials with two warmups and repeated stage calls showed:

| Model | Parse | Presolve | Matched presolved solve |
|---|---:|---:|---:|
| `neos-1445765` | 0.93x | 1.21x | 1.29x |
| `graph20-20-1rand` | 1.13x | 0.95x | 1.11x |

The presolve adapter previously used Rust's `LinearProgram::clone()` where
native and the public Rust solver use `PopulateFromLinearProgram()`. Correcting
that harness mismatch reduced the measured `neos` presolve ratio from about
1.25x to 1.21x; most of the gap remains. Sampling places most of its presolve
time in the same doubleton-free-column sparse merge as upstream. Caching the
two input lengths in that merge, matching the upstream loop, had no clear
timing effect. Raw stage runs are under `target/nmm-two-stage-*.json`.

An optional iteration cap in both presolved-solve diagnostic adapters gives
the following 100-repetition comparison (milliseconds per solve; cap zero
takes a distinct early-exit path, so it is not a literal setup-time intercept):

| Model | Cap | Native | Rust |
|---|---:|---:|---:|
| `neos-1445765` | 100 | 0.94 | 2.47 |
| `neos-1445765` | 500 | 2.42 | 3.96 |
| `neos-1445765` | 1295 | 9.18 | 11.87 |
| `graph20-20-1rand` | 100 | 2.14 | 3.24 |
| `graph20-20-1rand` | 500 | 10.24 | 12.97 |
| `graph20-20-1rand` | 3092 | 146.87 | 163.39 |

The initial basic-value computation in Rust allocated a separate dense RHS
and copied it into a scattered solve vector. It now accumulates directly in
that vector, traverses the nonbasic bitset, and calls the compact column-add
helper so zero-valued nonbasic columns are skipped like GLOP. The latter was
an actual hot-loop fidelity gap. `DynamicMaximum::add_or_update` is now marked
inline like the upstream header method, though this hint had no separately
measurable effect. Status, pivot counts, and objective bits on the two focus
models, the full workspace tests, and fast-96 strict trajectories pass. The
final seven-trial stage recheck measured 1.211x `neos` presolve, 1.269x `neos`
solve, and 1.102x `graph` solve; the paired run is
[the focus-stage fixture](phase8-nmm0-50k-focus-stages.json). The remaining `neos` gap is concentrated in
early-solve/pricing setup plus a smaller later increment; `graph`'s gap is
chiefly repeated pivot work. Neither model is yet at parity.
