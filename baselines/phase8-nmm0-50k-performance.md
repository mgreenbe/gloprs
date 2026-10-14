# NMM 0–50k: M2 performance triage

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
