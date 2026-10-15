# Iterator / counted-loop audit — 15 October 2026

Baseline: gloprs `77b83ac`; upstream GLOP
`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`.
Apple M2, macOS 15.6.1 arm64, Rust 1.95.0; release, native CPU target,
one codegen unit, thin LTO (`Cargo.toml` and `.cargo/config.toml`).

## Scope and conclusion

The production-source inventory covers `glop/src` and `lp_data/src`, stopping
before each module's test section. It finds iterator adapters in 35 modules:
244 `.iter()`, 22 `.iter_mut()`, 26 `.zip()`, 21 `.rev()`, 27 `.fold()`,
106 `.map()`, 19 `.filter()`, 68 `.enumerate()`, and eight `.chunks_exact()`
calls. These are syntax counts, not independent loops or measurements of cost;
see `iterator-audit-2026-10-15-inventory.json`.

The CLI driver has no data-processing loops of its own; it delegates to these
core crates. Benchmark/example code is treated as measurement infrastructure.

The audit prioritized numerical and simplex hot paths, then checked startup,
preprocessing, and diagnostic adapters for comparable issues. It does not
claim to have benchmarked every iterator or every architecture.

The focused audit initially identified two candidates. **The later full-corpus
retention gate rejected both. No production code change remains relative to
`77b83ac`.** The candidate mechanisms were:

- The rejected packed rank-one prototype used upstream's counted four-accumulator loop
  and fixed 1–3-entry tail, replacing paired `chunks_exact(4)` iterators.
  Four accumulators, FMA order, entry order, packed storage, and RHS validation
  are unchanged. Both private entry arrays cover the same update span; the
  counted cursor bounds their unchecked reads. No additional per-pivot scan,
  allocation, or change to deterministic operation accounting is introduced.
- The rejected pricing-heap prototype used libc++'s counted descending
  parent cursor. It replaces a reverse **inclusive** range, including its
  exhaustion bookkeeping, while preserving every sift, tie, and random draw.

Ordinary exclusive ranges, slice iteration, and `zip` are not generally a
performance problem. The existing lower-transpose repair removed a reverse
inclusive range from a substitution loop. That result did not generalize to
upper substitution or compact-column iteration in this audit.

## Profile evidence

Six-second, 1-ms baseline process samples of repeated matched-presolved solves
put these functions at the top of the stack (sample counts, not elapsed-time
percentages):

| Function | graph20-20-1rand | neos-1171737 |
|---|---:|---:|
| Triangular transpose solve | 781 | 700 |
| Row-wise update row | below threshold | 887 |
| Hypersparse row-wise update row | 465 | 71 |
| Primal edge-norm update | 360 | 623 |
| Harris ratio test | 601 | 296 |
| Packed rank-one dense left solve | 153 | 271 |
| Packed rank-one sparse left solve | 113 | 67 |
| Packed rank-one sparse right solve | 96 | 130 |

These samples justify testing rank-one and sparse traversal loops; they do not
by themselves demonstrate iterator overhead. Collapsed sample summaries are
saved in `iterator-audit-2026-10-15-profiles.txt`. Full local profiles remain
under `target/iterator-audit-{graph,neos}.sample.txt`.

## Candidate audit

| Rust location / construct | Pinned counterpart and finding | Decision |
|---|---|---|
| `triangular_matrix.rs`: exclusive reverse upper-solve range | `sparse.cc::UpperSolveInternal` uses a signed counted column loop. Explicit exclusive cursor measured 1.001–1.018x baseline across eight upper cases. Other triangular modes were controls. | Reverted: no gain. |
| `triangular_matrix.rs`: lower/upper transpose, hypersparse loops | Lower-transpose's descending substitution loop is already repaired. Forward transpose uses a contiguous cursor; hypersparse reverse support corresponds to upstream `Reverse(non_zero_rows)`, not a dense counted scan. | Preserve order and sparse support. |
| `sparse.rs::CompactColumn::iter`: paired slice zip | `sparse.cc`/`update_row.cc` traverse compact entry-index ranges. Replacing zip with one integer entry cursor produced the same hot row-wise/hypersparse instructions, apart from anonymous diagnostic addresses. Baseline already has one countdown in the inner loop. | Reverted: no code-generation benefit. |
| `update_row.rs`: signed-index conversion inside row iterator | Validated transpose permits removing the repeated fallible conversion. Prototype removed the sign test and enabled a different vectorized loop, but neos solve slowed to 1.023x baseline (+6.14 ms). Graph/train were 0.996x/0.993x. | Reverted: measured workload regression. |
| `rank_one_update.rs::packed_scalar_product`: zipped chunks | `sparse.h::ConstView::ColumnScalarProduct` has one counted four-entry loop and fixed tail. A first counted variant with a looped tail was inconsistent, including a 1.104x short-column microcase. The final fixed-tail variant has smaller dense-left code (the counted-loop prototype was 205 versus 220 disassembly lines) and repeatable small matched-solve gains. | Rejected by the later full-corpus gate; reverted. |
| `rank_one_update.rs`: reverse update traversal and zipped output entries | Upstream counts elementary matrices backward and traverses compact output columns. These are ordinary slice/exclusive traversals; profile cost includes actual sparse arithmetic. | No blanket rewrite. |
| `pricing.rs::make_heap`: reverse inclusive parent range | Upstream `std::make_heap` delegates to libc++'s counted descending parent loop. Small pricing benchmarks improve 4.4% (31 candidates) and 2.7% (64); 1024/4096-candidate scans are neutral. | Rejected by the later pricing-only full-corpus gate; reverted. |
| `initial_basis.rs::CandidateHeap::from_vec`, `libcpp_sort.rs::heap_sort`: inclusive ranges | Other libc++ heap loops are candidates for a future startup/adversarial-sort audit. Neither was a material hotspot in these profiles; the introsort heap fallback is conditional. | Identified, left unchanged without supporting measurements. |
| `primal_edge_norms.rs`, `dual_edge_norms.rs`, `reduced_costs.rs` | Hot incremental primal paths already follow compact indexed and bitset traversal. Dense dual `enumerate()` retains one index and follows the counted upstream recurrence. Precision reductions require their current operation order. | No measured iterator-state defect. |
| `lu_factorization.rs`, `markowitz.rs`, `basis_representation.rs` | Hot permute/gather and dense eta loops use contiguous arrays or exact sparse support, like upstream. Zipped equal-length dense eta slices are already length-checked. Permuted symbolic/sparse loops correspond to upstream sparse/range traversal. | Preserve sparse architecture and buffer reuse. |
| `lp_utils.rs`, `sparse_vector.rs`, `scattered_vector.rs`, `lp_types.rs` | Custom iterators walk sparse support or set bits, as native does. Counting all rows instead would change complexity. Norm kernels already reproduce native unrolling/contraction. | No broad replacement; public caller-controlled gathers stay checked. |
| Parser, preprocessing, scaling, model utilities, stats, diagnostics | Token/support/undo iterations generally have upstream range analogues; collections and allocations are a separate issue. Their timing was not implicated by matched-solve profiles. | No speculative syntax-only edits. |

## Initial focused measurements (superseded by the full retention gate)

All A/B trials alternate execution order with one warmup per binary. The
microbenchmarks exclude matrix/update construction and output, but include RHS
reset; pricing includes candidate population and first maximum selection.
Checksums must agree. Recorded solve comparisons are on the same presolved
model and exclude parse, presolve, and postsolve. The final harness requires
identical status, iteration count, and objective bits for Rust A/B runs.

The initial rank-one fixed-tail solve comparisons gave:

| Model | 21-pair after/before | Independent 31-pair after/before |
|---|---:|---:|
| graph20-20-1rand | 0.9992 | 0.9970 |
| neos-1171737 | 0.9905 | 0.9896 |
| traininstance2 | 0.9950 | 0.9911 |

The extended rank-one microbenchmark covers 8 and 96 updates, left/right
solves, and 1, 2, 3, 4, 5, 7, 16, 17, and 64 entries, with 21 alternating
trials. Ratios range from 0.972x to 1.026x: most are near neutral, with the
larger update sets and four/five/seven-entry columns commonly benefiting.
It would be misleading to claim a universal microkernel speedup.

Pricing-only full-solve ratios were 0.9994/0.9923/1.0032 on graph/neos/train.
Its neos result is not evidence that heap construction alone accounts for the
whole difference; most whole-solve differences are small and have code-layout
and system-noise sensitivity. Early exploratory runs overlapped some rebuilds;
the final serial combined confirmation below supersedes their timing claims.

The initial combined candidate, 31 alternating pairs after builds/tests finished:

| Model | Baseline median (ms) | Final median (ms) | Final/baseline | Trial ranges, baseline / final (ms) |
|---|---:|---:|---:|---|
| graph20-20-1rand | 149.340 | 149.661 | 1.0021 | 147.57–153.11 / 147.39–152.70 |
| neos-1171737 | 270.359 | 268.907 | 0.9946 | 268.17–290.25 / 267.62–283.54 |
| traininstance2 | 67.116 | 67.348 | 1.0034 | 65.68–71.05 / 66.53–69.33 |

Neos improves about 0.54% (1.45 ms); graph/train are within 0.35% of baseline.
The serial three-trial fast-96 sweep totals 9.015133 s baseline versus
9.031533 s final (1.0018x), with a median per-model ratio of 1.001.
All compared status/iteration outcomes agree. The largest apparent excesses
were fit2d (+13.86 ms, 1.0125x) and pilot87 (+11.20 ms, 1.0095x).
Independent 15-pair rechecks give **1.0018x and 0.9995x**, respectively;
the initial ~1% excesses do not repeat. Do not claim a broad solver speedup.

No builds or other test suites ran concurrently with the final 31-pair
confirmation or fast-96 sweep. A post-campaign process snapshot showed no
busy indexing/media services (highest unrelated process below 3% CPU). This
is still desktop timing, not a controlled dedicated-machine experiment.

One-shot peak RSS checks of the initial combined candidate, including
parsing/presolve, were 12,976,128 /
12,353,536 bytes before/after for graph and 14,909,440 / 15,122,432 for neos.
These startup-sensitive samples show no material memory change and do not
support a memory-reduction claim. Raw values are in the memory JSON. Parsing,
presolve, and factorization algorithms were not changed; no isolated LU
speedup is claimed. The checkpoint validation retains refactorization counts.

## Reproduction and artifacts

Build and save baseline binaries **before** changing sources. The enhanced
`lower_transpose_benchmark` example accepts `lower`, `upper`, `transpose-lower`
(default), and `transpose-upper` modes. It converts the same deterministic
lower input into upper storage when requested. `pricing_loop_benchmark` times
pricing-cache construction via the public API without diagnostic allocation.

```sh
cargo build --release -p gloprs-glop --example full_presolve_netlib \
  --example rank_one_solve_benchmark --example lower_transpose_benchmark \
  --example pricing_loop_benchmark
# Save each before/after executable under target/ before the next rebuild.
python3 tools/benchmark_iterator_audit.py --kind triangular \
  --before target/iterator-triangular-before --after target/iterator-triangular-upper \
  --trials 11 --output target/iterator-upper-micro.json
python3 tools/benchmark_iterator_audit.py --kind rank \
  --before target/iterator-rank-before --after target/iterator-rank-tail \
  --trials 21 --output target/iterator-rank-tail-extended.json
python3 tools/benchmark_iterator_audit.py --kind pricing \
  --before target/iterator-pricing-before --after target/iterator-pricing-counted \
  --trials 31 --output target/iterator-pricing-micro.json
python3 tools/benchmark_iterator_audit.py --kind solve \
  --before target/iterator-audit-before --after target/iterator-audit-final \
  --trials 31 --output baselines/iterator-audit-2026-10-15-final-solve.json
python3 tools/benchmark_phase8_ab.py --a target/iterator-audit-before \
  --b target/iterator-audit-final --fast96 --trials 3 --mode presolved
```

For the rejected upper variant, replace only the exclusive reverse column
`for` with `let mut column = dimension; while column > first { column -= 1; … }`.
For the rejected compact zip variant, use `(0..rows.len()).map(...)` with the
same paired private-array reads. For the rejected signed-index variant, use
`column.value() as usize` only in `compute_row_wise_inner::<true>` and retain
`to_usize()` in the checked fallback. No rejected production changes remain.

The JSON files with prefix `iterator-audit-2026-10-15-` retain per-case raw
trials, medians' ratios, compiler/platform information, and binary SHA-256s.
The inventory is syntactic; sample summaries are explanatory, not benchmarks.
The final fast-96 text artifact retains every model's raw timing trials.

## Full nmm0-50k retention gate

At the user's request, retention was reopened and gated on a full
public-pipeline comparison and material-outlier rechecks.
Fresh public adapters were built from `77b83ac` (both production files reset
to HEAD for the build) and from the two candidate source files, then saved
separately. A `finally` block restored the candidate source even on a build
failure. Provenance and SHA-256s are in
`iterator-audit-2026-10-15-nmm-builds.json`.

The first full campaign uses five alternating timed pairs per model, one
warmup per side, and a 60-second per-solve safety cap. Parsing and output are
excluded; the timed public solve includes presolve, simplex, and postsolve.
No cases reached the cap. In the existing harness, fields named `native` refer
to the **pre-change Rust** adapter and `rust` to the candidate, not native GLOP.

| Scope | Before | Candidate | Candidate / before |
|---|---:|---:|---:|
| Sum of 213 per-model median solve times | 23.160118 s | 23.083541 s | 0.996694 |
| Median model ratio | — | — | 0.999692 |
| sqrt(nnz)-weighted geometric ratio | — | — | 0.999016 |

All 1065/1065 timed pairs match status, iterations, and objective bits exactly;
213/213 models completed without wall limits. The first pass suggests a small
0.33% aggregate gain, not a broad large speedup. The material absolute
slowdowns are CMS750_4 (+7.06 ms, 1.0097x), qap10 (+2.59 ms, 1.0018x),
greenbea (+2.11 ms, 1.0079x), and d2q06c (+1.89 ms, 1.0045x). The largest gain
is rmatr100-p10 (-51.78 ms, 0.9377x); qap12 contributes another -21.04 ms
(0.9977x). The four slowdowns and rmatr gain were rechecked with 15 alternating pairs
and two warmups per side. Results and isolation are recorded below.

```sh
python3 tools/benchmark_phase5_netlib.py --corpus nmm0-50k --mode default \
  --trials 5 --warmups 1 --wall-seconds 60 \
  --native target/iterator-nmm-before-public --rust target/iterator-nmm-after-public \
  --output baselines/iterator-audit-2026-10-15-nmm-public-ab.json
python3 tools/benchmark_phase5_netlib.py --corpus nmm0-50k --mode default \
  --models CMS750_4 qap10 greenbea d2q06c rmatr100-p10 \
  --trials 15 --warmups 2 --wall-seconds 60 \
  --native target/iterator-nmm-before-public --rust target/iterator-nmm-after-public \
  --output baselines/iterator-audit-2026-10-15-nmm-public-recheck.json
```

The 15-pair combined-candidate recheck found:

| Model | Candidate / before | Excess (ms) |
|---|---:|---:|
| CMS750_4 | 1.01274 | +9.245 |
| qap10 | 1.00037 | +0.541 |
| greenbea | 1.00559 | +1.486 |
| d2q06c | 0.99366 | -2.657 |
| rmatr100-p10 | 0.93539 | -54.303 |

Two further builds isolate one change at a time. Eleven paired trials per
model, one warmup, give:

| Model | Rank-only / before | Excess (ms) | Pricing-only / before | Excess (ms) |
|---|---:|---:|---:|---:|
| CMS750_4 | 1.01279 | +9.283 | 1.00011 | +0.078 |
| greenbea | 1.00884 | +2.353 | 0.99811 | -0.502 |
| rmatr100-p10 | 0.93874 | -51.450 | 0.97698 | -19.303 |

**Revert the rank-one rewrite.** Its small focused improvements did not justify
these repeatable full-pipeline regressions. The original chunk iterators are
restored without altering public validation or arithmetic. The pricing-only
binary then underwent its own complete 213-model, five-pair campaign before
retention. The initial aggregate gain does not override per-model regressions.

Pricing-only full campaign, again five alternating pairs and one warmup on
all 213 models:

| Scope | Before | Pricing-only | Pricing-only / before |
|---|---:|---:|---:|
| Sum of 213 per-model median solve times | 22.960675 s | 22.882338 s | 0.996588 |
| Median model ratio | — | — | 0.998858 |
| sqrt(nnz)-weighted geometric ratio | — | — | 0.999227 |

All 1,065 timed pairs again match status, iterations, and objective bits
exactly, with zero timeouts. CMS750_4 is 1.00039x (+0.276 ms), greenbea
1.00089x (+0.236 ms), and rmatr100-p10 0.97458x (-21.257 ms). The initial
pricing-only slowdowns above 1 ms are cvs16r128-89 (1.00773x, +12.863 ms),
dfl001 (1.00146x, +4.320 ms), and 80bau3b (1.00901x, +1.008 ms); these are
checked with 15-pair/two-warmup reruns before the final decision.

Full and recheck raw artifacts are
`iterator-audit-2026-10-15-nmm-pricing-full.json` and
`iterator-audit-2026-10-15-nmm-pricing-recheck.json`. The same public benchmark
command above is used with `--rust target/iterator-nmm-pricing-only-public`;
the full run has no `--models` filter, and the recheck selects the three models
listed above with `--trials 15 --warmups 2`.

The final pricing-only rechecks are:

| Model | Pricing-only / before | Excess (ms) |
|---|---:|---:|
| cvs16r128-89 | 1.011278 | +18.621 |
| dfl001 | 1.000506 | +1.507 |
| 80bau3b | 0.980893 | -2.167 |

All 45 timed recheck pairs match exact outcomes. dfl001 is effectively
neutral and the initial 80bau3b regression does not repeat. The CVS slowdown
**does repeat** and is larger in the focused comparison than in the full pass.

**Final decision: revert the pricing rewrite as well.** The ~0.34% summed-time
gain and small pricing microbenchmark win are insufficient grounds to retain
its repeatable 1.13%/18.6-ms full-pipeline regression. Both production files
are byte-for-byte restored to HEAD; the source diff under `glop/src` and
`lp_data/src` is empty. Only audit documentation, raw measurement artifacts,
and benchmark infrastructure remain changed. No full-corpus performance
improvement is claimed for the final workspace: it retains the baseline.

These desktop measurements contain occasional outliers (for example, an early
dfl001 before trial at 3.83 s). Raw distributions are retained rather than
reporting only aggregate ratios. The retention decision is deliberately
conservative; it does not establish that either iterator form is slower on
every model or architecture.

## Validation

The initial combined candidate passed all of these correctness gates before
the full performance campaign rejected it:

- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
- `cargo test --workspace --all-features`;
- release `main_preprocessor_netlib` ignored tests: fast-96 bit-exact presolve
  and native status/iteration counts;
- release `netlib_trajectories`: every pivot of all fast-96 models;
- release `phase8_extra_trajectories`: all six selected extra-model paths;
- release `phase5_netlib_solution fast_96_public_postsolve_matches_native`:
  recovered public solutions and independent KKT checks;
- `validate_phase8_nmm_presolved_checkpoints.py`: all 21 pinned native/Rust
  state fingerprints, including variable values, reduced costs, basis,
  deterministic clocks, and refactorization counts;
- rank-one existing bit-pattern tests on lengths 0–28, sparse/dense update
  agreement, malformed/short RHS checks, and pricing's pinned tied-heap
  sequence, as part of the workspace tests.

Local logs are `target/iterator-{clippy,tests,differential,kkt,checkpoints}.log`.
The benchmark runner also compiles as Python and the Rust example targets
pass Clippy. After reverting both production edits, formatting, Clippy, and
the empty production-source diff were checked again. The final production
implementation is exactly the already-tested `77b83ac` baseline. No new
numerical fixtures or tolerance changes were required.

Phase 8 remains complete. This is a focused follow-up, not a new claim of
full-corpus or cross-platform performance parity. The later full 213-model campaign above compares Rust before/after; a fresh
213/166-model comparison to native GLOP was not run.
