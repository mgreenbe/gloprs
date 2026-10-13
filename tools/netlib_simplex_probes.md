# Netlib simplex numerical-kernel probes

Build the opt-in diagnostic and choose Netlib models:

```sh
cargo build --release --example netlib_simplex_probe
python3 tools/generate_netlib_simplex_probes.py --all --wall-seconds 600
```

Pass instance names instead of `--all` to select a subset. The complete
98-model Netlib run produces 2,593 factorization matrices, 2,593 solve-basis
matrices, 161,608 FTRAN right-hand sides, and 5,809 BTRAN right-hand sides.
Validate Matrix Market syntax, dimensions, and basis/RHS grouping with:

```sh
python3 tools/validate_netlib_simplex_probes.py
```

The default output is `../datasets/netlib-simplex-probes/`. Each directory is
`<instance>-<pivot>/`, where `pivot` is the iteration count of the resulting
basis (zero for the initial basis). If multiple factorizations occur at one pivot, their distinct
four-digit serials keep all files in that directory without overwriting data.
The generator refuses to overwrite a conflicting existing file, but an
identical run is safe to repeat.

Files are Matrix Market coordinate, real, general, with one-based row/column
coordinates and 17 digits after the decimal point in scientific notation:

- `factorization-SSSS.mtx`: the square basis matrix actually passed to LU.
- `basis-SSSS.mtx`: the square basis in the column order governing captured
  solves. It can differ from `factorization-SSSS.mtx` when GLOP absorbs a
  column permutation after constructing LU.
- `ftran-SSSS-NNNNNN.mtx`: an `n × 1` right-hand side for `B x = b`.
- `btran-SSSS-NNNNNN.mtx`: an `n × 1` right-hand side for `Bᵀ x = b`.

Only inputs taken before the next pivot, while that exact basis has no
rank-one updates, are recorded. Thus each RHS pairs with its same-serial
`basis-SSSS.mtx` for standalone FTRAN/BTRAN benchmarking. The recorder covers
ordinary dense/scattered solves, unit-row BTRAN, problem-column and tau
FTRAN, and FTRANs used for edge-norm initialization. It is entirely opt-in;
ordinary solves neither serialize nor materialize basis columns for probes.

These are bases of the *presolved* equation-form LP, including slack columns,
under the project's default GLOP parameters. They are not the original MPS
constraint matrix. The probe example runs preprocessing and scaling before
the direct revised-simplex solve and prints terminal status, iteration count,
and the post-initial-construction refactorization count. A wall limit can be
set with `--wall-seconds`; a truncated solve yields only the probes reached
before that limit.

For a single instance, the release example can also be run directly:

```sh
target/release/examples/netlib_simplex_probe ../datasets/netlib/mps/afiro.mps 120
```

For a direct, unscaled dual-simplex comparison matching `netlib_timing`, use
the opt-in factorization-only mode. The final positional argument is the
iteration cap (40,000 by default):

```sh
target/release/examples/netlib_simplex_probe \
  ../datasets/netlib/mps/qap15.mps 600 \
  target/phase8-qap15-direct-probes --direct-factorizations 40000
```

This mode skips preprocessing and scaling, saves only
`factorization-SSSS.mtx`, and avoids the large RHS corpus. It does not change
the default presolved/scaled probe behavior described above.

## Sparse FTRAN/BTRAN benchmark

The paired adapters benchmark the same sparse LU solve stages that GLOP uses
after absorbing its factorization column permutation. They load
`factorization-SSSS.mtx` and `basis-SSSS.mtx`, verify that the latter is the
former in the absorbed column order, and factorize the former **before** the
timer. Each right-hand side is read and scattered before its solve timer;
FTRAN times `RightSolveLWithNonZeros` plus `RightSolveUWithNonZeros`, and BTRAN
times `LeftSolveUWithNonZeros` plus `LeftSolveLWithNonZeros`. The harnesses do
not time Matrix Market parsing, factorization, RHS preparation, or checksums.
They benchmark a standardized sparse representation of every saved RHS;
the probe format does not record whether the original simplex call used a
dense, scattered, or specialized entry point.

Build and compare serially:

```sh
cargo build --release --example netlib_solve_probe_benchmark
python3 tools/build_glop_reference_adapter.py \
  --source tools/netlib_solve_probe_reference_adapter.cc \
  --output target/native/netlib_solve_probe_reference_adapter
python3 tools/benchmark_netlib_solve_probes.py \
  --trials 3 --repetitions 10 --warmups 1
```

The Python runner alternates native/Rust order by basis and trial, checks
dimension, input nonzero count, and three solution summaries per RHS, and
writes median per-RHS solve times to `target/netlib-solve-probe-timings.csv`.
Use `--models qap15 qap12`, `--limit-groups`, or `--rhs-limit` for subsets.
The printed totals sum the per-RHS medians; they are not wall times of the
runner. Very small/zero RHS solves are susceptible to timer granularity, so
inspect the nonzero-RHS totals and large-model breakdown as well.

On 2026-10-12 (Apple M2, Rust 1.95.0 release with thin LTO, Apple Clang 17,
native OR-Tools `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`), a complete
one-trial/ten-repetition pass measured 161,608
FTRAN RHS at 0.4134 s native versus 0.5385 s Rust (1.303), and 5,809 BTRAN
RHS at 0.2008 s versus 0.2649 s (1.319). All solution summaries agreed.
Three alternating trials on the four largest contributors gave 37,653 FTRAN
RHS at 0.3605 s versus 0.4684 s (1.300), and 2,435 BTRAN RHS at 0.1871 s
versus 0.2451 s (1.310). Those four models include more than 80% of the
full-corpus native solve time. The detailed CSVs are in `target/` (ignored by
Git). This is isolated sparse LU solve time, not a full simplex or original
call-site representation benchmark.

After the transpose-kernel bounds-check and traversal improvements, a new
one-trial/ten-repetition full-corpus pass measured 0.4157 s native versus
0.4644 s Rust for FTRAN (1.117), and 0.2016 s versus 0.2337 s for BTRAN
(1.159). All 167,417 RHS solution summaries matched. The per-RHS results are
in `target/netlib-solve-probe-full-after-kernels.csv`; this is a single serial
pass, so repeat paired trials for confidence intervals before treating the
small remaining gap as a stable hardware-independent ratio.

The final upper-transpose factor-entry optimization brought the next complete
serial pass to 0.4137 s native versus 0.4658 s Rust for FTRAN (1.126), and
0.2019 s versus 0.1996 s for BTRAN (0.989). Every solution summary still
matched. Per-RHS results are in `target/netlib-solve-probe-full-final.csv`.
The remaining FTRAN gap is concentrated most strongly in `qap15`, `qap12`,
`truss`, and `dfl001`; BTRAN is at parity for this isolated probe workload.
