# Netlib LU factorization probes

The 2,593 `factorization-*.mtx` files in
`../datasets/netlib-simplex-probes/<instance>-<pivot>/` are exact square inputs
to GLOP's LU constructor. The Rust and native harnesses read the same files
into `CompactSparseMatrix` without changing the entry order within columns.
Both use default GLOP parameters and a retained LU object, as the simplex
refactorization path does. Matrix Market parsing, column-vector construction,
and LU object initialization occur **before** the timer. The reported time is
the total time for the requested number of `ComputeFactorization` calls.

Build both release executables against the pinned upstream checkout:

```sh
cargo build --release --example netlib_factorization_benchmark
python3 tools/build_glop_reference_adapter.py \
  --source tools/netlib_factorization_reference_adapter.cc \
  --output target/native/netlib_factorization_reference_adapter
```

Run a full serial comparison. This writes per-matrix median time per
factorization, dimensions, nonzeros, and Rust/native ratio to a CSV in
`target/`, with separate `-raw.csv` and `-models.csv` files for every trial
and per-model summaries. The native and Rust processes run sequentially,
never concurrently; trial order alternates to reduce cache-order bias.

```sh
python3 tools/benchmark_netlib_factorizations.py
```

The defaults are one untimed warm-up, three timed factorizations per trial,
and three trials. Use `--warmups 0 --repetitions 1 --trials 1` to measure a
first construction for every probe; `--limit 20` makes a quick smoke test.
The executable interfaces are also usable directly:

```sh
target/release/examples/netlib_factorization_benchmark 3 1 ../datasets/netlib-simplex-probes/afiro-0/factorization-0000.mtx
target/native/netlib_factorization_reference_adapter 3 1 ../datasets/netlib-simplex-probes/afiro-0/factorization-0000.mtx
```

Each direct output row is tab-separated:
`path, dimension, input-nnz, repetitions, total-factorization-seconds, determinant`.
The Python runner checks matching dimensions, nonzeros, and finite
determinants before writing timings. Timings exclude file I/O, but include
all numerical and symbolic factorization work and internal workspace resets.

## Five-trial comparison, 2026-10-12

On an Apple M2 with Rust 1.95.0 and Apple Clang 17, using release builds and
the pinned OR-Tools commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`:

```sh
python3 tools/benchmark_netlib_factorizations.py \
  --trials 5 --repetitions 1 --warmups 1 \
  --output target/netlib-factorization-5trial.csv
```

All 2,593 matrices passed dimension, nonzero-count, and determinant checks in
every trial. The paired Rust/native aggregate ratios were 1.095, 1.093,
1.093, 1.088, and 1.114; summing per-matrix medians gives 36.720 s native
versus 40.134 s Rust (1.093). `qap15` accounts for 31.451 s native versus
34.939 s Rust (1.111) and more than the entire net gap; without `qap15`,
the aggregate ratio is approximately 0.986. The remaining 96 models after
also excluding `qap12` total about 1.68 s native versus 1.44 s Rust (0.86).
Small-model individual ratios are noisy and should be interpreted by absolute
time. The five-trial run is a controlled isolated-LU comparison, not an
end-to-end simplex timing claim.
