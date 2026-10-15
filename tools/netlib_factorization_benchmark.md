# Netlib LU factorization probes

The 2,593 `factorization-*.mtx` files in
`../datasets/netlib-simplex-probes/<instance>-<pivot>/` are exact square inputs
to GLOP's LU constructor. The Rust and native harnesses read the same files
into `CompactSparseMatrix` without changing the entry order within columns.
Both use default GLOP parameters and a retained LU object, as the simplex
refactorization path does. Matrix Market parsing, column-vector construction,
and LU object initialization occur **before** the timer. The reported time is
the total time for the requested number of `ComputeFactorization` calls.
These probes preserve the numerical basis entries and their within-column
order, but materialize each selected basis as a new contiguous matrix. In a
live solve, LU instead reads a selected-column view of the original compact
problem matrix. Thus the benchmark measures factorization parity on matched
materialized inputs, not necessarily the memory-access cost of live
refactorization.
The direct factorization-only probes additionally record the original compact
matrix and basis selections; use `--selected-source` to benchmark that live
view layout without timing file loading:

```sh
python3 tools/benchmark_netlib_factorizations.py \
  --root target/qap15-direct-probes \
  --selected-source target/qap15-direct-probes/qap15-source.mtx \
  --models qap15 --repetitions 1 --warmups 0 --trials 3 \
  --output target/qap15-selected-view-timings.csv
```

For a model that normally uses presolve, `netlib_simplex_probe` can capture
the selected compact views under the ordinary presolved solve parameters:

```sh
target/release/examples/netlib_simplex_probe \
  ../datasets/miplib2017/mps/fhnw-binpack4-48.mps 120 \
  target/fhnw-selected --presolved-factorizations
```

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
Use `--models qap15 --pivot-min 20000 --pivot-max 40000` to isolate saved
bases from a particular simplex interval (inclusive endpoints).
For live-lifecycle comparisons, `--retain-factorization --chronological`
reuses one LU object per solver and processes saved selections in numeric
pivot order. `--pivot-threshold 0.9` reproduces the high threshold that the
direct qap15 solve reaches after numerical imprecision; the default probe
threshold is 0.01. Choose the threshold to match the solver path being
investigated. `--evict-mib 32` evicts cache outside the timed region for a
cache-sensitivity check.
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
