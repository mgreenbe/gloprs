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
