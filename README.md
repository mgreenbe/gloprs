# gloprs

`gloprs` is a faithful Rust port of GLOP, the simplex-based linear-programming
solver in Google OR-Tools.

See [PLAN.md](PLAN.md) for the roadmap and [AGENTS.md](AGENTS.md) for
translation and validation policy.

## Workspace

- `lp_data`: linear-program models and sparse data structures;
- `glop`: numerical kernels and the GLOP solver;
- `cli`: command-line tools for solving and differential testing.

## Public API

Construct a `lp_data::lp_data::LinearProgram`, call `clean_up()`, and pass it
to `glop::lp_solver::LPSolver::solve_result()`. The returned result borrows
the solver's primal and dual vectors, statuses, statistics, and certificates;
inspect its status before using the numerical values. `export_basis()` and
`import_basis()` provide an owned basis for warm starts. Disable preprocessing
when importing a basis, as native GLOP recommends.

Runnable examples:

```text
cargo run -p gloprs-glop --example public_lp
cargo run -p gloprs-glop --example public_mps -- MODEL.mps
```

The command-line solver accepts `gloprs solve MODEL.mps` (historical raw
simplex mode), `gloprs solve --dual MODEL.mps`, and
`gloprs solve --default MODEL.mps` (the full default preprocessing pipeline).
For a stable model fingerprint, use `gloprs inspect --tsv MODEL.mps`.

Check the Rust parameter schema and defaults against the pinned upstream proto
with `python3 tools/validate_parameter_defaults.py`. Non-default parameters
without a faithful implementation are rejected with a named diagnostic.

## Full Netlib differential audit

With the shared `../datasets/netlib` corpus and pinned native build present:

```text
python3 tools/build_glop_reference_adapter.py --source tools/phase5_netlib_solution_reference_adapter.cc --output target/native/phase5_netlib_solution_reference_adapter
python3 tools/build_glop_reference_adapter.py --source tools/full_presolve_netlib_reference_adapter.cc --output target/native/full_presolve_netlib_reference_adapter
cargo build --release -p gloprs-glop --example phase5_netlib_solution_trace --example full_presolve_netlib
cargo build --release -p gloprs-cli
python3 tools/phase7_netlib_audit.py
```

The audit checks all 98 model checksums and parse statistics, compares all
98 presolve states (96 pinned and two live), and compares native/Rust public solve results with
per-model classifications. Its checked-in output is
`baselines/phase7-netlib-report.json`; known limitations are tracked in
`baselines/phase7-known-issues.md`. The `qap15` native terminal solve is pinned
in `baselines/phase7-qap15-terminal-native.json.gz`; the audit's Rust run can
take around two minutes, and its terminal path differs from native despite
both solutions being optimal and feasible. Run the independent original-model KKT
gate with `cargo test --release -p gloprs-glop --test phase5_netlib_solution
fast_96_public_postsolve_matches_native -- --ignored`; the opt-in `qap15` KKT
test is `qap15_terminal_solution_satisfies_original_model_kkt`. A narrower
opt-in capped-state regression is
`python3 tools/validate_phase7_qap15_boundary.py`.

## Additional LP corpora

The same native adapters and release Rust examples support the shared
`../datasets/miplib2017` and `../datasets/mittelmann-lpopt` MPS collections.
Their parser-verified `extra` selection contains 166 models with fewer than
150,000 constraint-matrix nonzeros, excluding names already in Netlib. The
shared manifests also count objective coefficients, so the pinned selection
is in `baselines/extra-lp-corpus.json`; regenerate it with
`python3 tools/generate_extra_lp_corpus.py`. MIPLIB instances
are solved as LP relaxations; GLOP does not enforce integrality.

```text
python3 -m unittest discover -s tools -p test_lp_corpus.py
python3 tools/test_extra_lp_corpus.py --corpus extra --wall-seconds 60
python3 tools/validate_extra_lp_baseline.py
python3 tools/benchmark_phase8_stages.py --corpus extra --stage all --wall-seconds 60 --output baselines/phase8-extra-stages.json
python3 tools/benchmark_phase5_netlib.py --corpus extra --mode default --wall-seconds 60 --output baselines/phase8-extra-public.json
```

Use `--corpus miplib2017` or `--corpus mittelmann-lpopt` for one source, and
`--models corpus/name ...` to select individual cases. The differential gate
compares native and Rust solution vectors and basis statuses in addition to
status, objective, and iteration count. A stage solve is skipped when presolve
already terminates the model; public-solver validation still covers it. Models
that hit the chosen wall limit are reported as inconclusive and excluded
from parity/timing aggregates, because their pivot count depends on elapsed
time. `validate_extra_lp_baseline.py` is a faster Rust-only regression gate
against the pinned native status, iteration, and objective outcomes. The
measured results, dispersion, and build fingerprints are in
[the extra-corpus performance baseline](baselines/phase8-extra-performance.md).

## Native GLOP reference

With the pinned OR-Tools checkout and native build described in
[UPSTREAM.md](UPSTREAM.md), build the basis-aware reference adapter and solve an
MPS model with normalized JSON output:

```text
python3 tools/build_glop_reference_adapter.py
python3 tools/run_glop_reference.py --summary MODEL.mps
```

Validate Rust MPS parsing and model construction against native GLOP across the
shared Netlib corpus with:

```text
cargo build -p gloprs-cli
python3 tools/validate_netlib_parse.py
```

Run the Phase-2 numerical-kernel benchmark with:

```text
cargo bench -p gloprs-glop --bench basis_kernels
```

## License

Apache-2.0. This project includes translated and derived work from Google
OR-Tools; see [NOTICE](NOTICE).
