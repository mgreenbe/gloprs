# Development and validation

## Routine checks

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/validate_parameter_defaults.py
```

Python corpus tests require the shared sibling `../datasets` manifests and MPS
files. Cargo tests that need external datasets or long solves are opt-in.

## Differential regression gates

Set up the pinned native build using [UPSTREAM.md](UPSTREAM.md), then build
adapters and Rust diagnostics:

```text
python3 tools/build_glop_reference_adapter.py --source tools/phase5_netlib_solution_reference_adapter.cc --output target/native/phase5_netlib_solution_reference_adapter
python3 tools/build_glop_reference_adapter.py --source tools/full_presolve_netlib_reference_adapter.cc --output target/native/full_presolve_netlib_reference_adapter
cargo build --release
cargo build --release --example phase5_netlib_solution_trace --example full_presolve_netlib
python3 tools/audit_netlib.py
```

The audit checks input checksums, parsed models, presolve states, public
solutions, bases, refactorizations and deterministic work. It writes to
`target/netlib-audit.json` and fails on mismatches; wall-limited solves are
inconclusive. Use `--models afiro ...` for a focused run. The full run includes
long qap solves. Strict pivot and original-model feasibility checks include:

```text
cargo test --release -p gloprs --test netlib_trajectories -- --ignored
cargo test --release -p gloprs --test phase8_extra_trajectories -- --ignored
cargo test --release -p gloprs --test main_preprocessor_netlib -- --ignored
cargo test --release -p gloprs --test phase5_netlib_solution fast_96_public_postsolve_matches_native -- --ignored
cargo test --release -p gloprs --test qap12_primal -- --ignored
cargo test --release -p gloprs --test qap15_presolved_trajectory -- --ignored
python3 tools/validate_phase8_nmm_presolved_checkpoints.py
```

Component `validate_*.py` tools compare focused Rust trace examples with native
reference adapters. Keep their fixtures and generators: phase numbers in some
fixture, example and test names are stable identifiers, not unfinished work.
See [baselines/README.md](baselines/README.md) for the fixture catalog.

## Performance measurements

Use release builds under quiet conditions. On macOS, inspect
`./benchmarkmode.sh --status` and verify live worker/index/sleep state before
measurements. The script's `--on --dry-run` previews preparation; `--on` applies
it and `--off` restores recorded settings. Do not disable a mode that was
already enabled by the user. The journal alone does not prove the live state.

```text
python3 tools/benchmark_solver.py --corpus nmm0-50k --trials 5 --warmups 1 --wall-seconds 60 --output target/solver-benchmark.json
python3 tools/benchmark_stages.py --corpus nmm100k-150k --models neos-950242 --stage all --trials 21 --warmups 2 --output target/stage-benchmark.json
```

Both harnesses accept `--native` and `--rust` paths to saved release binaries
for controlled before/after tests. Public solve timing includes presolve,
simplex and postsolve, excluding parsing/output. Stage timing separates parsing,
presolve and matched presolved solves. Wall-limited cases are excluded from
performance aggregates. Keep raw trials, outcome checks, compiler/machine data
and binary hashes with each local report. Never compare an old executable
against edited source accidentally.

Available selections are `netlib-fast96`, `extra`, `miplib2017`,
`mittelmann-lpopt`, `nmm0-50k`, `nmm50k-100k`, `nmm100k-150k`, and
`nmm150k-200k`. NMM bins use manifest-verified matrix nonzeros; MIP instances
are LP relaxations. Use `--models corpus/name ...` to disambiguate names.

`benchmark_memory.py` measures macOS peak memory, `profile_stages.py` takes
serial native/Rust stack samples, and `benchmark_kernel_ab.py` compares saved
kernel binaries. Other `benchmark_*.py` tools isolate individual kernels;
`cargo bench -p gloprs --bench basis_kernels` runs the Rust kernel suite.
See [factorization probes](tools/netlib_factorization_benchmark.md) and
[simplex probes](tools/netlib_simplex_probes.md) for captured numerical inputs.

Run the full `nmm0-50k` before retaining solver performance changes. Use total
completed-model time to assess practical benefit, while disclosing repeated
per-model costs. Profiles, abandoned experiments and trial output belong under
`target/` or `benchmark-results/local/`. Past parity evidence and measurements
are preserved in Git through commit `69d5ba7`; old timings are not promises
about a different compiler or machine.

CLI output is compared byte-for-byte with native-captured fixtures in
`tests/fixtures/cli`. Run `cargo test -p gloprs --test cli_output`; use
`-- --include-ignored` to include afiro with the shared dataset present.
Fixture provenance and native regeneration commands are documented in
[tests/fixtures/cli/README.md](tests/fixtures/cli/README.md).
