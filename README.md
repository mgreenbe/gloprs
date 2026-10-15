# gloprs

A Rust implementation of GLOP, the simplex linear-programming solver in
Google OR-Tools. The core port and performance-parity work are complete.
Development now focuses on maintaining numerical fidelity, improving performance,
and making the solver easier to use.

## Use the solver

```text
cargo run --release -p gloprs-glop --example public_lp
cargo run --release -p gloprs-glop --example public_mps -- MODEL.mps
cargo run --release -p gloprs-cli -- solve --default MODEL.mps
```

For the Rust API, build a `lp_data::lp_data::LinearProgram`, call `clean_up()`,
and pass it to `glop::lp_solver::LPSolver::solve_result()`. The result borrows
solution vectors, statuses, statistics, and certificates from the solver.
Check its status before using numerical values. `export_basis()` and
`import_basis()` support warm starts; disable preprocessing when importing a
basis, as native GLOP recommends.

The CLI also supports `solve --dual MODEL.mps`, raw simplex with
`solve MODEL.mps`, and model fingerprints with `inspect --tsv MODEL.mps`.
MIP models are solved as LP relaxations; integrality is not enforced.
Unsupported nondefault parameters are rejected with a named diagnostic.

## Workspace

- `lp_data/`: models, sparse structures, MPS parsing and transformations.
- `glop/`: numerical kernels, preprocessing and public solver API.
- `cli/`: command-line solver and model inspection.
- `tools/`: native reference adapters, regression validation and benchmarks.
- `baselines/`: pinned correctness fixtures, not an experiment archive.

## Development

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
python3 -m unittest discover -s tools -p 'test_*.py'
```

Corpus checks require the shared sibling `../datasets/` directory. Native
comparisons additionally require the pinned OR-Tools checkout and build.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and benchmarking,
[UPSTREAM.md](UPSTREAM.md) for reference setup, [PLAN.md](PLAN.md) for ongoing
priorities, and [AGENTS.md](AGENTS.md) for repository policy.

## License

Apache-2.0. Includes translated and derived work from Google OR-Tools;
see [NOTICE](NOTICE).
