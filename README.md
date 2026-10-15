# gloprs

A Rust implementation of GLOP, the simplex linear-programming solver in
Google OR-Tools. The core port and performance-parity work are complete.
Development now focuses on maintaining numerical fidelity, improving performance,
and making the solver easier to use.

## Use the solver

Install the `gloprs` executable on your PATH:

```sh
cargo install --path .
```

With Netlib's `afiro.mps` in the current directory, solve it with:

```sh
gloprs afiro.mps
```

```text
File        : 'afiro.mps'
Dimension   : 27 x 32
Status      : OPTIMAL
Objective   : -4.647531428571430e+02
Iterations  : 5
Primal error: 2.114974861910923e-14
Dual error  : 5.551115123125783e-17
```

Inspect the model's statistics without solving it:

```sh
gloprs inspect afiro.mps
```

```text
Number of rows                               : 27
Number of variables in file                  : 32
Number of entries (non-zeros)                : 83
Number of entries in the objective           : 5
Number of entries in the right-hand side     : 7
Number of <= constraints                     : 19
Number of >= constraints                     : 0
Number of = constraints                      : 8
Number of range constraints                  : 0
Number of non-negative variables             : 32
Number of boxed variables                    : 0
Number of free variables                     : 0
Number of fixed variables                    : 0
Number of other variables                    : 0
Number of integer variables                  : 0
Number of binary variables                   : 0
Number of non-binary integer variables       : 0
Number of continuous variables               : 32
```

These outputs were captured from actual release-build runs on Netlib's AFIRO
instance. Floating-point output and iteration counts may vary with platform or
solver version. The model is not bundled with this repository; use your own copy
of `afiro.mps`, or substitute its full path in the commands.

To run without installing, place `afiro.mps` in the repository root and run:

```sh
cargo run --release -- afiro.mps
cargo run --release -- inspect afiro.mps
```

Or use `cargo build --release` and run `./target/release/gloprs` directly.
API examples are available with `cargo run --release --example public_lp`
and `cargo run --release --example public_mps -- afiro.mps`.

The Rust API is exported directly from the package root:

```rust
use gloprs::{LinearProgram, LPSolver, ProblemStatus, parse_mps_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model: LinearProgram = parse_mps_file("afiro.mps")?;
    let mut solver = LPSolver::new();
    let result = solver.solve(&model);
    if result.status == ProblemStatus::Optimal {
        println!("Objective: {}", result.objective_value);
    }
    Ok(())
}
```

See [API.md](API.md) for the public types, configuration, result lifetimes,
error handling, and warm starts. Full Rust documentation is available with
`cargo doc --no-deps --open`.

Solving uses the default preprocessing and scaling pipeline. Use
`gloprs --dual afiro.mps` to select dual simplex, or `gloprs --raw afiro.mps`
to disable preprocessing and scaling (`--raw --dual` combines both). The former
`--default` flag and `solve` subcommand remain accepted for compatibility.
Use `gloprs --machine afiro.mps` for the compact machine-readable solve line,
and `gloprs inspect --tsv afiro.mps` for tab-separated model fingerprints.
MIP models are solved as LP relaxations; integrality is not enforced.
Unsupported nondefault parameters are rejected with a named diagnostic.

## Workspace

- `lp_data/`: models, sparse structures, MPS parsing and transformations.
- `src/`: the `gloprs` library and executable, numerical kernels and preprocessing.
- `examples/`, `tests/`, `benches/`: API examples, regressions and benchmarks.
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
