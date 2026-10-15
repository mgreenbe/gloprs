# Pinned regression fixtures

These files are inputs to maintained tests and native differential tools.
They are not a benchmark archive. Fixtures target the immutable revision in
[UPSTREAM.md](../UPSTREAM.md); never replace them silently to make a test pass.

| Fixtures | Purpose | Regeneration tool under `tools/` |
|---|---|---|
| `netlib-glop.json` | Parsed model fingerprints, native solutions and bases | `generate_netlib_baselines.py` |
| `netlib-dual-trajectories.json.gz` | Fast-96 ordered pivots and terminal state | `generate_netlib_trajectory_fixture.py` |
| `netlib-perturbed-dual.json.gz` | Perturbed dual terminal state | `generate_perturbed_dual_fixture.py` |
| `main-preprocessor-netlib-native.json` | Fast-96 bit-exact presolved models | `generate_main_preprocessor_netlib_fixture.py` |
| `phase5-netlib-solution-native.json.gz` | Public postsolve vectors and bases | `generate_phase5_netlib_solution_fixture.py` |
| `phase7-qap15-terminal-native.json.gz` | Long native qap15 terminal solution | `generate_phase7_qap15_terminal_fixture.py` |
| `qap12-primal.json` | Large primal-path fingerprints | `generate_qap12_primal_fixture.py` |
| `qap15-dual.json`, `qap15-clock-transition.json` | Capped dual path and clock boundary | `generate_qap15_dual_fixture.py` |
| `phase4-{cases,native}.json` | Revised simplex branch regressions | `generate_phase4_fixture.py` |
| `phase5-solve-{cases,native}.json` | Pipeline solve regressions | `generate_phase5_solve_fixture.py` |
| `preprocessor-{cases,native}.json` | Transformation regressions | `generate_preprocessor_fixture.py` |
| `preprocessor-basic-{cases,native}.json` | Basic preprocessing regressions | `generate_preprocessor_basic_fixture.py` |
| `extra-lp-corpus.json` | Deduplicated, parser-verified external corpus selection | `generate_extra_lp_corpus.py` |
| `phase8-extra-trajectories.json.gz` | Selected external-model pivot paths | `generate_phase8_extra_trajectories.py` |
| `extra-native-outcomes.json` | Completed native external-corpus statuses, iterations and objective bits | See below |

Generators require shared datasets and the appropriate pinned native adapters.
Inspect each tool's usage and diagnostic-build requirements before regenerating;
some trajectory generators need a temporary instrumented native build.

`extra-native-outcomes.json` retains only native outcomes from the validated
external-corpus campaign; it is consumed by `validate_extra_lp_baseline.py`.
To refresh it, run `benchmark_solver.py --corpus extra` against the intended
native revision, retain only nonlimited native outcomes, and review changed
checksums, statuses, iteration counts and objective bits. Timing samples and
Rust outcomes do not belong in this fixture.

Phase labels in fixture filenames remain for compatibility with tests and
generators. Historical porting plans, resolved-issue narratives and measurement
campaigns are available in Git through `69d5ba7`. New exploratory output goes
to `target/` or `benchmark-results/local/`; see
[CONTRIBUTING.md](../CONTRIBUTING.md) for current validation commands.
