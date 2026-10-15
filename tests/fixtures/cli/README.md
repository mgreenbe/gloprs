# CLI formatting fixtures

`model.mps` is a synthetic two-variable LP used by self-contained CLI tests.
`afiro.*.txt` captures the shared Netlib instance identified by SHA-256 in
`native.json`; the MPS itself remains in the shared dataset.

The native adapter obtains model statistics and solve results directly from
pinned GLOP. Inspect uses `GetPrettyProblemStats()` verbatim. The solve summary
uses upstream `ortools/linear_solver/solve.cc`'s 12-column labels and `%.15e`
precision, with GLOP's API status names, iteration count and feasibility errors.
It is a selected summary, not a transcript of the generic driver's full logs;
no timings are synthesized or compared.

Regenerate or independently check against the native build:

```text
python3 tools/build_glop_reference_adapter.py --source tools/cli_output_reference_adapter.cc --output target/native/cli_output_reference_adapter
python3 tools/generate_cli_output_fixtures.py
python3 tools/generate_cli_output_fixtures.py --check
cargo test -p gloprs --test cli_output -- --include-ignored
```

The afiro execution test is opt-in because it requires shared external data.
The synthetic model, layout snapshots, and README snapshot check run normally.
