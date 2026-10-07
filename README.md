# gloprs

`gloprs` is a faithful Rust port of GLOP, the simplex-based linear-programming
solver in Google OR-Tools.

The project is in its initial porting phase. See [PLAN.md](PLAN.md) for the
roadmap and [AGENTS.md](AGENTS.md) for translation and validation policy.

## Workspace

- `lp_data`: linear-program models and sparse data structures;
- `glop`: numerical kernels and the GLOP solver;
- `cli`: command-line tools for solving and differential testing.

## Native GLOP reference

With the pinned OR-Tools checkout and native build described in
[UPSTREAM.md](UPSTREAM.md), build the basis-aware reference adapter and solve an
MPS model with normalized JSON output:

```text
python3 tools/build_glop_reference_adapter.py
python3 tools/run_glop_reference.py --summary MODEL.mps
```

## License

Apache-2.0. This project includes translated and derived work from Google
OR-Tools; see [NOTICE](NOTICE).
