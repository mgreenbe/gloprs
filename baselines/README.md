# Reference baselines

`netlib-glop.json` contains one normalized result for every problem in the
shared `../datasets/netlib/manifest.json`, produced by the pinned native GLOP
build at OR-Tools commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`.

Regenerate it with:

```text
python3 tools/generate_netlib_baselines.py
```

Each result is tied to the expanded MPS checksum and records status, objective,
iterations, feasibility residuals, deterministic time, elapsed times, and peak
resident memory. Each result also contains the final status of every structural
variable and constraint slack in `basis.variables` and `basis.constraints`.
The `model` object records parsed dimensions, matrix nonzeros, and a stable
fingerprint covering all bounds and objective data.
Wall-clock and memory measurements describe the platform recorded in the file
and are expected to change on another machine. Published Netlib objectives are
retained for comparison; differences are not silently treated as failures
because the Netlib documentation records solver- and tolerance-dependent
alternatives for several instances.
