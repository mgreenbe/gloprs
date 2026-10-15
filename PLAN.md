# Development priorities

The core port and performance-parity work are complete. Continue against the
pinned upstream revision in [UPSTREAM.md](UPSTREAM.md); completed porting plans
and experiment reports are preserved in Git history.

- Maintain differential, pivot-trajectory, presolve and original-model KKT gates.
- Improve public API documentation, examples, diagnostics and integration ergonomics.
- Extend performance coverage to larger problems, including `nmm150k-200k`.
  Completed-model aggregate time is the primary practical performance measure;
  retain per-model measurements to identify costs hidden by aggregates.
- Profile concrete performance gaps before changing numerical kernels. Remaining
  follow-up candidates include `fit2d`, `graph20-20-1rand`, `neos-1171737`, and
  the small `CMS750_4` cost from transpose-buffer reuse.
- Evaluate upstream updates explicitly, with regenerated fixtures and a full
  differential pass; do not silently advance the reference.

Keep this file focused on outstanding work. Validation commands belong in
[CONTRIBUTING.md](CONTRIBUTING.md); benchmark outputs belong under `target/`
or `benchmark-results/local/` unless intentionally promoted to a fixture.
