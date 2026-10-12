# Phase 7 known differential limitations

- **P7-001 — qap15 late terminal trajectory divergence.** The native terminal
  solve is pinned in `phase7-qap15-terminal-native.json.gz`. With matching
  default parameters, both solvers reach `OPTIMAL`, but native takes 74,317
  pivots (527 refactorizations) and Rust takes 73,789 (521 refactorizations),
  with different terminal basis and solution bits. The full Phase 7 audit
  classifies each difference and checks the original-model primal violations;
  the opt-in terminal KKT test checks the Rust solution independently. The
  presolved model is bit-exact, and the fixed-cap fingerprints agree at
  67,016 iterations but differ at 67,017. Run
  `python3 tools/validate_phase7_qap15_boundary.py` to reproduce that narrow
  capped-state comparison. An iteration cap runs cleanup and reoptimization,
  so this is **not** evidence that live pivot 67,017 is the first divergent
  pivot; earlier capped snapshots even diverge and subsequently reconverge.
  The in-flight first divergence and its numerical cause remain open fidelity
  work. A 20-second wall-limited partial solve is not used as a trajectory
  comparator.

- **P7-002 — resolved: d2q06c deterministic-work drift.** Before the fix, the
  terminal Rust deterministic time exceeded native by about `4.04e-8` seconds.
  On the identical, bit-exact presolved model the first difference was at
  pivot 32, entirely in basis deterministic time (`-5.767925561e-9` Rust minus native), with
  the same number of basis updates. The native cached primal direction-left
  inverse includes a nonzero at row 1275 that Rust omits at that prefix, so
  this was a real intermediate sparse-support fidelity defect, not merely a
  clock-rounding difference. Rust's scattered-direction preparation marked
  temporary membership-mask bits that upstream leaves clear. Removing those
  bits restores the sparse support and deterministic clock at pivot 32 and at
  terminal solve. `tools/validate_phase7_clock_prefix.py` now requires exact
  equality of the formerly divergent prefix and all three transpose-solve
  stages on its predecessor.

- **P7-003 — resolved: sierra deterministic-work drift.** Before the fix, the
  terminal Rust deterministic time was about `5.728096677e-9` seconds below
  native. On the identical presolved model the clocks first differed at pivot
  229, entirely in basis deterministic time, with equal basis
  update counts. The cached direction-left inverse also differs in support at
  that prefix (including row 303). The same prefix fixture above pins the
  discrepancy; the same sparse-mask correction restores exact clock agreement
  at pivot 229 and at terminal solve.
