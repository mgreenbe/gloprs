# Phase 7 known differential limitations

- **P7-001 — resolved: qap15 late terminal trajectory divergence.** The
  former Rust solve took 73,789 pivots and 521 refactorizations versus native
  GLOP's 74,317 and 527. Live reduced-cost arrays first differed at pivot
  7,505, even though capped solves could clean up and reconverge. The initial
  mismatch was in `TestEnteringReducedCostPrecision()`: GLOP uses a persistent,
  basis-row-ordered basic-objective vector, and Apple Clang contracts its dense
  dot product in eight four-term blocks. Rust gathered objective entries through
  the basis permutation and accumulated them in a different floating-point
  order. Matching the native dense reduction and maintaining the basic-objective
  workspace (including lazy invalidation on basis permutation) restores the
  exact 7,506 pricing heap and eliminates the extra RNG draw at 65,430. The
  full presolved terminal fingerprints now agree bit-for-bit: `OPTIMAL`, 74,317
  pivots, 527 refactorizations, ordered basis, variable values, reduced costs,
  objective, and deterministic clock. The opt-in
  `qap15_presolved_trajectory` test pins the early price heap, later RNG state,
  and terminal state against the pinned native checkout. The native terminal
  solution remains in `phase7-qap15-terminal-native.json.gz`.

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
