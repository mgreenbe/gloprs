# Repository policy

## Scope

`gloprs` is a Rust implementation of GLOP with a completed core port. The
canonical repository is <https://github.com/mgreenbe/gloprs.git>. This is an
independent Git repository; sibling `../or-tools` and `../datasets` are shared
reference infrastructure and must not be modified as part of solver work.

Read [PLAN.md](PLAN.md) before substantial work. Keep it focused on outstanding
priorities. Use [CONTRIBUTING.md](CONTRIBUTING.md) for repeatable workflows and
[UPSTREAM.md](UPSTREAM.md) for the pinned source revision. Historical porting
ledgers and completed investigations belong in Git history.

## Fidelity and implementation

- Preserve upstream algorithms, sparse representations, asymptotic complexity,
  operation order, tie breaking, numerical safeguards and workspace lifetimes.
- Compare the complete relevant upstream implementation and associated tests
  before changing numerical behavior. Keep upstream names and module boundaries
  recognizable; retain attribution and comments explaining mathematical invariants.
- Prefer contiguous storage, slices and reusable workspaces. Avoid allocation
  inside simplex iterations unless upstream does comparable work.
- Keep row/column index newtypes distinct. Preserve sparse traversal order and
  deterministic behavior; do not introduce uncontrolled hashing in hot paths.
- Use stable Rust. Keep unsafe operations narrowly scoped, with a `// SAFETY:`
  explanation at each block and tests for the invariants that make it sound.
- Replace unavailable upstream infrastructure with focused Rust equivalents.
  Reject unsupported parameters rather than silently changing their semantics.
- Document deliberate upstream divergences where maintained, with measured
  consequences and validation evidence. Keep Apache-2.0 notices intact.

## Validation

Run the applicable checks in CONTRIBUTING.md. Numerical changes require
release-mode native differential evidence, including solver status, objective,
feasibility, basis, reduced costs and iteration/pivot behavior where affected.
Investigate path differences even when the final objective agrees. Keep
minimal regression cases for discovered discrepancies.

Performance changes require a correct baseline, profiles, equivalent release
builds, alternating before/after timings and raw trial distributions. Report
aggregate performance and per-model costs, with binary fingerprints and machine
metadata. Run the full `nmm0-50k` benchmark before retaining solver performance
changes. Never claim parity through weaker tolerances or termination checks.

## Change discipline

Preserve unrelated user changes. Keep changes reviewable and focused. Update
maintained docs when workflows or behavior change, without appending an
experiment diary. Retain correctness fixtures and their regeneration tools;
keep temporary profiles and benchmark output out of source control. State what
was tested and what remains unverified in each handoff.

Datasets stay under shared `../datasets`, with manifest-backed provenance,
checksums and deterministic selection. Never silently replace an instance or
mix package-specific measurement output into the shared corpus.
