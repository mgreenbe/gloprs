#!/usr/bin/env python3
"""Compare all pinned GLOP parameter-validation branches."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target/native/parameters_reference_adapter"
RUST = ROOT / "target/debug/examples/parameters_trace"

FINITE = [
    "degenerate_ministep_factor", "drop_tolerance", "dual_feasibility_tolerance",
    "dual_small_pivot_threshold", "dualizer_threshold", "harris_tolerance_ratio",
    "lu_factorization_pivot_threshold", "markowitz_singularity_threshold",
    "max_number_of_reoptimizations", "minimum_acceptable_pivot",
    "preprocessor_zero_tolerance", "primal_feasibility_tolerance",
    "ratio_test_zero_threshold", "recompute_edges_norm_threshold",
    "recompute_reduced_costs_threshold", "refactorization_threshold",
    "relative_cost_perturbation", "relative_max_cost_perturbation",
    "small_pivot_threshold", "solution_feasibility_tolerance",
    "max_valid_magnitude", "drop_magnitude",
]
NOT_NAN = ["objective_lower_limit", "objective_upper_limit"]
NONNEGATIVE = ["crossover_bound_snapping_distance", "initial_condition_number_threshold", "max_deterministic_time", "max_time_in_seconds"]
INTEGERS = ["basis_refactorization_period", "devex_weights_reset_period", "num_omp_threads", "random_seed"]

def run(exe: Path, case: str) -> str:
    return subprocess.run([exe], input=case, text=True, capture_output=True, check=True).stdout

def main() -> None:
    cases = [(name, value) for name in FINITE for value in ("nan", "inf", "-1", "0")]
    cases += [(name, value) for name in NOT_NAN for value in ("nan", "inf", "-inf", "0")]
    cases += [(name, value) for name in NONNEGATIVE for value in ("nan", "inf", "-1", "0")]
    cases += [(name, value) for name in INTEGERS for value in ("-1", "0")]
    cases += [("markowitz_zlatev_parameter", value) for value in ("0", "1")]
    cases += [("max_valid_magnitude", "1.1e100"), ("drop_magnitude", "1e-101")]
    for field, value in cases:
        case = f"{field} {value}\n"
        expected, actual = run(NATIVE, case), run(RUST, case)
        if expected != actual:
            raise AssertionError(f"{field}={value}: {expected!r} != {actual!r}")
    print(f"{len(cases)} parameter-validation cases agree")

if __name__ == "__main__": main()
