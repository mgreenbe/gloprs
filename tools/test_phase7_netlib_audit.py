"""Focused classification tests for the full-Netlib differential harness."""

import unittest
import gzip
import json
from pathlib import Path

from phase7_netlib_audit import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = [
            "OPTIMAL 2 4617315517961601024 0 0",
            "2 1 0 0 0 0 0 0",
            "primal 4617315517961601024 0",
            "reduced 0 0",
            "dual 0",
            "activity 4617315517961601024",
            "variable_status BASIC AT_LOWER_BOUND",
            "constraint_status AT_LOWER_BOUND",
        ]

    def test_identical(self) -> None:
        issues, metrics = compare(self.trace, self.trace)
        self.assertEqual(issues, [])
        self.assertEqual(metrics["native_iterations"], 2)

    def test_status_and_path_are_separate(self) -> None:
        changed = self.trace.copy()
        changed[0] = changed[0].replace("OPTIMAL 2", "IMPRECISE 3")
        issues, _ = compare(self.trace, changed)
        self.assertIn("incorrect_status", issues)
        self.assertIn("pivot_path_or_termination", issues)

    def test_vector_and_basis_are_separate(self) -> None:
        changed = self.trace.copy()
        changed[2] = "primal 0 0"
        changed[6] = "variable_status AT_LOWER_BOUND BASIC"
        issues, _ = compare(self.trace, changed)
        self.assertIn("primal", issues)
        self.assertIn("variable_basis", issues)
        self.assertNotIn("incorrect_status", issues)

    def test_checked_in_report_covers_full_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        names = json.loads((root.parent / "datasets/netlib/subsets/all.json").read_text())
        report = json.loads((root / "baselines/phase7-netlib-report.json").read_text())
        self.assertEqual(report["upstream_commit"], "100f66e6242ab8bf8d32feb8f3bf086db66ae2b5")
        self.assertEqual([case["name"] for case in report["results"]], names)
        self.assertEqual(report["summary"]["presolve_models_checked"], 98)
        exceptional = {case["name"]: case["known_issue"] for case in report["results"]
                       if case["classifications"] != ["match"]}
        self.assertEqual(exceptional, {"qap15": "P7-001"})
        self.assertEqual(report["summary"]["classifications"],
                         {"match": 97, "deterministic_time": 1, "dual": 1,
                          "pivot_path_or_termination": 1, "primal": 1,
                          "reduced_costs": 1, "refactorization_count": 1,
                          "variable_basis": 1})
        for case in report["results"]:
            self.assertIn("native_refactorizations", case)
            self.assertIn("rust_refactorizations", case)
            if case["name"] != "qap15":
                self.assertEqual(case["native_refactorizations"],
                                 case["rust_refactorizations"], case["name"])
        qap15 = next(case for case in report["results"] if case["name"] == "qap15")
        native_fixture = json.loads(gzip.decompress(
            (root / "baselines/phase7-qap15-terminal-native.json.gz").read_bytes()))
        self.assertEqual(native_fixture["upstream_commit"], report["upstream_commit"])
        self.assertEqual(native_fixture["model_sha256"], qap15["mps_sha256"])
        self.assertEqual(qap15["native_status"], "OPTIMAL")
        self.assertEqual(qap15["rust_status"], "OPTIMAL")
        self.assertEqual(qap15["native_iterations"], 74317)
        self.assertEqual(qap15["rust_iterations"], 73789)
        self.assertEqual(qap15["native_refactorizations"], 527)
        self.assertEqual(qap15["rust_refactorizations"], 521)
        self.assertLess(qap15["independent_primal_violation"], 1e-6)
        self.assertLess(qap15["maximum_dual_infeasibility"], 1e-6)
        self.assertEqual(qap15["deterministic_prefix_iterations"], 5000)
        self.assertEqual(
            qap15["deterministic_prefix_fingerprint"],
            "fingerprint INIT 5000 16686786820853128222 18085048207775952367 9781289146198669845 4627381663376136883 4626660475175328932 47",
        )
        self.assertEqual(
            qap15["presolved_sha256"],
            "c88f74fc92f114ffd23709e4bab3ca8b1f361bbe2d6c534bca3f549d6b05b8b4",
        )


if __name__ == "__main__":
    unittest.main()
