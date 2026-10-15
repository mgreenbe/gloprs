"""Focused classification tests for the full-Netlib differential harness."""

import unittest
import gzip
import json
from pathlib import Path

from audit_netlib import compare


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



    def test_native_fixtures_cover_full_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root.parent / "datasets/netlib/manifest.json").read_text())
        names = {case["name"] for case in manifest["problems"]}
        model_fixture = json.loads((root / "baselines/netlib-glop.json").read_text())
        self.assertEqual({case["name"] for case in model_fixture["results"]}, names)
        fast = json.loads(gzip.decompress(
            (root / "baselines/phase5-netlib-solution-native.json.gz").read_bytes()))
        self.assertEqual({case["name"] for case in fast["results"]} | {"qap12", "qap15"}, names)
        terminal = json.loads(gzip.decompress(
            (root / "baselines/phase7-qap15-terminal-native.json.gz").read_bytes()))
        qap15 = next(case for case in manifest["problems"] if case["name"] == "qap15")
        self.assertEqual(terminal["model_sha256"], qap15["mps_sha256"])
        self.assertEqual(terminal["upstream_commit"], model_fixture["upstream_commit"])
        self.assertEqual(fast["upstream_commit"], model_fixture["upstream_commit"])


if __name__ == "__main__":
    unittest.main()
