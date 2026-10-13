"""Selection and differential-comparison regressions for external LP corpora."""

import unittest

from lp_corpus import MAX_NONZEROS, models
from test_extra_lp_corpus import check


class CorpusTests(unittest.TestCase):
    def test_manifest_selection_is_strict_and_excludes_netlib(self):
        netlib = {case.name.casefold() for case in models()}
        miplib = models("miplib2017")
        mittelmann = models("mittelmann-lpopt")
        self.assertEqual((len(miplib), len(mittelmann)), (129, 34))
        for case in miplib + mittelmann:
            self.assertLess(case.nonzeros, MAX_NONZEROS)
            self.assertNotIn(case.name.casefold(), netlib)
            self.assertTrue(case.path.is_file(), case.path)
        self.assertEqual(models("extra"), miplib + mittelmann)
        self.assertEqual(len({case.key for case in models("extra")}), 163)

    def test_trace_comparison_rejects_path_and_value_divergence(self):
        base = ["OPTIMAL 2 4607182418800017408 0 0", "1 1", "primal 0",
                "reduced 0", "dual 0", "activity 0",
                "variable_status BASIC", "constraint_status BASIC", "elapsed 0.1"]
        check(base, base.copy())
        changed = base.copy()
        changed[0] = "OPTIMAL 3 4607182418800017408 0 0"
        with self.assertRaises(AssertionError):
            check(base, changed)
        changed = base.copy()
        changed[2] = "primal 4607182418800017408"
        with self.assertRaises(AssertionError):
            check(base, changed)


if __name__ == "__main__":
    unittest.main()
