"""Selection and differential-comparison regressions for external LP corpora."""

import unittest

from lp_corpus import MAX_NONZEROS, NMM_CORPORA, models
from test_extra_lp_corpus import check


class CorpusTests(unittest.TestCase):
    def test_nmm_bins_are_disjoint_and_complete(self):
        expected_counts = (213, 25, 20, 16)
        seen = set()
        for corpus, count, lower in zip(NMM_CORPORA, expected_counts,
                                        (0, 50_000, 100_000, 150_000), strict=True):
            selected = models(corpus)
            self.assertEqual(len(selected), count)
            for case in selected:
                self.assertLessEqual(lower, case.nonzeros)
                self.assertLess(case.nonzeros, lower + 50_000)
                self.assertNotIn(case.key, seen)
                self.assertTrue(case.path.is_file(), case.path)
                seen.add(case.key)

    def test_manifest_selection_is_strict_and_excludes_netlib(self):
        netlib = {case.name.casefold() for case in models()}
        miplib = models("miplib2017")
        mittelmann = models("mittelmann-lpopt")
        self.assertEqual((len(miplib), len(mittelmann)), (130, 36))
        for case in miplib + mittelmann:
            self.assertLess(case.nonzeros, MAX_NONZEROS)
            self.assertNotIn(case.name.casefold(), netlib)
            self.assertTrue(case.path.is_file(), case.path)
        self.assertEqual(models("extra"), miplib + mittelmann)
        self.assertEqual(len({case.key for case in models("extra")}), 166)
        self.assertEqual(
            {case.key for case in models("extra") if case.name in
             ("rmatr200-p5", "fome12", "nug08-3rd")},
            {"miplib2017/rmatr200-p5", "mittelmann-lpopt/fome12",
             "mittelmann-lpopt/nug08-3rd"},
        )

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
