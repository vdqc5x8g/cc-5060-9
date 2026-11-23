import unittest

import numpy as np

from nmfsep.metrics import (
    best_permutation_sdr,
    projection_leakage,
    sdr,
    si_sdr,
    spectral_convergence,
)


class MetricTests(unittest.TestCase):
    def test_sdr_perfect_is_large(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000)
        self.assertGreater(sdr(x, x.copy()), 100)
        self.assertGreater(si_sdr(x, x.copy()), 100)

    def test_sdr_scaling(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000)
        self.assertGreater(sdr(x, 3.0 * x), 100)

    def test_sdr_wrong_is_low(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(5000)
        y = rng.standard_normal(5000)
        self.assertLess(sdr(x, y), 5)

    def test_leakage_diagonal_dominates(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal(4000)
        b = rng.standard_normal(4000)
        est = [a + 0.01 * b, b]
        L = projection_leakage([a, b], est)
        self.assertEqual(L.shape, (2, 2))
        self.assertGreater(L[0, 0], L[0, 1])
        self.assertGreater(L[1, 1], L[1, 0])

    def test_best_permutation_finds_swap(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(4000)
        b = rng.standard_normal(4000)
        perm, score = best_permutation_sdr([a, b], [b, a])
        self.assertEqual(perm, [1, 0])
        self.assertGreater(score, 50)

    def test_spectral_convergence_zero_for_identical(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(8000)
        self.assertLess(spectral_convergence(x, x.copy()), 1e-10)

    def test_unmatched_estimates_are_not_scored(self):
        from nmfsep.pipeline import evaluate_stems
        rng = np.random.default_rng(5)
        refs = [rng.standard_normal(4000), rng.standard_normal(4000)]
        ests = [refs[0].copy(), refs[1].copy(), 0.01 * rng.standard_normal(4000)]
        ev = evaluate_stems(ests, refs, ["a", "b"], sample_rate=8000,
                            n_fft=512, hop_length=128)
        self.assertEqual(ev["n_matched"], 2)
        self.assertEqual(ev["n_estimates"], 3)
        unmatched = ev["per_estimate"][2]
        self.assertIsNone(unmatched["matched_reference"])
        self.assertIsNone(unmatched["sdr_db"])
        self.assertIsNone(unmatched["si_sdr_db"])
        self.assertIsNone(unmatched["spectral_convergence"])
        # mean is over the two matched pairs only
        self.assertGreater(ev["mean_sdr_db"], 40)


if __name__ == "__main__":
    unittest.main()
