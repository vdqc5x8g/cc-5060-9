import unittest

import numpy as np

from nmfsep.masks import (
    component_spectrograms,
    group_components,
    reconstruct_stems,
)
from nmfsep.stft import istft, stft


def _fixture(seed=0, n=16000, sr=8000):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    a = 0.3 * np.sin(2 * np.pi * 220 * t)
    b = 0.2 * np.sin(2 * np.pi * 660 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    mix = a + b
    W = np.stack([
        np.abs(stft(a, 512, 128)).mean(axis=1),
        np.abs(stft(b, 512, 128)).mean(axis=1),
    ], axis=1)
    W /= W.sum(axis=0, keepdims=True)
    X = stft(mix, 512, 128)
    return mix, X, W


class MaskTests(unittest.TestCase):
    def test_partition_of_unity_and_sample_consistency(self):
        mix, X, W = _fixture()
        H = np.abs(np.linalg.lstsq(W, np.abs(X), rcond=None)[0])
        for policy in ("wiener", "ratio", "binary"):
            ss = reconstruct_stems(
                W, H, X, mix, sample_rate=8000, n_fft=512, hop_length=128,
                policy=policy,
            )
            self.assertLess(ss.masks_sum_error, 1e-10, policy)
            self.assertLess(ss.component_sum_error, 1e-10, policy)
            self.assertEqual(len(ss.stems), 2)
            self.assertEqual(ss.policy, policy)

    def test_residual_policy_exact_sample_sum(self):
        mix, X, W = _fixture()
        H = np.abs(np.linalg.lstsq(W, np.abs(X), rcond=None)[0])
        ss = reconstruct_stems(
            W, H, X, mix, sample_rate=8000, n_fft=512, hop_length=128,
            policy="residual",
        )
        self.assertLess(ss.component_sum_error, 1e-12)
        self.assertGreaterEqual(ss.residual_energy_ratio, 0.0)
        self.assertLess(ss.residual_energy_ratio, 0.5)

    def test_binary_mask_one_hot(self):
        mix, X, W = _fixture()
        H = np.abs(np.linalg.lstsq(W, np.abs(X), rcond=None)[0])
        parts = component_spectrograms(W, H, X, policy="binary")
        # Outside the numerical floor, exactly one component claims each bin.
        active = np.abs(X) > 1e-6
        masks = np.array([np.abs(p) / np.abs(X) for p in parts])
        self.assertTrue(np.allclose(masks.sum(axis=0)[active], 1.0, atol=1e-9))

    def test_grouping_partitions_mixture(self):
        mix, X, W = _fixture()
        H = np.abs(np.linalg.lstsq(W, np.abs(X), rcond=None)[0])
        ss = group_components(
            W, H, X, mix, [[0], [1]], sample_rate=8000, n_fft=512, hop_length=128,
            labels=["g0", "g1"],
        )
        self.assertLess(ss.component_sum_error, 1e-10)

    def test_grouping_rejects_overlap(self):
        mix, X, W = _fixture()
        H = np.abs(np.linalg.lstsq(W, np.abs(X), rcond=None)[0])
        with self.assertRaises(ValueError):
            group_components(
                W, H, X, mix, [[0, 1], [1]], sample_rate=8000,
                n_fft=512, hop_length=128, labels=["a", "b"],
            )

    def test_zero_model_does_not_divide_by_zero(self):
        mix, X, W = _fixture()
        H = np.zeros((2, X.shape[1]))
        ss = reconstruct_stems(
            W, H, X, mix, sample_rate=8000, n_fft=512, hop_length=128,
        )
        # Nothing is attributed; the policy reports the consequence instead of
        # emitting NaN.
        self.assertGreater(ss.masks_sum_error, 0.99)
        self.assertTrue(all(np.all(np.isfinite(s)) for s in ss.stems))
        # Since nothing was attributed, the stems sum to zero, not to the mix.
        self.assertGreater(ss.component_sum_error, 0.99)

    def test_time_gate_segment(self):
        from nmfsep.server import _time_gate
        sr, n = 1000, 3000
        g = _time_gate(n, sr, 0.5, 1.5)
        self.assertEqual(g.shape, (n,))
        self.assertAlmostEqual(float(g[1000]), 1.0, places=6)
        self.assertAlmostEqual(float(g[100]), 0.0, places=6)
        self.assertAlmostEqual(float(g[2000]), 0.0, places=6)
        # edges are ramped, interior is flat
        self.assertTrue(0.0 <= float(g[505]) <= 1.0)
        self.assertAlmostEqual(float(g.sum()), 1000.0, delta=30.0)


if __name__ == "__main__":
    unittest.main()
