import unittest

import numpy as np

from nmfsep.nmf import (
    _divergence,
    compare_candidates,
    gain_ambiguity_probe,
    initialize,
    nmf,
)


def toy_V(seed=0, n_bins=40, n_frames=80, rank=3):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 1.0, (n_bins, rank))
    H = np.zeros((rank, n_frames))
    for k in range(rank):
        H[k, k * 20 : (k + 1) * 20] = rng.uniform(0.5, 1.5, 20)
    return W @ H + 0.01, W, H


class NMFtests(unittest.TestCase):
    def test_candidate_grid_recovers_low_rank(self):
        V, W, H = toy_V()
        rs = [
            nmf(V, 3, beta=1.0, n_iter=1500, init=init, seed=seed,
                patience=300, tol=1e-10, min_iter=20)
            for init, seed in [("nndsvda", 0), ("random", 1), ("random", 2)]
        ]
        rels = [np.linalg.norm(V - r.W @ r.H) / np.linalg.norm(V) for r in rs]
        self.assertLess(min(rels), 5e-3)
        for r in rs:
            h = r.loss_history
            self.assertTrue(all(b <= a * (1 + 1e-6) for a, b in zip(h, h[1:])))
        self.assertTrue(min(r.final_loss for r in rs)
                        < _divergence(V, W @ H, 1.0, 1e-12))

    def test_plain_nndsvd_can_stall_but_stays_monotone(self):
        V, _, _ = toy_V()
        r = nmf(V, 3, beta=1.0, n_iter=300, init="nndsvd", seed=0,
                patience=200, tol=1e-10, min_iter=20)
        h = r.loss_history
        self.assertTrue(all(b <= a * (1 + 1e-6) for a, b in zip(h, h[1:])))

    def test_loss_monotone(self):
        V, _, _ = toy_V()
        for beta in (0.0, 1.0, 2.0):
            r = nmf(V, 4, beta=beta, n_iter=200, init="nndsvd", seed=1,
                    patience=500, tol=0.0, min_iter=1)
            h = r.loss_history
            self.assertTrue(all(b <= a * (1 + 1e-6) for a, b in zip(h, h[1:])), beta)

    def test_anchors_are_fixed(self):
        V, _, _ = toy_V()
        rng = np.random.default_rng(4)
        anchors = rng.uniform(0.1, 1.0, (V.shape[0], 2))
        anchors /= anchors.sum(axis=0, keepdims=True)
        r = nmf(V, 4, anchors=anchors, beta=1.0, n_iter=50, init="random", seed=0)
        np.testing.assert_allclose(r.W[:, :2], anchors, rtol=1e-12, atol=1e-15)

    def test_silence_frames_get_zero_activation(self):
        V, _, _ = toy_V()
        V[:, 10:20] = 0.0
        r = nmf(V, 3, beta=1.0, n_iter=100, init="nndsvd", seed=0, silence_rel=1e-8)
        self.assertTrue(np.all(r.H[:, 10:20] == 0.0))
        self.assertTrue(set(range(10, 20)).issubset(r.silent_frames))

    def test_scale_invariance_of_relative_loss(self):
        V, _, _ = toy_V()
        r1 = nmf(V, 3, beta=1.0, n_iter=200, init="random", seed=2,
                 patience=100, tol=1e-10, min_iter=10)
        r2 = nmf(V * 100.0, 3, beta=1.0, n_iter=200, init="random", seed=2,
                 patience=100, tol=1e-10, min_iter=10)
        self.assertAlmostEqual(r1.final_loss / r2.final_loss, 1 / 100.0, places=6)
        np.testing.assert_allclose(r1.W, r2.W, rtol=1e-6)

    def test_duplicated_template_gain_unidentifiable(self):
        rng = np.random.default_rng(0)
        shape = rng.uniform(0.2, 1.0, 60)
        shape /= shape.sum()
        V = np.outer(shape, rng.uniform(0.5, 1.5, 100))
        W = np.stack([shape * 1.001, shape * 0.999], axis=1)
        W /= W.sum(axis=0, keepdims=True)
        H = np.zeros((2, 100))
        H[0] = 0.7
        H[1] = 0.3
        probes = gain_ambiguity_probe(V, W, H, beta=1.0, cos_threshold=0.99)
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0]["unidentifiable_gain_split"])
        self.assertLess(probes[0]["loss_relative_change"], 0.01)

    def test_comparison_groups_similar_solutions(self):
        V, _, _ = toy_V()
        rs = [nmf(V, 3, init="random", seed=s, n_iter=300, patience=200,
                  tol=1e-10, min_iter=20) for s in range(3)]
        cmp = compare_candidates(rs, retain_rtol=0.2, stability_cos=0.9)
        self.assertEqual(len(cmp.losses), 3)
        self.assertEqual(len(cmp.retained), 3)
        self.assertIn(cmp.best_index, range(3))

    def test_initialization_shapes_and_nonnegative(self):
        V, _, _ = toy_V()
        for method in ("nndsvd", "random"):
            W, H = initialize(V, 3, method, seed=0)
            self.assertEqual(W.shape, (V.shape[0], 3))
            self.assertEqual(H.shape, (3, V.shape[1]))
            self.assertTrue(np.all(W >= 0) and np.all(H >= 0))

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            nmf(-np.ones((5, 5)), 2)


if __name__ == "__main__":
    unittest.main()
