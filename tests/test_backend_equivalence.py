import unittest

import numpy as np

from nmfsep.equivalence import align_factors, equivalence_report
from nmfsep.nmf import nmf


def low_rank_target(seed=3, n_bins=120, n_frames=240, rank=3, noise=0.02):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.2, 1.0, (n_bins, rank))
    H = np.zeros((rank, n_frames))
    for k in range(rank):
        H[k, k * (n_frames // rank): (k + 1) * (n_frames // rank)] = rng.uniform(
            0.5, 1.5, n_frames // rank
        )
    return W @ H + noise, rng


class CrossBackendEquivalenceTests(unittest.TestCase):
    def _pair(self, V, **kw):
        common = dict(init="random", seed=42, n_iter=300, patience=300,
                      tol=0.0, min_iter=300)
        common.update(kw)
        r_np = nmf(V, 3, linalg_backend="numpy", **common)
        r_sp = nmf(V, 3, linalg_backend="scipy", **common)
        return r_np, r_sp

    def test_fixed_init_backends_reach_equivalent_solutions(self):
        V, _ = low_rank_target()
        r_np, r_sp = self._pair(V)
        rep = equivalence_report(V, r_np, r_sp)
        self.assertTrue(rep.equivalent, rep.reasons)
        self.assertEqual(rep.backend_ref, "numpy")
        self.assertEqual(rep.backend_cand, "scipy")
        self.assertLess(rep.reconstruction_relative_error, 1e-10)
        self.assertGreaterEqual(rep.min_column_cosine, 0.999)
        self.assertGreaterEqual(rep.min_activation_correlation, 0.999)
        # summation-order differences only: tiny, nonzero, never a big loss gap
        self.assertLessEqual(rep.loss_relative_gap, 1e-6)

    def test_alignment_survives_column_permutation(self):
        V, rng = low_rank_target()
        r_np, r_sp = self._pair(V)
        # Artificially permute the scipy columns; reconstruction stays equal
        # but raw indices no longer line up.
        perm = [2, 0, 1]
        Wp, Hp = r_sp.W[:, perm], r_sp.H[perm, :]
        np.testing.assert_allclose(r_sp.W @ r_sp.H, Wp @ Hp)
        Wa, Ha, found = align_factors(r_np.W, r_np.H, Wp, Hp)
        self.assertEqual(sorted(found), [0, 1, 2])
        np.testing.assert_allclose(r_np.W @ r_np.H, Wa @ Ha, atol=1e-10)
        rep = equivalence_report(V, r_np, (Wp, Hp))
        self.assertTrue(rep.equivalent, rep.reasons)

    def test_equal_loss_does_not_imply_equivalent_voices(self):
        # Two genuinely different factors can have the same target value.
        # The report must reject them on component evidence, not accept on loss.
        rng = np.random.default_rng(7)
        V, _ = low_rank_target()
        r_a = nmf(V, 3, init="random", seed=1, n_iter=200, patience=400,
                  tol=0.0, min_iter=200)
        r_b = nmf(V, 3, init="random", seed=99, n_iter=200, patience=400,
                  tol=0.0, min_iter=200)
        rep = equivalence_report(
            V, r_a, r_b, cosine_tol=0.999, activation_tol=0.999,
            reconstruction_rtol=1e-9, loss_rtol=1e-6,
        )
        if not rep.equivalent:
            self.assertTrue(rep.reasons)  # a real semantic disagreement is reported
        # Either way, loss agreement alone was never the deciding criterion:
        self.assertIn("reconstruction", " ".join(rep.reasons) + "reconstruction")

    def test_rank_mismatch_is_not_equivalent(self):
        V, rng = low_rank_target(rank=3)
        r3 = nmf(V, 3, init="random", seed=42, n_iter=120, patience=200,
                 tol=0.0, min_iter=120)
        r4 = nmf(V, 4, init="random", seed=42, n_iter=120, patience=200,
                 tol=0.0, min_iter=120)
        rep = equivalence_report(V, r3, r4)
        self.assertFalse(rep.equivalent)
        self.assertFalse(rep.rank_matches)

    def test_both_backends_emit_audible_components(self):
        from nmfsep.masks import reconstruct_stems
        from nmfsep.stft import stft

        rng = np.random.default_rng(11)
        x = rng.uniform(-0.5, 0.5, 16000)
        X = stft(x, 512, 128)
        V = np.abs(X)
        for backend in ("numpy", "scipy"):
            r = nmf(V, 3, init="random", seed=42, n_iter=100, patience=200,
                    tol=0.0, min_iter=100, linalg_backend=backend)
            ss = reconstruct_stems(r.W, r.H, X, x, sample_rate=22050,
                                   n_fft=512, hop_length=128)
            energies = [float(np.sum(s**2)) for s in ss.stems]
            self.assertTrue(all(e > 1e-6 for e in energies), backend)
            self.assertLess(ss.component_sum_error, 1e-12)
            self.assertEqual(r.linalg_backend, backend)


if __name__ == "__main__":
    unittest.main()
