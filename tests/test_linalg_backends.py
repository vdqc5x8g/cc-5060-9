import unittest

import numpy as np

from nmfsep.linalg import (
    BackendUnavailableError,
    NumpyBackend,
    ScipyBackend,
    available_backends,
    get_backend,
)


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.A = rng.uniform(0.1, 1.0, (40, 7))
        self.B = rng.uniform(0.1, 1.0, (7, 5))
        self.R = rng.uniform(0.1, 1.0, (40, 5))

    def test_both_backends_available(self):
        self.assertEqual(available_backends(), ["numpy", "scipy"])
        self.assertIsInstance(get_backend("numpy"), NumpyBackend)
        self.assertIsInstance(get_backend("scipy"), ScipyBackend)
        with self.assertRaises(ValueError):
            get_backend("does-not-exist")

    def test_unknown_precision_rejected(self):
        with self.assertRaises(ValueError):
            get_backend("numpy", "float16")

    def test_matmul_all_layouts_and_precisions(self):
        variants = {
            "C": (np.ascontiguousarray(self.A), np.ascontiguousarray(self.B)),
            "F": (np.asfortranarray(self.A), np.asfortranarray(self.B)),
            "slice": (self.A[::2], self.B),
        }
        for name in ("numpy", "scipy"):
            for dtype in ("float64", "float32"):
                be = get_backend(name, dtype)
                tol = 1e-5 if dtype == "float32" else 1e-12
                for layout, (x, y) in variants.items():
                    xx = x[:34] if layout == "slice" else x
                    out = be.matmul(xx, y)
                    # layout adaptation: canonical float64 C-order at the boundary
                    self.assertEqual(out.dtype, np.float64)
                    self.assertTrue(out.flags["C_CONTIGUOUS"])
                    np.testing.assert_allclose(
                        out, np.asarray(xx, np.float64) @ np.asarray(y, np.float64),
                        atol=tol, err_msg=f"{name}/{dtype}/{layout}",
                    )

    def test_svd_and_lstsq_match_reference(self):
        ref_s = np.linalg.svd(self.A, compute_uv=False)
        ref_x, *_ = np.linalg.lstsq(self.A, self.R, rcond=None)
        for name in ("numpy", "scipy"):
            for dtype in ("float64", "float32"):
                be = get_backend(name, dtype)
                tol = 1e-5 if dtype == "float32" else 1e-10
                _, s, _ = be.svd(self.A)
                np.testing.assert_allclose(s, ref_s, atol=tol)
                np.testing.assert_allclose(be.lstsq(self.A, self.R), ref_x, atol=tol)

    def test_nnls_clip_is_nonnegative(self):
        for name in ("numpy", "scipy"):
            be = get_backend(name)
            x = be.solve_nnls_clip(self.A, self.R - self.R.mean())
            self.assertTrue(np.all(x >= 0.0))

    def test_sum_matches_value_layout_independent(self):
        for name in ("numpy", "scipy"):
            be = get_backend(name)
            np.testing.assert_allclose(be.sum(self.A, axis=0), self.A.sum(axis=0), atol=1e-9)
            # layout independence: C- and Fortran-ordered inputs reduce identically
            np.testing.assert_allclose(
                be.sum(self.A, axis=0), be.sum(np.asfortranarray(self.A), axis=0),
                atol=1e-12,
            )
            kept = be.sum(self.A, axis=1, keepdims=True)
            self.assertEqual(kept.shape, (self.A.shape[0], 1))
            self.assertAlmostEqual(be.sum(self.A), float(self.A.sum()), places=10)

    def test_scipy_reduction_order_genuinely_differs(self):
        # The alternative backend must be a real alternative: its naive
        # accumulation order rounds differently from numpy's pairwise sum.
        # Mixed magnitudes make the last-bit difference deterministic.
        rng = np.random.default_rng(0)
        x = np.concatenate([
            rng.uniform(1e-12, 1e-10, 1500),
            rng.uniform(1.0, 2.0, 1500),
        ])
        self.assertFalse(np.equal(ScipyBackend().sum(x), np.sum(x)))

    def test_normalize_columns_preserves_product(self):
        rng = np.random.default_rng(2)
        for name in ("numpy", "scipy"):
            be = get_backend(name)
            W = rng.uniform(0.1, 1.0, (10, 4)).copy()
            H = rng.uniform(0.1, 1.0, (4, 9)).copy()
            product = be.matmul(W, H)
            free = np.array([True, True, False, True])
            be.normalize_columns(W, H, free, 1e-12)
            np.testing.assert_allclose(be.matmul(W, H), product, atol=1e-12)
            np.testing.assert_allclose(W[:, 0].sum(), 1.0)
            np.testing.assert_allclose(W[:, 3].sum(), 1.0)
            self.assertGreater(abs(W[:, 2].sum() - 1.0), 1e-6)

    def test_describe_carries_identity(self):
        self.assertEqual(
            get_backend("scipy", "float32").describe(),
            {"backend": "scipy", "working_dtype": "float32"},
        )


if __name__ == "__main__":
    unittest.main()
