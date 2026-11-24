"""Pluggable linear-algebra backend for the matrix solves used by the NMF path.

Two concrete backends implement one contract:

``numpy`` (reference)
    ``numpy.linalg`` SVD/lstsq and ``numpy`` matmul/sum.  This is the historical
    numerics of the project and remains the default, so saved runs are
    reproducible bit-for-bit.

``scipy`` (alternative)
    LAPACK drivers through ``scipy.linalg``: ``?gesdd`` SVD, ``?gelsd`` least
    squares and BLAS ``?gemm`` products.  It is a genuinely different solve
    path (different driver, different block/accumulation order inside GEMM),
    not a renamed numpy call.

Layout adaptation
-----------------
Every entry point accepts C- or Fortran-ordered (and generally non-contiguous)
inputs; each backend copies into the layout its driver wants and every result is
returned canonicalised to C-contiguous float64, so downstream code never sees a
backend-dependent stride.

Precision adaptation
--------------------
``working_dtype`` lets a caller request float32 driver precision; the work is
done in that precision but results are promoted back to float64 at the boundary,
keeping the on-disk contract (float64 ``W``/``H``) stable.  The NMF loop itself
runs float64 on both backends; single precision is exercised by the backend
tests and the equivalence harness.

Normalisation
-------------
Column normalisation (unit L1, scale transferred to the partner factor) is a
backend operation so the same divide/multiply semantics hold regardless of
driver; see :meth:`LinAlgBackend.normalize_columns`.

Semantic equivalence across backends must never be decided by comparing target
values alone (the objective is permutation/scale invariant).  Use
``nmfsep.equivalence.align_factors`` / ``equivalence_report`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend's native library is not importable."""


# Canonical output dtype/order for every backend boundary.
CANONICAL_DTYPE = np.float64


class LinAlgBackend:
    """Backend contract: matmul / sum / svd / lstsq / normalisation."""

    name = "base"

    def __init__(self, working_dtype: np.dtype | type | str = CANONICAL_DTYPE):
        self.working_dtype = np.dtype(working_dtype)
        if self.working_dtype not in (np.float64, np.float32):
            raise ValueError("working_dtype must be float64 or float32")

    # -- boundary helpers -------------------------------------------------
    def canonical(self, a: np.ndarray) -> np.ndarray:
        """Return ``a`` as C-contiguous float64 regardless of input layout."""
        out = np.asarray(a)
        if out.dtype != CANONICAL_DTYPE:
            out = out.astype(CANONICAL_DTYPE)
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)
        return out

    def _work(self, a: np.ndarray) -> np.ndarray:
        """Cast to working precision; contiguous so the driver layout is known."""
        out = np.asarray(a, dtype=self.working_dtype)
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)
        return out

    # -- primitives -------------------------------------------------------
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sum(self, a: np.ndarray, axis=None, keepdims: bool = False):
        raise NotImplementedError

    def svd(self, a: np.ndarray):
        """Thin SVD returning ``(U, s, Vt)`` with singular values descending."""
        raise NotImplementedError

    def lstsq(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Least-squares solution ``x`` to ``a x ≈ b`` (no rcond truncation)."""
        raise NotImplementedError

    def solve_nnls_clip(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Unconstrained least squares clipped to the nonnegative orthant."""
        return np.maximum(self.lstsq(a, b), 0.0)

    # -- normalisation contract ------------------------------------------
    def normalize_columns(
        self, w: np.ndarray, h: np.ndarray, free: np.ndarray, eps: float
    ) -> None:
        """Unit-L1 normalise free columns of ``w`` in place, scale moved to ``h``.

        ``w[:, k] /= s`` and ``h[k, :] *= s`` leave the product ``w @ h``
        unchanged.  Columns at or below ``eps`` are left alone.
        """
        for k in np.flatnonzero(free):
            s = float(self.sum(w[:, k]))
            if s > eps:
                w[:, k] = self.canonical(w[:, k] / s)
                h[k, :] = self.canonical(h[k, :] * s)

    def describe(self) -> dict:
        return {"backend": self.name, "working_dtype": np.dtype(self.working_dtype).name}


class NumpyBackend(LinAlgBackend):
    """Reference backend: numpy linalg drivers and numpy reduction order."""

    name = "numpy"

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.canonical(np.dot(np.asarray(a), np.asarray(b)))

    def sum(self, a: np.ndarray, axis=None, keepdims: bool = False):
        out = np.sum(np.asarray(a), axis=axis, keepdims=keepdims)
        if isinstance(out, np.ndarray):
            return self.canonical(out)
        return float(out)

    def svd(self, a: np.ndarray):
        u, s, vt = np.linalg.svd(self._work(a), full_matrices=False)
        return self.canonical(u), self.canonical(s), self.canonical(vt)

    def lstsq(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x, *_ = np.linalg.lstsq(self._work(a), self._work(b), rcond=None)
        return self.canonical(x)


class ScipyBackend(LinAlgBackend):
    """Alternative backend: explicit LAPACK/BLAS drivers via scipy.linalg."""

    name = "scipy"

    def __init__(self, working_dtype=CANONICAL_DTYPE):
        try:
            import scipy.linalg  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise BackendUnavailableError(f"scipy backend unavailable: {exc!r}") from exc
        super().__init__(working_dtype)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """BLAS GEMM product.

        ``scipy.linalg.blas`` is called directly instead of numpy's dispatcher,
        so the alternative backend accumulates partial products through the
        BLAS implementation's own block order rather than numpy's.  Transposed
        inputs are handled through the GEMM transpose flags rather than copies.
        """
        from scipy.linalg import blas

        aw, bw = self._work(a), self._work(b)
        prefix = "s" if self.working_dtype == np.float32 else "d"
        gemm = getattr(blas, f"{prefix}gemm")
        # scipy's GEMM wrapper interprets an array as column-major storage.
        # * Fortran-ordered operands are genuine column-major matrices: a plain
        #   GEMM yields the physical product A @ B.
        # * C-ordered (row-major) operands use the identity
        #   (A @ B)_row = (B^T @ A^T)_col: swap the operands and set both
        #   transpose flags.  The driver returns the column-major view of the
        #   transposed physical shape, so one final transpose restores the
        #   row-major product.  The accumulation still runs through the BLAS
        #   block order rather than numpy's dispatcher.
        if aw.flags["F_CONTIGUOUS"] and bw.flags["F_CONTIGUOUS"]:
            out = gemm(1.0, aw, bw)
        else:
            aw = np.ascontiguousarray(aw)
            bw = np.ascontiguousarray(bw)
            out = gemm(1.0, bw, aw, trans_a=True, trans_b=True).T
        return self.canonical(out)

    def sum(self, a: np.ndarray, axis=None, keepdims: bool = False):
        """Naive sequential accumulation, deliberately not pairwise blocked.

        numpy's reduction uses pairwise/blocked summation; here each output
        element is accumulated one term at a time along the moved reduction
        axes, so rounding genuinely differs while remaining layout independent
        (C- and Fortran-ordered inputs give the same sequence).
        """
        arr = np.asarray(a, dtype=np.float64)
        if axis is None:
            total = 0.0
            for v in np.ascontiguousarray(arr).reshape(-1, order="C"):
                total += float(v)
            return total
        axis = axis if isinstance(axis, tuple) else (axis,)
        axis = tuple(ax % arr.ndim for ax in axis)
        moved = np.moveaxis(arr, axis, tuple(range(arr.ndim - len(axis), arr.ndim)))
        reduce_ndim = len(axis)
        leading = moved.shape[: moved.ndim - reduce_ndim]
        flat_reduce = int(np.prod(moved.shape[moved.ndim - reduce_ndim:]))
        view = moved.reshape(*leading, flat_reduce)
        out = np.zeros(leading, dtype=np.float64)
        for i in range(flat_reduce):
            slab = view[..., i]
            for idx in np.ndindex(leading):
                out[idx] += float(slab[idx])
        if keepdims:
            for ax in sorted(axis):
                out = np.expand_dims(out, ax)
        return self.canonical(out)

    def svd(self, a: np.ndarray):
        import scipy.linalg as sla

        u, s, vt = sla.svd(self._work(a), full_matrices=False, lapack_driver="gesdd")
        return self.canonical(u), self.canonical(s), self.canonical(vt)

    def lstsq(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import scipy.linalg as sla

        x, *_ = sla.lstsq(self._work(a), self._work(b), lapack_driver="gelsd")
        return self.canonical(x)


@dataclass(frozen=True)
class LinAlgConfig:
    """Selectable backend description carried through the pipeline."""

    backend: str = "numpy"
    working_dtype: str = "float64"

    def build(self) -> LinAlgBackend:
        return get_backend(self.backend, self.working_dtype)


def get_backend(name: str = "numpy", working_dtype=CANONICAL_DTYPE) -> LinAlgBackend:
    key = str(name).lower()
    if key in ("numpy", "np", "reference"):
        return NumpyBackend(working_dtype)
    if key in ("scipy", "lapack", "alternative"):
        return ScipyBackend(working_dtype)
    raise ValueError(
        f"unknown linalg backend {name!r}; expected 'numpy' or 'scipy'"
    )


def available_backends() -> list[str]:
    names = ["numpy"]
    try:
        ScipyBackend()
        names.append("scipy")
    except BackendUnavailableError:
        pass
    return names
