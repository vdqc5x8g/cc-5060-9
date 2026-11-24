"""Semantic equivalence of factorisations across linear-algebra backends.

A backend switch can return the *same* factorisation with its columns in a
different order and with W/H scale moved between the two factors.  The
objective value and the reconstruction ``W H`` are invariant to both
transformations, so comparing losses (or checking ``W_a == W_b`` element-wise)
cannot decide equivalence and must never be used to claim that two components
are the same voice.

The procedure here is:

1. **Column-wise normalisation matching.**  Each W column is scaled to unit L1
   and the inverse scale is applied to its H row, so both solutions live in the
   same gauge before any comparison.
2. **Permutation alignment.**  Columns of the candidate solution are assigned to
   reference columns by Hungarian matching on cosine similarity (the same
   similarity used for candidate retention), giving an explicit, auditable
   permutation.
3. **Reconstruction-space comparison.**  Equivalence is decided on the aligned
   products ``W H`` (which are gauge/permutation invariant), plus per-component
   shape and activation agreement on the alignment.
4. **Tolerance for summation order.**  GEMM and the explicit reduction order of
   the alternative backend change only last-bit rounding; tolerances are
   relative and reported alongside the raw gaps rather than silently widened.

Nothing here assigns instrument names: an aligned column pair means "same
factor", and whether that factor is a voice remains the operator's call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .linalg import LinAlgBackend, NumpyBackend


def _unit_l1_gauge(W: np.ndarray, H: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy in the unit-L1 W gauge (scale moved into H)."""
    Wn = np.array(W, dtype=np.float64, copy=True)
    Hn = np.array(H, dtype=np.float64, copy=True)
    for k in range(Wn.shape[1]):
        s = float(Wn[:, k].sum())
        if s > eps:
            Wn[:, k] /= s
            Hn[k, :] *= s
    return Wn, Hn


def _column_cosine(W_ref_n: np.ndarray, W_cand_n: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of two sets of unit-L1 columns."""
    R = W_ref_n / np.maximum(np.linalg.norm(W_ref_n, axis=0, keepdims=True), 1e-12)
    C = W_cand_n / np.maximum(np.linalg.norm(W_cand_n, axis=0, keepdims=True), 1e-12)
    return R.T @ C


@dataclass
class EquivalenceReport:
    equivalent: bool
    rank_matches: bool
    permutation: list[int]                 # ref column k -> cand column p[k]
    column_cosines: list[float]            # cosine after alignment, per ref col
    min_column_cosine: float
    activation_correlation: list[float]    # H-row correlation on the alignment
    min_activation_correlation: float
    reconstruction_relative_error: float   # ||WH_ref - WH_cand_aligned|| / ||WH_ref||
    loss_ref: float
    loss_cand: float
    loss_relative_gap: float
    backend_ref: str
    backend_cand: str
    tolerances: dict
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "equivalent": self.equivalent,
            "rank_matches": self.rank_matches,
            "permutation": self.permutation,
            "column_cosines": [float(v) for v in self.column_cosines],
            "min_column_cosine": float(self.min_column_cosine),
            "activation_correlation": [float(v) for v in self.activation_correlation],
            "min_activation_correlation": float(self.min_activation_correlation),
            "reconstruction_relative_error": float(self.reconstruction_relative_error),
            "loss_ref": float(self.loss_ref),
            "loss_cand": float(self.loss_cand),
            "loss_relative_gap": float(self.loss_relative_gap),
            "backend_ref": self.backend_ref,
            "backend_cand": self.backend_cand,
            "tolerances": dict(self.tolerances),
            "reasons": list(self.reasons),
        }


def align_factors(
    W_ref: np.ndarray, H_ref: np.ndarray,
    W_cand: np.ndarray, H_cand: np.ndarray,
    *,
    eps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Align a candidate factorisation to the reference's column ordering.

    Both inputs are first put in the unit-L1 W gauge, then candidate columns are
    permuted by maximum total cosine similarity.  Returns the aligned
    ``(W_cand_aligned, H_cand_aligned, permutation)`` where ``permutation[k]`` is
    the candidate index matched to reference column ``k``.
    """
    Wr, Hr = _unit_l1_gauge(np.asarray(W_ref, np.float64), np.asarray(H_ref, np.float64),
                            eps or 1e-12)
    Wc, Hc = _unit_l1_gauge(np.asarray(W_cand, np.float64), np.asarray(H_cand, np.float64),
                            eps or 1e-12)
    if Wr.shape[1] != Wc.shape[1]:
        raise ValueError(
            f"rank mismatch: reference has {Wr.shape[1]} columns, "
            f"candidate has {Wc.shape[1]}"
        )
    C = _column_cosine(Wr, Wc)
    row, col = linear_sum_assignment(-C)
    permutation = [int(col[list(row).index(k)]) for k in range(Wr.shape[1])]
    return Wc[:, permutation], Hc[permutation, :], permutation


def _row_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= eps or nb <= eps:
        # Both silent rows are vacuously "the same"; one silent / one active is
        # a genuine disagreement and the denominator keeps cosine at 0.
        return 1.0 if (na <= eps and nb <= eps) else 0.0
    return float(np.dot(a, b) / (na * nb))


def equivalence_report(
    V: np.ndarray,
    ref,
    cand,
    *,
    cosine_tol: float = 0.999,
    activation_tol: float = 0.999,
    reconstruction_rtol: float = 1e-6,
    loss_rtol: float = 1e-6,
    beta: float = 1.0,
    eps: float | None = None,
    backend: LinAlgBackend | None = None,
) -> EquivalenceReport:
    """Decide whether two NMF results are the same solution in another gauge.

    ``ref``/``cand`` are either :class:`~nmfsep.nmf.NMFResult` objects or plain
    ``(W, H)`` tuples.  Equal target values alone never make two results
    equivalent: the aligned reconstruction, per-column shapes and per-row
    activations must all agree within tolerances chosen to absorb summation
    order differences but not genuine alternative optima.
    """
    from .nmf import NMFResult, _divergence  # local import: avoid a cycle

    be = backend or NumpyBackend()
    V = np.asarray(V, dtype=np.float64)
    if eps is None:
        pos = V[V > 0]
        eps = max(float(pos.mean()) * 1e-9 if pos.size else 1e-12, 1e-12)

    def _wh(x):
        if isinstance(x, NMFResult):
            return x.W, x.H, x.linalg_backend
        return x[0], x[1], "external"

    W_ref, H_ref, name_ref = _wh(ref)
    W_cand, H_cand, name_cand = _wh(cand)
    rank_matches = W_ref.shape == W_cand.shape and H_ref.shape == H_cand.shape

    Wr, Hr = _unit_l1_gauge(W_ref, H_ref, eps)
    reasons: list[str] = []
    if not rank_matches:
        reasons.append(
            f"rank/shape mismatch: ref {W_ref.shape}/{H_ref.shape} vs "
            f"cand {W_cand.shape}/{H_cand.shape}"
        )
        permutation = list(range(min(W_ref.shape[1], W_cand.shape[1])))
        return EquivalenceReport(
            equivalent=False, rank_matches=False, permutation=permutation,
            column_cosines=[], min_column_cosine=0.0, activation_correlation=[],
            min_activation_correlation=0.0, reconstruction_relative_error=float("inf"),
            loss_ref=float("nan"), loss_cand=float("nan"), loss_relative_gap=float("inf"),
            backend_ref=name_ref, backend_cand=name_cand,
            tolerances={"cosine": cosine_tol, "activation": activation_tol,
                        "reconstruction_rtol": reconstruction_rtol, "loss_rtol": loss_rtol},
            reasons=reasons,
        )

    Wc_aligned, Hc_aligned, permutation = align_factors(W_ref, H_ref, W_cand, H_cand, eps=eps)
    cos = _column_cosine(Wr, Wc_aligned)
    col_cosines = [float(cos[k, k]) for k in range(Wr.shape[1])]
    act_corr = [_row_correlation(Hr[k], Hc_aligned[k]) for k in range(Hr.shape[0])]

    rec_ref = be.matmul(Wr, Hr)
    rec_cand = be.matmul(Wc_aligned, Hc_aligned)
    denom = max(float(np.linalg.norm(rec_ref)), eps)
    rec_err = float(np.linalg.norm(rec_ref - rec_cand) / denom)

    loss_ref = _divergence(V, np.maximum(be.matmul(W_ref, H_ref), eps), beta, eps, be)
    loss_cand = _divergence(V, np.maximum(be.matmul(W_cand, H_cand), eps), beta, eps, be)
    loss_gap = abs(loss_ref - loss_cand) / max(abs(loss_ref), eps)

    min_cos = min(col_cosines)
    min_act = min(act_corr)
    checks = {
        "column cosine": (min_cos >= cosine_tol, f"min cosine {min_cos:.6f} < {cosine_tol}"),
        "activation correlation": (min_act >= activation_tol,
                                   f"min activation corr {min_act:.6f} < {activation_tol}"),
        "reconstruction": (rec_err <= reconstruction_rtol,
                           f"reconstruction rel err {rec_err:.3e} > {reconstruction_rtol:.0e}"),
        "loss gap": (loss_gap <= loss_rtol,
                     f"loss relative gap {loss_gap:.3e} > {loss_rtol:.0e}"),
    }
    for label, (ok, detail) in checks.items():
        if not ok:
            reasons.append(f"{label} check failed: {detail}")
    return EquivalenceReport(
        equivalent=not reasons,
        rank_matches=True,
        permutation=permutation,
        column_cosines=col_cosines,
        min_column_cosine=min_cos,
        activation_correlation=act_corr,
        min_activation_correlation=min_act,
        reconstruction_relative_error=rec_err,
        loss_ref=loss_ref,
        loss_cand=loss_cand,
        loss_relative_gap=loss_gap,
        backend_ref=name_ref,
        backend_cand=name_cand,
        tolerances={"cosine": cosine_tol, "activation": activation_tol,
                    "reconstruction_rtol": reconstruction_rtol, "loss_rtol": loss_rtol},
        reasons=reasons,
    )
