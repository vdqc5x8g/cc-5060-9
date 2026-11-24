"""Nonnegative matrix factorisation of a magnitude spectrogram.

Model
-----
Given a nonnegative matrix ``V`` of shape ``(n_bins, n_frames)`` we seek
nonnegative ``W`` ``(n_bins, n_templates)`` and ``H``
``(n_templates, n_frames)`` minimising a beta-divergence ``D_beta(V || W H)``.

Supported objectives (all element-wise sums, so they are defined even when the
model is not normalised):

* ``beta=1`` generalised Kullback-Leibler divergence
  ``sum(V log(V / Vhat) - V + Vhat)``
* ``beta=2`` squared Frobenius error ``0.5 * sum((V - Vhat) ** 2)``
* ``beta=0`` Itakura-Saito divergence
  ``sum(V / Vhat - log(V / Vhat) - 1)``

Updates are the standard multiplicative rules for each beta; see
``_update_w`` / ``_update_h``.  Every update is monotone in the objective for
nonnegative V (numerically, a line-search-free multiplicative step may show
tiny increases; the result records the best observed state instead).

Anchoring
---------
``anchors`` is an optional ``(n_bins, n_anchor)`` matrix of *fixed* spectral
templates (columns).  They occupy template indices ``0..n_anchor-1`` and are
never updated; all other columns are free.  Anchor columns are normalised to
unit L1 on entry, so a unit of an anchor activation means "one unit of that
spectral prototype".  This makes activation gains comparable between anchored
solutions.

Scale handling
--------------
Free template columns are re-normalised to unit L1 after every update and the
removed scale is transferred to the corresponding row of H:
``W[:, k] *= s`` -> ``H[k, :] /= s``.  The model ``W H`` and the objective are
unchanged.  This removes the W/H scale ambiguity as a source of comparison
noise; a template's "gain" is then the energy of its H row, which we report.

Silence and tiny denominators
-----------------------------
Frames whose input energy is at most ``silence_rel * max_frame_energy`` are
marked silent.  Their H columns are fixed at zero (they carry no evidence),
they are excluded from loss reporting, and we report their count.  All
divisions use an additive floor ``eps``; ``eps`` is relative to the mean
positive input magnitude so the behaviour is invariant to global gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .linalg import LinAlgBackend, NumpyBackend, get_backend


@dataclass
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    loss_history: list[float]
    n_iter: int
    stop_reason: str
    beta: float
    init: str
    seed: int | None
    anchor_indices: list[int] = field(default_factory=list)
    silent_frames: list[int] = field(default_factory=list)
    eps: float = 0.0
    n_parameters: int = 0
    linalg_backend: str = "numpy"
    linalg_dtype: str = "float64"

    @property
    def reconstruction(self) -> np.ndarray:
        return self.W @ self.H

    @property
    def final_loss(self) -> float:
        return float(self.loss_history[-1]) if self.loss_history else float("nan")


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return a / np.maximum(b, eps)


def _divergence(
    V: np.ndarray, Vhat: np.ndarray, beta: float, eps: float,
    backend: LinAlgBackend | None = None,
) -> float:
    """Objective value.

    The element-wise terms are identical across backends; the final reduction
    goes through the selected backend's ``sum`` so a backend switch changes the
    accumulation order (and thus the last floating-point bits) rather than the
    mathematical objective.
    """
    be = backend or NumpyBackend()
    if beta == 2:
        return 0.5 * be.sum((V - Vhat) ** 2)
    if beta == 1:
        ratio = _safe_ratio(V, Vhat, eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(V > 0, V * np.log(np.maximum(ratio, 1e-300)), 0.0)
        return be.sum(term - V + Vhat)
    if beta == 0:
        ratio = _safe_ratio(V, Vhat, eps)
        return be.sum(ratio - np.log(np.maximum(ratio, 1e-300)) - 1.0)
    raise ValueError("beta must be 0, 1 or 2")


def _update_h(
    W: np.ndarray, H: np.ndarray, V: np.ndarray, beta: float, eps: float,
    backend: LinAlgBackend,
) -> np.ndarray:
    Vhat = np.maximum(backend.matmul(W, H), eps)
    if beta == 2:
        num = backend.matmul(W.T, V)
        den = backend.matmul(W.T, backend.matmul(W, H))
        return H * _safe_ratio(num, den, eps)
    if beta == 1:
        num = backend.matmul(W.T, (V / Vhat))
        den = backend.sum(W, axis=0)[:, None]
        return H * _safe_ratio(num, den, eps)
    # beta == 0 (Itakura-Saito)
    num = backend.matmul(W.T, (V / (Vhat**2)))
    den = backend.matmul(W.T, (1.0 / Vhat))
    return H * np.sqrt(_safe_ratio(num, den, eps))


def _update_w(
    W: np.ndarray, H: np.ndarray, V: np.ndarray, beta: float, eps: float,
    backend: LinAlgBackend,
) -> np.ndarray:
    Vhat = np.maximum(backend.matmul(W, H), eps)
    if beta == 2:
        num = backend.matmul(V, H.T)
        den = backend.matmul(backend.matmul(W, H), H.T)
        return W * _safe_ratio(num, den, eps)
    if beta == 1:
        num = backend.matmul((V / Vhat), H.T)
        den = backend.sum(H, axis=1)[None, :]
        return W * _safe_ratio(num, den, eps)
    num = backend.matmul((V / (Vhat**2)), H.T)
    den = backend.matmul((1.0 / Vhat), H.T)
    return W * np.sqrt(_safe_ratio(num, den, eps))


def initialize(
    V: np.ndarray, rank: int, method: str = "nndsvda", seed: int = 0,
    backend: LinAlgBackend | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic initialisation.

    ``nndsvd``: standard NNDSVD (Boutsidis & Gallopoulos 2008): the largest
    singular triplet contributes its nonnegative part, later triplets use the
    positive or negative residual side, whichever has more mass.  Exact zeros
    are kept, so components can die under multiplicative updates.
    ``nndsvda``: NNDSVD with every zero replaced by the mean of the positive
    entries of V.  This keeps all components alive; it is the default.
    ``random``: seeded uniform values scaled to the data.

    The SVD is taken through the selected linear-algebra backend, so both
    backends share the NNDSVD construction but use their own drivers.
    """
    be = backend or NumpyBackend()
    if method == "random":
        rng = np.random.default_rng(seed)
        scale = np.sqrt(max(V.mean(), 1e-12) / rank)
        W = rng.uniform(0.0, 1.0, (V.shape[0], rank)) * scale
        H = rng.uniform(0.0, 1.0, (rank, V.shape[1])) * scale
        return W, H
    if method not in {"nndsvd", "nndsvda"}:
        raise ValueError(f"unknown init {method!r}")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(max(V.mean(), 1e-12) / rank)
    U, s, Vt = be.svd(V)
    k = min(rank, len(s))
    W = np.zeros((V.shape[0], rank))
    H = np.zeros((rank, V.shape[1]))
    W[:, 0] = np.sqrt(s[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(s[0]) * np.abs(Vt[0, :])
    for j in range(1, k):
        x, y = U[:, j], Vt[j, :]
        xp, xn = np.maximum(x, 0), np.maximum(-x, 0)
        yp, yn = np.maximum(y, 0), np.maximum(-y, 0)
        xpn, xnn = np.linalg.norm(xp), np.linalg.norm(xn)
        ypn, ynn = np.linalg.norm(yp), np.linalg.norm(yn)
        mp, mn = xpn * ypn, xnn * ynn
        if mp >= mn and xpn > 1e-12 and ypn > 1e-12:
            u, v, sigma = xp / xpn, yp / ypn, mp
        elif xnn > 1e-12 and ynn > 1e-12:
            u, v, sigma = xn / xnn, yn / ynn, mn
        else:
            W[:, j] = rng.uniform(0.0, 1.0, V.shape[0]) * scale
            H[j, :] = rng.uniform(0.0, 1.0, V.shape[1]) * scale
            continue
        W[:, j] = np.sqrt(s[j]) * u
        H[j, :] = np.sqrt(s[j]) * v
    for j in range(rank):
        if W[:, j].sum() <= 1e-12:
            W[:, j] = rng.uniform(0.0, 1.0, V.shape[0]) * scale
        if H[j, :].sum() <= 1e-12:
            H[j, :] = rng.uniform(0.0, 1.0, V.shape[1]) * scale
    if method == "nndsvda":
        avg = float(V[V > 0].mean()) if np.any(V > 0) else 1.0
        W[W == 0] = avg
        H[H == 0] = avg
    return W, H


def initialize_h(
    V: np.ndarray, W: np.ndarray, method: str, seed: int = 0,
    backend: LinAlgBackend | None = None,
) -> np.ndarray:
    """Activation initialisation.

    ``nnls`` solves a clipped least-squares fit of V on the initial W.  This is
    deterministic and avoids the slow start that random H gives anchored runs.
    The least-squares solve itself is dispatched through the selected backend.
    """
    be = backend or NumpyBackend()
    rank = W.shape[1]
    if method == "random":
        rng = np.random.default_rng(seed)
        scale = np.sqrt(max(V.mean(), 1e-12) / rank)
        return rng.uniform(0.0, 1.0, (rank, V.shape[1])) * scale
    if method != "nnls":
        raise ValueError(f"unknown H init {method!r}")
    return be.solve_nnls_clip(W, V)


def _normalize_free_columns(
    W: np.ndarray, H: np.ndarray, free: np.ndarray, eps: float,
    backend: LinAlgBackend | None = None,
) -> None:
    """In place unit-L1 normalisation of free W columns, scale moved to H."""
    be = backend or NumpyBackend()
    be.normalize_columns(W, H, free, eps)


def nmf(
    V: np.ndarray,
    rank: int,
    *,
    anchors: np.ndarray | None = None,
    beta: float = 1.0,
    n_iter: int = 300,
    init: str = "nndsvda",
    seed: int = 0,
    eps: float | None = None,
    silence_rel: float = 1e-8,
    tol: float = 1e-5,
    patience: int = 15,
    min_iter: int = 10,
    h_init: str = "nnls",
    linalg_backend: str = "numpy",
    linalg_dtype: str = "float64",
    backend: LinAlgBackend | None = None,
) -> NMFResult:
    """Factorise ``V`` into ``W H`` with optional fixed anchor columns.

    Stopping condition: relative improvement of the objective over the last
    ``patience`` iterations is below ``tol`` (and at least ``min_iter``
    iterations have run), or ``n_iter`` is reached, or the numerical objective
    fails to be finite / decreases.  The returned ``W, H`` are the best states
    observed.

    ``linalg_backend`` selects the matrix-solve implementation (``numpy``
    reference or ``scipy`` LAPACK/BLAS alternative); both share this same
    algorithm, init grid and stopping contract.  ``backend`` accepts a
    pre-built backend instance and takes precedence when supplied.
    """
    be = backend or get_backend(linalg_backend, linalg_dtype)
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError("V must be 2-D")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if np.any(V < 0):
        raise ValueError("V must be nonnegative")
    if eps is None:
        positive = V[V > 0]
        eps = max(float(positive.mean()) * 1e-9 if positive.size else 1e-12, 1e-12)

    frame_energy = V.sum(axis=0)
    max_frame = float(frame_energy.max()) if frame_energy.size else 0.0
    silent = np.flatnonzero(frame_energy <= silence_rel * max(max_frame, 1e-30))
    active = np.setdiff1d(np.arange(V.shape[1]), silent)

    n_anchor = 0
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=np.float64)
        if anchors.ndim != 2 or anchors.shape[0] != V.shape[0]:
            raise ValueError("anchors must be (n_bins, n_anchor)")
        if np.any(anchors < 0):
            raise ValueError("anchors must be nonnegative")
        if anchors.shape[1] > rank:
            raise ValueError("number of anchors cannot exceed rank")
        n_anchor = anchors.shape[1]
        if n_anchor:
            colsum = anchors.sum(axis=0)
            if np.any(colsum <= 0):
                raise ValueError("anchor columns must have positive energy")
            anchors = anchors / colsum[None, :]
    if n_anchor == rank:
        # Still need free columns for a meaningful joint factorisation?  It is
        # allowed: H is estimated on fixed bases only.
        pass

    W0, H0 = initialize(V, rank, method=init, seed=seed, backend=be)
    W = W0.copy()
    H = H0.copy()
    if n_anchor:
        W[:, :n_anchor] = anchors
    H = initialize_h(V, W, h_init, seed=seed, backend=be)
    free = np.ones(rank, dtype=bool)
    free[:n_anchor] = False
    H[:, silent] = 0.0
    _normalize_free_columns(W, H, free, eps, backend=be)

    loss_history: list[float] = []
    stop_reason = "max_iter"
    best_loss = np.inf
    best_W, best_H = W.copy(), H.copy()
    stale = 0

    for it in range(1, n_iter + 1):
        H = _update_h(W, H, V, beta, eps, be)
        H[:, silent] = 0.0
        if free.any():
            W_free = _update_w(W, H, V, beta, eps, be)
            W[:, free] = W_free[:, free]
        _normalize_free_columns(W, H, free, eps, backend=be)
        H[:, silent] = 0.0

        vhat = be.matmul(W, H)
        if active.size:
            loss = _divergence(V[:, active], vhat[:, active], beta, eps, be)
        else:
            loss = 0.0
        if not np.isfinite(loss):
            stop_reason = "non_finite_loss"
            break
        if loss < best_loss:
            best_loss = loss
            best_W, best_H = W.copy(), H.copy()
        loss_history.append(float(loss))

        if it >= min_iter and len(loss_history) > patience:
            ref = loss_history[-patience - 1]
            denom = max(abs(ref), eps)
            improvement = (ref - loss) / denom
            if improvement < tol:
                stop_reason = "relative_tolerance"
                break
        if it > 2 and loss > loss_history[-2] * (1.0 + 1e-6):
            # Multiplicative updates should not increase the objective; if
            # they do beyond the numerical floor, keep the best state and stop.
            stop_reason = "loss_increase"
            break

    W, H = best_W, best_H
    n_params = int(W.size + H.size)
    return NMFResult(
        W=W,
        H=H,
        loss_history=loss_history,
        n_iter=len(loss_history),
        stop_reason=stop_reason,
        beta=beta,
        init=init,
        seed=seed,
        anchor_indices=list(range(n_anchor)),
        silent_frames=silent.tolist(),
        eps=float(eps),
        n_parameters=n_params,
        linalg_backend=be.name,
        linalg_dtype=np.dtype(be.working_dtype).name,
    )


def heldout_loss(
    V_train: np.ndarray,
    W: np.ndarray,
    V_hold: np.ndarray,
    *,
    beta: float = 1.0,
    n_iter: int = 200,
    anchors_count: int = 0,
    eps: float | None = None,
    tol: float = 1e-6,
    linalg_backend: str = "numpy",
    linalg_dtype: str = "float64",
    backend: LinAlgBackend | None = None,
) -> float:
    """Estimate held-out frames with W fixed, then report their divergence.

    This gives a model-complexity-aware comparison across template counts:
    more templates may lower the training loss without improving held-out
    frames.
    """
    be = backend or get_backend(linalg_backend, linalg_dtype)
    V_train = np.asarray(V_train, dtype=np.float64)
    V_hold = np.asarray(V_hold, dtype=np.float64)
    if eps is None:
        positive = V_train[V_train > 0]
        eps = max(float(positive.mean()) * 1e-9 if positive.size else 1e-12, 1e-12)
    rank = W.shape[1]
    rng = np.random.default_rng(12345)
    scale = np.sqrt(max(V_train.mean(), 1e-12) / rank)
    H = rng.uniform(0.0, 1.0, (rank, V_hold.shape[1])) * scale
    prev = np.inf
    for _ in range(n_iter):
        H = _update_h(W, H, V_hold, beta, eps, be)
        vhat = be.matmul(W, H)
        loss = _divergence(V_hold, vhat, beta, eps, be)
        if np.isfinite(prev) and abs(prev - loss) <= tol * max(abs(prev), eps):
            prev = loss
            break
        prev = loss
    return float(prev)


def gain_ambiguity_probe(
    V: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    beta: float = 1.0,
    eps: float | None = None,
    cos_threshold: float = 0.995,
    loss_rtol: float = 0.01,
    linalg_backend: str = "numpy",
    linalg_dtype: str = "float64",
    backend: LinAlgBackend | None = None,
) -> list[dict]:
    """Test whether two templates' gain split is observable.

    For each pair of templates whose shapes are nearly identical
    (cosine >= ``cos_threshold``), move *all* activation mass from template
    ``j`` onto template ``k`` while keeping total reconstruction fixed:
    ``W_k' = normalized(W_k + W_j)``; the pair's combined contribution
    ``W_k H_k + W_j H_j`` is unchanged only if the shapes match exactly.  If the
    resulting objective stays within ``loss_rtol`` of the original, the data
    cannot distinguish how much gain belongs to each template: their gain split
    is unidentifiable.  This probe is computed from the fitted model and V
    alone; it does not use any reference stems.
    """
    be = backend or get_backend(linalg_backend, linalg_dtype)
    if eps is None:
        pos = V[V > 0]
        eps = max(float(pos.mean()) * 1e-9 if pos.size else 1e-12, 1e-12)
    Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    C = be.matmul(Wn.T, Wn)
    base = _divergence(V, np.maximum(be.matmul(W, H), eps), beta, eps, be)
    out: list[dict] = []
    for k in range(W.shape[1]):
        for j in range(k + 1, W.shape[1]):
            if C[k, j] < cos_threshold:
                continue
            combined = be.matmul(W[:, [k, j]], H[[k, j], :])
            shape = Wn[:, k] + Wn[:, j]
            shape = shape / max(np.linalg.norm(shape), 1e-12)
            g = np.maximum(be.matmul(shape[None, :], combined)[0], 0.0)
            merged = np.outer(shape, g)
            probe = be.matmul(W, H) - combined + merged
            loss = _divergence(V, np.maximum(probe, eps), beta, eps, be)
            rel = (loss - base) / max(abs(base), eps)
            out.append(
                {
                    "templates": [k, j],
                    "template_cosine": float(C[k, j]),
                    "loss_relative_change": float(rel),
                    "unidentifiable_gain_split": bool(rel <= loss_rtol),
                    "gains_db": [
                        float(20 * np.log10(max(np.linalg.norm(H[k]), 1e-12))),
                        float(20 * np.log10(max(np.linalg.norm(H[j]), 1e-12))),
                    ],
                }
            )
    return out


@dataclass
class CandidateComparison:
    losses: list[float]
    relative_gaps: list[float]
    best_index: int
    assignment: list[int]
    similarities: np.ndarray
    gain_ratios: np.ndarray
    retained: list[bool]
    stable: bool


def compare_candidates(
    results: list[NMFResult], retain_rtol: float = 0.01, stability_cos: float = 0.98
) -> CandidateComparison:
    """Compare candidate solutions without assigning source identities.

    Templates are matched between candidates by maximising the mean cosine
    similarity of W columns (Hungarian assignment).  Activations are compared
    on that assignment via their gains (L2 norms of H rows).  A set of
    candidates is reported ``stable`` only when every retained candidate is
    template-wise similar to the best solution *and* per-template gains agree
    within 3 dB; otherwise the ambiguity is real and both solutions are kept.
    """
    if not results:
        raise ValueError("need at least one candidate")
    best = int(np.argmin([r.final_loss for r in results]))
    best_loss = results[best].final_loss
    losses = [r.final_loss for r in results]
    gaps = [(l - best_loss) / max(abs(best_loss), 1e-30) for l in losses]
    retained = [g <= retain_rtol for g in gaps]
    ref = results[best]
    assignment_rows = []
    sim_rows = []
    gain_rows = []
    for r in results:
        Wc = ref.W / np.maximum(np.linalg.norm(ref.W, axis=0, keepdims=True), 1e-12)
        Wr = r.W / np.maximum(np.linalg.norm(r.W, axis=0, keepdims=True), 1e-12)
        C = Wc.T @ Wr
        row, col = linear_sum_assignment(-C)
        mapping = [int(col[row.tolist().index(k)]) if k in row else k for k in range(ref.W.shape[1])]
        sims = np.array([C[k, mapping[k]] for k in range(ref.W.shape[1])])
        g_ref = np.linalg.norm(ref.H, axis=1)
        g_r = np.linalg.norm(r.H[mapping, :], axis=1)
        gain = g_r / np.maximum(g_ref, 1e-12)
        assignment_rows.append(mapping)
        sim_rows.append(sims)
        gain_rows.append(gain)
    similarities = np.array(sim_rows)
    gain_ratios = np.array(gain_rows)
    used = [i for i, keep in enumerate(retained) if keep]
    stable = True
    for i in used:
        if i == best:
            continue
        if similarities[i].min() < stability_cos or np.max(np.abs(np.log2(
            np.maximum(gain_ratios[i], 1e-12)
        ))) > 0.5:
            stable = False
            break
    return CandidateComparison(
        losses=losses,
        relative_gaps=gaps,
        best_index=best,
        assignment=assignment_rows,
        similarities=similarities,
        gain_ratios=gain_ratios,
        retained=retained,
        stable=stable,
    )
