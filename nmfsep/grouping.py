"""Group NMF components into higher-level stems.

Nothing here assigns an instrument name.  Grouping is either manual (the UI)
or a documented, reversible rule that the user can inspect:

``activation_cluster``
    Correlation distance between activation rows (H), average linkage,
    ``n_groups`` clusters.  Components that rise and fall together are grouped.
``spectral_cluster``
    Cosine distance between templates (W columns).
``oracle``
    Evaluation-only: assign each component to the reference stem it explains
    best (projection energy).  This uses ground truth and is reported as an
    upper bound on attribution quality, never as an automatic result.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def activation_cluster(H: np.ndarray, n_groups: int, method: str = "average") -> list[list[int]]:
    X = np.asarray(H, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    if X.shape[0] <= n_groups:
        return [[i] for i in range(X.shape[0])]
    dist = pdist(X, metric="correlation")
    labels = fcluster(linkage(dist, method=method), n_groups, criterion="maxclust")
    groups = [list(np.flatnonzero(labels == g)) for g in sorted(set(labels))]
    return [g for g in groups if g]


def spectral_cluster(W: np.ndarray, n_groups: int, method: str = "average") -> list[list[int]]:
    X = np.asarray(W, dtype=np.float64).T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    if X.shape[0] <= n_groups:
        return [[i] for i in range(X.shape[0])]
    dist = pdist(X, metric="cosine")
    labels = fcluster(linkage(dist, method=method), n_groups, criterion="maxclust")
    return [list(np.flatnonzero(labels == g)) for g in sorted(set(labels))]


def oracle_grouping(
    component_images: np.ndarray, references: list[np.ndarray]
) -> list[list[int]]:
    """Evaluation-only component->reference assignment.

    ``component_images`` has shape ``(n_components, n_samples)`` (component
    waveforms).  Each component is assigned to the reference with maximal
    projection energy.  Returns groups ordered by reference index.
    """
    refs = np.stack([np.asarray(r, dtype=np.float64) for r in references])
    images = np.asarray(component_images, dtype=np.float64)
    n = min(refs.shape[1], images.shape[1])
    refs, images = refs[:, :n], images[:, :n]
    gram = refs @ refs.T
    coef = np.linalg.solve(gram + 1e-10 * np.eye(len(refs)), refs @ images.T)
    proj = coef * np.sum(refs[:, None, :] ** 2, axis=2)[:, :]
    # proj[j, k] = coefficient(j,k) * ||ref_j||^2 = projection energy of image k on ref j
    assign = np.argmax(proj, axis=0)
    groups: list[list[int]] = [[] for _ in references]
    for k, ref_idx in enumerate(assign):
        groups[int(ref_idx)].append(k)
    return groups


def grouping_diagnostics(groups: list[list[int]], H: np.ndarray, W: np.ndarray) -> list[dict]:
    out = []
    for gi, g in enumerate(groups):
        h = H[g]
        w = W[:, g]
        coh_h = _mean_abs_corr(h) if len(g) > 1 else 1.0
        coh_w = _mean_abs_corr(w.T) if len(g) > 1 else 1.0
        out.append(
            {
                "group": gi,
                "components": [int(i) for i in g],
                "gain": float(np.linalg.norm(h)),
                "activation_coherence": float(coh_h),
                "template_coherence": float(coh_w),
            }
        )
    return out


def _mean_abs_corr(M: np.ndarray) -> float:
    if M.shape[0] < 2:
        return 1.0
    X = M - M.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(X, axis=1)
    X = X / np.maximum(denom[:, None], 1e-12)
    C = np.abs(X @ X.T)
    iu = np.triu_indices_from(C, k=1)
    return float(C[iu].mean())
