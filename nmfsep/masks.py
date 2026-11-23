"""Turn NMF components into audio, with an explicit reconstruction policy.

Policies
--------
``wiener``
    Soft mask ``M_k = (W_k H_k)^p / sum_j (W_j H_j)^p`` with ``p=1`` by default,
    then each component's complex spectrogram is ``M_k * X`` and components are
    inverted.  Because the masks are a partition of unity per bin/frame, the
    sum of component spectrograms equals the mixture spectrogram *before*
    inversion; after independent inverse STFTs the sum equals the mixture's
    own inverse-STFT to floating point error.  This is the default.
``ratio``
    Mask ``M_k = (W_k H_k) / sum_j (W_j H_j)`` without exponent (p=1 ratio),
    optionally with a floor.  Identical to ``wiener`` at p=1 but kept as a
    separate name for readability.
``binary``
    One-hot mask on the arg-max component per bin/frame.  Also a partition of
    unity, so mixture consistency holds, but artifacts are larger.
``residual``
    Reconstruct each component with a soft mask, then subtract their sum from
    the *original mixture waveform* and add the residual to the component with
    the largest energy.  This makes ``sum(components) == mixture`` exactly in
    the sample domain, at the cost of putting model error into one stem.  The
    residual energy is reported so callers can judge that cost.

All policies return components whose sum is checked by
``reconstruction_report``.  Where a denominator is below ``eps`` the mask is
zero (a bin/frame where the model explains nothing is not attributed).

Regrouping
----------
NMF components are not instruments.  ``group_components`` combines a
user-supplied partition (list of lists of template indices) into stems and
reports the implied template-level masks, so a user can audition "template k
contribution" *and* "my group A attribution".  No component is ever labelled
with a source name automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .stft import istft


@dataclass
class StemSet:
    stems: list[np.ndarray]
    labels: list[str]
    sample_rate: int
    masks_sum_error: float
    component_sum_error: float
    residual_energy_ratio: float = 0.0
    policy: str = "wiener"


def component_spectrograms(
    W: np.ndarray, H: np.ndarray, X: np.ndarray, policy: str = "wiener", power: float = 1.0, eps: float | None = None
) -> list[np.ndarray]:
    """Complex spectrograms for each NMF component under ``policy``."""
    if policy not in {"wiener", "ratio", "binary"}:
        raise ValueError("component_spectrograms handles wiener/ratio/binary")
    if eps is None:
        pos = np.abs(X)
        eps = max(float(pos.mean()) * 1e-12, 1e-15)
    parts = np.maximum(W, 0.0)[:, :, None] * np.maximum(H, 0.0)[None, :, :]
    parts = parts.transpose(1, 0, 2)  # (n_templates, n_bins, n_frames)
    total = parts.sum(axis=0, keepdims=True)
    if policy == "binary":
        idx = np.argmax(parts, axis=0)
        masks = np.zeros_like(parts)
        np.put_along_axis(masks, idx[None, :, :], 1.0, axis=0)
    else:
        p = max(float(power), 1e-6)
        powered = np.maximum(parts, 0.0) ** p
        denom = powered.sum(axis=0, keepdims=True)
        masks = np.divide(powered, denom, out=np.zeros_like(powered), where=denom > eps)
        if total.min() <= eps:
            # Frames/bin rows where the model is silent get zero mask, which
            # the divide already produced (where=...).  The remainder is
            # attributed nowhere; residual policy below captures it.
            pass
    return [masks[k] * X for k in range(parts.shape[0])]


def reconstruct_stems(
    W: np.ndarray,
    H: np.ndarray,
    X: np.ndarray,
    mixture: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    labels: Sequence[str] | None = None,
    policy: str = "wiener",
    power: float = 1.0,
) -> StemSet:
    """Reconstruct one stem per NMF component (no instrument names)."""
    n = len(mixture)
    spec_parts = component_spectrograms(W, H, X, "wiener" if policy == "residual" else policy, power)
    raw = [istft(s, n_fft=n_fft, hop_length=hop_length, length=n) for s in spec_parts]
    if policy == "residual":
        approx = np.sum(raw, axis=0)
        residual = mixture - approx
        energies = [float(np.sum(r * r)) for r in raw]
        target = int(np.argmax(energies)) if energies else 0
        if raw:
            raw[target] = raw[target] + residual
        res_ratio = float(
            np.sum(residual * residual) / max(np.sum(mixture * mixture), 1e-12)
        )
        spec_sum_err = 0.0  # mixture consistency moved to the sample domain
    else:
        res_ratio = 0.0
        spec_sum_err = float(
            np.linalg.norm(np.sum(spec_parts, axis=0) - X)
            / max(np.linalg.norm(X), 1e-12)
        )
    if labels is None:
        labels = [f"component_{k:02d}" for k in range(W.shape[1])]
    stems = [np.asarray(r, dtype=np.float64) for r in raw]
    # ``masks_sum_error`` is always measured on the complex spectrograms, so it
    # directly reports how much of X the chosen policy failed to attribute
    # (1.0 means "nothing was attributed", e.g. an all-zero model).
    spec_sum_err = float(
        np.linalg.norm(np.sum(spec_parts, axis=0) - X) / max(np.linalg.norm(X), 1e-12)
    )
    comp_sum_err = float(
        np.linalg.norm(np.sum(stems, axis=0) - mixture)
        / max(np.linalg.norm(mixture), 1e-12)
    )
    return StemSet(
        stems=stems,
        labels=list(labels),
        sample_rate=sample_rate,
        masks_sum_error=spec_sum_err,
        component_sum_error=comp_sum_err,
        residual_energy_ratio=res_ratio,
        policy=policy,
    )


def group_components(
    W: np.ndarray,
    H: np.ndarray,
    X: np.ndarray,
    mixture: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    labels: Sequence[str],
    policy: str = "wiener",
    power: float = 1.0,
) -> StemSet:
    """Combine component activations into user-defined groups.

    ``groups`` is e.g. ``[[0, 2], [1]]``.  Components missing from all groups
    are not attributed to any stem; the reported ``masks_sum_error`` /
    ``component_sum_error`` expose that, so the user can see the declared
    policy's consequence rather than discovering it by ear.
    """
    n = len(mixture)
    used = [k for g in groups for k in g]
    if len(set(used)) != len(used):
        raise ValueError("a component may appear in only one group")
    if any(k < 0 or k >= W.shape[1] for k in used):
        raise ValueError("group index out of range")
    parts = np.maximum(W, 0.0)[:, :, None] * np.maximum(H, 0.0)[None, :, :]
    parts = parts.transpose(1, 0, 2)  # (n_templates, n_bins, n_frames)
    grouped = np.zeros((len(groups), parts.shape[1], parts.shape[2]))
    for gi, g in enumerate(groups):
        for k in g:
            grouped[gi] += parts[k]
    total = grouped.sum(axis=0, keepdims=True)
    p = max(float(power), 1e-6)
    powered = np.maximum(grouped, 0.0) ** p
    denom = powered.sum(axis=0, keepdims=True)
    eps = max(float(np.abs(X).mean()) * 1e-12, 1e-15)
    masks = np.divide(powered, denom, out=np.zeros_like(powered), where=denom > eps)
    spec_parts = [masks[i] * X for i in range(len(groups))]
    raw = [
        istft(s, n_fft=n_fft, hop_length=hop_length, length=n) for s in spec_parts
    ]
    if policy == "residual":
        approx = np.sum(raw, axis=0)
        residual = mixture - approx
        energies = [float(np.sum(r * r)) for r in raw]
        target = int(np.argmax(energies)) if energies else 0
        if raw:
            raw[target] = raw[target] + residual
        res_ratio = float(
            np.sum(residual * residual) / max(np.sum(mixture * mixture), 1e-12)
        )
    else:
        res_ratio = 0.0
    spec_sum_err = float(
        np.linalg.norm(np.sum(spec_parts, axis=0) - X) / max(np.linalg.norm(X), 1e-12)
    )
    stems = [np.asarray(r, dtype=np.float64) for r in raw]
    comp_sum_err = float(
        np.linalg.norm(np.sum(stems, axis=0) - mixture)
        / max(np.linalg.norm(mixture), 1e-12)
    )
    return StemSet(
        stems=stems,
        labels=[str(l) for l in labels],
        sample_rate=sample_rate,
        masks_sum_error=spec_sum_err,
        component_sum_error=comp_sum_err,
        residual_energy_ratio=res_ratio,
        policy=policy,
    )
