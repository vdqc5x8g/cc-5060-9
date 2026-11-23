"""Spectrogram / signal metrics used by the evaluation and the CLI."""

from __future__ import annotations

import numpy as np

from .stft import stft


def sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-invariant SDR in dB (energy ratio of reference to error)."""
    r = np.asarray(reference, dtype=np.float64)
    e = np.asarray(estimate, dtype=np.float64)
    n = min(len(r), len(e))
    r, e = r[:n], e[:n]
    alpha = float(np.dot(e, r) / max(np.dot(r, r), eps))
    target = alpha * r
    err = e - target
    return float(10.0 * np.log10(max(np.sum(target**2), eps) / max(np.sum(err**2), eps)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-invariant SDR (zero-mean before projection, as in Le Roux 2019)."""
    r = np.asarray(reference, dtype=np.float64)
    e = np.asarray(estimate, dtype=np.float64)
    n = min(len(r), len(e))
    r, e = r[:n] - np.mean(r[:n]), e[:n] - np.mean(e[:n])
    alpha = float(np.dot(e, r) / max(np.dot(r, r), eps))
    target = alpha * r
    err = e - target
    return float(10.0 * np.log10(max(np.sum(target**2), eps) / max(np.sum(err**2), eps)))


def best_permutation_sdr(
    references: list[np.ndarray], estimates: list[np.ndarray]
) -> tuple[list[int], float]:
    """Global permutation maximising mean SDR (ground-truth evaluation only)."""
    from itertools import permutations

    n = min(len(references), len(estimates))
    best = None
    for perm in permutations(range(n), n):
        vals = [sdr(references[perm[i]], estimates[i]) for i in range(n)]
        score = float(np.mean(vals))
        if best is None or score > best[1]:
            best = (list(perm), score)
    assert best is not None
    return best


def projection_leakage(
    references: list[np.ndarray], estimates: list[np.ndarray], eps: float = 1e-12
) -> np.ndarray:
    """Energy ratio of estimate i attributable to reference j, in dB.

    Row i, column j = 10 log10( ||P_j e_i||^2 / ||e_i||^2 ), where P_j is the
    orthogonal projection of the estimate onto the ground-truth reference j.
    The diagonal is the wanted contribution; off-diagonal entries are leakage
    from other known sources (a ground-truth-only diagnostic).
    """
    refs = np.stack([np.asarray(r, dtype=np.float64) for r in references])
    ests = np.stack([np.asarray(e, dtype=np.float64) for e in estimates])
    n = min(refs.shape[1], ests.shape[1])
    refs, ests = refs[:, :n], ests[:, :n]
    gram = refs @ refs.T
    coef = np.linalg.solve(gram + eps * np.eye(len(refs)), refs @ ests.T).T  # (est, ref)
    out = np.zeros((len(ests), len(refs)))
    for i in range(len(ests)):
        energy = float(np.sum(ests[i] ** 2)) or eps
        for j in range(len(refs)):
            proj = coef[i, j] * refs[j]
            out[i, j] = 10.0 * np.log10(max(float(np.sum(proj**2)), eps) / energy)
    return out


def spectral_convergence(reference: np.ndarray, estimate: np.ndarray, *,
                         n_fft: int = 1024, hop_length: int = 256, eps: float = 1e-12) -> float:
    """|| |STFT(r)| - |STFT(e)| ||_F / || |STFT(r)| ||_F."""
    R = np.abs(stft(reference, n_fft=n_fft, hop_length=hop_length))
    E = np.abs(stft(estimate, n_fft=n_fft, hop_length=hop_length))
    k = min(R.shape[1], E.shape[1])
    R, E = R[:, :k], E[:, :k]
    return float(np.linalg.norm(R - E) / max(np.linalg.norm(R), eps))


def spectrogram_mse(reference: np.ndarray, estimate: np.ndarray, *,
                    n_fft: int = 1024, hop_length: int = 256) -> float:
    R = np.abs(stft(reference, n_fft=n_fft, hop_length=hop_length))
    E = np.abs(stft(estimate, n_fft=n_fft, hop_length=hop_length))
    k = min(R.shape[1], E.shape[1])
    return float(np.mean((R[:, :k] - E[:, :k]) ** 2))


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray, *,
                          n_fft: int = 1024, hop_length: int = 256, eps: float = 1e-8) -> float:
    R = np.abs(stft(reference, n_fft=n_fft, hop_length=hop_length))
    E = np.abs(stft(estimate, n_fft=n_fft, hop_length=hop_length))
    k = min(R.shape[1], E.shape[1])
    R, E = R[:, :k], E[:, :k]
    num = np.linalg.norm(np.log10(R + eps) - np.log10(E + eps))
    den = np.linalg.norm(np.log10(R + eps))
    return float(num / max(den, eps))


def energy_db(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(10.0 * np.log10(max(float(np.mean(np.asarray(x) ** 2)), eps)))
