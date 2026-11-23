"""Real STFT / inverse STFT with explicit, documented conventions.

Conventions
-----------
* ``n_fft`` samples per frame, periodic Hann window (scipy's ``get_window``).
* ``hop_length`` samples between frame starts.  With ``center=True`` the signal
  is zero-padded by ``n_fft // 2`` on both sides so frame ``t`` is centred on
  sample ``t * hop_length``.
* Frames are *not* individually normalised; the window is applied once on
  analysis and again (as synthesis window) on reconstruction, then divided by
  the overlap-add of the squared window and the FFT length.  This makes
  ``istft(stft(x))`` numerically equal to ``x`` to ~1e-6 for any window/hop
  pair with nonzero OLA coverage (tested in ``tests/test_stft.py``).
* Magnitude ``|S|`` is what the factorisation models.  Phase from the mixture
  is reused; masked reconstructions are therefore not phase-aware estimators
  but they do satisfy the declared consistency policy exactly (see
  ``nmfsep.reconstruct``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def stft(
    x: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    center: bool = True,
) -> np.ndarray:
    """Complex STFT of a mono signal.

    Returns an array of shape ``(n_fft // 2 + 1, n_frames)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("stft expects a mono (1-D) signal")
    if hop_length <= 0 or hop_length > n_fft:
        raise ValueError("hop_length must be in (0, n_fft]")
    win = get_window("hann", n_fft, fftbins=True)
    if center:
        pad = n_fft // 2
        xp = np.pad(x, (pad, pad), mode="constant")
    else:
        xp = x
    if len(xp) < n_fft:
        xp = np.pad(xp, (0, n_fft - len(xp)), mode="constant")
    n_frames = 1 + int(np.ceil((len(xp) - n_fft) / hop_length))
    tail = n_fft + hop_length * (n_frames - 1) - len(xp)
    if tail > 0:
        xp = np.pad(xp, (0, tail), mode="constant")
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = xp[idx] * win[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1).T
    return spec


def istft(
    spec: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    center: bool = True,
    length: int | None = None,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse STFT.  ``length`` truncates to the original sample count."""
    if spec.ndim != 2:
        raise ValueError("istft expects (n_bins, n_frames)")
    if window is None:
        window = get_window("hann", n_fft, fftbins=True)
    n_frames = spec.shape[1]
    if n_frames < 1:
        return np.zeros(0 if length is None else length)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1)
    frames *= window[None, :]
    out_len = n_fft + hop_length * (n_frames - 1)
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    wsq = window**2
    for t in range(n_frames):
        start = t * hop_length
        out[start : start + n_fft] += frames[t]
        wsum[start : start + n_fft] += wsq
    nonzero = wsum > 1e-12
    out[nonzero] /= wsum[nonzero]
    if center:
        pad = n_fft // 2
        out = out[pad : len(out) - pad if len(out) > 2 * pad else len(out)]
    if length is not None:
        if len(out) < length:
            out = np.pad(out, (0, length - len(out)))
        else:
            out = out[:length]
    return out


def stft_consistency_error(
    x: np.ndarray, spec: np.ndarray, n_fft: int, hop_length: int
) -> float:
    """Relative L2 error between ``x`` and ``istft(spec)`` (sanity metric)."""
    rec = istft(spec, n_fft=n_fft, hop_length=hop_length, length=len(x))
    denom = np.linalg.norm(x) or 1.0
    return float(np.linalg.norm(rec - x) / denom)
