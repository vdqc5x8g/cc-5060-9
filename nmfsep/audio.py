"""Audio loading/saving helpers (mono, float64 in memory)."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from math import gcd

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0)


def load_audio(path: str, *, mono: bool = True, offset: float = 0.0,
               duration: float | None = None, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    start = int(offset * info.samplerate)
    stop = None if duration is None else start + int(duration * info.samplerate)
    data, sr = sf.read(
        path,
        dtype="float64",
        always_2d=True,
        start=start,
        stop=stop,
    )
    if mono:
        data = data.mean(axis=1)
    if target_sr is not None and target_sr != sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr
    if not mono:
        data = data.T
    return data, sr


def save_audio(path: str, x: np.ndarray, sr: int, *, subtype: str = "PCM_16") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak * 1.0
    sf.write(path, x, sr, subtype=subtype)


def normalize_peak(x: np.ndarray, target: float = 0.9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-12:
        return x
    return x / peak * target
