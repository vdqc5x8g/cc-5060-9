"""Deterministic synthetic mixtures with ground-truth stems.

The synth tracks are built from oscillators + envelopes so the project needs no
downloaded source material to produce *evaluated* data.  They are intentionally
not instrument recordings: they are stand-ins whose exact stems we know.  The
unknown-recording demo uses a public-domain piano recording instead.

Two mixture types are provided:

``separable``
    Three timbres with clearly different spectral envelopes (a low harmonic
    stack, a high bell-like partial set, a noise-like band).  Each has its own
    note pattern, with deliberate time overlaps.
``ambiguous``
    Two sources built from *the same* two spectral templates with different
    activation patterns, plus a third distinguishing source.  Any 2-template
    solution can explain the overlap equally well, so this is the case used to
    demonstrate non-unique solutions and gain-scaling comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 22050


@dataclass
class Track:
    name: str
    signal: np.ndarray
    descriptor: dict = field(default_factory=dict)


def _adsr(n: int, sr: int, attack=0.01, decay=0.08, sustain=0.7, release=0.15) -> np.ndarray:
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def harmonic_note(freq: float, dur: float, sr: int, partials: list[tuple[float, float]],
                  attack=0.01, decay=0.08, sustain=0.7, release=0.15,
                  vibrato_hz=0.0, vibrato_depth=0.0, detune=0.0) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    phase = 2 * np.pi * freq * t
    if vibrato_hz:
        phase += (vibrato_depth / vibrato_hz) * np.sin(2 * np.pi * vibrato_hz * t)
    y = np.zeros(n)
    for k, amp in partials:
        f = freq * k * (1.0 + detune)
        y += amp * np.sin(2 * np.pi * f * t + 0.13 * k)
    return y * _adsr(n, sr, attack, decay, sustain, release)


def noise_note(dur: float, sr: int, center: float, bandwidth: float, rng: np.random.Generator,
               attack=0.005, decay=0.05, sustain=0.8, release=0.1) -> np.ndarray:
    n = int(dur * sr)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    shape = np.exp(-0.5 * ((freqs - center) / max(bandwidth, 1e-6)) ** 2)
    y = np.fft.irfft(spec * shape, n=n)
    y /= max(np.max(np.abs(y)), 1e-12)
    return y * _adsr(n, sr, attack, decay, sustain, release)


def _place(canvas: np.ndarray, sig: np.ndarray, start: int, gain: float) -> None:
    end = min(len(canvas), start + len(sig))
    if end <= start:
        return
    canvas[start:end] += gain * sig[: end - start]


def _notes_from_pattern(spec: list[tuple[float, float, float]], sr: int, total: int,
                        note_fn) -> np.ndarray:
    out = np.zeros(total)
    for start_s, dur_s, gain in spec:
        sig = note_fn(dur_s)
        _place(out, sig, int(start_s * sr), gain)
    return out


def make_separable(duration: float = 8.0, sr: int = SR, seed: int = 7) -> tuple[np.ndarray, list[Track]]:
    rng = np.random.default_rng(seed)
    total = int(duration * sr)

    low_pattern = [(0.0, 1.0, 0.9), (1.25, 0.7, 0.8), (2.0, 1.0, 1.0), (3.4, 1.2, 0.85),
                   (4.9, 0.8, 0.9), (5.7, 1.4, 1.0), (7.2, 0.7, 0.75)]
    low = _notes_from_pattern(low_pattern, sr, total, lambda d: harmonic_note(
        110.0, d, sr, [(1, 1.0), (2, 0.5), (3, 0.35), (4, 0.22), (5, 0.12)],
        attack=0.02, decay=0.12, sustain=0.65, release=0.2))

    bell_pattern = [(0.5, 1.2, 0.8), (2.1, 0.9, 0.7), (2.9, 1.1, 0.9), (4.2, 1.0, 0.6),
                    (5.9, 1.2, 0.8), (7.0, 0.9, 0.7)]
    bell = _notes_from_pattern(bell_pattern, sr, total, lambda d: harmonic_note(
        660.0, d, sr, [(1, 0.5), (1.51, 0.7), (2.76, 0.9), (4.07, 0.6), (5.43, 0.35)],
        attack=0.004, decay=0.3, sustain=0.25, release=0.6, detune=0.001))

    noise_pattern = [(0.9, 1.4, 0.55), (2.4, 1.6, 0.5), (4.5, 1.8, 0.6), (6.3, 1.5, 0.5)]
    noise = _notes_from_pattern(noise_pattern, sr, total, lambda d: noise_note(
        d, sr, center=3200.0, bandwidth=1100.0, rng=rng))

    mix = low + bell + noise
    peak = np.max(np.abs(mix)) or 1.0
    mix = 0.85 * mix / peak
    tracks = [
        Track("low_harmonic", low / peak * 0.85, {"kind": "synthetic", "descriptor": "110 Hz harmonic stack"}),
        Track("high_bell", bell / peak * 0.85, {"kind": "synthetic", "descriptor": "inharmonic bell-like partials"}),
        Track("band_noise", noise / peak * 0.85, {"kind": "synthetic", "descriptor": "3 kHz band-limited noise"}),
    ]
    return mix, tracks


def make_ambiguous(duration: float = 8.0, sr: int = SR, seed: int = 11) -> tuple[np.ndarray, list[Track]]:
    """Two sources with *identical magnitude spectra*, plus a distinct third.

    ``twin_A`` and ``twin_B`` play the same pitch with the same partial
    amplitudes and the same note times, differing only in phase (and a 0.3%
    detune).  Their individual waveforms are linearly independent, but their
    magnitude spectra are indistinguishable to any magnitude-domain method.
    The gain split between the two twins is therefore unobservable: every split
    ``a + b = total`` fits the mixture equally well.  A distinct narrowband
    source is added so the experiment still has a recoverable component.

    This is a deliberate failure case, not a benchmark of what NMF can do.
    """
    rng = np.random.default_rng(seed)
    total = int(duration * sr)
    partials = [(1, 1.0), (2, 0.6), (3, 0.28), (4, 0.12)]

    def twin_notes(rng_local: np.random.Generator, detune: float) -> np.ndarray:
        out = np.zeros(total)
        for start_s, dur_s, gain in [(0.3, 1.1, 0.8), (2.0, 1.3, 0.75),
                                     (3.8, 1.0, 0.8), (5.5, 1.2, 0.75),
                                     (6.9, 0.9, 0.6)]:
            sig = harmonic_note(
                196.0, dur_s, sr, partials, attack=0.03, decay=0.15,
                sustain=0.6, release=0.35, detune=detune,
            )
            _place(out, sig, int(start_s * sr), gain)
        return out

    twin_a = twin_notes(rng, 0.0)
    twin_b = twin_notes(rng, 0.003)
    dist = _notes_from_pattern(
        [(1.1, 1.6, 0.55), (4.4, 1.6, 0.55)], sr, total,
        lambda d: noise_note(d, sr, center=5200.0, bandwidth=700.0, rng=rng),
    )
    mix = twin_a + twin_b + dist
    peak = np.max(np.abs(mix)) or 1.0
    mix = 0.85 * mix / peak
    tracks = [
        Track("twin_A", twin_a / peak * 0.85,
              {"kind": "synthetic", "descriptor": "196 Hz, identical magnitude spectrum to twin_B"}),
        Track("twin_B", twin_b / peak * 0.85,
              {"kind": "synthetic", "descriptor": "196 Hz, identical magnitude spectrum to twin_A"}),
        Track("distinct_narrowband", dist / peak * 0.85,
              {"kind": "synthetic", "descriptor": "5.2 kHz narrowband bursts"}),
    ]
    return mix, tracks


def make_test_tone(freq: float = 440.0, duration: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    y = 0.4 * np.sin(2 * np.pi * freq * t)
    y += 0.1 * np.sin(2 * np.pi * 2 * freq * t)
    y *= _adsr(len(y), sr, attack=0.05, decay=0.1, sustain=0.8, release=0.3)
    return y
