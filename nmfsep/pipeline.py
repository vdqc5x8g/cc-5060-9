"""End-to-end pipeline: mix -> STFT -> NMF candidates -> stems + diagnostics.

The two ambiguity strategies requested are implemented here:

1. **Multiple candidates near the optimum** ("retain candidates when solutions
   are similar"): several initialisations (deterministic NNDSVD at different
   seeds, random at different seeds, and an anchored variant when anchors are
   supplied) are run; every solution within ``retain_rtol`` of the best loss is
   retained and compared template-by-template with a Hungarian match.  If
   template shapes and gains agree, the solution is reported as stable; if not,
   the ambiguity is reported and both are kept for audition.

2. **Fixing bases of observable sources as anchors** ("anchor on a fixed
   observable source basis"): when the caller can supply an isolated reference
   (e.g. a calibration tone, a known stationary source, or a previously
   validated template), its spectral prototype becomes a fixed NMF column.  The
   remaining columns adapt.  Anchors make that source's attribution independent
   of the free-factorisation ambiguity; a paired unanchored run is computed so
   the effect on other components is visible.

Neither path converts a component into an instrument name.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import grouping
from .masks import StemSet, component_spectrograms, group_components, reconstruct_stems
from .metrics import (
    energy_db,
    projection_leakage,
    sdr,
    si_sdr,
    spectral_convergence,
    spectrogram_mse,
)
from .nmf import CandidateComparison, NMFResult, compare_candidates, gain_ambiguity_probe, heldout_loss, nmf
from .stft import istft, stft


@dataclass
class Config:
    n_fft: int = 1024
    hop_length: int = 256
    n_templates: int = 4
    beta: float = 1.0
    n_iter: int = 300
    tol: float = 1e-5
    patience: int = 15
    min_iter: int = 10
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    init: str = "grid"
    h_init: str = "nnls"
    retain_rtol: float = 0.01
    stability_cos: float = 0.98
    mask_power: float = 1.0
    policy: str = "wiener"
    silence_rel: float = 1e-8
    include_unanchored: bool = True
    include_anchored: bool = True
    prefer_anchored: bool = True
    anchor_source: str | None = None  # restrict anchors to this key; None = all supplied
    anchor_keys: tuple[str, ...] | None = None
    heldout_fraction: float = 0.15
    # Selectable linear-algebra backend for the matrix solves (SVD/lstsq/GEMM).
    # "numpy" keeps the historical reference numerics; "scipy" runs the same
    # contract through LAPACK/BLAS drivers.  Both write float64 C-order W/H.
    linalg_backend: str = "numpy"
    linalg_dtype: str = "float64"


@dataclass
class PipelineResult:
    config: Config
    sample_rate: int
    mixture: np.ndarray
    X: np.ndarray
    magnitude: np.ndarray
    best: NMFResult
    candidates: list[NMFResult]
    comparison: CandidateComparison
    stem_set: StemSet
    anchored: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    grouping: list[list[int]] | None = None
    grouping_rule: str | None = None
    group_stems: StemSet | None = None


def _select_heldout(n_frames: int, fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if fraction <= 0 or n_frames < 10:
        idx = np.arange(n_frames)
        return idx, idx
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(n_frames * fraction)))
    hold = np.sort(rng.choice(n_frames, size=n_hold, replace=False))
    train = np.setdiff1d(np.arange(n_frames), hold)
    return train, hold


def anchor_templates_from_signals(
    anchors: dict[str, np.ndarray], n_fft: int, hop_length: int, sr: int,
) -> np.ndarray:
    """Turn isolated anchor signals into (n_bins, n_anchor) unit-L1 columns.

    The anchor is a *spectral prototype*: the mean magnitude spectrum over
    frames with above-median energy.  It is fixed during NMF.  Callers must
    still decide (and we report) how the anchor was obtained.
    """
    cols = []
    for name, sig in anchors.items():
        if len(sig) < n_fft:
            raise ValueError(f"anchor {name!r} is shorter than n_fft")
        S = np.abs(stft(sig, n_fft=n_fft, hop_length=hop_length))
        e = S.sum(axis=0)
        if e.max() <= 0:
            raise ValueError(f"anchor {name!r} is silent")
        keep = e >= np.median(e[e > 0])
        col = S[:, keep].mean(axis=1)
        s = col.sum()
        if s <= 0:
            raise ValueError(f"anchor {name!r} has no energy")
        cols.append(col / s)
    return np.stack(cols, axis=1)


def run_pipeline(
    mixture: np.ndarray,
    sample_rate: int,
    cfg: Config,
    *,
    anchors: dict[str, np.ndarray] | None = None,
    reference_stems: list[np.ndarray] | None = None,
    reference_labels: list[str] | None = None,
) -> PipelineResult:
    """Run candidate NMF, pick/retain candidates, reconstruct stems.

    ``reference_stems`` is optional and used *only* for evaluation metrics and
    oracle grouping diagnostics; it never enters the factorisation or the
    reconstruction.
    """
    mixture = np.asarray(mixture, dtype=np.float64)
    X = stft(mixture, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    magnitude = np.abs(X)

    anchor_mat = None
    anchor_names: list[str] = []
    if cfg.anchor_source and anchors:
        if cfg.anchor_source in anchors:
            anchor_mat = anchor_templates_from_signals(
                {cfg.anchor_source: anchors[cfg.anchor_source]},
                cfg.n_fft, cfg.hop_length, sample_rate,
            )
            anchor_names = [cfg.anchor_source]
    elif anchors:
        # Anchor every supplied source, deterministic order by key.
        keys = sorted(anchors)
        anchor_mat = anchor_templates_from_signals(
            {k: anchors[k] for k in keys}, cfg.n_fft, cfg.hop_length, sample_rate
        )
        anchor_names = keys

    train_idx, hold_idx = _select_heldout(magnitude.shape[1], cfg.heldout_fraction)

    candidates: list[NMFResult] = []
    candidate_meta: list[dict[str, Any]] = []

    def _run(anchors_arg, init: str, seed: int, kind: str) -> None:
        res = nmf(
            magnitude,
            cfg.n_templates,
            anchors=anchors_arg,
            beta=cfg.beta,
            n_iter=cfg.n_iter,
            init=init,
            seed=seed,
            tol=cfg.tol,
            patience=cfg.patience,
            min_iter=cfg.min_iter,
            silence_rel=cfg.silence_rel,
            h_init=cfg.h_init,
            linalg_backend=cfg.linalg_backend,
            linalg_dtype=cfg.linalg_dtype,
        )
        candidates.append(res)
        candidate_meta.append({"kind": kind, "init": init, "seed": seed,
                               "anchored": bool(res.anchor_indices),
                               "linalg_backend": res.linalg_backend,
                               "linalg_dtype": res.linalg_dtype})

    # Deterministic grid: NNDSVD plus seeded random starts.  The grid exists
    # because multiplicative NMF has many local optima; which one you land in
    # is itself a source of non-uniqueness and is reported per candidate.
    if cfg.include_unanchored:
        grid = [("nndsvda", cfg.seeds[0])] + [("random", s) for s in cfg.seeds[1:]]
        if cfg.init not in {"grid", "nndsvd", "nndsvda", "random"}:
            raise ValueError(f"unknown init {cfg.init!r}")
        if cfg.init != "grid":
            grid = [(cfg.init, s) for s in cfg.seeds]
        for init, seed in grid:
            _run(None, init, seed, "unanchored")
    if anchor_mat is not None and cfg.include_anchored:
        for seed in cfg.seeds:
            _run(anchor_mat, "random", seed, "anchored")

    comparison = compare_candidates(
        candidates, retain_rtol=cfg.retain_rtol, stability_cos=cfg.stability_cos
    )
    # Selection rule: the overall best candidate is chosen by objective, but
    # when anchors were supplied the caller asked specifically about the
    # anchored constraint, so the best *anchored* candidate is the reported
    # decomposition and the best unanchored one is kept as a reference.  Both
    # are compared template-by-template; the unanchored run is never silently
    # dropped, because it can explain the mixture equally well.
    anchored_idx = [i for i, c in enumerate(candidates) if c.anchor_indices]
    unanchored_idx = [i for i, c in enumerate(candidates) if not c.anchor_indices]
    best_unanchored = (
        min(unanchored_idx, key=lambda i: candidates[i].final_loss)
        if unanchored_idx else None
    )
    if cfg.prefer_anchored and anchored_idx:
        best_index = min(anchored_idx, key=lambda i: candidates[i].final_loss)
        best = candidates[best_index]
    else:
        best = candidates[comparison.best_index]
        best_index = comparison.best_index
    anchored = len(best.anchor_indices) > 0

    # Anchored and unanchored candidates optimise over different feasible sets,
    # so a single loss threshold across both is not meaningful.  Retention and
    # stability are therefore evaluated separately per kind.
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("unanchored", "anchored"):
        idxs = [i for i, m in enumerate(candidate_meta) if m["kind"] == kind]
        if not idxs:
            continue
        comp = compare_candidates(
            [candidates[i] for i in idxs],
            retain_rtol=cfg.retain_rtol, stability_cos=cfg.stability_cos,
        )
        by_kind[kind] = {
            "indices": idxs,
            "best_index": idxs[comp.best_index],
            "best_loss": candidates[idxs[comp.best_index]].final_loss,
            "losses": [candidates[i].final_loss for i in idxs],
            "relative_gaps": comp.relative_gaps,
            "retained": comp.retained,
            "n_retained": int(sum(comp.retained)),
            "stable": bool(comp.stable),
        }
    selected_kind = candidate_meta[best_index]["kind"] if candidate_meta else None
    kind_stability = by_kind.get(selected_kind, {}).get("stable", comparison.stable)

    labels = [f"template_{k:02d}" for k in range(best.W.shape[1])]
    for k in best.anchor_indices:
        if k < len(anchor_names):
            labels[k] = f"anchor_{anchor_names[k]}"
    stem_set = reconstruct_stems(
        best.W, best.H, X, mixture,
        sample_rate=sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        labels=labels, policy=cfg.policy, power=cfg.mask_power,
    )

    # Model selection diagnostics: held-out loss across template counts.
    train_V = magnitude[:, train_idx]
    hold_V = magnitude[:, hold_idx]
    hold_loss = heldout_loss(
        train_V, best.W, hold_V, beta=cfg.beta, n_iter=200,
        anchors_count=len(best.anchor_indices),
        linalg_backend=cfg.linalg_backend, linalg_dtype=cfg.linalg_dtype,
    ) if hold_idx.size != train_idx.size else float("nan")
    train_loss = best.final_loss

    diagnostics: dict[str, Any] = {
        "sample_rate": sample_rate,
        "n_samples": int(len(mixture)),
        "n_frames": int(magnitude.shape[1]),
        "n_bins": int(magnitude.shape[0]),
        "silent_frames": best.silent_frames,
        "silent_frame_count": len(best.silent_frames),
        "eps": best.eps,
        "best_loss": best.final_loss,
        "train_loss": train_loss,
        "heldout_loss": hold_loss,
        "stop_reason": best.stop_reason,
        "n_iter": best.n_iter,
        "n_parameters": best.n_parameters,
        "loss_history": best.loss_history,
        "candidate_losses": comparison.losses,
        "candidate_gaps": comparison.relative_gaps,
        "candidate_retained": comparison.retained,
        "candidate_meta": candidate_meta,
        "solution_stable": comparison.stable,
        "candidate_kind_stats": by_kind,
        "selected_candidate_index": int(best_index),
        "selected_candidate_kind": selected_kind,
        "selected_kind_stable": bool(kind_stability),
        "selected_kind_retained": int(by_kind.get(selected_kind, {}).get("n_retained", 0)),
        "selected_kind_n_candidates": int(len(by_kind.get(selected_kind, {}).get("indices", []))),
        "best_unanchored_index": best_unanchored,
        "best_unanchored_loss": (
            candidates[best_unanchored].final_loss if best_unanchored is not None else None
        ),
        "best_unanchored_gain_db": (
            [float(20 * np.log10(max(float(np.linalg.norm(h)), 1e-12)))
             for h in candidates[best_unanchored].H]
            if best_unanchored is not None else None
        ),
        "candidate_assignment": comparison.assignment,
        "candidate_similarities": comparison.similarities.tolist(),
        "candidate_gain_ratios": comparison.gain_ratios.tolist(),
        "anchored": anchored,
        "anchor_names": anchor_names,
        "anchor_indices": best.anchor_indices,
        "policy": cfg.policy,
        "mask_sum_error": stem_set.masks_sum_error,
        "component_sum_error": stem_set.component_sum_error,
        "residual_energy_ratio": stem_set.residual_energy_ratio,
        "stem_energy_db": [energy_db(s) for s in stem_set.stems],
        "template_gain_db": [
            float(20 * np.log10(max(float(np.linalg.norm(h)), 1e-12)))
            for h in best.H
        ],
        "template_total_energy_db": [
            float(10 * np.log10(max(float(np.sum((best.W[:, k, None] * best.H[k][None, :]) ** 2)), 1e-12)))
            for k in range(best.W.shape[1])
        ],
        "template_peak_bins": [int(np.argmax(best.W[:, k])) for k in range(best.W.shape[1])],
        "linalg_backend": best.linalg_backend,
        "linalg_dtype": best.linalg_dtype,
    }
    diagnostics["gain_ambiguity_probes"] = gain_ambiguity_probe(
        magnitude, best.W, best.H, beta=cfg.beta, eps=best.eps,
        loss_rtol=cfg.retain_rtol,
        linalg_backend=cfg.linalg_backend, linalg_dtype=cfg.linalg_dtype,
    )
    diagnostics["any_unidentifiable_gain_split"] = any(
        p["unidentifiable_gain_split"] for p in diagnostics["gain_ambiguity_probes"]
    )
    # Most similar template pairs regardless of the probe threshold.  A high
    # cosine below 0.995 is not proof of an unidentifiable gain split, but it is
    # a strong cue that two components may describe the same source and could be
    # grouped by the user (e.g. one source split across two templates).
    Wn = best.W / np.maximum(np.linalg.norm(best.W, axis=0, keepdims=True), 1e-12)
    C = Wn.T @ Wn
    pairs = [
        {"templates": [k, j], "cosine": float(C[k, j]),
         "gains_db": [float(20 * np.log10(max(float(np.linalg.norm(best.H[k])), 1e-12))),
                      float(20 * np.log10(max(float(np.linalg.norm(best.H[j])), 1e-12)))]}
        for k in range(best.W.shape[1]) for j in range(k + 1, best.W.shape[1])
    ]
    pairs.sort(key=lambda p: p["cosine"], reverse=True)
    diagnostics["similar_template_pairs"] = pairs[:5]

    result = PipelineResult(
        config=cfg,
        sample_rate=sample_rate,
        mixture=mixture,
        X=X,
        magnitude=magnitude,
        best=best,
        candidates=candidates,
        comparison=comparison,
        stem_set=stem_set,
        anchored=anchored,
        diagnostics=diagnostics,
    )

    if reference_stems is not None:
        diagnostics["evaluation"] = evaluate_stems(
            stem_set.stems, reference_stems, reference_labels or
            [f"ref_{i}" for i in range(len(reference_stems))],
            sample_rate=sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        )
        # Oracle grouping (ground truth) as an attribution upper bound.
        try:
            oracle_groups = grouping.oracle_grouping(
                np.stack(stem_set.stems, axis=0), reference_stems
            )
            oracle_stems = group_components(
                best.W, best.H, X, mixture, oracle_groups,
                sample_rate=sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                labels=reference_labels or [f"group_{i}" for i in range(len(oracle_groups))],
                policy=cfg.policy, power=cfg.mask_power,
            )
            diagnostics["evaluation"]["oracle_grouping"] = oracle_groups
            diagnostics["evaluation"]["oracle_grouped_sdr"] = [
                sdr(ref, est) for ref, est in zip(reference_stems, oracle_stems.stems)
            ]
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            diagnostics["evaluation"]["oracle_grouping_error"] = repr(exc)
    return result


def apply_grouping(
    result: PipelineResult,
    groups: list[list[int]],
    labels: list[str],
    policy: str | None = None,
) -> StemSet:
    cfg = result.config
    return group_components(
        result.best.W, result.best.H, result.X, result.mixture, groups,
        sample_rate=result.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        labels=labels, policy=policy or cfg.policy, power=cfg.mask_power,
    )


def evaluate_stems(
    estimates: list[np.ndarray],
    references: list[np.ndarray],
    reference_labels: list[str],
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> dict[str, Any]:
    """Ground-truth metrics for a set of component images.

    Reported per estimate and per best-permutation assignment:
    SDR, SI-SDR, spectral convergence, spectrogram MSE, and a leakage matrix
    (projection energy onto each reference).
    """
    n_est = len(estimates)
    n_ref = len(references)
    leak = projection_leakage(references, estimates)
    perm, mean_sdr = _best_perm(estimates, references)
    n_matched = min(len(estimates), len(references))
    matched_sdr = [sdr(references[perm[i]], estimates[i]) for i in range(n_matched)]
    mean_sdr = float(np.mean(matched_sdr)) if matched_sdr else float("nan")
    per_est = []
    for i, est in enumerate(estimates):
        order = np.argsort(leak[i])[::-1]
        matched = i < len(perm)
        ref = references[perm[i]] if matched else None
        per_est.append(
            {
                "estimate_index": i,
                "matched_reference": reference_labels[perm[i]] if matched else None,
                "matched_reference_index": int(perm[i]) if matched else None,
                # With more estimates than references, the surplus estimates are
                # not attributable; scoring them against a zero reference would
                # produce meaningless numbers, so they are left unscored.
                "sdr_db": sdr(ref, est) if matched else None,
                "si_sdr_db": si_sdr(ref, est) if matched else None,
                "spectral_convergence": spectral_convergence(
                    ref, est, n_fft=n_fft, hop_length=hop_length) if matched else None,
                "spectrogram_mse": spectrogram_mse(
                    ref, est, n_fft=n_fft, hop_length=hop_length) if matched else None,
                "leakage_db_to": {
                    reference_labels[j]: float(leak[i, j]) for j in range(n_ref)
                },
                "dominant_leak_reference": reference_labels[int(order[0])] if n_ref else None,
            }
        )
    return {
        "reference_labels": list(reference_labels),
        "permutation": [int(p) for p in perm],
        "n_matched": int(n_matched),
        "n_estimates": int(n_est),
        "mean_sdr_db": float(mean_sdr),
        "mean_sdr_note": "mean over matched estimate/reference pairs only",
        "per_estimate": per_est,
        "leakage_matrix_db": leak.tolist(),
    }


def _best_perm(estimates: list[np.ndarray], references: list[np.ndarray]) -> tuple[list[int], float]:
    from itertools import permutations

    n = min(len(estimates), len(references))
    best_perm = list(range(n))
    best_score = -np.inf
    for p in permutations(range(len(references)), n):
        vals = [sdr(references[p[i]], estimates[i]) for i in range(n)]
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_perm = list(p)
    return best_perm, best_score


def config_summary(cfg: Config) -> dict[str, Any]:
    return {
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "window": "hann(periodic)",
        "n_templates": cfg.n_templates,
        "objective": {1: "generalised KL", 2: "squared Frobenius", 0: "Itakura-Saito"}.get(
            cfg.beta, f"beta={cfg.beta}"),
        "beta": cfg.beta,
        "n_iter_cap": cfg.n_iter,
        "tol": cfg.tol,
        "patience": cfg.patience,
        "min_iter": cfg.min_iter,
        "seeds": list(cfg.seeds),
        "init": cfg.init,
        "retain_rtol": cfg.retain_rtol,
        "stability_cos": cfg.stability_cos,
        "mask_power": cfg.mask_power,
        "policy": cfg.policy,
        "silence_rel": cfg.silence_rel,
        "anchor_source": cfg.anchor_source,
        "heldout_fraction": cfg.heldout_fraction,
        "linalg_backend": cfg.linalg_backend,
        "linalg_dtype": cfg.linalg_dtype,
    }
