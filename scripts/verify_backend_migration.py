"""Backend-migration verification, runnable from a clean directory.

It regenerates its own inputs (a low-rank nonnegative target and a sparse
spectral target), runs the *same* NMF contract on both linear-algebra backends
with fixed identical initialisation, and writes auditable evidence rather than
printing a pass/fail claim:

* objective value per backend and their relative gap (summation-order tolerance);
* column-wise normalised + permutation-aligned component cosines and
  activation correlations;
* reconstruction-space relative error (the actual equivalence criterion);
* per-component stem energies proving the new path emits audible components;
* the two-voice identity-keyed automation pair and a simulated backend column
  swap, demonstrating the fade follows the voice rather than a stale index.

Nothing here reads hand-edited data: every matrix is generated in-process.

Usage
-----
    python3 scripts/verify_backend_migration.py --out <clean_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmfsep.automation import two_voice_automations, apply_automation  # noqa: E402
from nmfsep.equivalence import equivalence_report  # noqa: E402
from nmfsep.linalg import available_backends  # noqa: E402
from nmfsep.masks import reconstruct_stems  # noqa: E402
from nmfsep.nmf import nmf  # noqa: E402
from nmfsep.stft import istft, stft  # noqa: E402


def _low_rank_target(rng, n_bins=120, n_frames=240, rank=3):
    W = rng.uniform(0.2, 1.0, (n_bins, rank))
    H = np.zeros((rank, n_frames))
    block = n_frames // rank
    for k in range(rank):
        H[k, k * block:(k + 1) * block] = rng.uniform(0.5, 1.5, block)
    return W @ H + 0.02


def _sparse_spectral_target(rng, n_bins=120, n_frames=240, rank=4):
    """Sparse spectrum: a few narrow active bins and short activation bursts."""
    W = np.zeros((n_bins, rank))
    for k in range(rank):
        center = int((k + 0.5) * n_bins / rank)
        W[max(center - 2, 0):center + 2, k] = rng.uniform(0.5, 1.0, 4)
    H = np.zeros((rank, n_frames))
    for k in range(rank):
        starts = rng.integers(0, n_frames - 12, size=4)
        for s in starts:
            H[k, s:s + 8] = rng.uniform(0.4, 1.0, 8)
    return W @ H + 0.01


def _component_errors(V, result):
    """Per-component reconstruction contribution error (Frobenius, relative)."""
    errs = []
    for k in range(result.W.shape[1]):
        part = np.outer(result.W[:, k], result.H[k, :])
        errs.append(float(np.linalg.norm(part) / max(np.linalg.norm(V), 1e-12)))
    return errs


def _audible_stems(V_shape_signal, result, sr=22050):
    n = int(sr * 2)
    rng = np.random.default_rng(9)
    x = rng.uniform(-0.4, 0.4, n)
    X = stft(x, n_fft=512, hop_length=128)
    mag = np.abs(X)
    fitted = nmf(mag, result.W.shape[1], init="random", seed=42, n_iter=80,
                 patience=200, tol=0.0, min_iter=80,
                 linalg_backend=result.linalg_backend)
    ss = reconstruct_stems(fitted.W, fitted.H, X, x, sample_rate=sr,
                           n_fft=512, hop_length=128)
    return {
        "stem_energy": [float(np.sum(s**2)) for s in ss.stems],
        "audible": all(float(np.sum(s**2)) > 1e-9 for s in ss.stems),
        "component_sum_error": ss.component_sum_error,
        "backend": result.linalg_backend,
    }


def verify_target(name, V, rank, out_dir):
    common = dict(init="random", seed=42, n_iter=300, patience=300,
                  tol=0.0, min_iter=300)
    t0 = time.time()
    r_np = nmf(V, rank, linalg_backend="numpy", **common)
    t1 = time.time()
    r_sp = nmf(V, rank, linalg_backend="scipy", **common)
    t2 = time.time()
    rep = equivalence_report(V, r_np, r_sp)

    # Simulate a backend-induced column swap and prove identity-keyed
    # automation still routes the fade to the same voice.
    perm = list(range(r_sp.W.shape[1]))
    if len(perm) >= 2:
        perm[0], perm[1] = perm[1], perm[0]
    Wp, Hp = r_sp.W[:, perm], r_sp.H[perm, :]
    curves = two_voice_automations(r_sp.W, r_sp.H.shape[1],
                                   indices=(0, min(1, rank - 1)))
    applied = apply_automation(curves[0], Wp, Hp)
    swap_check = {
        "applied_permutation": perm,
        "reconstruction_unchanged": bool(np.allclose(r_sp.W @ r_sp.H, Wp @ Hp)),
        "fade_followed_voice_to_index": applied["index"],
        "naive_index_would_be": 0,
        "routed_correctly": applied["index"] == perm[0],
    }

    evidence = {
        "target": name,
        "shape": list(V.shape),
        "rank": rank,
        "backends": available_backends(),
        "objective": {
            "numpy_loss": r_np.final_loss,
            "scipy_loss": r_sp.final_loss,
            "relative_gap": rep.loss_relative_gap,
            "stop_reason": {"numpy": r_np.stop_reason, "scipy": r_sp.stop_reason},
            "n_iter": {"numpy": r_np.n_iter, "scipy": r_sp.n_iter},
        },
        "component_evidence": {
            "permutation_ref_to_scipy": rep.permutation,
            "aligned_column_cosines": rep.column_cosines,
            "min_column_cosine": rep.min_column_cosine,
            "aligned_activation_correlation": rep.activation_correlation,
            "min_activation_correlation": rep.min_activation_correlation,
            "reconstruction_relative_error": rep.reconstruction_relative_error,
            "numpy_component_energy_ratio": _component_errors(V, r_np),
            "scipy_component_energy_ratio": _component_errors(V, r_sp),
        },
        "equivalence": {
            "equivalent": rep.equivalent,
            "tolerances": rep.tolerances,
            "reasons": rep.reasons,
        },
        "audible_output": {
            "numpy": _audible_stems(V.shape, r_np),
            "scipy": _audible_stems(V.shape, r_sp),
        },
        "automation_swap_check": swap_check,
        "runtime_s": {"numpy": round(t1 - t0, 3), "scipy": round(t2 - t1, 3)},
    }
    with open(out_dir / f"{name}_evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
    np.save(out_dir / f"{name}_W_numpy.npy", r_np.W)
    np.save(out_dir / f"{name}_W_scipy.npy", r_sp.W)
    return evidence


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="clean output directory (created fresh)")
    args = p.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        print(f"refusing to write into non-empty directory {out_dir}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(2026)
    summary = []
    for name, maker, rank in (
        ("low_rank", _low_rank_target, 3),
        ("sparse_spectrum", _sparse_spectral_target, 4),
    ):
        V = maker(rng)
        ev = verify_target(name, V, rank, out_dir)
        summary.append({
            "target": name,
            "equivalent": ev["equivalence"]["equivalent"],
            "loss_gap": ev["objective"]["relative_gap"],
            "reconstruction_rel_err": ev["component_evidence"]["reconstruction_relative_error"],
            "scipy_audible": ev["audible_output"]["scipy"]["audible"],
            "automation_routed_correctly": ev["automation_swap_check"]["routed_correctly"],
        })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"results": summary}, f, indent=2, sort_keys=True)
    print(json.dumps({"results": summary}, indent=2))
    ok = all(
        row["equivalent"] and row["scipy_audible"] and row["automation_routed_correctly"]
        for row in summary
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
