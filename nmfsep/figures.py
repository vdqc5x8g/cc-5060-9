"""Static figures for the report, generated from saved run artifacts."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stft import stft  # noqa: E402


def _load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "diagnostics.json"), encoding="utf-8") as f:
        diag = json.load(f)
    return {
        "diag": diag,
        "W": np.load(os.path.join(run_dir, "templates.npy")),
        "H": np.load(os.path.join(run_dir, "activations.npy")),
        "config": json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8")),
        "dir": run_dir,
    }


def _dim(x):
    return 10 * np.log10(np.maximum(x, np.max(x) * 1e-5 + 1e-12))


def _norm_db(col: np.ndarray, floor_db: float = -45.0) -> np.ndarray:
    """dB relative to a column's own maximum, floored (for shape comparison)."""
    peak = float(np.max(col)) or 1.0
    db = 20 * np.log10(np.maximum(col, 1e-12) / peak)
    return np.maximum(db, floor_db)


def generate_for_run(
    run_dir: str,
    *,
    truth_npz: str | None = None,
    title: str | None = None,
    out_dir: str | None = None,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> list[str]:
    """Generate the standard figure set for one run and record a manifest.

    The manifest (``figures.json`` inside the run directory) is what the UI
    reads, so figures from similarly-named runs can never be mixed up.
    """
    import json

    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    n_fft = int(cfg.get("n_fft", n_fft))
    hop_length = int(cfg.get("hop_length", hop_length))
    diag = json.load(open(os.path.join(run_dir, "diagnostics.json"), encoding="utf-8"))
    sr = int(diag.get("sample_rate", sr))
    slug = os.path.basename(os.path.abspath(run_dir))
    prefix = slug.replace(" ", "_")
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "artifacts", "figures")
    os.makedirs(out_dir, exist_ok=True)
    names = []
    names.append(os.path.basename(plot_templates_and_activations(
        run_dir, os.path.join(out_dir, f"{prefix}_templates.png"),
        title=title or slug, sr=sr, n_fft=n_fft, hop_length=hop_length)))
    names.append(os.path.basename(plot_loss_curves(
        run_dir, os.path.join(out_dir, f"{prefix}_loss.png"))))
    names.append(os.path.basename(plot_source_reconstruction(
        run_dir, truth_npz, os.path.join(out_dir, f"{prefix}_spectrograms.png"),
        sr=sr, n_fft=n_fft, hop_length=hop_length)))
    leak = plot_leakage(run_dir, os.path.join(out_dir, f"{prefix}_leakage.png"))
    if leak:
        names.append(os.path.basename(leak))
    with open(os.path.join(run_dir, "figures.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)
    return names


def plot_templates_and_activations(run_dir: str, out: str, sr: int = 22050,
                                   n_fft: int = 1024, hop_length: int = 256,
                                   title: str | None = None) -> str:
    r = _load_run(run_dir)
    W, H = r["W"], r["H"]
    n = W.shape[1]
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    times = np.arange(H.shape[1]) * hop_length / sr
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n]
    for k in range(n):
        axes[0].semilogx(freqs[1:], _norm_db(W[1:, k]), color=colors[k],
                         lw=1.3, label=f"template {k} (peak {freqs[int(np.argmax(W[:, k]))]:.0f} Hz)")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("dB relative to template peak")
    axes[0].set_title((title or run_dir) + " — spectral templates W (shapes only; no source names)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_ylim(-45, 3)
    for k in range(n):
        g = float(np.linalg.norm(H[k]))
        axes[1].plot(times, H[k] / max(g, 1e-12), color=colors[k], lw=1.0,
                     label=f"template {k} gain {20 * np.log10(max(g, 1e-12)):.1f} dB")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("normalised activation")
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_loss_curves(run_dir: str, out: str) -> str:
    r = _load_run(run_dir)
    cand_dir = os.path.join(run_dir, "candidates")
    files = sorted(f for f in os.listdir(cand_dir) if f.endswith(".json"))
    fig, ax = plt.subplots(figsize=(9, 5))
    for f in files:
        info = json.load(open(os.path.join(cand_dir, f), encoding="utf-8"))
        meta = info.get("meta", {})
        label = f"{f.split('_')[1].split('.')[0]}: {meta.get('kind','?')}/{meta.get('init','?')}/s{meta.get('seed','?')}"
        keep = meta.get("kind") == "anchored"
        ax.plot(info["loss_history"], lw=1.1, ls="--" if keep else "-",
                label=f"{label} ({info['loss']:.4g})")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective (generalised KL)")
    ax.set_title("Candidate trajectories (dashed = anchored)")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_source_reconstruction(run_dir: str, reference_npz: str | None, out: str,
                               sr: int = 22050, n_fft: int = 1024,
                               hop_length: int = 256) -> str:
    """Mixture + component spectrograms, with ground truth if available."""
    r = _load_run(run_dir)
    W, H = r["W"], r["H"]
    stems_dir = os.path.join(run_dir, "stems")
    from .audio import load_audio

    stem_files = sorted(os.listdir(stems_dir))
    stems = [load_audio(os.path.join(stems_dir, f))[0] for f in stem_files]
    mix, _ = load_audio(os.path.join(run_dir, "mixture.wav"))
    n_plot = len(stems) + 1
    has_truth = reference_npz and os.path.exists(reference_npz)
    if has_truth:
        data = np.load(reference_npz)
        refs = data["stems"]
        labels = [str(x) for x in data["labels"]]
        n_plot += len(refs)
    fig, axes = plt.subplots(n_plot, 1, figsize=(11, 2.0 * n_plot))
    def show(ax, sig, name):
        S = np.abs(stft(sig, n_fft, hop_length))
        ax.imshow(_dim(S), origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(sig) / sr, 0, sr / 2])
        ax.set_ylabel("Hz")
        ax.set_title(name, fontsize=9, loc="left")
    show(axes[0], mix, "mixture")
    for i, s in enumerate(stems):
        show(axes[i + 1], s, f"component: {stem_files[i]}  (no source name assigned)")
    if has_truth:
        for j in range(len(refs)):
            show(axes[len(stems) + 1 + j], refs[j], f"ground truth: {labels[j]} (evaluation only)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_leakage(run_dir: str, out: str) -> str | None:
    ev_path = os.path.join(run_dir, "evaluation.json")
    if not os.path.exists(ev_path):
        return None
    ev = json.load(open(ev_path, encoding="utf-8"))
    L = np.array(ev["leakage_matrix_db"])
    labels = ev["reference_labels"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(L, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(L.shape[0]), [f"est {i}" for i in range(L.shape[0])])
    ax.set_title("Projection energy of each estimate on each reference (dB)")
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L[i, j]:.1f}", ha="center", va="center",
                    color="white" if L[i, j] < L.max() - 10 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_unknown_sweep(sweep_json: str, out: str) -> str:
    """K-sweep diagnostics for a recording with no ground truth.

    Without references there is no SDR dimension.  Instead this shows that the
    fit keeps improving with K (over-decomposition is always rewarded in
    training), how held-out loss behaves, and whether candidate stability and
    gain identifiability survive.  Those are the failure signals a user has to
    read instead of accuracy.
    """
    rows = json.load(open(sweep_json, encoding="utf-8"))
    ks = [r["n_templates"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    axes[0].plot(ks, [r["best_loss"] for r in rows], "o-", label="train loss")
    axes[0].plot(ks, [r["heldout_loss"] for r in rows], "s--", label="held-out loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("objective")
    axes[0].legend()
    axes[0].set_title("No ground truth: fit improves with K")

    axes[1].plot(ks, [r["retained"] for r in rows], "o-", label="candidates retained")
    axes[1].plot(ks, [r["candidates"] for r in rows], "s--", alpha=0.6, label="candidates run")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].set_title("Retained within 1% loss threshold")

    amb = [1 if r.get("gain_ambiguity") else 0 for r in rows]
    axes[2].scatter(ks, amb, s=70,
                    color=["#ef7b7b" if a else "#74d99f" for a in amb], zorder=3)
    axes[2].set_yticks([0, 1], ["identifiable", "unidentifiable"])
    axes[2].set_ylim(-0.3, 1.3)
    # Stability is the other failure signal: mark K where candidates disagree.
    for k, r in zip(ks, rows):
        if not r.get("stable", True):
            axes[1].scatter([k], [r["retained"]], s=140, facecolors="none",
                            edgecolors="#f0b46c", linewidths=2, zorder=4)
    axes[2].set_title("Gain-split probe")
    for ax in axes:
        ax.set_xlabel("templates K")
        ax.grid(alpha=0.3)
    fig.suptitle("Unknown recording: no accuracy metric exists; read stability and identifiability")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_sweep(sweep_json: str, out: str) -> str:
    rows = json.load(open(sweep_json, encoding="utf-8"))
    ks = [r["n_templates"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(ks, [r["best_loss"] for r in rows], "o-", label="train loss")
    axes[0].plot(ks, [r["heldout_loss"] for r in rows], "s--", label="held-out loss")
    axes[0].set_xlabel("templates K")
    axes[0].set_ylabel("objective")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("Fit vs K (held-out frames)")
    sdr_key = "oracle_grouped_mean_sdr"
    if sdr_key in rows[0] and rows[0][sdr_key] is not None:
        axes[1].plot(ks, [r.get("mean_sdr_db") for r in rows], "o-", label="mean SDR (ungrouped)")
        axes[1].plot(ks, [r.get(sdr_key) for r in rows], "s--", label="oracle-grouped mean SDR")
        axes[1].set_ylabel("dB")
        axes[1].legend()
        axes[1].set_title("Separation vs K (ground-truth eval)")
    axes[2].plot(ks, [r["params_per_frame"] for r in rows], "o-")
    axes[2].set_ylabel("parameters / frame")
    axes[2].set_title("Model size")
    for ax in axes:
        ax.set_xlabel("templates K")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
