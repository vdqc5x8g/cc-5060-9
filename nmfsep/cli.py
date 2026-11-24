"""Command-line entry point.

Subcommands
-----------
``synth``   evaluate on a local mixture with a known ground truth (a .npz with
            ``mixture``, ``sr`` and ``stems``); used for the synthetic benchmark.
``demo``    build the shipped synthetic mixtures, evaluate both, and write the
            full report bundle used by the UI.
``unknown`` run the pipeline on a recording with no ground truth (an external
            audio file), producing the failure-case report.
``sweep``   run the same mixture for a range of template counts and write a
            comparison table (the "template count must be comparable" check).

All runs write into ``runs/<run_id>/``:
    config.json, diagnostics.json, candidate_*.json, stems/*.wav,
    components/*.wav, activation_*.npy, templates_*.npy, report.json
and, for evaluated runs, ``evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from . import __version__, grouping
from .audio import load_audio, save_audio
from .linalg import available_backends
from .metrics import sdr
from .nmf import nmf
from .pipeline import (
    Config,
    PipelineResult,
    apply_grouping,
    config_summary,
    evaluate_stems,
    run_pipeline,
)
from .stft import istft, stft
from .synth import make_ambiguous, make_separable


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    return obj


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def _save_run(
    outdir: str,
    result: PipelineResult,
    *,
    mixture_name: str,
    anchors_infos: list[dict] | None = None,
    ground_truth_used_for_metrics: bool = False,
) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cfg = result.config
    diag = dict(result.diagnostics)
    diag["mixture_name"] = mixture_name
    diag["anchors"] = anchors_infos or []
    diag["ground_truth_used_for_metrics"] = ground_truth_used_for_metrics
    diag["nmfsep_version"] = __version__
    diag["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Audio stems (per template) and grouped stems if present.
    for i, stem in enumerate(result.stem_set.stems):
        save_audio(os.path.join(outdir, "stems", f"{result.stem_set.labels[i]}.wav"),
                   stem, result.sample_rate)
    save_audio(os.path.join(outdir, "mixture.wav"), result.mixture, result.sample_rate)
    np.save(os.path.join(outdir, "mixture.npy"), result.mixture)
    np.save(os.path.join(outdir, "activations.npy"), result.best.H)
    np.save(os.path.join(outdir, "templates.npy"), result.best.W)
    write_json(os.path.join(outdir, "config.json"), config_summary(cfg))
    write_json(os.path.join(outdir, "diagnostics.json"), diag)
    for ci, cand in enumerate(result.candidates):
        write_json(
            os.path.join(outdir, "candidates", f"candidate_{ci:02d}.json"),
            {
                "meta": diag["candidate_meta"][ci] if ci < len(diag.get("candidate_meta", []))
                        else {},
                "loss": cand.final_loss,
                "loss_history": cand.loss_history,
                "n_iter": cand.n_iter,
                "stop_reason": cand.stop_reason,
                "eps": cand.eps,
                "silent_frames": cand.silent_frames,
                "anchor_indices": cand.anchor_indices,
                "template_peak_bins": [int(np.argmax(cand.W[:, k])) for k in range(cand.W.shape[1])],
                "template_gain_db": [
                    float(20 * np.log10(max(float(np.linalg.norm(h)), 1e-12))) for h in cand.H
                ],
            },
        )
        np.save(os.path.join(outdir, "candidates", f"candidate_{ci:02d}_W.npy"), cand.W)
        np.save(os.path.join(outdir, "candidates", f"candidate_{ci:02d}_H.npy"), cand.H)

    if "evaluation" in diag:
        write_json(os.path.join(outdir, "evaluation.json"), diag["evaluation"])

    write_json(os.path.join(outdir, "diagnostics.json"), diag)
    return diag


def _save_npz(path: str, mixture, sr, tracks) -> None:
    np.savez_compressed(
        path,
        mixture=mixture,
        sr=sr,
        stems=np.stack([t.signal for t in tracks], axis=0),
        labels=np.array([t.name for t in tracks]),
        descriptors=np.array([json.dumps(t.descriptor) for t in tracks]),
    )


def cmd_demo(args) -> int:
    outroot = args.out or "runs/demo"
    os.makedirs(outroot, exist_ok=True)
    summary = {}
    cfg = Config(n_templates=args.templates, n_iter=args.iters, seeds=tuple(range(args.seeds)),
                 linalg_backend=args.backend, linalg_dtype=args.precision)
    for name, maker, K in (
        ("separable", make_separable, 3),
        ("ambiguous", make_ambiguous, 2),
    ):
        mix, tracks = maker(8.0)
        outdir = os.path.join(outroot, name)
        _save_npz(os.path.join(outroot, f"{name}_ground_truth.npz"), mix,
                  int(22050), tracks)
        c = Config(**{**asdict(cfg), "n_templates": K})
        t0 = time.time()
        res = run_pipeline(
            mix, 22050, c,
            reference_stems=[t.signal for t in tracks],
            reference_labels=[t.name for t in tracks],
        )
        diag = _save_run(outdir, res, mixture_name=name,
                         ground_truth_used_for_metrics=True)
        text = (
            f"{name}: K={K} best_loss={diag['best_loss']:.2f} "
            f"stable={diag['selected_kind_stable']} "
            f"retained={diag['selected_kind_retained']}/{diag['selected_kind_n_candidates']} "
            f"mean_SDR={diag['evaluation']['mean_sdr_db']:.2f} dB "
            f"oracle_SDR={np.round(diag['evaluation'].get('oracle_grouped_sdr', []), 2).tolist()} "
            f"mask_err={diag['mask_sum_error']:.2e} ({time.time() - t0:.1f}s)"
        )
        print(text)
        summary[name] = diag

    # Anchored variant on the separable mixture.  Anchors come from isolated
    # *synthetic* stems, which stand in for an observable calibration source.
    # This is stated explicitly in the report; it proves the anchor mechanism,
    # not that anchors are available for real recordings.
    mix, tracks = make_separable(8.0)
    refs = [t.signal for t in tracks]
    anchor_keys = ("low_harmonic", "band_noise")
    anchors = {t.name: t.signal for t in tracks if t.name in anchor_keys}
    c = Config(**{**asdict(cfg), "n_templates": 4})
    t0 = time.time()
    res = run_pipeline(
        mix, 22050, c, anchors=anchors,
        reference_stems=refs, reference_labels=[t.name for t in tracks],
    )
    diag = _save_run(
        os.path.join(outroot, "separable_anchored"), res, mixture_name="separable_anchored",
        anchors_infos=[{"name": k, "origin": "isolated synthetic stem",
                        "note": "stand-in for an observable calibration source"} for k in anchors],
        ground_truth_used_for_metrics=True,
    )
    print(
        f"separable_anchored: anchored={diag['anchored']} anchors={diag['anchor_names']} "
        f"best_loss={diag['best_loss']:.2f} oracle_SDR="
        f"{np.round(diag['evaluation'].get('oracle_grouped_sdr', []), 2).tolist()} "
        f"({time.time() - t0:.1f}s)"
    )
    summary["separable_anchored"] = diag

    # Unidentifiable-gain demo: anchor the two magnitude-identical twins.  Any
    # gain split between them is consistent with the mixture, so the anchors
    # stay fixed but the reported per-anchor gains are one representative.
    mix2, tracks2 = make_ambiguous(8.0)
    refs2 = [t.signal for t in tracks2]
    anchors2 = {"twin_A": refs2[0], "twin_B": refs2[1]}
    c2 = Config(**{**asdict(cfg), "n_templates": 3})
    t0 = time.time()
    res2 = run_pipeline(
        mix2, 22050, c2, anchors=anchors2,
        reference_stems=refs2, reference_labels=[t.name for t in tracks2],
    )
    diag2 = _save_run(
        os.path.join(outroot, "ambiguous_twins_anchored"), res2,
        mixture_name="ambiguous_twins_anchored",
        anchors_infos=[{"name": k, "origin": "isolated synthetic stem",
                        "note": "twins have identical magnitude spectra by construction"} for k in anchors2],
        ground_truth_used_for_metrics=True,
    )
    print(
        f"ambiguous_twins_anchored: unidentifiable_gain_split="
        f"{diag2['any_unidentifiable_gain_split']} gains_db="
        f"{np.round(diag2['template_gain_db'], 2).tolist()} ({time.time() - t0:.1f}s)"
    )
    summary["ambiguous_twins_anchored"] = diag2

    write_json(os.path.join(outroot, "summary.json"), {
        k: {
            "best_loss": v["best_loss"],
            "heldout_loss": v["heldout_loss"],
            "solution_stable": v["solution_stable"],
            "selected_candidate_kind": v.get("selected_candidate_kind"),
            "selected_kind_stable": v.get("selected_kind_stable"),
            "selected_kind_retained": v.get("selected_kind_retained"),
            "retained": v["candidate_retained"],
            "mean_sdr_db": v["evaluation"]["mean_sdr_db"],
            "oracle_grouped_sdr": v["evaluation"].get("oracle_grouped_sdr"),
            "gain_ambiguity": v["gain_ambiguity_probes"],
            "mask_sum_error": v["mask_sum_error"],
            "component_sum_error": v["component_sum_error"],
        } for k, v in summary.items()
    })
    return 0


def cmd_unknown(args) -> int:
    outdir = args.out or "runs/unknown"
    offset = args.offset
    x, sr = load_audio(args.audio, mono=True, offset=offset, duration=args.duration,
                       target_sr=args.sample_rate)
    if args.max_seconds and len(x) > args.max_seconds * sr:
        x = x[: int(args.max_seconds * sr)]
    cfg = Config(n_templates=args.templates, n_iter=args.iters,
                 seeds=tuple(range(args.seeds)),
                 linalg_backend=args.backend, linalg_dtype=args.precision)
    res = run_pipeline(x, sr, cfg)
    diag = _save_run(outdir, res, mixture_name=os.path.basename(args.audio),
                     ground_truth_used_for_metrics=False)
    print(
        f"unknown: K={args.templates} best_loss={diag['best_loss']:.2f} "
        f"stable={diag['selected_kind_stable']} "
        f"retained={diag['selected_kind_retained']}/{diag['selected_kind_n_candidates']} "
        f"mask_err={diag['mask_sum_error']:.2e} "
        f"gain_ambiguity={diag['any_unidentifiable_gain_split']} "
        f"heldout={diag['heldout_loss']:.2f}"
    )
    return 0


def cmd_synth(args) -> int:
    outdir = args.out or "runs/synth"
    mix, sr = load_audio(args.mixture, mono=True, target_sr=args.sample_rate)
    data = np.load(args.truth)
    stems = data["stems"]
    labels = [str(s) for s in data["labels"]]
    cfg = Config(n_templates=args.templates, n_iter=args.iters,
                 seeds=tuple(range(args.seeds)),
                 linalg_backend=args.backend, linalg_dtype=args.precision)
    res = run_pipeline(
        mix, sr, cfg,
        reference_stems=[stems[i] for i in range(len(stems))],
        reference_labels=labels,
    )
    diag = _save_run(outdir, res, mixture_name=os.path.basename(args.mixture),
                     ground_truth_used_for_metrics=True)
    print(json.dumps({
        "mean_sdr_db": diag["evaluation"]["mean_sdr_db"],
        "oracle_grouped_sdr": diag["evaluation"].get("oracle_grouped_sdr"),
        "solution_stable": diag["solution_stable"],
        "retained": diag["candidate_retained"],
        "mask_sum_error": diag["mask_sum_error"],
    }, indent=2))
    return 0


def cmd_sweep(args) -> int:
    outroot = args.out or "runs/sweep"
    os.makedirs(outroot, exist_ok=True)
    if args.audio:
        mix, sr = load_audio(args.audio, mono=True, target_sr=args.sample_rate,
                             offset=args.offset, duration=args.duration)
        refs, labels = None, None
    else:
        mix, tracks = make_separable(8.0)
        sr = 22050
        refs = [t.signal for t in tracks]
        labels = [t.name for t in tracks]
    rows = []
    for K in range(args.min_templates, args.max_templates + 1):
        cfg = Config(n_templates=K, n_iter=args.iters, seeds=tuple(range(args.seeds)),
                     linalg_backend=args.backend, linalg_dtype=args.precision)
        t0 = time.time()
        res = run_pipeline(mix, sr, cfg, reference_stems=refs, reference_labels=labels)
        d = res.diagnostics
        row = {
            "n_templates": K,
            "best_loss": d["best_loss"],
            "heldout_loss": d["heldout_loss"],
            "per_template_loss": d["best_loss"] / K,
            "params": d["n_parameters"],
            "params_per_frame": d["n_parameters"] / max(d["n_frames"], 1),
            "stable": d["solution_stable"],
            "retained": sum(d["candidate_retained"]),
            "candidates": len(d["candidate_retained"]),
            "mask_sum_error": d["mask_sum_error"],
            "component_sum_error": d["component_sum_error"],
            "gain_ambiguity": d["any_unidentifiable_gain_split"],
            "stop_reason": d["stop_reason"],
            "iters": d["n_iter"],
            "elapsed_s": round(time.time() - t0, 2),
        }
        if "evaluation" in d:
            row["mean_sdr_db"] = d["evaluation"]["mean_sdr_db"]
            row["oracle_grouped_sdr"] = d["evaluation"].get("oracle_grouped_sdr")
            row["oracle_grouped_mean_sdr"] = (
                float(np.mean(row["oracle_grouped_sdr"]))
                if row["oracle_grouped_sdr"] else None
            )
        rows.append(row)
        print({k: row[k] for k in ("n_templates", "best_loss", "heldout_loss",
                                   "mean_sdr_db", "oracle_grouped_mean_sdr")
               if k in row})
    write_json(os.path.join(outroot, "sweep.json"), rows)
    _sweep_csv(os.path.join(outroot, "sweep.csv"), rows)
    return 0


def _sweep_csv(path: str, rows: list[dict]) -> None:
    keys = ["n_templates", "best_loss", "heldout_loss", "per_template_loss", "params",
            "params_per_frame", "mean_sdr_db", "oracle_grouped_mean_sdr", "stable",
            "retained", "candidates", "gain_ambiguity", "stop_reason", "iters",
            "mask_sum_error", "component_sum_error", "elapsed_s"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(k)) for k in keys) + "\n")


def _csv_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def cmd_nmf(args) -> int:
    """Auxiliary single NMF solve (the graphical window is the main entry).

    Runs one factorisation on an audio file or a synthetic mixture with the
    chosen linear-algebra backend, writes W/H and per-component stems, and
    records the backend identity plus the objective/component diagnostics so a
    backend migration can be audited from the command line.
    """
    from .masks import reconstruct_stems

    outdir = args.out or "runs/nmf"
    if args.audio:
        x, sr = load_audio(args.audio, mono=True, offset=args.offset or 0.0,
                           duration=args.duration, target_sr=args.sample_rate)
        mixture_name = os.path.basename(args.audio)
    else:
        mix, tracks = make_separable(8.0)
        x, sr = mix, 22050
        mixture_name = "synthetic_separable"
    X = stft(x, n_fft=1024, hop_length=256)
    V = np.abs(X)
    t0 = time.time()
    res = nmf(
        V, args.templates, beta=1.0, n_iter=args.iters, init="nndsvda",
        seed=0, linalg_backend=args.backend, linalg_dtype=args.precision,
    )
    ss = reconstruct_stems(
        res.W, res.H, X, x, sample_rate=sr, n_fft=1024, hop_length=256,
        labels=[f"component_{k:02d}" for k in range(res.W.shape[1])],
    )
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "templates.npy"), res.W)
    np.save(os.path.join(outdir, "activations.npy"), res.H)
    np.save(os.path.join(outdir, "mixture.npy"), x)
    for i, stem in enumerate(ss.stems):
        save_audio(os.path.join(outdir, "stems", f"{ss.labels[i]}.wav"), stem, sr)
    report = {
        "mixture_name": mixture_name,
        "nmfsep_version": __version__,
        "linalg_backend": res.linalg_backend,
        "linalg_dtype": res.linalg_dtype,
        "available_backends": available_backends(),
        "n_iter": res.n_iter,
        "stop_reason": res.stop_reason,
        "final_loss": res.final_loss,
        "component_energy_db": [float(10 * np.log10(max(float(np.sum(s**2)), 1e-12)))
                                for s in ss.stems],
        "component_sum_error": ss.component_sum_error,
        "elapsed_s": round(time.time() - t0, 3),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(os.path.join(outdir, "nmf_report.json"), report)
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nmfsep", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("demo", "unknown", "synth", "sweep"):
        sp = sub.add_parser(name)
        sp.add_argument("--templates", type=int, default=4)
        sp.add_argument("--iters", type=int, default=600)
        sp.add_argument("--seeds", type=int, default=5)
        sp.add_argument("--out", default=None)
        sp.add_argument("--sample-rate", type=int, default=22050)
        sp.add_argument("--backend", choices=available_backends(), default="numpy",
                        help="linear-algebra backend for the NMF matrix solves")
        sp.add_argument("--precision", choices=("float64", "float32"), default="float64",
                        help="working precision of the backend drivers")
        if name in ("unknown", "sweep"):
            sp.add_argument("--audio", default=None)
            sp.add_argument("--offset", type=float, default=0.0)
            sp.add_argument("--duration", type=float, default=None)
        if name == "unknown":
            sp.add_argument("--max-seconds", type=float, default=None)
        if name == "synth":
            sp.add_argument("--mixture", required=True)
            sp.add_argument("--truth", required=True)
        if name == "sweep":
            sp.add_argument("--min-templates", type=int, default=2)
            sp.add_argument("--max-templates", type=int, default=8)

    nmf_parser = sub.add_parser(
        "nmf",
        help="auxiliary single NMF solve (the inspection window is the main entry)",
    )
    nmf_parser.add_argument("--audio", default=None)
    nmf_parser.add_argument("--offset", type=float, default=0.0)
    nmf_parser.add_argument("--duration", type=float, default=None)
    nmf_parser.add_argument("--templates", type=int, default=3)
    nmf_parser.add_argument("--iters", type=int, default=400)
    nmf_parser.add_argument("--out", default=None)
    nmf_parser.add_argument("--sample-rate", type=int, default=22050)
    nmf_parser.add_argument("--backend", choices=available_backends(), default="numpy")
    nmf_parser.add_argument("--precision", choices=("float64", "float32"), default="float64")

    args = p.parse_args(argv)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "unknown":
        if not args.audio:
            p.error("unknown requires --audio")
        return cmd_unknown(args)
    if args.cmd == "synth":
        return cmd_synth(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "nmf":
        return cmd_nmf(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
