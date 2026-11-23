"""Inspection UI server (stdlib only).

Endpoints
---------
GET  /                          HTML page
GET  /api/runs                  available runs (from ``runs/``)
GET  /api/run/<id>              diagnostics, templates, activations, stems
GET  /api/run/<id>/audio/<name> WAV bytes (mixture, stems, grouped stems)
POST /api/run/<id>/group        body: {"groups": [[0,1],[2]], "labels": [...],
                                       "policy": "wiener"|"residual"|..., "power": 1.0}
                                returns grouped stems written to disk + metrics
POST /api/run/<id>/solo         body: {"stems": [0,2], "gains_db": {...}}
                                renders a mixed WAV from selected stems

The server never names a component by an instrument.  All grouping is explicit
user input or the labelled clustering rules from ``nmfsep.grouping``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .audio import load_audio, save_audio
from .masks import group_components
from .stft import stft

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
UI_DIR = ROOT / "ui"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


def _figure_prefix(run_id: str) -> str:
    # runs/demo/separable -> "separable_"; runs/unknown/foo -> "foo_"
    return run_id.rstrip("/").rsplit("/", 1)[-1] + "_"


def _run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise ValueError("invalid run id")
    d = (RUNS / run_id).resolve()
    if RUNS.resolve() not in d.parents and d != RUNS.resolve():
        raise ValueError("run outside runs directory")
    if not d.is_dir():
        raise FileNotFoundError(run_id)
    return d


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_runs() -> list[dict]:
    out = []
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.rglob("diagnostics.json")):
        run_dir = d.parent
        rel = run_dir.relative_to(RUNS).as_posix()
        diag = _read_json(d, {})
        out.append({
            "id": rel,
            "name": diag.get("mixture_name", rel),
            "sample_rate": diag.get("sample_rate"),
            "n_templates": len(diag.get("template_peak_bins", []) or []),
            "policy": diag.get("policy"),
            "stable": diag.get("solution_stable"),
            "anchored": diag.get("anchored"),
            "has_evaluation": (run_dir / "evaluation.json").exists(),
            "has_ground_truth": any(diag.get("mixture_name", "").startswith(n)
                                    for n in ("separable", "ambiguous")),
        })
    return out


def _load_mixture(d: Path, sr: int) -> np.ndarray:
    """Prefer the lossless float mixture saved by the CLI over the 16-bit WAV."""
    npy = d / "mixture.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    mixture, _ = load_audio(str(d / "mixture.wav"), mono=True, target_sr=sr)
    return mixture


def _template_shapes(W: np.ndarray, max_bins: int = 256, floor_db: float = -45.0) -> list[dict]:
    """Downsample each template to <= max_bins log-spaced bands for the UI.

    Values are dB relative to the template's own peak, floored; the UI shows
    shapes only, because the absolute scale lives in the activation gain.
    """
    n_bins = W.shape[0]
    edges = np.unique(np.round(np.geomspace(1, n_bins - 1, num=min(max_bins, n_bins - 1) + 1)).astype(int))
    edges = np.unique(np.concatenate([[0], edges]))
    out = []
    for k in range(W.shape[1]):
        col = W[:, k]
        vals = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = float(col[a:b].mean()) if b > a else float(col[a])
            vals.append(v)
        peak = max(vals) if vals else 1.0
        db = [round(max(float(20 * np.log10(max(v, 1e-12) / max(peak, 1e-12))), floor_db), 2)
              for v in vals]
        out.append({"bin_edges": [int(e) for e in edges], "db": db})
    return out


def _activation_summary(H: np.ndarray) -> dict:
    """Frame-resolution, min/max-normalised activation matrix for plotting."""
    n = H.shape[0]
    out = []
    for k in range(n):
        row = H[k]
        lo, hi = float(row.min()), float(row.max())
        norm = (row - lo) / (hi - lo) if hi > lo else np.zeros_like(row)
        out.append([round(float(v), 5) for v in norm])
    return {"n_frames": int(H.shape[1]), "rows": out,
            "gains_db": [float(20 * np.log10(max(float(np.linalg.norm(h)), 1e-12)))
                         for h in H]}


def run_payload(run_id: str) -> dict:
    d = _run_dir(run_id)
    diag = _read_json(d / "diagnostics.json", {})
    cfg = _read_json(d / "config.json", {})
    W = np.load(d / "templates.npy")
    H = np.load(d / "activations.npy")
    ev = _read_json(d / "evaluation.json")
    stems_dir = d / "stems"
    stems = sorted(p.name for p in stems_dir.glob("*.wav")) if stems_dir.exists() else []
    grouped_dir = d / "grouped"
    grouped = sorted(p.name for p in grouped_dir.glob("*.wav")) if grouped_dir.exists() else []
    manifest = d / "figures.json"
    if manifest.exists():
        figures = _read_json(manifest, [])
    else:
        figures = sorted(
            p.name for p in (ROOT / "artifacts" / "figures").glob("*.png")
            if p.name.startswith(_figure_prefix(run_id))
        )
    return {
        "id": run_id,
        "diagnostics": diag,
        "config": cfg,
        "evaluation": ev,
        "templates": {
            "peak_bins": diag.get("template_peak_bins", []),
            "peak_hz": [float(b * diag.get("sample_rate", 1) / (2 * (len(W) - 1)))
                        for b in diag.get("template_peak_bins", [])],
            "gain_db": diag.get("template_gain_db", []),
            "total_energy_db": diag.get("template_total_energy_db", []),
            "n_templates": int(W.shape[1]),
            "n_bins": int(W.shape[0]),
            "epsilon": diag.get("eps"),
            "shapes_db": _template_shapes(W),
        },
        "activation_summary": _activation_summary(H),
        "stems": stems,
        "grouped": grouped,
        "figures": figures,
        "has_ground_truth": ev is not None,
    }


def group_payload(run_id: str, body: dict) -> dict:
    d = _run_dir(run_id)
    groups = body.get("groups")
    labels = body.get("labels")
    policy = body.get("policy", "wiener")
    power = float(body.get("power", 1.0))
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty list of lists of template indices")
    for g in groups:
        if not isinstance(g, list) or not all(isinstance(i, int) for i in g):
            raise ValueError("each group must be a list of integer template indices")
    W = np.load(d / "templates.npy")
    H = np.load(d / "activations.npy")
    diag = _read_json(d / "diagnostics.json", {})
    cfg = _read_json(d / "config.json", {})
    sr = int(diag.get("sample_rate", 22050))
    n_fft = int(cfg.get("n_fft", 1024))
    hop = int(cfg.get("hop_length", 256))
    mixture = _load_mixture(d, sr)
    # Recompute the complex STFT from the lossless mixture: storing it would
    # duplicate several MB per run and is a pure function of (mixture, n_fft,
    # hop), both of which are already saved.
    if (d / "mixture_stft.npy").exists():
        X = np.load(d / "mixture_stft.npy")
    else:
        X = stft(mixture, n_fft=n_fft, hop_length=hop)
    if labels is None:
        labels = [f"group_{i}" for i in range(len(groups))]
    if len(labels) != len(groups):
        raise ValueError("labels must match groups in length")
    safe = [_safe_label(l) for l in labels]
    ss = group_components(
        W, H, X, mixture, groups, sample_rate=sr, n_fft=n_fft, hop_length=hop,
        labels=safe, policy=policy, power=power,
    )
    grouped_dir = d / "grouped"
    grouped_dir.mkdir(exist_ok=True)
    for f in grouped_dir.glob("*.wav"):
        f.unlink()
    for i, stem in enumerate(ss.stems):
        save_audio(str(grouped_dir / f"{i:02d}_{safe[i]}.wav"), stem, sr)
    report = {
        "groups": groups,
        "labels": safe,
        "policy": ss.policy,
        "mask_sum_error": ss.masks_sum_error,
        "component_sum_error": ss.component_sum_error,
        "residual_energy_ratio": ss.residual_energy_ratio,
        "stems": [f"{i:02d}_{safe[i]}.wav" for i in range(len(ss.stems))],
        "note": "Group labels are user-supplied. They are not verified source identities.",
    }
    ev = _read_json(d / "evaluation.json")
    if ev:
        refs = _load_reference_stems(d)
        if refs is not None:
            from .metrics import projection_leakage, sdr
            rr, labels_ref = refs
            from itertools import permutations
            best = None
            n = min(len(ss.stems), len(rr))
            for perm in permutations(range(len(rr)), n):
                score = float(np.mean([sdr(rr[perm[i]], ss.stems[i]) for i in range(n)]))
                if best is None or score > best[1]:
                    best = (list(perm), score)
            report["ground_truth_eval"] = {
                "warning": "ground truth is used here only to score the user's grouping",
                "permutation": best[0],
                "mean_sdr_db": best[1],
                "per_group_sdr_db": [sdr(rr[best[0][i]], ss.stems[i]) for i in range(n)],
                "reference_labels": labels_ref,
            }
    _write_json(d / "grouped" / "grouping.json", report)
    return report


def _time_gate(n: int, sr: int, start_s: float, end_s: float, fade_s: float = 0.01) -> np.ndarray:
    """Raised-cosine window that is 1 inside [start, end] and 0 outside.

    Used to audition an activation segment.  The fade is short (10 ms) only to
    avoid clicks; it is not part of the decomposition and is reported back.
    """
    t = np.arange(n) / float(sr)
    w = ((t >= start_s) & (t < end_s)).astype(np.float64)
    fade = max(int(fade_s * sr), 1)
    if fade > 1:
        rise = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        i0 = int(round(start_s * sr))
        lo, hi = max(i0, 0), min(i0 + fade, n)
        if hi > lo:
            w[lo:hi] *= rise[: hi - lo]
        i1 = int(round(end_s * sr))
        lo, hi = max(i1 - fade, 0), min(i1, n)
        if hi > lo:
            w[lo:hi] *= rise[::-1][fade - (hi - lo):]
    return w


def solo_payload(run_id: str, body: dict) -> dict:
    d = _run_dir(run_id)
    stem_ids = body.get("stems", [])
    gains = body.get("gains_db", {})
    grouped = body.get("grouped", False)
    start_s = body.get("start_s")
    end_s = body.get("end_s")
    sub = "grouped" if grouped else "stems"
    base = d / sub
    files = sorted(p for p in base.glob("*.wav") if not p.name.endswith(".json"))
    if not files:
        raise ValueError(f"no stems in {sub}")
    diag = _read_json(d / "diagnostics.json", {})
    sr = int(diag.get("sample_rate", 22050))
    chosen = []
    for sid in stem_ids:
        if not isinstance(sid, int) or sid < 0 or sid >= len(files):
            raise ValueError("stem index out of range")
        x, _ = load_audio(str(files[sid]), mono=True, target_sr=sr)
        g = float(gains.get(str(sid), gains.get(sid, 0.0)))
        if g < -120 or g > 24:
            raise ValueError("gain out of range [-120, 24] dB")
        chosen.append(x * (10 ** (g / 20.0)))
    if not chosen:
        raise ValueError("select at least one stem")
    n = max(len(x) for x in chosen)
    out = np.zeros(n)
    for x in chosen:
        out[: len(x)] += x
    segment = None
    if start_s is not None or end_s is not None:
        if start_s is None or end_s is None:
            raise ValueError("start_s and end_s must be given together")
        start_s, end_s = float(start_s), float(end_s)
        if not (0.0 <= start_s < end_s <= n / sr + 1e-9):
            raise ValueError("segment outside the mixture or empty")
        gate = _time_gate(n, sr, start_s, end_s)
        out = out * gate
        segment = {"start_s": start_s, "end_s": end_s, "fade_s": 0.01}
    out_dir = d / "solo"
    out_dir.mkdir(exist_ok=True)
    name = "solo.wav"
    save_audio(str(out_dir / name), out, sr)
    return {"file": name, "stems": stem_ids, "gains_db": {str(k): v for k, v in gains.items()},
            "segment": segment, "peak": float(np.max(np.abs(out))) if out.size else 0.0}


def _safe_label(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._") or "group"
    return s[:40]


def _load_reference_stems(d: Path):
    """Find the ground-truth bundle for this run, if any.

    Anchored runs (e.g. ``separable_anchored``) share the base mixture's truth
    file (``separable_ground_truth.npz``), so the run name is a prefix rather
    than an exact match.  Pick the longest matching prefix; this is used only
    to score a user grouping, never for reconstruction.
    """
    best = None
    for npz in d.parent.glob("*_ground_truth.npz"):
        prefix = npz.stem[: -len("_ground_truth")]
        if d.name.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, npz)
    if best is None:
        return None
    data = np.load(best[1])
    return ([data["stems"][i] for i in range(len(data["stems"]))],
            [str(x) for x in data["labels"]])


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class Handler(BaseHTTPRequestHandler):
    server_version = "nmfsep-ui/0.1"

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                self._json(200, {"runs": list_runs()})
                return
            m = re.match(r"^/figures/([^/]+)$", path)
            if m:
                name = urllib.parse.unquote(m.group(1))
                if not re.match(r"^[A-Za-z0-9_.-]+\.png$", name):
                    self._json(400, {"error": "bad figure name"})
                    return
                fp = (ROOT / "artifacts" / "figures" / name).resolve()
                if fp.exists() and (ROOT / "artifacts" / "figures").resolve() in fp.parents:
                    self._send(200, fp.read_bytes(), "image/png")
                    return
                self._json(404, {"error": "figure not found"})
                return
            m = re.match(r"^/api/run/(.+?)/audio/([^/]+)$", path)
            if m:
                run_id = urllib.parse.unquote(m.group(1))
                name = urllib.parse.unquote(m.group(2))
                d = _run_dir(run_id)
                candidates = [d / name, d / "stems" / name, d / "grouped" / name,
                              d / "solo" / name]
                for c in candidates:
                    c = c.resolve()
                    if d in c.parents and c.exists() and c.suffix == ".wav":
                        self._send(200, c.read_bytes(), "audio/wav")
                        return
                self._json(404, {"error": "audio not found"})
                return
            m = re.match(r"^/api/run/(.+?)$", path)
            if m:
                self._json(200, run_payload(urllib.parse.unquote(m.group(1))))
                return
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            self._json(404, {"error": "not found"})
        except FileNotFoundError:
            self._json(404, {"error": "not found"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})

    do_HEAD = do_GET

    def do_POST(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                self._json(413, {"error": "body too large"})
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            m = re.match(r"^/api/run/(.+?)/group$", path)
            if m:
                self._json(200, group_payload(urllib.parse.unquote(m.group(1)), body))
                return
            m = re.match(r"^/api/run/(.+?)/solo$", path)
            if m:
                self._json(200, solo_payload(urllib.parse.unquote(m.group(1)), body))
                return
            self._json(404, {"error": "not found"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="nmfsep-ui")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}/  (runs from {RUNS})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
