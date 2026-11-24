"""In-process NMF job manager for the spectral-editing window.

The graphical window is the primary entry for running the new NMF
implementation: a job is submitted over HTTP, executed on a background thread
with visible status, and its results are written as an ordinary run under
``runs/`` so every existing inspection/audition view keeps working unchanged.

Job state is intentionally explicit (``queued``/``running``/``done``/``error``)
so the UI can show progress and surface anomalies (silent components, unstable
or unidentifiable voices) instead of blocking on a request.  The CLI only
wraps the same library functions; it does not own the workflow.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import save_audio
from .linalg import available_backends
from .masks import reconstruct_stems
from .nmf import nmf
from .stft import stft

VALID_STATUSES = ("queued", "running", "done", "error")


@dataclass
class NMFJob:
    job_id: str
    run_id: str
    backend: str
    precision: str
    params: dict
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "backend": self.backend,
            "precision": self.precision,
            "params": dict(self.params),
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self, runs_root: Path):
        self._jobs: dict[str, NMFJob] = {}
        self._lock = threading.Lock()
        self._runs_root = Path(runs_root)
        self._counter = 0

    def list_backends(self) -> list[str]:
        return available_backends()

    def submit(
        self,
        *,
        mixture: np.ndarray,
        sample_rate: int,
        run_id: str,
        backend: str,
        precision: str,
        n_templates: int,
        n_iter: int,
        init: str,
        seed: int,
        beta: float,
        mixture_name: str,
        nmfsep_version: str,
    ) -> NMFJob:
        if backend not in available_backends():
            raise ValueError(f"backend {backend!r} unavailable; have {available_backends()}")
        with self._lock:
            self._counter += 1
            job_id = f"nmf-{int(time.time())}-{self._counter}"
        params = {
            "n_templates": int(n_templates), "n_iter": int(n_iter),
            "init": init, "seed": int(seed), "beta": float(beta),
            "n_fft": 1024, "hop_length": 256, "mixture_name": mixture_name,
        }
        job = NMFJob(
            job_id=job_id, run_id=run_id, backend=backend, precision=precision,
            params=params,
        )
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job, mixture, sample_rate, run_id, backend, precision,
                  n_templates, n_iter, init, seed, beta, mixture_name, nmfsep_version),
            daemon=True,
        )
        thread.start()
        return job

    def get(self, job_id: str) -> NMFJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _set(self, job: NMFJob, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(job, k, v)

    def _run(self, job, mixture, sr, run_id, backend, precision,
             n_templates, n_iter, init, seed, beta, mixture_name, nmfsep_version):
        try:
            self._set(job, status="running", started_at=time.time(),
                      message="computing STFT", progress=0.05)
            X = stft(mixture, n_fft=1024, hop_length=256)
            V = np.abs(X)
            self._set(job, message=f"factorising with {backend} backend", progress=0.2)
            res = nmf(
                V, n_templates, beta=beta, n_iter=n_iter, init=init, seed=seed,
                linalg_backend=backend, linalg_dtype=precision,
            )
            self._set(job, progress=0.75, message="reconstructing component stems")
            labels = [f"template_{k:02d}" for k in range(n_templates)]
            stems = reconstruct_stems(
                res.W, res.H, X, mixture, sample_rate=sr,
                n_fft=1024, hop_length=256, labels=labels,
            )
            self._set(job, progress=0.9, message="writing run artifacts")
            out_dir = self._runs_root / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "stems").mkdir(exist_ok=True)
            np.save(out_dir / "templates.npy", res.W)
            np.save(out_dir / "activations.npy", res.H)
            np.save(out_dir / "mixture.npy", mixture)
            save_audio(str(out_dir / "mixture.wav"), mixture, sr)
            for i, stem in enumerate(stems.stems):
                save_audio(str(out_dir / "stems" / f"{labels[i]}.wav"), stem, sr)

            # Voice-activation / separation anomalies, computed from the fit
            # alone (no ground truth): silent components and near-duplicate
            # shapes the mixer cannot distinguish.
            gains = [float(np.linalg.norm(h)) for h in res.H]
            silent_components = [k for k, g in enumerate(gains) if g <= res.eps]
            Wn = res.W / np.maximum(np.linalg.norm(res.W, axis=0, keepdims=True), 1e-12)
            C = Wn.T @ Wn
            ambiguous_pairs = [
                {"templates": [k, j], "cosine": float(C[k, j])}
                for k in range(n_templates) for j in range(k + 1, n_templates)
                if float(C[k, j]) >= 0.995
            ]
            diagnostics = {
                "mixture_name": mixture_name,
                "sample_rate": int(sr),
                "n_samples": int(len(mixture)),
                "n_frames": int(V.shape[1]),
                "n_bins": int(V.shape[0]),
                "best_loss": res.final_loss,
                "train_loss": res.final_loss,
                "stop_reason": res.stop_reason,
                "n_iter": res.n_iter,
                "silent_frames": res.silent_frames,
                "silent_frame_count": len(res.silent_frames),
                "eps": res.eps,
                "n_parameters": res.n_parameters,
                "policy": stems.policy,
                "mask_sum_error": stems.masks_sum_error,
                "component_sum_error": stems.component_sum_error,
                "residual_energy_ratio": stems.residual_energy_ratio,
                "template_peak_bins": [int(np.argmax(res.W[:, k])) for k in range(n_templates)],
                "template_gain_db": [
                    float(20 * np.log10(max(g, 1e-12))) for g in gains
                ],
                "linalg_backend": res.linalg_backend,
                "linalg_dtype": res.linalg_dtype,
                "nmfsep_version": nmfsep_version,
                "job_id": job.job_id,
                "voice_activation": {
                    "component_gains": gains,
                    "silent_components": silent_components,
                    "ambiguous_template_pairs": ambiguous_pairs,
                    "n_ambiguous_pairs": len(ambiguous_pairs),
                },
            }
            import json
            with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, sort_keys=True)
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "n_fft": 1024, "hop_length": 256, "n_templates": n_templates,
                    "beta": beta, "n_iter_cap": n_iter, "init": init,
                    "objective": {1: "generalised KL", 2: "squared Frobenius",
                                  0: "Itakura-Saito"}.get(beta, f"beta={beta}"),
                    "linalg_backend": backend, "linalg_dtype": precision,
                    "policy": "wiener", "seeds": [seed],
                }, f, indent=2, sort_keys=True)
            anomalies = []
            if silent_components:
                anomalies.append(f"silent component(s): {silent_components}")
            if ambiguous_pairs:
                anomalies.append(
                    f"{len(ambiguous_pairs)} indistinguishable template pair(s); "
                    "operator confirmation needed before writing grouped stems"
                )
            self._set(
                job, status="done", progress=1.0, finished_at=time.time(),
                message="factorisation complete",
                result={
                    "run_id": run_id,
                    "diagnostics": diagnostics,
                    "anomalies": anomalies,
                    "stem_energies": [float(np.sum(s**2)) for s in stems.stems],
                    "audible": all(float(np.sum(s**2)) > 1e-9 for s in stems.stems),
                },
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            self._set(job, status="error", finished_at=time.time(),
                      error=repr(exc), message="factorisation failed",
                      result={"traceback": traceback.format_exc()})
