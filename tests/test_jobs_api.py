import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from nmfsep import jobs as jobs_mod
from nmfsep.audio import save_audio
from nmfsep.server import Handler, ThreadingHTTPServer
from nmfsep.stft import stft


def _request(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


class NMFJobApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name) / "runs"
        self.runs.mkdir()
        # A minimal source run with a lossless mixture the job can re-factor.
        src = self.runs / "src"
        (src / "stems").mkdir(parents=True)
        rng = np.random.default_rng(0)
        mixture = rng.uniform(-0.4, 0.4, 8000)
        np.save(src / "mixture.npy", mixture)
        save_audio(str(src / "mixture.wav"), mixture, 22050)
        with open(src / "diagnostics.json", "w") as f:
            json.dump({"sample_rate": 22050, "mixture_name": "src"}, f)
        self._prev_runs = jobs_mod_mod = None
        import nmfsep.server as srv
        self._srv_mod = srv
        self._old_runs = srv.RUNS
        self._old_jobs_runs = srv.JOBS._runs_root
        srv.RUNS = self.runs
        srv.JOBS._runs_root = self.runs
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._srv_mod.RUNS = self._old_runs
        self._srv_mod.JOBS._runs_root = self._old_jobs_runs
        self.tmp.cleanup()

    def test_backends_listed(self):
        code, payload = _request(self.base, "GET", "/api/backends")
        self.assertEqual(code, 200)
        self.assertEqual(payload["backends"], ["numpy", "scipy"])

    def test_nmf_job_runs_new_path_and_writes_audible_run(self):
        code, job = _request(self.base, "POST", "/api/nmf", {
            "run_id": "out/job1", "source_run": "src",
            "backend": "scipy", "n_templates": 3, "n_iter": 80,
        })
        self.assertEqual(code, 200)
        self.assertIn(job["status"], ("queued", "running"))
        final = None
        for _ in range(60):
            code, final = _request(self.base, "GET", f"/api/jobs/{job['job_id']}")
            if final["status"] in ("done", "error"):
                break
            import time
            time.sleep(0.25)
        self.assertEqual(final["status"], "done", final.get("error"))
        self.assertTrue(final["result"]["audible"])
        self.assertEqual(final["result"]["diagnostics"]["linalg_backend"], "scipy")
        # the output run is a normal run directory consumable by the UI
        out = self.runs / "out" / "job1"
        self.assertTrue((out / "templates.npy").exists())
        self.assertTrue((out / "activations.npy").exists())
        self.assertEqual(len(list((out / "stems").glob("*.wav"))), 3)
        code, payload = _request(self.base, "GET", "/api/run/out/job1")
        self.assertEqual(code, 200)
        self.assertEqual(payload["templates"]["n_templates"], 3)
        # voice activation diagnostics are present and anomaly-flagged
        va = final["result"]["diagnostics"]["voice_activation"]
        self.assertIn("silent_components", va)
        self.assertIn("ambiguous_template_pairs", va)

    def test_invalid_backend_rejected(self):
        code, payload = _request(self.base, "POST", "/api/nmf", {
            "run_id": "out/bad", "source_run": "src", "backend": "torch",
        })
        self.assertEqual(code, 400)
        self.assertIn("unavailable", payload["error"])

    def test_group_write_gate_requires_confirmation(self):
        # Build a run whose first two templates are magnitude-indistinguishable.
        run = self.runs / "twins"
        run.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(2)
        mix = rng.uniform(-0.3, 0.3, 4000)
        X = stft(mix, n_fft=1024, hop_length=256)
        n_bins, n_frames = X.shape
        # two magnitude-indistinguishable templates over the real STFT grid
        W = np.zeros((n_bins, 2))
        W[: n_bins // 3, 0] = 1.0
        W[: n_bins // 3, 1] = 1.0001
        W /= W.sum(axis=0, keepdims=True)
        H = rng.uniform(0.2, 1.0, (2, n_frames))
        np.save(run / "templates.npy", W)
        np.save(run / "activations.npy", H)
        np.save(run / "mixture.npy", mix)
        np.save(run / "mixture_stft.npy", X)
        with open(run / "diagnostics.json", "w") as f:
            json.dump({"sample_rate": 22050}, f)
        with open(run / "config.json", "w") as f:
            json.dump({"n_fft": 1024, "hop_length": 256}, f)
        # merging the twins must be blocked first
        code, payload = _request(self.base, "POST", "/api/run/twins/group",
                                 {"groups": [[0, 1]], "labels": ["merged"]})
        self.assertEqual(code, 409)
        self.assertTrue(payload["confirmation_required"])
        # operator confirmation allows the write
        code, payload = _request(self.base, "POST", "/api/run/twins/group",
                                 {"groups": [[0, 1]], "labels": ["merged"],
                                  "confirm_write": True})
        self.assertEqual(code, 200)
        self.assertEqual(len(payload["stems"]), 1)


if __name__ == "__main__":
    unittest.main()
