# nmfsep — inspectable nonnegative spectral decomposition

A from-scratch magnitude-spectrogram NMF prototype for **part extraction from a
mono mix**, with two explicit strategies for non-uniqueness:
1. **Retain candidates near the optimum.** Several deterministic initialisations
   are run; every solution within a loss threshold of the best is kept and
   compared template-by-template (Hungarian cosine match + activation-gain
   ratio). If they disagree, that disagreement is reported instead of being
   hidden behind one arbitrary solution.
2. **Anchor a fixed basis of an observable source.** An isolated reference signal
   (calibration tone, known stationary source, previously validated template) is
   turned into a fixed NMF column. The remaining columns adapt. A paired
   unanchored fit is kept for comparison.

Components are **basis spectra × activations, not instruments**. The code never
attaches a source name automatically. Grouping is user input; metrics that use
ground truth are labelled as evaluation-only and never enter the factorisation.

Everything here is our own STFT, NMF, masking, metrics and rendering. No
separation model or black-box stem API is called.

---

## Pluggable linear-algebra backend (migration v0.2.0)

All matrix work inside the NMF solve — GEMM products, reductions, the NNDSVD
SVD, the NNLS least-squares solve, and the unit-L1 column gauge — goes through a
single backend contract (`nmfsep/linalg.py`), selectable per run:

| backend | driver path | role |
|---------|-------------|------|
| `numpy` (default) | `numpy.linalg` / `@` / pairwise `sum` | reference implementation |
| `scipy` | LAPACK `gesdd` SVD, `gelsd` lstsq, BLAS `gemm`, explicit naive-order reductions | alternative driver, not a rename |

**Layout.** The scipy path calls BLAS directly, so it distinguishes C- and
Fortran-ordered arrays: F-contiguous operands use a plain `gemm`; C-order
operands use the row-major identity `(A·B)_row = (Bᵀ·Aᵀ)_col` via
`gemm(trans_a, trans_b).T`. Every backend returns the canonical boundary type —
float64, C-contiguous — regardless of the working layout, so the rest of the
code never sees a foreign memory order.

**Precision.** `--precision float32` runs the working solve in single precision
(including `sgemm`) and promotes results back to canonical float64 at the
boundary. `float16` and any other dtype are rejected at backend construction.

**Normalisation.** Unit-L1 column normalisation is a backend method: free `W`
columns are rescaled in place and the same scale is moved onto the `H` rows, so
`W·H` is preserved bit-for-bit within tolerance (covered by test). Anchored
columns are never rescaled.

**Summation order is genuinely different.** The scipy reduction accumulates
scalar terms in naive sequential order, which rounds differently from numpy's
pairwise blocked sum on mixed-magnitude data (a test asserts the last bits
actually differ, so the alternative path cannot be a library alias). C- and
F-ordered inputs reduce identically.

**Cross-backend equivalence is checked in reconstruction space, not by objective
value.** `nmfsep/equivalence.py` first fixes the unit-L1 gauge, matches columns
by unit-L2 template cosine with Hungarian assignment, and only then compares:
aligned column cosines, activation-row correlations, the relative error of
`W·H`, and the objective gap under a summation-order tolerance. Two solutions
with the *same loss* but swapped columns are explicitly **not** equivalent —
equal target values never identify a voice. A rank mismatch is reported as
non-equivalent rather than raised. Fixed identical initialisation (`seed=42`)
is used for every cross-backend comparison so the optimisation target, not the
starting point, is what differs.

**The new path emits audible components.** Both backends run the full
STFT → NMF → Wiener-mask → ISTFT pipeline; per-stem energy and the partition
of unity are asserted, so switching backends changes the driver, not the
output contract.

`scripts/verify_backend_migration.py` regenerates a low-rank and a sparse
spectral target in-process (no hand-edited data), refuses a non-empty output
directory, runs both backends from the same init, and writes per-target evidence
JSON (objective gap, aligned cosines/correlations, reconstruction error,
component energies, audibility, automation swap check) plus a summary. It exits
nonzero if any equivalence or audibility check fails.

### Identity-keyed mix automation and the write gate

A backend switch can permute NMF columns while leaving `W·H` identical. Any
fade stored by column index would then be written onto the wrong instrument.
Automation curves (`nmfsep/automation.py`) are therefore bound to a component's
**unit-L2 spectral signature**, not its index: every application re-claims the
target by cosine match. If the mixer feedback cannot uniquely identify a voice
(multiple templates above the match threshold), the call raises
`AmbiguousClaimError`; no stem is written until an operator explicitly confirms.
The server exposes the same rule as an HTTP **409 `confirmation_required`**
response on grouped-stem writes, accepted only when the request is re-sent with
`confirm_write: true`. The two separated voices get distinct default curves
(exponential fade-out vs. swell).

---

## Quick start

```bash
# local Python deps (numpy, scipy, soundfile, matplotlib) live in .pylibs here
export PYTHONPATH=/workspace/.pylibs:/workspace

# every solve accepts --backend {numpy,scipy} and --precision {float64,float32}
python3 -m nmfsep.cli demo  --templates 4 --iters 600 --seeds 5 --out runs/demo
python3 -m nmfsep.cli demo  --backend scipy --precision float32 \
    --templates 4 --iters 600 --seeds 5 --out runs/demo_scipy
python3 -m nmfsep.cli unknown \
    --audio data/external/beethoven_op101_mvt1.ogg \
    --offset 0 --duration 20 --templates 6 --iters 400 --seeds 4 \
    --out runs/unknown/beethoven_op101
# over-decomposition demonstration on the same excerpt
python3 -m nmfsep.cli unknown \
    --audio data/external/beethoven_op101_mvt1.ogg \
    --offset 0 --duration 20 --templates 9 --iters 400 --seeds 4 \
    --out runs/unknown/beethoven_op101_k9
python3 -m nmfsep.cli sweep --min-templates 2 --max-templates 9 --iters 400 --seeds 4 \
    --out runs/sweep
python3 -m nmfsep.cli sweep --audio data/external/beethoven_op101_mvt1.ogg \
    --offset 0 --duration 20 --min-templates 2 --max-templates 9 --iters 400 --seeds 4 \
    --out runs/unknown_sweep

# backend-migration evidence in a clean directory (generates its own inputs)
python3 scripts/verify_backend_migration.py --out /tmp/nmfsep_migration_check

# CLI is auxiliary to NMF: a single solve, writing nmf_report.json with the
# backend identity, per-component energy and audibility evidence
python3 -m nmfsep.cli nmf --backend scipy --templates 3 --iters 200 \
    --audio data/external/beethoven_op101_mvt1.ogg --duration 10 \
    --out runs/cli_nmf

# figures (reads saved runs; truth npz only for the synthetic mixtures)
python3 - <<'PY'
from nmfsep.figures import generate_for_run, plot_sweep, plot_unknown_sweep
generate_for_run('runs/demo/separable', truth_npz='runs/demo/separable_ground_truth.npz')
generate_for_run('runs/demo/ambiguous', truth_npz='runs/demo/ambiguous_ground_truth.npz')
generate_for_run('runs/demo/separable_anchored')
generate_for_run('runs/demo/ambiguous_twins_anchored')
generate_for_run('runs/unknown/beethoven_op101')
generate_for_run('runs/unknown/beethoven_op101_k9')
plot_sweep('runs/sweep/sweep.json', 'artifacts/figures/sweep_synthetic.png')
plot_unknown_sweep('runs/unknown_sweep/sweep.json', 'artifacts/figures/unknown_sweep.png')
PY

# tests (stdlib unittest; pytest is not installed)
python3 -m unittest discover -s tests -v

# inspection UI — the graphical NMF entry point (Run NMF panel drives the new
# backend-pluggable implementation; the CLI above is only auxiliary)
python3 -m nmfsep.server --port 8010
#   -> http://127.0.0.1:8010/?run=demo/separable
```

Runs are plain directories under `runs/` (config, diagnostics, `templates.npy`,
`activations.npy`, `mixture.npy`, per-template WAVs, candidate JSON/NPY,
`evaluation.json` where truth exists, and a per-run `figures.json` manifest).

---

## Model, objective, initialisation, stopping

**Signal model.** `x` → STFT `X` (periodic Hann, `n_fft=1024`, `hop=256`,
center-padded; exact overlap-add round-trip). We factor the **magnitude**
`V = |X|` as `V ≈ W H`, `W ≥ 0` (`n_bins × K`), `H ≥ 0` (`K × n_frames`).

**Objective.** Beta-divergence, element-wise sum `D_β(V || WH)`:

| β | divergence | note |
|---|------------|------|
| 1 | `Σ(V log(V/V̂) − V + V̂)` | generalised KL (default) |
| 2 | `0.5 Σ(V − V̂)²` | squared Frobenius |
| 0 | `Σ(V/V̂ − log(V/V̂) − 1)` | Itakura–Saitsu |

Updates are the standard multiplicative rules (`_update_h`, `_update_w` in
`nmfsep/nmf.py`); the returned `W,H` are the **best observed state** so a
numerical blip cannot make the reported objective worse.

**Initialisation.** Free columns: `nndsvda` (NNDSVD with zeros replaced by the
positive mean; plain NNDSVD leaves dead components here and stalls) or seeded
random. `H` is then initialised by NNLS against `W`. Anchors are unit-L1 fixed
columns and are not updated. The candidate grid is
`[(nndsvda, seed0)] + [(random, seed_i)]`, plus anchored `random` runs when
anchors are supplied — deterministic given `seeds`.

**Stopping.** Stop when the relative improvement over the last `patience=15`
iterations is `< tol=1e-5` after `min_iter=10`, or at `n_iter`, or
`non_finite_loss`, or (a guard) `loss_increase` beyond `1e-6`. `stop_reason` and
`n_iter` are recorded per candidate.

**Scale, silence, tiny denominators.** Free columns are re-normalised to unit L1
every iteration and the scale is moved into `H`; a template's "gain" is
`‖H_k‖` (reported in dB), which removes W/H scale ambiguity from comparisons.
Frames with energy ≤ `silence_rel·max_frame_energy` (`1e-8`) are marked silent,
their `H` columns are fixed at 0, and they are excluded from the reported loss.
All divisions use an additive `eps` relative to the mean positive magnitude
(`~1e-9·mean`), so results are invariant to global gain.

**Template counts are comparable.** Every sweep row reports train loss,
held-out loss (15 % of frames held out, `H` re-estimated on the training frames
only), parameters, parameters/frame, retention, stability, stop reason and
runtime at the same STFT settings. Held-out loss is the model-selection signal;
train loss is not comparable across K in the usual "more templates fit better"
sense, and we do not pretend otherwise.

---

## Reconstruction policy (declared, checked)

Per-component complex spectrogram is a mask applied to `X`:

* `wiener` (default): `M_k = (W_k H_k)^p / Σ_j (W_j H_j)^p`, partition of unity
  per bin/frame; compressing `X` with it and inverting gives stems whose sum
  equals the mixture's own inverse STFT to floating point.
* `ratio`: same at `p=1`, kept as a readable name.
* `binary`: one-hot arg-max mask (still a partition of unity, more artefacts).
* `residual`: soft masks, then add `mixture − Σ stems` to the largest-energy
  stem, making `Σ stems == mixture` **exactly in the sample domain**; the
  residual energy ratio is reported so the cost is visible.

Where the model explains nothing at a bin/frame the denominator is below `eps`
and the mask is zero (that energy is simply not attributed). Every reconstruction
reports `mask_sum_error` (complex-spectrogram sum vs `X`) and
`component_sum_error` (sample sum vs mixture). Observed on these runs:
`~1e-16` for wiener, exactly `0` for residual apart from the reported residual
ratio, and `~1` (nothing attributed) for an all-zero model — see
`tests/test_masks.py`.

---

## Two non-uniqueness strategies, concretely

### A. Candidate retention (`compare_candidates`)

For each candidate, templates are matched to the best solution by Hungarian
assignment on cosine similarity, and `H` row gains are compared on that
assignment. A candidate is **retained** if its loss is within `retain_rtol`
(default 1 %) of the best. The set is **stable** only if every other retained
candidate has all template cosines ≥ `stability_cos` (0.98) and all gain ratios
within ±3 dB (factor 2 in power). Otherwise both solutions are kept for audition
and the disagreement is shown in the UI.

Because anchored and unanchored candidates optimise over **different feasible
sets**, retention and stability are computed **per kind** (`candidate_kind_stats`)
rather than across one combined loss threshold; the UI reports the selected
candidate's kind.

### B. Fixed anchors + gain-ambiguity probe

`gain_ambiguity_probe` finds template pairs with cosine ≥ 0.995 and merges them,
re-estimating `H`. If the objective changes by less than the retention
threshold, the data cannot decide how gain is split between them: the split is
**unidentifiable**, and the reported gain pair is one representative only.

---

## Synthetic benchmark (known stems)

Mix and reference stems are generated by `nmfsep.synth`; references are used
**only** for evaluation and for oracle-grouping upper bounds.

* `separable` (8 s, 3 sources: 110 Hz harmonic, inharmonic bell at 660 Hz,
  3.2 kHz band noise, overlapping note patterns), K=3.
* `ambiguous` (8 s, three sources: twin_A and twin_B with **magnitude-identical**
  partials and note times — phase / +0.3 % detune only; measured template cosine
  0.9993 — plus distinct narrowband bursts), K=2.

Results (best candidate, β=1, 5 seeds, 600 iters, check `runs/demo/*/diagnostics.json`):

| run | K | train loss | held-out | mean SDR* | oracle-grouped SDR | mask Σ err |
|-----|---|-----------|----------|-----------|--------------------|------------|
| separable | 3 | 14231.0 | 2063.7 | 25.69 dB | 43.03 / 21.98 / 12.06 dB | 1.0e-16 |
| ambiguous | 2 | 15816.8 | 2462.5 | 36.23 dB | 0.00 / 0.82 / 71.63 dB | 7.5e-17 |
| separable_anchored | 4 | 12876.2 | 1927.6 | 14.47 dB | 38.67 / 22.38 / 12.54 dB | 1.0e-16 |
| ambiguous_twins_anchored | 3 | 15851.1 | 2480.8 | 22.10 dB | 0.27 / 0.36 / 65.66 dB | 9.5e-17 |

Per-kind retention (same folds; anchored and unanchored counted separately).
"Best unanchored loss" is the best loss among unanchored candidates only; it is
not comparable to the anchored loss when the anchor constraint binds, so it is
shown for transparency rather than used to pick a winner:

| run | selected kind | selected kind stable | retained (kind) | best unanchored loss | unanchored stable |
|-----|---------------|----------------------|-----------------|----------------------|-------------------|
| separable | unanchored | yes | 1/5 | 14231.1 (same kind) | yes |
| ambiguous | unanchored | yes | 4/5 | 15816.8 (same kind) | yes |
| separable_anchored | anchored | yes | 2/5 | 14146.0 (9.86 % above anchored) | **no** (45.8 % spread within kind) |
| ambiguous_twins_anchored | anchored | yes | 5/5 (all within 0.04 %) | 10300.4 (lower; the anchor constraint costs fit) | yes |

\* Permutation-maximised mean over the *ungrouped* components; it is a summary,
not a per-source claim. Read the per-estimate and leakage matrices in the UI.

**Interpretation.**

* On `separable`, a good candidate reaches 43.0 / 22.0 / 12.1 dB oracle; plain
  NNDSVD stalls far below, which is exactly why the candidate grid is retained.
* On `ambiguous`, no decomposition can separate the twins (their magnitude
  spectra are identical); the "best" 71.6 dB belongs to the distinct narrowband
  source, and the twins stay ~0 dB. This is a correct failure, not a bug.
* Anchoring on two known synthetic stems (`low_harmonic`, `band_noise`) lowers
  the objective (12876 vs 14231) and keeps each anchored template dedicated to
  its source. It also reveals a real consequence of forcing K=4 onto a 3-source
  mix: the free columns split the bell source across templates 2 and 3 (both
  peak at the same bin, template cosine 0.852). Ungrouped, those two components
  score −7.8 dB each; the oracle grouping `[[1],[2,3],[0]]` merges them and
  restores 22.38 dB. The UI lists the most similar template pairs precisely so
  this split is visible and can be grouped by the user. (The 0.995 gain-probe
  does not trigger at cosine 0.852 — that probe is for *unidentifiable* gain
  splits, a stricter condition than "same likely source".)
* Anchoring the twins yields an explicit `any_unidentifiable_gain_split = True`
  (template cosine 0.9993, merged loss change 0.25 %), with gains
  67.29 / 68.44 / 68.08 dB — one representative of an unobservable split.
* On the twins mix the anchored fit's objective (15851) is *higher* than a free
  K=3 fit (10300). That gap is instructive, not a defect: the free fit reaches
  it by splitting the twins' 196 Hz fundamental (bin 9) and 392 Hz harmonic
  (bin 18) into two different templates, i.e. it decomposes one pair of sources
  by partial, which is over-decomposition rather than separation. The anchored
  fit keeps both twin templates on the same partial set and adds only the
  narrowband source, so it is the more parsimonious description even though its
  loss is higher. This is why loss alone is never presented as a quality measure.

**Leakage / spectral error** are in `evaluation.json` and the UI: projection
leakage matrix (dB), spectral convergence, spectrogram MSE, SDR and SI-SDR per
estimate, with the permutation used for matching recorded. When a run has more
components than reference sources (e.g. the K=4 anchored run against 3 known
stems), surplus estimates are marked unmatched and left unscored — scoring them
against silence would produce meaningless numbers — and the reported mean SDR is
over matched pairs only (`n_matched`, `n_estimates`).

---

## Unknown recording — failure analysis (no ground truth)

Input: Beethoven, Piano Sonata Op. 101, mvt. I, Musopen recording from Wikimedia
Commons (public domain; file `data/external/beethoven_op101_mvt1.ogg`). 20 s
excerpt, 22.05 kHz mono, K=6 for the main run. No separation accuracy can be
computed, so **no accuracy is claimed**; the failure signals are the model
diagnostics themselves.

K-sweep on the same excerpt (`runs/unknown_sweep/sweep.json`):

| K | train loss | held-out | retained / run | stable | gain unident. |
|---|-----------|----------|----------------|--------|---------------|
| 2 | 38444.9 | 5383.1 | 2/4 | no | no |
| 3 | 31376.5 | 4364.0 | 1/4 | yes | no |
| 4 | 27947.7 | 3909.6 | 1/4 | yes | no |
| 5 | 25251.3 | 3430.2 | 2/4 | no | no |
| 6 | 21583.0 | 2972.6 | 1/4 | yes | no |
| 7 | 19114.1 | 2582.4 | 1/4 | yes | no |
| 8 | 16661.1 | 2258.7 | 1/4 | yes | no |
| 9 | 15615.1 | 2090.5 | 1/4 | yes | no |

Two honest conclusions:

1. **No held-out optimum.** Unlike the synthetic mix, held-out loss keeps falling
   through K=9. A piano recording is dense and reverberant; the model class
   (shared, time-invariant spectral templates) can always buy fit by adding
   components. Template count cannot be chosen from this data by held-out loss,
   so any "best K" would be arbitrary. We therefore fix K=6 by inspection and
   say so.
2. **Attribution is fragile where it matters.** Candidate stability is
   non-monotone (unstable at K=2 and K=5) even though the objective is smooth in
   K: the fit is stable but *which* local optimum you land in is not. On a real
   recording this is the failure mode to watch, not the loss value. The UI
   exposes it as the retained-candidate and stability pills.

Additional K=9 run (`runs/unknown/beethoven_op101_k9`, loss 15615.1, held-out
2090.5) demonstrates over-decomposition: nine components explain the excerpt
with lower loss, but each template is a narrower set of partials (e.g. extra
peaks at 258, 366, 495, 754 Hz appear, splitting what K=6 covers with broader
templates) and the activations fragment into more short bursts. There are no
reference stems, so this is illustrated by
`figures/beethoven_op101_k9_templates.png` and is a caution, not a quality claim.

What a user can still do: group templates by listening (`Solo / audition`),
audition a **time segment** by dragging on the activation plot, and type their
own group labels. Every grouping response reports the mask/sample errors and the
declared policy's residual, and — for synthetic runs only — a ground-truth score
of *that exact grouping*.

---

## UI

`python3 -m nmfsep.server --port 8010`, then open
`http://127.0.0.1:8010/?run=demo/separable`.

**The inspection window is the graphical NMF entry point.** The *Run NMF —
spectral decomposition* panel submits a real backend-pluggable factorisation job
(backend, precision, K, iterations, init and seed selectors) to a threaded job
manager. Run state is visible while the job is queued/running/done; on
completion the new run loads automatically with its separated stems ready to
audition. Voice-activation and separation anomalies are flagged inline: silent
components and indistinguishable template pairs (cosine ≥ 0.995) that require
operator confirmation before any grouped write. The run-state pills show the
active linalg backend and working dtype.

* Run state: stability (per selected candidate kind), gain identifiability,
  retained count, stop reason, iterations, silent frames, anchored names,
  linalg backend/dtype, mask and sample sum errors, selected candidate.
* Template shapes (per-template dB-relative, **no source names**) and activation
  traces over time.
* Solo/audition: per-template audio, solo checkbox, per-stem gain dB, server-side
  mixdown; **drag on the activation canvas to select a time segment** and render
  only that segment (10 ms raised-cosine gate, reported in the response).
* Mix automation: one identity-keyed curve row per template with distinct
  presets; applying re-claims the target by spectral signature, and an
  unclaimable voice opens an operator-confirmation dialog instead of writing.
* Grouping editor: user-defined groups, duplicate/missing-template hints, policy
  and mask power; returns grouped stems plus mask/sample errors and (synthetic
  only) a ground-truth check of that grouping. A merge of indistinguishable
  voices answers HTTP 409 first and writes only after explicit confirmation.
* Candidate table (kind/init/seed/loss/gap/retained/shown/stop), template
  similarity and gain-ratio table, gain-ambiguity table, full ground-truth
  evaluation table with leakage matrix and oracle grouping, per-run figures.

All UI numbers are read from saved diagnostics or computed from the saved
`W, H, X`; nothing is fabricated in the page. `scripts/ui_check_backend.mjs`
drives the whole flow (backend selector, scipy job completion, anomaly line,
automation apply, 409 confirmation dialog) in a real browser and writes
`artifacts/ui_backend_migration.png`.

---

## Tests

```bash
python3 -m unittest discover -s tests -v    # 53 tests
```

Covers STFT round-trip/shape/length, NMF candidate recovery, monotonicity,
anchors fixed, silence-zeroed `H`, scale invariance, duplicated-template gain
detection, candidate comparison, initialisation shapes; mask partition of unity,
residual exactness, binary one-hot, grouping overlap rejection, zero-model
finite output, time-gate segment; SDR/SI-SDR identities, leakage diagonal,
permutation, spectral convergence.

The v0.2.0 migration adds four suites:

* `test_linalg_backends.py` — backend availability and invalid-precision
  rejection; matmul across C/F/sliced layouts in float32 and float64 with
  canonical float64 C-order outputs; SVD/lstsq against the numpy reference;
  NNLS non-negativity; layout-independent reductions; a test that the scipy
  summation order really differs in the last bits; gauge normalisation
  preserving `W·H`; backend identity reporting.
* `test_backend_equivalence.py` — fixed-init numpy/scipy equivalence in
  reconstruction space; permutation alignment; the rule that equal loss with
  swapped columns is not equivalence; rank mismatch reported, not raised;
  audible stems from both backends.
* `test_automation.py` — distinct curves per voice; the fade following the
  voice across a column swap while a naive index would fail; ambiguity
  blocking until confirmed; missing-target blocking; preset validity.
* `test_jobs_api.py` — backend listing; a full scipy job writing an audible,
  UI-consumable run with voice-activation diagnostics; invalid backend rejected
  with HTTP 400; the grouped-write 409 gate and confirmed 200 rewrite.

---

## Provenance / licensing

* `data/external/beethoven_op101_mvt1.ogg` — Wikimedia Commons, Musopen
  recording, verified **public domain**; used as the unknown input.
  `beethoven_symphony5_mvt1.ogg` is also public domain and unused.
* Synthetic mixtures and stems are generated by this repository (`nmfsep.synth`).
* No reference stems of any real recording were used to construct any estimated
  output.

## Build locking and migration marker

`requirements-lock.txt` pins the exact Python dependency set used with the
migration (numpy 2.4.6, scipy 1.17.1, soundfile, matplotlib and transitive
pins); install with `python3 -m pip install -r requirements-lock.txt`.
`package.json` carries a `linalgBackendMigration` marker recording version
0.2.0, the two backend names and the lockfile, and the package/module version
is bumped to 0.2.0 (`nmfsep/__init__.py`). Every run written by the new path
records `linalg_backend`, `linalg_dtype` and `nmfsep_version` in its
diagnostics, so output artifacts identify the driver that produced them.

## Layout

```
nmfsep/{stft,nmf,masks,metrics,synth,grouping,pipeline,audio,cli,figures,
        server,linalg,equivalence,automation,jobs}.py
ui/index.html
tests/            (unittest, 53 tests)
runs/             generated runs (not required for the code to run)
artifacts/figures generated figures; UI screenshots
data/external/    downloaded public-domain audio
scripts/          verification + Playwright UI checks (need node_modules/ +
                  Chromium libs; the browser check runs headless)
requirements-lock.txt  pinned dependency set for the v0.2.0 migration
```

`runs/`, `artifacts/`, `data/external/`, `.pylibs/`, `node_modules/` and the
generated `package*.json` are local artifacts; only the source needs to ship.
