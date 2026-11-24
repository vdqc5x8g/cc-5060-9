"""Identity-keyed mix automation and the stem write-confirmation gate.

The failure this module prevents
----------------------------------
NMF columns are exchangeable: switching the linear-algebra backend (or merely
changing the initialisation) can return an identical reconstruction with the
component columns permuted.  A fade drawn against a *column index* would then be
applied to a different instrument even though ``W H`` is bit-for-bit the same.

Automation here is therefore bound to a **component identity** derived from the
fitted template shape (its cosine signature), never to an index.  Before a curve
is rendered, its target is re-claimed against the current factorisation by
column-wise normalised matching:

* exactly one column matches above ``match_cosine``  -> claimed unambiguously;
* no column matches                                    -> ``AutomationTargetError``;
* several columns match (the mixer cannot tell voices apart, e.g. the
  magnitude-identical twins)                            ->
  :class:`AmbiguousClaimError`, carrying the candidates.  The caller (UI) must
  obtain explicit operator confirmation and pass ``confirm=True`` before any
  stem is written to disk.

Two separated voices get two *different* automation curves; the shipped
presets are deliberately distinct shapes so the swap bug is audible when it
occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


class AutomationTargetError(LookupError):
    """The automation's target component no longer exists in the factorisation."""


class AmbiguousClaimError(RuntimeError):
    """More than one live component matches the automation target."""

    def __init__(self, message: str, *, candidates: list[dict], automation_id: str):
        super().__init__(message)
        self.candidates = candidates
        self.automation_id = automation_id


def component_signature(W: np.ndarray, index: int, eps: float = 1e-12) -> np.ndarray:
    """Unit-L2 template shape used as the component's backend-independent identity.

    Scale-invariant (a column's L1/L2 normalisation does not change under the
    NMF gauge moves) and permutation-stable, so a backend switch that only
    reorders columns keeps each signature attached to the same voice.
    """
    col = np.asarray(W, dtype=np.float64)[:, index]
    return col / max(float(np.linalg.norm(col)), eps)


def claim_component(
    signature: np.ndarray, W: np.ndarray, *, match_cosine: float = 0.98
) -> dict:
    """Re-claim an automation target against a current factorisation.

    Returns ``{"index", "cosine", "candidates"}``; raises
    :class:`AmbiguousClaimError` when more than one live column matches, and
    :class:`AutomationTargetError` when none does.
    """
    Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    sims = Wn.T @ np.asarray(signature, dtype=np.float64)  # (K,)
    order = np.argsort(-sims)
    hits = [
        {"index": int(k), "cosine": float(sims[k])}
        for k in order if float(sims[k]) >= match_cosine
    ]
    if not hits:
        raise AutomationTargetError(
            f"no component matches the automation target "
            f"(best cosine {float(sims[order[0]]):.4f} < {match_cosine})"
        )
    if len(hits) > 1:
        raise AmbiguousClaimError(
            "mixer feedback cannot uniquely identify the target voice: "
            f"{len(hits)} components match within cosine {match_cosine}; "
            "operator confirmation is required before writing stems",
            candidates=hits,
            automation_id="",
        )
    return {"index": hits[0]["index"], "cosine": hits[0]["cosine"], "candidates": hits}


@dataclass
class AutomationCurve:
    """A gain envelope (linear amplitude, 0..1) bound to a component identity.

    ``preset`` names the shape; ``signature`` is the identity captured at draw
    time.  The curve is never stored against a column index, so backend-induced
    permutations cannot route the fade to the wrong instrument.
    """

    automation_id: str
    signature: np.ndarray
    preset: str
    n_points: int
    gain: np.ndarray
    label: str = ""
    confirmed_index: int | None = None
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "automation_id": self.automation_id,
            "preset": self.preset,
            "label": self.label,
            "n_points": int(self.n_points),
            "gain": [float(g) for g in self.gain],
            "confirmed_index": self.confirmed_index,
            "history": list(self.history),
        }


def fade_out_curve(n_points: int, *, shape: str = "exponential") -> np.ndarray:
    """Distinct fade-out envelope for one voice."""
    n = int(n_points)
    if n < 2:
        return np.ones(max(n, 1), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    if shape == "linear":
        return 1.0 - t
    if shape == "exponential":
        # Long level head, quick tail: reaches ~0.05 at the end.
        return np.maximum(1.0 - t, 1e-3) ** 2.2
    if shape == "raised_cosine":
        return 0.5 * (1.0 + np.cos(np.pi * t))
    raise ValueError(f"unknown fade shape {shape!r}")


def swell_curve(n_points: int) -> np.ndarray:
    """Distinct automation shape for a second voice: swell then settle."""
    n = int(n_points)
    if n < 2:
        return np.ones(max(n, 1), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    return 0.45 + 0.55 * np.sin(np.pi * t) ** 1.5


def make_automation(
    automation_id: str,
    W: np.ndarray,
    component_index: int,
    n_points: int,
    preset: str,
    *,
    label: str = "",
) -> AutomationCurve:
    """Capture an identity-keyed automation curve for one component."""
    sig = component_signature(W, component_index)
    if preset == "fade_out_linear":
        gain = fade_out_curve(n_points, shape="linear")
    elif preset == "fade_out_exponential":
        gain = fade_out_curve(n_points, shape="exponential")
    elif preset == "fade_out_cosine":
        gain = fade_out_curve(n_points, shape="raised_cosine")
    elif preset == "swell":
        gain = swell_curve(n_points)
    else:
        raise ValueError(f"unknown automation preset {preset!r}")
    return AutomationCurve(
        automation_id=automation_id, signature=sig, preset=preset,
        n_points=int(n_points), gain=gain, label=label,
        confirmed_index=int(component_index),
    )


def _curve_to_frames(gain: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a curve (any length) onto STFT frame positions."""
    if gain.size == n_frames:
        return gain.astype(np.float64)
    xp = np.linspace(0.0, 1.0, gain.size)
    x = np.linspace(0.0, 1.0, n_frames)
    return np.interp(x, xp, gain)


def apply_automation(
    curve: AutomationCurve,
    W: np.ndarray,
    H: np.ndarray,
    *,
    match_cosine: float = 0.98,
    confirm: bool = False,
) -> dict:
    """Apply an identity-keyed curve to the matching component's activation.

    The target is re-claimed against ``W`` every call, so a backend-induced
    column swap is followed rather than mis-routed.  Returns the scaled H row
    and the claim record.  ``confirm=True`` is required to resolve an
    :class:`AmbiguousClaimError`; the chosen index is then recorded on the
    curve so the decision is auditable.
    """
    try:
        claim = claim_component(curve.signature, W, match_cosine=match_cosine)
    except AmbiguousClaimError as exc:
        if not confirm or curve.confirmed_index is None:
            exc.automation_id = curve.automation_id
            raise
        idx = int(curve.confirmed_index)
        sig = component_signature(W, idx)
        cos = float(np.dot(sig, curve.signature))
        claim = {"index": idx, "cosine": cos,
                 "candidates": exc.candidates, "confirmed": True}
    idx = claim["index"]
    env = _curve_to_frames(curve.gain, H.shape[1])
    H_new = np.array(H, dtype=np.float64, copy=True)
    H_new[idx, :] = H_new[idx, :] * env
    record = {
        "automation_id": curve.automation_id,
        "preset": curve.preset,
        "target_index": idx,
        "match_cosine": claim["cosine"],
        "confirmed": claim.get("confirmed", False),
        "ambiguous_candidates": claim.get("candidates", []),
    }
    curve.history.append(record)
    return {"index": idx, "H": H_new, "env": env, "claim": record}


def write_gate(
    curves: list[AutomationCurve],
    W: np.ndarray,
    *,
    match_cosine: float = 0.98,
    confirmations: dict[str, int] | None = None,
) -> dict:
    """Pre-flight check before stems are written to disk.

    Every automation must resolve to exactly one live component, or carry an
    operator confirmation.  Returns a report with ``writable`` plus per-curve
    status; nothing is written here — the caller only proceeds when the gate is
    open.  Equal target values from a mixer comparison are not used: claims are
    decided on shape signatures.
    """
    confirmations = confirmations or {}
    report = {"writable": True, "curves": []}
    for curve in curves:
        try:
            claim = claim_component(curve.signature, W, match_cosine=match_cosine)
            report["curves"].append({
                "automation_id": curve.automation_id,
                "status": "claimed",
                "target_index": claim["index"],
                "match_cosine": claim["cosine"],
            })
        except AmbiguousClaimError as exc:
            confirmed = confirmations.get(curve.automation_id)
            if confirmed is None:
                report["writable"] = False
                report["curves"].append({
                    "automation_id": curve.automation_id,
                    "status": "needs_confirmation",
                    "candidates": exc.candidates,
                })
            else:
                sig = component_signature(W, int(confirmed))
                report["curves"].append({
                    "automation_id": curve.automation_id,
                    "status": "confirmed",
                    "target_index": int(confirmed),
                    "match_cosine": float(np.dot(sig, curve.signature)),
                })
        except AutomationTargetError as exc:
            report["writable"] = False
            report["curves"].append({
                "automation_id": curve.automation_id,
                "status": "missing",
                "error": str(exc),
            })
    return report


def two_voice_automations(
    W: np.ndarray, n_frames: int, *, indices: tuple[int, int] = (0, 1)
) -> list[AutomationCurve]:
    """The requested pair of *different* automations for two separated voices."""
    i, j = indices
    return [
        make_automation("voice_a_fade", W, i, n_frames,
                        "fade_out_exponential", label="voice A · exponential fade-out"),
        make_automation("voice_b_swell", W, j, n_frames,
                        "swell", label="voice B · swell and settle"),
    ]
