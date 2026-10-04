"""Command-adapter hook of the guard set: how the guard feeds a route command to a candidate that has a command channel.

Shipped openpilot has no command channel (docs/openpilot-interface.md, `command.channel: none`), so for a candidate whose
candidates.json entry has `"command_adapter": null` the open-loop lines `drift` and `negatives` run the plain model (no command
exists). A candidate with a command channel (image-drawn command, route-polyline input, intent token, ...) names an adapter class as
a dotted path `"package.module:Class"` (importable with the repo root on sys.path); the guard then asks that adapter for plans under
three conditions on the same frames:

  (a) no command                       -> `drift` (vs the original model's plan) and the baseline of `negatives`
  (b) the correct command              (the logged route of the frame, `Command.poly` = `poly_pos` of the negatives sample)
  (c) a negative command               (lib/route_neg.py: a route into an exit / side road that does not exist, onto the oncoming
                                        side, or a U-turn; `Command.poly` = `poly` of experiments/op_route_cmd/results/negatives_sample.npz)

The adapter owns everything model-specific: how the command is encoded (overlay drawn into every history frame, polyline tensor,
intent bias), how the frames are rendered for it, and which runtime serves the model. The guard owns frame selection and scoring.
Plans come back as openpilot's own plan MDN mean (n, 33, 15) in the calibrated camera frame (x fwd, y right, channel 11 = yaw),
exactly what the plain-model paths return, so the guard converts and scores them with the same code.

No adapter exists yet: `StubAdapter` below documents the contract and raises. The first command candidate (merged fine-tune,
tmp/2026-10-05-merged-finetune-plan.md) implements `CommandAdapter` and registers itself in candidates.json.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

Kind = Literal["none", "correct", "negative"]


@dataclass(frozen=True)
class Frame:
    """One evaluation frame. domain: nav | carla | wod. token: NAVSIM token / CARLA sample token / WOD `<sequence>-<frame>`.
    source: where the guard read the plain-model frame from (e.g. 'op_img_cmd/ft/bank/skytrain#123', 'op_lb/lb_guardneg#7'), so an
    adapter that renders its own images can find the same sample."""
    domain: str
    token: str
    source: str = ""
    extra: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Command:
    """kind: none | correct | negative. poly: (k, 2) route polyline in the t0 ego frame (rear axle, x fwd, y left, metres; the
    navigation-level route of experiments/op_route_cmd, 10 m vertices up to 150 m), None for `none`. meta: e.g. the negative's
    kind / tier from lib/route_neg.py, or a discrete exit class for adapters that take one."""
    kind: Kind
    poly: np.ndarray | None = None
    meta: dict = field(default_factory=dict, compare=False)


@runtime_checkable
class CommandAdapter(Protocol):
    """What a command candidate must provide. Constructed once per line run as `Class(candidate, gpu=..., cpus=...)` with the
    resolved candidates.json entry (`name`, `onnx`, `command_adapter`, plus any adapter-specific keys the entry carries)."""

    def plans(self, frames: Sequence[Frame], commands: Sequence[Command]) -> np.ndarray:
        """Plan MDN mean (len(frames), 33, 15), openpilot calib frame, one plan per (frame, command) pair, zero model state and the
        board's history protocol (the guard compares against the original model run on the same protocol). Must be deterministic."""
        ...

    def describe(self) -> dict:
        """Provenance for the line file: encoding, renderer, runtime, versions."""
        ...


class StubAdapter:
    """Placeholder registered by nobody: documents the contract, refuses to run. A line that meets it writes status `stub`."""

    def __init__(self, candidate: dict, gpu: int = 0, cpus: str = ""):
        self.candidate = candidate

    def plans(self, frames, commands):
        raise NotImplementedError("no command adapter implemented yet: see experiments/op_guard/scripts/cmd_adapter.py")

    def describe(self):
        return {"adapter": "stub"}


def load(candidate: dict, gpu: int = 0, cpus: str = "") -> CommandAdapter | None:
    """The candidate's adapter instance, or None when it has no command channel (`command_adapter` null / missing)."""
    spec = candidate.get("command_adapter")
    if not spec:
        return None
    mod, _, cls = spec.partition(":")
    if not cls:
        raise SystemExit(f"command_adapter {spec!r}: expected 'package.module:Class'")
    ad = getattr(importlib.import_module(mod), cls)(candidate, gpu=gpu, cpus=cpus)
    if not isinstance(ad, CommandAdapter):
        raise SystemExit(f"{spec} does not implement CommandAdapter (plans, describe)")
    return ad
