"""Model registry: every model the benchmarks run, by name, with how it is loaded / served, what it reads and where its
weights live (resolved from $DATA_DIR). `resolve("P2-F-s0@gimm")` returns a `Model`; `python -m jevdrive.bench models`
lists the entries.

Name syntax: `<name>[@<frames>][:<opt>]`
  frames  NAVSIM front protocol of the open-loop readouts (docs/bench.md "Frame protocols"): gimm (G, GIMM-synthesised 0.2 s
          pairs), warp (W, CPU ego-motion warp), keys (N, the 2 Hz keyframes only), real (lb_hq_navtestX only), vh140 (1.40 m
          virtual camera). HUGSIM / CARLA render their own frames, so the key of a closed-loop run ignores it.
  opt     parity models: `tsA` / `tsB` / `ts0` put the decision-191 turn selector (N7, 19 candidates around the SH30 plan) behind the
          export (navsim only): A every token, B only where the model's own 4 s heading change is >= 20 deg, 0 forced identity
          (experiments/op_parity/plans/2026-10-08-turn-selector-bench-prereg.md; SH30-F-s* only);
          `noside` masks every side / rear camera (memory arms: the memory bank) at test time (navsim plans only); `mshuf` (memory arms
          only) feeds each token the bank row of a token of another log; `lm` exports the plan through
          the lead standstill margin (jevdrive/openpilot/lead_margin.py, op_interp adapter `lm`; navsim only, HUGSIM takes it as
          the agent option `{"lead_margin": {}}`).

Families
  onnx            shipped openpilot models (cinque, lebowski, small) and op_guard candidates (an adapted ONNX without a
                  command adapter, e.g. fw-S3): navsim plans by scripts/op_lb.py run (envs/openpilot, TensorRT),
                  HUGSIM by experiments/hugsim/archive/hugsim_zs_server.py <base> [--onnx X]
  parity          op_parity arms (Cinque + lib/parity_adapter): P0 (shipped weights through the parity path, no adapter),
                  P1-init / P2-init / P3-init (untrained adapter: bias exactly 0), and every pp_train / pp_unfreeze
                  checkpoint under $DATA_DIR/runs/op_parity/runs/<tag>/ckpt-final.pt (P1-F-s0, P2-F-s0, P2H10-F-s0, HP-F-s0,
                  T1P-F-s0, PX-s0, UF-U2-s0 ...). navsim plans: torch port on the pp_prep token cache
                  (experiments/op_parity/scripts/pp_train.py PModel, fp16; UF-* arms from pixels via pp_unfreeze.py plans);
                  HUGSIM: the arm's ONNX (pp_hugsim.py onnx: trained initializers + `intent_bias` input) on the policy server
                  plus the arm's bias server (pp_hugsim.py serve, envs/op-train) fed by lib/parity_hugsim.py.
                  A checkpoint with a trajectory head next to it (thead.pt, pp_train --thead, lib/traj_head.py) is served by that head
                  on navsim: its 8 rear-axle poses are written as the prediction file (stage `poses`, no export); not on HUGSIM
  vt              vis_train arms with a memory branch or their own encoder (VT-A0 / A / B / C / W-s<seed>[-k<NN>]: checkpoints under
                  runs/op_parity/runs/ that carry memory tokens or trained vision weights): navsim plans by
                  experiments/vis_train/scripts/vt.py plans (cached slot tokens + the pre-rendered pixel cache, protocol W only);
                  `:noside` masks the memory at test time, `:mshuf` feeds the memory of a token of another log, `:sideoff` (VT-W
                  only: three-view branch, F0 from the pixel cache, CAM_L0 / CAM_R0 from lib/side_store.py) masks the two side
                  views and keeps the F0 view. VT-F / VT-F0 are plain P2 checkpoints (family parity). Not on HUGSIM
  wajepa          WA-JEPA released checkpoint ($DATA_DIR/models/wajepa): stored navtest / navhard references (its own runner,
                  experiments/top10), HUGSIM through its shipped client (zs_run agent `wajepa`, exam preset)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRAMES = ("gimm", "warp", "keys", "real", "vh140")
TS_OPTS = ("tsA", "tsB", "ts0")                     # turn selector gates (decision 191 N7 on SH30), see navsim.select_stage
VT_OPTS = ("noside", "mshuf", "sideoff")            # vis_train memory arms: memory masked / taken from a token of another log / (W) side views masked


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", Path.home() / "data"))


@dataclass(frozen=True)
class Model:
    name: str                       # registry name (P2-F-s0, cinque, fw-S3, WA-JEPA)
    family: str                     # onnx | parity | wajepa
    frames: str = ""                # navsim front protocol (default per family / tag)
    opt: str = ""                   # parity: noside | mshuf (memory arms) | lm | sg (stop gate, serving) | dn (navtest-mean bias subtracted, serving) | tsA | tsB | ts0 (turn selector)
    base: str = "cinque"            # openpilot base model (onnx family; parity arms are Cinque)
    onnx: str = ""                  # onnx family: serving ONNX ("" = the shipped file)
    ckpt: str = ""                  # parity: checkpoint (.pt); "" for P0 / *-init
    note: str = ""
    inputs: tuple = ()
    benches: tuple = ("navtest", "navhard", "hugsim")
    stored: dict = field(default_factory=dict, compare=False, hash=False)   # bench -> stored result path (references)

    @property
    def spec(self) -> str:
        """Canonical name incl. frames and opt: the identity of an open-loop run."""
        return self.name + (f"@{self.frames}" if self.frames else "") + (f":{self.opt}" if self.opt else "")

    def key(self, bench: str) -> str:
        """Run-dir name of this model on a benchmark (closed loop ignores frames / opt)."""
        if bench in ("hugsim", "b2d"):
            return self.name + (f"-{self.opt}" if self.opt in TS_OPTS else "")     # turn selector: its own run dir (served by the same ONNX)
        return self.spec.replace(":", "_")

    @property
    def shipped(self) -> bool:
        return self.family == "parity" and (self.name == "P0" or self.name.endswith("-init"))

    @property
    def unfreeze(self) -> bool:
        return self.family == "parity" and self.name.startswith("UF-")

    @property
    def thead(self) -> bool:
        """A parity checkpoint whose plan is its trajectory head (thead.pt next to ckpt-final.pt; lib/traj_head.py)."""
        return self.family == "parity" and bool(self.ckpt) and Path(self.ckpt).with_name("thead.pt").exists()


OP_INPUTS = ("road + wide camera frame pairs (12 x 128 x 256 YUV, 0.2 s context)", "desire (one-hot, none by default)",
             "traffic convention", "lateral control params / prev desired curvature")
PARITY_INPUTS = OP_INPUTS + ("P2+: ego velocity / acceleration, 4-pose history (2 Hz), NAVSIM command",
                             "P3: CAM_L0 / CAM_R0 / CAM_B0 tokens from Cinque's own frozen encoder")
STATIC = {
    "cinque": Model("cinque", "onnx", "gimm", base="cinque", note="shipped openpilot Cinque (comma v3, 382 M), ONNX / TensorRT",
                    inputs=OP_INPUTS),
    "lebowski": Model("lebowski", "onnx", "gimm", base="lebowski", note="shipped openpilot Lebowski, ONNX / TensorRT (3.3 s warp pre-roll)",
                      inputs=OP_INPUTS),
    "small": Model("small", "onnx", "gimm", base="small", note="shipped openpilot small model, ONNX / TensorRT fp32", inputs=OP_INPUTS),
    "WA-JEPA": Model("WA-JEPA", "wajepa", note="WA-JEPA released checkpoint ($DATA_DIR/models/wajepa); NAVSIM path fp32, HUGSIM bf16",
                     inputs=("CAM_L0 / F0 / R0 / B0 x 4 frames at 2 Hz", "4 history poses", "ego velocity / acceleration", "command"),
                     stored={"navtest": "runs/top10_t2/navsim/wajepa/20260926-122804/v2/2026.09.26.16.32.53.csv",
                             "navhard": "runs/op_parity/navhard/harness/wajepa",
                             "hugsim:exam": "experiments/hugsim/results/hugsim-exam/scored_wajepa.csv#wajepa"}),
}
ALIASES = {"wajepa": "WA-JEPA", "wa-jepa": "WA-JEPA", "shipped": "cinque"}


def parity_root() -> Path:
    return data_dir() / "runs" / "op_parity"


def parity_ckpt(tag: str) -> Path:
    return parity_root() / "runs" / tag / "ckpt-final.pt"


def parity_frames(tag: str) -> str:
    """Default front protocol of a parity tag: the protocol it was trained on (Stage B G / W / N, the pilot's GIMM, V's 1.40 m camera)."""
    if "-G-" in tag or re.fullmatch(r"P\d-s\d+", tag):
        return "gimm"
    if "-N-" in tag:
        return "keys"
    if tag.startswith("UF-V"):
        return "vh140"
    return "warp"


def guard_candidates() -> dict:
    f = REPO / "experiments/op_guard/candidates.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def expand(p: str) -> str:
    return os.path.expandvars(p.replace("$DATA_DIR", str(data_dir()))) if p else p


def resolve(spec: str, check: bool = False) -> Model:
    """'P2-F-s0' / 'P0@gimm' / 'P3-F-s0:noside' / 'cinque' / 'fw-S3' / 'WA-JEPA' -> Model. check: the weights must exist."""
    name, _, opt = spec.partition(":")
    name, _, frames = name.partition("@")
    name = ALIASES.get(name.lower(), name)
    if frames and frames not in FRAMES:
        raise ValueError(f"{spec}: unknown frame protocol {frames!r} (one of {FRAMES})")
    if name in STATIC:
        m = STATIC[name]
    elif re.fullmatch(r"P0|P[123]-init", name):
        m = Model(name, "parity", "warp", note="shipped Cinque through the parity path" + (" (untrained adapter, bias 0)" if name != "P0" else ""),
                  inputs=PARITY_INPUTS)
    elif name.startswith("H-") and re.fullmatch(r"H-[\w.-]+", name):
        m = Model(name, "adapt_h", ckpt=str(data_dir() / "runs/op_adapt_H/runs" / name[2:] / "ckpt-final.pt"),
                  note="op_adapt H checkpoint; no-adapter ONNX export with stream equivalence gate", inputs=OP_INPUTS, benches=("hugsim",))
    elif name in guard_candidates() and name != "shipped":
        c = guard_candidates()[name]
        if c.get("command_adapter") or c.get("route_adapter"):
            raise ValueError(f"{name}: op_guard candidate with a command / route adapter; jevdrive.bench serves plain ONNX candidates only")
        m = Model(name, "onnx", "gimm", base="cinque", onnx=expand(c["onnx"]), note=c.get("note", ""), inputs=OP_INPUTS)
    elif re.fullmatch(r"VT-(A0|A2|B2|A|B|C|W)-s\d+(-k\d+)?", name):       # A2 / B2: A / B continued under a new tag (vis_train prereg amendment 7)
        m = Model(name, "vt", "warp", ckpt=str(parity_ckpt(name)), note="vis_train checkpoint (vt.py plans: memory branch / own encoder)",
                  inputs=OP_INPUTS + ("ego velocity / acceleration, 4-pose history (2 Hz), NAVSIM command",), benches=("navtest", "navhard"))
    else:
        ck = parity_ckpt(name)
        if not ck.exists() and not re.fullmatch(r"[A-Z][\w.]*-[\w.-]+", name):
            raise ValueError(f"unknown model {spec!r}: not in the registry ({', '.join(STATIC)}), not P0 / P*-init, not an op_guard "
                             f"candidate and no checkpoint {ck}")
        m = Model(name, "parity", parity_frames(name), ckpt=str(ck), note="op_parity checkpoint (pp_train / pp_unfreeze format)",
                  inputs=PARITY_INPUTS)
    if frames:
        if m.family == "vt" and frames != "warp":
            raise ValueError(f"{spec}: vis_train arms read the protocol-W pixel cache only")
        m = replace(m, frames=frames)
    if opt and m.family == "vt":
        if opt not in VT_OPTS or m.name.startswith("VT-C-") or (opt == "sideoff" and not m.name.startswith("VT-W-")):
            raise ValueError(f"{spec}: vis_train memory arms (A0 / A / B / W) take {VT_OPTS[:2]}, W also sideoff; C has no memory channel")
        m = replace(m, opt=opt)
    elif opt:
        if m.family != "parity" or opt not in ("noside", "mshuf", "lm", "sg", "dn") + TS_OPTS:
            raise ValueError(f"{spec}: option {opt!r} is only defined for parity models (noside, mshuf, lm, sg, dn, tsA, tsB, ts0)")
        if opt in TS_OPTS and not re.fullmatch(r"SH30-F-s\d+", m.name):
            raise ValueError(f"{spec}: the turn selector options are defined for SH30-F-s* only")
        m = replace(m, opt=opt, benches=("navtest", "navhard", "hugsim")) if opt in TS_OPTS else replace(m, opt=opt)
    if check:
        if m.ckpt and not Path(m.ckpt).exists():
            raise FileNotFoundError(f"{spec}: checkpoint {m.ckpt} missing")
        if m.onnx and not Path(m.onnx).exists():
            raise FileNotFoundError(f"{spec}: ONNX {m.onnx} missing")
    return m


def listing() -> list:
    """Rows for `python -m jevdrive.bench models`: static entries, P0 / *-init, op_guard candidates, parity checkpoints on disk."""
    rows = [dict(name=m.name, family=m.family, frames=m.frames, weights=m.onnx or "shipped", note=m.note) for m in STATIC.values()]
    rows += [dict(name=n, family="parity", frames="warp", weights="shipped", note=resolve(n).note) for n in ("P0", "P1-init", "P2-init", "P3-init")]
    for n, c in guard_candidates().items():
        if n != "shipped" and not (c.get("command_adapter") or c.get("route_adapter")):
            rows.append(dict(name=n, family="onnx", frames="gimm", weights=expand(c["onnx"]), note=c.get("note", "")[:80]))
    runs = parity_root() / "runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir()):
            if (d / "ckpt-final.pt").exists() and not d.name.startswith(("smoke", "eqtest", "time-")):
                rows.append(dict(name=d.name, family="parity", frames=parity_frames(d.name), weights=str(d / "ckpt-final.pt"), note=""))
    for ck in sorted((data_dir() / "runs/op_adapt_H/runs").glob("*/ckpt-final.pt")):
        rows.append(dict(name=f"H-{ck.parent.name}", family="adapt_h", frames="", weights=str(ck), note="HUGSIM; checked no-adapter export"))
    return rows
