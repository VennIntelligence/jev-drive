"""Unit sets of the benchmarks: HUGSIM scenario lists by name, NAVSIM token sets (jevdrive.data.splits) and their log shards.

HUGSIM sets (scenario paths relative to $DATA_DIR/datasets/hugsim/scenarios):
  all64    the exam's 64 scenarios (experiments/hugsim/scripts/derot_all64.txt; the headline set of every HUGSIM readout)
  spin10   the 10 PR #57 Cinque spinners (derot_spin10.txt)
  spec29   spin10 + the 19 other decision-118 stuck scenarios (experiments/op_parity/scripts/pp_hugsim_spec.txt)
  small11  the 11-scenario guard / leaderboard-audit subset (experiments/leaderboard_audit/scripts/unified_hugsim_small.txt)
  turn23   the 23 turning scenarios of all64: recorded route heading range >= 30 deg (experiments/op_probe/results/turn-gain.md),
           frozen in jevdrive/bench/lists/hugsim_turn23.txt
  <file>   any .txt list of scenario paths; <scenario stem> for one scenario; a comma list of stems
"""
from __future__ import annotations

from pathlib import Path

from .models import REPO, data_dir

HERE = Path(__file__).resolve().parent
HUGSIM_SETS = {
    "all64": REPO / "experiments/hugsim/scripts/derot_all64.txt",
    "spin10": REPO / "experiments/hugsim/scripts/derot_spin10.txt",
    "spec29": REPO / "experiments/op_parity/scripts/pp_hugsim_spec.txt",
    "small11": REPO / "experiments/leaderboard_audit/scripts/unified_hugsim_small.txt",
    "turn23": HERE / "lists/hugsim_turn23.txt",
}
TURN_DEG = 30.0


def hugsim_scenarios(name: str = "all64") -> list:
    """Scenario paths ('nuscenes/scene-0013-medium-00.yaml'), in list order."""
    if name in HUGSIM_SETS:
        return Path(HUGSIM_SETS[name]).read_text().split()
    p = Path(name)
    if p.suffix == ".txt" and p.exists():
        return p.read_text().split()
    every = {Path(s).stem: s for s in HUGSIM_SETS["all64"].read_text().split()}
    stems = [s.strip() for s in name.split(",") if s.strip()]
    missing = [s for s in stems if s not in every and not s.endswith(".yaml")]
    if missing:
        raise ValueError(f"unknown HUGSIM set / scenarios {missing} (sets: {', '.join(HUGSIM_SETS)})")
    return [every.get(s, s) for s in stems]


def route_turn_deg(routes: dict) -> dict:
    """scene -> heading range (deg) of its recorded route (routes.json of experiments/hugsim/scripts/spin_export_routes.py)."""
    import numpy as np
    return {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in routes.items()}


# ---------------------------------------------------------------- NAVSIM
NAVSIM = {"navtest": dict(data="lb_navtest", split="navtest", unit="token", cluster="log"),
          "navhard": dict(data="lb_navhard", split="navhard_two_stage", unit="group", cluster="log of the stage-1 token")}


def navsim_tokens(bench: str, subset: str = "") -> tuple:
    """(tokens, logs) of a NAVSIM bench in the op_lb row order; subset = a jevdrive.data.splits name (token or log unit) or ''."""
    import json
    import numpy as np
    mt = json.loads((data_dir() / "runs" / "op_lb" / NAVSIM[bench]["data"] / "meta.json").read_text())
    toks = np.asarray(mt["names"])
    tab = np.load(data_dir() / "runs" / "op_parity" / "cache" / NAVSIM[bench]["data"] / "tab.npz")
    assert tab["names"].tolist() == toks.tolist(), "pp_prep tab and op_lb meta disagree on the token order"
    logs = np.asarray(tab["log"])
    if subset:
        from ..data import splits
        s = splits.load(subset)
        keep = s.mask(logs) if s.unit == "log" else s.mask(toks)
        toks, logs = toks[keep], logs[keep]
    return toks, logs


def log_shards(logs, k: int) -> list:
    """k shards of whole logs (the devkit pairs consecutive frames of a log for EC), balanced by token count, deterministic."""
    import numpy as np
    u, c = np.unique(np.asarray(logs), return_counts=True)
    order = np.argsort(-c, kind="stable")
    load, out = np.zeros(k), [[] for _ in range(k)]
    for i in order:                                              # longest-processing-time first
        j = int(np.argmin(load))
        out[j].append(u[i])
        load[j] += c[i]
    return [sorted(x) for x in out]
