"""P0 reproduction gate of a third-party navtest subject (jevdrive/bench/agents/<model>.json), two stages on the pool.

  submit --model M [--priority -1]   smoke run on navsim/p0-smoke-logs (8 logs, 540 tokens) via jevdrive.bench, then a CPU gate
                                     job after its collect that runs `gate` (and submits the full navtest run if the smoke passes)
  gate --model M                     smoke verdict -> results/p0_subjects/<M>_smoke.json; pass -> `bench run` on all of navtest
  verdict --model M                  full-run verdict -> results/p0_subjects/<M>_full.json (|PDMS - published| <= tolerance)

Smoke pass (fixed before any run, shortlist.md): with a released per-token CSV (DiffusionDrive) the smoke PDMS on the same tokens
is within 1.0 of the published CSV's mean on those tokens; otherwise the smoke PDMS is >= published - 3. A model that fails is
reported, not tuned. Waiting: one blocking waiter on <full run>/DONE or ERROR, never a polling loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive import bench  # noqa: E402
from jevdrive.bench.models import agent_cfg  # noqa: E402

SMOKE = "navsim/p0-smoke-logs"
RES = REPO / "experiments/wm_surplus/results/p0_subjects"


def cmd_submit(a):
    d = bench.run(a.model, "navtest", subset=SMOKE, priority=a.priority)
    ids = json.loads((d / "jobs.json").read_text())
    from jevdrive.cl import pool as P
    after = [ids["collect"]] if ids.get("collect") not in (None, "done") else []
    jid = P.submit([str(REPO / ".venv/bin/python"), __file__, "gate", "--model", a.model, "--priority", str(a.priority)],
                   name=f"p0-gate-{a.model}"[:60], owner="p0-subjects", vram_gb=0.5, cpu=1, ram_gb=4, after=after,
                   log_dir=str(d / "pool" / "p0-gate"), cwd=str(REPO), priority=a.priority)
    print(json.dumps(dict(smoke_run=str(d), jobs=ids, gate=jid), indent=1))


def smoke_verdict(model: str) -> dict:
    import pandas as pd
    c = agent_cfg(model)
    d = bench.run_dir(model, "navtest", subset=SMOKE)
    s = json.loads((d / "summary.json").read_text())
    pub = c["published"]
    v = dict(model=model, smoke_run=str(d), n=s["n"], PDMS_v1=s["PDMS_v1"], EPDMS=s["EPDMS"], published=pub["value"])
    if pub.get("value") is None:
        v["rule"] = "no published navtest number: execution check only (all tokens scored)"
        v["passed"] = bool(s["missing"] == 0)
    elif pub.get("per_token_csv") and Path(pub["per_token_csv"]).exists():
        ref = pd.read_csv(pub["per_token_csv"])
        ref = ref[~ref.token.astype(str).str.startswith("average")].set_index("token")
        u = pd.read_csv(d / "units_v1.csv").set_index("token")
        common = u.index.intersection(ref.index)
        v |= dict(ref_tokens=len(common), published_csv_mean_on_smoke=100 * float(ref.loc[common, "score"].fillna(0).mean()),
                  ours_on_same=100 * float(u.loc[common, "score"].fillna(0).mean()))
        v["per_token_abs_diff_mean"] = float((u.loc[common, "score"].fillna(0) - ref.loc[common, "score"].fillna(0)).abs().mean())
        v["delta"] = v["ours_on_same"] - v["published_csv_mean_on_smoke"]
        v["rule"] = "|smoke - published CSV on the same tokens| <= 1.0 PDMS"
        v["passed"] = bool(len(common) == s["n"] and abs(v["delta"]) <= 1.0)
    else:
        v["delta"] = s["PDMS_v1"] - pub["value"]
        v["rule"] = "smoke PDMS >= published - 3"
        v["passed"] = bool(v["delta"] >= -3.0)
    return v


def cmd_gate(a):
    v = smoke_verdict(a.model)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{a.model}_smoke.json").write_text(json.dumps(v, indent=1) + "\n")
    print(json.dumps(v, indent=1))
    if not v["passed"]:
        sys.exit(f"{a.model}: smoke outside the gate ({v['rule']}); full run not submitted")
    d = bench.run(a.model, "navtest", priority=a.priority)
    print(json.dumps(dict(full_run=str(d), jobs=json.loads((d / "jobs.json").read_text())), indent=1))


def cmd_verdict(a):
    c = agent_cfg(a.model)
    d = bench.run_dir(a.model, "navtest")
    s = json.loads((d / "summary.json").read_text())
    tol = float(c.get("tolerance", 0.5))
    v = dict(model=a.model, kind=c["kind"], run=str(d), n=s["n"], missing=s["missing"], PDMS_v1=s["PDMS_v1"],
             published=s.get("published"), delta=s.get("delta_vs_published"), tolerance=tol,
             reproduced=(None if s.get("published") is None else bool(s["missing"] == 0 and abs(s.get("delta_vs_published", 99)) <= tol)),
             EPDMS_v2=s["EPDMS"], v1_subscores={k: s[k] for k in s if k.startswith("v1_")},
             v2_subscores={k: s[k] for k in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC") if k in s},
             poses_wall_h=None)
    try:
        import numpy as np
        v["poses_wall_h"] = float(np.load(d / "preds.npz")["poses_wall_s"]) / 3600
    except Exception:
        pass
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{a.model}_full.json").write_text(json.dumps(v, indent=1) + "\n")
    print(json.dumps(v, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "gate", "verdict"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--priority", type=float, default=-1.0)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
