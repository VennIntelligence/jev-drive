"""Night queue 4 OPL: tables of the op_native_launch arms (openpilot native plan + TCP launch partner, P7).
Registration: todos/2026-09-26-night-queue-3.md, CL section, the [OPL] entries (written before any of these numbers).

Reads every arm under runs/nq4/opl/arms/<arm>/s<seed>/ (a b2d_run --out directory) with jevdrive.nq3_cl_report's
collector (official record of the finished attempt: DS, RC, Bench2Drive SR, hazard groups) and adds the shared-control
readout from each finished attempt's ticks.jsonl (the agent logs "driver" / "w_model" / "partner_why" every tick):

  partner_tick_share   ticks on which the partner's control has weight (w_model < 1: partner or blend) / all ticks
  model_dist_share     distance driven while w_model = 1 (speed x 0.05 s per tick) / all distance
  t_move_s             first tick above 0.5 m/s, from the route start
  handovers            partner -> model hand-backs; why: ticks per partner reason (warmup / standstill / model_not_ready)
  infractions by driver  each located infraction goes to the driver of the nearest tick (as zeroshot_b2d_junctions.py)

Writes runs/nq4/opl/results/{per_route.csv, arms.csv, paired.csv, summary.md}; the paired rows against lane B's CL2 /
CL7 / CL5d (same routes, same seed) are descriptive.

    .venv/bin/python -m jevdrive.nq4_opl_report                        # tables
    .venv/bin/python -m jevdrive.nq4_opl_report accept <arm dir> ...   # the registered acceptance gates -> JSON
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import nq3_cl_report as R
from .common import data_dir

LABEL = {"cl2p": "CL2p openpilot Cinque native + TCP launch", "cl7p": "CL7p openpilot Lebowski native + TCP launch",
         "cl5dp": "CL5dp openpilot native + bypass desire + TCP launch"}
PAIRS = (("cl2p", "cl2", "all"), ("cl7p", "cl7", "all"), ("cl5dp", "cl5d", "obstacle"), ("cl5dp", "cl5", "obstacle"),
         ("cl7p", "cl2p", "all"))


def root() -> Path:
    return data_dir() / "runs" / "nq4" / "opl"


def shared_control(adir: Path) -> dict:
    """The per-tick shared-control readout of one finished attempt."""
    ticks = [json.loads(line) for line in open(adir / "ticks.jsonl")]
    if not ticks:
        return {}
    w = np.array([float(t.get("w_model", 1.0)) for t in ticks])
    v = np.array([max(float(t["v"]), 0.0) for t in ticks])
    drv = [t.get("driver", "model") for t in ticks]
    step = v * 0.05
    total = float(step.sum()) or 1.0
    why = {}
    for t in ticks:
        if t.get("partner_why"):
            why[t["partner_why"]] = why.get(t["partner_why"], 0) + 1
    moving = np.flatnonzero(v > 0.5)
    t0 = float(ticks[0]["t"])
    model = np.flatnonzero(w >= 1.0)
    r = {"ticks": len(ticks), "partner_tick_share": float(np.mean(w < 1.0)), "model_dist_share": float(step[w >= 1.0].sum() / total),
         "model_dist_m": float(step[w >= 1.0].sum()), "dist_m": float(step.sum()),
         "t_move_s": float(ticks[moving[0]]["t"] - t0) if len(moving) else np.nan,
         "t_first_model_s": float(ticks[model[0]]["t"] - t0) if len(model) else np.nan,
         "handovers": int(np.sum((w[1:] >= 1.0) & (w[:-1] < 1.0))), "why": json.dumps(why)}
    try:
        rec = json.loads((adir / "results.json").read_text())["_checkpoint"]["records"][0]
    except (OSError, ValueError, KeyError, IndexError):
        return r
    txy = np.array([t["truth"][:2] if "truth" in t else [np.nan, np.nan] for t in ticks], float)
    by = {}
    for kind, items in rec.get("infractions", {}).items():
        if kind == "min_speed_infractions":
            continue
        for text in items:
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", str(text))
            who = "?"
            if m and np.isfinite(txy).any():
                who = drv[int(np.nanargmin(np.linalg.norm(txy - [float(m.group(1)), float(m.group(2))], axis=1)))]
            by.setdefault(who, []).append(kind)
    r["infractions_by_driver"] = json.dumps(by)
    r["blend_collisions"] = sum(k.startswith("collisions") for k in by.get("blend", []))
    return r


def collect(base: Path) -> pd.DataFrame:
    df = R.collect(base)
    if not len(df):
        return df
    rows = []
    for x in df.itertuples():
        d = base / x.arm / f"s{x.seed}"
        done = d / "done" / f"{x.route}.json"
        sc = shared_control(d / "attempts" / x.route / str(json.loads(done.read_text())["attempt"])) if x.finished else {}
        rows.append(sc)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    t = R.arm_table(df).set_index(["arm", "seed"])
    f = df[df.finished]
    g = f.groupby(["arm", "seed"])
    t["label"] = [LABEL.get(a, R.LABEL.get(a, a)) for a, _ in t.index]
    t["partner_tick_share"] = g.apply(lambda x: (x.partner_tick_share * x.ticks).sum() / x.ticks.sum())
    t["model_dist_share"] = g.apply(lambda x: x.model_dist_m.sum() / max(x.dist_m.sum(), 1e-9))
    t["routes_model_drove"] = g.apply(lambda x: int((x.model_dist_m >= 10).sum()))
    t["t_move_median_s"] = g.t_move_s.median()
    t["blend_collisions"] = g.blend_collisions.sum()
    return t.reset_index()


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive paired differences against lane B's arms (runs/nq3/b/arms), same routes and seeds."""
    b = R.collect(R.root() / "arms")
    cols = ["arm", "seed", "route", "DS", "success", "group", "scenario"]
    both = pd.concat([df[cols]] + ([b[cols]] if len(b) else []), ignore_index=True)
    rows = []
    for a, c, subset in PAIRS:
        A, B = both[both.arm == a], both[both.arm == c]
        seeds = sorted(set(A.seed) & set(B.seed))
        if not seeds:
            continue
        per = lambda X: X[X.seed.isin(seeds)].groupby("route").agg(  # noqa: E731
            DS=("DS", "mean"), SR=("success", "mean"), group=("group", "first"), scenario=("scenario", "first"))
        pa, pb = per(A), per(B)
        common = pa.index.intersection(pb.index)
        pa, pb = pa.loc[common], pb.loc[common]
        if subset == "obstacle":
            keep = pa.scenario.isin(R.OBSTACLE_SCEN)
            pa, pb = pa[keep], pb[keep]
        for metric in ("DS", "SR"):
            mean, lo, hi = R.boot_mean((pa[metric] - pb[metric]).to_numpy())
            rows.append({"a": a, "b": c, "seeds": ",".join(map(str, seeds)), "subset": subset, "metric": metric,
                         "n_routes": len(pa), "diff": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def write(out: Path | None = None) -> None:
    out = out or root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = collect(root() / "arms")
    if not len(df):
        return
    df.to_csv(out / "per_route.csv", index=False)
    at = arm_table(df)
    at.to_csv(out / "arms.csv", index=False)
    pt = paired(df)
    pt.to_csv(out / "paired.csv", index=False)
    fmt = lambda x, k=1: "" if pd.isna(x) else f"{k * x:.1f}"  # noqa: E731
    lines = ["# Night queue 4 OPL: openpilot native + TCP launch partner (generated by jevdrive.nq4_opl_report; do not edit)",
             "", "DS over the requested routes (missing = 0); SR = Bench2Drive merge_route_json; P7 execution layer. "
             "partner ticks = share of ticks on which TCP's control has weight (partner or blend), the registered readout.", "",
             "| arm | seed | finished / requested | DS | DS (finished) | SR % | sudden | yield | obstacle | other | collisions | "
             "partner ticks % | model distance % | routes model drove >= 10 m | median t_move s | blend collisions | retries |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in at.itertuples():
        lines.append(f"| {r.label} | {r.seed} | {r.finished} / {r.requested} | {fmt(r.DS)} | {fmt(r.DS_finished)} | {fmt(r.SR, 100)} | "
                     f"{fmt(r.SR_sudden, 100)} ({r.n_sudden}) | {fmt(r.SR_yield, 100)} ({r.n_yield}) | {fmt(r.SR_obstacle, 100)} "
                     f"({r.n_obstacle}) | {fmt(r.SR_other, 100)} ({r.n_other}) | {fmt(r.collisions)} | {fmt(r.partner_tick_share, 100)} | "
                     f"{fmt(r.model_dist_share, 100)} | {r.routes_model_drove} | {fmt(r.t_move_median_s)} | {fmt(r.blend_collisions)} | "
                     f"{r.retries} |")
    if len(pt):
        lines += ["", "## Paired differences against lane B (a - b, same routes and seed; route bootstrap 10 000; descriptive)", "",
                  "| a - b | seeds | subset | metric | n | diff [95% CI] |", "|:--|:--|:--|:--|--:|:--|"]
        for r in pt.itertuples():
            k = 100 if r.metric == "SR" else 1
            lines.append(f"| {r.a} - {r.b} | {r.seeds} | {r.subset} | {r.metric} | {r.n_routes} | "
                         f"{k * r.diff:+.1f} [{k * r.lo:+.1f}, {k * r.hi:+.1f}] |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def accept(dirs: list[str]) -> dict:
    """The registered acceptance gates ([OPL] 2026-09-26 entry) on smoke arm directories (b2d_run --out dirs), per arm:
    G1 every requested route finished with plans; G2 >= all-but-one routes above 0.5 m/s within 20 s; G3 the model drove
    >= 10 m on >= all-but-two routes; G5 no collision while blending. G4 (rule 8) is scripts/nq4_opl_check.py."""
    res = {}
    for d in map(Path, dirs):
        req = json.loads((d / "requested.json").read_text())
        rows = []
        for rid in req:
            done = d / "done" / f"{rid}.json"
            if not done.exists():
                rows.append({"route": rid, "finished": False})
                continue
            adir = d / "attempts" / rid / str(json.loads(done.read_text())["attempt"])
            rec = R._record(adir)
            n_plans = sum(1 for _ in open(adir / "plans.jsonl")) if (adir / "plans.jsonl").exists() else 0
            rows.append(dict(route=rid, finished=rec is not None and n_plans > 0, n_plans=n_plans,
                             DS=rec["scores"]["score_composed"] if rec else None, RC=rec["scores"]["score_route"] if rec else None,
                             status=rec["status"] if rec else None, **shared_control(adir)))
        t = pd.DataFrame(rows)
        n = len(req)
        fin = t[t.finished] if "finished" in t else t.iloc[:0]
        g = {"G1_infrastructure": bool(len(fin) == n),
             "G2_launch_20s": int((fin.t_move_s <= 20).sum()) if len(fin) else 0,
             "G3_model_drove_10m": int((fin.model_dist_m >= 10).sum()) if len(fin) else 0,
             "G5_blend_collisions": int(fin.blend_collisions.sum()) if len(fin) and "blend_collisions" in fin else 0}
        g["pass"] = bool(g["G1_infrastructure"] and g["G2_launch_20s"] >= n - 1 and g["G3_model_drove_10m"] >= n - 2
                         and g["G5_blend_collisions"] == 0)
        g.update(routes=n, partner_tick_share=float((fin.partner_tick_share * fin.ticks).sum() / max(fin.ticks.sum(), 1)),
                 model_dist_share=float(fin.model_dist_m.sum() / max(fin.dist_m.sum(), 1e-9)),
                 DS_mean=float(fin.DS.mean()) if len(fin) else None, RC_mean=float(fin.RC.mean()) if len(fin) else None,
                 per_route=t.replace({np.nan: None}).to_dict("records"))
        res[str(d)] = g
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "accept":
        print(json.dumps(accept(sys.argv[2:]), indent=1, default=str))
    else:
        write()
