"""Night queue 3, lane B: closed-loop tables (todos/2026-09-26-night-queue-3.md, rules 9-10 and section CL).

Reads every arm under runs/nq3/b/arms/<arm>/s<seed>/ (a b2d_run --out directory; the score of a route is the official
record of its finished attempt, results.json) and writes to runs/nq3/b/results/:

  per_route.csv   arm, seed, route, scenario, family, group, status, DS, RC, success (Bench2Drive merge_route_json:
                  Completed / Perfect and no infraction other than min_speed), collisions, attempts
  arms.md / .csv  per arm: routes finished / requested, DS over the requested routes (a missing route scores 0, as the
                  official merge divides by 220) and over the finished ones, SR, SR per hazard group, retries
  paired.md / .csv  the registered paired differences, route-grouped bootstrap (10 000), seeds averaged per route first

Hazard groups (decision 38's families, jevdrive.tfv6_rules.FAMILY): sudden = its five sudden-hazard families;
yield = unprotected_turn + merge_lane_change + emergency_vehicle; obstacle = obstacle_bypass; other = routine_control.
Every table carries CL1 (the expert track replayed through P7) as the execution-layer ceiling row; P7 arms and the
author-executor arms (CL10) are only shown side by side (rule 9 (c)).

    .venv/bin/python -m jevdrive.nq3_cl_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir

N_BOOT = 10_000
GROUPS = {"sudden": ("vru_emerging", "vru_crossing", "cut_in", "lead_hard_brake", "junction_violator"),
          "yield": ("unprotected_turn", "merge_lane_change", "emergency_vehicle"),
          "obstacle": ("obstacle_bypass",), "other": ("routine_control",)}
OBSTACLE_SCEN = ("Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays", "ParkedObstacle",
                 "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays")
AUTHOR = ("tfv6", "bridgedrive", "simlingo", "blue")
LABEL = {"cl1": "CL1 expert track via P7 (ceiling)", "cl2": "CL2 openpilot Cinque native plan", "cl3": "CL3 Cinque + ridge_late",
         "cl4": "CL4 Cinque + M-C", "cl5": "CL5 Cinque + Q2 bypass head", "cl5d": "CL5d openpilot native + bypass desire",
         "cl6": "CL6 E5 student (image-plane)", "cl7": "CL7 openpilot Lebowski native plan", "cl8": "CL8 Alpamayo 1.5",
         "mc_real0": "mc_real0 (Q4a real-frame zero constraint)", "tfv6": "TFv6 (author executor)",
         "bridgedrive": "BridgeDrive (author executor)", "simlingo": "SimLingo (author executor)", "blue": "BLUE (author executor)"}
# (a, b, route subset, registered reading) for a - b
PAIRS = (("cl4", "cl3", "all", "criterion 1: sudden-group SR CI low > 0 (3 seeds) and total DS CI low > -3"),
         ("cl5", "cl4", "obstacle", "criterion 2: obstacle SR CI low > 0 (3 seeds), collisions not above CL4"),
         ("cl5d", "cl5", "obstacle", "criterion 3 (second half): obstacle SR >= CL5 - 10 pp"),
         ("cl5", "cl2", "obstacle", "descriptive"), ("cl2", "cl1", "all", "criterion 4: gap to the ceiling, no gate"),
         ("cl3", "cl2", "all", "descriptive"), ("cl6", "cl4", "all", "descriptive"), ("cl7", "cl2", "all", "descriptive"),
         ("mc_real0", "cl4", "all", "Q4a arm vs M-C, same routes and seeds"),
         ("bridgedrive", "tfv6", "all", "criterion 5: author column, closed-loop check of T3"),
         ("blue", "simlingo", "all", "criterion 5: author column, closed-loop check of T3"))


def root() -> Path:
    return data_dir() / "runs" / "nq3" / "b"


def route_table() -> pd.DataFrame:
    from .tfv6_rules import route_table as rt
    t = rt()
    fam2grp = {f: g for g, fs in GROUPS.items() for f in fs}
    t["group"] = t.family.map(fam2grp)
    return t.set_index("route_id")


def _record(adir: Path) -> dict | None:
    try:
        recs = json.loads((adir / "results.json").read_text())["_checkpoint"]["records"]
    except (OSError, ValueError, KeyError):
        return None
    return recs[0] if recs else None


def collect(base: Path | None = None) -> pd.DataFrame:
    base = base or root() / "arms"
    rt = route_table()
    rows = []
    for d in sorted(base.glob("*/s*")):
        arm, seed = d.parent.name, int(d.name[1:])
        req = json.loads((d / "requested.json").read_text()) if (d / "requested.json").exists() else None
        done = {p.stem: json.loads(p.read_text()) for p in (d / "done").glob("*.json")}
        for rid in (req or sorted(done)):
            n_att = len(list((d / "attempts" / rid).glob("*"))) if (d / "attempts" / rid).exists() else 0
            r = {"arm": arm, "seed": seed, "route": rid, "attempts": n_att, "finished": rid in done}
            rec = _record(d / "attempts" / rid / str(done[rid]["attempt"])) if rid in done else None
            if rec is not None:
                inf = rec.get("infractions", {})
                r.update(status=rec["status"], DS=float(rec["scores"]["score_composed"]),
                         RC=float(rec["scores"]["score_route"]),
                         success=rec["status"] in ("Completed", "Perfect")
                         and not any(len(v) for k, v in inf.items() if k != "min_speed_infractions"),
                         collisions=sum(len(inf.get(k, [])) for k in ("collisions_layout", "collisions_pedestrian",
                                                                      "collisions_vehicle")))
            else:
                r.update(status="missing", DS=0.0, RC=0.0, success=False, collisions=np.nan)
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.join(rt[["scenario", "family", "group"]], on="route")
    return df


def boot_mean(x: np.ndarray, n: int = N_BOOT, seed: int = 0):
    x = np.asarray(x, float)
    if not len(x):
        return np.nan, np.nan, np.nan
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arm, seed), g in df.groupby(["arm", "seed"], sort=False):
        f = g[g.finished]
        r = {"arm": arm, "seed": seed, "label": LABEL.get(arm, arm), "executor": "author" if arm in AUTHOR else "P7",
             "requested": len(g), "finished": int(g.finished.sum()), "DS": g.DS.mean(),
             "DS_finished": f.DS.mean() if len(f) else np.nan, "SR": g.success.mean(), "retries": int((g.attempts - 1).clip(0).sum())}
        for grp in GROUPS:
            s = g[g.group == grp]
            r[f"SR_{grp}"] = s.success.mean() if len(s) else np.nan
            r[f"n_{grp}"] = len(s)
        r["collisions"] = g.collisions.sum()
        out.append(r)
    t = pd.DataFrame(out)
    order = {a: i for i, a in enumerate(LABEL)}
    return t.sort_values(["arm", "seed"], key=lambda c: c.map(order) if c.name == "arm" else c) if len(t) else t


def paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b, subset, reading in PAIRS:
        A, B = df[df.arm == a], df[df.arm == b]
        if not len(A) or not len(B):
            continue
        seeds = sorted(set(A.seed) & set(B.seed))
        if not seeds:
            continue
        A, B = A[A.seed.isin(seeds)], B[B.seed.isin(seeds)]
        per = lambda X: X.groupby("route").agg(DS=("DS", "mean"), SR=("success", "mean"), col=("collisions", "mean"),  # noqa: E731
                                               group=("group", "first"), scenario=("scenario", "first"))
        pa, pb = per(A), per(B)
        common = pa.index.intersection(pb.index)
        pa, pb = pa.loc[common], pb.loc[common]
        if subset == "obstacle":
            keep = pa.scenario.isin(OBSTACLE_SCEN)
            pa, pb = pa[keep], pb[keep]
        for scope in ("all",) + tuple(GROUPS):
            m = np.ones(len(pa), bool) if scope == "all" else (pa.group == scope).to_numpy()
            if not m.any():
                continue
            for metric in ("DS", "SR"):
                d = (pa[metric] - pb[metric]).to_numpy()[m]
                mean, lo, hi = boot_mean(d)
                rows.append({"a": a, "b": b, "seeds": ",".join(map(str, seeds)), "subset": subset, "scope": scope,
                             "metric": metric, "n_routes": int(m.sum()), "diff": mean, "lo": lo, "hi": hi,
                             "collisions_a": float(pa.col.to_numpy()[m].sum()), "collisions_b": float(pb.col.to_numpy()[m].sum()),
                             "reading": reading})
    return pd.DataFrame(rows)


def write(out: Path | None = None) -> None:
    out = out or root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = collect()
    if not len(df):
        return
    df.to_csv(out / "per_route.csv", index=False)
    at = arm_table(df)
    at.to_csv(out / "arms.csv", index=False)
    pt = paired(df)
    pt.to_csv(out / "paired.csv", index=False)
    fmt = lambda x: "" if pd.isna(x) else f"{x:.1f}"  # noqa: E731
    lines = ["# Night queue 3, CL arms (generated by jevdrive.nq3_cl_report; do not edit)", "",
             "DS over the requested routes (missing = 0, the official merge); SR = Bench2Drive merge_route_json; "
             "group SR in % (n routes in the header). P7 and author-executor rows are side by side only (rule 9 (c)).", "",
             "| arm | seed | executor | finished / requested | DS | DS (finished) | SR % | sudden | yield | obstacle | other | "
             "collisions | retries |", "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in at.itertuples():
        lines.append(f"| {r.label} | {r.seed} | {r.executor} | {r.finished} / {r.requested} | {fmt(r.DS)} | {fmt(r.DS_finished)} | "
                     f"{fmt(100 * r.SR)} | {fmt(100 * r.SR_sudden)} ({r.n_sudden}) | {fmt(100 * r.SR_yield)} ({r.n_yield}) | "
                     f"{fmt(100 * r.SR_obstacle)} ({r.n_obstacle}) | {fmt(100 * r.SR_other)} ({r.n_other}) | "
                     f"{fmt(r.collisions)} | {r.retries} |")
    if len(pt):
        lines += ["", "## Paired differences (a - b, same routes and seeds; route-grouped bootstrap 10 000, seeds averaged "
                  "per route first; SR in pp)", "",
                  "| a - b | seeds | subset | scope | metric | n | diff [95% CI] | collisions a / b | reading |",
                  "|:--|:--|:--|:--|:--|--:|:--|:--|:--|"]
        for r in pt.itertuples():
            k = 100 if r.metric == "SR" else 1
            lines.append(f"| {r.a} - {r.b} | {r.seeds} | {r.subset} | {r.scope} | {r.metric} | {r.n_routes} | "
                         f"{k * r.diff:+.1f} [{k * r.lo:+.1f}, {k * r.hi:+.1f}] | {r.collisions_a:.0f} / {r.collisions_b:.0f} | "
                         f"{r.reading} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write()
