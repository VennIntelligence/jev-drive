"""Analysis of the SimLingo catalogue experiment (todos/2026-09-25-simlingo-catalogue).

Reads the b2d_run.py directories $ROOT/{official,simlingo}/seed*/ and writes to --out:
  per_route.csv   one row per (arm, seed, route); columns aligned with research/results/b2d-family/public_routes.csv
  summary.json    per-arm scores under both merge scripts, paired deltas with route-cluster bootstrap CIs,
                  attribution of the delta to the tick cap (D1), the completion threshold (D2) and the merge script (D3)
  results.md      the same as tables
  figs            simlingo-catalogue-waterfall / -routes (PDF + PNG) via research/plot_style.py

Scoring follows each tree's shipped tools/merge_route_json.py:
  official  DS = sum(DS)/220, SR = successes/220, a route with no record counts 0
  simlingo  records with status 'Failed - Agent crashed' dropped, DS and SR divided by the routes that remain
Success is Bench2Drive's definition: status Completed/Perfect and no infraction other than min_speed_infractions.

The D1 counterfactual re-scores a simlingo-arm run as if the 4000-tick cap had been on, from the same trajectory:
RC at tick 4000 (rc_trace.jsonl), penalties of the criterion events with frame <= that tick's frame, and a route
not completed by then fails with TickRuntime. Penalty factors are Bench2Drive's PENALTY_VALUE_DICT
(outside-lane and min-speed events carry no penalty there). Routes that ended before tick 4000 keep their record,
and the recomputed penalty is checked against the recorded one on them.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PENALTY = {"COLLISION_PEDESTRIAN": .5, "COLLISION_VEHICLE": .6, "COLLISION_STATIC": .65,
           "TRAFFIC_LIGHT_INFRACTION": .7, "STOP_INFRACTION": .8, "SCENARIO_TIMEOUT": .7,
           "YIELD_TO_EMERGENCY_VEHICLE": .7}
CAP = 4000
ARMS = ("official", "simlingo")
N_ROUTES = 220


def route_meta(xml: Path) -> pd.DataFrame:
    rows = []
    for r in ET.parse(xml).getroot().iter("route"):
        sc = r.find("scenarios")
        rows.append({"route_id": r.get("id"), "town": r.get("town"),
                     "scenario": sc[0].get("type") if sc is not None and len(sc) else ""})
    return pd.DataFrame(rows)


def success(rec: dict) -> bool:
    return rec["status"] in ("Completed", "Perfect") and not any(
        v for k, v in rec["infractions"].items() if k != "min_speed_infractions")


def read_trace(adir: Path) -> list:
    f = adir / "rc_trace.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def counterfactual_cap(adir: Path, rec: dict, ticks: int) -> dict:
    """Score of this trajectory under the official 4000-tick cap (see module docstring)."""
    out = {"cf_ds": rec["scores"]["score_composed"], "cf_rc": rec["scores"]["score_route"],
           "cf_success": success(rec), "cf_status": rec["status"], "cf_check": np.nan}
    ev = adir / "criterion_events.json"
    events = json.loads(ev.read_text()).get("events", []) if ev.exists() else []
    pen = float(np.prod([PENALTY.get(e["type"], 1.0) for e in events]))
    out["cf_check"] = abs(pen - rec["scores"]["score_penalty"])  # recomputation vs record, whole route
    if ticks is None or ticks <= CAP:
        return out
    tr = [r for r in read_trace(adir) if r[0] <= CAP]
    if not tr:
        out.update(cf_ds=np.nan, cf_rc=np.nan, cf_success=False, cf_status="no_trace")
        return out
    last = max(tr, key=lambda r: r[0])
    rc, frame = last[2], last[1]
    pen = float(np.prod([PENALTY.get(e["type"], 1.0) for e in events if e["frame"] is not None and e["frame"] <= frame]))
    completed = rc >= 100
    n_inf = sum(1 for e in events if e["frame"] is not None and e["frame"] <= frame
                and e["type"] not in ("ROUTE_COMPLETION", "MIN_SPEED_INFRACTION"))
    out.update(cf_rc=rc, cf_ds=rc * pen, cf_success=completed and n_inf == 0,
               cf_status=("Completed" if n_inf else "Perfect") if completed else "Failed - TickRuntime")
    return out


def threshold_completion(adir: Path) -> float:
    """Completion percentage when RouteCompletionTest declared success (before its override to 100); <= 99 means
    only the 90 % threshold could have granted it."""
    rows = [r for r in read_trace(adir) if len(r) == 5]
    return rows[0][4] if rows else np.nan


def load(root: Path, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for sdir in sorted((root / arm).glob("seed*")):
            seed = int(sdir.name[4:])
            for rid in meta.route_id:
                row = {"arm": arm, "seed": seed, "route_id": rid, "status": "missing", "ds": np.nan, "rc": np.nan,
                       "penalty": np.nan, "success": False, "ticks": np.nan, "wall_s": np.nan, "attempts": 0}
                att = sorted((sdir / "attempts" / rid).glob("*"), key=lambda p: int(p.name)) if (
                    sdir / "attempts" / rid).exists() else []
                row["attempts"] = len(att)
                done = sdir / "done" / f"{rid}.json"
                if done.exists():
                    adir = sdir / "attempts" / rid / str(json.loads(done.read_text()).get("attempt", len(att)))
                    res = adir / "results.json"
                    recs = json.loads(res.read_text())["_checkpoint"]["records"] if res.exists() else []
                    if recs:
                        rec = recs[-1]
                        rr = json.loads((adir / "route_result.json").read_text())
                        ticks = (rr.get("profile") or {}).get("ticks")
                        row.update(status=rec["status"], ds=rec["scores"]["score_composed"],
                                   rc=rec["scores"]["score_route"], penalty=rec["scores"]["score_penalty"],
                                   success=success(rec), ticks=ticks, wall_s=rr.get("wall_s"),
                                   duration_game=rec["meta"].get("duration_game"),
                                   **{"n_" + k: len(v) for k, v in rec["infractions"].items()},
                                   **counterfactual_cap(adir, rec, ticks), rc_before_completion=threshold_completion(adir))
                rows.append(row)
    df = pd.DataFrame(rows).merge(meta, on="route_id", how="left")
    df["thr_granted"] = df.rc_before_completion <= 99  # official needs > 99
    return df


def merge_scores(g: pd.DataFrame, script: str, ds="ds", succ="success", rc="rc") -> dict:
    """One run (one arm x seed, 220 rows) scored by one of the two shipped merge scripts."""
    if script == "official":
        return {"DS": g[ds].fillna(0).sum() / N_ROUTES, "SR": 100 * g[succ].sum() / N_ROUTES,
                "RC": g[rc].fillna(0).sum() / N_ROUTES, "n": N_ROUTES}
    keep = g[(g.status != "Failed - Agent crashed") & (g.status != "missing")]
    return {"DS": keep[ds].mean(), "SR": 100 * keep[succ].mean(), "RC": keep[rc].mean(), "n": len(keep)}


def per_route(df: pd.DataFrame, arm: str, col: str, crash_drop=False) -> pd.Series:
    """Seed-averaged per-route value; crash_drop reproduces the simlingo script (crashed runs excluded)."""
    d = df[df.arm == arm].copy()
    if col in ("ds", "rc", "cf_ds", "cf_rc"):
        d[col] = d[col].fillna(0)
    if col in ("success", "cf_success"):
        d[col] = d[col].fillna(False).astype(float)
    if crash_drop:
        d = d[(d.status != "Failed - Agent crashed") & (d.status != "missing")]
    return d.groupby("route_id")[col].mean().astype(float)


def boot(x: np.ndarray, n=10000, seed=0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(df: pd.DataFrame) -> dict:
    out = {"runs": {}, "arms": {}, "paired": {}, "attribution": {}, "noise": {}}
    for (arm, seed), g in df.groupby(["arm", "seed"]):
        out["runs"][f"{arm}/seed{seed}"] = {
            "official_merge": merge_scores(g, "official"), "simlingo_merge": merge_scores(g, "simlingo"),
            "missing": int((g.status == "missing").sum()), "crashed": int((g.status == "Failed - Agent crashed").sum()),
            "status": g.status.value_counts().to_dict(), "ticks_gt_cap": int((g.ticks > CAP).sum()),
            "thr_granted": int(g.thr_granted.sum())}
        if arm == "simlingo":
            out["runs"][f"{arm}/seed{seed}"]["cf_cap_official_merge"] = merge_scores(
                g, "official", ds="cf_ds", succ="cf_success", rc="cf_rc")
    for arm in ARMS:
        a = {}
        for col, name in (("ds", "DS"), ("rc", "RC"), ("success", "SR")):
            v = per_route(df, arm, col).values * (100 if col == "success" else 1)
            a[name] = boot(v)
        out["arms"][arm] = a
        runs = [r["official_merge"]["DS"] for k, r in out["runs"].items() if k.startswith(arm)]
        out["noise"][arm] = {"DS_per_seed": runs, "DS_sd_across_seeds": float(np.std(runs, ddof=1)) if len(runs) > 1 else None}
    # paired contrasts only over routes that have a record in every run of both arms
    ran = df[df.status != "missing"].groupby("route_id").apply(lambda g: set(zip(g.arm, g.seed)))
    need = set(zip(df.arm, df.seed))
    common = sorted(r for r, s_ in ran.items() if s_ == need)
    out["n_paired_routes"] = len(common)
    for col, name, k in (("ds", "DS", 1), ("rc", "RC", 1), ("success", "SR", 100)):
        d = (per_route(df, "simlingo", col)[common] - per_route(df, "official", col)[common]).values * k
        out["paired"][name + " (S-O, both official merge)"] = boot(d)
    # headline as each side reports it: O official merge vs S simlingo merge
    o = per_route(df, "official", "ds")[common]
    out["paired"]["DS headline (S simlingo merge - O official merge)"] = (
        float(df[df.arm == "simlingo"].groupby("seed").apply(lambda g: merge_scores(g, "simlingo")["DS"]).mean()
              - df[df.arm == "official"].groupby("seed").apply(lambda g: merge_scores(g, "official")["DS"]).mean()))
    # attribution on DS, official merge unless stated
    s = per_route(df, "simlingo", "ds")[common]
    s_cf = per_route(df, "simlingo", "cf_ds")[common]
    out["attribution"] = {
        "D1 tick cap (S - S@cap4000)": boot((s - s_cf).values),
        "rest: S@cap4000 - O (D2 + rerun noise)": boot((s_cf - o).values),
        "D3 merge script (S simlingo merge - S official merge)": float(
            df[df.arm == "simlingo"].groupby("seed").apply(lambda g: merge_scores(g, "simlingo")["DS"] - merge_scores(g, "official")["DS"]).mean()),
        "D3 on O runs": float(
            df[df.arm == "official"].groupby("seed").apply(lambda g: merge_scores(g, "simlingo")["DS"] - merge_scores(g, "official")["DS"]).mean()),
        "SR: D1 (S - S@cap4000)": boot(100 * (per_route(df, "simlingo", "success")[common]
                                               - per_route(df, "simlingo", "cf_success")[common]).values),
        "D2 threshold-granted completions (S runs)": int(df[df.arm == "simlingo"].thr_granted.sum()),
        "D2 threshold-granted completions (O runs, should be 0)": int(df[df.arm == "official"].thr_granted.sum()),
        "cf penalty recomputation max abs error (routes <= cap)": float(
            df[(df.arm == "simlingo") & (df.ticks <= CAP)].cf_check.max()),
    }
    return out


def fmt(t) -> str:
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if isinstance(t, (tuple, list)) and len(t) == 3 else (
        f"{t:.2f}" if isinstance(t, float) else str(t))


def write_md(res: dict, path: Path):
    L = ["| run | DS (official merge) | SR | RC | DS (simlingo merge) | SR | n | crashed | missing | >4000 ticks | thr-90 completions |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in res["runs"].items():
        o, s = r["official_merge"], r["simlingo_merge"]
        L.append(f"| {k} | {o['DS']:.2f} | {o['SR']:.1f} | {o['RC']:.2f} | {s['DS']:.2f} | {s['SR']:.1f} | {s['n']} | "
                 f"{r['crashed']} | {r['missing']} | {r['ticks_gt_cap']} | {r['thr_granted']} |")
    L += ["", "| arm | DS | RC | SR (%) |", "|---|---:|---:|---:|"]
    for arm, a in res["arms"].items():
        L.append(f"| {arm} | {fmt(a['DS'])} | {fmt(a['RC'])} | {fmt(a['SR'])} |")
    L += ["", "| paired contrast | mean [95% CI] |", "|---|---:|"]
    L += [f"| {k} | {fmt(v)} |" for k, v in res["paired"].items()]
    L += ["", "| attribution | value |", "|---|---:|"]
    L += [f"| {k} | {fmt(v)} |" for k, v in res["attribution"].items()]
    path.write_text("\n".join(L) + "\n")


def figures(df: pd.DataFrame, res: dict, out: Path, repo: Path):
    sys.path.insert(0, str(repo / "research"))
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    runs = res["runs"]
    mean = lambda arm, key, sub="DS": float(np.mean([r[key][sub] for k, r in runs.items() if k.startswith(arm)]))
    o = mean("official", "official_merge")
    s_cf = mean("simlingo", "cf_cap_official_merge")
    s = mean("simlingo", "official_merge")
    s_sl = mean("simlingo", "simlingo_merge")
    steps = [("Official\n(measured)", o, None), ("+ threshold 90\n+ rerun noise", s_cf, "d"), ("+ no 4000-tick\ncap (D1)", s, "d"),
             ("+ SimLingo\nmerge (D3)", s_sl, "d"), ("SimLingo copy\n(measured)", s_sl, None)]
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.2))
    prev = 0
    for i, (lab, v, kind) in enumerate(steps):
        if kind is None:
            ax.bar(i, v, color=ps.BASELINE if i == 0 else ps.PALETTE["blue"], width=.62)
        else:
            lo, hi = sorted((prev, v))
            ax.bar(i, hi - lo, bottom=lo, color=ps.PALETTE["vermillion"] if v >= prev else ps.PALETTE["green"], width=.62)
        ax.text(i, v + .6, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        prev = v
    ax.set_xticks(range(len(steps)), [s_[0] for s_ in steps], fontsize=6.5)
    ax.set_ylabel("Driving Score (mean of 3 runs)")
    lo = min(v for _, v, _ in steps)
    ax.set_ylim(max(0, lo - 12), max(v for _, v, _ in steps) + 5)
    ps.bars(ax)
    fig.tight_layout(pad=.3)
    ps.save(fig, out / "simlingo-catalogue-waterfall")
    plt.close(fig)

    t_s = df[df.arm == "simlingo"].groupby("route_id").ticks.median()
    d = per_route(df, "simlingo", "ds") - per_route(df, "official", "ds")
    j = pd.DataFrame({"ticks_s": t_s, "d": d}).dropna()
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.2))
    ax.scatter(j.ticks_s / 20, j.d, s=6, color=ps.PALETTE["blue"], alpha=.7, linewidths=0)
    ax.axvline(CAP / 20, color=ps.PALETTE["vermillion"], linewidth=.7, linestyle="--")
    ax.text(CAP / 20, ax.get_ylim()[1], " 4000-tick cap", color=ps.PALETTE["vermillion"], fontsize=7, va="top")
    ps.zero_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Route duration in the SimLingo-copy arm (s, median of 3 runs)")
    ax.set_ylabel(r"$\Delta$DS per route (S $-$ O)")
    fig.tight_layout(pad=.3)
    ps.save(fig, out / "simlingo-catalogue-routes")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding official/ and simlingo/")
    ap.add_argument("--routes", required=True, help="bench2drive220.xml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(a.root), route_meta(Path(a.routes)))
    pr = df.assign(method="SimLingo-" + df.arm, run="seed" + df.seed.astype(str), source="jev-rerun")
    lead = ["method", "run", "route_id", "ds", "rc", "success", "source"]
    pr[lead + [c for c in pr.columns if c not in lead]].to_csv(out / "per_route.csv", index=False)
    res = analyse(df)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    write_md(res, out / "results.md")
    figures(df, res, out, Path(a.repo))
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
