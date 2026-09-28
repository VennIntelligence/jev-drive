"""op-arb readouts: openpilot closed-loop diagnosis (phase 1) and the arbitration arms (phase 2).
Plan: todos/2026-09-28-op-closedloop.md; write-up: research/openpilot-closedloop-integration.md.

Reads runs/op_arb/arms/p<k>-<arm>/ (b2d_run --out dirs of scripts/op_arb.sh; per finished attempt results.json,
plans.jsonl with one record per openpilot step, ticks.jsonl) and writes small CSV / Markdown tables:

  diag   start: standstill steps by ground-truth context (free / lead <= 15 m / red-yellow light <= 30 m), openpilot's
         signals there and their AUC for "should go"; cruise: plan speed vs measured speed on free road; hazard: AUC of
         openpilot's signals for "must slow" while moving; turn: plan heading vs route heading in turn zones, with the
         route desire and without (the twin session); native junction passes
  eval   per route and per arm: DS, RC, SR, infractions, share of moving steps where an openpilot constraint binds,
         infractions by the source binding at that moment, paired DS vs base

    .venv/bin/python -m jevdrive.op_arb_report diag|eval [--root DIR] [--out DIR]
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir

INF_KEYS = {"collisions_vehicle": "coll_veh", "collisions_pedestrian": "coll_ped", "collisions_layout": "coll_layout",
            "red_light": "red_light", "stop_infraction": "stop_sign", "outside_route_lanes": "off_lane",
            "route_dev": "route_dev", "vehicle_blocked": "blocked", "route_timeout": "timeout",
            "yield_emergency_vehicle_infractions": "yield_ev", "scenario_timeouts": "scen_timeout",
            "min_speed_infractions": "min_speed"}
OP_SRC = ("lead", "plan", "latch", "op")


def root() -> Path:
    return data_dir() / "runs" / "op_arb"


def auc(pos, neg) -> float:
    """P(signal of a positive > signal of a negative), ties 0.5 (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]
    r = pd.Series(allv).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def attempts(arm_dir: Path) -> dict:
    out = {}
    for p in sorted((arm_dir / "done").glob("*.json")):
        out[p.stem] = arm_dir / "attempts" / p.stem / str(json.loads(p.read_text())["attempt"])
    return out


def record(adir: Path) -> dict:
    try:
        rec = json.loads((adir / "results.json").read_text())["_checkpoint"]["records"][0]
    except (OSError, ValueError, KeyError, IndexError):
        return {"status": "missing"}
    inf = rec.get("infractions", {})
    r = {"status": rec["status"], "DS": float(rec["scores"]["score_composed"]), "RC": float(rec["scores"]["score_route"])}
    for k, v in INF_KEYS.items():
        r[v] = len(inf.get(k, []))
    r["SR"] = int(rec["status"] in ("Completed", "Perfect") and not any(len(v) for k, v in inf.items() if k != "min_speed_infractions"))
    r["_inf"] = inf
    return r


def plans(adir: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(adir / "plans.jsonl")] if (adir / "plans.jsonl").exists() else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ctx = pd.DataFrame([r.get("ctx") or {} for r in rows])
    for c in ("lead_gap", "lead_v", "ped_gap", "ped_v", "tl_dist", "tl", "junc"):
        df["c_" + c] = ctx[c] if c in ctx else np.nan
    df["x5"] = [p[3][0] for p in df.op_xy]
    for k, t in enumerate((0, 1, 2, 3, 5)):
        df[f"vp{t}"] = [v[k] for v in df.vplan]
    df["lp0"] = [x[0] for x in df.lp]
    df["lead_x"] = [x[0][0] for x in df.lead]
    df["lead_v_op"] = [x[0][2] for x in df.lead]
    df["gas0"], df["gas2"] = [x[0] for x in df.gas], [x[1] for x in df.gas]      # meta gas / brake press at t = 0 and 2 s
    df["brk0"], df["brk2"] = [x[0] for x in df.brk], [x[1] for x in df.brk]
    df["hb"] = [max(x) for x in df.hb3]
    df["cmd_k"] = [c[0] for c in df.cmd]
    df["cmd_d"] = [c[1] if c[1] is not None else np.nan for c in df.cmd]
    return df


def context(df: pd.DataFrame) -> pd.Series:
    """Ground-truth context of a step: red (red / yellow light <= 30 m ahead), lead (vehicle in the path <= 15 m),
    ped (walker in the path <= 20 m), free (none of these within 30 m), else other."""
    red = df.c_tl.isin([1, 2]) & (df.c_tl_dist <= 30)
    lead = df.c_lead_gap <= 15
    ped = df.c_ped_gap <= 20
    free = ~red & ~(df.c_lead_gap <= 30) & ~(df.c_ped_gap <= 30) & ~(df.c_tl.isin([1, 2]) & (df.c_tl_dist <= 40))
    return pd.Series(np.select([red, lead, ped, free], ["red", "lead", "ped", "free"], "other"), index=df.index)


def load(arm_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs, steps = [], []
    for rid, adir in attempts(arm_dir).items():
        r = record(adir)
        r.pop("_inf", None)
        recs.append({"route": rid, **r})
        p = plans(adir)
        if len(p):
            p["route"] = rid
            steps.append(p)
    return pd.DataFrame(recs), (pd.concat(steps, ignore_index=True) if steps else pd.DataFrame())


# ------------------------------------------------------------------------------------------------ diagnosis
def diag(rootdir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    arms = {d.name.split("-", 1)[1]: load(d) for d in sorted(rootdir.glob("arms/p1-*"))}
    res = {}
    # ---- 1. standstill
    rows = []
    for arm, (_, st) in arms.items():
        if not len(st):
            continue
        s = st[(st.v < 0.2) & ~st.warm].copy()
        s["ctx"] = context(s)
        s["arm"] = arm
        rows.append(s)
    ss = pd.concat(rows, ignore_index=True)
    sig = {"plan v@1s": "vp1", "plan v@3s": "vp3", "plan v@5s": "vp5", "plan x@5s": "x5", "action accel": "act_a",
           "lead_prob": "lp0", "lead x": "lead_x", "gas_press@0": "gas0", "gas_press@2s": "gas2", "brake_press@0": "brk0",
           "brake_press@2s": "brk2", "vision v": "pose_v"}
    tab = []
    for (arm, c), g in ss.groupby(["arm", "ctx"]):
        tab.append({"arm": arm, "context": c, "steps": len(g), "routes": g.route.nunique(),
                    **{k: round(float(g[v].median()), 3) for k, v in sig.items()}})
    t1 = pd.DataFrame(tab)
    t1.to_csv(out / "standstill.csv", index=False)
    go, wait = ss[ss.ctx == "free"], ss[ss.ctx.isin(["lead", "red"])]
    t1a = pd.DataFrame([{"signal": k, "AUC go vs wait": round(auc(go[v], wait[v]), 3), "n go": len(go), "n wait": len(wait),
                         "AUC go vs lead": round(auc(go[v], ss[ss.ctx == "lead"][v]), 3),
                         "AUC go vs red": round(auc(go[v], ss[ss.ctx == "red"][v]), 3)} for k, v in sig.items()])
    t1a.to_csv(out / "standstill_auc.csv", index=False)
    res["standstill"], res["standstill_auc"] = t1, t1a
    # ---- 2. cruising on free road: does the plan keep speed?
    rows = []
    for arm, (_, st) in arms.items():
        if not len(st):
            continue
        m = st[(st.v > 1.0) & ~st.warm & ~st.zone].copy()
        m["ctx"] = context(m)
        m = m[m.ctx == "free"]
        m["bin"] = pd.cut(m.v, [1, 2, 4, 6, 8, 12], labels=["1-2", "2-4", "4-6", "6-8", "8-12"])
        for b, g in m.groupby("bin", observed=True):
            rows.append({"arm": arm, "v bin (m/s)": b, "steps": len(g), "v": round(g.v.median(), 2),
                         "vision v": round(g.pose_v.median(), 2), "plan v@0": round(g.vp0.median(), 2),
                         "plan v@3s": round(g.vp3.median(), 2), "plan v@5s": round(g.vp5.median(), 2),
                         "plan v@5s - v@0": round((g.vp5 - g.vp0).median(), 2), "action accel": round(g.act_a.median(), 2)})
    t2 = pd.DataFrame(rows)
    t2.to_csv(out / "cruise.csv", index=False)
    res["cruise"] = t2
    # ---- 3. hazard sensitivity while moving
    rows = []
    for arm, (_, st) in arms.items():
        if not len(st):
            continue
        m = st[(st.v > 2.0) & ~st.warm].copy()
        m["ctx"] = context(m)
        ttc = m.c_lead_gap / np.maximum(m.v - m.c_lead_v, 0.1)
        m["must_lead"] = (m.c_lead_gap < 30) & (ttc < 4)
        m["must_red"] = m.c_tl.isin([1, 2]) & (m.c_tl_dist < 30)
        m["must_ped"] = m.c_ped_gap < 20
        free = m.ctx == "free"
        m["decel"] = m.vp0 - m.vp3            # the plan's own speed drop over 3 s
        m["neg_a"] = -m.act_a
        for lab in ("must_lead", "must_red", "must_ped"):
            pos = m[m[lab]]
            for k, v in {"plan speed drop 0-3 s": "decel", "-action accel": "neg_a", "lead_prob": "lp0",
                         "brake_press@0": "brk0", "brake_press@2s": "brk2", "hard_brake": "hb"}.items():
                rows.append({"arm": arm, "event": lab[5:], "signal": k, "n event": len(pos), "n free": int(free.sum()),
                             "routes": pos.route.nunique(), "AUC": round(auc(pos[v], m[free][v]), 3),
                             "median event": round(float(pos[v].median()), 3) if len(pos) else np.nan,
                             "median free": round(float(m[free][v].median()), 3)})
    t3 = pd.DataFrame(rows)
    t3.to_csv(out / "hazard_auc.csv", index=False)
    res["hazard"] = t3
    # ---- 4. turns: plan heading vs route heading
    rows = []
    for arm, (_, st) in arms.items():
        if not len(st) or "tw_xy" not in st:
            continue
        m = st[(st.v > 1.0) & ~st.warm].copy()
        turn = m.cmd_k.isin([1, 2]) & (m.cmd_d <= 20)
        straight = m.cmd_k.isin([3, 4]) | m.cmd_k.isna()
        for name, g in (("turn zone", m[turn]), ("elsewhere", m[~turn & straight])):
            if not len(g):
                continue
            r_ang, o_ang, t_ang, side = [], [], [], []
            for _, r in g.iterrows():
                base = np.array(r.base_xy)
                for pts, acc in ((r.op_xy, o_ang), (r.tw_xy, t_ang)):
                    p = np.array(pts)[2]                   # 3 s
                    acc.append(math.degrees(math.atan2(p[1], max(p[0], 1e-3))))
                p = np.array(r.op_xy)[2]
                d = float(np.hypot(*p))
                ba = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[[[0, 0]], base], axis=0), axis=1))]
                q = np.array([np.interp(d, ba, np.r_[0, base[:, k]]) for k in range(2)])
                r_ang.append(math.degrees(math.atan2(q[1], max(q[0], 1e-3))))
                side.append(1 if r.cmd_k == 1 else -1 if r.cmd_k == 2 else 0)
            r_ang, o_ang, t_ang, side = map(np.array, (r_ang, o_ang, t_ang, side))
            row = {"arm": arm, "where": name, "steps": len(g), "routes": g.route.nunique(),
                   "route bearing @3s-arc (deg, median |.|)": round(float(np.median(np.abs(r_ang))), 1),
                   "|plan - route| (deg)": round(float(np.median(np.abs(o_ang - r_ang))), 1),
                   "|twin - route| (deg)": round(float(np.median(np.abs(t_ang - r_ang))), 1)}
            if name == "turn zone":
                row["plan turns >= half route bearing"] = round(float(np.mean(o_ang * np.sign(r_ang) >= 0.5 * np.abs(r_ang))), 3)
                row["twin turns >= half route bearing"] = round(float(np.mean(t_ang * np.sign(r_ang) >= 0.5 * np.abs(r_ang))), 3)
                row["P(turn desire_pred) plan"] = round(float(np.mean([dp[0][1] + dp[0][2] for dp in g.dp])), 3)
                row["P(turn desire_pred) twin"] = round(float(np.mean([dp[0][1] + dp[0][2] for dp in g.tw_dp])), 3)
            rows.append(row)
    t4 = pd.DataFrame(rows)
    t4.to_csv(out / "turn.csv", index=False)
    res["turn"] = t4
    # ---- 5. per route outcomes of the phase-1 arms
    t5 = pd.concat([r.assign(arm=a) for a, (r, _) in arms.items() if len(r)], ignore_index=True)
    first = {}
    for arm, (_, st) in arms.items():
        for rid, g in st.groupby("route") if len(st) else ():
            mv = g[g.v > 0.5]
            first[(arm, rid)] = round(float(mv.t.iloc[0] - g.t.iloc[0]), 1) if len(mv) else np.nan
    t5["t_first_move_s"] = [first.get((a, r), np.nan) for a, r in zip(t5.arm, t5.route)]
    t5.to_csv(out / "p1_routes.csv", index=False)
    res["routes"] = t5
    md = ["# op-arb phase 1 diagnosis (generated by jevdrive.op_arb_report diag; do not edit)", ""]
    for k, t in res.items():
        md += [f"## {k}", "", t.to_markdown(index=False), ""]
    (out / "diag.md").write_text("\n".join(md))
    return res


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(rootdir: Path, out: Path, phase: str = "p2") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows, share = [], []
    for d in sorted(rootdir.glob(f"arms/{phase}-*")):
        arm = d.name.split("-", 1)[1]
        for rid, adir in attempts(d).items():
            r = record(adir)
            inf = r.pop("_inf", {})
            st = plans(adir)
            row = {"arm": arm, "route": rid, **r}
            if len(st):
                live = st[~st.warm]
                mv = live[live.v > 0.5]
                dist = (live.v * 0.05)
                row.update(op_bind_share=round(float(live.src.isin(OP_SRC).mean()), 3),
                           op_bind_dist_share=round(float(dist[live.src.isin(OP_SRC)].sum() / max(dist.sum(), 1e-6)), 3),
                           moving_share=round(float(len(mv) / max(len(live), 1)), 3),
                           latch_share=round(float(live.latch.mean()), 3),
                           t_first_move_s=round(float(mv.t.iloc[0] - st.t.iloc[0]), 1) if len(mv) else np.nan)
                # infractions by the source binding at the nearest step (ground-truth rear axle is not in plans.jsonl:
                # use the steps' frames through ticks.jsonl)
                row["inf_by_src"] = json.dumps(attribute(adir, inf, st))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_route.csv", index=False)
    agg = df.groupby("arm").agg(routes=("route", "nunique"), DS=("DS", "mean"), RC=("RC", "mean"), SR=("SR", "mean"),
                                coll_veh=("coll_veh", "sum"), coll_ped=("coll_ped", "sum"), coll_layout=("coll_layout", "sum"),
                                red_light=("red_light", "sum"), stop_sign=("stop_sign", "sum"), blocked=("blocked", "sum"),
                                timeout=("timeout", "sum"), route_dev=("route_dev", "sum"), off_lane=("off_lane", "sum"),
                                op_bind_share=("op_bind_share", "mean"), op_bind_dist_share=("op_bind_dist_share", "mean"),
                                t_first_move_s=("t_first_move_s", "median")).round(3).reset_index()
    agg.to_csv(out / "arms.csv", index=False)
    pair = []
    if "base" in set(df.arm):
        b = df[df.arm == "base"].set_index("route")
        for arm, g in df[df.arm != "base"].groupby("arm"):
            g = g.set_index("route")
            common = g.index.intersection(b.index)
            dd = (g.loc[common, "DS"] - b.loc[common, "DS"]).to_numpy()
            rng = np.random.default_rng(0)
            boot = [rng.choice(dd, len(dd)).mean() for _ in range(10000)] if len(dd) else [np.nan]
            pair.append({"arm - base": arm, "n routes": len(common), "mean dDS": round(float(dd.mean()), 1) if len(dd) else np.nan,
                         "95% CI (route bootstrap)": "[%.1f, %.1f]" % tuple(np.percentile(boot, [2.5, 97.5])),
                         "routes better / worse / same": "%d / %d / %d" % ((dd > 0.5).sum(), (dd < -0.5).sum(), (np.abs(dd) <= 0.5).sum())})
    tp = pd.DataFrame(pair)
    tp.to_csv(out / "paired.csv", index=False)
    md = ["# op-arb evaluation (generated by jevdrive.op_arb_report eval; do not edit)", "",
          "## arms", "", agg.to_markdown(index=False), "", "## paired vs base", "", tp.to_markdown(index=False), "",
          "## per route", "", df.drop(columns=["inf_by_src"], errors="ignore").to_markdown(index=False), ""]
    (out / "eval.md").write_text("\n".join(md))
    return {"arms": agg, "paired": tp, "routes": df}


def attribute(adir: Path, inf: dict, st: pd.DataFrame) -> dict:
    """Each located infraction -> the arbitration source of the nearest step (by ground-truth position in ticks.jsonl)."""
    try:
        ticks = [json.loads(line) for line in open(adir / "ticks.jsonl")]
    except OSError:
        return {}
    fr = np.array([t["frame"] for t in ticks])
    xy = np.array([t["truth"][:2] if "truth" in t else [np.nan, np.nan] for t in ticks], float)
    src = st.set_index("frame").src
    out = {}
    for kind, items in inf.items():
        if kind == "min_speed_infractions":
            continue
        for text in items:
            m = re.search(r"x=(-?[\d.]+), y=(-?[\d.]+)", str(text))
            who = "?"
            if m and np.isfinite(xy).any():
                f = fr[int(np.nanargmin(np.linalg.norm(xy - [float(m.group(1)), float(m.group(2))], axis=1)))]
                k = src.index.searchsorted(f)
                who = str(src.iloc[min(k, len(src) - 1)]) if len(src) else "?"
            out.setdefault(INF_KEYS.get(kind, kind), []).append(who)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("diag", "eval"))
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--phase", default="p2")
    a = ap.parse_args()
    rd = Path(a.root) if a.root else root()
    out = Path(a.out) if a.out else rd / "results"
    res = diag(rd, out) if a.step == "diag" else evaluate(rd, out, a.phase)
    for k, t in res.items():
        print(f"== {k}\n{t.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
