"""op-adapt L in Bench2Drive: per-run readouts, paired judgement and the stage checklists
(todos/2026-10-01-op-adapt-L-b2d-prereg.md; definitions in section 5 there).

Reads $OPL_ROOT (default $DATA_DIR/runs/op_l_b2d) / arms/<tag>-<arm>-s<seed>/ (b2d_run --out dirs of scripts/op_arb.sh) and writes
small CSV / Markdown under $OPL_ROOT/results/. Reuses the op-drive per-run row (jevdrive.op_arb_report.drive_row) and adds the
behaviour readouts: start releases by cause, ghost stops, plan-vs-route agreement in the turn window, intent consistency, turn passes.

  .venv/bin/python -m jevdrive.op_l_b2d_report unit <tag> <arm> <seed>   append the per-run rows of one finished unit (chain calls it)
  .venv/bin/python -m jevdrive.op_l_b2d_report check stage1|stage2        the written checklists, PASS / FAIL per item
  .venv/bin/python -m jevdrive.op_l_b2d_report judge [--set dev|h]        all tables: arms, paired lines, behaviour lines, pacing
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir
from .op_arb_report import INF_KEYS, attempts, context, cross_track, drive_row, plans, record

ROOT = Path(os.environ.get("OPL_ROOT") or data_dir() / "runs" / "op_l_b2d")
RES = ROOT / "results"
FAMILY = {"drive": "ld", "lmain": "lm", "lnoint": "ln", "ldw10": "lw", "lmain1": "l1", "lmain2": "l2"}
BAD = ["coll_veh", "coll_ped", "coll_layout", "red_light", "stop_sign"]
LEFT, RIGHT = 1, 2                                   # CARLA route commands (scripts/b2d_zeroshot_agent.py)
END_M, STAND_S, TURN_WIN_M, AGREE_M, TURN_XT_M = 15.0, 1.0, 15.0, 1.0, 1.5
INTENT_BEFORE_M, INTENT_AFTER_M = 20.0, 5.0


# ------------------------------------------------------------------------------------------------ per run
def route_arrays(adir: Path):
    r = json.loads((adir / "route.json").read_text())
    xy = np.asarray(r["xy"], float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))], np.asarray(r["cmd"], int)


def cmd_runs(cmd: np.ndarray, kinds=(LEFT, RIGHT)):
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] in kinds:
            j = i
            while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
                j += 1
            out.append((i, j, int(cmd[i])))
            i = j + 1
        else:
            i += 1
    return out


def episodes(live: pd.DataFrame, s: np.ndarray):
    """Standstills of >= STAND_S after the car first moved: (onset index, end index, onset context, near the route end)."""
    if not (live.v > 1.0).any():
        return []
    st = ((live.v < 0.2) & (live.t > live.t[live.v > 1.0].min())).to_numpy()
    on = np.flatnonzero(np.diff(np.r_[0, st.astype(int)]) == 1)
    off = np.flatnonzero(np.diff(np.r_[st.astype(int), 0]) == -1)
    ctx = context(live).to_numpy()
    ri = live.ri.to_numpy()
    return [(i, j, ctx[i], bool(s[-1] - s[min(ri[i], len(s) - 1)] < END_M)) for i, j in zip(on, off) if j - i + 1 >= STAND_S * 20 - 1]


def run_extras(adir: Path) -> dict:
    st = plans(adir)
    out = {}
    if not len(st):
        return out
    live = st[~st.warm].reset_index(drop=True)
    try:
        s, cmd = route_arrays(adir)
    except (OSError, ValueError):
        return out
    eps = [e for e in episodes(live, s)]
    free = [(i, j) for i, j, c, end in eps if c == "free" and not end]
    km = max((live.v * 0.05).sum() / 1e3, 1e-6)
    out.update(ghost=len(free), ghost_per_km=round(len(free) / km, 3), free_stand_s=json.dumps([round((j - i + 1) * 0.05, 1) for i, j in free]),
               stands=len([e for e in eps if not e[3]]), stands_red=len([e for e in eps if e[2] == "red" and not e[3]]))
    # plan vs route in the turn window: next LEFT / RIGHT command within TURN_WIN_M (0 inside), moving
    w = live.cmd_k.isin([LEFT, RIGHT]) & (live.cmd_d <= TURN_WIN_M) & (live.v > 1.0)
    out.update(turn_steps=int(w.sum()), turn_agree_steps=int((w & (live["div"] <= AGREE_M)).sum()),
               turn_zone_lat_op=int((w & (live.lat == "op")).sum()), turn_v_mean=round(float(live.v[w].mean()), 2) if w.any() else np.nan)
    # intent consistency with the route (causal rule of the agent), only for arms that send one
    if "intent" in live and (live.intent > 0).any():
        zones = [(s[i] - INTENT_BEFORE_M, s[j] + INTENT_AFTER_M, 2 if k == LEFT else 3) for i, j, k in cmd_runs(cmd)]
        sp = s[np.minimum(live.ri.to_numpy(), len(s) - 1)]
        exp = np.ones(len(live), int)
        for a, b, k in zones:
            exp[(sp >= a) & (sp <= b)] = k
        out.update(intent_steps=len(live), intent_bad=int((exp != live.intent.to_numpy()).sum()), intent_turn_steps=int((live.intent > 1).sum()))
    # turn passes (per LEFT / RIGHT command run): progress passes the run end and the ground truth stays within TURN_XT_M of the route
    try:
        xt = cross_track(adir)
        ticks = [json.loads(line) for line in open(adir / "ticks.jsonl")]
        fr = np.array([t["frame"] for t in ticks if "truth" in t])
        on = st.set_index("frame").reindex(fr, method="ffill")
        ri = on.ri.to_numpy()
        runs = cmd_runs(cmd)
        ok, n_op = 0, 0
        for i, j, _ in runs:
            m = (ri >= i) & (ri <= j)
            passed = bool(np.nanmax(st.ri.to_numpy()) >= j)
            ok += int(passed and (xt[m].max() <= TURN_XT_M if m.any() else True))
            n_op += int(((on.lat == "op").to_numpy() & m).any())
        out.update(turn_runs=len(runs), turn_ok=ok, turn_runs_op=n_op)
    except (OSError, KeyError, ValueError, IndexError):
        pass
    ms = st.ms[~st.warm]
    out.update(ms_med=round(float(ms.median()), 1), ms_p99=round(float(ms.quantile(0.99)), 1))
    return out


def run_row(adir: Path) -> dict:
    r = drive_row(adir)
    r.update(run_extras(adir))
    return r


def arm_rows(tag: str, arm: str, seed: int) -> list[dict]:
    d = ROOT / "arms" / f"{tag}-{arm}-s{seed}"
    rows = []
    for rid, adir in attempts(d).items():
        rows.append({"tag": tag, "arm": arm, "seed": seed, "route": rid, **run_row(adir)})
    return rows


def unit(tag: str, arm: str, seed: int):
    RES.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(arm_rows(tag, arm, seed))
    p = RES / "per_run.csv"
    if p.exists():
        old = pd.read_csv(p)
        old = old[~((old.tag == tag) & (old.arm == arm) & (old.seed == seed))]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(p, index=False)
    u = rows[(rows.tag == tag) & (rows.arm == arm) & (rows.seed == seed)]
    line = {"tag": tag, "arm": arm, "seed": seed, "routes": len(u), "DS": round(u.DS.mean(), 1), "RC": round(u.RC.mean(), 1),
            "v_mean": round(u.v_mean.mean(), 2), "viol": int(u[BAD].sum().sum()), "blocked": int(u.blocked.sum()),
            "attempts_note": "", "t": pd.Timestamp.now().strftime("%F %T")}
    q = RES / "units.csv"
    pd.concat([pd.read_csv(q) if q.exists() else pd.DataFrame(), pd.DataFrame([line])]).to_csv(q, index=False)
    print("unit", line)


# ------------------------------------------------------------------------------------------------ pace control resolution
def final_slow(tag: str, seed: int) -> str | None:
    """The family's control for a seed: the last dbaseslow version that has been run (registration P: the one that terminated)."""
    vs = sorted(p.name[len(tag) + 1:-len(f"-s{seed}")] for p in (ROOT / "arms").glob(f"{tag}-dbaseslow*-s{seed}") if (p / "DONE").exists())
    vs.sort(key=lambda x: int(x[len("dbaseslow"):] or 1))
    return vs[-1] if vs else None


def load_all() -> pd.DataFrame:
    p = RES / "per_run.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def by_route(df: pd.DataFrame, tag: str, arm: str, col: str, how="mean") -> pd.Series:
    """Per-route value of an arm in a tag: over the TM seeds, mean (DS, rates) or sum (counts)."""
    g = df[(df.tag == tag) & (df.arm == arm)].groupby("route")[col]
    return g.mean() if how == "mean" else g.sum()


def control_rows(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    parts = []
    for seed in sorted(df[df.tag == tag].seed.unique()):
        a = final_slow(tag, int(seed))
        if a:
            parts.append(df[(df.tag == tag) & (df.arm == a) & (df.seed == seed)].assign(arm="CTRL"))
    return pd.concat(parts) if parts else pd.DataFrame(columns=df.columns)


def spec_rows(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    """spec 'arm' (its own tag), 'slow:arm' (the arm's final pace control) or 'tag/arm'."""
    if spec.startswith("slow:"):
        return control_rows(df, FAMILY.get(spec[5:], "lm"))
    if "/" in spec:
        t, a = spec.split("/")
        return df[(df.tag == t) & (df.arm == a)]
    tag = FAMILY.get(spec) or {"dnod": "ld", "dtz": "ld", "dbase": "ld", "lkd": "lm", "ltz": "lm"}[spec]
    return df[(df.tag == tag) & (df.arm == spec)]


def boot(x: np.ndarray, n=10000, seed=0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_ds(df, a: str, b: str, col="DS") -> dict:
    ra, rb = spec_rows(df, a), spec_rows(df, b)
    ga, gb = ra.groupby("route")[col].mean(), rb.groupby("route")[col].mean()
    k = ga.index.intersection(gb.index)
    if not len(k):
        return {}
    d = (ga[k] - gb[k]).to_numpy()
    m, lo, hi = boot(d)
    bad = lambda r: int(r[BAD].sum().sum())  # noqa: E731
    return {"pair": f"{a} - {b}", "routes": len(k), "d": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2),
            "better/worse/same": "%d/%d/%d" % ((d > 0.5).sum(), (d < -0.5).sum(), (np.abs(d) <= 0.5).sum()), "viol_a": bad(ra), "viol_b": bad(rb)}


def did(df, x: str) -> dict:
    """[DS(x) - DS(slow x)] - [DS(drive) - DS(slow drive)] on the routes all four share."""
    g = {k: spec_rows(df, s).groupby("route").DS.mean() for k, s in (("x", x), ("xs", f"slow:{x}"), ("d", "drive"), ("ds", "slow:drive"))}
    k = g["x"].index
    for v in g.values():
        k = k.intersection(v.index)
    if not len(k):
        return {}
    d = ((g["x"][k] - g["xs"][k]) - (g["d"][k] - g["ds"][k])).to_numpy()
    m, lo, hi = boot(d)
    return {"pair": f"DiD {x}", "routes": len(k), "d": round(m, 2), "lo": round(lo, 2), "hi": round(hi, 2)}


def ratio_diff(df, x: str, ref: str, num: str, den: str, mult=1.0, n=10000) -> dict:
    """Pooled rate num / den of two arms and its difference x - ref, route-clustered bootstrap (routes shared by both)."""
    rx, rr = spec_rows(df, x), spec_rows(df, ref)
    sx, sr = rx.groupby("route")[[num, den]].sum(), rr.groupby("route")[[num, den]].sum()
    k = sx.index.intersection(sr.index)
    if not len(k):
        return {}
    sx, sr = sx.loc[k].to_numpy(), sr.loc[k].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(len(k), size=(n, len(k)))
    f = lambda s, i: s[i, 0].sum(1) / np.maximum(s[i, 1].sum(1), 1e-9) * mult  # noqa: E731
    bx, br = f(sx, idx), f(sr, idx)
    dd = bx - br
    return {"x": round(float(sx[:, 0].sum() / max(sx[:, 1].sum(), 1e-9) * mult), 4), "ref": round(float(sr[:, 0].sum() / max(sr[:, 1].sum(), 1e-9) * mult), 4),
            "d": round(float(np.nanmean(dd)), 4), "lo": round(float(np.nanpercentile(dd, 2.5)), 4), "hi": round(float(np.nanpercentile(dd, 97.5)), 4),
            "bx_lo": round(float(np.nanpercentile(bx, 2.5)), 4), "bx_hi": round(float(np.nanpercentile(bx, 97.5)), 4),
            "br_lo": round(float(np.nanpercentile(br, 2.5)), 4), "br_hi": round(float(np.nanpercentile(br, 97.5)), 4)}


# ------------------------------------------------------------------------------------------------ tables
def arms_table(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame()
    sums = BAD + ["blocked", "timeout", "off_lane", "route_dev", "latches", "rel_signal", "rel_lead", "rel_resume", "ghost", "stands", "stands_red",
                  "turn_steps", "turn_agree_steps", "turn_runs", "turn_ok", "turn_runs_op", "intent_steps", "intent_bad"]
    means = ["DS", "RC", "v_mean", "km", "lat_op", "lat_zone", "lat_div", "lon_op", "xt_op_med", "ms_med", "ms_p99", "stop_s", "ghost_per_km"]
    rows = []
    for (tag, arm), g in df.groupby(["tag", "arm"]):
        r = {"tag": tag, "arm": arm, "runs": len(g)}
        r.update({c: int(g[c].sum()) for c in sums if c in g})
        r.update({c: round(float(g[c].mean()), 3) for c in means if c in g})
        r["turn_agree"] = round(r.get("turn_agree_steps", 0) / max(r.get("turn_steps", 0), 1), 3)
        rel = r.get("rel_signal", 0) + r.get("rel_lead", 0) + r.get("rel_resume", 0)
        r["timer_share"] = round(r.get("rel_resume", 0) / rel, 3) if rel else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def judge(which="dev"):
    RES.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if not len(df):
        print("no per-run rows")
        return
    arms = arms_table(df)
    arms.to_csv(RES / "arms.csv", index=False)
    lines, beh, pace = [], [], []
    xs = [a for a in ("lmain", "lnoint", "ldw10", "lmain1", "lmain2") if len(spec_rows(df, a)) and len(spec_rows(df, f"slow:{a}"))]
    dr = spec_rows(df, "drive")
    for x in xs:
        s2 = paired_ds(df, x, f"slow:{x}")
        s1 = paired_ds(df, x, "drive")
        rx = spec_rows(df, x)
        safe = {"lat_op": round(rx.lat_op.mean(), 3), "lat_op_drive": round(dr.lat_op.mean(), 3), "xt_op_med": round(rx.xt_op_med.mean(), 3),
                "offroute": int(rx.off_lane.sum() + rx.route_dev.sum()), "offroute_drive": int(dr.off_lane.sum() + dr.route_dev.sum()),
                "blocked": int(rx.blocked.sum()), "blocked_drive": int(dr.blocked.sum())}
        safe_ok = safe["lat_op"] >= safe["lat_op_drive"] - 0.10 and safe["xt_op_med"] <= 0.5 and safe["offroute"] <= safe["offroute_drive"] + 2 \
            and safe["blocked"] <= safe["blocked_drive"] + 2
        ok2 = bool(s2) and s2["d"] > 0 and s2["viol_a"] < s2["viol_b"]
        ok1 = bool(s1) and s1["d"] > 0 and int(s1["better/worse/same"].split("/")[0]) >= int(s1["better/worse/same"].split("/")[1])
        dd = did(df, x)
        lines.append({"arm": x, "L-S2 d": s2.get("d"), "L-S2 CI": f"[{s2.get('lo')}, {s2.get('hi')}]", "viol arm/ctrl": f"{s2.get('viol_a')}/{s2.get('viol_b')}", "L-S2": ok2,
                      "L-S1 d": s1.get("d"), "L-S1 CI": f"[{s1.get('lo')}, {s1.get('hi')}]", "b/w/s": s1.get("better/worse/same"), "L-S1": ok1,
                      "DiD": dd.get("d"), "DiD CI": f"[{dd.get('lo')}, {dd.get('hi')}]", "L-Safe": safe_ok, "safe detail": json.dumps(safe)})
    # behaviour lines vs drive (pooled rates, route-clustered CI)
    d2 = df.assign(rel_total=df.rel_signal + df.rel_lead + df.rel_resume, n_runs=1.0)
    for x in ["lmain", "lnoint", "ldw10", "lmain1", "lmain2", "dnod", "lkd", "dtz", "ltz"]:
        if not len(spec_rows(d2, x)) or not len(dr):
            continue
        b = {"arm": x}
        for name, num, den, mult in (("timer_share", "rel_resume", "rel_total", 1), ("signal_per_run", "rel_signal", "n_runs", 1),
                                     ("ghost_per_km", "ghost", "km", 1), ("stops_red_per_run", "stands_red", "n_runs", 1),
                                     ("red_light_per_run", "red_light", "n_runs", 1), ("turn_agree", "turn_agree_steps", "turn_steps", 1),
                                     ("turn_ok_rate", "turn_ok", "turn_runs", 1)):
            r = ratio_diff(d2, x, "drive", num, den, mult)
            for k2, v in r.items():
                b[f"{name}.{k2}"] = v
        beh.append(b)
    pairs = [paired_ds(df, a, b) for a, b in (("lmain", "lnoint"), ("lnoint", "dnod"), ("lkd", "lmain"), ("lmain", "dnod"), ("ltz", "dtz"), ("drive", "slow:drive"),
                                                ("drive", "dbase"), ("lmain", "lmain1"), ("lmain", "lmain2"))]
    for p in pairs:
        if p:
            lines.append({"arm": "(" + p["pair"] + ")", "L-S2 d": p["d"], "L-S2 CI": f"[{p['lo']}, {p['hi']}]", "b/w/s": p["better/worse/same"]})
    for tag in sorted(df.tag.unique()):
        for seed in sorted(df[df.tag == tag].seed.unique()):
            for f in sorted((ROOT / "arms").glob(f"{tag}-dbaseslow*-s{int(seed)}/PACE")):
                pace.append({"tag": tag, "seed": int(seed), "control": f.parent.name[len(tag) + 1:-len(f"-s{int(seed)}")], "v_ratio": float(f.read_text())})
    pd.DataFrame(lines).to_csv(RES / "paired.csv", index=False)
    pd.DataFrame(beh).to_csv(RES / "behaviour.csv", index=False)
    pd.DataFrame(pace).to_csv(RES / "pacing.csv", index=False)
    md = ["# op-adapt L in B2D (generated by jevdrive.op_l_b2d_report judge; do not edit)", "", "## arms", "", arms.to_markdown(index=False), "", "## judged lines", "",
          pd.DataFrame(lines).to_markdown(index=False), "", "## behaviour readouts (pooled, difference vs drive)", "", pd.DataFrame(beh).to_markdown(index=False), "",
          "## pacing", "", pd.DataFrame(pace).to_markdown(index=False) if pace else "-", ""]
    (RES / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------ checklists
def chk(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return bool(ok)


def check(stage: str):
    df = load_all()
    res = []
    if stage == "stage1":
        a, d = df[(df.tag == "s1") & (df.arm == "lmain")], df[(df.tag == "s1") & (df.arm == "drive")]
        res.append(chk("C4 result rows for both arms", len(a) == 1 and len(d) == 1))
        if len(a) and len(d):
            a, d = a.iloc[0], d.iloc[0]
            res.append(chk("C2 intent consistent with the route (100%), a turn seen", a.get("intent_bad", 1) == 0 and a.get("intent_turn_steps", 0) > 0,
                           f"bad {a.get('intent_bad')} of {a.get('intent_steps')}, turn steps {a.get('intent_turn_steps')}"))
            res.append(chk("C3 latency ms median <= 1.2x drive, p99 <= 1.5x", a.ms_med <= 1.2 * d.ms_med and a.ms_p99 <= 1.5 * d.ms_p99,
                           f"lmain {a.ms_med}/{a.ms_p99} drive {d.ms_med}/{d.ms_p99}"))
            res.append(chk("C4 status", str(a.status) not in ("missing",) and "Fail" not in str(a.status) and "rash" not in str(a.status), f"{a.status}"))
        adir = {arm: next(iter(attempts(ROOT / "arms" / f"s1-{arm}-s0").values()), None) for arm in ("lmain", "drive")}
        if all(adir.values()):
            st = {k: plans(v) for k, v in adir.items()}
            free = {k: v[(~v.warm) & (v.v > 2) & (v.c_lead_gap.isna() | (v.c_lead_gap > 30)) & (v.c_ped_gap.isna() | (v.c_ped_gap > 30)) & ~(v.c_tl.isin([1, 2]) & (v.c_tl_dist < 40))]
                    for k, v in st.items()}
            rs = {k: float(v.vp5.median()) if len(v) else np.nan for k, v in free.items()}
            y2 = float(np.median([abs(p[1][1]) for p in free["lmain"].op_xy])) if len(free["lmain"]) else np.nan
            res.append(chk("C5 plan v(5 s) ratio to drive in [0.8, 1.25] on free road", 0.8 <= rs["lmain"] / max(rs["drive"], 1e-6) <= 1.25, f"{rs}"))
            res.append(chk("C5 |y@2 s| median <= 0.6 m", y2 <= 0.6, f"{y2:.3f}"))
            nan = any(np.isnan(np.asarray(v.vplan.tolist(), float)).any() for v in st.values())
            ln = {k: float(np.median(np.minimum(*np.array(v.lane.tolist())[:, 1:3].T))) for k, v in st.items()}
            res.append(chk("C5 no NaN; inner lane prob median >= 0.5x drive's (D1)", not nan and ln["lmain"] >= 0.5 * ln["drive"], f"{ln}"))
            ph = {k: float(((v.lp0 > 0.5) & v.c_lead_gap.isna()).mean()) for k, v in st.items() if len(v)}
            res.append(chk("C5 phantom lead rate <= 2x drive (+0.02)", ph["lmain"] <= 2 * ph["drive"] + 0.02, f"{ph}"))
            need = {"lat", "lat_why", "div", "go", "intent"}
            res.append(chk("C6 fields", need <= set(st["lmain"].columns)))
    else:
        a, d, b = (df[(df.tag == t) & (df.arm == r) & (df.seed == 0)] for t, r in (("lm", "lmain"), ("ld", "drive"), ("ld", "dbase")))
        res.append(chk("every unit has 10 run rows", len(a) == len(d) == len(b) == 10, f"{len(a)} {len(d)} {len(b)}"))
        res.append(chk("no agent crash", not any(("Fail" in str(s) or "rash" in str(s) or s == "missing") for x in (a, d, b) for s in x.status), ""))
        res.append(chk("dbase DS in 57.7 +- 15", abs(b.DS.mean() - 57.7) <= 15, f"{b.DS.mean():.1f}"))
        res.append(chk("drive DS in 63.2 +- 15", abs(d.DS.mean() - 63.2) <= 15, f"{d.DS.mean():.1f}"))
        res.append(chk("C2 intent 100% on all routes", int(a.intent_bad.sum()) == 0 and a.intent_turn_steps.sum() > 0, f"bad {int(a.intent_bad.sum())} of {int(a.intent_steps.sum())}"))
        res.append(chk("L-Safe: lat_op >= drive - 0.10", a.lat_op.mean() >= d.lat_op.mean() - 0.10, f"{a.lat_op.mean():.3f} vs {d.lat_op.mean():.3f}"))
        res.append(chk("L-Safe: xt_op_med <= 0.5", a.xt_op_med.mean() <= 0.5, f"{a.xt_op_med.mean():.3f}"))
        res.append(chk("L-Safe: off_lane + route_dev <= drive + 2", a.off_lane.sum() + a.route_dev.sum() <= d.off_lane.sum() + d.route_dev.sum() + 2,
                       f"{a.off_lane.sum() + a.route_dev.sum()} vs {d.off_lane.sum() + d.route_dev.sum()}"))
        res.append(chk("L-Safe: blocked <= drive + 2", a.blocked.sum() <= d.blocked.sum() + 2, f"{a.blocked.sum()} vs {d.blocked.sum()}"))
        res.append(chk("speed ratio lmain / drive in [0.5, 2]", 0.5 <= a.v_mean.mean() / max(d.v_mean.mean(), 1e-6) <= 2.0, f"{a.v_mean.mean():.2f} / {d.v_mean.mean():.2f}"))
    print("ALL PASS" if all(res) else "CHECKLIST FAILED")
    return all(res)


if __name__ == "__main__":
    if sys.argv[1] == "unit":
        unit(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    elif sys.argv[1] == "judge":
        judge()
