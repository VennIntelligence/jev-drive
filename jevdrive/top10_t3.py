"""Top-10 T3: BridgeDrive and BLUE on the P5 v1 BA pairs, re-recorded with their own rigs
([T3] in todos/2026-09-26-top10-intersection.md; pre-registration 5.1-5.3 there).

  need       the worlds the BA exam references and the last tick k each needs -> runs/top10_t3/need.json, ids.txt
  check-det  re-recorded expert vs the original recording, tick by tick (and the camera-tick grid)
  check-bd   BridgeDrive: the 5 Hz shadow vs the author's run_step on every tick (two recordings of one world)
  check-blue BLUE: the runner's shortcut vs the author's run_step on every tick; cache vs fresh setup
  blue-plan  the offline BLUE runner's plan (world, attempt dir, referenced ticks)
  judge      p5_exam.exam unchanged + elicit_e4's per-pair window; creep and gate-open frames singled out

Readouts (scalar speed at 2 s, as the exam's Delta): BridgeDrive target speed (expectation over its classes), the
decoded scalar it drives with (brake threshold applied), waypoint speed 1.75 -> 2.0 s; BLUE speed-waypoint speed
1.75 -> 2.0 s (10 points at 0.25 s).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import p5_pairs as P
from . import p5v1
from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "top10-exams"
BD = {"BridgeDrive target speed": "ts", "BridgeDrive target speed (decoded scalar)": "ts_scalar",
      "BridgeDrive waypoint speed 2 s": "wp2"}
BLUE = {"BLUE waypoint speed 2 s": "wp2"}
CREEP = {"bd": 1100, "blue": 800}          # author stuck thresholds in 20 Hz frames (config_closed_loop / config_simlingo)
STILL = 0.1                                # both authors' "stopped" speed, m/s


def root(*p) -> Path:
    d = data_dir() / "runs" / "top10_t3" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _refs() -> dict:
    """{world id: sorted referenced ticks k} over the BA exam's pair and null frames."""
    obs, null, _ = p5v1.load("ba")
    ks = {}
    for df, cols in ((obs, ("fn_plus", "fn_minus")), (null, ("fn_plus", "fn_null"))):
        for c in cols:
            for rid, k in zip(df[c].str.split("-").str[0], df.k):
                ks.setdefault(rid, set()).add(int(k))
    return {r: sorted(v) for r, v in ks.items()}


def need():
    refs = _refs()
    order = [v for v in p5v1.variants(p5v1.cases()) if v in refs]
    (root() / "need.json").write_text(json.dumps({r: refs[r][-1] for r in order}))
    (root() / "refs.json").write_text(json.dumps(refs))
    (root() / "ids.txt").write_text(",".join(order))
    log.info("%d worlds, %d referenced frames, %d ticks to record", len(order), sum(len(v) for v in refs.values()),
             sum(refs[r][-1] + 3 for r in order))


def attempt(gen: Path, rid: str) -> Path | None:
    """The finished attempt of a re-recorded world: the one b2d_run's done/<rid>.json names, else (a world still
    running or never finished) None; without a done/ directory (checks), the attempt with the most pose lines."""
    done = gen / "done" / (rid + ".json")
    if done.exists():
        a = gen / "attempts" / rid / str(json.loads(done.read_text())["attempt"])
        return a if (a / "t3_summary.json").exists() else None
    if (gen / "done").exists():
        return None
    best, n = None, -1
    for a in sorted((gen / "attempts" / rid).glob("*")):
        if not (a / "t3_summary.json").exists():
            continue
        m = sum(1 for _ in open(a / "pose.jsonl"))
        if m > n:
            best, n = a, m
    return best


def _pose(adir: Path) -> pd.DataFrame:
    p = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    p["k"] = (p.t / P.TICK).round().astype(int)
    return p.set_index("k")


def check_det(gen: Path, ids=None) -> pd.DataFrame:
    """Per world: ticks compared, max position / heading / speed difference, first tick past P5's divergence
    threshold (1 cm or 0.1 deg), and whether the camera-tick grids agree."""
    need_k = json.loads((root() / "need.json").read_text())
    g0, rows = p5v1.gen("ba"), []
    for rid in ids or sorted(need_k):
        a, o = attempt(gen, rid), P.attempt(g0, rid)
        if a is None or o is None:
            continue
        A, B = _pose(a), _pose(o)
        m = A[["x", "y", "yaw", "vx", "vy"]].join(B[["x", "y", "yaw", "vx", "vy"]], rsuffix="_o", how="inner")
        m = m[m.index <= need_k[rid]]
        d = np.hypot(m.x - m.x_o, m.y - m.y_o)
        dy = np.abs((m.yaw - m.yaw_o + 180) % 360 - 180)
        dv = np.hypot(m.vx - m.vx_o, m.vy - m.vy_o)
        bad = np.flatnonzero((d >= P.DIV_M) | (dy >= P.DIV_DEG))
        fa = pd.read_json(a / "frames.jsonl", lines=True)
        fo = pd.read_json(o / "frames.jsonl", lines=True)
        ko = set((fo.t / P.TICK).round().astype(int))
        ka = set(fa.k)
        lim = need_k[rid]
        rows.append({"rid": rid, "ticks": len(m), "need_k": lim, "max_pos_m": float(d.max()), "max_yaw_deg": float(dy.max()),
                     "max_dv_mps": float(dv.max()), "first_div_k": int(m.index[bad[0]]) if len(bad) else None,
                     "cam_grid_same": {k for k in ka if k <= lim} == {k for k in ko if k <= lim}})
    return pd.DataFrame(rows)


def bd_rows(adir: Path) -> pd.DataFrame:
    meta = json.loads((adir / "meta.json").read_text())
    cls = np.array(meta["bridgedrive"]["target_speed_classes"], float)
    r = pd.read_json(adir / "bridgedrive.jsonl", lines=True)
    if not len(r):
        return pd.DataFrame(columns=["k", "ts", "ts_scalar", "wp2", "p_stop"])
    dist = np.stack(r.pred_target_speed_distribution.map(lambda d: np.ravel(d)).to_numpy())
    wp = np.stack(r.pred_future_waypoints.map(np.asarray).to_numpy())
    return pd.DataFrame({"k": r.k, "ts": dist @ cls, "ts_scalar": r.pred_target_speed_scalar.map(lambda d: float(np.ravel(d)[0])),
                         "wp2": np.linalg.norm(wp[:, 7] - wp[:, 6], axis=-1) / 0.25, "p_stop": dist[:, 0],
                         "throttle": r.throttle, "brake": r.brake})


def blue_rows(f: Path) -> pd.DataFrame:
    d = json.loads(f.read_text())["frames"]
    if not d:
        return pd.DataFrame(columns=["k", "wp2", "gate", "gate_score", "language"])
    w = np.array([x["wps"] for x in d])
    return pd.DataFrame({"k": [x["k"] for x in d], "wp2": np.linalg.norm(w[:, 7] - w[:, 6], axis=-1) / 0.25,
                         "gate": [x["gate"] for x in d], "gate_score": [x["gate_score"] for x in d],
                         "language": [x["language"] for x in d]})


def creep_flags(adir: Path) -> pd.DataFrame:
    """Would the author's creep be active at tick k, had the author's agent seen this (expert-driven) speed history
    at 20 Hz: stuck counter (+1 per tick below 0.1 m/s, reset otherwise) above the author's threshold, or inside the
    creep duration after it (BridgeDrive 20 frames, BLUE 15)."""
    p = _pose(adir)
    still = (np.hypot(p.vx, p.vy) < STILL).to_numpy()
    out = {"k": p.index.to_numpy()}
    for name, thr, dur in (("bd", CREEP["bd"], 20), ("blue", CREEP["blue"], 15)):
        c, fm, flag = 0, 0, []
        for s in still:
            c = c + 1 if s else 0
            if c > thr:
                fm = dur
            flag.append(fm > 0)
            fm = max(fm - 1, 0)
        out["creep_" + name] = flag
    return pd.DataFrame(out)


def blue_plan(gen: Path):
    refs = json.loads((root() / "refs.json").read_text())
    plan = []
    for rid in json.loads((root() / "need.json").read_text()):
        a = attempt(gen, rid)
        if a is not None:
            plan.append({"rid": rid, "adir": str(a), "ks": refs[rid]})
    (root() / "blue_plan.json").write_text(json.dumps(plan))
    log.info("BLUE plan: %d worlds", len(plan))


def check_bd(a5: Path, a20: Path) -> dict:
    """Same world recorded with the 5 Hz shadow and with the author's run_step on every tick: outputs on the shared
    camera ticks."""
    x, y = bd_rows(a5).set_index("k"), bd_rows(a20).set_index("k")
    m = x.join(y, rsuffix="_ref", how="inner")
    return {"ticks": len(m), **{f"max_abs_{c}": float(np.abs(m[c] - m[c + "_ref"]).max()) for c in ("ts", "ts_scalar", "wp2")},
            "pose_max_m": float(np.hypot(*(_pose(a5)[["x", "y"]] - _pose(a20)[["x", "y"]]).dropna().to_numpy().T).max())}


def check_blue(fast: Path, ref: Path) -> dict:
    a, b = json.loads(fast.read_text())["frames"], json.loads(ref.read_text())["frames"]
    kb = {x["k"]: x for x in b}
    d = [(np.abs(np.array(x["wps"]) - np.array(kb[x["k"]]["wps"])).max(), x["gate"] == kb[x["k"]]["gate"],
          x["prompt"] == kb[x["k"]]["prompt"]) for x in a if x["k"] in kb]
    return {"frames": len(d), "max_abs_wps_m": float(max(z[0] for z in d)) if d else None,
            "gate_same": int(sum(z[1] for z in d)), "prompt_same": int(sum(z[2] for z in d))}


# ---------------------------------------------------------------- the exam

def readouts(gen: Path, blue_dir: Path) -> pd.DataFrame:
    """One row per (world, k): every examinee's scalar, creep flags, BLUE gate."""
    rows = []
    for rid in json.loads((root() / "need.json").read_text()):
        a = attempt(gen, rid)
        if a is None:
            continue
        r = creep_flags(a).merge(bd_rows(a).rename(columns={v: k for k, v in BD.items()}), on="k", how="left")
        f = blue_dir / (rid + ".json")
        if f.exists():
            b = blue_rows(f).rename(columns={"wp2": list(BLUE)[0]})
            r = r.merge(b, on="k", how="left")
        rows.append(r.assign(rid=rid))
    return pd.concat(rows, ignore_index=True)


def judge(rl, gen: Path, blue_dir: Path):
    from . import elicit_e4 as E4, elicit_i3 as I, p5_exam as E
    with I.p5_set("carla_p5v1_ba"):
        t, _, _, obs, null, pairs = E.load()
    R = readouts(gen, blue_dir).set_index(["rid", "k"])
    R.to_parquet(rl.dir / "readouts.parquet")
    o, n = E.deltas(obs, null, t, {})
    ours = list(BD) + list(BLUE)

    def look(fn, k, col):
        idx = pd.MultiIndex.from_arrays([fn.str.split("-").str[0], k])
        return R[col].reindex(idx).to_numpy(dtype=float)

    for col in ours:
        o[col] = look(o.fn_plus, o.k, col) - look(o.fn_minus, o.k, col)
        n[col] = look(n.fn_plus, n.k, col) - look(n.fn_null, n.k, col)
    for tag, (df, other) in {"o": (o, "fn_minus"), "n": (n, "fn_null")}.items():
        for c in ("creep_bd", "creep_blue"):
            df[c] = (np.nan_to_num(look(df.fn_plus, df.k, c)) > 0) | (np.nan_to_num(look(df[other], df.k, c)) > 0)
        gp, go = look(df.fn_plus, df.k, "gate"), look(df[other], df.k, "gate")
        df["blue_gate_open"] = np.where(np.isnan(gp) | np.isnan(go), np.nan, ((gp > 0) | (go > 0)).astype(float))
    ex = list(E.TFV6) + ours
    res = E.exam(o, n, pairs, ex)
    d = rl.dir
    res["flips"].to_csv(d / "flip_rates.csv", index=False)
    res["obs"].to_parquet(d / "obs_scored.parquet", index=False)
    n.to_parquet(d / "null_scored.parquet", index=False)
    (d / "summary.json").write_text(json.dumps({"tau_exp": res["tau_exp"], "pooled_families": res["pooled_families"],
                                                **{f"tau {k}": v for k, v in res["taus"].items()}}, indent=1))
    fl = res["flips"]
    rl.log.info("tau_exp %.3f, pooled %s\n%s", res["tau_exp"], res["pooled_families"],
                fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    # E4 (b): per pair, null per case (the [T1] judge, unchanged)
    ob = res["obs"].copy()
    ob["t_vis"] = pairs.set_index(["base_id", "seed"]).t_vis.reindex(pd.MultiIndex.from_frame(ob[["base_id", "seed"]])).to_numpy()
    s0 = pairs[pairs.seed == 0].set_index("base_id")
    nn = n.copy()
    nn["t_vis"] = s0.t_vis.fillna(s0.t_trig).reindex(nn.base_id).to_numpy()
    lat = E4.latency
    E4.latency = lambda e: 0.0 if e in ours else lat(e)
    pp = E4.score(ob, nn, res["taus"], res["pooled_families"], ex, nn)
    E4.latency = lat
    pp = pp[pp.window != "(a) gated"]
    pp.to_csv(d / "per_pair.csv", index=False)
    rl.log.info("per frame / per pair\n%s", pp.pivot_table(index="examinee", columns=["scope", "window"], values="flip",
                                                            sort=False).to_markdown(floatfmt=".3f"))
    # singled out: creep frames (reported apart, not in the main reading) and BLUE's gate
    ro = res["obs"]
    side = []
    for name in ours + list(E.TFV6):
        tau = res["taus"][name]
        creep = ro["creep_blue" if name.startswith("BLUE") else "creep_bd"] if not name.startswith("TFv6") else ro.creep_bd & False
        pooled = ro.family.isin(res["pooled_families"]) & ro[name].notna()
        for split, m in [("creep frames", creep), ("no creep", ~creep)] + (
                [("gate open (either world)", ro.blue_gate_open == 1), ("gate closed (both)", ro.blue_gate_open == 0)]
                if name.startswith("BLUE") else []):
            s = ro[pooled & m]
            r = s[s.reactive]
            ok = ((np.sign(r[name]) == np.sign(r.d_expert)) & E._moved(r[name], tau)).astype(float)
            fr, lo, hi = E.boot_ratio(ok.to_numpy(), np.ones(len(r)), r.base_id.to_numpy()) if len(r) else (np.nan,) * 3
            side.append({"examinee": name, "split": split, "frames": len(s), "reactive": len(r), "flip": fr, "lo": lo, "hi": hi,
                         "false_flip_nonreactive": float(E._moved(s[~s.reactive][name], tau).mean()) if (~s.reactive).any() else np.nan})
    side = pd.DataFrame(side)
    side.to_csv(d / "splits.csv", index=False)
    rl.log.info("creep / gate splits (pooled families)\n%s", side.to_markdown(index=False, floatfmt=".3f"))
    gate = [{"frames": scope, "n": int(m.sum()), "gate_open_share": float(np.nanmean(ro.blue_gate_open[m]))}
            for scope, m in (("all pair frames", ro.blue_gate_open.notna()), ("reactive", ro.reactive & ro.blue_gate_open.notna()),
                             ("non-reactive", ~ro.reactive & ro.blue_gate_open.notna()))]
    gate.append({"frames": "null frames", "n": int(n.blue_gate_open.notna().sum()), "gate_open_share": float(np.nanmean(n.blue_gate_open))})
    pd.DataFrame(gate).to_csv(d / "gate.csv", index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fl.to_csv(RESULTS / "p5_t3_flip_rates.csv", index=False)
    pp.to_csv(RESULTS / "p5_t3_per_pair.csv", index=False)
    side.to_csv(RESULTS / "p5_t3_splits.csv", index=False)
    pd.DataFrame(gate).to_csv(RESULTS / "p5_t3_gate.csv", index=False)
    return res


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("need", "check-det", "check-bd", "check-blue", "blue-plan", "judge"))
    ap.add_argument("--gen", default=str(data_dir() / "runs/top10_t3/gen"))
    ap.add_argument("--blue", default=str(data_dir() / "runs/top10_t3/blue"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--a", nargs=2, default=None, help="check-bd: attempt dirs (5 Hz, ref20); check-blue: json files (fast, ref)")
    a = ap.parse_args()
    gen = Path(a.gen)
    if a.cmd == "need":
        need()
    elif a.cmd == "check-det":
        df = check_det(gen, a.ids.split(",") if a.ids else None)
        print(df.to_markdown(index=False, floatfmt=".4g"))
        df.to_csv(gen / "check_det.csv", index=False)
    elif a.cmd == "check-bd":
        print(json.dumps(check_bd(Path(a.a[0]), Path(a.a[1])), indent=1))
    elif a.cmd == "check-blue":
        print(json.dumps(check_blue(Path(a.a[0]), Path(a.a[1])), indent=1))
    elif a.cmd == "blue-plan":
        blue_plan(gen)
    else:
        judge(RunLog("top10_t3", "judge"), gen, Path(a.blue))


if __name__ == "__main__":
    main()
