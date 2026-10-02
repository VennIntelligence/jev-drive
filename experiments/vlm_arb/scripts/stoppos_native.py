"""What openpilot's native outputs say about stop lines and junctions, and how faithful the 2 Hz replay is (plan 2026-10-03-stoppos-probe.md,
Heads and readouts 1 and Features).  Project venv, CPU.

  native_logs(out_dir)   logged heads of the `drive` units (20 Hz plans.jsonl) against the true distance to the stop line / junction
  fidelity(out_dir)      replayed heads of the CL frames (stored `native` array) against the 20 Hz values logged at the same time
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoppos_data as D  # noqa: E402
from jevdrive import stats  # noqa: E402

ARMS = D.DATA / "runs/vlm_arb/arms"
REAR = 1.3886 + 2.4508
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
DRIVE_UNITS = ("eval-drive-s0", "eval-drive-s0-rep1", "eval-drive-s0-rep2", "eval-drive-s0-rep3-partial", "v2-drive-s0-tgt", "v2-drive-s1-dev",
               "v2-drive-s1-tgt", "shadow-drive-s0")
# slices of the 18452-d Cinque output (ONNX metadata `output_slices`)
SL = dict(lane_lines=(0, 528), lane_lines_prob=(528, 536), road_edges=(536, 800), meta=(800, 855), desire_pred=(855, 887), pose=(887, 899),
          wide_from_device_euler=(899, 905), road_transform=(905, 917), plan=(917, 1907), lead=(1907, 2051), lead_prob=(2051, 2054),
          desire_state=(2054, 2062), action=(2062, 2066), hidden_state=(2066, 18450))


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -11, np.inf)))


def decode_native(nat: np.ndarray) -> dict:
    """Heads of a (n, 2066) native array: plan speed at 1, 2, 3, 5 s, brake-press probabilities, lead probability, lane probabilities."""
    plan = nat[:, SL["plan"][0]:SL["plan"][0] + 495].reshape(-1, 33, 15)
    vx = plan[:, :, 3]
    v = {s: np.array([np.interp(s, T_IDXS, r) for r in vx]) for s in (0, 1, 2, 3, 5)}
    meta = sig(nat[:, SL["meta"][0]:SL["meta"][1]])
    lead_p = sig(nat[:, SL["lead_prob"][0]:SL["lead_prob"][1]])
    lane_p = sig(nat[:, SL["lane_lines_prob"][0]:SL["lane_lines_prob"][1]])[:, 1::2]
    return dict(v0=v[0], v1=v[1], v2=v[2], v3=v[3], v5=v[5], brk0=meta[:, 32], brk2=meta[:, 36], hb3=meta[:, 4], lead_p=lead_p[:, 0],
                lane1=lane_p[:, 1], lane2=lane_p[:, 2])


def jl(p):
    out = []
    try:
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _attempt(d):
    d = Path(d)
    P = jl(d / "plans.jsonl")
    if not P:
        return None
    V = jl(d / "vlm_decisions.jsonl")
    g = {round(r["t"], 3): r.get("gt", {}) for r in V if "gt" in r and "speed" in r}
    rows = []
    for p in P:
        if p.get("warm"):
            continue
        c = p.get("ctx") or {}
        gt = g.get(round(p["t"], 3), {})
        nj = gt.get("next_junc_dist")
        rows.append(dict(t=p["t"], v=p["v"], v3=p["vplan"][3], v5=p["vplan"][4], dv3=p["vplan"][3] - p["v"], dv5=p["vplan"][4] - p["v"],
                         brk0=p["brk"][0], brk2=p["brk"][1], hb3=p["hb3"][0], lead_p=p["lp"][0], lane1=p["lane"][1], lane2=p["lane"][2],
                         d_stop=c.get("tl_dist", np.nan), tl=c.get("tl", np.nan),
                         d_junc=(nj - REAR) if nj is not None and nj < 999 else np.nan, ego_s=(p.get("pc") or {}).get("ego_s", np.nan)))
    df = pd.DataFrame(rows)
    df["unit"], df["route"], df["att"] = d.parts[-4], d.parts[-2], d.parts[-1]
    return df


def native_logs(out_dir: Path) -> dict:
    dirs = [d for u in DRIVE_UNITS for d in sorted(glob.glob(f"{ARMS}/{u}/attempts/*/*"))]
    frames = [x for x in map(_attempt, dirs) if x is not None]
    df = pd.concat(frames, ignore_index=True)
    df["id"] = "cl-" + df.route
    df["key"] = df.unit + "/" + df.route + "/" + df.att
    df["ratio3"] = df.v3 / df.v.clip(lower=0.5)
    # approach group: nearest light ahead's state, or "no light" for junction approaches without a light on the route
    light_routes = set(df[df.d_stop.notna()].id)
    df["group"] = np.where(df.d_stop.notna() & df.tl.isin([1, 2]), "red/yellow", np.where(df.d_stop.notna() & (df.tl == 0), "green",
                                                                                           np.where(df.d_junc.notna() & ~df.id.isin(light_routes), "no light", "other")))
    rolling = df[(df.v >= 3.0)]
    metrics = ["ratio3", "dv3", "dv5", "brk0", "brk2", "hb3", "lead_p", "lane1", "lane2"]
    rows, rho_rows = [], []
    edges = [-2, 0, 5, 10, 15, 20, 30, 40]
    for grp, dcol in (("red/yellow", "d_stop"), ("green", "d_stop"), ("no light", "d_junc")):
        g = rolling[rolling.group == grp]
        g = g[g[dcol].between(-2, 40)]
        g = g.assign(bin=pd.cut(g[dcol], edges, right=False))
        for b, gb in g.groupby("bin", observed=True):
            r = dict(group=grp, dist=dcol, bin=str(b), frames=len(gb), routes=gb.id.nunique(), attempts=gb.key.nunique())
            for m in metrics:
                per = gb.groupby(["id", "key"])[m].median().groupby(level=0).median()
                r[m] = float(per.median())
            rows.append(r)
        # within-route Spearman between distance and each metric
        for m in metrics:
            rhos = {}
            for rid, gr in g.groupby("id"):
                if len(gr) >= 20 and gr[dcol].nunique() > 5 and gr[m].nunique() > 3:
                    rhos[rid] = gr[dcol].corr(gr[m], method="spearman")
            if rhos:
                b = stats.bootstrap(np.array(list(rhos.values())))
                rho_rows.append(dict(group=grp, dist=dcol, metric=m, routes=len(rhos), rho=b["mean"], lo=b["lo"], hi=b["hi"],
                                     routes_pos=int(sum(v > 0 for v in rhos.values())), routes_neg=int(sum(v < 0 for v in rhos.values()))))
    tab, rho = pd.DataFrame(rows), pd.DataFrame(rho_rows)
    tab.to_csv(out_dir / "native_by_distance.csv", index=False)
    rho.to_csv(out_dir / "native_spearman.csv", index=False)
    # discrimination: native scalar separating red-light frames within 10 m of the line from 20-40 m (v >= 3 m/s)
    disc = []
    g = rolling[rolling.group == "red/yellow"]
    near, far = g[g.d_stop.between(0, 10)], g[g.d_stop.between(20, 40)]
    for m in metrics:
        d = pd.concat([near.assign(y=1), far.assign(y=0)])
        d = d[d[m].notna()]
        if d.y.nunique() < 2:
            continue
        w = D.group_weights(d)
        disc.append(dict(metric=m, near_frames=int((d.y == 1).sum()), far_frames=int((d.y == 0).sum()), auc_high_near=D.auc(d.y, d[m], w),
                         routes_near=d[d.y == 1].id.nunique(), routes_far=d[d.y == 0].id.nunique()))
    disc = pd.DataFrame(disc)
    disc.to_csv(out_dir / "native_discrimination.csv", index=False)
    summ = dict(attempts=int(df.key.nunique()), ticks=len(df), rolling_ticks=len(rolling),
                red_frames_0_40=int(((rolling.group == "red/yellow") & rolling.d_stop.between(0, 40)).sum()),
                green_frames_0_40=int(((rolling.group == "green") & rolling.d_stop.between(0, 40)).sum()),
                nolight_frames_0_40=int(((rolling.group == "no light") & rolling.d_junc.between(0, 40)).sum()))
    df.to_parquet(out_dir / "native_logs.parquet")
    json.dump(summ, open(out_dir / "native_summary.json", "w"), indent=1)
    return dict(tab=tab, rho=rho, disc=disc, summary=summ)


def fidelity(out_dir: Path, frames: pd.DataFrame) -> pd.DataFrame:
    """Replayed (2 Hz frames held for 10 steps) against logged (20 Hz) heads at the same query time, CL frames."""
    cl = frames[(frames.src == "cl")]
    rows = []
    for key, g in cl.groupby("key", sort=False):
        p = D.FEATS / (key.replace("/", "__") + ".npz")
        if not p.exists():
            continue
        _, unit, route, att = key.split("/")
        P = pd.DataFrame(jl(ARMS / unit / "attempts" / route / att / "plans.jsonl"))
        if P.empty:
            continue
        with np.load(p) as z:
            dec = decode_native(z["native"])
        pt = P.t.to_numpy()
        i = np.abs(pt[None, :] - g.t.to_numpy()[:, None]).argmin(1)
        ok = np.abs(pt[i] - g.t.to_numpy()) < 0.03
        vp = np.array(P.vplan.tolist())[i]
        lg = dict(v0=vp[:, 0], v1=vp[:, 1], v2=vp[:, 2], v3=vp[:, 3], v5=vp[:, 4], brk0=np.array(P.brk.tolist())[i][:, 0],
                  brk2=np.array(P.brk.tolist())[i][:, 1], hb3=np.array(P.hb3.tolist())[i][:, 0], lead_p=np.array(P.lp.tolist())[i][:, 0],
                  lane1=np.array(P.lane.tolist())[i][:, 1], lane2=np.array(P.lane.tolist())[i][:, 2])
        for m in lg:
            r = pd.DataFrame(dict(key=key, id=g.id.iloc[0], t=g.t.to_numpy(), metric=m, logged=lg[m], replay=dec[m]))
            rows.append(r[ok])
    d = pd.concat(rows, ignore_index=True)
    out = []
    for m, g in d.groupby("metric"):
        out.append(dict(metric=m, n=len(g), corr=float(np.corrcoef(g.logged, g.replay)[0, 1]), mean_abs_diff=float((g.logged - g.replay).abs().mean()),
                        logged_sd=float(g.logged.std()), bias=float((g.replay - g.logged).mean())))
    t = pd.DataFrame(out)
    t.to_csv(out_dir / "fidelity.csv", index=False)
    return t
