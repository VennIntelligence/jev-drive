#!/usr/bin/env python3
"""WAIT-CAP (plans/2026-10-16-wait-cap-prereg.md): does each stage of our lineage wait where the human waits, launch where the human goes,
and does any stage bypass a stopped car laterally? Open loop on NAVSIM *test-log* frames (all frames have cameras on the box, the navtest
filter is not applied), plus the closed-loop counterpart from stored rollouts. No training.

  select   (box, op-train python, CPU)  wc.py select            per frame of the 147 navsim test logs: human cells (wait / go / stop / follow),
                                        in-path vehicles from the frame's own annotations, the human's reference path; sampled rows
                                        -> $DATA_DIR/runs/wait_cap/select/rows.npz, tokens.txt, cells.md
  index    (box, navsim2 python)        wc.py index             navsim_zs_index.entry of every selected token (SceneLoader, frame_interval 1,
                                        4 history + 8 future frames) -> runs/navsim_zs/index/waitcap_test{.pkl,_future.npz}
  (then)   pp_prep.py --data idx_waitcap_test --frames warp --no-side     (GPU: Cinque's frozen encoder on the W protocol)
           rm_build.py nav --data idx_waitcap_test                         (AlpaSim venv, CPU: the route memory of every row)
  plans    (box, op-train python, GPU)  wc.py plans --tags T ..   per checkpoint the rear-axle plan (8 poses, 0.5 .. 4 s) of every row
                                        -> runs/wait_cap/plans/<tag>.npz; gate: P0 = shipped port; route-memory tags fed ego ++ route features
  report   (Mac or box, CPU)            wc.py report --rows R --plans DIR --out RESULTS   the registered classes, rates, paired CIs

Frames: ego / rear axle at t0, x forward, y left (OpenScene lidar2ego is the identity). Plan class (stationary cells): wait = plan arc to
4 s < 1.0 m; else bypass = max lateral distance of the plan poses from the human's reference path >= 1.0 m; else launch-straight.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
NAV = DATA / "datasets/navsim"
OUT = DATA / "runs/wait_cap"
SPLIT = "waitcap_test"
FRONT = 4.049                                   # rear axle -> front bumper (slp_probe / rf_lonol convention)
CELLS = ("W_L", "W_N", "G_L", "G_N", "S_L", "S_N", "F_L")
CAP, GAP_FR, SEED = 1200, 4, 20261016           # rows per cell, min frame spacing of rows of one cell in one log (2 s), sampling seed
REF_N, REF_DS = 81, 0.5                         # reference path: 0 .. 40 m every 0.5 m
VEH = ("vehicle", "bicycle")


# ---------------------------------------------------------------- select (CPU)
def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_ego(p, o, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(p, float) - o
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def resample(P, n=REF_N, ds=REF_DS):
    """Polyline P (m, 2) from the origin -> n points every ds m of arc; past its end extended straight along the last heading (or x)."""
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    keep = np.r_[True, np.diff(s) > 1e-3]
    P, s = P[keep], s[keep]
    q = np.arange(n) * ds
    if len(P) < 2 or s[-1] < 1.0:
        return np.stack([q, np.zeros(n)], -1), 0.0
    out = np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], -1)
    d = P[-1] - P[max(0, np.searchsorted(s, s[-1] - 2.0) - 1)]
    d = d / max(np.hypot(*d), 1e-6)
    far = q > s[-1]
    out[far] = P[-1] + (q[far] - s[-1])[:, None] * d
    return out, float(s[-1])


def lat_s(R, pts):
    """Unsigned lateral distance and arc of points (k, 2) to the reference polyline R (n, 2) (nearest segment, clamped)."""
    a, d = R[:-1], R[1:] - R[:-1]
    t = np.clip((((pts[:, None] - a[None]) * d[None]).sum(-1)) / np.maximum((d * d).sum(-1), 1e-12)[None], 0, 1)
    q = a[None] + t[..., None] * d[None]
    D = np.hypot(*(q - pts[:, None]).transpose(2, 0, 1))
    j = D.argmin(1)
    seg = np.r_[0.0, np.cumsum(np.hypot(*d.T))]
    return D[np.arange(len(pts)), j], seg[j] + t[np.arange(len(pts)), j] * np.hypot(*d[j].T)


def log_rows(path):
    """One test log -> per eligible frame dict of arrays (cells decided later)."""
    fr = pickle.load(open(path, "rb"))
    n = len(fr)
    ts = np.array([f["timestamp"] for f in fr], float) * 1e-6
    P = np.array([f["ego2global_translation"][:2] for f in fr], float)
    Y = np.array([yaw_of(f["ego2global_rotation"]) for f in fr])
    V = np.array([np.hypot(*np.asarray(f["ego_dynamic_state"][:2], float)) for f in fr])
    route = np.array([len(f["roadblock_ids"]) > 0 for f in fr])
    out = []
    for i in range(3, n - 8):
        if np.abs(np.diff(ts[i - 3:i + 9]) - 0.5).max() > 0.05 or not route[i]:
            continue
        fut = to_ego(P[i + 1:i + 9], P[i], Y[i])
        d2, d4 = float(np.hypot(*fut[3])), float(np.hypot(*fut[7]))
        a4 = float(np.r_[np.hypot(*fut[0]), np.hypot(*np.diff(fut, axis=0).T)].sum())
        vend = float(np.hypot(*(fut[7] - fut[6])) / 0.5)
        j = i
        while j + 1 < n and abs(ts[j + 1] - ts[j] - 0.5) < 0.05 and np.hypot(*(P[j + 1] - P[i])) < 45.0:
            j += 1
        R, rlen = resample(to_ego(P[i:j + 1], P[i], Y[i]))
        an = fr[i]["anns"]
        B, nm, vel = np.asarray(an["gt_boxes"], float).reshape(-1, 7), np.asarray(an["gt_names"]), np.asarray(an["gt_velocity_3d"], float).reshape(-1, 3)
        lead = dict(gap=np.inf, v=np.nan, x=np.nan, y=np.nan, yaw=np.nan, l=np.nan, w=np.nan, s=np.nan)
        lead_any = np.inf
        if len(B):
            isv = np.isin(nm, VEH)
            B, vel = B[isv], vel[isv]
            if len(B):
                dl, sa = lat_s(R, B[:, :2])
                gap = sa - B[:, 3] / 2 - FRONT
                inp = (dl < 1.5) & (gap > -1.0) & (B[:, 0] > 0)
                if inp.any():
                    k = np.flatnonzero(inp)[np.argmin(gap[inp])]
                    lead = dict(gap=float(gap[k]), v=float(np.hypot(*vel[k, :2])), x=float(B[k, 0]), y=float(B[k, 1]), yaw=float(B[k, 6]),
                                l=float(B[k, 3]), w=float(B[k, 4]), s=float(sa[k]))
                    lead_any = float(gap[k])
        out.append(dict(token=fr[i]["token"], log=fr[i]["log_name"], frame=i, v0=float(V[i]), d2=d2, d4=d4, a4=a4, vend=vend, fut=fut.astype(np.float32),
                        ref=R.astype(np.float32), ref_len=rlen, red_any=bool(any(r for _, r in fr[i]["traffic_lights"])),
                        **{f"lead_{k}": v for k, v in lead.items()}, lead_any=lead_any))
    return out


def cell_of(r):
    st, mv = r["v0"] < 0.3, 2.0 <= r["v0"] <= 12.0
    sl = r["lead_gap"] <= 15.0 and r["lead_v"] < 0.5
    free30 = r["lead_any"] > 30.0
    if st and r["d4"] < 1.0:
        return "W_L" if sl else ("W_N" if free30 else None)
    if st and r["d2"] > 0.5 and r["d4"] > 2.0:
        return "G_L" if (r["lead_gap"] <= 15.0 and r["lead_v"] < 3.0) else ("G_N" if free30 else None)
    if mv and r["vend"] < 0.5:
        if 2.0 <= r["lead_gap"] <= 40.0 and r["lead_v"] < 0.5 and r["a4"] <= r["lead_gap"] + 1.0:       # stops behind the stopped car
            return "S_L"
        return "S_N" if r["lead_any"] > r["a4"] + 15.0 else None
    if mv and r["vend"] > max(2.0, 0.5 * r["v0"]) and r["lead_gap"] <= 30.0 and r["lead_v"] > 2.0:
        return "F_L"
    return None


def _log_job(p):
    return log_rows(p)


def cmd_select(a):
    from jevdrive import par
    from jevdrive.run import Run
    logs = sorted((NAV / "navsim_logs/test").glob("*.pkl"))
    with Run("wait_cap", "select", config=vars(a), seed=SEED) as run:
        res = par.pmap(_log_job, logs, run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        rows = [r for v in res.values for r in v]
        for r in rows:
            r["cell"] = cell_of(r)
        run.info(f"{len(rows)} eligible frames, {len(logs)} logs")
        rng = np.random.default_rng(SEED)
        keep = []
        census = {}
        for c in CELLS:
            cand = sorted((r for r in rows if r["cell"] == c), key=lambda r: (r["log"], r["frame"]))
            thin, last = [], {}
            for r in cand:                                          # >= GAP_FR frames between rows of one cell in one log
                if r["frame"] - last.get(r["log"], -99) >= GAP_FR:
                    thin.append(r)
                    last[r["log"]] = r["frame"]
            pick = [thin[k] for k in sorted(rng.permutation(len(thin))[:CAP])]
            census[c] = dict(frames=len(cand), thinned=len(thin), rows=len(pick), logs=len({r["log"] for r in pick}))
            keep += pick
        out = OUT / "select"
        out.mkdir(parents=True, exist_ok=True)
        cols = [k for k in keep[0] if k not in ("fut", "ref")]
        np.savez(out / "rows.npz", fut=np.stack([r["fut"] for r in keep]), ref=np.stack([r["ref"] for r in keep]),
                 **{k: np.array([r[k] for r in keep]) for k in cols})
        (out / "tokens.txt").write_text("\n".join(r["token"] for r in keep) + "\n")
        allc = {c: sum(r["cell"] == c for r in rows) for c in CELLS}
        md = ["| cell | frames | thinned | rows | logs |", "|---|--:|--:|--:|--:|"] + [f"| {c} | {v['frames']} | {v['thinned']} | {v['rows']} | {v['logs']} |" for c, v in census.items()]
        (out / "cells.md").write_text(f"eligible frames {len(rows)} in {len(logs)} test logs; stationary (v0 < 0.3) {sum(r['v0'] < 0.3 for r in rows)}\n\n" + "\n".join(md) + "\n")
        run.summary |= dict(eligible=len(rows), census=census, all=allc)
        print((out / "cells.md").read_text())


# ---------------------------------------------------------------- index (navsim2 venv)
def cmd_index(a):
    import importlib.util
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(NAV))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    spec = importlib.util.spec_from_file_location("nzi", REPO / "experiments/zeroshot_openloop/archive/navsim_zs_index.py")
    nzi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nzi)
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from jevdrive.navsim_zs import root
    z = np.load(OUT / "select/rows.npz")
    toks, logs = z["token"].tolist(), sorted(set(z["log"].tolist()))
    sf = SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1, has_route=True, log_names=logs, tokens=toks)
    real = NAV / "sensor_blobs/test"
    L = SceneLoader(NAV / "navsim_logs/test", real, sf)
    got = set(L.tokens_stage_one)
    miss = [t for t in toks if t not in got]
    out, fut = [], []
    for t in toks:
        if t in miss:
            continue
        frames = L.scene_frames_dicts[t]
        meta = {"log_name": frames[3]["log_name"], "synthetic": False, "map": frames[3]["map_location"], "timestamp": int(frames[3]["timestamp"])}
        out.append(nzi.entry(t, "one", L.get_agent_input_from_token(t), [f["cams"] for f in frames[:4]], real, meta))
        fut.append(L.get_scene_from_token(t).get_future_trajectory(8).poses)
    nmiss = sum(not Path(c["CAM_F0"]["path"]).exists() for e in out for c in e["cams"])
    with open(root("index") / f"{SPLIT}.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)
    np.savez(root("index") / f"{SPLIT}_future.npz", tokens=np.array([e["token"] for e in out]), poses=np.stack(fut).astype(np.float32))
    print(f"{SPLIT}: {len(out)} tokens, not loaded {len(miss)}, missing CAM_F0 files {nmiss}")
    assert nmiss == 0


def cmd_define(a):
    """Register the row set as navsim/waitcap_test (jevdrive.data.splits; pp_prep's run.use_split needs it)."""
    from jevdrive.data import splits
    z = np.load(OUT / "select/rows.npz")
    splits.define("navsim", SPLIT, sorted(z["token"].tolist()), unit="token", origin="wait_cap select (experiments/wait_cap/scripts/wc.py): sampled cells "
                  "of NAVSIM test-log frames, plans/2026-10-16-wait-cap-prereg.md; rows $DATA_DIR/runs/wait_cap/select/rows.npz", status="frozen",
                  used_by=["experiments/wait_cap"], notes="open-loop evaluation rows only; test logs, never trained on")


# ---------------------------------------------------------------- plans (GPU)
def cmd_plans(a):
    import torch
    sys.path[:0] = [str(REPO / p) for p in ("lib", "scripts", "experiments/op_adapt_r2/lib", "experiments/op_parity/scripts", "experiments/route_mem/lib",
                                             "experiments/route_mem/scripts")]
    import pp_train as T
    import rmem as RM
    from jevdrive.run import Run
    from jevdrive.data import splits
    data = f"idx_{SPLIT}"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    out = OUT / "plans"
    out.mkdir(parents=True, exist_ok=True)
    with Run("wait_cap", "plans", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{SPLIT}"))
        S = T.Store([data], dev, need_side=False, frames="warp")
        z = np.load(DATA / "runs/route_mem/feat" / f"{data}.npz")
        assert np.array_equal(z["names"], S.tab["names"]), "route features rows != cache rows"
        F = z["f"].astype(np.float32)
        F[~z["ok"]] = 0
        F = torch.as_tensor(F, device=dev)
        W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        for tag in a.tags:
            f = out / f"{tag}.npz"
            if f.exists() and not a.force:
                run.info(f"{tag}: exists")
                continue
            if tag == "P0":
                model = T.load_pmodel("P0", dev)
                br = False
            else:
                ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
                model = T.PModel(ck["model"]["arm"]).to(dev).eval()
                br = RM.has_branch(tag)
                if br:
                    sd = json.loads((T.proot("runs", tag) / RM.SIDECAR).read_text())
                    assert not sd.get("reflex") and not sd["head"].get("order") and not sd["head"].get("latfeat"), f"{tag}: unsupported branch variant"
                    RM.attach(model, **{k: v for k, v in sd["head"].items() if k not in ("order", "latfeat")})
                    model.to(dev)
                model.load_state(ck["model"])
                model.eval()
            sl = model.net.slices
            pi = torch.arange(sl["plan"].start, sl["plan"].start + 495, device=dev)
            xs, ys, ps = [], [], []
            with torch.no_grad():
                for i in range(0, S.n, 256):
                    r = torch.arange(i, min(i + 256, S.n), device=dev)
                    e = torch.cat([S.ego[r].float(), F[r]], 1) if br else S.ego[r]
                    p = model(S.front[r], e, S.tc[r], None, None, nv=None).float()[:, pi].view(-1, 33, 15)
                    x, y, psi = T.rear(p, S.cam_x[r], W)
                    xs.append(x.cpu()), ys.append(y.cpu()), ps.append(psi.cpu())
            x, y, psi = (torch.cat(v).numpy() for v in (xs, ys, ps))
            fut = S.tab["fut"]
            ade = float(np.nanmean(np.hypot(x - fut[..., 0], y - fut[..., 1])))
            np.savez(f, names=S.tab["names"], plan=np.stack([x, y, psi], -1).astype(np.float32), branch=br)
            run.info(f"{tag}: rows {S.n}, branch {br}, ADE vs log {ade:.3f}")
            run.summary[tag] = dict(ade=ade, branch=br)
            del model
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- report (CPU)
def classify(plan, ref, lead, stationary):
    """plan (n, 8, 3) rear-axle poses; ref (n, REF_N, 2) the human's reference path; lead dict of (n,) arrays.
    -> dict of per-row arrays: arc4, lat (max lateral distance from ref), cls (0 wait, 1 launch-straight, 2 bypass), into (footprint reaches the lead)."""
    P = plan[..., :2]
    arc = np.hypot(P[:, 0, 0], P[:, 0, 1]) + np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)).sum(1)
    lat = np.zeros(len(P))
    s_end = np.zeros(len(P))
    for k in range(len(P)):
        d, s = lat_s(ref[k], P[k])
        lat[k], s_end[k] = d.max(), s[-1]
    wait = arc < 1.0
    byp = ~wait & (lat >= 1.0)
    cls = np.where(wait, 0, np.where(byp, 2, 1))
    # footprint front reaches the lead's rear while laterally overlapping it (lead box half width + ego half width 1.0 m)
    yaw = plan[..., 2]
    fx, fy = P[..., 0] + FRONT * np.cos(yaw), P[..., 1] + FRONT * np.sin(yaw)
    into = np.zeros(len(P), bool)
    ok = np.isfinite(lead["x"])
    for k in np.flatnonzero(ok):
        c, s_ = np.cos(lead["yaw"][k]), np.sin(lead["yaw"][k])
        dx, dy = fx[k] - lead["x"][k], fy[k] - lead["y"][k]
        u, v = c * dx + s_ * dy, -s_ * dx + c * dy
        into[k] = bool(((np.abs(u) <= lead["l"][k] / 2 + 0.2) & (np.abs(v) <= lead["w"][k] / 2 + 1.0)).any())
    return dict(arc4=arc, lat=lat, cls=cls, into=into, s_end=s_end)


def cmd_report(a):
    from jevdrive import stats
    z = dict(np.load(a.rows, allow_pickle=False))
    n = len(z["token"])
    lead = {k: z[f"lead_{k}"] for k in ("x", "y", "yaw", "l", "w")}
    tags = [t for t in a.tags if (Path(a.plans) / f"{t}.npz").exists()]
    R = {}
    for t in tags:
        p = np.load(Path(a.plans) / f"{t}.npz")
        pos = {k: i for i, k in enumerate(p["names"].tolist())}
        idx = np.array([pos.get(k, -1) for k in z["token"].tolist()])
        assert (idx >= 0).all(), f"{t}: rows missing"
        R[t] = classify(p["plan"][idx], z["ref"], lead, None)
    hum_arc = z["a4"]
    lines, rows_csv = [], []
    cells = [c for c in CELLS if (z["cell"] == c).any()]
    for c in cells:
        m = z["cell"] == c
        g = z["log"][m]
        lines += [f"\n### {c} (n {m.sum()}, logs {len(set(g))})", "", "| arm | wait | launch-straight | bypass (lat >= 1 m) | into lead | arc4 med (m) | lat p90 (m) |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for t in tags:
            r = R[t]
            cl = r["cls"][m]
            fm = lambda x: stats.fmt(stats.bootstrap(x.astype(float), groups=g))  # noqa: E731
            lines.append(f"| {t} | {fm(cl == 0)} | {fm(cl == 1)} | {fm(cl == 2)} | {fm(r['into'][m]) if np.isfinite(lead['x'][m]).any() else '-'} | "
                         f"{np.median(r['arc4'][m]):.2f} | {np.percentile(r['lat'][m], 90):.2f} |")
            rows_csv.append(dict(cell=c, arm=t, n=int(m.sum()), wait=float((cl == 0).mean()), straight=float((cl == 1).mean()), bypass=float((cl == 2).mean()),
                                 into=float(r["into"][m].mean()), arc4_med=float(np.median(r["arc4"][m])), lat_p90=float(np.percentile(r["lat"][m], 90)),
                                 over_med=float(np.median(r["arc4"][m] - hum_arc[m]))))
        lines.append(f"\nhuman: arc4 median {np.median(hum_arc[m]):.2f} m")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    # paired stage-to-stage differences on the registered quantities
    pl = ["\n## paired differences (later - earlier), log-clustered bootstrap\n", "| cell | quantity | pair | diff |", "|---|---|---|--:|"]
    for c, q, f_ in (("W_L", "wait", lambda r: r["cls"] == 0), ("W_N", "wait", lambda r: r["cls"] == 0), ("W_L", "bypass", lambda r: r["cls"] == 2),
                     ("S_L", "bypass", lambda r: r["cls"] == 2), ("G_L", "wait", lambda r: r["cls"] == 0), ("G_N", "wait", lambda r: r["cls"] == 0),
                     ("W_L", "into", lambda r: r["into"])):
        m = z["cell"] == c
        if not m.any():
            continue
        for t0, t1 in [(tags[i], tags[i + 1]) for i in range(len(tags) - 1)] + [("P0", t) for t in tags if t != "P0" and "P0" in tags]:
            if t0 not in R or t1 not in R:
                continue
            r = stats.paired(f_(R[t1])[m].astype(float), f_(R[t0])[m].astype(float), groups=z["log"][m])
            pl.append(f"| {c} | {q} | {t1} - {t0} | {stats.fmt(r)} |")
    (out / "wait_cap.md").write_text("# WAIT-CAP open loop\n" + "\n".join(lines) + "\n" + "\n".join(pl) + "\n")
    print((out / "wait_cap.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--workers", type=int, default=16)
    sp.add_parser("index")
    sp.add_parser("define")
    p = sp.add_parser("plans")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--rows", required=True)
    p.add_argument("--plans", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "index": cmd_index, "define": cmd_define, "plans": cmd_plans, "report": cmd_report}[a.cmd](a)
