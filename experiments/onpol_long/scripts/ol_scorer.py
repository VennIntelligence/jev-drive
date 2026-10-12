#!/usr/bin/env python3
"""ONPOL-LONG Amendment 12: feasibility of a candidate-scoring line on the 43 zero events of YS-L18-s1 (held 452, 18 decisions). No new rollouts.

Box; X = $DATA_DIR/runs/onpol_long/scorer/<tag> (pseudo run dir X/run: onpol_sim.pkl + the replay's driver-logs):
  prep    (AlpaSim venv)  events + E1 onset (anchor_offset per_decision.parquet, |o10| >= 4.21 m) -> X/events.json; c1_extract logs / msgs
  replay  (op-train, GPU) the driver class on the stored driver messages (c1_replay's loop), SH30_DUMP on: per decision the driver's tokens, valid,
          ego, served plan -> X/tok/<scene>.npz; served plan against the original drive.jsonl -> X/replay.json
  geo     (navsim2)       on_rows.geo_one unedited on X/run: simulator actor boxes, road-and-lane SDF, logged path per decision -> X/geo.pkl
  score   (op-train, GPU) candidate family, privileged labels (body1 sweep.py), d223 two-seed head scores -> X/cand.npz, X/score.json, X/score.md
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
TAG = "YS-L18-s1"
X = DATA / "runs/onpol_long/scorer" / TAG
RUN = DATA / "runs/onpol_long/read" / TAG / "run"
E1 = DATA / "runs/anchor_offset/e1/20261012-124821/per_decision.parquet"
EVENTS = R / "experiments/zeros_mech/results/events_YS-L18-s1.csv"
THR10, WIN, CORR, T_JUDGE = 4.21, 4, 4.0, 1.0
VBINS = (1.0, 4.0, 8.0, 12.0)
LAT = (0.5, 1.0, 1.5, 2.0, 3.0)
YAW = (2.5, 5.0, 10.0)
VS = (0.0, 0.5, 0.7, 0.85, 1.15, 1.3)


def cmd_prep(a):
    import csv
    import polars as pl
    e1 = pl.read_parquet(E1).filter(pl.col("label") == TAG)
    ev = []
    for r in csv.DictReader(open(EVENTS)):
        k_ev = int(float(r["k_ev"]))
        o = e1.filter((pl.col("scene") == r["scene"]) & (pl.col("k") <= k_ev)).sort("k")
        al = [int(k) for k, v in zip(o["k"].to_list(), o["o10"].to_list()) if v is not None and np.isfinite(v) and abs(v) >= THR10]
        k_on = al[0] if al else k_ev
        ev.append(dict(scene=r["scene"], log=r["log"], type=r["type"], k_ev=k_ev, k_on=k_on, e1_alarm=bool(al), cls=r.get("cls", ""), fits=r.get("fits", ""),
                       ks=list(range(max(0, k_on - WIN), k_ev + 1))))
    (X / "run").mkdir(parents=True, exist_ok=True)
    (X / "events.json").write_text(json.dumps(ev, indent=1) + "\n")
    (X / "scenes.txt").write_text("\n".join(e["scene"] for e in ev) + "\n")
    print(len(ev), "events;", sum(e["e1_alarm"] for e in ev), "with an E1 alarm", flush=True)
    py, ex = sys.executable, str(R / "experiments/alpasim/scripts/c1_extract.py")
    subprocess.run([py, ex, "msgs", "--run", str(RUN), "--scenes", str(X / "scenes.txt"), "--out", str(X / "msgs")], check=True)
    subprocess.run([py, ex, "logs", "--run", str(RUN), "--out", str(X / "run/onpol_sim.pkl"), "--jobs", "16"], check=True)


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a):
    sys.path.insert(0, str(R / "experiments/alpasim/lib"))
    os.environ.setdefault("SH30_TAG", TAG)
    import sh30_driver as D
    ev = json.loads((X / "events.json").read_text())
    orig = {}
    for line in open(RUN / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "drive":
            orig[(r["scene"], r["k"])] = np.array(r["poses"])
    dl = X / "run/driver-logs"
    assert not (dl / "drive.jsonl").exists(), f"{dl} already replayed"
    core = D.C.Core(TAG, "cuda", os.environ.get("SH30_COLD", "backwarp"), float(os.environ.get("SH30_MOTION", "1")), os.environ.get("SH30_SYNTH", "gpu"))
    z = np.zeros(D.C.FRAME, np.uint8)
    decode = D.tune(core, lambda: [core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5]) for m in (1, 4, 4)])
    drv = D.Driver(core, dl, 1000, False)                                # the live driver's settings (sh30_driver main): gpu synthesis, compiled passes
    drv.decode = decode
    pb, last, plan0 = D.egodriver_pb2, {}, core.plan

    def plan(*x, **k):
        last["o"] = plan0(*x, **k)
        return last["o"]
    core.plan = plan
    (X / "tok").mkdir(exist_ok=True)
    ctx, chk = Ctx(), {}
    for e in ev:
        f = X / "msgs" / f"{e['scene']}.pkl"
        if not f.exists():
            chk[e["scene"]] = "no msgs"
            continue
        k, uuid, T = 0, None, dict(k=[], tokens=[], valid=[], ego=[], poses=[])
        for kind, raw in pickle.load(open(f, "rb")):
            if kind == "driver_session_request":
                req = pb.DriveSessionRequest.FromString(raw)
                uuid = req.session_uuid
                drv.start_session(req, ctx)
            elif kind == "driver_camera_image":
                drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
            elif kind == "driver_ego_trajectory":
                drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
            elif kind == "route_request":
                drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
            elif kind == "driver_request":
                drv.drive(pb.DriveRequest.FromString(raw), ctx)
                o = last["o"]
                V = np.zeros((8, 32, 512), np.float16)
                V[np.flatnonzero(o["valid"])] = o["tokens"].float().cpu().numpy().astype(np.float16)
                for q, v in (("k", k), ("tokens", V), ("valid", np.asarray(o["valid"], bool)), ("ego", o["ego"]), ("poses", o["poses"])):
                    T[q].append(v)
                k += 1
        drv.sessions.pop(uuid, None)
        np.savez(X / "tok" / f"{e['scene']}.npz", **{q: np.stack(v) for q, v in T.items()})
        d = [float(np.abs(p - orig[(e["scene"], kk)]).max()) for kk, p in zip(T["k"], T["poses"]) if (e["scene"], kk) in orig]
        chk[e["scene"]] = dict(n=len(T["k"]), n_orig=sum(1 for s, _ in orig if s == e["scene"]), max_diff=max(d) if d else None,
                               bad_k=[kk for kk, x in zip(T["k"], d) if x > 1e-3])
        print(e["scene"], json.dumps(chk[e["scene"]]), flush=True)
    (X / "replay.json").write_text(json.dumps(chk, indent=1) + "\n")


def cmd_geo(a):
    sys.path[:0] = [str(R / "experiments/onpol_nuplan/scripts")]
    import on_rows as OR
    from concurrent.futures import ProcessPoolExecutor
    ev = json.loads((X / "events.json").read_text())
    keep = ("k", "excl", "box", "bval", "cls", "sdf_road", "sdf_nav", "gpath", "poses", "ego", "e0", "s0", "t_us", "err")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(OR.geo_one, [(TAG, str(X / "run"), e["scene"]) for e in ev]))
    out = {e["scene"]: {r["k"]: {q: r[q] for q in keep if q in r} for r in g["rows"]} for e, g in zip(ev, res)}
    pickle.dump(out, open(X / "geo.pkl", "wb"), protocol=4)
    print({s: sum("box" in v for v in d.values()) for s, d in out.items()}, flush=True)


# ---------------------------------------------------------------- candidates
def arcs(P):
    xy = np.concatenate([np.zeros_like(P[..., :1, :2]), P[..., :2]], -2)
    return np.cumsum(np.hypot(*np.moveaxis(np.diff(xy, axis=-2), -1, 0)), -1)


def along(P, s_new):
    """Poses at arc lengths s_new (8,) along the polyline origin -> P (8, 3), extrapolated straight past the end."""
    xy = np.concatenate([np.zeros((1, 2)), P[:, :2]])
    yaw = np.r_[0.0, P[:, 2]]
    s = np.r_[0.0, arcs(P)]
    s = np.maximum.accumulate(s)
    out = []
    for q in s_new:
        if q <= s[-1] or s[-1] < 1e-3:
            j = int(np.clip(np.searchsorted(s, q) - 1, 0, len(s) - 2))
            w = 0.0 if s[j + 1] - s[j] < 1e-6 else np.clip((q - s[j]) / (s[j + 1] - s[j]), 0, 1)
            out.append([*(xy[j] * (1 - w) + xy[j + 1] * w), yaw[j] * (1 - w) + yaw[j + 1] * w])
        else:
            out.append([*(xy[-1] + (q - s[-1]) * np.array([np.cos(yaw[-1]), np.sin(yaw[-1])])), yaw[-1]])
    return np.array(out)


def ramp(P, A):
    s = arcs(P)
    s4 = max(s[-1], 1e-3)
    off = A * s / s4
    return np.c_[P[:, 0] - off * np.sin(P[:, 2]), P[:, 1] + off * np.cos(P[:, 2]), P[:, 2] + np.arctan2(A, s4)]


def rotz(P, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1], P[:, 2] + np.radians(deg)]


def family(P):
    """p0 + local variants -> (n, 8, 3), names."""
    C, N = [P], ["p0"]
    for A in LAT:
        for sg in (1, -1):
            C.append(ramp(P, sg * A)), N.append(f"lat{sg * A:+g}")
    for d in YAW:
        for sg in (1, -1):
            C.append(rotz(P, sg * d)), N.append(f"yaw{sg * d:+g}")
    s = arcs(P)
    for v in VS:
        C.append(along(P, v * s)), N.append(f"v{v:g}")
    for A in (1.0, 2.0):
        for sg in (1, -1):
            C.append(ramp(along(P, 0.7 * s), sg * A)), N.append(f"lat{sg * A:+g}v0.7")
    return np.stack(C), N


def vocab(dev, k=256, n_max=200_000, it=20):
    import torch
    F, V = [], []
    for i in range(4):
        z = np.load(DATA / f"runs/op_parity/cache/navtrain_full.s{i}of12/tab.npz")
        F.append(z["fut"]), V.append(z["speed"])
    F, V = np.concatenate(F).astype(np.float32), np.concatenate(V).astype(np.float32)
    ok = np.isfinite(F).all((1, 2)) & np.isfinite(V)
    F, V = F[ok], V[ok]
    b = np.digitize(V, VBINS)
    out, rng = [], np.random.default_rng(0)
    for j in range(len(VBINS) + 1):
        f = F[b == j]
        f = f[rng.permutation(len(f))[:n_max]]
        x = torch.as_tensor(f[:, :, :2].reshape(len(f), -1), device=dev)
        c = x[torch.as_tensor(rng.choice(len(f), k, replace=False), device=dev)].clone()
        for _ in range(it):
            a = torch.cdist(x, c).argmin(1)
            for q in range(k):
                m = a == q
                if m.any():
                    c[q] = x[m].mean(0)
        a = torch.cdist(x, c).argmin(1).cpu().numpy()
        Y = np.stack([np.median(f[a == q, :, 2], 0) if (a == q).any() else np.zeros(8, np.float32) for q in range(k)])
        out.append(np.concatenate([c.cpu().numpy().reshape(k, 8, 2), Y[..., None]], -1))
    return np.stack(out)                                                           # (5, k, 8, 3)


def gdist(d, gp):
    """dense poses (..., 3), logged path gp (86, 2) -> distance to the path (...), station (m, 0 = the row origin's foot point is not used)."""
    D = np.hypot(d[..., None, 0] - gp[:, 0], d[..., None, 1] - gp[:, 1])
    return D.min(-1), D.argmin(-1).astype(np.float64)


def cmd_score(a):
    import torch
    sys.path[:0] = [str(R / "experiments/body1/lib"), str(R / "lib")]
    import serve_body as SB
    import sweep as SW
    dev = torch.device("cuda")
    ev = json.loads((X / "events.json").read_text())
    G = pickle.load(open(X / "geo.pkl", "rb"))
    rp = json.loads((X / "replay.json").read_text())
    VOC = vocab(dev)
    body = SB.Body(dev)
    rows, recs = [], []
    for e in ev:
        sc = e["scene"]
        chk = rp.get(sc)
        if not isinstance(chk, dict) or not (X / "tok" / f"{sc}.npz").exists():
            e["drop"] = "no replay"
            continue
        T = np.load(X / "tok" / f"{sc}.npz")
        for k in e["ks"]:
            g = G.get(sc, {}).get(k)
            if k in chk["bad_k"] or k not in T["k"] or g is None or "box" not in g or "sdf_road" not in g:
                recs.append(dict(scene=sc, k=k, drop="replay" if k in chk["bad_k"] else (g or {}).get("excl", "no row") or "no raster"))
                continue
            i = int(np.flatnonzero(T["k"] == k)[0])
            P = T["poses"][i].astype(np.float64)
            assert np.abs(P - g["poses"]).max() < 1e-3
            v = float(np.hypot(T["ego"][i][4], T["ego"][i][5]) * 10)
            L, names = family(P)
            Q = np.concatenate([L, VOC[np.digitize(v, VBINS)]]).astype(np.float32)
            names = names + [f"voc{j}" for j in range(VOC.shape[1])]
            lab = SW.labels(Q[None], np.zeros((1, 2), np.float32), g["box"][None], g["bval"][None], g["cls"][None], g["sdf_road"][None].astype(np.float32))
            dd = SW.dense(Q[None], np.zeros((1, len(Q), 2), np.float32))[0]           # (Q, 41, 3)
            late = SW.TAU >= T_JUDGE
            m, ins = SW.corner_margins(dd[None], g["sdf_road"][None].astype(np.float32))
            m = np.where(ins.all(-1), m.min(-1), np.inf)[0]                          # (Q, 41)
            off = (m[:, late] < -0.20).any(1)
            gd, st = gdist(dd, g["gpath"].astype(np.float64))
            cor = (gd[:, late] > CORR).any(1)
            _, st0 = gdist(np.zeros((1, 3)), g["gpath"].astype(np.float64))
            prog = st[:, -1] - st0[0]
            col = lab["a_hit"][0] | lab["a_rear"][0]
            pp = prog[0]
            progok = prog >= (0.8 * pp if pp >= 1.0 else pp - 0.2)
            safe = ~col & ~off & ~cor
            za, zb = body.score_many(T["tokens"][i][T["valid"][i]], T["valid"][i], T["ego"][i], Q.astype(np.float64))
            r = np.maximum(1 / (1 + np.exp(-za)), 1 / (1 + np.exp(-zb)))
            arc = arcs(Q.astype(np.float64))[:, -1]
            recs.append(dict(scene=sc, k=k, v=v, n=len(Q), p0_col=bool(col[0]), p0_off=bool(off[0]), p0_cor=bool(cor[0]), p0_prog=float(pp),
                             n_safe=int(safe.sum()), n_safe_prog=int((safe & progok).sum()), n_safe_prog_local=int((safe & progok)[:len(L)].sum()),
                             r0=float(r[0]), rank_p0=float((r < r[0]).mean())))
            rows.append(dict(scene=sc, k=k, za=za, zb=zb, r=r, col=col, off=off, cor=cor, prog=prog, progok=progok, arc=arc, names=np.array(names)))
    pickle.dump(dict(events=ev, recs=recs, rows=rows), open(X / "cand.pkl", "wb"), protocol=4)
    report(ev, recs, rows)


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    from scipy.stats import rankdata
    rk = rankdata(s)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def report(ev, recs, rows):
    by = {(r["scene"], r["k"]): r for r in rows}
    E = []
    for e in ev:
        if e.get("drop"):
            continue
        ks = [k for k in e["ks"] if (e["scene"], k) in by]
        sp = {k: int((~by[e["scene"], k]["col"] & ~by[e["scene"], k]["off"] & ~by[e["scene"], k]["cor"] & by[e["scene"], k]["progok"]).sum()) for k in ks}
        pre = [k for k in ks if k <= e["k_on"] and sp[k] > 0]
        x = dict(scene=e["scene"][-16:-8], type=e["type"], k_ev=e["k_ev"], k_on=e["k_on"], e1=e["e1_alarm"], ks=ks, n_sp=[sp[k] for k in ks],
                 k_first=min([k for k in ks if sp[k] > 0], default=None), recoverable=bool(pre), k_r=max(pre) if pre else None)
        if pre:
            R = by[e["scene"], max(pre)]
            sp_mask = ~R["col"] & ~R["off"] & ~R["cor"] & R["progok"]
            feas = R["arc"] >= 0.8 * R["arc"][0]
            j = int(np.where(feas, R["r"], np.inf).argmin())
            x |= dict(pick=str(R["names"][j]), pick_safe_prog=bool(sp_mask[j]), pick_below_p0=bool(R["r"][j] < R["r"][0]), p0_safe_prog=bool(sp_mask[0]),
                      head_ok=bool(sp_mask[j] and R["r"][j] < R["r"][0]), r0=float(R["r"][0]), rpick=float(R["r"][j]),
                      safe_above_p0=int((sp_mask & (R["r"] < R["r"][0])).sum()), auc_safe=auc(~sp_mask, R["r"]))
        E.append(x)
    n = len(E)
    nr = sum(x["recoverable"] for x in E)
    nh = sum(x.get("head_ok", False) for x in E)
    A = {}
    for kind, col, s in (("collision", "col", "za"), ("off-raster", "off", "zb"), ("corridor", "cor", "r")):
        y = np.concatenate([r[col] for r in rows])
        z = np.concatenate([r[s] for r in rows])
        per = [auc(r[col], r[s]) for r in rows]
        per = [p for p in per if p is not None]
        A[kind] = dict(pooled=auc(y, z), positives=int(y.sum()), n=int(len(y)), decisions_with_both=len(per), median_per_decision=float(np.median(per)) if per else None)
    g1, g2 = nr / max(n, 1) >= 0.6, nh / max(nr, 1) >= 0.5
    res = dict(events=n, dropped=[e["scene"] for e in ev if e.get("drop")], recoverable=nr, head_ok=nh, gate_recoverable=g1, gate_rank=g2,
               passed=bool(g1 and g2 and n >= 30), auc=A, per_event=E, decisions=recs)
    (X / "score.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    md = [f"# ONPOL-LONG Amendment 12: candidate-scoring feasibility on the zero events of {TAG} (held 452, 18 decisions)", "",
          f"{n} events read ({len(res['dropped'])} dropped: {res['dropped']}); {len(rows)} decisions, {sum(len(r['r']) for r in rows)} candidates "
          f"({len(rows[0]['r']) if rows else 0} per decision: p0, 31 local variants, 256 vocabulary trajectories of the speed bin).", "",
          f"**Gate 1 (recoverable, line >= 60 %)**: {nr} / {n} = {nr / max(n, 1):.2f} -> {'pass' if g1 else 'FAIL'}.  "
          f"**Gate 2 (head ranks a safe progressing candidate above p0, line >= 50 % of the recoverable)**: {nh} / {nr} = {nh / max(nr, 1):.2f} -> {'pass' if g2 else 'FAIL'}.  "
          f"**Line: {'CONTINUE' if res['passed'] else 'STOP'}**.", "",
          "| kind | pooled AUC (candidate commits it vs not; collision: agent logit, off-raster: boundary logit, corridor: risk r) | positives / candidates | decisions with both | median per-decision AUC |",
          "|:--|--:|--:|--:|--:|"]
    md += [f"| {k} | {f(v['pooled'])} | {v['positives']} / {v['n']} | {v['decisions_with_both']} | {f(v['median_per_decision'])} |" for k, v in A.items()]
    md += ["", "Per event: window decisions with the count of safe and progressing candidates; k_first = earliest decision with one; k_r = the latest such "
           "decision at or before the onset k_on (E1 alarm, else k_ev); at k_r the head's pick (lowest risk among candidates with own arc >= 0.8 x p0's).", "",
           "| scene | type | k_ev | k_on (E1) | safe+progress counts over the window | k_first | recoverable | k_r | p0 safe+prog at k_r | head pick | pick safe+prog | r(pick) / r(p0) | safe above p0 | AUC r (unsafe vs safe+prog) |",
           "|:--|:--|--:|:--|:--|--:|:--|--:|:--|:--|:--|:--|--:|--:|"]
    for x in E:
        md.append(f"| {x['scene']} | {x['type']} | {x['k_ev']} | {x['k_on']}{' (E1)' if x['e1'] else ''} | " + " ".join(f"k{k}:{c}" for k, c in zip(x["ks"], x["n_sp"]))
                  + f" | {x['k_first']} | {x['recoverable']} | {x['k_r'] if x['k_r'] is not None else '-'} | "
                  + (f"{x['p0_safe_prog']} | {x['pick']} | {x['pick_safe_prog']} | {x['rpick']:.2f} / {x['r0']:.2f} | {x['safe_above_p0']} | {f(x['auc_safe'])} |" if x["recoverable"] else "- | - | - | - | - | - |"))
    md += ["", "Open-loop counterfactuals over a 4 s horizon with logged (non-reactive) traffic; labels from the simulator's actor boxes, the road-and-lane raster "
           "(t >= 1 s) and the logged path (4 m, t >= 1 s); the head (d223, two seeds, no retraining) was trained on P2H10 student plans.", ""]
    (X / "score.md").write_text("\n".join(md))
    print("\n".join(md[:12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "replay", "score"):
        sub.add_parser(c)
    p = sub.add_parser("geo")
    p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"prep": cmd_prep, "replay": cmd_replay, "geo": cmd_geo, "score": cmd_score}[a.cmd](a)
