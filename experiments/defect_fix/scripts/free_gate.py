#!/usr/bin/env python3
"""NR1FREE offline gate (NR1 prereg supplement 15; lib/nr1_free.py). Box, repo root, AlpaSim's venv (torch + polars), CPU:

  $AP free_gate.py gate --out DIR       -> DIR/gate.json (+ DIR/sim_rows.npz), printed; exit 0 = pass, 2 = fail
  $TR free_gate.py pilot --tag P --ctl C --out MD   the supplement 15 pilot rule (wait set v2 + held 460 @18, P against its control C) -> MD
      (+ .json), printed

Three sets, every one scored with nr1_free.terms(parts=True) exactly as the trainer scores a pool row:
  human    the pool on1_JT's S rows with the human's own paths as the plan: `fut` (the human made reachable) and `fut_reg` (his raw travel);
           fires / rows = the false-fire rate (gate <= 1 % for each). Also the rate when the reference is one of the two alone (description).
  late     the launch_diag dumps (runs/defect_fix/launch_diag/dump/<tag>/all) of JTN-L18-s0 and RTJA0-L18-s0, wait set v2 variant b, lead
           classes: the decisions at h_launch_s <= t < the ego's own launch (controller speed > 1.0 m/s; never = the run's end) in windows the
           ego launched later than the human; the plan = the logged model plan (drive.jsonl `poses`). Gate: fires on >= 50 % (both tags pooled).
  front    the JT arena reads (runs/defect_fix/jt/arena/JT10-L18-s0 / JT00-L18-s0 @ leash3 / off), held split, windows whose zero type is
           AF-front (arena.py failure_type, incl. AF-front+corridor): the decisions in the 3 s up to the first collision_front event. A window
           fires when any of them fires. Gate: <= 10 % of the windows (pooled).
Closed-loop states: the scene time t maps to the log frame start_idx + round(20 t) (wait_set / arena convention, checked: the frame start_idx +
30 is the id's token); the controller trace (scene frame) is moved onto the log by the rigid transform that puts its first pose on the log's
pose at start_idx (residual against the log on the first 1.5 s of rollouts that follow the human is reported). Reference: the human's logged
4 s arc from the frame (raw) and its reachable version (the log's speed blended from the ego's controller speed over 2 s, 20 Hz integration);
s_ref = the smaller, per 0.5 s step. Objects: the log's vehicle / bicycle / pedestrian / generic_object boxes, the 32 nearest to the ego at the
frame followed by track token at +0.5 .. 4 s, in the ego's rear-axle frame at the decision.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/alpasim_proxy/scripts")]
import nr1_free as NF  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RAW = DATA / "datasets/nuplan_raw/test"
O = DATA / "runs/defect_fix"
WSET = REPO / "experiments/alpasim_proxy/results/wait_set/set.json"
ASET = DATA / "runs/alpasim_proxy/arena/set/set.csv.gz"
KEEP = ("vehicle", "generic_object", "pedestrian", "bicycle")
K, HZ = 32, 20
LATE_TAGS = ("JTN-L18-s0", "RTJA0-L18-s0")
FRONT_READS = ("JT10-L18-s0@leash3", "JT10-L18-s0@off", "JT00-L18-s0@leash3", "JT00-L18-s0@off")


def yaw_q(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class Log:
    def __init__(self, log):
        self.c = c = sqlite3.connect(f"file:{RAW}/{log}.db?mode=ro", uri=True)
        r = c.execute("select l.token, e.x, e.y, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy from lidar_pc l join ego_pose e on e.token = l.ego_pose_token "
                      "order by l.timestamp").fetchall()
        self.tok = [x[0] for x in r]
        a = np.array([x[1:] for x in r], float)
        self.xy, self.yaw, self.v = a[:, :2], yaw_q(*a[:, 2:6].T), np.hypot(a[:, 6], a[:, 7])
        self.cache = {}

    def boxes(self, f):
        """frame -> (track hex list, (n, 5) x, y, yaw, length, width global)."""
        if f not in self.cache:
            q = ("select lb.track_token, lb.x, lb.y, lb.yaw, lb.length, lb.width from lidar_box lb join track t on t.token = lb.track_token "
                 "join category c on c.token = t.category_token where c.name in (%s) and lb.lidar_pc_token = ?" % ",".join("?" * len(KEEP)))
            r = self.c.execute(q, [*KEEP, self.tok[f]]).fetchall()
            self.cache[f] = ([x[0].hex() for x in r], np.array([x[1:] for x in r], float).reshape(-1, 5))
        return self.cache[f]


def ctrl(path):
    rows = list(csv.reader(open(path)))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    return dict(t=A[:, h["timestamp_us"]] / 1e6, xy=A[:, [h["x"], h["y"]]], yaw=np.unwrap(yaw_q(A[:, h["qw"]], A[:, h["qx"]], A[:, h["qy"]], A[:, h["qz"]])),
                v=np.hypot(A[:, h["vx"]], A[:, h["vy"]]))


def state_row(L, s0, C, t, poses):
    """One closed-loop decision -> (poses (8, 3), s_ref (8,), box (9, K, 5), val (9, K), info) or None when the log ends before 4 s."""
    f = s0 + int(round(HZ * t))
    if f + 4 * HZ >= len(L.xy):
        return None
    dyaw = L.yaw[s0] - C["yaw"][0]
    R = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
    i = np.clip(np.searchsorted(C["t"], t), 1, len(C["t"]) - 1)
    w = np.clip((t - C["t"][i - 1]) / max(C["t"][i] - C["t"][i - 1], 1e-6), 0, 1)
    pxy = C["xy"][i - 1] + w * (C["xy"][i] - C["xy"][i - 1])
    exy = R @ (pxy - C["xy"][0]) + L.xy[s0]
    eyaw = C["yaw"][i - 1] + w * (C["yaw"][i] - C["yaw"][i - 1]) + dyaw
    ve = float(C["v"][i - 1] + w * (C["v"][i] - C["v"][i - 1]))
    seg = np.hypot(*np.diff(L.xy[f:f + 4 * HZ + 1], axis=0).T)
    raw = np.cumsum(seg)[HZ // 2 - 1::HZ // 2]
    tau = np.arange(4 * HZ) / HZ
    vb = ve + (L.v[f:f + 4 * HZ] - ve) * np.minimum(tau / 2, 1)
    reach = np.cumsum(vb / HZ)[HZ // 2 - 1::HZ // 2]
    s_ref = np.minimum(raw, reach)
    tk0, b0 = L.boxes(f)
    box, val = np.zeros((9, K, 5), np.float32), np.zeros((9, K), bool)
    if len(tk0):
        near = np.argsort(np.hypot(*(b0[:, :2] - exy).T))[:K]
        trk = [tk0[j] for j in near]
        c, s = np.cos(eyaw), np.sin(eyaw)
        for j in range(9):
            tk, b = L.boxes(f + j * HZ // 2)
            pos = {x: r for r, x in enumerate(tk)}
            for k, x in enumerate(trk):
                r = pos.get(x)
                if r is None:
                    continue
                d = b[r, :2] - exy
                box[j, k] = (c * d[0] + s * d[1], -s * d[0] + c * d[1], b[r, 2] - eyaw, b[r, 3], b[r, 4])
                val[j, k] = True
    return np.asarray(poses, np.float32), s_ref.astype(np.float32), box, val, dict(f=f, v_ego=ve, raw4=float(raw[-1]), reach4=float(reach[-1]))


def run_rollouts(run):
    """run dir -> {scene: (ctrl dict, [(k, now_s, poses)], rollout dict)}."""
    S = json.load(open(Path(run) / "aggregate/results-summary.json"))["rollouts"]
    D = defaultdict(list)
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "drive":
            D[r["scene"]].append((int(r["k"]), r["now"] / 1e6, r["poses"]))
    out = {}
    for r in S:
        f = Path(run) / f"controller/alpasim_controller_{r['rollout_id']}.csv"
        if f.exists() and r["clipgt_id"] in D:
            out[r["clipgt_id"]] = (ctrl(f), sorted(D[r["clipgt_id"]]), r)
    return out


def score(rows):
    import torch
    if not rows:
        return dict(n=0)
    P, Sr, B, V = (torch.as_tensor(np.stack([r[i] for r in rows])) for i in range(4))
    v, pr = NF.terms(P, Sr, torch.ones(len(P), dtype=torch.bool), B, V, parts=True)
    return {k: x.numpy() for k, x in pr.items()} | dict(value=v.numpy(), s_ref4=Sr[:, -1].numpy())


def cmd_human(res):
    import torch
    z = dict(np.load(DATA / "runs/op_parity/cache/on1_JT/tab.npz"))
    a = dict(np.load(DATA / "runs/op_parity/cache/on1_JT/agent.npz"))
    S = (z["mode"].astype(str) == "self") & a["ok"]
    s_ref, fin = NF.ref_arc(z["fut"], z["fut_reg"])
    m = np.flatnonzero(S & fin)
    alt = dict(ref_min=s_ref, ref_fut=np.nan_to_num(NF.cum_arc(z["fut"])).astype(np.float32),
               ref_reg=np.nan_to_num(NF.cum_arc(z["fut_reg"])).astype(np.float32))
    out = dict(rows=int(len(m)))
    for plan in ("fut", "fut_reg"):
        for rn, R in alt.items():
            fire = []
            for i in range(0, len(m), 4096):
                j = m[i:i + 4096]
                _, pr = NF.terms(torch.as_tensor(z[plan][j]), torch.as_tensor(R[j]), torch.ones(len(j), dtype=torch.bool),
                                 torch.as_tensor(a["box"][j]), torch.as_tensor(a["valid"][j]), parts=True)
                fire.append(pr["fire"].numpy())
            fire = np.concatenate(fire)
            out[f"{plan}|{rn}"] = dict(fires=int(fire.sum()), rate=float(fire.mean()))
            print(f"human plan {plan} ref {rn}: {fire.sum()} / {len(fire)} = {fire.mean():.4f}", flush=True)
    res["human"] = out


def cmd_late(res, rows_out):
    W = {w["id"]: w for w in json.load(open(WSET))["windows"] if w["variant"] == "b" and w["cls"] in ("other_lead", "red_queue")}
    out = {}
    for tag in LATE_TAGS:
        R = run_rollouts(O / "launch_diag/dump" / tag / "all")
        rows, info, nwin, nlate = [], [], 0, 0
        logs = {}
        for sc, (C, dec, _) in R.items():
            w = W.get(sc)
            if w is None or w["h_launch_s"] is None:                     # 3 windows have no logged human launch (wait_set collect)
                continue
            nwin += 1
            go = np.flatnonzero(C["v"] > 1.0)
            t_go = C["t"][go[0]] if len(go) else np.inf
            if t_go <= w["h_launch_s"]:
                continue
            nlate += 1
            L = logs.get(w["log"]) or logs.setdefault(w["log"], Log(w["log"]))
            assert L.tok[w["start_idx"] + 30].hex() == w["token"], sc
            for k, t, p in dec:
                if w["h_launch_s"] <= t < t_go:
                    r = state_row(L, w["start_idx"], C, t, p)
                    if r is not None:
                        rows.append(r)
                        info.append(dict(scene=sc, k=k, t=t, cls=w["cls"], **r[4]))
        sc_ = score(rows)
        fire = sc_.get("fire", np.zeros(0, bool))
        out[tag] = dict(windows=nwin, late_windows=nlate, decisions=len(rows), fires=int(fire.sum()), rate=float(fire.mean()) if len(fire) else None,
                        slow=int(sc_["slow"].sum()) if len(rows) else 0, free_of_slow=int((sc_["free"] & sc_["slow"]).sum()) if len(rows) else 0,
                        windows_with_fire=len({i["scene"] for i, f in zip(info, fire) if f}),
                        plan4_mean=float(sc_["s_plan"].mean()) if len(rows) else None, ref4_mean=float(sc_["s_ref4"].mean()) if len(rows) else None)
        rows_out[f"late_{tag}"] = (info, sc_)
        print(tag, json.dumps(out[tag]), flush=True)
    f = sum(v["fires"] for v in out.values())
    n = sum(v["decisions"] for v in out.values())
    out["pooled"] = dict(decisions=n, fires=f, rate=f / max(n, 1))
    res["late"] = out


def first_front(run, sc, rid):
    import polars as pl
    mp = glob.glob(f"{run}/rollouts/{sc}/{rid}/metrics.parquet") or glob.glob(f"{run}/rollouts/{sc}/*/metrics.parquet")
    if not mp:
        return None
    M = pl.read_parquet(mp[0], columns=["name", "timestamps_us", "values"]).filter((pl.col("name") == "collision_front") & (pl.col("values") > 0))
    return float(M["timestamps_us"].cast(pl.Float64).min()) / 1e6 if len(M) else None


def cmd_front(res, rows_out):
    import arena as A
    import pandas as pd
    Sx = pd.read_csv(ASET).set_index("id")
    held = set(Sx.index[Sx.split == "held"])
    out, align = {}, []
    for rd in FRONT_READS:
        rows, info, nwin, notime = [], [], 0, 0
        logs = {}
        for run in sorted(glob.glob(str(O / "jt/arena" / rd / "c*/run"))):
            if not (Path(run) / "aggregate/results-summary.json").exists():
                continue
            R = run_rollouts(run)
            for sc, (C, dec, r) in R.items():
                if sc not in held:
                    continue
                w = Sx.loc[sc]
                if (r.get("score") or 0) >= 0.98 and len(align) < 400 and not (r.get("metrics") or {}).get("collision_any"):
                    L = logs.get(w.log) or logs.setdefault(w.log, Log(w.log))
                    dyaw = L.yaw[w.start_idx] - C["yaw"][0]
                    Rm = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
                    m = C["t"] <= 1.5
                    g = (Rm @ (C["xy"][m] - C["xy"][0]).T).T + L.xy[w.start_idx]
                    fr = np.clip(w.start_idx + np.round(HZ * C["t"][m]).astype(int), 0, len(L.xy) - 1)
                    align.append(float(np.median(np.hypot(*(g - L.xy[fr]).T))))
                if not A.failure_type(r).startswith("AF-front"):
                    continue
                nwin += 1
                tc = first_front(run, sc, r["rollout_id"])
                if tc is None:
                    notime += 1
                    continue
                L = logs.get(w.log) or logs.setdefault(w.log, Log(w.log))
                assert L.tok[int(w.start_idx) + 30].hex() == w.token, sc
                for k, t, p in dec:
                    if tc - 3.0 <= t <= tc:
                        q = state_row(L, int(w.start_idx), C, t, p)
                        if q is not None:
                            rows.append(q)
                            info.append(dict(scene=sc, k=k, t=t, tc=tc, read=rd, **q[4]))
        sc_ = score(rows)
        fire = sc_.get("fire", np.zeros(0, bool))
        wins = {i["scene"] for i in info}
        out[rd] = dict(af_front_windows=nwin, no_event_time=notime, windows_with_decisions=len(wins), decisions=len(rows),
                       windows_fired=len({i["scene"] for i, f in zip(info, fire) if f}), decisions_fired=int(fire.sum()),
                       slow_decisions=int(sc_["slow"].sum()) if len(rows) else 0)
        rows_out[f"front_{rd}"] = (info, sc_)
        print(rd, json.dumps(out[rd]), flush=True)
    nw = sum(v["windows_with_decisions"] for v in out.values())
    fw = sum(v["windows_fired"] for v in out.values())
    out["pooled"] = dict(windows=nw, windows_fired=fw, rate=fw / max(nw, 1), decisions=sum(v["decisions"] for v in out.values()),
                         decisions_fired=sum(v["decisions_fired"] for v in out.values()))
    res["front"] = out
    res["align_residual_m"] = dict(n=len(align), median=float(np.median(align)) if align else None, p90=float(np.percentile(align, 90)) if align else None)


def cmd_pilot(a):
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import df
    import rtja
    W = {t: rtja.wait_stats(t) for t in (a.tag, a.ctl)}
    H = {t: df.cl_read(t, [a.ctl, "RSG-L18-s0"] if t == a.tag else ["RSG-L18-s0"])["held"] for t in (a.tag, a.ctl)}
    P, C = W[a.tag], W[a.ctl]
    LEAD = rtja.LEAD
    g = dict(held_zeros=H[a.tag]["zeros"] <= H[a.ctl]["zeros"] + 3, held_front=H[a.tag]["coll"]["front"] <= H[a.ctl]["coll"]["front"] + 2,
             **{f"launched_{c}": P[c]["launched"] >= C[c]["launched"] - 0.03 for c in LEAD},
             **{f"delay_{c}": P[c]["delay_med"] <= C[c]["delay_med"] + 0.05 for c in LEAD},
             **{f"wait_{c}": P[c]["wait"] >= C[c]["wait"] - 0.05 for c in LEAD})
    eff = any(P[c]["delay_med"] <= C[c]["delay_med"] - 0.10 for c in LEAD)
    ok = all(g.values())
    verdict = ("pass + effect: candidate for the joint arm" if ok and eff else "pass, no effect at this dose: not into the joint arm" if ok
               else "fail: stop the line")
    refs = {t: rtja.wait_stats(t) for t in ("JTN-L18-s0", "JTNS1K-L18-s0", "LDNTV1K-L18-s0", "RTB1ARO-L18-s0")}
    L = [f"# NR1FREE pilot ({a.tag} against {a.ctl}; NR1 prereg supplement 15)", "",
         "| arm | a wait other_lead / red_queue | b launched other_lead / red_queue | b delay med s | a zeros | b mean | held zeros | held mean | held front coll | held vs control [CI] | held vs RSG [CI] |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t in (a.tag, a.ctl, *refs):
        w = W.get(t) or refs[t]
        h = H.get(t)
        pc = h["pair"].get(a.ctl) if h and t == a.tag else None
        pr = h["pair"].get("RSG-L18-s0") if h else None
        L.append(f"| {t}{'' if t in W else ' (stored)'} | {w['other_lead']['wait']:.3f} / {w['red_queue']['wait']:.3f} | {w['other_lead']['launched']:.3f} / "
                 f"{w['red_queue']['launched']:.3f} | {w['other_lead']['delay_med']:+.2f} / {w['red_queue']['delay_med']:+.2f} | {w['a_zeros']} | {w['b_score']:.3f} | "
                 + (f"{h['zeros']} | {h['mean']:.4f} | {h['coll']['front']} | " if h else "- | - | - | ")
                 + (f"{pc['mean']:+.4f} [{pc['lo']:+.4f}, {pc['hi']:+.4f}] | " if pc else "- | ")
                 + (f"{pr['mean']:+.4f} [{pr['lo']:+.4f}, {pr['hi']:+.4f}] |" if pr else "- |"))
    L += ["", "Gates (supplement 15): " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in g.items()),
          f"Effect (a lead-class delay at least 0.10 s earlier than the control): {'yes' if eff else 'no'}", "", f"**Verdict: {verdict}**", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L))
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(tag=a.tag, ctl=a.ctl, wait=W, held=H, gates=g, effect=eff, verdict=verdict), indent=1, default=str))
    print("\n".join(L))


def cmd_gate(a):
    import torch
    torch.set_num_threads(16)
    res, rows_out = dict(frac=NF.FRAC, s_min=NF.S_MIN, d0=NF.D0, th=NF.TH, ns=NF.NS), {}
    for k in a.only.split(","):
        dict(human=lambda: cmd_human(res), late=lambda: cmd_late(res, rows_out), front=lambda: cmd_front(res, rows_out))[k]()
    g = dict(human=all(res["human"][f"{p}|ref_min"]["rate"] <= 0.01 for p in ("fut", "fut_reg")) if "human" in res else None,
             late=res["late"]["pooled"]["rate"] >= 0.5 if "late" in res else None,
             front=res["front"]["pooled"]["rate"] <= 0.10 if "front" in res else None)
    res["gate"] = g | dict(pass_=all(v is True for v in g.values()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(Path(a.out) / "sim_rows.npz", **{f"{k}__info": np.array(json.dumps(v[0])) for k, v in rows_out.items()},
                        **{f"{k}__{q}": x for k, v in rows_out.items() for q, x in v[1].items() if isinstance(x, np.ndarray)})
    print(json.dumps(res["gate"]))
    raise SystemExit(0 if res["gate"]["pass_"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True), p.add_argument("--only", default="human,late,front")
    p = sp.add_parser("pilot")
    p.add_argument("--tag", required=True), p.add_argument("--ctl", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(gate=cmd_gate, pilot=cmd_pilot)[a.cmd](a)
