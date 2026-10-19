#!/usr/bin/env python3
"""LEADSTART-PROBE (plans/2026-10-19-leadstart-probe-prereg.md): when a stopped lead starts to move, at which decision after its onset do the
policy's frozen tokens (sh30 2 Hz CAM_F0 path, synthesised slots included) and its lead head show it? Render (R1 wait set v2 lead classes,
JTN-L18-s0 closed loop; R2 arena held launch / static windows with a close lead, JTN-L18-s1 native) against real trainval frames (T,
navsim/tv-clean-train-logs, the same warp slot protocol). Results: results/leadstart_probe.md. No training.

  $AP leadstart_probe.py select     (box, CPU) R1 / R2 windows, the lead at the window start and its 20 Hz track from the raw log -> $O/lead.json;
                                    dump lists $O/lists/{wsa,ar_s<shard>_<j>}.txt (R1 b reuses launch_diag's JTN-L18-s0 dump)
  $TR leadstart_probe.py tvsel      (box, CPU) T events (ego standing, a stopped lead <= 15 m that starts) from the trainval pickles -> $O/tv/rows.npz;
                                    split navsim/leadstart_tv (tokens)
  $NS leadstart_probe.py tvindex    (box) navsim_zs index entries of the T tokens (wt.cmd_index's call) -> runs/navsim_zs/index/leadstart_tv{.pkl,_future.npz}
      (then pp_prep.py --data idx_leadstart_tv --frames warp --no-side --synth cpu)
  $TR leadstart_probe.py feats      (box, GPU) R: every used dumped decision -> tokens (8, 32, 512) fp16, ego, keys, lead heads (RSG-L18-s0, JTN-L18-s1,
                                    route memory zero), plan v0, gate (the dumping model with the dumped memory against the dumped plan) -> $O/feats_R.npz;
                                    T: the cached warp tokens + tab -> the same heads -> $O/feats_T.npz
  $TR leadstart_probe.py probe      (box, GPU) log-fold cross-fitted probes per source, AUC by time since onset, lead-head change -> $O/probe/probe.json
  python leadstart_probe.py report [--src DIR]   probe.json -> markdown tables (stdout)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/alpasim_proxy/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix/leadstart_probe"
LD = DATA / "runs/defect_fix/launch_diag"
AR = DATA / "runs/alpasim_proxy/arena"
WSET = REPO / "experiments/alpasim_proxy/results/wait_set/set.json"
NAV = DATA / "datasets/navsim"
CENSUS = DATA / "runs/self_iter/tv_census/frames.npz"
SPLIT_T, SPLIT_TV = "navsim/tv-clean-train-logs", "leadstart_tv"
HEADS = {"RSG": "RSG-L18-s0", "JTN1": "JTN-L18-s1"}
DUMPER = {"R1": "JTN-L18-s0", "R2": "JTN-L18-s1"}
FRONT = 4.049                                     # rear axle -> front bumper (wait_set / wc)
MOVE, REST0, GAP_MAX, LAT = 0.3, 0.1, 15.0, 1.8   # onset threshold, max start drift (R: first 0.5 s), lead gap, lead lateral window
V_STAND, D_STAND = 0.3, 0.3                       # ego standing: fed v0 and the oldest fed key's distance from t0
BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0))
T_NEG = -3.0                                      # T: negatives within 3 s before the onset
FOLDS, B_BOOT, FPR, PER_LIST = 5, 2000, 0.10, 60


def fold_of(logs):
    return np.array([int(hashlib.sha1(x.encode()).hexdigest(), 16) % FOLDS for x in logs])


# ---------------------------------------------------------------- select (R1 / R2 lead tracks, AlpaSim venv)
def lead_track(L, w0, n_fr):
    """The lead at frame w0 (launch_diag's rule, gap <= GAP_MAX) and its displacement from its w0 position at every frame w0 .. w0 + n_fr - 1
    (nan where the track has no box) -> dict or None."""
    import wait_set as WS
    fr = list(range(w0, min(w0 + n_fr, len(L.tok))))
    B = L.boxes(fr)
    lead = None
    for tk, x, y, yw, ln, wd in B[w0]:
        ex, ey = WS.to_ego(L.xy[w0], L.yaw[w0], (x, y))
        gap = ex - ln / 2 - WS.FRONT
        if abs(ey) < LAT and gap > -0.5 and gap < GAP_MAX and (lead is None or gap < lead[1]):
            lead = (tk, gap, np.array([x, y]))
    if lead is None:
        return None
    pos = {f: next((np.array([x, y]) for tk, x, y, *_ in B[f] if tk == lead[0]), None) for f in fr}
    disp = np.array([np.nan if pos[f] is None else float(np.hypot(*(pos[f] - lead[2]))) for f in fr])
    return dict(track=lead[0], gap0=round(float(lead[1]), 3), disp=disp)


def onset_of(disp, hz=20):
    i = np.flatnonzero(np.nan_to_num(disp, nan=-1) >= MOVE)
    return None if not len(i) else float(i[0] / hz)


def cmd_select(a):
    import csv
    import gzip
    from collections import defaultdict
    import wait_set as WS
    O.mkdir(parents=True, exist_ok=True)
    jobs = defaultdict(list)                                   # log -> [(source, window id, w0, n_frames, meta)]
    for w in json.loads(WSET.read_text())["windows"]:
        if w["cls"] in ("other_lead", "red_queue"):
            jobs[w["log"]].append(("R1", w["id"], int(w["start_idx"]), 16 * WS.HZ, dict(variant=w["variant"], cls=w["cls"])))
    with gzip.open(AR / "set/set.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["split"] == "held" and r["stratum"] in ("launch", "static") and r["lead"] == "True" and float(r["lead_gap"] or 99) <= GAP_MAX:
                jobs[r["log"]].append(("R2", r["id"], int(r["start_idx"]), WS.WIN, dict(stratum=r["stratum"], shard=int(r["shard"]))))
    out, why = {}, defaultdict(int)
    for log, ws in sorted(jobs.items()):
        L = WS.Log(log)
        for src, wid, w0, n, meta in ws:
            t = lead_track(L, w0, n)
            if t is None:
                why[f"{src} no lead <= {GAP_MAX} m at w0"] += 1
                continue
            d = t.pop("disp")
            if not np.isfinite(d[:11]).all() or np.nanmax(d[:11]) >= REST0:
                why[f"{src} lead moving at w0"] += 1
                continue
            out[wid] = dict(src=src, log=log, w0=w0, onset_s=onset_of(d), disp20=np.round(d, 3).tolist(), **t, **meta)
        L.c.close()
    (O / "lead.json").write_text(json.dumps(out))
    (O / "lists").mkdir(exist_ok=True)
    wsa = sorted(k for k, v in out.items() if v["src"] == "R1" and v["variant"] == "a")
    (O / "lists/wsa.txt").write_text("\n".join(wsa) + "\n")
    by = defaultdict(list)
    for k, v in out.items():
        if v["src"] == "R2":
            by[v["shard"]].append(k)
    for s, ids in sorted(by.items()):
        for j, c in enumerate(np.array_split(np.array(sorted(ids)), int(np.ceil(len(ids) / PER_LIST)))):
            (O / "lists" / f"ar_s{s}_{j}.txt").write_text("\n".join(c) + "\n")
    summ = defaultdict(lambda: [0, 0, set()])
    for v in out.values():
        key = (v["src"], v.get("variant") or v.get("stratum"))
        summ[key][0] += 1
        summ[key][1] += v["onset_s"] is not None
        summ[key][2].add(v["log"])
    for k, (n, m, lg) in sorted(summ.items()):
        print(f"{k}: windows {n}, lead reaches {MOVE} m {m}, logs {len(lg)}")
    print("dropped:", dict(why))
    print("lists:", " ".join(sorted(p.stem for p in (O / "lists").glob("*.txt"))))


# ---------------------------------------------------------------- tvsel (T events, op-train)
def _tv_log(p):
    import pickle
    sys.path.insert(0, str(REPO / "experiments/wait_cap/scripts"))
    import wc
    fr = pickle.load(open(p, "rb"))
    n = len(fr)
    ts = np.array([f["timestamp"] for f in fr], float) * 1e-6
    v = np.array([np.hypot(*np.asarray(f["ego_dynamic_state"][:2], float)) for f in fr])
    P = np.array([f["ego2global_translation"][:2] for f in fr], float)
    Y = np.array([wc.yaw_of(f["ego2global_rotation"]) for f in fr])
    glob_ = []                                                     # per frame: {track: global xy} of vehicles, and the in-path lead
    for i, f in enumerate(fr):
        an = f["anns"]
        B, nm, tk = np.asarray(an["gt_boxes"], float).reshape(-1, 7), np.asarray(an["gt_names"]), np.asarray(an["track_tokens"])
        c, s = np.cos(Y[i]), np.sin(Y[i])
        g = {t: P[i] + np.array([c * b[0] - s * b[1], s * b[0] + c * b[1]]) for t, b in zip(tk, B)}
        isv = nm == "vehicle"
        gap = B[:, 0] - B[:, 3] / 2 - FRONT
        q = isv & (B[:, 0] > 0) & (np.abs(B[:, 1]) < LAT) & (gap > -0.5) & (gap < GAP_MAX)
        lead = (str(tk[q][np.argmin(gap[q])]), float(gap[q].min())) if q.any() else None
        glob_.append((g, lead))
    ok = np.r_[False, np.abs(np.diff(ts) - 0.5) < 0.05]               # frame i follows i - 1 at 2 Hz
    rows, i = [], 0
    while i < n:
        if v[i] >= V_STAND:
            i += 1
            continue
        e = i
        while e + 1 < n and ok[e + 1] and v[e + 1] < V_STAND:
            e += 1
        s0, lead = i, glob_[i][1]
        i = e + 1
        if lead is None or e - s0 < 4:
            continue
        tk, gap0 = lead
        p0 = glob_[s0][0][tk]
        d = np.array([np.hypot(*(glob_[j][0][tk] - p0)) if tk in glob_[j][0] else np.nan for j in range(s0, e + 1)])
        if not (np.isfinite(d[1:3]).all() and d[1:3].max() < 0.15):
            continue
        hit = np.flatnonzero(np.nan_to_num(d, nan=-1) >= MOVE)
        if not len(hit) or hit[0] == 0 or not np.isfinite(d[hit[0] - 1]):
            continue
        j1 = hit[0]
        tau = ts[s0 + j1 - 1] + (MOVE - d[j1 - 1]) / max(d[j1] - d[j1 - 1], 1e-6) * 0.5
        for j in range(3, e - s0 + 1):                                 # the 4 history keys standing inside the run
            dt = ts[s0 + j] - tau
            if T_NEG <= dt < BINS[-1][1] and np.isfinite(d[j]):
                rows.append((fr[s0 + j]["token"], fr[s0 + j]["log_name"], s0 + j, f"{fr[s0]['log_name']}:{s0}", float(dt), bool(dt >= 0),
                             float(d[j]), gap0, float(v[s0 + j])))
    return rows


def cmd_tvsel(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    tr = splits.load(SPLIT_T)
    with Run("defect_fix", "leadstart-tvsel", config=vars(a)) as run:
        run.use_split(tr)
        logs = sorted(set(tr.members))[:a.limit or None]
        res = par.pmap(_tv_log, [NAV / "navsim_logs/trainval" / f"{l}.pkl" for l in logs], run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        R = [x for v in res.values for x in v]
        cols = ("token", "log", "frame", "event", "dt", "pos", "disp", "gap0", "v0")
        T = {k: np.array([r[j] for r in R]) for j, k in enumerate(cols)}
        z = np.load(CENSUS)
        idx = {t: i for i, t in enumerate(z["token"].astype(str).tolist())}
        ci = np.array([idx.get(t, -1) for t in T["token"].tolist()])
        keep = ci >= 0
        keep[keep] &= z["cam4"][ci[keep]].astype(bool)
        run.info(f"rows {len(ci)}, eligible in the census (4 s future, route, 4 CAM_F0) {int(keep.sum())}")
        T = {k: v[keep] for k, v in T.items()}
        ci = ci[keep]
        # light flag per event: the census' first stop line at the event's first used row is a light line within 60 m
        lt = (z["ln_type"][ci] == 2) & (z["ln_dF"][ci] <= 60)
        ev = T["event"]
        first = {}
        for k, (e, f) in enumerate(zip(ev.tolist(), T["frame"].tolist())):
            if e not in first or f < T["frame"][first[e]]:
                first[e] = k
        T["light"] = np.array([bool(lt[first[e]]) for e in ev.tolist()])
        T["cell"] = z["cell"][ci].astype(str)
        if a.limit:
            print(f"smoke: rows {len(ev)}, events {len(set(ev.tolist()))}, pos {int(T['pos'].sum())}, light rows {int(T['light'].sum())}; nothing written")
            return
        (O / "tv").mkdir(parents=True, exist_ok=True)
        np.savez(O / "tv/rows.npz", **T)
        sp = splits.define("navsim", SPLIT_TV, sorted(set(T["token"].tolist())), unit="token",
                           origin="LEADSTART-PROBE tvsel (experiments/defect_fix/scripts/leadstart_probe.py): trainval frames of navsim/tv-clean-train-logs "
                                  "where the ego stands and a stopped lead <= 15 m starts (3 s before to 2 s after its 0.3 m onset); rows "
                                  "$DATA_DIR/runs/defect_fix/leadstart_probe/tv/rows.npz",
                           status="frozen", used_by=["experiments/defect_fix"], notes="probe frames only; never trained, never an evaluation set")
        gl = set(np.flatnonzero(z["cell"].astype(str) == "G_L"))
        s = dict(rows=int(len(ev)), events=int(len(set(ev.tolist()))), logs=int(len(set(T["log"].tolist()))), pos=int(T["pos"].sum()),
                 light_events=int(len({e for e, l in zip(ev.tolist(), T["light"].tolist()) if l})),
                 rows_in_GL_cell=int(sum(int(c) in gl for c in ci)), split=f"{sp.dataset}/{sp.name}@v{sp.version}")
        for lo, hi in BINS:
            s[f"pos [{lo}, {hi})"] = int(((T["dt"] >= lo) & (T["dt"] < hi)).sum())
        run.summary.update(s)
        print(json.dumps(s, indent=1))


# ---------------------------------------------------------------- tvindex (navsim2): wt.cmd_index's call on the T tokens
def cmd_tvindex(a):
    import importlib.util
    import pickle
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(NAV))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    spec = importlib.util.spec_from_file_location("nzi", REPO / "experiments/zeroshot_openloop/archive/navsim_zs_index.py")
    nzi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nzi)
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from jevdrive.navsim_zs import root
    z = np.load(O / "tv/rows.npz")
    toks = sorted(set(z["token"].tolist()))
    logs = sorted(set(z["log"].tolist()))
    sf = SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1, has_route=True, log_names=logs, tokens=toks)
    real = NAV / "sensor_blobs/trainval"
    L = SceneLoader(NAV / "navsim_logs/trainval", real, sf)
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
    with open(root("index") / f"{SPLIT_TV}.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)
    np.savez(root("index") / f"{SPLIT_TV}_future.npz", tokens=np.array([e["token"] for e in out]), poses=np.stack(fut).astype(np.float32))
    print(f"{SPLIT_TV}: {len(out)} tokens, not loaded {len(miss)}, missing CAM_F0 files {nmiss}")
    assert nmiss == 0 and len(miss) <= 0.01 * len(toks)


# ---------------------------------------------------------------- feats (GPU)
def standing(ego):
    return 10 * float(ego[4]) < V_STAND and 10 * float(np.hypot(ego[8], ego[9])) < D_STAND


def label(dt):
    """-> keep, positive (rows with an onset: negatives before it, positives up to 2 s after; windows without one: all negative)."""
    if dt is None:
        return True, False
    return dt < BINS[-1][1], dt >= 0


def cmd_feats(a):
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO / "lib")]
    import redline_probe as RP
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.run import Run
    lead = json.loads((O / "lead.json").read_text())
    with Run("defect_fix", "leadstart-feats", config=vars(a)) as run:
        tags = sorted(set(HEADS.values()) | set(DUMPER.values()))
        cores = {t: SC.Core(t, "cuda", "backwarp") for t in tags}

        def fwd(core, H, ego, rm=None):
            """H (B, 8, 32, 512) cuda; ego (B, 20) -> lead (B, 75) [72 means, 3 prob logits], plan mu (B, 33, 15)."""
            row = ego if not core.rm else np.c_[ego, np.zeros((len(ego), core.rm_dim), np.float32) if rm is None else rm]
            tc = torch.tensor([[1.0, 0.0]], device="cuda").expand(len(ego), 2)
            o = core.model(H, torch.from_numpy(np.ascontiguousarray(row, np.float32)).cuda(), tc)
            o = (o[0] if isinstance(o, tuple) else o).float()
            sl = core.model.net.slices
            ld = o[:, sl["lead"]]
            return torch.cat([ld[:, :ld.shape[1] // 2], o[:, sl["lead_prob"]]], 1).cpu().numpy(), o[:, core.pi].reshape(-1, 33, 15).cpu().numpy()

        if a.part in ("R", "all"):
            src = {"R1": {**RP.dumped(LD / "dump/JTN-L18-s0", "all/driver-logs/drive.jsonl"), **RP.dumped(O / "dump/wsa", "run/driver-logs/drive.jsonl")},
                   "R2": RP.dumped(O / "dump", "ar_*/run/driver-logs/drive.jsonl")}
            F = {k: [] for k in ("id", "src", "log", "k", "t", "dt", "pos", "disp", "gap0", "nkeys", "ego", "tok", "cam_x", "gate")}
            F |= {f"lead_{h}": [] for h in HEADS} | {f"pv0_{h}": [] for h in HEADS}
            miss = 0
            for wid, L in sorted(lead.items())[:a.limit or None]:
                if wid not in src[L["src"]]:
                    miss += 1
                    continue
                f, ses_n = src[L["src"]][wid]
                recs = RP.records(f, ses_n, set(range(64)))
                for k in sorted(recs):
                    r = recs[k]
                    p = Path(f).parent / "dump" / f"s{ses_n:02d}_k{k}.npz"
                    if k == 0 or not p.exists():
                        continue
                    t = r["now"] / 1e6
                    fi = int(round(t * 20))
                    if fi >= len(L["disp20"]) or L["disp20"][fi] is None or not np.isfinite(L["disp20"][fi]):
                        continue
                    dt = None if L["onset_s"] is None else t - L["onset_s"]
                    keep, pos = label(dt)
                    ego = np.asarray(r["ego"], np.float32)
                    if not keep or not standing(ego):
                        continue
                    d = np.load(p)
                    assert d["valid"].all(), (wid, k)
                    cur = d["cur"]
                    prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
                    cd = cores[DUMPER[L["src"]]]
                    with torch.no_grad():
                        pp, cu = (torch.from_numpy(x).cuda() for x in (prev, cur))
                        H = cd.model.net.run_batched(cd.A.vision_feeds(pp, cu), ["view_39"])["view_39"].reshape(1, 8, *cd.A.H_SHAPE)
                        rm = np.asarray(r.get("rmem") or [], np.float32)[None] if cd.rm else None
                        _, mu = fwd(cd, H, d["ego"][None].astype(np.float32), rm if rm is not None and rm.shape[1] == cd.rm_dim else None)
                        P = I.to_rear(mu[0, :, 0:3], mu[0, :, 11], I.T_IDXS, d["cam_t"][:2].astype(np.float64), Z.T_OUT, "lever")
                        F["gate"].append(float(np.abs(P[:, :2] - d["poses"][:, :2]).max()))
                        for h, tg in HEADS.items():
                            ld, mu = fwd(cores[tg], H, d["ego"][None].astype(np.float32))
                            F[f"lead_{h}"].append(ld[0]), F[f"pv0_{h}"].append(float(mu[0, 0, 3]))
                        F["tok"].append(H[0].half().cpu().numpy())
                    nk = int(r.get("n_keys", 4))
                    for key, val in (("id", wid), ("src", L["src"]), ("log", L["log"]), ("k", k), ("t", t), ("dt", np.nan if dt is None else dt),
                                     ("pos", pos), ("disp", float(L["disp20"][fi])), ("gap0", L["gap0"]), ("nkeys", nk), ("ego", d["ego"].astype(np.float32)),
                                     ("cam_x", float(d["cam_t"][0]))):
                        F[key].append(val)
                run.info(f"{wid}: rows {len(F['id'])}")
            g = np.array(F["gate"])
            run.info(f"R rows {len(g)}, windows without a dump {miss}; gate (dumper, dumped memory vs dumped plan) max {g.max():.4f} m, p99 {np.quantile(g, 0.99):.4f}")
            assert g.max() <= 0.15, "gate failed: recomputed plan differs from the dumped plan"
            if a.limit:
                run.info(f"smoke: {len(g)} rows, gate max {g.max():.4f}, pos {sum(F['pos'])}, standing rows kept")
                return
            np.save(O / "tok_R.npy", np.stack(F.pop("tok")))
            np.savez(O / "feats_R.npz", **{k: np.stack(v) if k == "ego" or k.startswith("lead_") else np.array(v) for k, v in F.items()})
            run.summary.update(R_rows=int(len(g)), R_gate_max=float(g.max()), R_missing=miss)

        if a.part in ("T", "all"):
            CR = DATA / "runs/op_parity/cache"
            tab = np.load(CR / f"idx_{SPLIT_TV}/tab.npz")
            front = np.load(CR / f"idx_{SPLIT_TV}@warp/front.npy", mmap_mode="r")
            z = np.load(O / "tv/rows.npz")
            pos_ = {t: i for i, t in enumerate(tab["names"].astype(str).tolist())}
            j = np.array([pos_.get(t, -1) for t in z["token"].astype(str).tolist()])
            ok = j >= 0
            ok[ok] = [standing(e) for e in tab["ego"][j[ok]]]               # the R rows' standing rule on the fed row
            run.info(f"T rows {len(j)}, with tokens {int(ok.sum())}")
            j = j[ok]
            ego = tab["ego"][j].astype(np.float32)
            n = len(j)
            G = {f"lead_{h}": np.zeros((n, 75), np.float32) for h in HEADS} | {f"pv0_{h}": np.zeros(n, np.float32) for h in HEADS}
            tok = np.lib.format.open_memmap(O / "tok_T.npy", "w+", np.float16, (n, 8, 32, 512))
            for b in range(0, n, 256):
                s = slice(b, min(b + 256, n))
                Hn = np.ascontiguousarray(front[np.sort(j[s])])                       # sorted reads from the memmap, then back to row order
                Hn = Hn[np.argsort(np.argsort(j[s]))]
                tok[s] = Hn
                with torch.no_grad():
                    H = torch.from_numpy(Hn).cuda()
                    for h, tg in HEADS.items():
                        ld, mu = fwd(cores[tg], H, ego[s])
                        G[f"lead_{h}"][s], G[f"pv0_{h}"][s] = ld, mu[:, 0, 3]
            tok.flush()
            np.savez(O / "feats_T.npz", **{k: z[k][ok] for k in z.files}, ego=ego, cam_x=tab["cam"][j, 0].astype(np.float32),
                     nkeys=np.full(n, 4), **G)
            run.summary.update(T_rows=int(n))


# ---------------------------------------------------------------- probe
def fit_tok(Ttr, Ktr, ytr, Tte, Kte, seed, slot=None):
    """redline_probe.fit_tok's net (per-token LN -> 16 -> MLP 256, batch 64, 30 epochs) on one slot (slot j) or all 8 (slot None:
    8 x 32 x 16 flattened). Ttr / Tte: cuda fp16 (n, 8, 32, 512)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    ns = 8 if slot is None else 1
    mu, sd = Ktr.mean(0), Ktr.std(0) + 1e-6
    net = nn.ModuleDict(dict(ln=nn.LayerNorm(512, elementwise_affine=False), red=nn.Linear(512, 16),
                             mlp=nn.Sequential(nn.Linear(ns * 32 * 16 + Ktr.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)))).cuda()
    sel = (lambda t: t) if slot is None else (lambda t: t[:, slot])  # noqa: E731
    f = lambda t, k: net["mlp"](torch.cat([net["red"](net["ln"](sel(t).float())).flatten(1), k], 1))[:, 0]  # noqa: E731
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = torch.tensor(float((1 - ytr).sum() / max(ytr.sum(), 1)), device="cuda")
    Kt, Ke = (torch.from_numpy(((x - mu) / sd).astype(np.float32)).cuda() for x in (Ktr, Kte))
    yt = torch.from_numpy(ytr.astype(np.float32)).cuda()
    g = np.random.default_rng(seed)
    for _ in range(30):
        net.train()
        for b in np.array_split(g.permutation(len(ytr)), max(1, len(ytr) // 64)):
            b = torch.from_numpy(b).cuda()
            loss = nn.functional.binary_cross_entropy_with_logits(f(Ttr[b], Kt[b]), yt[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        ev = lambda T, K: torch.cat([f(T[i:i + 1024], K[i:i + 1024]) for i in range(0, len(K), 1024)]).cpu().numpy()  # noqa: E731
        return ev(Ttr, Kt), ev(Tte, Ke)


def acc_reads(Z, h):
    """dRel (lead x - camera to front bumper), vLead (v_ego + lead v - plan v0), lead prob of head h (serve_fix's radard rule)."""
    ld = Z[f"lead_{h}"]
    mu = ld[:, :72].reshape(-1, 3, 6, 4)
    drel = mu[:, 0, 0, 0] - (FRONT - Z["cam_x"])
    vlead = 10 * Z["ego"][:, 4] + mu[:, 0, 0, 2] - Z[f"pv0_{h}"]
    return drel, vlead, 1 / (1 + np.exp(-np.clip(ld[:, 72], -30, 30)))


def cmd_probe(a):
    import torch
    import rf_stopprobe as RS
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = O / "probe"
    out.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "leadstart-probe", config=vars(a)) as run:
        run.use_split(splits.load(SPLIT_T))
        zr, zt = np.load(O / "feats_R.npz"), np.load(O / "feats_T.npz")
        TK = {"R": np.load(O / "tok_R.npy", mmap_mode="r"), "T": np.load(O / "tok_T.npy", mmap_mode="r")}
        assert set(zt["log"].tolist()) <= set(splits.load(SPLIT_T).members), "a T row outside the registered clean train logs"
        srcs = {"R1": zr["src"] == "R1", "R2": zr["src"] == "R2", "R": np.ones(len(zr["src"]), bool), "T": None}
        J = dict(sources={}, change={})
        allscores = {}
        for name, m in srcs.items():
            z = zt if name == "T" else zr
            m = np.ones(len(z["log"]), bool) if m is None else m
            Z = {k: z[k][m] for k in z.files}
            im = np.flatnonzero(m)
            tk = TK["T" if name == "T" else "R"]
            tok = torch.cat([torch.from_numpy(np.ascontiguousarray(tk[im[i:i + 2048]])).cuda() for i in range(0, len(im), 2048)])
            n = int(m.sum())
            log, y = Z["log"].astype(str), Z["pos"].astype(bool)
            dt = Z["dt"].astype(float)
            grp = Z["event"].astype(str) if name == "T" else Z["id"].astype(str)
            fold = fold_of(log)
            kin = np.c_[Z["ego"], np.eye(5, dtype=np.float32)[np.clip(Z["nkeys"].astype(int), 0, 4)][:, 1:]]
            S, TAU = {}, {}
            lin = {"KIN": kin} | {f"LEAD-{h}": np.c_[kin, Z[f"lead_{h}"]] for h in HEADS}
            dr, vl, lp = acc_reads(Z, "RSG")
            first = {}
            for i in np.argsort(Z["t"] if name != "T" else Z["frame"], kind="stable"):
                first.setdefault(grp[i], i)
            S["ACC-v"], S["ACC-dd"] = vl, dr - dr[[first[g] for g in grp]]
            lin["TOK-lin"] = np.c_[kin, torch.cat([tok[i:i + 1024].float().mean(2).flatten(1) for i in range(0, n, 1024)]).cpu().numpy()]
            jobs = [(k, "lin", X) for k, X in lin.items()] + [("TOK", "tok", None)] + [(f"TOK-s{j}", "tok", j) for j in range(8)]
            for nm in ("ACC-v", "ACC-dd"):
                TAU[nm] = [float(np.quantile(S[nm][~y & (fold != k)], 1 - FPR)) for k in range(FOLDS)]
            for nm, kind, X in jobs:
                sc, tau = np.full(n, np.nan), []
                for k in range(FOLDS):
                    tr, te = fold != k, fold == k
                    if kind == "lin":
                        s_tr, s_te = RS.fit_lin(X[tr], y[tr], X[te])
                    else:
                        s_tr = s_te = 0
                        itr, ite = (torch.from_numpy(np.flatnonzero(q)).cuda() for q in (tr, te))
                        Ttr, Tte = tok[itr], tok[ite]
                        for seed in range(3):
                            a_, b_ = fit_tok(Ttr, kin[tr], y[tr].astype(int), Tte, kin[te], seed, X)
                            s_tr, s_te = s_tr + a_ / 3, s_te + b_ / 3
                        del Ttr, Tte
                    sc[te] = s_te
                    tau.append(float(np.quantile(s_tr[~y[tr]], 1 - FPR)))
                S[nm], TAU[nm] = sc, tau
                run.info(f"{name} {nm} done")
            del tok
            torch.cuda.empty_cache()
            allscores[name] = S
            sets = ["KIN", *[f"LEAD-{h}" for h in HEADS], "ACC-v", "ACC-dd", "TOK-lin", "TOK", *[f"TOK-s{j}" for j in range(8)]]

            def table(mask):
                mm = mask & np.all([np.isfinite(S[s]) for s in sets], 0)
                r = dict(n_pos=int((y & mm).sum()), n_neg=int((~y & mm).sum()), logs_pos=int(len(set(log[mm & y]))))
                if r["n_pos"] < 5 or r["n_neg"] < 5:
                    return r
                res, draws = RS.boot_auc({s: S[s][mm] for s in sets}, y[mm], log[mm])
                r["auc"] = res
                lb = max([f"LEAD-{h}" for h in HEADS], key=lambda s: res[s][0])
                r["lead_best"] = lb
                for b in ("KIN", lb):
                    d = draws["TOK"] - draws[b]
                    r[f"TOK-{'LEAD' if b == lb else b}"] = (res["TOK"][0] - res[b][0], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
                return r
            D = dict(rows=n, windows_or_events=int(len(set(grp))), logs=int(len(set(log))), pos=int(y.sum()), neg=int((~y).sum()), bins={}, extra={})
            for lo, hi in BINS:
                D["bins"][f"[{lo}, {hi})"] = table(~y | ((dt >= lo) & (dt < hi)))
            D["all"] = table(np.ones(n, bool))
            if name == "T":
                for nm_, q in (("no light", ~Z["light"].astype(bool)), ("light", Z["light"].astype(bool))):
                    D["extra"][nm_] = {f"[{lo}, {hi})": table(q & (~y | ((dt >= lo) & (dt < hi)))) for lo, hi in BINS}
            if name == "R1":
                lead = json.loads((O / "lead.json").read_text())
                q = np.array([lead[i]["variant"] == "b" for i in Z["id"].astype(str)])
                D["extra"]["variant b only"] = {f"[{lo}, {hi})": table(q & (~y | ((dt >= lo) & (dt < hi)))) for lo, hi in BINS}
            g0 = Z["gap0"].astype(float)
            for glo, ghi in ((-1, 6), (6, 10), (10, 15.01)):
                q = (g0 >= glo) & (g0 < ghi)
                D["extra"][f"gap0 [{glo}, {ghi})"] = {f"[{lo}, {hi})": table(q & (~y | ((dt >= lo) & (dt < hi)))) for lo, hi in BINS}
            D["fire_at_10fpr"] = {f"[{lo}, {hi})": {s: float(np.mean([S[s][i] >= TAU[s][fold[i]] for i in np.flatnonzero(y & (dt >= lo) & (dt < hi))]))
                                                    if (y & (dt >= lo) & (dt < hi)).any() else None for s in ("KIN", "LEAD-RSG", "LEAD-JTN1", "ACC-v", "TOK")}
                                  for lo, hi in BINS}
            D["fire_at_10fpr"]["neg"] = {s: float(np.mean([S[s][i] >= TAU[s][fold[i]] for i in np.flatnonzero(~y)])) for s in ("KIN", "LEAD-RSG", "LEAD-JTN1", "ACC-v", "TOK")}
            J["sources"][name] = D
            # lead-head change from the last decision before the onset (ACC's signal), against the logged lead displacement
            C = {}
            for h in HEADS:
                drh, vlh, lph = acc_reads(Z, h)
                tt = Z["t"] if name != "T" else Z["frame"] * 0.5
                ref, jit = {}, []
                o = np.lexsort((tt, grp))
                for i0, i1 in zip(o[:-1], o[1:]):
                    if grp[i0] == grp[i1] and not y[i0] and not y[i1] and abs(tt[i1] - tt[i0] - 0.5) < 0.05:
                        jit.append(abs(drh[i1] - drh[i0]))
                for i in o:
                    if not y[i] and dt[i] >= -1.0:
                        ref[grp[i]] = i                                  # the last pre-onset row (within 1 s of the onset)
                jit = np.array(jit)
                Ch = dict(jitter_p90=float(np.quantile(jit, 0.9)) if len(jit) else None, jitter_n=int(len(jit)), bins={})
                for lo, hi in BINS[:2]:
                    q = np.flatnonzero(y & (dt >= lo) & (dt < hi) & np.isin(grp, list(ref)))
                    if len(q) < 5:
                        Ch["bins"][f"[{lo}, {hi})"] = dict(n=int(len(q)))
                        continue
                    r0 = np.array([ref[grp[i]] for i in q])
                    vals = dict(d_drel=drh[q] - drh[r0], d_vlead=vlh[q] - vlh[r0], d_prob=lph[q] - lph[r0], d_logged=Z["disp"][q] - Z["disp"][r0])
                    lg = log[q]
                    u = np.unique(lg)
                    per = [np.flatnonzero(lg == l) for l in u]
                    rng = np.random.default_rng(0)
                    bt = {k: [] for k in vals}
                    for _ in range(B_BOOT):
                        pick = np.concatenate([per[c] for c in rng.integers(0, len(u), len(u))])
                        for k, v in vals.items():
                            bt[k].append(np.median(v[pick]))
                    Ch["bins"][f"[{lo}, {hi})"] = dict(n=int(len(q)), logs=int(len(u)),
                                                       **{k: (float(np.median(v)), float(np.quantile(bt[k], 0.025)), float(np.quantile(bt[k], 0.975)))
                                                          for k, v in vals.items()},
                                                       frac_drel_up_over_jitter=float(np.mean(vals["d_drel"] > (Ch["jitter_p90"] or 0))))
                C[h] = Ch
            J["change"][name] = C
            np.savez(out / f"scores_{name}.npz", **S, y=y, dt=dt, log=log, grp=grp)
        json.dump(J, open(out / "probe.json", "w"), indent=1, default=float)
        run.summary.update({f"{s}_TOK_bins": {b: r.get("auc", {}).get("TOK") for b, r in D["bins"].items()} for s, D in J["sources"].items()})


# ---------------------------------------------------------------- report
def cmd_report(a):
    J = json.load(open(Path(a.src) / "probe.json"))
    f = lambda t: f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"  # noqa: E731
    g = lambda t: f"{t[0]:+.3f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731
    L = []
    for s, D in J["sources"].items():
        L += ["", f"## {s}: rows {D['rows']}, windows / events {D['windows_or_events']}, logs {D['logs']}, pos {D['pos']}, neg {D['neg']}", ""]
        bins = list(D["bins"])
        sets = list(next(r for r in D["bins"].values() if "auc" in r)["auc"]) if any("auc" in r for r in D["bins"].values()) else []
        L += ["| set | " + " | ".join(bins) + " |", "|---|" + "--:|" * len(bins)]
        L.append("| pos (logs) | " + " | ".join(f"{D['bins'][b]['n_pos']} ({D['bins'][b]['logs_pos']})" for b in bins) + " |")
        for st in sets:
            L.append(f"| {st} | " + " | ".join(f(D["bins"][b]["auc"][st]) if "auc" in D["bins"][b] else "-" for b in bins) + " |")
        for k in ("TOK-KIN", "TOK-LEAD"):
            L.append(f"| **{k}** | " + " | ".join(g(D["bins"][b][k]) if k in D["bins"][b] else "-" for b in bins) + " |")
        for nm, E in D["extra"].items():
            L.append(f"| TOK, {nm} | " + " | ".join(f"{f(E[b]['auc']['TOK'])} n {E[b]['n_pos']}" if "auc" in E[b] else f"n {E[b]['n_pos']}" for b in bins) + " |")
            L.append(f"| LEAD best, {nm} | " + " | ".join(f(E[b]['auc'][E[b]['lead_best']]) if "auc" in E[b] else "-" for b in bins) + " |")
        L += ["", "fire at 10 % FPR: " + json.dumps(D["fire_at_10fpr"])]
        for h, C in J["change"][s].items():
            L += ["", f"lead-head change {s} / {h}: jitter p90 {C['jitter_p90']} (n {C['jitter_n']})"]
            for b, r in C["bins"].items():
                L.append(f"- {b}: n {r['n']}" + ("" if "d_drel" not in r else
                         f", logs {r['logs']}, d dRel {g(r['d_drel'])}, d vLead {g(r['d_vlead'])}, d prob {g(r['d_prob'])}, logged {g(r['d_logged'])}, "
                         f"share dRel up > jitter {r['frac_drel_up_over_jitter']:.2f}"))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "tvsel", "tvindex", "feats", "probe", "report"))
    ap.add_argument("--src", default=str(O / "probe"))
    ap.add_argument("--part", default="all", choices=("R", "T", "all"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N logs (tvsel) / windows (feats R, nothing written)")
    a = ap.parse_args()
    dict(select=cmd_select, tvsel=cmd_tvsel, tvindex=cmd_tvindex, feats=cmd_feats, probe=cmd_probe, report=cmd_report)[a.cmd](a)
