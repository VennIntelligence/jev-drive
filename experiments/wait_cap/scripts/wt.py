#!/usr/bin/env python3
"""WAIT-TRAIN (plans/2026-10-16-wait-train-prereg.md, design plans/2026-10-16-wait-train-design.md): same-domain wait / launch / stop rows from
the downloaded NAVSIM trainval camera archives, packed as on-policy pool rows that carry ONLY an arc-length speed loss (lib/wc_arc.py), appended
to RSG-L18's pool on1_L18RB0 -> on1_L18RBW; the pilot / closed-loop gates. Box, repo root; chain in wt.sh.

  select   (op-train, CPU)   the 111 logs of results/trainval_logs.txt minus the logs holding a navtrain dev token; per frame wc.log_rows /
                             wc.cell_of (cells unchanged), CAM_F0 of the 4 history frames on disk; same log same cell >= 2 s apart, <= 5 000 per
                             cell (seed 20261016) -> runs/wait_cap/train/select/{rows.npz, tokens.txt, cells.md}; the leak check (logs vs held
                             460, public 1491, PAI 281 + train160, waitcap_test) -> leak.json; split navsim/waitcap_train
  index    (navsim2)         navsim_zs index entries of every selected token (trainval SceneLoader) -> runs/navsim_zs/index/waitcap_train{.pkl,_future.npz}
  (then)   pp_prep.py --data idx_waitcap_train --frames warp --no-side; rm_build.py nav --data idx_waitcap_train
  teacher  (op-train, GPU)   shipped Cinque (port fp16) on the rows' warp tokens, pp_prep / on_rows make_teacher's call -> cache/idx_waitcap_train@warp/teacher.npz
  pack     (op-train, CPU)   pool on1_WAIT (names on:<token>; S rows fed the constant-velocity history; agent / road / route labels ok False;
                             tab `cell`, `s_h`) -> merged on1_L18RBW = on1_L18RB0 rows first (prefix s) ++ on1_WAIT (prefix w) (rf_pait.py merge),
                             <pool>/wc_arc.npz, the draw flags per dose -> runs/wait_cap/train/flags_<dose>.txt, pack.json
  bitcmp                     the 60-step identity runs WTGG-off (RSG-L18 recipe, pool on1_L18RB0) vs WTGG-w0 (on1_L18RBW, new rows weight 0,
                             --on-lon wc_arc): scalars, weights, on_hash_60 = RSG-L18-s0's stored hash -> results/wait_train/identity.json
  devplans --tags T..  (GPU) plans of the moving (>= 2 m/s) navtrain dev rows of shard s0 -> runs/wait_cap/train/devplans/<tag>.npz
  gate --j T.. --c T [--submit]   (CPU) the registered pilot gates -> results/wait_train/pilot_gate.{json,md}; --submit: write the picked dose
                             (j2 | j5 | STOP) to runs/wait_cap/train/pilot_pick.txt. Amendment C: it no longer submits the WAIT-only formal
                             (wt.sh full); the composed WAIT + desire pilot takes the dose, `full` runs only as the registered fallback
  clgate --stage held|sub|pub|pai --j T0,T1 --c T0,T1 [--follow DIR] [--submit]   (CPU) closed-loop stage lines; --submit: the next stage
                             on PASS; --follow (Amendment C 6, the composed arm): held stage also needs the held turn completion line
                             (rf_follow in DIR/follow: 2-seed mean completed J >= C - 2)
  Amendment C (the composed WAIT + desire arm WD):
  bitcmp --a T --b T --out NAME   the same identity comparison for any two 60-step runs (default WTGG-off vs WTGG-w0 -> identity.json)
  补记 D (exploratory, after the pilot STOP; outputs under runs/wait_cap/train/explore, no registered verdict file touched):
  devall --tags T..   (GPU)  the gate-4 dev rows (dev_eval's call) -> explore/devall.npz (plans, human future, P0 teacher plan, agent context)
  xheld / xsum --j T,T --c T [--submit]   (CPU) held 460 @18 lines on one seed -> explore/held.json (+ public reads for J meeting them);
                             every closed-loop read -> explore/cl.{json,md}, explore/FINISHED
  wdgate --j T1 T3 --c T --lam-of 1=T1,3=T3 [--submit]   the composed pilot: every WAIT pilot gate + launch-lock (cmd_gate's per-arm read)
                             and the desire offline gates (dt.desire_read) -> results/wait_train/wd_pilot_gate.{json,md}; --submit: a passing
                             arm (both: D2 weight 1) -> wt.sh wd-full <lam>; none -> the registered fallback WT_FALLBACK=1 wt.sh full <dose>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
from jevdrive.tokens import open_front  # noqa: E402
import wc  # noqa: E402

DATA = wc.DATA
NAV = wc.NAV
OUT = DATA / "runs/wait_cap/train"
CR = DATA / "runs/op_parity/cache"
FEAT = DATA / "runs/route_mem/feat"
RES = REPO / "experiments/wait_cap/results/wait_train"
SPLIT = "waitcap_train"
CAP = 5000
STAT, STOP, YIELD = ("W_L", "W_N", "G_L", "G_N"), ("S_L", "S_N"), ("Y_P", "Y_V")      # Y = decision 331's yield cells (Amendment A)
BASE, POOL, NEW = "on1_L18RB0", "on1_L18RBW", "on1_WAIT"
K_ON = 13
DOSES = {"j2": 2, "j5": 5}                      # expected new rows among the 13 pool rows of a batch
RSG_HASH60 = "ea8cacc3e451c577"                  # RSG-L18-s0's stored on_hash_60 (rf_jdagger bitcmp)
STATIC_PUB = "2021.09.16.15.12.03_veh-42_01037_01434-b5b8d20688385790"     # d325 / d326: the public scene where the human never moves
STOP17 = "054b5901 095cf563 127c4f85 2d10a08d 35e082a4 3fe2bac8 44df49fe 4cb7f7ac 4ecc114d 559d4ad6 68c3be44 7d994ed3 9f8acd00 ba40be62 bc309fd6 cbd87234 fc9fde87".split()
PAI_CH = DATA / "runs/alpasim/pai_box/tokyo/r281/rsg/chunks"
PAI_REF = DATA / "runs/route_follow/pai_g10/acc"                         # RSG-L18-s0 under the PAI submission config (d328)


def db_of(scene):
    return scene.rsplit("-", 1)[0]


# ---------------------------------------------------------------- select (CPU)
def _led():
    import importlib.util
    spec = importlib.util.spec_from_file_location("led", REPO / "experiments/cap_ledger/scripts/led.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


WC_KEYS = None


def _log_job(p):
    """One trainval log -> wait_cap rows with one cell each: wc.cell_of first (W_L .. S_N), else decision 331's Y_P, else Y_V (led.cells_of,
    unchanged), so a yield frame already in S_L / S_N stays there once; frames whose 4 history CAM_F0 are not on disk are dropped."""
    led = _led()
    fr = pickle.load(open(p, "rb"))
    ok = [(NAV / "sensor_blobs/trainval" / f["cams"]["CAM_F0"]["data_path"]).exists() for f in fr]
    out = []
    for r in led.log_cells(p):
        if not all(ok[r["frame"] - 3:r["frame"] + 1]):
            continue
        c = wc.cell_of(r)
        if c is None:
            lc = led.cells_of(r)
            c = "Y_P" if lc["Y_P"] else "Y_V" if lc["Y_V"] else None
        if c is None:
            continue
        q = {k: v for k, v in r.items() if not (k.startswith(("ln_", "lane", "grp")) or k in ("dpsi2", "dpsi4", "vmin4", "junc", "off0", "off2", "off4"))}
        q["cell"] = c
        out.append(q)
    return out, sum(ok), len(fr)


def leak(logs):
    """Every held / public / PAI / waitcap_test set against the training logs (exact nuPlan db name, and the drive session date_veh)."""
    L = set(logs)
    ses = {"_".join(l.split("_")[:2]) for l in L}
    sets = {}
    held = {s.strip() for f in glob.glob(str(DATA / "runs/onpol_long/lists/held_n6/c*.txt")) for s in open(f) if s.strip()}
    pub = {s.strip() for s in open(DATA / "runs/onpol_long/lists/all1491.txt") if s.strip()}
    pai = {s.strip() for f in glob.glob(str(PAI_CH / "*.txt")) for s in open(f) if s.strip()}
    wt = set(np.load(DATA / "runs/wait_cap/select/rows.npz")["log"].tolist())
    for nm, S, conv in (("held460", held, db_of), ("public1491", pub, db_of), ("pai281", pai, lambda s: s), ("waitcap_test", wt, lambda s: s)):
        dbs = {conv(s) for s in S}
        sets[nm] = dict(n=len(S), db_overlap=sorted(L & dbs), session_overlap=sorted(ses & {"_".join(d.split("_")[:2]) for d in dbs}))
    return dict(logs=len(L), sets=sets, clean=all(not v["db_overlap"] for v in sets.values()))


def bsplit(b):
    return f"waitcap_tr{b}"


def used_logs():
    """The 111 listed logs minus the ones holding a navtrain dev token -> (used logs, dropped)."""
    from jevdrive.data import splits
    want = [l for l in (REPO / "experiments/wait_cap/results/trainval_logs.txt").read_text().split() if l]
    dv = splits.load("navsim/op-parity-full-dev")
    devlogs = set()
    for i in range(12):
        z = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        devlogs |= set(z["log"][dv.mask(z["names"])].tolist())
    return [l for l in want if l not in devlogs], sorted(set(want) & devlogs)


def cmd_select(a):
    """Batch B: the used logs of the archives whose .done exists and that no earlier batch took; every cell row after the 2 s thinning (the
    5 000 cap is applied over the union at pack) -> runs/wait_cap/train/batches/<B>/{rows.npz, logs.txt, cells.md, select.json}, split navsim/waitcap_tr<B>."""
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    logs_all, dropped = used_logs()
    amap = json.loads((REPO / "experiments/wait_cap/results/trainval_archive_logs.json").read_text())
    done = {k for k in amap if (NAV / "_archives/state" / f"{k}.done").exists()}
    ready = {l for k in done for l in amap[k]}
    out = OUT / "batches" / a.batch
    assert not (out / "rows.npz").exists(), f"batch {a.batch} exists"
    taken = {l for f in (OUT / "batches").glob("*/logs.txt") for l in f.read_text().split()}
    logs = [l for l in logs_all if l in ready and l not in taken]
    assert logs, "no new log ready"
    with Run("wait_cap", f"train-select-{a.batch}", config=vars(a), seed=wc.SEED) as run:
        res = par.pmap(_log_job, [NAV / f"navsim_logs/trainval/{l}.pkl" for l in logs], run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        rows = [r for v, _, _ in res.values for r in v]
        cams = dict(frames=sum(n for _, _, n in res.values), cam_f0=sum(c for _, c, _ in res.values))
        keep, census = [], {}
        for c in STAT + STOP + YIELD:
            cand = sorted((r for r in rows if r["cell"] == c), key=lambda r: (r["log"], r["frame"]))
            thin, last = [], {}
            for r in cand:
                if r["frame"] - last.get(r["log"], -99) >= wc.GAP_FR:
                    thin.append(r)
                    last[r["log"]] = r["frame"]
            census[c] = dict(frames=len(cand), thinned=len(thin), logs=len({r["log"] for r in thin}))
            keep += thin
        out.mkdir(parents=True, exist_ok=True)
        cols = [k for k in keep[0] if k not in ("fut", "ref")]
        np.savez(out / "rows.npz", fut=np.stack([r["fut"] for r in keep]), ref=np.stack([r["ref"] for r in keep]), **{k: np.array([r[k] for r in keep]) for k in cols})
        (out / "logs.txt").write_text("\n".join(logs) + "\n")
        lk = leak(sorted({r["log"] for r in keep}))
        rec = dict(batch=a.batch, archives_done=sorted(done), logs=len(logs), dropped_dev_logs=dropped, cell_frames=len(rows), **cams, census=census,
                   rows=len(keep), leak=lk)
        md = [f"batch {a.batch}: {len(logs)} logs ({len(done)} archives done); frames {cams['frames']}, CAM_F0 on disk {cams['cam_f0']}; frames in a cell with "
              f"history cameras {len(rows)} (one cell per frame: wc cell first, then Y_P, then Y_V); rows after the 2 s thinning {len(keep)}", "",
              "| cell | frames | rows | logs |", "|---|--:|--:|--:|"] + [f"| {c} | {v['frames']} | {v['thinned']} | {v['logs']} |" for c, v in census.items()]
        md += ["", "leak check (exact nuPlan db / drive session date_veh): " + "; ".join(
            f"{k} {len(v['db_overlap'])} / {len(v['session_overlap'])}" for k, v in lk["sets"].items()) + f" -> clean {lk['clean']}"]
        (out / "cells.md").write_text("\n".join(md) + "\n")
        (out / "select.json").write_text(json.dumps(rec, indent=1))
        run.summary |= dict(rows=len(keep), census=census, leak_clean=lk["clean"])
        print((out / "cells.md").read_text())
        assert lk["clean"], "a training log is an evaluation log"
        splits.define("navsim", bsplit(a.batch), sorted(r["token"] for r in keep), unit="token",
                      origin=f"WAIT-TRAIN select batch {a.batch} (experiments/wait_cap/scripts/wt.py): wait / launch / stop / yield cells of NAVSIM trainval "
                             f"frames of the downloaded camera archives; rows $DATA_DIR/runs/wait_cap/train/batches/{a.batch}/rows.npz",
                      status="frozen", used_by=["experiments/wait_cap"], notes="training row candidates (pool on1_WAIT after the cap); never an evaluation set")


# ---------------------------------------------------------------- index (navsim2 venv)
def cmd_index(a):
    import importlib.util
    split, sel = bsplit(a.batch), OUT / "batches" / a.batch / "rows.npz"
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(NAV))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    spec = importlib.util.spec_from_file_location("nzi", REPO / "experiments/zeroshot_openloop/archive/navsim_zs_index.py")
    nzi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nzi)
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from jevdrive.navsim_zs import root
    z = np.load(sel)
    toks, logs = z["token"].tolist(), sorted(set(z["log"].tolist()))
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
    with open(root("index") / f"{split}.pkl", "wb") as f:
        pickle.dump(out, f, protocol=4)
    np.savez(root("index") / f"{split}_future.npz", tokens=np.array([e["token"] for e in out]), poses=np.stack(fut).astype(np.float32))
    print(f"{split}: {len(out)} tokens, not loaded {len(miss)}, missing CAM_F0 files {nmiss}")
    assert nmiss == 0 and len(miss) <= 0.01 * len(toks)


# ---------------------------------------------------------------- teacher (GPU)
def cmd_teacher(a):
    import torch
    sys.path[:0] = [str(REPO / p) for p in ("lib", "scripts", "experiments/op_adapt_r2/lib", "experiments/op_parity/scripts")]
    import pp_prep as PP
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    from jevdrive.run import Run
    d = f"idx_{bsplit(a.batch)}"
    tab = np.load(CR / d / "tab.npz")
    front = open_front(CR / f"{d}@warp/front.npy")
    n = len(tab["names"])
    assert front.shape == (n, 8, 32, 512), front.shape
    dev = torch.device("cuda")
    with Run("wait_cap", f"train-teacher-{a.batch}", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{bsplit(a.batch)}"))
        net, _ = PP.encoder(dev)
        di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
        o_, plan = np.zeros((n, len(di)), np.float32), np.zeros((n, 33, 15), np.float32)
        tc_all = np.where(tab["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)
        with torch.no_grad():
            for i in range(0, n, 256):
                r = slice(i, min(i + 256, n))
                H = torch.from_numpy(np.ascontiguousarray(front[r])).to(dev)
                H = torch.cat([torch.zeros_like(H[:, :1]), H], 1)
                valid = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
                valid[:, 0] = False
                o = A._policy(net, H, (0.275, 0.525), torch.from_numpy(tc_all[r]).to(dev), valid)["outputs"].float()
                o_[r], plan[r] = o[:, di].cpu().numpy(), o[:, pi].cpu().numpy().reshape(-1, 33, 15)
        np.savez(CR / f"{d}@warp/teacher.npz", out=o_, plan=plan, di=di, pi=pi)
        # the same call on 64 rows of on1_L18RB0 reproduces their stored teacher (on_rows make_teacher, the protocol check)
        zb = np.load(CR / f"{BASE}@warp/teacher.npz")
        fb = open_front(CR / f"{BASE}@warp/front.npy")
        tb = np.load(CR / BASE / "tab.npz")
        with torch.no_grad():
            H = torch.from_numpy(np.ascontiguousarray(fb[:64])).to(dev)
            H = torch.cat([torch.zeros_like(H[:, :1]), H], 1)
            valid = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
            valid[:, 0] = False
            tc = torch.from_numpy(np.where(tb["lht"][:64, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)).to(dev)
            o = A._policy(net, H, (0.275, 0.525), tc, valid)["outputs"].float()[:, di].cpu().numpy()
        err = float(np.abs(o - zb["out"][:64]).max())
        run.info(f"teacher {n} rows; protocol check on 64 on1_L18RB0 rows: max |diff| {err:.3g}")
        run.summary |= dict(rows=n, protocol_max_abs=err)
        assert np.array_equal(di, zb["di"]) and np.array_equal(pi, zb["pi"]) and err < 1e-2, err


# ---------------------------------------------------------------- pack (CPU)
def flat_rows(E):
    """rf_lonol.flat_rows: vx kept, vy = ax = ay = 0, the 3 past poses re-spaced along their own polyline to v x (0.5, 1.0, 1.5) s (vhist.respace)."""
    sys.path.insert(0, str(REPO / "experiments/alpasim/lib"))
    import vhist as VH
    F = E.astype(np.float64).copy()
    F[:, 5:8] = 0.0
    for i in range(len(F)):
        P = E[i, 8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
        F[i, 8:20] = (VH.respace(P, max(float(E[i, 4]) * 10.0, 0.0)) / [10.0, 10.0, 1.0]).reshape(12)
    return F.astype(np.float32)


def human_arc(fut):
    P = np.concatenate([np.zeros((len(fut), 1, 2)), fut[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1).astype(np.float32)


def cmd_pack(a):
    """Union of the batches (every used log covered), the 5 000 cap per cell over the union (seed 20261016, rows sorted by log / frame), then
    on1_WAIT from the batch caches, merged on1_L18RBW, wc_arc.npz, flags; split navsim/waitcap_train = the packed tokens."""
    from jevdrive.data import splits
    assert not (CR / POOL / "tab.npz").exists() or os.environ.get("WT_FORCE"), f"{POOL} exists (a training may read it): WT_FORCE=1"
    bs = a.batches.split(",")
    logs_all, dropped = used_logs()
    got = [l for b in bs for l in (OUT / "batches" / b / "logs.txt").read_text().split()]
    assert sorted(got) == sorted(logs_all), f"batches cover {len(set(got))} of {len(logs_all)} used logs"
    parts = []                                                        # (batch, cache tab, sel rows by token)
    for b in bs:
        d = f"idx_{bsplit(b)}"
        tb = dict(np.load(CR / d / "tab.npz"))
        sl = np.load(OUT / "batches" / b / "rows.npz")
        pos = {t: i for i, t in enumerate(sl["token"].tolist())}
        j = np.array([pos[t] for t in tb["names"].tolist()])
        parts.append((b, d, tb, {k: sl[k][j] for k in ("cell", "log", "frame")}))
    cell_all = np.concatenate([p[3]["cell"] for p in parts]).astype("U4")
    key = [(str(l), int(f)) for p in parts for l, f in zip(p[3]["log"], p[3]["frame"])]
    rng = np.random.default_rng(wc.SEED)
    take = np.zeros(len(cell_all), bool)
    census = {}
    for c in STAT + STOP + YIELD:
        idx = sorted(np.flatnonzero(cell_all == c), key=lambda i: key[i])
        pick = [idx[k] for k in sorted(rng.permutation(len(idx))[:CAP])]
        take[pick] = True
        census[c] = dict(rows_before_cap=len(idx), rows=len(pick), logs=len({key[i][0] for i in pick}))
    off = np.cumsum([0] + [len(p[2]["names"]) for p in parts])
    sel = [np.flatnonzero(take[off[k]:off[k + 1]]) for k in range(len(parts))]
    cat = lambda k: np.concatenate([p[2][k][s_] for p, s_ in zip(parts, sel)])  # noqa: E731
    tb = {k: cat(k) for k in ("names", "log", "ego", "pose", "vel", "acc", "cmd", "fut", "cam", "lht", "speed")}
    cell = np.concatenate([cell_all[off[k]:off[k + 1]][s_] for k, s_ in enumerate(sel)])
    names = tb["names"]
    n = len(names)
    splits.define("navsim", SPLIT, sorted(names.tolist()), unit="token",
                  origin="WAIT-TRAIN pack (experiments/wait_cap/scripts/wt.py): the batches' rows after the 5 000 cap per cell; pool on1_WAIT",
                  status="frozen", used_by=["experiments/wait_cap"], notes="training rows; never an evaluation set")
    assert np.isfinite(tb["fut"]).all(), "rows without a full future"
    stop = np.isin(cell, STOP + YIELD)                  # S and Y rows: constant-velocity history, over-the-human-only arc loss
    ego = tb["ego"].copy()
    ego[stop] = flat_rows(tb["ego"][stop])
    s_h = human_arc(tb["fut"])
    kind = np.where(stop, 2, 1).astype(np.int8)
    src, dw = CR / NEW, CR / f"{NEW}@warp"
    src.mkdir(exist_ok=True), dw.mkdir(exist_ok=True)
    pn = np.array([f"on:{t}" for t in names.tolist()])
    T = dict(names=pn, log=tb["log"], ego=ego, pose=tb["pose"], vel=tb["vel"], acc=tb["acc"], cmd=tb["cmd"], fut=tb["fut"], cam=tb["cam"], lht=tb["lht"],
             speed=tb["speed"], cell=cell, src=np.full(n, "wait", "U16"), s_h=s_h, kind=kind, ego_measured=tb["ego"])
    np.savez(src / "tab.npz", **T)
    Ba = np.load(CR / BASE / "agent.npz", mmap_mode="r")
    np.savez(src / "agent.npz", tokens=pn, ok=np.zeros(n, bool), box=np.zeros((n,) + Ba["box"].shape[1:], np.float32),
             valid=np.zeros((n,) + Ba["valid"].shape[1:], bool), cls=np.zeros((n,) + Ba["cls"].shape[1:], np.int8))
    Br = np.load(CR / BASE / "road.npz", mmap_mode="r")
    np.savez(src / "road.npz", tokens=pn, ok=np.zeros(n, bool), sdf=np.zeros((n,) + Br["sdf"].shape[1:], np.float16), x0=Br["x0"], y0=Br["y0"], res=Br["res"])
    Bq = np.load(CR / BASE / "route.npz", mmap_mode="r")
    np.savez(src / "route.npz", tokens=pn, ok=np.zeros(n, bool), poly=np.zeros((n,) + Bq["poly"].shape[1:], np.float32),
             seglen=np.zeros((n,) + Bq["seglen"].shape[1:], np.float32), dz=np.zeros(n, np.float32), cov20=np.zeros(n, bool))
    FZ, TH, FR = [], [], []
    for (b, d, t_, _), s_ in zip(parts, sel):
        fz = np.load(FEAT / f"{d}.npz")
        assert np.array_equal(fz["names"], t_["names"]), f"{d}: route features rows != cache rows"
        FZ.append({k: fz[k][s_] for k in ("f", "k", "ok")})
        th = np.load(CR / f"{d}@warp/teacher.npz")
        assert np.array_equal(th["di"], np.load(CR / f"{parts[0][1]}@warp/teacher.npz")["di"])
        TH.append((th["out"][s_], th["plan"][s_], th["di"], th["pi"]))
        FR.append((open_front(CR / f"{d}@warp/front.npy"), s_))
    np.savez(FEAT / f"{NEW}.npz", names=pn, f=np.concatenate([z["f"] for z in FZ]).astype(np.float16), k=np.concatenate([z["k"] for z in FZ]).astype(np.int64),
             ok=np.concatenate([z["ok"] for z in FZ]))
    fz = dict(ok=np.concatenate([z["ok"] for z in FZ]))
    np.savez(dw / "teacher.npz", out=np.concatenate([t[0] for t in TH]), plan=np.concatenate([t[1] for t in TH]), di=TH[0][2], pi=TH[0][3])
    o = np.lib.format.open_memmap(dw / "front.npy", "w+", FR[0][0].dtype, (n,) + FR[0][0].shape[1:])
    i0 = 0
    for f_, s_ in FR:
        for i in range(0, len(s_), 1024):
            o[i0 + i:i0 + min(i + 1024, len(s_))] = f_[s_[i:i + 1024]]
        i0 += len(s_)
    o.flush()
    fr = o
    p = subprocess.run([sys.executable, str(REPO / "experiments/route_follow/scripts/rf_pait.py"), "merge", "--pools", f"{BASE}:s", f"{NEW}:w", "--out", POOL[4:]],
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"rf_pait merge failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    M = dict(np.load(CR / POOL / "tab.npz"))
    B = np.load(CR / BASE / "tab.npz")
    ns = len(B["names"])
    pool = M["pool"].astype(str)
    chk = dict(base_rows=ns, new_rows=n, merged=len(M["names"]), base_first=bool((pool[:ns] == "s").all() and (pool[ns:] == "w").all()),
               base_names=bool(all(m[5:] == b[3:] for m, b in zip(M["names"][:ns].tolist(), B["names"].tolist()))),
               base_ego_equal=bool(np.array_equal(M["ego"][:ns], B["ego"])), base_fut_equal=bool(np.array_equal(M["fut"][:ns], B["fut"])),
               keys=sorted(M))
    Fm, Fb = open_front(CR / f"{POOL}@warp/front.npy"), open_front(CR / f"{BASE}@warp/front.npy")
    rs = np.random.default_rng(0).choice(ns, 64, replace=False)
    chk["base_front_equal_64"] = bool(all(np.array_equal(Fm[i], Fb[i]) for i in rs))
    chk["new_front_equal_64"] = bool(all(np.array_equal(Fm[ns + i], fr[i]) for i in rs % n))
    tm, tbz = np.load(CR / f"{POOL}@warp/teacher.npz"), np.load(CR / f"{BASE}@warp/teacher.npz")
    chk["base_teacher_equal"] = bool(np.array_equal(tm["out"][:ns], tbz["out"]))
    fm, fb = np.load(FEAT / f"{POOL}.npz"), np.load(FEAT / f"{BASE}.npz")
    chk["base_feat_equal"] = bool(np.array_equal(fm["f"][:ns], fb["f"]) and np.array_equal(fm["ok"][:ns], fb["ok"]))
    for f in ("agent.npz", "road.npz"):
        z = np.load(CR / POOL / f, mmap_mode="r")
        chk[f"{f[:-4]}_ok_new"] = int(z["ok"][ns:].sum())
        chk[f"{f[:-4]}_ok_base_equal"] = bool(np.array_equal(z["ok"][:ns], np.load(CR / BASE / f, mmap_mode="r")["ok"]))
    np.savez(CR / POOL / "wc_arc.npz", tokens=M["names"], s_h=np.concatenate([np.full((ns, 8), np.nan, np.float32), s_h]),
             kind=np.concatenate([np.zeros(ns, np.int8), kind]))
    assert all(v for k, v in chk.items() if k not in ("keys", "agent_ok_new", "road_ok_new") and isinstance(v, bool)) and chk["agent_ok_new"] == 0 \
        and chk["road_ok_new"] == 0, chk
    # draw weights: base rows weight 1 (mass ns); new rows' mass q ns / (1 - q), q = dose / 13; inside the new rows W 1/3, G 1/3 (wait : go = 1 : 1),
    # S + Y 1/3 split by their row counts, Y raised to 1/12 if under (Amendment A); natural proportions inside each group
    cnt = {c: int((cell == c).sum()) for c in STAT + STOP + YIELD}
    grp = {"W": ("W_L", "W_N"), "G": ("G_L", "G_N"), "S": STOP, "Y": YIELD}
    ng = {g: sum(cnt[c] for c in cs) for g, cs in grp.items()}
    assert all(ng.values()), ng
    sh = {"W": 1 / 3, "G": 1 / 3, "Y": max(ng["Y"] / (ng["S"] + ng["Y"]) / 3, 1 / 12)}
    sh["S"] = 1 / 3 - sh["Y"]
    flags, wts = {}, {}
    for dose, k in DOSES.items():
        q = k / K_ON
        mass = q * ns / (1 - q)
        w = {c: mass * sh[g] / ng[g] for g, cs in grp.items() for c in cs if cnt[c]}
        wts[dose] = w
        flags[dose] = f"--on {POOL} --on-k {K_ON} --on-mode self --on-lon wc_arc --on-weight " + ",".join(f"cell={c}:{w[c]:.6f}" for c in w)
    flags["w0"] = f"--on {POOL} --on-k {K_ON} --on-mode self --on-lon wc_arc --on-weight pool=w:0"
    for k_, f in flags.items():
        (OUT / f"flags_{k_}.txt").write_text(f + "\n")
    rec = dict(batches=bs, census=census, check=chk, cells=cnt, groups=ng, shares=sh, y_raised=bool(sh["Y"] > ng["Y"] / (ng["S"] + ng["Y"]) / 3 + 1e-12), s_y_rows_cv_history=int(stop.sum()), flags=flags, weights=wts,
               expected_new_per_batch={d: k for d, k in DOSES.items()}, route_feat_ok_new=float(fz["ok"].mean()))
    (OUT / "pack.json").write_text(json.dumps(rec, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pack.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- identity (CPU)
def run_of(tag):
    import torch
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{tag}/*/"))[-1]
    sc = {}
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc[(e["tag"], e["step"])] = e["value"]
    return sc, json.load(open(d + "DONE")), torch.load(f"{DATA}/runs/op_parity/runs/{tag}/ckpt-final.pt", map_location="cpu", weights_only=False)["model"]


def cmd_bitcmp(a):
    (sa, da, ma), (sb, db, mb) = run_of(a.a), run_of(a.b)
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith(("throughput/", "loss/drop/"))]
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g])
    same = all(set(ma[g]) == set(mb[g]) for g in ("net", "parity"))
    extra = sorted({k[0] for k in sb} - {k[0] for k in sa})
    wz = [sb[k] for k in sb if k[0].startswith("on/wcarc")]
    r = dict(a=a.a, b=a.b, note=a.note or f"RSG-L18 recipe 60 steps vs pool {POOL} with the new rows at weight 0 (--on-lon wc_arc)", scalars=len(keys),
             scalar_max_abs_diff=dl, weight_max_abs_diff=dw, weight_keys_equal=same, on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")],
             rsg_l18_s0_on_hash_60=RSG_HASH60, dev_ade=[da.get("dev_ade"), db.get("dev_ade")], b_only_scalars=extra,
             wcarc_scalars_max_abs=max(map(abs, wz)) if wz else None)
    r["identical"] = bool(dl == 0 and dw == 0 and same and r["on_hash_60"][0] == r["on_hash_60"][1] == RSG_HASH60)
    RES.mkdir(parents=True, exist_ok=True)
    for p in (OUT / f"{a.out}.json", RES / f"{a.out}.json"):
        p.write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))
    raise SystemExit(0 if r["identical"] else 2)


# ---------------------------------------------------------------- dev plans (GPU)
def cmd_devplans(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from types import SimpleNamespace
    sys.path[:0] = [str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    out = OUT / "devplans"
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in a.tags if not (out / f"{t}.npz").exists()]
    if not todo:
        return
    dev = torch.device("cuda")
    ns, _, _, _, _ = RT.build(SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on=BASE, init="", rm_from=todo[0], tag=todo[0], exp="wait_cap",
                                              lr_scale=1.0, prefetch=1, dev_card_gb=4.0), dev)
    S, model = ns.S, ns.model
    W = ns.losses(ns.mode).W
    pi = torch.as_tensor(S.pi, device=dev)
    rows = ns.dv_o[S.tab["speed"][ns.dv_o] >= 2.0]
    for j, tag in enumerate(todo):
        if j:
            model.load_state(torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                r = rows[i:i + 256]
                rt = torch.as_tensor(r, device=dev)
                p = model(S.front[r], S.ego[rt], S.tc[rt], None, None, nv=None).float()[:, pi].view(-1, 33, 15)
                x, y, psi = T.rear(p, S.cam_x[rt], W)
                res.append(torch.stack([x, y, psi], -1).cpu())
        P = torch.cat(res).numpy()
        f = S.fut[torch.as_tensor(rows, device=dev)].cpu().numpy()
        np.savez(out / f"{tag}.npz", rows=rows, names=S.tab["names"][rows], log=S.tab["log"][rows], poses=P.astype(np.float32), fut=f.astype(np.float32))
        print(tag, len(rows), "ADE", float(np.hypot(P[..., 0] - f[..., 0], P[..., 1] - f[..., 1]).mean()), flush=True)


# ---------------------------------------------------------------- 补记 D (exploratory, after the pilot STOP; no registered file is touched)
XO = OUT / "explore"


def lead_ctx(names):
    """Agent-label context at t0 per dev row (rear-axle frame, runs/op_parity/agent_labels/navtrain_all-k32.npz): gap to the nearest vehicle in the
    lane corridor (|y| < 1.8 m, 0 < x < 20 m; inf if none), its closing speed from t0 -> 0.5 s, and a pedestrian within x 0-20 m, |y| < 5 m."""
    z = np.load(DATA / "runs/op_parity/agent_labels/navtrain_all-k32.npz")
    pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
    cls = z["classes"].tolist()
    veh, ped = cls.index("vehicle"), cls.index("pedestrian")
    zb, zv, zc, zo = z["box"], z["valid"], z["cls"], z["ok"]                   # once: every z[...] access decompresses the whole array
    gap, close, pd_, ok = (np.full(len(names), np.inf), np.zeros(len(names)), np.zeros(len(names), bool), np.zeros(len(names), bool))
    for n, t in enumerate(names.tolist()):
        i = pos.get(t)
        if i is None or not zo[i]:
            continue
        ok[n] = True
        b, v, c = zb[i], zv[i], zc[i]
        x, y = b[0, :, 0], b[0, :, 1]
        m = v[0] & (c == veh) & (np.abs(y) < 1.8) & (x > 0) & (x < 20)
        if m.any():
            k = np.flatnonzero(m)[np.argmin(x[m])]
            gap[n] = x[k]
            close[n] = (x[k] - b[1, k, 0]) / 0.5 if v[1, k] else 0.0      # > 0: the gap shrinks (in the t0 frame; ego motion not removed)
        pd_[n] = bool((v[0] & (c == ped) & (x > 0) & (x < 20) & (np.abs(y) < 5)).any())
    return dict(lead_gap=gap, lead_close=close, ped=pd_, ctx_ok=ok)


def cmd_devall(a):
    """The gate-4 dev rows (rm_train's dev_eval call: every navtrain dev row with a future, route memory on) for each tag -> XO/devall.npz with
    the plans, the human future, shipped Cinque's stored teacher plan (P0) and the context; asserts each tag reproduces its DONE dev_ade."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from types import SimpleNamespace
    sys.path[:0] = [str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    dev = torch.device("cuda")
    ns, _, _, _, _ = RT.build(SimpleNamespace(**(REC | dict(data=RT.OT.JT.FULL, yr_mass=0.1)), road_labels=RT.OT.JT.ROAD, on=BASE, init="",
                                              rm_from=a.tags[0], tag=a.tags[0], exp="wait_cap", lr_scale=1.0, prefetch=1, dev_card_gb=4.0), dev)
    S, model, W = ns.S, ns.model, ns.losses(ns.mode).W
    Sd = T.dev_store(S, ns.dv_rows, 4.0)
    pi = torch.as_tensor(S.pi, device=dev)
    rows = ns.dv_o[S.has_fut[torch.as_tensor(ns.dv_o, device=dev)].cpu().numpy().astype(bool)]
    out = {}
    for j, tag in enumerate(a.tags):
        if j:
            model.load_state(torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        res, tp = [], []
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                r = torch.as_tensor(rows[i:i + 128], device=dev)
                side = Sd.side[r] if Sd.side is not None else (Sd.mem[r] if getattr(Sd, "mem", None) is not None else None)
                p = model(Sd.front[r], Sd.ego[r], Sd.tc[r], side, None, inputs_on=True, nv=None if Sd.nv is None else Sd.nv[r]).float()[:, pi].view(-1, 33, 15)
                res.append(torch.stack(T.rear(p, Sd.cam_x[r], W), -1).cpu())
                if not j:
                    tp.append(torch.stack(T.rear(Sd.t_plan[r], Sd.cam_x[r], W), -1).cpu())
        out[tag] = torch.cat(res).numpy().astype(np.float32)
        if not j:
            out["P0"] = torch.cat(tp).numpy().astype(np.float32)
            fut = S.fut[torch.as_tensor(rows, device=dev)].cpu().numpy().astype(np.float32)
        ade = float(np.hypot(*(out[tag][..., :2] - fut[..., :2]).transpose(2, 0, 1)).mean())
        ref = dev_ade(tag)
        print(tag, len(rows), "ADE", ade, "DONE", ref, flush=True)
        assert abs(ade - ref) < 1e-3, f"{tag}: dev ADE {ade} does not reproduce DONE {ref}"
    names = S.tab["names"][rows]
    XO.mkdir(parents=True, exist_ok=True)
    np.savez(XO / "devall.npz", rows=rows, names=names, log=S.tab["log"][rows], fut=fut, speed=S.tab["speed"][rows], tags=np.array(["P0"] + a.tags),
             **{f"plan_{k}": v for k, v in out.items()}, **lead_ctx(names))


def metrics(dirs):
    """scene -> (score, progress_rel, at-fault collision, offroad) of the first rollout per scene over the run dirs (aggregate/results-summary.json)."""
    M = {}
    for D in dirs:
        f = Path(D) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                m = r.get("metrics") or {}
                M.setdefault(r["clipgt_id"], (float(r.get("score") or 0.0), float(m.get("progress_rel", np.nan)), float(m.get("collision_at_fault", np.nan)),
                                              float(m.get("offroad", np.nan))))
    return M


def xdirs(stage, tag):
    return dict(held=f"{DATA}/runs/route_mem/read/{tag}/c*/run", pub=f"{XO}/read10/{tag}/b*", sub=f"{XO}/sub/{tag}/run",
                pai=f"{XO}/paisub/{tag}/runs/*/sim")[stage]


def xpair(MJ, MC):
    """Paired J vs C on the common scenes: zeros, mean score with the session-clustered CI, progress, at-fault collisions, offroad."""
    from jevdrive import stats
    common = sorted(set(MJ) & set(MC))
    A, B = (np.array([M[s] for s in common]) for M in (MJ, MC))
    p = stats.paired(A[:, 0], B[:, 0], groups=np.array(["_".join(s.split("_")[:2]) for s in common]))
    f = lambda X: dict(zeros=int((X[:, 0] == 0).sum()), mean=float(X[:, 0].mean()), progress_rel=float(np.nanmean(X[:, 1])),  # noqa: E731
                       at_fault=int(np.nansum(X[:, 2] > 0)), offroad=int(np.nansum(X[:, 3] > 0)))
    return dict(scenes=len(common), J=f(A), C=f(B), paired=stats.fmt(p), paired_lo=float(p["lo"]), zero_flips=dict(
        J_only=[s for s, x, y in zip(common, A[:, 0], B[:, 0]) if x == 0 < y], C_only=[s for s, x, y in zip(common, A[:, 0], B[:, 0]) if y == 0 < x]))


def cmd_xheld(a):
    """补记 D (a): held 460 @18 of each J vs C (one seed); the registered held lines moved to one seed (zeros J <= C + 3, turns J >= C - 2);
    --submit: public 1491 @10 (wt.sh explore-pub) for the J tags meeting both, then the summary job."""
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    import rf_recov_post as RP
    J, C = a.j.split(","), a.c
    MC = metrics(glob.glob(xdirs("held", C)))
    res = dict(C=C, arms={})
    want = set((RES / "held_turns24.txt").read_text().split())
    for t in J:
        r = xpair(metrics(glob.glob(xdirs("held", t))), MC)
        h = RP.h2_held(XO, [t], [C])
        r["turns"] = None if h is None else dict(completed={k: v["completed"] for k, v in h["counts"].items()},
                                                 turns={k: v["turns"] for k, v in h["counts"].items()}, paired=h["paired"], common=h["common_turns"])
        if h is not None:
            import pandas as pd
            S = pd.read_csv(XO / "follow/scenes.csv")
            T_ = S[S.turn.astype(str).str.lower().isin(("true", "1", "1.0"))]
            r["turns"]["set_is_turns24"] = {k: set(T_[T_.label == f"{k}@18"].scene) == want for k in (t, C)}
        r["line_zeros"] = bool(r["J"]["zeros"] <= r["C"]["zeros"] + 3)
        r["line_turns"] = bool(h is not None and h["counts"][t]["completed"] >= h["counts"][C]["completed"] - 2)
        r["meets"] = r["line_zeros"] and r["line_turns"]
        res["arms"][t] = r
    res["public_for"] = [t for t in J if res["arms"][t]["meets"]]
    (XO / "held.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    if a.submit:
        subprocess.run(["bash", str(HERE / "wt.sh"), "explore-pub", ",".join(res["public_for"]) or "-", C], check=True)


def cmd_xsum(a):
    """补记 D: every closed-loop read -> XO/cl.{json,md}; then XO/FINISHED."""
    J, C = a.j.split(","), a.c
    H = json.loads((XO / "held.json").read_text())
    ref = metrics(glob.glob(f"{PAI_REF}/runs/*/sim"))
    full = {pai_full(p): p for p in STOP17}
    res = dict(held=H, static={}, stop17={}, pub={})
    for t in J + [C]:
        res["static"][t] = metrics(glob.glob(xdirs("sub", t))).get(STATIC_PUB)
        m = metrics(glob.glob(xdirs("pai", t)))
        res["stop17"][t] = dict(zeros=sum(m[s][0] == 0 for s in full if s in m), n=sum(s in m for s in full),
                                mean=float(np.mean([m[s][0] for s in full if s in m])) if m else None, per={full[s]: m.get(s, (None,))[0] for s in full})
    res["stop17"]["RSG-L18-s0 (archive)"] = dict(zeros=sum(ref[s][0] == 0 for s in full if s in ref), n=sum(s in ref for s in full),
                                                 per={full[s]: ref.get(s, (None,))[0] for s in full})
    MC = metrics(glob.glob(xdirs("pub", C)))
    for t in H["public_for"]:
        r = xpair(metrics(glob.glob(xdirs("pub", t))), MC)
        r["line"] = bool(r["paired_lo"] > -0.01 and r["J"]["zeros"] <= r["C"]["zeros"] + 5)
        res["pub"][t] = r
    L = ["# 补记 D closed-loop reads (exploratory; 1 000-step pilots, seed 0; control WTCP-L18-s0)", "",
         "| tag | held zeros | held mean | paired vs C | progress_rel | at-fault | offroad | turns /24 | static scene @10 | stop17 zeros |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t in J + [C]:
        r = H["arms"].get(t) or H["arms"][J[0]]
        side = "J" if t in J else "C"
        tc = (r.get("turns") or {}).get("completed", {}).get(t)
        st = res["static"][t]
        L.append(f"| {t} | {r[side]['zeros']} | {r[side]['mean']:.3f} | {r['paired'] if t in J else '-'} | {r[side]['progress_rel']:.3f} | {r[side]['at_fault']} | "
                 f"{r[side]['offroad']} | {tc} | {st[0] if st else None} | {res['stop17'][t]['zeros']} / {res['stop17'][t]['n']} |")
    L.append(f"| RSG-L18-s0 archive (PAI config) | | | | | | | | | {res['stop17']['RSG-L18-s0 (archive)']['zeros']} / {res['stop17']['RSG-L18-s0 (archive)']['n']} |")
    L += ["", "public 1491 @10 (only J tags meeting the held lines):"]
    for t, r in res["pub"].items():
        L.append(f"- {t}: {r['scenes']} scenes, zeros {r['J']['zeros']} vs {r['C']['zeros']}, mean {r['J']['mean']:.4f} vs {r['C']['mean']:.4f}, paired {r['paired']}, "
                 f"progress {r['J']['progress_rel']:.3f} vs {r['C']['progress_rel']:.3f}, at-fault {r['J']['at_fault']} vs {r['C']['at_fault']}; line {'met' if r['line'] else 'not met'}")
    if not res["pub"]:
        L.append("- none ran")
    (XO / "cl.json").write_text(json.dumps(res, indent=1, default=float))
    (XO / "cl.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    (XO / "FINISHED").write_text("ok\n")


# ---------------------------------------------------------------- pilot gate (CPU)
def cap_read(tag):
    """Open-loop capability on the waitcap_test rows (wc.classify): wait W, wait G, D, into lead W_L, S_N stop / over median."""
    z = dict(np.load(DATA / "runs/wait_cap/select/rows.npz", allow_pickle=False))
    p = np.load(DATA / f"runs/wait_cap/plans/{tag}.npz")
    pos = {k: i for i, k in enumerate(p["names"].tolist())}
    r = wc.classify(p["plan"][np.array([pos[k] for k in z["token"].tolist()])], z["ref"], {k: z[f"lead_{k}"] for k in ("x", "y", "yaw", "l", "w")}, None)
    C, g = z["cell"], z["log"]
    W, G = np.isin(C, ("W_L", "W_N")), np.isin(C, ("G_L", "G_N"))
    wait = (r["cls"] == 0).astype(float)
    return dict(wait_W=float(wait[W].mean()), go_G=float(1 - wait[G].mean()), wait_G=float(wait[G].mean()), D=float(wait[W].mean() - wait[G].mean()),
                into_WL=float(r["into"][C == "W_L"].mean()), bypass_WLSL=float((r["cls"] == 2)[np.isin(C, ("W_L", "S_L"))].mean()),
                stop_SN=float((r["arc4"] <= z["a4"] + 2.0)[C == "S_N"].mean()), over_SN_med=float(np.median((r["arc4"] - z["a4"])[C == "S_N"])),
                stop_SL=float((r["arc4"] <= z["a4"] + 2.0)[C == "S_L"].mean())), dict(wait=wait, W=W, G=G, g=g, into=r["into"].astype(float), C=C)


def dev_read(tag, ctag):
    a, b = np.load(OUT / f"devplans/{tag}.npz"), np.load(OUT / f"devplans/{ctag}.npz")
    assert np.array_equal(a["rows"], b["rows"])
    lat = np.abs(a["poses"][:, 3, 1] - b["poses"][:, 3, 1])
    arc = lambda P: np.hypot(P[:, 0, 0], P[:, 0, 1]) + np.hypot(*np.diff(P[..., :2], axis=1).transpose(2, 0, 1)).sum(1)  # noqa: E731
    return dict(lat2s_med_absdiff=float(np.median(lat)), lat2s_p90_absdiff=float(np.percentile(lat, 90)), rows=int(len(lat)),
                arc4_med_ratio=float(np.median(arc(a["poses"]) / np.maximum(arc(b["poses"]), 0.1))),
                ade_moving=float(np.hypot(*(a["poses"][..., :2] - a["fut"][..., :2]).transpose(2, 0, 1)).mean()),
                ade_moving_c=float(np.hypot(*(b["poses"][..., :2] - b["fut"][..., :2]).transpose(2, 0, 1)).mean()))


def dev_ade(tag):
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{tag}/*/DONE"))[-1]
    return float(json.load(open(d))["dev_ade"])


def ledger_read(tag):
    """Decision 331's ledger cells S_L, Y_P, Y_V on the F read (constant-velocity history; capledger_test rows, led.py measures / success
    unchanged) -> {cell: (per-row success, logs)}; registered secondary reads, not gates."""
    led = _led()
    z = dict(np.load(DATA / "runs/cap_ledger/select/rows.npz", allow_pickle=False))
    p = np.load(DATA / f"runs/cap_ledger/plans/F/{tag}.npz")
    pos = {k: i for i, k in enumerate(p["names"].tolist())}
    m = led.measures(p["plan"][np.array([pos[k] for k in z["token"].tolist()])], z)
    out = {}
    for c in ("S_L",) + YIELD:
        msk = z["in_" + c]
        zc = {k: v[msk] for k, v in z.items()}
        out[c] = (led.success(c, {k: v[msk] for k, v in m.items()}, zc).astype(float), z["log"][msk])
    return out


def wait_arm(t, c, Cc, cx, c_ade):
    """The registered WAIT pilot gates of one arm t against the control c (section 6, unchanged): five gates + launch-lock, secondary ledger F."""
    from jevdrive import stats
    Jc, jx = cap_read(t)
    dv = dev_read(t, c)
    ade = dev_ade(t)
    dW = stats.paired(jx["wait"][jx["W"]], cx["wait"][cx["W"]], groups=jx["g"][jx["W"]])
    dG = stats.paired(jx["wait"][jx["G"]], cx["wait"][cx["G"]], groups=jx["g"][jx["G"]])
    dWm = Jc["wait_W"] - Cc["wait_W"]
    dD = Jc["D"] - Cc["D"]
    g = dict(wait_W=Jc["wait_W"] >= 0.60, go_G=Jc["go_G"] >= 0.85, into_WL=Jc["into_WL"] <= 0.05, dev_ade=ade <= c_ade + 0.01,
             lat2s=dv["lat2s_med_absdiff"] <= 0.05)
    lock = bool(dWm >= 0.20 and dD < 0.5 * dWm)
    r = dict(cap=Jc, dev=dv, dev_ade=ade, d_wait_W=stats.fmt(dW), d_wait_G=stats.fmt(dG), dD=dD, gates=g, launch_lock=lock,
             passed=bool(all(g.values()) and not lock))
    try:                                                               # secondary (not a gate): d331's ledger cells on the F read
        lj, lc = ledger_read(t), ledger_read(c)
        r["ledger_F"] = {k: dict(J=float(lj[k][0].mean()), C=float(lc[k][0].mean()), n=int(len(lj[k][0])),
                                 diff=stats.fmt(stats.paired(lj[k][0], lc[k][0], groups=lj[k][1]))) for k in lj}
    except FileNotFoundError as e:
        r["ledger_F"] = f"missing: {e}"
    return r


def cmd_gate(a):
    Cc, cx = cap_read(a.c)
    res = dict(control=a.c, C=Cc | dict(dev_ade=dev_ade(a.c)), arms={})
    for t in a.j:
        res["arms"][t] = wait_arm(t, a.c, Cc, cx, res["C"]["dev_ade"])
    ok = [t for t in a.j if res["arms"][t]["passed"]]
    if len(ok) == 2:
        d0, d1 = (res["arms"][t]["cap"]["D"] for t in ok)
        pick = ok[0] if abs(d0 - d1) < 0.03 else ok[int(d1 > d0)]           # larger D; within 0.03 the smaller dose (listed first)
    else:
        pick = ok[0] if ok else None
    res |= dict(passed=ok, pick=pick, verdict="PASS" if pick else ("STOP: launch-lock (d167 / d174 shape)" if any(res["arms"][t]["launch_lock"] for t in a.j)
                                                                     else "STOP: both doses fail"))
    L = [f"# WAIT-TRAIN pilot gates (1 000 steps, seed 0; open loop on waitcap_test, dev rows of navtrain s0)", "",
         "| arm | wait W (>= 0.60) | go G (>= 0.85) | into W_L (<= 0.05) | D | dev ADE (<= C + 0.01) | lat 2 s med abs diff vs C (<= 0.05 m) | S_N stop / over med | launch-lock | pass |",
         "|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|",
         f"| {a.c} (C) | {Cc['wait_W']:.3f} | {Cc['go_G']:.3f} | {Cc['into_WL']:.3f} | {Cc['D']:.3f} | {res['C']['dev_ade']:.4f} | - | {Cc['stop_SN']:.3f} / {Cc['over_SN_med']:.2f} | - | - |"]
    for t in a.j:
        r = res["arms"][t]
        L.append(f"| {t} | {r['cap']['wait_W']:.3f} | {r['cap']['go_G']:.3f} | {r['cap']['into_WL']:.3f} | {r['cap']['D']:.3f} | {r['dev_ade']:.4f} | "
                 f"{r['dev']['lat2s_med_absdiff']:.3f} | {r['cap']['stop_SN']:.3f} / {r['cap']['over_SN_med']:.2f} | {'yes' if r['launch_lock'] else 'no'} | "
                 f"{'PASS' if r['passed'] else 'fail'} |")
    L += ["", "secondary (not gates), decision 331's ledger cells on the F read (capledger_test, success share, J vs C paired):"]
    for t in a.j:
        lf = res["arms"][t].get("ledger_F")
        L.append(f"- {t}: " + ("; ".join(f"{c} {v['J']:.3f} vs {v['C']:.3f} (n {v['n']}), diff {v['diff']}" for c, v in lf.items()) if isinstance(lf, dict) else str(lf)))
    L += ["", "paired vs C (log-clustered bootstrap): " + "; ".join(f"{t}: wait W {res['arms'][t]['d_wait_W']}, wait G {res['arms'][t]['d_wait_G']}" for t in a.j),
          "", f"verdict: {res['verdict']}" + (f"; dose {pick}" if pick else "")]
    RES.mkdir(parents=True, exist_ok=True)
    for p in (OUT / "pilot_gate.json", RES / "pilot_gate.json"):
        p.write_text(json.dumps(res, indent=1, default=float))
    (RES / "pilot_gate.md").write_text("\n".join(L) + "\n")
    (OUT / "pilot_gate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.submit:                    # Amendment C: the gate no longer submits the WAIT-only formal; it hands the picked dose to the composed line
        dose = ({v: k for k, v in a.dose_of.items()}[pick] if a.dose_of else pick) if pick else "STOP"
        (OUT / "pilot_pick.txt").write_text(dose + "\n")   # j2 | j5 | STOP; the composed pilot (wt.sh wd-pilot) waits on this file


def cmd_wdgate(a):
    """Amendment C 4: the composed pilot arms (D2 weight 1 / 3) on every WAIT gate + the desire offline gates; pick; fallback (C 5)."""
    sys.path.insert(0, str(REPO / "experiments/desire_cap/scripts"))
    import dt
    Cc, cx = cap_read(a.c)
    res = dict(control=a.c, C=Cc | dict(dev_ade=dev_ade(a.c)), arms={}, dose=(OUT / "pilot_pick.txt").read_text().strip())
    for t in a.j:
        r = wait_arm(t, a.c, Cc, cx, res["C"]["dev_ade"])
        r["desire"] = dt.desire_read(t)
        r["wait_passed"], r["passed"] = r["passed"], bool(r["passed"] and r["desire"]["passed"])
        res["arms"][t] = r
    lam = {v: k for k, v in a.lam_of.items()}
    ok = [t for t in a.j if res["arms"][t]["passed"]]
    pick = min(ok, key=lambda t: float(lam[t])) if ok else None              # both pass: D2 weight 1
    res |= dict(passed=ok, pick=pick, pick_lam=lam.get(pick), verdict="PASS" if pick else "FAIL: fallback to the WAIT-only formal (Amendment C 5)")
    P0 = next(iter(res["arms"].values()))["desire"]["P0"]
    L = ["# WAIT + desire composed pilot gates (Amendment C 4; 1 000 steps, seed 0; WAIT dose " + res["dose"] + ")", "",
         "| arm | wait W (>= 0.60) | go G (>= 0.85) | into W_L (<= 0.05) | D | dev ADE (<= C + 0.01) | lat 2 s (<= 0.05 m) | launch-lock | "
         "LC 4 s (>= 0.9 x P0) | LCL 4 s (>= 0.8) | LCR 4 s (>= 0.8) | dir (>= 0.95) | PAI P_lca 4 s (report, 1.5) | pass |",
         "|---|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|--:|--:|:-:|",
         f"| {a.c} (C) | {Cc['wait_W']:.3f} | {Cc['go_G']:.3f} | {Cc['into_WL']:.3f} | {Cc['D']:.3f} | {res['C']['dev_ade']:.4f} | - | - | - | - | - | - | - | - |",
         f"| P0 | - | - | - | - | - | - | - | {P0['mean4']:.2f} | {P0['lcl4']:.2f} | {P0['lcr4']:.2f} | {P0['dir']:.3f} | {P0['pai_lca4']:.2f} | - |"]
    for t in a.j:
        r, d = res["arms"][t], res["arms"][t]["desire"]["arm"]
        L.append(f"| {t} (D2 {lam[t]}) | {r['cap']['wait_W']:.3f} | {r['cap']['go_G']:.3f} | {r['cap']['into_WL']:.3f} | {r['cap']['D']:.3f} | "
                 f"{r['dev_ade']:.4f} | {r['dev']['lat2s_med_absdiff']:.3f} | {'yes' if r['launch_lock'] else 'no'} | {d['mean4']:.2f} | {d['lcl4']:.2f} | "
                 f"{d['lcr4']:.2f} | {d['dir']:.3f} | {d['pai_lca4']:.2f} | {'PASS' if r['passed'] else 'fail'} |")
    L += ["", f"verdict: {res['verdict']}" + (f"; D2 weight {res['pick_lam']} ({pick})" if pick else "")]
    RES.mkdir(parents=True, exist_ok=True)
    for p in (OUT / "wd_pilot_gate.json", RES / "wd_pilot_gate.json"):
        p.write_text(json.dumps(res, indent=1, default=float))
    for p in (OUT / "wd_pilot_gate.md", RES / "wd_pilot_gate.md"):
        p.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not a.submit:
        return
    fb = OUT / "FALLBACK"                                                    # written by whichever of this gate / wt.sh wd-deadline falls back first
    if pick and not fb.exists():
        (OUT / "WD_PASS").write_text(f"{pick} D2 {res['pick_lam']}\n")
        subprocess.run(["bash", str(HERE / "wt.sh"), "wd-full", res["pick_lam"]], check=True)
    elif pick:
        print(f"composed arm {pick} passed after the 24 h fallback ({fb.read_text().strip()}): no composed formal", flush=True)
    elif not fb.exists():
        fb.write_text("composed pilot failed\n")
        subprocess.run(["bash", str(HERE / "wt.sh"), "full", res["dose"]], check=True, env=dict(os.environ, WT_FALLBACK="1"))


def held_turns(d, J, C):
    """Amendment C 6: rf_follow's completed held turns (DIR/follow) per tag, 2-seed means, the line J >= C - 2, paired (rf_recov_post.h2_held)."""
    sys.path.insert(0, str(REPO / "experiments/route_follow/scripts"))
    import rf_recov_post as RP
    import pandas as pd
    h = RP.h2_held(Path(d), J, C)
    assert h is not None, f"{d}/follow: rf_follow output missing for {J + C}"
    turns = {t: h["counts"][t]["turns"] for t in J + C}
    assert len(set(turns.values())) == 1, f"turn sets differ between tags: {turns}"
    S = pd.read_csv(Path(d) / "follow/scenes.csv")
    T = S[S.turn.astype(str).str.lower().isin(("true", "1", "1.0"))]
    want = set((RES / "held_turns24.txt").read_text().split())                # decisions 303 / 307: RSG-L18's 24 held turns (rf_follow)
    for t in J + C:
        assert set(T[T.label == f"{t}@18"].scene) == want, f"{t}: held turn scenes differ from held_turns24.txt"
    cj, cc = (float(np.mean([h["counts"][t]["completed"] for t in X])) for X in (J, C))
    return dict(turns=turns[J[0]], completed={t: h["counts"][t]["completed"] for t in J + C}, mean_J=cj, mean_C=cc, line=bool(cj >= cc - 2),
                paired=h["paired"], common_turns=h["common_turns"], logs=h["logs"])


# ---------------------------------------------------------------- closed-loop stage lines (CPU)
def scores(dirs):
    S = {}
    for D in dirs:
        f = Path(D) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                S.setdefault(r["clipgt_id"], float(r.get("score") or 0.0))
    return S


def tag_scores(stage, tag):
    if stage == "held":
        return scores(glob.glob(f"{DATA}/runs/route_mem/read/{tag}/c*/run"))
    if stage == "pub":
        return scores(glob.glob(f"{OUT}/read10/{tag}/b*"))
    if stage == "sub":
        return scores(glob.glob(f"{OUT}/sub/{tag}/run"))
    if stage in ("pai", "paisub"):
        return scores(glob.glob(f"{OUT}/{stage}/{tag}/runs/*/sim"))
    raise SystemExit(stage)


def pai_full(prefix):
    ids = {s.strip() for f in glob.glob(str(PAI_CH / "*.txt")) for s in open(f) if s.strip()}
    out = [s for s in ids if s.startswith(f"clipgt-{prefix}")]
    assert len(out) == 1, (prefix, out)
    return out[0]


def cmd_clgate(a):
    from jevdrive import stats
    J, C = a.j.split(","), a.c.split(",")
    st = a.stage
    res = dict(stage=st, J=J, C=C)
    if st in ("held", "pub"):
        SJ, SC = [tag_scores(st, t) for t in J], [tag_scores(st, t) for t in C]
        common = sorted(set.intersection(*map(set, SJ + SC)))
        z = lambda S: sum(S[s] == 0 for s in common)  # noqa: E731
        zj, zc = np.mean([z(s) for s in SJ]), np.mean([z(s) for s in SC])
        mj = np.array([np.mean([s[k] for s in SJ]) for k in common])
        mc = np.array([np.mean([s[k] for s in SC]) for k in common])
        pd_ = stats.paired(mj, mc, groups=np.array([s.split("_")[0] + "_" + s.split("_")[1] for s in common]))
        res |= dict(scenes=len(common), zeros_J=[z(s) for s in SJ], zeros_C=[z(s) for s in SC], zeros_J_mean=zj, zeros_C_mean=zc,
                    mean_J=float(mj.mean()), mean_C=float(mc.mean()), paired=stats.fmt(pd_), paired_lo=float(pd_["lo"]))
        if st == "held":
            res["passed"] = bool(zj <= zc + 3)
            res["static_scene"] = None
            if a.follow:                                              # Amendment C 6: the held turn completion line
                res["held_turns"] = held_turns(a.follow, J, C)
                res["zeros_line"] = res["passed"]
                res["passed"] = bool(res["passed"] and res["held_turns"]["line"])
        else:
            res["passed"] = bool(pd_["lo"] > -0.01 and zj <= zc + 5)
    elif st == "sub":                                            # stage 2: the static public scene (4 tags) + PAI stop17 (J s0) under the submission config
        res["static"] = {t: tag_scores("sub", t).get(STATIC_PUB) for t in J + C}
        sj = tag_scores("paisub", J[0])
        ref = scores(glob.glob(f"{PAI_REF}/runs/*/sim"))
        full = {p: pai_full(p) for p in STOP17}
        res["stop17"] = {p: dict(J=sj.get(full[p]), ref=ref.get(full[p])) for p in STOP17}
        zj = sum(v["J"] == 0 for v in res["stop17"].values() if v["J"] is not None)
        zr = sum(v["ref"] == 0 for v in res["stop17"].values() if v["ref"] is not None)
        res |= dict(stop17_zeros_J=zj, stop17_zeros_ref=zr, static_J_nonzero=sum((res["static"][t] or 0) > 0 for t in J))
        res["passed"] = bool(res["static_J_nonzero"] >= 1 or zj < zr)
    elif st == "pai":
        sj, ref = tag_scores("pai", J[0]), scores(glob.glob(f"{PAI_REF}/runs/*/sim"))
        common = sorted(set(sj) & set(ref))
        a_, b_ = np.array([sj[k] for k in common]), np.array([ref[k] for k in common])
        pd_ = stats.paired(a_, b_)
        full = {pai_full(p) for p in STOP17}
        res |= dict(scenes=len(common), mean_J=float(a_.mean()), mean_ref=float(b_.mean()), paired=stats.fmt(pd_), zeros_J=int((a_ == 0).sum()),
                    zeros_ref=int((b_ == 0).sum()), stop17_zeros_J=int(sum(sj.get(s, 1) == 0 for s in full)), stop17_zeros_ref=int(sum(ref.get(s, 1) == 0 for s in full)))
        res["passed"] = None                                       # final stage: reported, no further stage
    RES.mkdir(parents=True, exist_ok=True)
    for p in (OUT / f"cl_{st}.json", RES / f"cl_{st}.json"):
        p.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    nxt = {"held": "stage-sub", "sub": "stage-pub", "pub": "stage-pai"}.get(st)
    if a.submit and res["passed"] and nxt:
        subprocess.run(["bash", str(HERE / "wt.sh"), nxt, a.j, a.c], check=True)
    elif a.submit and nxt:
        (OUT / "STOPPED").write_text(f"stage {st} failed its line: no {nxt}\n")


# ---------------------------------------------------------------- yield census (navsim2, CPU; human-only, no model output)
def _ycount_job(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location("led", REPO / "experiments/cap_ledger/scripts/led.py")
    led = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(led)
    fr = pickle.load(open(p, "rb"))
    ok = [(NAV / "sensor_blobs/trainval" / f["cams"]["CAM_F0"]["data_path"]).exists() for f in fr]
    out = []
    for r in led.log_cells(p):
        c = led.cells_of(r)
        out.append(dict(log=r["log"], frame=r["frame"], cam=bool(all(ok[r["frame"] - 3:r["frame"] + 1])), v0=r["v0"],
                        **{k: bool(c[k]) for k in ("Y_P", "Y_V", "S_L", "S_R")}, wc=wc.cell_of(r) or ""))
    return out


def cmd_ycount(a):
    """Decision 331's yield cells (cap_ledger led.py cells_of, unchanged) on the WAIT-TRAIN logs: frames, 2 s-thinned rows, logs, and how many
    have their 4 history CAM_F0 on disk now -> runs/wait_cap/train/ycount.{json,md} (+ results/wait_train/)."""
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    want = [l for l in (REPO / "experiments/wait_cap/results/trainval_logs.txt").read_text().split() if l]
    dv = splits.load("navsim/op-parity-full-dev")
    devlogs = set()
    for i in range(12):
        z = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        devlogs |= set(z["log"][dv.mask(z["names"])].tolist())
    logs = [l for l in want if l not in devlogs]
    with Run("wait_cap", "train-ycount", config=vars(a)) as run:
        res = par.pmap(_ycount_job, [NAV / f"navsim_logs/trainval/{l}.pkl" for l in logs], run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        rows = [r for v in res.values for r in v]
        rec = dict(logs=len(logs), frames=len(rows), cam_frames=sum(r["cam"] for r in rows))
        md = [f"WAIT-TRAIN yield census (human-only; led.py cells_of on {len(logs)} trainval logs, {len(rows)} eligible frames, "
              f"{rec['cam_frames']} with history cameras on disk now)", "", "| cell | frames | rows (same log >= 2 s) | logs | rows with cameras now | frames also in a wc cell |",
              "|---|--:|--:|--:|--:|--:|"]
        for c in ("Y_P", "Y_V", "S_L", "S_R"):
            cand = sorted((r for r in rows if r[c]), key=lambda r: (r["log"], r["frame"]))
            thin, last = [], {}
            for r in cand:
                if r["frame"] - last.get(r["log"], -99) >= wc.GAP_FR:
                    thin.append(r)
                    last[r["log"]] = r["frame"]
            rec[c] = dict(frames=len(cand), rows=len(thin), logs=len({r["log"] for r in thin}), rows_cam=sum(r["cam"] for r in thin),
                          in_wc_cell={k: sum(r["wc"] == k for r in cand) for k in ("S_L", "S_N") if any(r["wc"] == k for r in cand)})
            md.append(f"| {c} | {len(cand)} | {len(thin)} | {rec[c]['logs']} | {rec[c]['rows_cam']} | {rec[c]['in_wc_cell']} |")
        yy = {(r["log"], r["frame"]) for r in rows if r["Y_P"] or r["Y_V"]}
        rec["Y_any_frames"] = len(yy)
        md.append(f"\nY_P or Y_V frames {len(yy)}")
        OUT.mkdir(parents=True, exist_ok=True)
        RES.mkdir(parents=True, exist_ok=True)
        for d in (OUT, RES):
            (d / "ycount.json").write_text(json.dumps(rec, indent=1))
            (d / "ycount.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch", required=True)
    for c in ("index", "teacher"):
        sp.add_parser(c).add_argument("--batch", required=True)
    sp.add_parser("pack").add_argument("--batches", required=True)
    p = sp.add_parser("bitcmp")
    p.add_argument("--a", default="WTGG-off"), p.add_argument("--b", default="WTGG-w0"), p.add_argument("--out", default="identity")
    p.add_argument("--note", default="")
    p = sp.add_parser("ycount")
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("devplans")
    p.add_argument("--tags", nargs="+", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--j", nargs="+", required=True)
    p.add_argument("--c", required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--dose-of", type=lambda s: dict(x.split("=") for x in s.split(",")), default={}, help="dose=tag,...")
    p = sp.add_parser("wdgate")
    p.add_argument("--j", nargs="+", required=True)
    p.add_argument("--c", required=True)
    p.add_argument("--lam-of", type=lambda s: dict(x.split("=") for x in s.split(",")), required=True, help="lam=tag,...")
    p.add_argument("--submit", action="store_true")
    p = sp.add_parser("clgate")
    p.add_argument("--stage", required=True, choices=("held", "sub", "pub", "pai"))
    p.add_argument("--j", required=True)
    p.add_argument("--c", required=True)
    p.add_argument("--follow", default="", help="Amendment C 6: dir holding rf_follow's follow/ of the held reads (held stage)")
    p.add_argument("--submit", action="store_true")
    sp.add_parser("devall").add_argument("--tags", nargs="+", required=True)
    for c in ("xheld", "xsum"):
        p = sp.add_parser(c)
        p.add_argument("--j", required=True)
        p.add_argument("--c", required=True)
        p.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    dict(select=cmd_select, index=cmd_index, teacher=cmd_teacher, pack=cmd_pack, bitcmp=cmd_bitcmp, devplans=cmd_devplans, gate=cmd_gate, ycount=cmd_ycount,
         clgate=cmd_clgate, wdgate=cmd_wdgate, devall=cmd_devall, xheld=cmd_xheld, xsum=cmd_xsum)[a.cmd](a)
