#!/usr/bin/env python3
"""SELF-ITER: open-loop hard-example mining with the human log as the label (plans/2026-10-17-self-iter-prereg.md). Box, repo root; chain in si.sh.
Rows, index, tokens, route memory and teacher reuse the WAIT-TRAIN tooling (experiments/wait_cap/scripts/wt.py index / teacher, pp_prep.py,
rm_build.py nav, rf_pait.py merge); the loss is the RSG-L18 recipe's, the draw is lib/self_iter.py (rm_train --si-scan).

  select [--workers N] [--batch-rows R]  (op-train, CPU) every NAVSIM trainval log: d331's per-frame row (cap_ledger led.log_cells, cells_of,
                     wait_cap wc.cell_of) -> runs/self_iter/cells.npz (all frames: navtrain tokens included, the class labels of the scan);
                     new rows = frames whose 4 history CAM_F0 are on disk, token not in navtrain, log neither a navtrain dev log nor an evaluation
                     log (wt.leak: held 460, public 1491, PAI 281, waitcap_test; no log of navsim_logs/test); every frame (2 Hz), no thinning;
                     hold = crc32(log) % 10 == 0 (held-out trainval logs: the gate's unseen rows). Logs already in a batch are skipped, so
                     a rerun after more archives adds batches; new logs -> batches of <= R rows (by log) -> batches/<B>/rows.npz,
                     split navsim/selfiter_tr<B>
  index --batch B    (navsim2) wt.py index for the batch; teacher --batch B (GPU) wt.py teacher
  pack --batches A,B.. [--name SITV --merged L18RBSI]   pool on1_<name> (mode both: imitation of the log only, agent / road / route labels ok
                     False; tab `hold`, `cls`, `cell`) -> merged on1_<merged> = on1_L18RB0 (s, self) ++ on1_<name> (t, both) (rf_pait merge);
                     the pool's own front copy is removed after the merge; splits navsim/selfiter_tv<suffix> / selfiter_tvhold<suffix>; pack.json
                     with the per-batch slot count k_tv (new rows in a 128 batch = 102 x n_tv / (n_nav + n_tv), at most 51) -> runs/self_iter/flags_<pool>.txt
  scan --policy T [--pool on1_..] [--sets all|dev] [--name N]   (GPU) T's plan on every row of the sets (all: navtrain train + dev, trainval
                     train + hold; dev: navtrain dev + trainval hold) -> runs/self_iter/scan/<N or T>[.dev].npz (Store names, set, log, v0,
                     class, lib/self_iter.errors); the navtrain dev ADE is checked against T's DONE dev_ade
  report --scan F    (CPU) error distribution per set, class / d331-cell / speed composition of the top decile -> results/scan_<stem>.{json,md}
  gate --j T.. --c C [--pol RSG-L18-s0] --name N   (CPU) the registered open-loop lines on the dev scans -> results/gate_<N>.{json,md},
                     runs/self_iter/pass_<N>.txt (passing arms, best top-decile drop first)
  bitcmp --a T --b T --out N   wt.py bitcmp (scalars, weights, on_hash_60 = RSG-L18-s0's) -> results/<N>.json
  clsum --j T        nx.py cont + wide on runs/self_iter/cl/<T> -> results/cl_<T>.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/wait_cap/scripts")]
import self_iter as SI  # noqa: E402
import wc  # noqa: E402
import wt  # noqa: E402

DATA, NAV, CR, FEAT = wt.DATA, wt.NAV, wt.CR, wt.FEAT
OUT = DATA / "runs/self_iter"
RES = REPO / "experiments/self_iter/results"
BASE, NAV_TRAIN, NN0 = "on1_L18RB0", 101499, 102            # RSG-L18's pool; its normal train rows and normal slots per batch (trainer log)
SETS = ("nav_tr", "nav_dv", "tv_tr", "tv_hold")


def bsplit(b):
    return f"selfiter_tr{b}"


wt.OUT, wt.bsplit = OUT, bsplit                                       # wt.py index / teacher on this topic's batches


def held_out(log):
    return zlib.crc32(log.encode()) % 10 == 0


# ---------------------------------------------------------------- select (CPU)
_LED = None


def _log_job(p):
    global _LED
    if _LED is None:
        _LED = wt._led()
    fr = pickle.load(open(p, "rb"))
    ok = [(NAV / "sensor_blobs/trainval" / f["cams"]["CAM_F0"]["data_path"]).exists() for f in fr]
    out = []
    for r in _LED.log_cells(p):
        i = r["frame"]
        lc = _LED.cells_of(r)
        out.append(dict(token=r["token"], log=r["log"], frame=i, v0=r["v0"], a4=r["a4"], vend=r["vend"], dpsi4=r["dpsi4"], lead_gap=r["lead_gap"],
                        lead_any=r["lead_any"], cell=wc.cell_of(r) or "", bits=sum(1 << j for j, k in enumerate(_LED.CELLS) if lc[k]),
                        cam=bool(all(ok[i - 3:i + 1])), fut=r["fut"]))
    return out, sum(ok), len(fr)


def nav_tokens():
    """navtrain tokens (every navtrain_full cache row, the navtrain split) and the logs holding a navtrain dev token (wt.used_logs' rule)."""
    from jevdrive.data import splits
    dv = splits.load("navsim/op-parity-full-dev")
    toks, devlogs = set(splits.load("navsim/navtrain").members), set()
    for i in range(12):
        z = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        toks |= set(z["names"].astype(str).tolist())
        devlogs |= set(z["log"][dv.mask(z["names"])].astype(str).tolist())
    return toks, devlogs


def cmd_select(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    paths = sorted((NAV / "navsim_logs/trainval").glob("*.pkl"))
    test = {p.stem for p in (NAV / "navsim_logs/test").glob("*.pkl")}
    taken = {l for f in (OUT / "batches").glob("*/logs.txt") for l in f.read_text().split()}
    with Run("self_iter", "select", config=vars(a)) as run:
        res = par.pmap(_log_job, paths, run=run, workers=a.workers, desc="logs")
        failed = [str(paths[i].stem) for i in res.errors]
        vals = [v for v in res.values if v is not None]
        rows = [r for v, _, _ in vals for r in v]
        cols = [k for k in rows[0] if k != "fut"]
        C = {k: np.array([r[k] for r in rows]) for k in cols}
        fut = np.stack([r["fut"] for r in rows]).astype(np.float32)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "cells.npz", cells=np.array(wt._led().CELLS), **C)
        toks, devlogs = nav_tokens()
        log = C["log"].astype(str)
        tok = C["token"].astype(str)
        new = C["cam"] & ~np.isin(tok, list(toks)) & ~np.isin(log, sorted(devlogs | test)) & ~np.isin(log, sorted(taken))
        logs = sorted(set(log[new].tolist()))
        lk = wt.leak(logs)
        assert lk["clean"] and not set(logs) & test, "a new-row log is an evaluation log"
        hold = np.array([held_out(l) for l in log])
        rec = dict(logs_total=len(paths), logs_failed=failed, frames=len(rows), cam_frames=int(C["cam"].sum()), frames_on_disk=sum(c for _, c, _ in vals),
                   navtrain_tokens_labelled=int(np.isin(tok, list(toks)).sum()), new_rows=int(new.sum()), new_logs=len(logs),
                   new_hold_rows=int((new & hold).sum()), new_hold_logs=len({l for l in logs if held_out(l)}), logs_already_batched=len(taken),
                   dev_logs_excluded=len(devlogs), leak=lk, batches={})
        if new.any():
            prev = sorted(p.name for p in (OUT / "batches").glob("*")) if (OUT / "batches").exists() else []
            nxt = chr(ord(prev[-1]) + 1) if prev else "A"
            per = {l: int(n) for l, n in zip(*np.unique(log[new], return_counts=True))}
            groups, cur, n = [], [], 0
            for l in logs:
                if cur and n + per[l] > a.batch_rows:
                    groups.append(cur)
                    cur, n = [], 0
                cur.append(l)
                n += per[l]
            groups.append(cur)
            for g in groups:
                B, nxt = nxt, chr(ord(nxt) + 1)
                m = new & np.isin(log, g)
                o = np.argsort([f"{l}:{f:06d}" for l, f in zip(log[m], C["frame"][m])])
                d = OUT / "batches" / B
                d.mkdir(parents=True)
                np.savez(d / "rows.npz", fut=fut[m][o], hold=hold[m][o], **{k: C[k][m][o] for k in cols})
                (d / "logs.txt").write_text("\n".join(g) + "\n")
                splits.define("navsim", bsplit(B), sorted(tok[m].tolist()), unit="token",
                              origin=f"SELF-ITER select batch {B} (experiments/self_iter/scripts/si.py): every frame of NAVSIM trainval logs with the 4 history "
                                     f"CAM_F0 on disk, not a navtrain token, not a navtrain dev log, no evaluation log; rows $DATA_DIR/runs/self_iter/batches/{B}/rows.npz",
                              status="frozen", used_by=["experiments/self_iter"], notes="training row candidates (hold = crc32(log) % 10 == 0 rows are never trained)")
                rec["batches"][B] = dict(logs=len(g), rows=int(m.sum()), hold=int((m & hold).sum()))
        (OUT / "select.json").write_text(json.dumps(rec, indent=1))
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "select.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "leak"} | dict(leak_clean=lk["clean"]), indent=1))
        run.summary |= {k: v for k, v in rec.items() if k != "leak"}
        print(json.dumps({k: v for k, v in rec.items() if k != "leak"}, indent=1))


def cmd_index(a):
    wt.cmd_index(a)


def cmd_teacher(a):
    wt.cmd_teacher(a)


# ---------------------------------------------------------------- pack (CPU)
def cmd_pack(a):
    from jevdrive.data import splits
    name, merged = f"on1_{a.name}", f"on1_{a.merged}"
    assert not (CR / merged / "tab.npz").exists(), f"{merged} exists"
    bs = a.batches.split(",")
    parts = []
    for b in bs:
        d = f"idx_{bsplit(b)}"
        tb = dict(np.load(CR / d / "tab.npz"))
        sl = np.load(OUT / "batches" / b / "rows.npz")
        pos = {t: i for i, t in enumerate(sl["token"].astype(str).tolist())}
        j = np.array([pos[t] for t in tb["names"].astype(str).tolist()])
        parts.append((d, tb, {k: sl[k][j] for k in ("hold", "cell", "bits")}))
    cat = lambda k: np.concatenate([p[1][k] for p in parts])  # noqa: E731
    tb = {k: cat(k) for k in ("names", "log", "ego", "pose", "vel", "acc", "cmd", "fut", "cam", "lht", "speed")}
    hold = np.concatenate([p[2]["hold"] for p in parts]).astype(bool)
    cell = np.concatenate([p[2]["cell"] for p in parts]).astype("U4")
    n = len(tb["names"])
    assert np.isfinite(tb["fut"]).all(), "rows without a full future"
    pn = np.array([f"on:{t}" for t in tb["names"].astype(str).tolist()])
    src, dw = CR / name, CR / f"{name}@warp"
    src.mkdir(exist_ok=True), dw.mkdir(exist_ok=True)
    cls = SI.classes(tb["fut"], tb["speed"])
    T = {k: v for k, v in tb.items() if k != "names"} | dict(names=pn, src=np.full(n, "selfiter", "U16"), hold=np.where(hold, "1", "0").astype("U1"),
                                                          cls=cls.astype("U8"), cell=cell)
    np.savez(src / "tab.npz", **T)
    Ba = np.load(CR / BASE / "agent.npz", mmap_mode="r")
    np.savez(src / "agent.npz", tokens=pn, ok=np.zeros(n, bool), box=np.zeros((n,) + Ba["box"].shape[1:], np.float32),
             valid=np.zeros((n,) + Ba["valid"].shape[1:], bool), cls=np.zeros((n,) + Ba["cls"].shape[1:], np.int8))
    Br = np.load(CR / BASE / "road.npz", mmap_mode="r")
    np.savez(src / "road.npz", tokens=pn, ok=np.zeros(n, bool), sdf=np.zeros((n,) + Br["sdf"].shape[1:], np.float16), x0=Br["x0"], y0=Br["y0"], res=Br["res"])
    Bq = np.load(CR / BASE / "route.npz", mmap_mode="r")
    np.savez(src / "route.npz", tokens=pn, ok=np.zeros(n, bool), poly=np.zeros((n,) + Bq["poly"].shape[1:], np.float32),
             seglen=np.zeros((n,) + Bq["seglen"].shape[1:], np.float32), dz=np.zeros(n, np.float32), cov20=np.zeros(n, bool))
    FZ = [np.load(FEAT / f"{d}.npz") for d, t_, _ in parts]
    for (d, t_, _), z in zip(parts, FZ):
        assert np.array_equal(z["names"], t_["names"]), f"{d}: route feature rows != cache rows"
    fok = np.concatenate([z["ok"] for z in FZ])
    np.savez(FEAT / f"{name}.npz", names=pn, f=np.concatenate([z["f"] for z in FZ]).astype(np.float16), k=np.concatenate([z["k"] for z in FZ]).astype(np.int64), ok=fok)
    TH = [np.load(CR / f"{d}@warp/teacher.npz") for d, _, _ in parts]
    assert all(np.array_equal(t["di"], TH[0]["di"]) and np.array_equal(t["pi"], TH[0]["pi"]) for t in TH)
    np.savez(dw / "teacher.npz", out=np.concatenate([t["out"] for t in TH]), plan=np.concatenate([t["plan"] for t in TH]), di=TH[0]["di"], pi=TH[0]["pi"])
    FR = [np.load(CR / f"{d}@warp/front.npy", mmap_mode="r") for d, _, _ in parts]
    o = np.lib.format.open_memmap(dw / "front.npy", "w+", np.float16, (n,) + FR[0].shape[1:])
    i0 = 0
    for f_ in FR:
        for i in range(0, len(f_), 4096):
            o[i0 + i:i0 + min(i + 4096, len(f_))] = f_[i:i + 4096]
        i0 += len(f_)
    o.flush()
    del o
    p = subprocess.run([sys.executable, str(REPO / "experiments/route_follow/scripts/rf_pait.py"), "merge", "--pools", f"{BASE}:s:self", f"{name}:t:both",
                        "--out", a.merged], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"rf_pait merge failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    M = np.load(CR / merged / "tab.npz")
    B = np.load(CR / BASE / "tab.npz")
    ns = len(B["names"])
    chk = dict(base_rows=ns, new_rows=n, merged=len(M["names"]), base_first=bool((M["pool"][:ns] == "s").all() and (M["pool"][ns:] == "t").all()),
               modes=dict(zip(*[x.tolist() for x in np.unique(M["mode"], return_counts=True)])), base_ego_equal=bool(np.array_equal(M["ego"][:ns], B["ego"])),
               base_fut_equal=bool(np.array_equal(M["fut"][:ns], B["fut"])), hold_rows=int((M["hold"] == "1").sum()))
    Fm, Fb = np.load(CR / f"{merged}@warp/front.npy", mmap_mode="r"), np.load(CR / f"{BASE}@warp/front.npy", mmap_mode="r")
    rs = np.random.default_rng(0).choice(ns, 64, replace=False)
    chk["base_front_equal_64"] = bool(all(np.array_equal(Fm[i], Fb[i]) for i in rs))
    chk["new_front_equal_64"] = bool(all(np.array_equal(Fm[ns + i], FR[0][i]) for i in rs % len(FR[0])))
    assert chk["base_first"] and chk["base_ego_equal"] and chk["base_fut_equal"] and chk["base_front_equal_64"] and chk["new_front_equal_64"], chk
    (dw / "front.npy").unlink()                                       # the merged pool holds the copy (disk)
    sfx = a.name[4:] if a.name.startswith("SITV") else a.name
    for nm_, m_ in ((f"selfiter_tv{sfx}", ~hold), (f"selfiter_tvhold{sfx}", hold)):
        splits.define("navsim", nm_, sorted(tb["names"][m_].astype(str).tolist()), unit="token",
                      origin=f"SELF-ITER pack {a.name} (experiments/self_iter/scripts/si.py): trainval rows of batches {a.batches}" + (" (held-out logs)" if nm_.startswith("selfiter_tvhold") else ""),
                      status="frozen", used_by=["experiments/self_iter"], notes="held-out gate rows, never trained" if "hold" in nm_ else "training rows (imitation of the log)")
    n_tv = int((~hold).sum())
    k_tv = min(int(round(NN0 * n_tv / (NAV_TRAIN + n_tv))), NN0 // 2)   # proportional, capped: navtrain keeps >= half the normal slots
    flags = f"--on {merged} --on-k {13 + k_tv} --on-mode mixed --si-k-s 13 --si-split navsim/selfiter_tv{sfx}"
    (OUT / f"flags_{a.merged}.txt").write_text(flags + "\n")
    (OUT / f"flags_{a.merged}_w0.txt").write_text(f"--on {merged} --on-k 13 --on-mode mixed --on-weight pool=t:0\n")
    rec = dict(batches=bs, pool=name, merged=merged, rows=n, train_rows=n_tv, hold_rows=int(hold.sum()), logs=len(set(tb["log"].tolist())),
               route_feat_ok=float(fok.mean()), classes=dict(zip(*[x.tolist() for x in np.unique(cls[~hold], return_counts=True)])),
               k_tv=k_tv, normal_slots=NN0 - k_tv, flags=flags, check=chk)
    (OUT / f"pack_{a.merged}.json").write_text(json.dumps(rec, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"pack_{a.merged}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- scan (GPU)
def cmd_scan(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from types import SimpleNamespace
    sys.path[:0] = [str(REPO / "experiments/route_mem/scripts"), str(REPO / "experiments/route_follow/scripts")]
    import rm_train as RT
    import pp_train as T
    from rf_lonread import REC
    from jevdrive.run import Run
    dev = torch.device("cuda")
    pol = a.policy
    ns, _, _, _, _ = RT.build(SimpleNamespace(**(REC | dict(data=RT.OT.JT.FULL, on_mode="mixed")), road_labels=RT.OT.JT.ROAD, on=a.pool, init="",
                                              rm_from=pol, tag=pol, exp="self_iter", lr_scale=1.0, prefetch=1, dev_card_gb=4.0), dev)
    S, model, W = ns.S, ns.model, ns.losses(ns.mode).W
    pi = torch.as_tensor(S.pi, device=dev)
    tb = np.load(CR / a.pool / "tab.npz")
    pool = tb["pool"].astype(str) if "pool" in tb.files else np.full(len(tb["names"]), "s")
    hold = tb["hold"].astype(str) == "1" if "hold" in tb.files else np.zeros(len(pool), bool)
    has = S.has_fut.cpu().numpy().astype(bool)
    R = dict(nav_tr=ns.tr_o, nav_dv=ns.dv_o[has[ns.dv_o]], tv_tr=ns.on0 + np.flatnonzero((pool == "t") & ~hold), tv_hold=ns.on0 + np.flatnonzero((pool == "t") & hold))
    want = SETS if a.sets == "all" else ("nav_dv", "tv_hold")
    rows = np.concatenate([np.sort(R[k]) for k in want])
    sets = np.concatenate([np.full(len(R[k]), k) for k in want])
    stem = (a.name or pol) + ("" if a.sets == "all" else ".dev")
    with Run("self_iter", f"scan-{stem}", config=vars(a)) as run:
        model.eval()
        t0 = time.time()
        P = np.zeros((len(rows), 8, 3), np.float32)
        with torch.no_grad():
            for i in range(0, len(rows), a.bs):
                r = rows[i:i + a.bs]
                rt = torch.as_tensor(r, device=dev)
                p = model(S.front[r], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
                P[i:i + len(r)] = torch.stack(T.rear(p, S.cam_x[rt], W), -1).cpu().numpy()
                if i // a.bs % 50 == 0:
                    run.status(f"{i + len(r)} / {len(rows)} rows")
        G = S.fut[torch.as_tensor(rows, device=dev)].cpu().numpy()
        v0 = np.asarray(S.tab["speed"])[rows]
        E = SI.errors(P, G)
        dv = sets == "nav_dv"
        try:
            ref = wt.dev_ade(pol)
        except Exception:                                             # noqa: BLE001  (no DONE: no reference)
            ref = None
        rec = dict(policy=pol, pool=a.pool, rows={k: int((sets == k).sum()) for k in want}, ade={k: float(E["ade"][sets == k].mean()) if (sets == k).any() else None for k in want},
                   forward_s=time.time() - t0,
                   nav_dev_ade=float(E["ade"][dv].mean()), done_dev_ade=ref)
        run.info(f"scan {pol}: {json.dumps(rec)}")
        if ref is not None and abs(rec["nav_dev_ade"] - ref) > 1e-3:
            run.info(f"WARNING: navtrain dev ADE {rec['nav_dev_ade']:.4f} != DONE {ref:.4f}")
        (OUT / "scan").mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "scan" / f"{stem}.npz", names=S.tab["names"][rows], set=sets.astype("U8"), log=S.tab["log"][rows], v0=v0.astype(np.float32),
                 cls=SI.classes(G, v0), policy=np.array(pol), plan4=P[:, 7], **E)
        run.summary |= rec


# ---------------------------------------------------------------- report (CPU)
def load_scan(p):
    z = np.load(p if str(p).startswith("/") else OUT / "scan" / f"{p}.npz")
    return {k: z[k] for k in z.files}


def tok_of(names):
    return np.array([x.split(":")[-1] for x in names.astype(str).tolist()])


def d331(names):
    """d331 cell names per row (join on cells.npz by token): the first set bit's cell in led.CELLS order, '' if none / unlabelled."""
    C = np.load(OUT / "cells.npz")
    pos = {t: i for i, t in enumerate(C["token"].astype(str).tolist())}
    cn = C["cells"].astype(str).tolist()
    j = np.array([pos.get(t, -1) for t in tok_of(names).tolist()])
    bits = np.where(j >= 0, C["bits"][np.maximum(j, 0)], 0)
    wcc = np.where(j >= 0, C["cell"][np.maximum(j, 0)].astype(str), "")
    lab = np.array([next((cn[k] for k in range(len(cn)) if b >> k & 1), "") for b in bits.tolist()])
    lab = np.where(np.isin(wcc, ("W_L", "W_N", "G_L", "G_N")), wcc, lab)       # d326's W / G cells first
    return lab, j >= 0, np.where(j >= 0, C["lead_any"][np.maximum(j, 0)], np.nan)


def shares(lab, m):
    u, c = np.unique(lab[m], return_counts=True)
    return {k: float(v / m.sum()) for k, v in sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1])}


def cmd_report(a):
    z = load_scan(a.scan)
    stem = Path(a.scan).stem
    lab, labelled, _ = d331(z["names"])
    res, L = dict(policy=str(z["policy"]), sets={}), [f"# Error scan of {z['policy']} ({stem})", ""]
    vb = np.digitize(z["v0"], [0.5, 3, 8, 15])
    vnames = np.array(["<0.5", "0.5-3", "3-8", "8-15", ">=15"])[vb]
    for s in np.unique(z["set"]).tolist():
        m = z["set"] == s
        e = z["ade"][m]
        q90 = np.quantile(e, 0.9)
        top = m & (z["ade"] >= q90)
        r = dict(rows=int(m.sum()), ade_mean=float(e.mean()), ade_q=dict(zip(("p50", "p75", "p90", "p95", "p99"), np.quantile(e, [0.5, 0.75, 0.9, 0.95, 0.99]).round(3).tolist())),
                 top_share_of_error=float(z["ade"][top].sum() / e.sum()), cls_all=shares(z["cls"], m), cls_top=shares(z["cls"], top),
                 d331_labelled=float(labelled[m].mean()), d331_all=shares(lab, m & labelled), d331_top=shares(lab, top & labelled),
                 v0_all=shares(vnames, m), v0_top=shares(vnames, top),
                 by_cls={c: dict(rows=int((m & (z["cls"] == c)).sum()), ade=float(z["ade"][m & (z["cls"] == c)].mean()),
                                 arc4=float(z["arc4"][m & (z["cls"] == c)].mean()), cr2_abs=float(np.abs(z["cr2"][m & (z["cls"] == c)]).mean()),
                                 lift=float((top & (z["cls"] == c)).sum() / max(top.sum(), 1) / max((m & (z["cls"] == c)).mean() / m.mean(), 1e-9)))
                         for c in SI.CLASSES if (m & (z["cls"] == c)).any()},
                 top_err=dict(al4_abs=float(np.abs(z["al4"][top]).mean()), cr4_abs=float(np.abs(z["cr4"][top]).mean()), arc4=float(z["arc4"][top].mean())))
        res["sets"][s] = r
        L += [f"## {s}: {r['rows']} rows, ADE mean {r['ade_mean']:.3f} m, quantiles {r['ade_q']}; the top decile carries {r['top_share_of_error']:.2f} of the error sum", "",
              "| class | rows | share | share in top 10 % | lift | ADE | 4 s arc bias | |cross 2 s| |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for c, v in r["by_cls"].items():
            L.append(f"| {c} | {v['rows']} | {r['cls_all'].get(c, 0):.3f} | {r['cls_top'].get(c, 0):.3f} | {v['lift']:.2f} | {v['ade']:.3f} | {v['arc4']:+.3f} | {v['cr2_abs']:.3f} |")
        L += ["", f"top decile error split: |along 4 s| {r['top_err']['al4_abs']:.2f} m, |cross 4 s| {r['top_err']['cr4_abs']:.2f} m, 4 s arc bias {r['top_err']['arc4']:+.2f} m",
              f"v0 bins, all: {json.dumps({k: round(v, 3) for k, v in r['v0_all'].items()})}; top decile: {json.dumps({k: round(v, 3) for k, v in r['v0_top'].items()})}",
              f"d331 cells (labelled {r['d331_labelled']:.2f} of the rows; '' = no cell), all: {json.dumps({k or '-': round(v, 3) for k, v in list(r['d331_all'].items())[:10]})}",
              f"top decile: {json.dumps({k or '-': round(v, 3) for k, v in list(r['d331_top'].items())[:10]})}", ""]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"scan_{stem}.json").write_text(json.dumps(res, indent=1))
    (RES / f"scan_{stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- gate (CPU)
TOP_DROP, ADE_TOL, FREE_TOL, LAT_TOL, START_TOL = 0.03, 0.01, 0.10, 0.05, 0.02


def arm_reads(z, top, free, start):
    nd = z["set"] == "nav_dv"
    return dict(dev_ade=float(z["ade"][nd].mean()), top_tv=float(z["ade"][top].mean()), free_arc4=float(z["arc4"][free].mean()),
                lat2_med=float(np.median(np.abs(z["cr2"]))), start_arc4=float(z["arc4"][start].mean()))


def cmd_gate(a):
    P = load_scan(a.pol)
    Zs = {t: load_scan(f"{t}{a.sfx}.dev") for t in a.j + [a.c]}
    names = Zs[a.c]["names"].astype(str)
    for t, z in Zs.items():
        assert np.array_equal(z["names"].astype(str), names), f"{t}: dev scan rows differ from {a.c}'s"
    kp = {n: i for i, n in enumerate(P["names"].astype(str).tolist())}
    pi = np.array([kp.get(x, -1) for x in names.tolist()])
    assert (pi >= 0).all(), f"{a.pol}: does not cover the dev rows"
    z0 = Zs[a.c]
    tvh = z0["set"] == "tv_hold"
    pe = P["ade"][pi]
    top = tvh & (pe >= np.quantile(pe[tvh], 0.9))                    # the policy's top decile of the held-out trainval rows
    lab, _, _ = d331(names)
    free = lab == "C"                                                 # d331's free-road cruise cell
    start = (z0["v0"] < 0.5) & (z0["cls"] == "starting")              # the human launches from standstill
    logs = z0["log"].astype(str)
    rng = np.random.default_rng(0)
    ul, li = np.unique(logs[top], return_inverse=True)
    out = dict(control=a.c, policy=a.pol, rows=dict(dev=int((z0["set"] == "nav_dv").sum()), tv_hold=int(tvh.sum()), top=int(top.sum()), free=int(free.sum()),
                                                    start=int(start.sum())), arms={})
    rc = arm_reads(z0, top, free, start)
    out["C"] = rc
    for t in a.j:
        z = Zs[t]
        r = arm_reads(z, top, free, start)
        d = z["ade"][top] - z0["ade"][top]
        sb, nb = np.bincount(li, d, len(ul)), np.bincount(li, minlength=len(ul))
        bs = []
        for _ in range(2000):
            c = np.bincount(rng.integers(0, len(ul), len(ul)), minlength=len(ul))
            bs.append((c * sb).sum() / max((c * nb).sum(), 1))
        r["top_drop_rel"] = float(1 - r["top_tv"] / rc["top_tv"])
        r["top_diff_ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        lines = dict(dev_ade=r["dev_ade"] <= rc["dev_ade"] + ADE_TOL, top_decile=r["top_drop_rel"] >= TOP_DROP,
                     free_arc4=abs(r["free_arc4"] - rc["free_arc4"]) <= FREE_TOL, lateral=r["lat2_med"] - rc["lat2_med"] <= LAT_TOL,
                     start_arc4=abs(r["start_arc4"]) <= abs(rc["start_arc4"]) + START_TOL)
        r["lines"] = {k: bool(v) for k, v in lines.items()}
        r["pass"] = bool(all(lines.values()))
        out["arms"][t] = r
    passing = sorted((t for t, r in out["arms"].items() if r["pass"]), key=lambda t: -out["arms"][t]["top_drop_rel"])
    out["passing"] = passing
    L = [f"# SELF-ITER gate {a.name}: arms vs {a.c} (top decile by {a.pol} on the held-out trainval rows)", "",
         f"rows: {json.dumps(out['rows'])}", "",
         f"| arm | dev ADE (<= C + {ADE_TOL}) | top-decile ADE (drop >= {TOP_DROP:.0%}) | diff CI | free-road 4 s arc (C +- {FREE_TOL}) | median |cross 2 s| "
         f"(<= C + {LAT_TOL}) | standstill-launch 4 s arc (|.| <= |C| + {START_TOL}) | PASS |", "|---|--:|--:|--:|--:|--:|--:|:-:|",
         f"| C {a.c} | {rc['dev_ade']:.4f} | {rc['top_tv']:.3f} | - | {rc['free_arc4']:+.3f} | {rc['lat2_med']:.3f} | {rc['start_arc4']:+.3f} | - |"]
    for t, r in out["arms"].items():
        f = lambda k: "" if r["lines"][k] else " x"  # noqa: E731
        L.append(f"| {t} | {r['dev_ade']:.4f}{f('dev_ade')} | {r['top_tv']:.3f} ({r['top_drop_rel']:+.1%}){f('top_decile')} | [{r['top_diff_ci'][0]:+.3f}, {r['top_diff_ci'][1]:+.3f}] | "
                 f"{r['free_arc4']:+.3f}{f('free_arc4')} | {r['lat2_med']:.3f}{f('lateral')} | {r['start_arc4']:+.3f}{f('start_arc4')} | {'yes' if r['pass'] else 'no'} |")
    L += ["", f"passing (best top-decile drop first): {passing or 'none'}"]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"gate_{a.name}.json").write_text(json.dumps(out, indent=1))
    (RES / f"gate_{a.name}.md").write_text("\n".join(L) + "\n")
    (OUT / f"pass_{a.name}.txt").write_text(" ".join(passing) + "\n")
    print("\n".join(L))


def cmd_bitcmp(a):
    wt.RES = RES
    wt.cmd_bitcmp(a)


def cmd_clsum(a):
    D = OUT / "cl" / a.j
    for c, extra in (("cont", []), ("wide", ["--no-pai"])):
        o = f"si_{c}_{a.j}"
        subprocess.run([sys.executable, str(REPO / "experiments/wait_cap/scripts/nx.py"), c, "--j", a.j, "--dir", str(D), "--out", o] + extra, check=True)
        for s in (".json", ".md"):
            shutil.copy(REPO / "experiments/wait_cap/results/next" / f"{o}{s}", RES / f"cl_{c}_{a.j}{s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--workers", type=int, default=48), p.add_argument("--batch-rows", type=int, default=45000)
    for c in ("index", "teacher"):
        sp.add_parser(c).add_argument("--batch", required=True)
    p = sp.add_parser("pack")
    p.add_argument("--batches", required=True), p.add_argument("--name", default="SITV"), p.add_argument("--merged", default="L18RBSI")
    p = sp.add_parser("scan")
    p.add_argument("--policy", default="RSG-L18-s0"), p.add_argument("--pool", default="on1_L18RBSI"), p.add_argument("--sets", default="all", choices=("all", "dev"))
    p.add_argument("--name", default=""), p.add_argument("--bs", type=int, default=512)
    sp.add_parser("report").add_argument("--scan", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--j", nargs="+", required=True), p.add_argument("--c", required=True), p.add_argument("--pol", default="RSG-L18-s0")
    p.add_argument("--name", required=True), p.add_argument("--sfx", default="", help="the dev scans' name suffix (si.sh SFX)")
    p = sp.add_parser("bitcmp")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", required=True), p.add_argument("--note", default="")
    sp.add_parser("clsum").add_argument("--j", required=True)
    a = ap.parse_args()
    dict(select=cmd_select, index=cmd_index, teacher=cmd_teacher, pack=cmd_pack, scan=cmd_scan, report=cmd_report, gate=cmd_gate, bitcmp=cmd_bitcmp,
         clsum=cmd_clsum)[a.cmd](a)
