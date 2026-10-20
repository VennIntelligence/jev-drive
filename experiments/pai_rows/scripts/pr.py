#!/usr/bin/env python3
"""PAI-ROWS (plans/2026-10-20-pai-rows-prereg.md): PAI human-state imitation rows on every training-eligible NuRec scene on the box, their
frame-stretched variants, the scene table / halves, and the gates. Lane script: scripts/pr.sh.

  scenes   (Mac, repo root) results/usdz_<date>.txt (box `stat -c "%n %s"` of datasets/nurec/all-usdzs) + pai_av nurec_select.csv + the exam
           exclusions -> results/scenes.tsv (scene_id, cls, d402, half, held, uuid, size_gb) and results/scenes.md; --register defines the
           splits physical_ai_av/pai-rows@v1 (all) and physical_ai_av/pai-rows-half-a@v1 (half A)
  scenes-n (Mac, repo root) the second batch (results/download_queue_v2.csv: 64 NuRec scenes, each on the same drive as a PAI test281 scene)
           -> results/scenes_n.tsv (scenes.tsv's columns + t281, t281_gap_s, t281_why; --partner FILE adds the test281 partner) and, with
           --register, the splits physical_ai_av/pai-rows-n@v1 (the 64) and physical_ai_av/pai-rows-x@v1 (pai-rows@v1 + the 64)
  partner  --scenes TSV --meta DIR --out TSV   (box, CPU) the test281 scene each new scene shares a drive with and the clock gap: the rule of
           pai_av's nurecsel (pose_record epoch or log-session clock within 600 s and the start-to-start distance drivable in that time)
  tag      --pool P [--scenes TSV]   (CPU) tab.npz of on1_<P> gets t281 (U8: first 8 characters of the partner test281 scene's id after
           "clipgt-", "" = none) per row, keyed by the row's scene (`log`); stretch pools inherit it from their plain pool
  xfix     --pool P --ref R --parts a,b  (CPU) after a zero-copy merge: the first rows of P equal R (every file), agent ok of those rows from R,
           ok False on the rows of the --parts (the PAI human rows are imitation rows, as round2's r2.py fix)
  mcheck   --pool P --ref R [--skip-names]   (CPU) the first rows of P equal R (tab, agent / road / route, feat, teacher, token sample)
  check    --root R --n N --min-ok K [--log] --out FILE   (box) roll stage check: >= K of the N listed scenes scored without an RPC error,
           frame dumps and rollout.asl present; --log: every drive record from k = 20 on returned the log with log_poses (pai_gap.sh's gate)
  stretch  --pool P --rows DIR --f F --name OUT [--limit n]   (GPU pool job) the frame-stretched variant of the human-state pool on1_<P>:
           per row the 8 slots re-picked from the 10 Hz rendered frames at t0 + F x (-1.4 .. 0) s (exact on 10 Hz for F = 0.5 / 1.0 / 1.5),
           the ego row from the history at t0 + F x (-1.5, -1.0, -0.5) s (the driven anchors, linear in time between the 10 Hz decisions;
           F = 1: the stored history), velocity x F, acceleration x F^2; tokens and teacher by onp_cache.encode; the label dilated the same way
           (supplement 2: the log's pose at t0 + F x 0.5 (j + 1) s; rows without that much log dropped); route memory and route labels of the
           plain row unchanged, actor boxes kept but ok False for F != 1 (they are on the undilated clock) -> on1_<OUT>, @warp, feat/on1_<OUT>.npz
  gate1    --plain P --f1 P10 --out FILE   (CPU) F = 1 against the plain pool, row by row: ego, fut, front tokens, teacher bit-equal
  feas     --pools P:f.. --out DIR   (CPU) the unchanged label against the fed speed: mean acceleration needed over 0 .. 0.5 s
           (2 (|fut_0| - f v0 0.5) / 0.25) and 0 .. 1 s, shares beyond 3 / 6 / 9 m/s^2, by class and speed
  report   --pools P.. --out DIR   (CPU) rows per pool x class x half (held = crc32(scene) % 100 < 15 counted apart), disk
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/onpol_pai/scripts"), str(R_ / "experiments/onpol_pai/lib"), str(R_ / "experiments/defect_fix/scripts")]
from jevdrive.tokens import open_front  # noqa: E402
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, FEAT = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
RES = R_ / "experiments/pai_rows/results"
DS = "physical_ai_av"
CLASSES = ("stop_nolead", "turn_sharp", "launch", "bend", "rest")
SLOT_T = np.round(np.arange(-7, 1) * 0.2, 3)                  # sh30_core.SLOT_T: the 8 policy slots, oldest first
T_KEY = np.array([-1.5, -1.0, -0.5, 0.0])                       # op_interp.T_KEY: the ego history keys
TOL_US = 30_000
PART_VER = 3                                                     # stretch parts: 2 = encoded over the row file's whole slot table, 3 = + dilated label
HOLD = 15                                                        # the trainers' --on-hold-pct: crc32(scene) % 100 < 15 is never drawn


def held(s: str) -> bool:
    return zlib.crc32(s.encode()) % 100 < HOLD


# ---------------------------------------------------------------- scenes (Mac)
def cmd_scenes(a):
    from jevdrive.data import splits
    have = {}
    for l in open(a.usdz):
        n, sz = l.split()
        have[n[:-5]] = int(sz)
    sel = {r["scene_id"]: r for r in csv.DictReader(open(R_ / "experiments/pai_av/results/nurec_select.csv"))}
    tr, va, te = (set(splits.load(f"{DS}/{n}")) for n in ("curated-train", "curated-val", "test281"))
    F = R_ / "experiments/pai_fix/scenes"
    exam = set(open(F / "exam_scenes.txt").read().split())
    tex = {r["scene_id"] for r in csv.DictReader(open(F / "train_exclude.csv"))}
    dv = {r["scene_id"] for r in csv.DictReader(open(F / "drives_val160.csv"))}
    d402 = {r["scene_id"] for r in csv.DictReader(open(R_ / "experiments/practice_pai/results/curated/scenes570.tsv"), delimiter="\t")}
    on = {s for s, r in sel.items() if r["uuid"] in have}
    trunc = sorted(s for s in on if abs(have[sel[s]["uuid"]] / 1e9 - float(sel[s]["size_gb"])) > 1e-3)
    on_tr = (on & tr) - set(trunc)
    drop = dict(exam=on_tr & exam, train_exclude=on_tr & tex, drives_val160=on_tr & dv, curated_val=on_tr & va, test281=on_tr & te)
    el = sorted(on_tr - set().union(*drop.values()))
    by = defaultdict(list)
    for s in el:
        by[sel[s]["cls"]].append(s)
    half = {}
    for c, ss in by.items():                                     # stratified by class: alternate A / B in a fixed hash order
        for i, s in enumerate(sorted(ss, key=lambda x: hashlib.sha256(f"pai-rows-half:{x}".encode()).hexdigest())):
            half[s] = "A" if i % 2 == 0 else "B"
    rows = [dict(scene_id=s, cls=sel[s]["cls"], d402=s in d402, half=half[s], held=held(s), uuid=sel[s]["uuid"], size_gb=sel[s]["size_gb"]) for s in el]
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "scenes.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    cnt = {(h, c): sum(1 for r in rows if r["half"] == h and r["cls"] == c) for h in "AB" for c in CLASSES}
    md = [f"# PAI-ROWS scene table ({a.usdz.split('/')[-1]})\n",
          f"usdz on the box {len(have)}; listed in nurec_select.csv {len(on)}; truncated {len(trunc)}; curated-train {len(on & tr)}; "
          f"dropped: " + ", ".join(f"{k} {len(v)}" for k, v in drop.items()) + f" (union {len(on_tr) - len(el)}); **eligible {len(el)}**; "
          f"of them in the d402 pool {sum(r['d402'] for r in rows)}, new {sum(not r['d402'] for r in rows)}; d402 scenes now excluded "
          f"{len(d402 - set(el))} ({', '.join(sorted(s[7:15] for s in d402 - set(el)))}).\n",
          "| class | all | half A | half B | new (not d402) | held (crc < 15) |\n|---|--:|--:|--:|--:|--:|\n"]
    for c in CLASSES + ("all",):
        rr = [r for r in rows if c == "all" or r["cls"] == c]
        md.append(f"| {c} | {len(rr)} | {sum(r['half'] == 'A' for r in rr)} | {sum(r['half'] == 'B' for r in rr)} | {sum(not r['d402'] for r in rr)} | "
                  f"{sum(r['held'] for r in rr)} |\n")
    (RES / "scenes.md").write_text("".join(md))
    print("".join(md))
    if a.register:
        common = dict(status="frozen", used_by=["experiments/pai_rows"])
        full = splits.define(DS, "pai-rows", el, "scene", f"experiments/pai_rows/results/scenes.tsv ({a.usdz.split('/')[-1]}): curated-train NuRec "
                             "scenes with a complete usdz on the GPU box, minus pai_fix/scenes exam_scenes.txt, train_exclude.csv, drives_val160.csv "
                             "(scene_id column), curated-val and test281", notes="PAI-ROWS human-state / stretch / S rows", **common)
        ha = splits.define(DS, "pai-rows-half-a", [s for s in el if half[s] == "A"], "scene", "half A of physical_ai_av/pai-rows@v1: per nurec_select "
                           "class, alternate scenes in sha256('pai-rows-half:' + scene) order", notes="PAI-ROWS data-scaling arm (half data)", **common)
        for x in ("curated-val", "test281"):
            splits.check_disjoint(full, splits.load(f"{DS}/{x}"))
        print(full.id, ha.id)


# ---------------------------------------------------------------- check (box)
def cmd_check(a):
    import pai281 as P
    want = set(" ".join(p.read_text() for p in sorted(Path(a.root, "chunks").glob("*.txt"))).split())
    J, bad = P.load(a.root)
    sc = {s for s in J if s not in bad}
    n_dump = len(list(Path(a.root).glob("runs/*/sim/driver-logs/dump/*")))
    n_asl = len(list(Path(a.root).glob("runs/*/sim/**/rollout.asl")))
    rec = dict(root=a.root, scenes=len(want), scored=len(sc & want), rpc=sorted(bad), missing=sorted(want - sc), dump_entries=n_dump, asl=n_asl)
    ok = len(want) == a.n and len(sc & want) >= a.min_ok and n_dump > 0 and n_asl > 0
    if a.log:
        late = lg = inf = 0
        for f in glob.glob(f"{a.root}/runs/*/driver/drive.jsonl"):
            for line in open(f):
                if '"drive"' not in line:
                    continue
                x = json.loads(line)
                if x.get("kind") == "drive" and x.get("k", 0) >= 20:
                    late += 1
                    lg += x.get("ret") == "log" and x.get("log_poses") is not None
                    inf += bool(x.get("infer"))
        rec |= dict(late=late, returned_log=lg, inferences=inf)
        ok = ok and late > 0 and lg == late and inf > 0
    out = subprocess.run(["du", "-sb", a.root], capture_output=True, text=True).stdout.split()
    rec |= dict(gb=round(int(out[0]) / 1e9, 2) if out else None, pass_=bool(ok))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    raise SystemExit(0 if ok else 2)


# ---------------------------------------------------------------- stretch (GPU)
def to_rig(a0, a):
    """Local-frame poses a (n, 3) -> the rig frame of the local pose a0 (3,); yaw relative, unwrapped from a0."""
    c, s = np.cos(a0[2]), np.sin(a0[2])
    dx, dy = a[:, 0] - a0[0], a[:, 1] - a0[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, a[:, 2] - a0[2]], 1)


def stretch_row(z, n, f, yaw_u):
    """Row n of a row file at spacing factor f -> (slot (8,), P (4, 3), V (4, 2), acc (2,)) or a reason string. f = 1: the stored history."""
    t0, ft = int(z["t0_us"][n]), z["frame_ts"]
    tt = t0 + np.round(f * SLOT_T * 1e6).astype(np.int64)
    j = np.abs(ft[None] - tt[:, None]).argmin(1)
    if not (np.abs(ft[j] - tt) <= TOL_US).all():
        return "frames"
    if f == 1.0:
        return j.astype(np.int32), z["P"][n], z["V"][n], z["acc"][n]
    th = t0 + np.round(f * T_KEY[:3] * 1e6).astype(np.int64)
    T = z["t0_us"]
    if th[0] < T[0]:
        return "history"
    A = np.stack([np.interp(th, T, z["anchor"][:, 0]), np.interp(th, T, z["anchor"][:, 1]), np.interp(th, T, yaw_u)], 1)
    a0 = np.r_[z["anchor"][n, :2], yaw_u[n]]
    P = np.concatenate([to_rig(a0, A), z["P"][n][3:]], 0)
    return j.astype(np.int32), P, f * z["V"][n], f * f * z["acc"][n]


def dilated_label(path, anchor, t0, f):
    """Supplement 2: the label of a row played at f x = the log's rig pose at t0 + f x 0.5 (j + 1) s, j = 0..7, in the rig frame at the
    row's anchor (onp_rows' log_fut rule, onp_geom.Path.at_time; f = 1 reproduces log_fut bit for bit) -> (8, 3) f4 or None (log too short)."""
    import onp_geom as G
    fp, _, ok = path.at_time(np.int64(t0) + np.round(f * G.T_FUT).astype(np.int64))
    if not ok.all():
        return None
    return G.pose_to_rig(fp, anchor).astype(np.float32)


def hist_err(z, yaw_u):
    """The anchor history at f = 1 against the stored history on rows with a real 1.5 s history (k >= 15): max |diff| (x, y in m, yaw rad)."""
    T, e = z["t0_us"], np.zeros(3)
    for n in np.flatnonzero(z["k"] >= 15):
        th = int(T[n]) + np.round(T_KEY[:3] * 1e6).astype(np.int64)
        A = np.stack([np.interp(th, T, z["anchor"][:, 0]), np.interp(th, T, z["anchor"][:, 1]), np.interp(th, T, yaw_u)], 1)
        e = np.maximum(e, np.abs(to_rig(np.r_[z["anchor"][n, :2], yaw_u[n]], A) - z["P"][n][:3]).max(0))
    return e


def cmd_stretch(a):
    import torch
    import onp_cache as OC
    import parity_adapter as PA
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    f = float(a.f)
    src, out = CR / f"on1_{a.pool}", CR / f"on1_{a.name}"
    tab = np.load(src / "tab.npz")
    names = tab["names"].astype(str)
    pre = names[0].split(":")[1]
    sc = np.array([x.split(":")[2] for x in names])
    kk = np.array([int(x.split(":")[3]) for x in names])
    prefix = a.prefix or f"{pre}{('%g' % f).replace('.', '')}"
    parts = D / "runs/pai_rows/stretch" / a.name / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    scenes = list(dict.fromkeys(sc.tolist()))[:a.limit or None]
    with Run("pai_rows", f"stretch-{a.name}", config=vars(a)) as run:
        dev = torch.device("cuda")
        net = A.load("cinque", torch.float16).to(dev).eval()
        di = A.distill_index(net.slices)
        n_new, why, herr, slot_eq, t_enc = 0, Counter(), np.zeros(3), [0, 0], 0.0
        for s in run.tqdm(scenes, desc="scenes"):
            rf = Path(a.rows) / f"{s}.npz"
            base = np.flatnonzero(sc == s)
            pf, tf = parts / f"{s}.npz", parts / f"{s}.tok.npy"
            sig = np.array(OC.sig(rf))
            if pf.exists() and tf.exists() and not a.force:
                o = np.load(pf)
                if o["src"].tolist() == sig.tolist() and np.array_equal(o["names_all"], names[base]) and float(o["f"]) == f and int(o.get("ver", 0)) == PART_VER:
                    continue
            zz = np.load(rf)
            z = {k: zz[k] for k in ("frames", "frame_ts", "t0_us", "k", "slot", "P", "V", "acc", "cmd", "anchor", "ego_in", "lht", "cold_rows", "cold_cur",
                                    "log_t_us", "log_xyyaw")}
            import onp_geom as G
            lpath = G.Path(z["log_t_us"], z["log_xyyaw"])
            yaw_u = np.unwrap(z["anchor"][:, 2])
            herr = np.maximum(herr, hist_err(z, yaw_u))
            pos = {int(k): i for i, k in enumerate(z["k"])}
            keep, rows_n, ego, fut, slot2 = [], [], [], [], z["slot"].copy()
            for b in base:
                n = pos[int(kk[b])]
                if (z["slot"][n] < 0).any():                           # cold-start rows (back-warped slots): no rendered frames to re-pick
                    why["cold"] += 1
                    continue
                r = stretch_row(z, n, f, yaw_u)
                if isinstance(r, str):
                    why[r] += 1
                    continue
                j, P, V, acc = r
                lf = dilated_label(lpath, z["anchor"][n], z["t0_us"][n], f)
                if lf is None:
                    why["label"] += 1
                    continue
                if f == 1.0:
                    slot_eq[0] += int(np.array_equal(j, z["slot"][n]))
                    slot_eq[1] += 1
                keep.append(b), rows_n.append(n), fut.append(lf)
                slot2[n] = j
                ego.append(PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(z["cmd"][n], np.float32)))
            if not keep:
                np.savez(pf, ver=PART_VER, src=sig, names_all=names[base], f=f, base=np.zeros(0, np.int64), tix=np.zeros((0, 8), np.int64),
                         t_out=np.zeros((0, len(di)), np.float32), ego=np.zeros((0, 20), np.float32), fut=np.zeros((0, 8, 3), np.float32))
                np.save(tf, np.zeros((0, *A.H_SHAPE), np.float16))
                continue
            # the row file's whole slot table (stretched slots on the kept rows, the stored ones and the cold rows' frames elsewhere): pair order
            # and encoder / teacher batches are then those of onp_cache build, so f = 1 is the plain cache bit for bit (the encoder is not
            # batch-invariant at the last fp16 step: probe 1 of the prereg); rows not kept cost encoder time only
            q = dict(slot=slot2, frames=z["frames"], cold_rows=z["cold_rows"], cold_cur=z["cold_cur"])
            prev, cur, tix = OC.pairs(q)
            t1 = time.time()
            tok, t_out = OC.encode(net, A, di, prev, cur, tix, bool(z["lht"]), dev, a.bs)
            tix, t_out = tix[rows_n], t_out[rows_n]
            t_enc += time.time() - t1
            np.save(tf.with_suffix(".tmp.npy"), tok)
            os.replace(tf.with_suffix(".tmp.npy"), tf)
            np.savez(pf.with_suffix(".tmp.npz"), ver=PART_VER, src=sig, names_all=names[base], f=f, base=np.array(keep, np.int64), tix=tix, t_out=t_out,
                     ego=np.stack(ego).astype(np.float32), fut=np.stack(fut))
            os.replace(pf.with_suffix(".tmp.npz"), pf)
            n_new += 1
        # ---- assemble the pool from every scene's part (also the parts of earlier runs: appending scenes = rerun)
        P_ = [np.load(parts / f"{s}.npz") for s in scenes]
        base = np.concatenate([p["base"] for p in P_])
        n = len(base)
        new_names = np.array([f"on:{prefix}:{sc[b]}:{kk[b]}" for b in base])
        (out).mkdir(parents=True, exist_ok=True)
        (CR / f"on1_{a.name}@warp").mkdir(parents=True, exist_ok=True)
        fr = np.lib.format.open_memmap(CR / f"on1_{a.name}@warp" / "front.tmp.npy", "w+", np.float16, (n, 8, *A.H_SHAPE))
        i0 = 0
        for s, p in zip(scenes, P_):
            tok = np.load(parts / f"{s}.tok.npy", mmap_mode="r")
            for i in range(0, len(p["base"]), 256):
                fr[i0 + i:i0 + i + len(p["tix"][i:i + 256])] = tok[p["tix"][i:i + 256]]
            i0 += len(p["base"])
        fr.flush()
        del fr
        os.replace(CR / f"on1_{a.name}@warp" / "front.tmp.npy", CR / f"on1_{a.name}@warp" / "front.npy")
        te = np.load(CR / f"on1_{a.pool}@warp" / "teacher.npz")
        np.savez(CR / f"on1_{a.name}@warp" / "teacher.npz", out=np.concatenate([p["t_out"] for p in P_]), plan=np.zeros((n, 33, 15), np.float32),
                 di=te["di"], pi=te["pi"])
        T = {k: tab[k] for k in tab.files}
        newt = {k: v[base] for k, v in T.items() if len(v) == len(names)} | dict(names=new_names, ego=np.concatenate([p["ego"] for p in P_]))
        newt["speed"] = (f * T["speed"][base]).astype(np.float32)
        newt["fut"] = np.concatenate([p["fut"] for p in P_])                # supplement 2: the dilated label (= log_fut at f = 1)
        fut_eq_f1 = bool(np.array_equal(newt["fut"], T["fut"][base])) if f == 1.0 else None
        newt |= dict(f=np.full(n, f, np.float32), base=np.array([names[b] for b in base]))
        np.savez(out / "tab.npz", **newt)
        for lf in ("agent", "road", "route"):
            L = np.load(src / f"{lf}.npz")
            assert np.array_equal(L["tokens"], names), f"{lf}.npz rows differ from tab"
            d = {k: L[k][base] for k in L.files if k != "tokens"} | dict(tokens=new_names)
            if lf == "agent" and f != 1.0:                              # actor boxes are at t0 + 0.5 j, not on the dilated clock: no agent label
                d["ok"] = np.zeros(n, bool)
            np.savez(out / f"{lf}.npz", **d)
        Fz = np.load(FEAT / f"on1_{a.pool}.npz")
        assert np.array_equal(Fz["names"], names)
        np.savez(FEAT / f"on1_{a.name}.npz", **({k: Fz[k][base] for k in Fz.files if k != "names"} | dict(names=new_names)))
        rec = dict(name=a.name, pool=a.pool, f=f, prefix=prefix, scenes=len(scenes), scenes_encoded_now=n_new, rows=int(n), rows_plain=int(len(names)),
                   dropped=dict(why), hist_src_err_f1=dict(x=float(herr[0]), y=float(herr[1]), yaw=float(herr[2])),
                   slot_repick_equal_f1=None if f != 1.0 else f"{slot_eq[0]} / {slot_eq[1]}", fut_equal_log_fut_f1=fut_eq_f1, encode_s=round(t_enc, 1),
                   front_gb=round(n * 8 * 32 * 512 * 2 / 1e9, 2), out=str(out))
        (out / "build.json").write_text(json.dumps(rec, indent=1))
        run.summary.update({k: v for k, v in rec.items() if not isinstance(v, dict)})
        run.info(json.dumps(rec))


# ---------------------------------------------------------------- gate1 (CPU): f = 1 against the plain pool, bit for bit
def cmd_gate1(a):
    P, Q = CR / f"on1_{a.plain}", CR / f"on1_{a.f1}"
    tp, tq = np.load(P / "tab.npz"), np.load(Q / "tab.npz")
    idx = {x: i for i, x in enumerate(tp["names"].astype(str))}
    b = np.array([idx[x] for x in tq["base"].astype(str)])
    fp, fq = (open_front(x.parent / f"{x.name}@warp" / "front.npy") for x in (P, Q))
    ep, eq = (np.load(x.parent / f"{x.name}@warp" / "teacher.npz")["out"] for x in (P, Q))
    rs = np.arange(len(b))
    fr_eq = np.array([np.array_equal(fp[b[i]], fq[i]) for i in rs])
    fr_d = max((float(np.abs(fp[b[i]].astype(np.float32) - fq[i].astype(np.float32)).max()) for i in rs[~fr_eq][:2000]), default=0.0)
    res = dict(plain=a.plain, f1=a.f1, rows=int(len(b)), rows_plain=int(len(tp["names"])), ego_equal=bool(np.array_equal(tp["ego"][b], tq["ego"])),
               fut_equal=bool(np.array_equal(tp["fut"][b], tq["fut"])), front_equal_share=float(fr_eq.mean()), front_max_abs_diff=fr_d,
               teacher_equal=bool(np.array_equal(ep[b], eq)), teacher_max_abs_diff=float(np.abs(ep[b] - eq).max()),
               build=json.loads((Q / "build.json").read_text()))
    res["pass_"] = bool(res["ego_equal"] and res["fut_equal"] and res["front_equal_share"] == 1.0 and res["teacher_equal"]
                        and res["build"]["slot_repick_equal_f1"].split(" / ")[0] == res["build"]["slot_repick_equal_f1"].split(" / ")[1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["pass_"] else 2)


# ---------------------------------------------------------------- feas / report (CPU)
def scene_table():
    """Scene -> row of results/scenes.tsv (the 853) and results/scenes_n.tsv (the second batch of 64, when present)."""
    S = {}
    for f in ("scenes.tsv", "scenes_n.tsv"):
        if (RES / f).exists():
            S |= {r["scene_id"]: r for r in csv.DictReader(open(RES / f), delimiter="\t")}
    return S


def cmd_feas(a):
    """Supplement 2: required constant acceleration from the fed speed f v0 to the label's pose at t (cumulative chord d_t):
    a_t = 2 (d_t - f v0 t) / t^2 at t = 0.5 / 1 / 4 s; shares beyond 3 / 6 / 9 m/s^2, and per row against the plain row it came from
    (median |a_t^f| / (f^2 |a_t^1|) over rows whose plain |a_t| >= 0.5: a consistent dilation gives f^2)."""
    S = scene_table()
    TT = {0.5: 0, 1.0: 1, 4.0: 7}
    acc = {}
    md = ["# PAI-ROWS: required acceleration under the label (supplement 2: the label dilated with the clip)\n\n",
          "a_t = 2 (d_t - f v0 t) / t^2: the constant acceleration that takes the fed speed f v0 to the label's pose at t (d_t = cumulative chord). "
          "Ratio = median over rows of |a_t(f)| / (f^2 |a_t(plain)|) on the same decision (rows with plain |a_t| >= 0.5 m/s^2); 1.0 = the human's own times f^2.\n\n",
          "| pool | f | rows | v0 median | " + " | ".join(f"abs a{t:g} > 3 / 6 / 9" for t in TT) + " | " + " | ".join(f"ratio a{t:g}" for t in TT) + " |\n|---|--:|--:|--:|" + "---|" * (2 * len(TT)) + "\n"]
    out, by = {}, []
    for spec in a.pools:
        p, f = spec.split(":")
        f = float(f)
        t = np.load(CR / f"on1_{p}" / "tab.npz")
        v0 = t["speed"] / f
        xy = np.concatenate([np.zeros((len(v0), 1, 2)), t["fut"][..., :2]], 1)
        d = np.cumsum(np.hypot(*np.diff(xy, axis=1).transpose(2, 0, 1)), 1)
        A_ = {tt: 2 * (d[:, k] - f * v0 * tt) / tt ** 2 for tt, k in TT.items()}
        names = t["base"].astype(str) if "base" in t.files else t["names"].astype(str)
        acc[p] = (f, names, A_)
        sh = lambda x: [float((np.abs(x) > q).mean()) for q in (3, 6, 9)]  # noqa: E731
        cls = np.array([S.get(x, {}).get("cls", "?") for x in t["log"].astype(str)])
        out[p] = dict(f=f, rows=int(len(v0)), v0_med=float(np.median(v0)), over={str(tt): sh(x) for tt, x in A_.items()},
                      by_cls={c: {str(tt): sh(x[cls == c]) for tt, x in A_.items()} for c in CLASSES if (cls == c).any()},
                      by_v0={b: {str(tt): sh(x[m]) for tt, x in A_.items()} for b, m in (("<2", v0 < 2), ("2-5", (v0 >= 2) & (v0 < 5)),
                             ("5-10", (v0 >= 5) & (v0 < 10)), (">=10", v0 >= 10)) if m.any()})
    plain = [p for p, (f, _, _) in acc.items() if f == 1.0]
    for p, (f, names, A_) in acc.items():
        r = {}
        if plain and f != 1.0:
            f1, n1, A1 = acc[plain[0]]
            ix = {x: i for i, x in enumerate(n1)}
            j = np.array([ix.get(x, -1) for x in names])
            for tt in TT:
                m = (j >= 0) & (np.abs(A1[tt][np.maximum(j, 0)]) >= 0.5)
                r[str(tt)] = float(np.median(np.abs(A_[tt][m]) / (f * f * np.abs(A1[tt][j[m]])))) if m.any() else float("nan")
        out[p]["ratio_f2"] = r
        o_ = out[p]
        md.append(f"| {p} | {f:g} | {o_['rows']} | {o_['v0_med']:.1f} | " + " | ".join(" / ".join(f"{x:.3f}" for x in o_["over"][str(tt)]) for tt in TT)
                  + " | " + " | ".join(f"{r[str(tt)]:.2f}" if str(tt) in r else "-" for tt in TT) + " |\n")
        by += [f"| {p} | {k} | " + " | ".join(" / ".join(f"{x:.3f}" for x in v[str(tt)]) for tt in TT) + " |\n" for k, v in {**o_["by_cls"], **o_["by_v0"]}.items()]
    md += ["\n| pool | class or v0 bin (m/s) | " + " | ".join(f"abs a{t:g} > 3 / 6 / 9" for t in TT) + " |\n|---|---|" + "---|" * len(TT) + "\n"] + by
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "feas.json").write_text(json.dumps(out, indent=1))
    (Path(a.out) / "feas.md").write_text("".join(md))
    print("".join(md))


def cmd_report(a):
    S = scene_table()
    rec, md = {}, ["# PAI-ROWS: rows per pool, class and half\n\n`drawn` = outside the trainers' held scenes (crc32(scene) % 100 < 15).\n\n",
                   "| pool | scenes | rows | drawn | " + " | ".join(CLASSES) + " | half A rows (drawn) | half B rows (drawn) | GB |\n|---|" + "--:|" * (8 + len(CLASSES)) + "\n"]
    for p in a.pools:
        t = np.load(CR / f"on1_{p}" / "tab.npz")
        lg = t["log"].astype(str)
        cls = np.array([S.get(x, {}).get("cls", "?") for x in lg])
        hf = np.array([S.get(x, {}).get("half", "?") for x in lg])
        hd = np.array([held(x) for x in lg])
        gb = sum(int(subprocess.run(["du", "-sbL", str(CR / x)], capture_output=True, text=True).stdout.split()[0]) for x in (f"on1_{p}", f"on1_{p}@warp")) / 1e9
        rec[p] = dict(scenes=int(len(set(lg))), rows=int(len(lg)), drawn=int((~hd).sum()), by_cls={c: int((cls == c).sum()) for c in CLASSES},
                      by_cls_scenes={c: int(len(set(lg[cls == c]))) for c in CLASSES}, half={h: dict(rows=int((hf == h).sum()), drawn=int(((hf == h) & ~hd).sum()),
                      scenes=int(len(set(lg[hf == h])))) for h in "AB"}, unknown=int((cls == "?").sum()), gb=round(gb, 1))
        if "t281" in t.files:                                   # second batch: rows whose scene shares a drive with a PAI test281 scene
            rec[p]["t281_rows"], rec[p]["t281_scenes"] = int((t["t281"] != "").sum()), int(len(set(lg[t["t281"] != ""])))
        r = rec[p]
        md.append(f"| {p} | {r['scenes']} | {r['rows']} | {r['drawn']} | " + " | ".join(f"{r['by_cls'][c]} ({r['by_cls_scenes'][c]})" for c in CLASSES)
                  + f" | {r['half']['A']['rows']} ({r['half']['A']['drawn']}) | {r['half']['B']['rows']} ({r['half']['B']['drawn']}) | {r['gb']} |\n")
    md.append("\nClass cells: rows (scenes).\n")
    if any("t281_rows" in r for r in rec.values()):
        md.append("\nRows tagged t281 (scene on the same drive as a PAI test281 scene): " + ", ".join(
            f"{p} {r['t281_rows']} rows / {r['t281_scenes']} scenes" for p, r in rec.items() if "t281_rows" in r) + "\n")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "rows.json").write_text(json.dumps(rec, indent=1))
    (Path(a.out) / "rows.md").write_text("".join(md))
    print("".join(md))


# ---------------------------------------------------------------- second batch: scenes, test281 partner, tag, merge checks
POSE_LEAD_S = 4.95                                              # pai_av pav.py: pose_record starts ~4.95 s before the scene's 20 s
NCOLS = ("scene_id", "cls", "d402", "half", "held", "uuid", "size_gb", "t281", "t281_gap_s", "t281_why")


def cmd_scenes_n(a):
    from jevdrive.data import splits
    q = list(csv.DictReader(open(RES / "download_queue_v2.csv")))
    ids = [r["scene_id"] for r in q]
    assert len(ids) == len(set(ids)), "duplicate scenes in the queue"
    par = {r["scene_id"]: r for r in csv.DictReader(open(a.partner), delimiter="\t")} if a.partner else {}
    tr, va, te, full = (set(splits.load(f"{DS}/{n}")) for n in ("curated-train", "curated-val", "test281", "pai-rows"))
    F = R_ / "experiments/pai_fix/scenes"
    bad = set(open(F / "exam_scenes.txt").read().split()) | {r["scene_id"] for f in ("train_exclude.csv", "drives_val160.csv")
                                                             for r in csv.DictReader(open(F / f))}
    s = set(ids)
    assert s <= tr and not s & (va | te | full | bad), f"not curated-train {len(s - tr)}, in val / test281 / pai-rows / excluded {len(s & (va | te | full | bad))}"
    rows = []
    for r in q:
        x = par.get(r["scene_id"], {})
        rows.append(dict(scene_id=r["scene_id"], cls=r["cls"], d402=False, half="N", held=held(r["scene_id"]), uuid=r["uuid"], size_gb=r["size_gb"],
                         t281=x.get("t281", ""), t281_gap_s=x.get("t281_gap_s", ""), t281_why=x.get("t281_why", "")))
    with open(RES / "scenes_n.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, NCOLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["cls"] for r in rows)
    print(f"{len(rows)} scenes {dict(c)}; held {sum(r['held'] for r in rows)}; with a test281 partner {sum(bool(r['t281']) for r in rows)}")
    if a.register:
        assert par and all(r["t281"] for r in rows), "register after the partner table (every scene shares a drive with a test281 scene)"
        common = dict(status="frozen", used_by=["experiments/pai_rows"])
        n = splits.define(DS, "pai-rows-n", ids, "scene", "experiments/pai_rows/results/scenes_n.tsv: the 64 curated-train NuRec scenes of "
                          "download_queue_v2.csv (stop_nolead / turn_sharp / launch, each on the same drive 2-10 min from a PAI test281 scene), "
                          "complete on the GPU box 2026-10-21; disjoint from curated-val, test281, pai-rows@v1 and the pai_fix exclusions",
                          notes="PAI-ROWS second batch (human-state rows on1_PAIHN*)", **common)
        x = splits.define(DS, "pai-rows-x", sorted(full | s), "scene", "physical_ai_av/pai-rows@v1 + pai-rows-n@v1 (853 + 64)",
                          notes="PAI-ROWS all training-eligible scenes with human-state rows", **common)
        for y in ("curated-val", "test281"):
            splits.check_disjoint(n, splits.load(f"{DS}/{y}"))
            splits.check_disjoint(x, splits.load(f"{DS}/{y}"))
        print(n.id, x.id)


def _km(la1, lo1, la2, lo2):
    return 6371 * np.hypot(np.radians(la2 - la1), np.radians(lo2 - lo1) * np.cos(np.radians(la1)))


def _clock(meta, scene):
    """(lat0, lon0, epoch0 s, session_start s) of a scene from its nurec_meta json, as pai_av's nurecsel reads them."""
    m = json.loads((Path(meta) / f"{scene}.json").read_text())
    t = np.array(m["t_us"], np.int64)
    k = (((t - t[0]) * 1e-6 - POSE_LEAD_S >= 0) & ((t - t[0]) * 1e-6 - POSE_LEAD_S <= 20))
    ep = t[0] * 1e-6 if t[0] > 1e14 else np.nan
    return float(np.array(m["lat"])[k][0]), float(np.array(m["lon"])[k][0]), ep, float((m.get("time_range") or {}).get("start", np.nan)) * 1e-6


def cmd_partner(a):
    """For every scene of --scenes: the nearest (smallest clock gap) test281 scene on the same drive by pai_av's rule, and the same for
    val160 as a check (none expected: those links were removed); --queue's exam_gap_s must equal the smaller of the two."""
    from jevdrive.data import splits
    S = [r["scene_id"] for r in csv.DictReader(open(a.scenes), delimiter="\t")]
    gap_q = {r["scene_id"]: float(r["exam_gap_s"]) for r in csv.DictReader(open(a.queue))}
    E = {n: sorted(splits.load(f"{DS}/{n}")) for n in ("test281", "curated-val")}
    X = {g: {s: _clock(a.meta, s) for s in ss} for g, ss in E.items()}

    def best(c, grp):
        la, lo, ep, ss = c
        out = (np.inf, "", "")
        for s, (la2, lo2, ep2, ss2) in X[grp].items():
            d = _km(la, lo, la2, lo2)
            w, g = [], np.inf
            de, ds = abs(ep2 - ep), abs(ss2 - ss)
            if np.isfinite(ep) and np.isfinite(ep2) and de < 600 and d < 0.3 + 0.03 * de:
                w.append("epoch")
                g = min(g, de)
            if np.isfinite(ss) and np.isfinite(ss2) and ds < 600 and d < 0.3 + 0.03 * ds:
                w.append("session")
                g = min(g, ds)
            if w and g < out[0]:
                out = (g, s, "+".join(w))
        return out
    rows, mism = [], []
    for s in S:
        c = _clock(a.meta, s)
        g1, p1, w1 = best(c, "test281")
        g2, p2, w2 = best(c, "curated-val")
        rows.append(dict(scene_id=s, t281=p1[7:15], t281_scene=p1, t281_gap_s=round(g1, 1), t281_why=w1, val160_gap_s=round(g2, 1) if np.isfinite(g2) else "",
                         val160_scene=p2, csv_exam_gap_s=round(gap_q[s], 1)))
        if abs(min(g1, g2) - gap_q[s]) > 0.5:
            mism.append(s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    gg = np.array([r["t281_gap_s"] for r in rows], float)
    print(f"{len(rows)} scenes; test281 partner found {int(np.isfinite(gg).sum())}; gap s min {gg.min():.0f} median {np.median(gg):.0f} max {gg.max():.0f}; "
          f"also a val160 partner {sum(r['val160_gap_s'] != '' for r in rows)}; queue exam_gap_s differs from min(test281, val160) on {len(mism)} {mism[:5]}")
    raise SystemExit(0 if np.isfinite(gg).all() and not mism else 2)


def cmd_tag(a):
    """Per-row t281 tag in the tab of a pool built from the second batch: the first 8 characters of the partner test281 scene (after clipgt-)."""
    T = CR / f"on1_{a.pool}" / "tab.npz"
    tb = dict(np.load(T))
    sc = {r["scene_id"]: r["t281"] for r in csv.DictReader(open(a.scenes), delimiter="\t")}
    lg = tb["log"].astype(str)
    assert set(lg) <= set(sc) and all(sc[x] for x in set(lg)), "a row of a scene without a test281 partner"
    tb["t281"] = np.array([sc[x] for x in lg], "U8")
    np.savez(T, **tb)
    print(json.dumps(dict(pool=a.pool, rows=len(lg), scenes=len(set(lg)), tagged=int((tb["t281"] != "").sum()))))


def _same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")


def _prefix_diff(P, R, skip_names, tmp=None):
    """Mismatching items between the first len(R) rows of pool P and pool R: tab keys, agent / road / drivable / route files, route feat, teacher,
    4000 sampled token rows. Items of P beyond that (new rows) are not looked at."""
    tp, tr = dict(np.load(CR / f"on1_{P}" / "tab.npz")), np.load(CR / f"on1_{R}" / "tab.npz")
    n = len(tr["names"])
    bad = [f"tab.{k} missing" for k in tr.files if k not in tp]
    bad += [f"tab.{k}" for k in tr.files if k in tp and not (skip_names and k == "names") and not _same(tp[k][:n], tr[k])]
    for f in ("agent.npz", "road.npz", "drivable.npz", "route.npz"):
        if not (CR / f"on1_{R}" / f).exists():
            continue
        zp, zr = np.load(CR / f"on1_{P}" / f), np.load(CR / f"on1_{R}" / f)
        bad += [f"{f}.{k}" for k in zr.files if not (skip_names and k == "tokens") and zr[k].ndim and len(zr[k]) == n and not _same(zp[k][:n], zr[k])
                and not (f == "agent.npz" and k == "ok")]
    fp, fr = np.load(FEAT / f"on1_{P}.npz"), np.load(FEAT / f"on1_{R}.npz")
    bad += [f"feat.{k}" for k in fr.files if not (skip_names and k == "names") and not _same(fp[k][:n], fr[k])]
    zp, zr = np.load(CR / f"on1_{P}@warp" / "teacher.npz"), np.load(CR / f"on1_{R}@warp" / "teacher.npz")
    bad += [f"teacher.{k}" for k in zr.files if not _same(zp[k][:n] if zr[k].ndim and len(zr[k]) == n else zp[k], zr[k])]
    i = np.sort(np.random.default_rng(0).choice(n, min(4000, n), replace=False))
    if not _same(np.asarray(open_front(CR / f"on1_{P}@warp" / "front.npy")[i]), np.asarray(open_front(CR / f"on1_{R}@warp" / "front.npy")[i])):
        bad.append("front sample")
    return n, tp, bad


def cmd_mcheck(a):
    n, tp, bad = _prefix_diff(a.pool, a.ref, a.skip_names)
    ag, ar = np.load(CR / f"on1_{a.pool}" / "agent.npz"), np.load(CR / f"on1_{a.ref}" / "agent.npz")
    ag_ok = _same(ag["ok"][:n], ar["ok"])
    nt = np.load(CR / f"on1_{a.pool}" / "tab.npz")
    rec = dict(pool=a.pool, ref=a.ref, ref_rows=n, rows=len(nt["names"]), parts={p: int((nt["pool"] == p).sum()) for p in np.unique(nt["pool"])},
               agent_ok_prefix_equal=ag_ok, mismatch=bad, pass_=not bad and ag_ok)
    out = Path(a.out) / f"mcheck_{a.pool}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    raise SystemExit(0 if rec["pass_"] else 2)


def cmd_xfix(a):
    """JTPH-style merge of the new human rows: prefix = --ref (every file), agent ok of the prefix as in --ref, ok False on --parts (as r2.py fix)."""
    n, tp, bad = _prefix_diff(a.pool, a.ref, False)
    P = CR / f"on1_{a.pool}"
    ag, ar = dict(np.load(P / "agent.npz")), np.load(CR / f"on1_{a.ref}" / "agent.npz")
    part = tp["pool"].astype(str)
    m = np.isin(part, a.parts.split(","))
    assert m[n:].all() and not m[:n].any(), "the new parts must be exactly the rows after the reference"
    before = int((ag["ok"][:n] != ar["ok"]).sum()), int(ag["ok"][n:].sum())
    ag["ok"] = np.concatenate([ar["ok"], np.zeros(len(part) - n, bool)])
    np.savez(P / "agent.npz", **ag)
    rd = np.load(P / "road.npz")
    rec = dict(pool=a.pool, ref=a.ref, ref_rows=n, rows=len(part), parts={p: int((part == p).sum()) for p in np.unique(part)}, agent_ok_prefix_fixed=before[0],
               agent_ok_new_before=before[1], agent_ok_new_after=int(ag["ok"][n:].sum()), road_ok_new=int(rd["ok"][n:].sum()),
               new_rows_mode=sorted(set(tp["mode"][n:].astype(str).tolist())), mismatch=bad, pass_=not bad and not rd["ok"][n:].any())
    out = Path(a.out) / f"xfix_{a.pool}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    raise SystemExit(0 if rec["pass_"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("scenes")
    p.add_argument("--usdz", default=str(RES / "usdz_20261020.txt"))
    p.add_argument("--register", action="store_true")
    p = sp.add_parser("scenes-n")
    p.add_argument("--partner", default="")
    p.add_argument("--register", action="store_true")
    p = sp.add_parser("partner")
    p.add_argument("--scenes", required=True)
    p.add_argument("--queue", default=str(RES / "download_queue_v2.csv"))
    p.add_argument("--meta", default=str(D / "runs/pai_av/nurec_meta"))
    p.add_argument("--out", required=True)
    p = sp.add_parser("tag")
    p.add_argument("--pool", required=True)
    p.add_argument("--scenes", default=str(RES / "scenes_n.tsv"))
    p = sp.add_parser("xfix")
    p.add_argument("--pool", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--parts", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("mcheck")
    p.add_argument("--pool", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--skip-names", action="store_true")
    p.add_argument("--out", required=True)
    p = sp.add_parser("check")
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--min-ok", type=int, required=True)
    p.add_argument("--log", action="store_true")
    p.add_argument("--out", required=True)
    p = sp.add_parser("stretch")
    p.add_argument("--pool", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--f", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("gate1")
    p.add_argument("--plain", required=True)
    p.add_argument("--f1", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("feas")
    p.add_argument("--pools", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--pools", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(scenes=cmd_scenes, **{"scenes-n": cmd_scenes_n}, partner=cmd_partner, tag=cmd_tag, xfix=cmd_xfix, mcheck=cmd_mcheck, check=cmd_check, stretch=cmd_stretch, gate1=cmd_gate1, feas=cmd_feas, report=cmd_report)[a.cmd](a)
