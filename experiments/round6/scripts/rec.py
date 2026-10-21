#!/usr/bin/env python3
"""ROUND6 line B: perturbation-recovery rows (ChauffeurNet-style; plans/2026-10-21-recovery-rows.md). Lane script: scripts/rec.sh. Box,
repo root, envs/op-train python (CPU subcommands; the GPU work is the pool jobs rec.sh submits). Geometry: lib/rec_geom.py; driver: lib/rec_driver.py.

  sched  --pass A|B|C   episodes on the plain PAI human pool's scenes (on1_PAIH, pai-rows scenes.tsv, held scenes out): candidate decisions k0
         (k0 >= 37 so the 2 s ramp starts after the runtime's 1.7 s log window; rows k0 - 4 .. k0 + 4 in on1_PAIH; log speed >= 2 m/s over
         t0 - 2 .. t0 + 2 s; plain raster ok_primary), class by the human's 4 s yaw (turn >= 30 deg, curve >= 8, straight), weighted random order
         (weight 1 / class share of the moving PAIH rows), greedy with >= 43 decisions between episodes (B: also >= 10 from A's, C: from A's and B's); per episode a
         perturbation drawn (type lat / head / both, |e0| in 0.5 / 1 / 1.5 m, |psi0| in 3 / 6 / 10 deg, signs at random) until it passes:
         added lateral acceleration v^2 |E''| <= 3 m/s^2 over ramp and recovery, the perturbed t0 footprint inside the plain row's road_pai
         raster (min SDF >= 0), no overlap with an actor box at t0 .. t0 + 2 s (0.5 s steps) -> $O/eps/{eps_<pass>.json, episodes_<pass>.csv,
         sched_<pass>.json}; A also the probe (11 scenes), eps_Z.json (the probe's episodes at e0 = psi0 = 0) and every pass's chunk lists
  select --pass A|B|C|Z [--prefix P]  the rows k0 - 4 .. k0 + 4 of every episode in the pass's row files, label = the recovery merge (rec_geom.merge
         from the driven offset at the row's time; = log_fut bit for bit at zero offset), rows at / after an at-fault collision dropped
         -> $O/select/<pass>.npz (names on:<prefix>:<scene>:<k>, fut, per-row meta) for rf_pai_pool --select
  g1     --pool PRECZT   G1: the zero pass reproduces on1_PAIH: row files (frames, slots, inputs, anchors, model plans, log_fut) and pool rows
         (ego, fut, 8 front tokens, teacher) bit-equal -> results/rec/g1.json
  g2     --pool P --pass A[,B]   G2: the recovery label in the plain row's raster (footprint min SDF >= 0 along [ego, 8 label poses] at 0.1 s) on
         >= 99 % of the rows whose plain twin has a raster and keeps its own label inside; lateral acceleration (three-point curvature of the
         label path x the human's speed) <= max(4 m/s^2, the plain twin's own) on >= 99 % -> results/rec/g2_<P>.json
  g3     --pool P --pass A[,B] --r1 F   G3: R1 (trainer forward = the driver's recorded NTV0 plan) passes, and NTV0's plan on the perturbed rows
         is not degenerate against its own plan on the plain twin (median 4 s path-length ratio in [0.8, 1.25], median world 4 s endpoint
         distance <= 3 m); recovering share (plan endpoint closer to the log than the ego) -> results/rec/g3_<P>.json
  gates  --probe   G1-G3 of the probe -> results/rec/gates_probe.{md,json}; exit 2 if any fails
  weights --pool on1_JTPHREC --k K --out W   draw file: R5LC's weights x 64 / (64 + K) on the on1_JTPH prefix, the recovery rows K / (64 + K)
         uniform (K = 0: R5LC's weights bit-equal, the identity) -> $DATA_DIR/runs/round6/rec/w/<W>.npz + results/rec/w_<W>.json
  smoke  --tag T   r5.py's smoke checks (arm A rules: batch 51 / 13 / 72, imitation rows 32 + 8, group shares) -> results/rec/smoke_R6REC.json
  rows   --pool P[,Q] --pass A[,B]   rows by perturbation type / size / class / speed, realised vs intended offset, drops -> results/rec/rows.{md,json}
  verdict --json J --md M --pcs Q   the registered adopt-candidate rule (both seeds) appended to M
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round6/lib"), str(REPO / "experiments/onpol_pai/lib"),
                str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/round3/scripts")]
import onp_geom as G  # noqa: E402
import rec_geom as RG  # noqa: E402
from drivable_hinge import CORNERS, interp_matrix  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, FEAT = D / "runs/op_parity/cache", D / "runs/route_mem/feat"
O = D / "runs/round6/rec"
ROWS_H, RP, W5 = D / "runs/pai_rows/rows/H", D / "runs/round5/road_pai", D / "runs/round5/r5/w"
RES = REPO / "experiments/round6/results/rec"
TSV = REPO / "experiments/pai_rows/results/scenes.tsv"
K_MIN, DROWS, GAP, AVOID = 37, np.arange(-4, 5), 43, 10
V_MIN, A_ADD, TRIES = 2.0, 3.0, 30
E_SET, PSI_SET, TYPES = (0.5, 1.0, 1.5), (3.0, 6.0, 10.0), ("lat", "head", "both")
YAW_TURN, YAW_CURVE = 30.0, 8.0
CLS = ("turn", "curve", "straight")
OFF, HL, HW = 1.4675, 2.6035, 1.0785                          # PAI ego box: rig -> box centre, half length, half width (rollout veh_sim)
X0, Y0, CELL = -8.0, -24.0, 0.5                               # road_pai raster (drivable_hinge layout)
HOLD = 15
CHUNK = 32
G2_LINE, G2_ACC = 0.99, 4.0


def held(s):
    return zlib.crc32(s.encode()) % 100 < HOLD


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def to_rig(p, a):
    """Local poses (..., 3) -> rig frame at the local pose a (3,)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = p[..., 0] - a[0], p[..., 1] - a[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - a[2]], -1)


def from_rig(p, a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.stack([a[0] + c * p[..., 0] - s * p[..., 1], a[1] + s * p[..., 0] + c * p[..., 1], p[..., 2] + a[2]], -1)


def sdf_at(R, xy):
    """Bilinear SDF of the raster R (128, 96) at points (..., 2) in its frame (grid_sample, align_corners False, border clamp)."""
    fi = np.clip((xy[..., 0] - X0) / CELL - 0.5, 0, 127)
    fj = np.clip((xy[..., 1] - Y0) / CELL - 0.5, 0, 95)
    i0, j0 = np.minimum(fi.astype(int), 126), np.minimum(fj.astype(int), 94)
    di, dj = fi - i0, fj - j0
    R = np.asarray(R, np.float32)
    return (R[i0, j0] * (1 - di) * (1 - dj) + R[i0 + 1, j0] * di * (1 - dj) + R[i0, j0 + 1] * (1 - di) * dj + R[i0 + 1, j0 + 1] * di * dj)


def foot_min(R, P):
    """Min SDF of the Pacifica footprint corners (drivable_hinge CORNERS, rear axle) over poses P (n, 3) in the raster frame."""
    c, s = np.cos(P[:, 2:3]), np.sin(P[:, 2:3])
    cx = P[:, :1] + c * CORNERS[None, :, 0] - s * CORNERS[None, :, 1]
    cy = P[:, 1:2] + s * CORNERS[None, :, 0] + c * CORNERS[None, :, 1]
    return float(sdf_at(R, np.stack([cx, cy], -1)).min())


def rect(cx, cy, yaw, hl, hw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[cx + c * a - s * b, cy + s * a + c * b] for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def overlap(A, B):
    for Q in (A, B):
        for i in range(4):
            e = Q[(i + 1) % 4] - Q[i]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def agent_hit(q, box, bok):
    """q (5, 3): the ego's rear-axle poses at t0 + 0.5 j in the plain row's frame; box (9, 32, 5) x, y, yaw, L, W, bok (9, 32)."""
    for j in range(len(q)):
        E = rect(q[j, 0] + OFF * np.cos(q[j, 2]), q[j, 1] + OFF * np.sin(q[j, 2]), q[j, 2], HL, HW)
        for b in box[j][bok[j]]:
            if np.hypot(b[0] - q[j, 0], b[1] - q[j, 1]) < 15 and overlap(E, rect(b[0], b[1], b[2], b[3] / 2, b[4] / 2)):
                return True
    return False


def cls_of(yaw4_rad):
    y = abs(np.degrees(yaw4_rad))
    return "turn" if y >= YAW_TURN else "curve" if y >= YAW_CURVE else "straight"


# ---------------------------------------------------------------- sched
_G = {}


def _load_globals():
    t = np.load(CR / "on1_PAIH" / "tab.npz")
    names = t["names"].astype(str)
    pk = defaultdict(set)
    for x in names:
        _, _, s, k = x.split(":")
        pk[s].add(int(k))
    r = np.load(RP / "road_on1_PAIH.npz")
    road = {x: (int(i), bool(o), float(m)) for x, i, o, m in zip(r["tokens"].astype(str), r["ix"], r["ok_primary"], r["m_lab"])}
    mv = t["speed"] >= V_MIN
    c = Counter(cls_of(y) for y in t["fut"][mv, 7, 2])
    tot = sum(c.values())
    _G.update(pk=pk, road=road, sdf=np.load(RP / str(r["sdf_file"]), mmap_mode="r"), wcls={k: tot / max(c[k], 1) / 3 for k in CLS},
              cls_share={k: c[k] / tot for k in CLS})


def _draw(z, i, path, ta, Sa, R, rng):
    t0, A0 = int(z["t0_us"][i]), z["anchor"][i]
    lt, lv = z["log_t_us"], z["log_v"].astype(np.float64)
    tg = t0 + np.arange(-20, 21) * 100_000
    tj = t0 + np.arange(5) * 500_000
    why = Counter()
    for n in range(TRIES):
        ty = TYPES[rng.integers(3)]
        e0 = float(E_SET[rng.integers(3)] * rng.choice((-1.0, 1.0))) if ty != "head" else 0.0
        ps = float(PSI_SET[rng.integers(3)] * rng.choice((-1.0, 1.0))) if ty != "lat" else 0.0
        tp = float(np.tan(np.radians(ps)))
        ep = [(t0, e0, tp)]
        _, _, D2 = RG.driven(tg, ta, Sa, ep)
        acc = float((np.interp(tg, lt, lv) ** 2 * np.abs(D2)).max())
        if acc > A_ADD:
            why["accel"] += 1
            continue
        Ej, Dj, _ = RG.driven(tj, ta, Sa, ep)
        q = to_rig(RG.offset(path.at_time(tj)[0], Ej, Dj), A0)
        m0 = foot_min(R, q[:1])
        if m0 < 0:
            why["offroad"] += 1
            continue
        if agent_hit(q, z["box"][i], z["box_ok"][i]):
            why["agent"] += 1
            continue
        return dict(t0=t0, e0=e0, psi=ps, tp=tp, type=ty, a_add=round(acc, 3), m_t0=round(m0, 3), tries=n + 1), why
    return None, why


def _sched_scene(arg):
    s, pas, avoid = arg
    pk, road, wcls = _G["pk"][s], _G["road"], _G["wcls"]
    z = np.load(ROWS_H / f"{s}.npz")
    z = {k: z[k] for k in ("k", "t0_us", "anchor", "log_t_us", "log_xyyaw", "log_v", "log_fut", "box", "box_ok")}
    path = G.Path(z["log_t_us"], z["log_xyyaw"])
    ta, Sa = RG.arc(z["log_t_us"], z["log_xyyaw"])
    lt, lv = z["log_t_us"], z["log_v"].astype(np.float64)
    cands, why = [], Counter()
    for i, kk in enumerate(z["k"].tolist()):
        if kk < K_MIN or not all(kk + d in pk for d in DROWS):
            continue
        t0 = int(z["t0_us"][i])
        tg = t0 + np.arange(-20, 21) * 100_000
        if tg[0] < lt[0] or tg[-1] > lt[-1] or np.interp(tg, lt, lv).min() < V_MIN:
            why["slow"] += 1
            continue
        r = road.get(f"on:paih:{s}:{kk}")
        if r is None or not r[1]:
            why["raster"] += 1
            continue
        cands.append((kk, i, cls_of(z["log_fut"][i, 7, 2]), r[0]))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"rec:{pas}:{s}".encode()).digest()[:8], "little"))
    out = []
    if cands:
        key = rng.random(len(cands)) ** (1.0 / np.array([wcls[c[2]] for c in cands]))
        chosen = []
        for j in np.argsort(-key):
            kk, i, c, ix = cands[j]
            if any(abs(kk - x) < GAP for x in chosen) or any(abs(kk - x) < AVOID for x in avoid):
                continue
            ep, w = _draw(z, i, path, ta, Sa, _G["sdf"][ix], rng)
            why.update(w)
            if ep is None:
                why["no_draw"] += 1
                continue
            chosen.append(kk)
            out.append(dict(scene=s, k0=kk, cls=c, v0=round(float(np.interp(ep["t0"], lt, lv)), 3), **ep))
    return s, sorted(out, key=lambda e: e["k0"]), len(cands), why


def write_chunks(pas, scenes, first=None):
    """Chunk lists of a pass: A / Z start with the probe (1, then 10 scenes), the rest 32 each."""
    root = O / "roll" / pas
    rest = [s for s in scenes if s not in (first or [])]
    groups = ([first[:1], first[1:]] if first else []) + [rest[i:i + CHUNK] for i in range(0, len(rest), CHUNK)]
    names = []
    for n, g in enumerate(groups):
        if not g:
            continue
        c = f"{pas.lower()}{n:03d}"
        (root / c / "chunks").mkdir(parents=True, exist_ok=True)
        (root / c / "chunks" / f"{c}.txt").write_text("\n".join(g) + "\n")
        names.append(c)
    return names


def cmd_sched(a):
    _load_globals()
    S = [r["scene_id"] for r in csv.DictReader(open(TSV), delimiter="\t") if not held(r["scene_id"]) and r["scene_id"] in _G["pk"]]
    avoid = defaultdict(list)
    for q in {"B": "A", "C": "AB"}.get(a.pas, ""):                       # later passes: other states than the earlier passes'
        for r in csv.DictReader(open(O / "eps" / f"episodes_{q}.csv")):
            avoid[r["scene"]].append(int(r["k0"]))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_sched_scene, [(s, a.pas, avoid[s]) for s in S], chunksize=4))
    eps = [e for _, ee, _, _ in res for e in ee]
    why = sum((w for _, _, _, w in res), Counter())
    (O / "eps").mkdir(parents=True, exist_ok=True)
    with open(O / "eps" / f"episodes_{a.pas}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(eps[0].keys()))
        w.writeheader()
        w.writerows(eps)
    js = {s: [[e["t0"], e["e0"], e["tp"]] for e in ee] for s, ee, _, _ in res if ee}
    (O / "eps" / f"eps_{a.pas}.json").write_text(json.dumps(js))
    scenes = [s for s in S if s in js]
    rec = dict(pass_=a.pas, scenes_eligible=len(S), scenes_with_episodes=len(scenes), episodes=len(eps), candidates=sum(n for _, _, n, _ in res),
               rows_expected=len(eps) * len(DROWS), why=dict(why), class_weight=_G["wcls"], class_share_moving_paih=_G["cls_share"],
               by_cls=dict(Counter(e["cls"] for e in eps)), by_type=dict(Counter(e["type"] for e in eps)),
               by_e0=dict(Counter(str(abs(e["e0"])) for e in eps)), by_psi=dict(Counter(str(abs(e["psi"])) for e in eps)))
    probe = None
    if a.pas == "A":
        many = [s for s in scenes if len(js[s]) >= 2]
        probe = sorted(many, key=lambda x: hashlib.sha256(f"rec-probe:{x}".encode()).hexdigest())[:11]
        (O / "eps" / "probe.txt").write_text("\n".join(probe) + "\n")
        (O / "eps" / "eps_Z.json").write_text(json.dumps({s: [[t0, 0.0, 0.0] for t0, _, _ in js[s]] for s in probe}))
        rec["chunks_Z"] = write_chunks("Z", [], probe)
        rec["probe"] = probe
    rec["chunks"] = write_chunks(a.pas, scenes, probe)
    dump(O / "eps" / f"sched_{a.pas}.json", rec)
    dump(RES / f"sched_{a.pas}.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("chunks", "chunks_Z")}, indent=1))


# ---------------------------------------------------------------- select
SEL_KEYS = ("k", "t0_us", "anchor", "log_t_us", "log_xyyaw", "s_ego", "t_event_us", "e0", "dpsi0", "v0")


def _sel_scene(arg):
    s, pas, prefix, eps = arg
    f = O / "rows" / pas / f"{s}.npz"
    if not f.exists():
        return s, None, Counter(missing_scene=len(eps) * len(DROWS))
    zz = np.load(f)
    z = {k: zz[k] for k in SEL_KEYS}
    path = G.Path(z["log_t_us"], z["log_xyyaw"])
    ta, Sa = RG.arc(z["log_t_us"], z["log_xyyaw"])
    ep_l = [(e["t0"], e["e0"], e["tp"]) for e in eps]
    pos = {int(k): i for i, k in enumerate(z["k"])}
    out, why = defaultdict(list), Counter()
    t_col = int(z["t_event_us"][0])                                          # collision_at_fault (onp_rows EVENTS order)
    for e in eps:
        for d in DROWS:
            i = pos.get(e["k0"] + int(d))
            if i is None:
                why["missing_row"] += 1
                continue
            t = int(z["t0_us"][i])
            if d == 0 and t != e["t0"]:
                why["clock"] += 1
                continue
            if t >= t_col:
                why["collision"] += 1
                continue
            Er, Dr, _ = RG.driven([t], ta, Sa, ep_l)
            tf = t + G.T_FUT
            fp, _, ok = path.at_time(tf)
            if not ok.all():
                why["log_short"] += 1
                continue
            El, Dl, _ = RG.merge(tf, ta, Sa, t, Er[0], Dr[0])
            fp = RG.offset(fp, El, Dl)
            A = z["anchor"][i:i + 1]
            h0 = path.at(np.float64(z["s_ego"][i]))[2]
            tw = 2 * np.pi * np.round((h0 - A[0, 2]) / (2 * np.pi))
            c, sn = np.cos(A[:, 2])[:, None], np.sin(A[:, 2])[:, None]                 # onp_rows.rig_n on one row
            dx, dy = fp[None, :, 0] - A[:, None, 0], fp[None, :, 1] - A[:, None, 1]
            lf = np.stack([c * dx + sn * dy, -sn * dx + c * dy, fp[None, :, 2] - A[:, None, 2]], -1)
            lf[..., 2] = lf[..., 2] - tw
            st = RG.offset(path.at_time(np.array([t]))[0], Er, Dr)[0]                 # the intended (driven) pose at t
            out["names"].append(f"on:{prefix}:{s}:{e['k0'] + int(d)}")
            out["fut"].append(np.asarray(lf[0], np.float32))
            for k, v in dict(scene=s, k=e["k0"] + int(d), k0=e["k0"], d=int(d), t=t, ep_e0=e["e0"], ep_psi=e["psi"], type=e["type"], cls=e["cls"],
                             E=float(Er[0]), D=float(Dr[0]), e_real=float(z["e0"][i]), dpsi_real=float(z["dpsi0"][i]), v0=float(z["v0"][i]),
                             start=to_rig(st, z["anchor"][i])).items():
                out[k].append(v)
    return s, out, why


def cmd_select(a):
    eps = defaultdict(list)
    src = "A" if a.pas == "Z" else a.pas
    for r in csv.DictReader(open(O / "eps" / f"episodes_{src}.csv")):
        eps[r["scene"]].append(dict(k0=int(r["k0"]), t0=int(r["t0"]), e0=float(r["e0"]), tp=float(r["tp"]), psi=float(r["psi"]), type=r["type"], cls=r["cls"]))
    if a.pas == "Z":                                                          # the probe at zero offset
        pz = set((O / "eps" / "probe.txt").read_text().split())
        eps = {s: [dict(e, e0=0.0, tp=0.0, psi=0.0, type="zero") for e in ee] for s, ee in eps.items() if s in pz}
    prefix = a.prefix or f"prec{a.pas.lower()}"
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_sel_scene, [(s, a.pas, prefix, ee) for s, ee in eps.items()], chunksize=2))
    why = sum((w for _, _, w in res), Counter())
    cat = defaultdict(list)
    for _, o, _ in res:
        if o:
            for k, v in o.items():
                cat[k] += v
    arr = {k: np.array(v) for k, v in cat.items()}
    arr["fut"] = np.stack(cat["fut"]).astype(np.float32)
    (O / "select").mkdir(parents=True, exist_ok=True)
    np.savez(O / "select" / f"{a.pas}.npz", **arr)
    rec = dict(pass_=a.pas, prefix=prefix, scenes=len(eps), scenes_with_rows=sum(1 for _, o, _ in res if o), rows=len(arr["names"]), dropped=dict(why))
    dump(O / "select" / f"{a.pas}.json", rec)
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- gates
def _pool(p):
    from jevdrive.tokens import open_front
    t = np.load(CR / f"on1_{p}" / "tab.npz")
    return t, open_front(CR / f"on1_{p}@warp" / "front.npy"), np.load(CR / f"on1_{p}@warp" / "teacher.npz")["out"]


def cmd_g1(a):
    keys = ("frames", "frame_ts", "slot", "valid", "cold_rows", "cold_cur", "t0_us", "k", "ego_in", "P", "V", "acc", "cmd", "anchor", "poses_model", "log_fut", "log_xyyaw", "log_t_us")
    probe = (O / "eps" / "probe.txt").read_text().split()
    files = {}
    for s in probe:
        f = O / "rows" / "Z" / f"{s}.npz"
        if not f.exists():
            files[s] = "missing"
            continue
        z, h = np.load(f), np.load(ROWS_H / f"{s}.npz")
        files[s] = {k: bool(z[k].shape == h[k].shape and np.array_equal(z[k], h[k])) for k in keys}
    tz, fz, ez = _pool(a.pool)
    tp, fp, ep = _pool("PAIH")
    ix = {x: i for i, x in enumerate(tp["names"].astype(str))}
    nz = tz["names"].astype(str)
    b = np.array([ix[x.replace(f":{x.split(':')[1]}:", ":paih:", 1)] for x in nz])
    fr = np.array([np.array_equal(np.asarray(fz[i]), np.asarray(fp[b[i]])) for i in range(len(nz))])
    sel = np.load(O / "select" / "Z.npz")
    res = dict(pool=a.pool, scenes=len(probe), rows=int(len(nz)), select_rows=int(len(sel["names"])), files=files,
               files_equal=all(isinstance(v, dict) and all(v.values()) for v in files.values()),
               ego_equal=bool(np.array_equal(tz["ego"], tp["ego"][b])), fut_equal=bool(np.array_equal(tz["fut"], tp["fut"][b])),
               front_equal_share=float(fr.mean()) if len(fr) else 0.0, teacher_equal=bool(np.array_equal(ez, ep[b])),
               fut_max_abs_diff=float(np.abs(tz["fut"] - tp["fut"][b]).max()) if len(b) else None)
    res["pass_"] = bool(res["files_equal"] and res["ego_equal"] and res["fut_equal"] and res["front_equal_share"] == 1.0 and res["teacher_equal"]
                        and len(nz) == len(sel["names"]) and len(nz) > 0)
    dump(RES / "g1.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=1))
    raise SystemExit(0 if res["pass_"] else 2)


def _meta(passes):
    M = defaultdict(list)
    for p in passes:
        z = np.load(O / "select" / f"{p}.npz")
        for k in z.files:
            M[k].append(z[k])
    return {k: np.concatenate(v) for k, v in M.items()}


def _three_point(P, v):
    """Path points P (9, 2) at 0.5 s steps and speeds v (9,) -> max |v^2 kappa| over the 7 interior points."""
    a, b, c = P[:-2], P[1:-1], P[2:]
    ab, bc, ac = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(c - a, axis=1)
    cr = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
    k = np.where(ab * bc * ac > 1e-6, 2 * cr / np.maximum(ab * bc * ac, 1e-9), 0.0)
    return float(np.abs(v[1:-1] ** 2 * k).max())


def _g2_scene(arg):
    s, rows, pas_of = arg
    road, sdf = _G["road"], _G["sdf"]
    H = np.load(ROWS_H / f"{s}.npz")
    hk = {int(k): i for i, k in enumerate(H["k"])}
    Hl = {k: H[k] for k in ("anchor", "log_fut", "log0", "log_t_us", "log_v", "t0_us")}
    Z = {p: np.load(O / "rows" / p / f"{s}.npz") for p in set(pas_of.values())}
    Zk = {p: {int(k): i for i, k in enumerate(z["k"])} for p, z in Z.items()}
    ZA = {p: z["anchor"] for p, z in Z.items()}
    M = interp_matrix()
    out = []
    for r in rows:
        p, k = pas_of[r["name"]], r["k"]
        i, j = Zk[p][k], hk[k]
        Ap, Ah = ZA[p][i], Hl["anchor"][j]
        fl = from_rig(np.r_[[[0.0, 0.0, 0.0]], r["fut"].astype(np.float64)], Ap)          # [ego, 8 label poses] in local
        q = to_rig(fl, Ah)
        q[:, 2] = np.unwrap(q[:, 2])
        dense = M @ q
        ix, okp, mlab_twin = road.get(f"on:paih:{s}:{k}", (-1, False, np.nan))
        m_rec = foot_min(sdf[ix], dense) if ix >= 0 else np.nan
        tt = Hl["t0_us"][j] + np.arange(9) * 500_000
        v = np.interp(tt, Hl["log_t_us"], Hl["log_v"].astype(np.float64))
        a_rec = _three_point(np.r_[r["start"][None, :2], r["fut"][:, :2]].astype(np.float64), v)
        a_tw = _three_point(np.r_[Hl["log0"][j][None, :2], Hl["log_fut"][j][:, :2]].astype(np.float64), v)
        out.append(dict(name=r["name"], okp=okp, m_twin=mlab_twin, m_rec=m_rec, a_rec=a_rec, a_twin=a_tw))
    return out


def _bysc(passes, pool):
    t = np.load(CR / f"on1_{pool}" / "tab.npz")
    names = set(t["names"].astype(str).tolist())
    M = _meta(passes)
    pas_of = {}
    for p in passes:
        for x in np.load(O / "select" / f"{p}.npz")["names"].astype(str):
            pas_of[x] = p
    by = defaultdict(list)
    for i, x in enumerate(M["names"].astype(str)):
        if x in names:
            by[str(M["scene"][i])].append(dict(name=x, k=int(M["k"][i]), fut=M["fut"][i], start=M["start"][i]))
    return M, by, pas_of


def cmd_g2(a):
    _load_globals()
    passes = a.pas.split(",")
    M, by, pas_of = _bysc(passes, a.pool)
    with ProcessPoolExecutor(a.jobs) as ex:
        R = [r for rr in ex.map(_g2_scene, [(s, rr, pas_of) for s, rr in by.items()], chunksize=2) for r in rr]
    okp = np.array([r["okp"] for r in R])
    mt, mr = np.array([r["m_twin"] for r in R]), np.array([r["m_rec"] for r in R])
    ar, at = np.array([r["a_rec"] for r in R]), np.array([r["a_twin"] for r in R])
    cond = okp & (mt >= 0)
    ins = float((mr[cond] >= 0).mean()) if cond.any() else float("nan")
    acc_ok = float((ar <= np.maximum(G2_ACC, at)).mean())
    res = dict(pool=a.pool, passes=passes, rows=len(R), rows_conditioned=int(cond.sum()), inside_share=ins, inside_share_all_raster=float((mr[okp] >= 0).mean()),
               twin_inside_share_all_raster=float((mt[okp] >= 0).mean()), acc_ok_share=acc_ok, acc_le4_share=float((ar <= G2_ACC).mean()),
               twin_acc_le4_share=float((at <= G2_ACC).mean()), a_rec_pct={q: float(np.percentile(ar, q)) for q in (50, 90, 99)},
               a_twin_pct={q: float(np.percentile(at, q)) for q in (50, 90, 99)}, m_rec_pct={q: float(np.nanpercentile(mr, q)) for q in (1, 5, 50)},
               line=G2_LINE)
    nm = np.array([r["name"] for r in R])
    pos = {x: i for i, x in enumerate(M["names"].astype(str))}
    ii = np.array([pos[x] for x in nm])
    res["by_type"] = {t: dict(rows=int((M["type"][ii] == t).sum()), inside=float((mr[cond & (M["type"][ii] == t)] >= 0).mean()) if (cond & (M["type"][ii] == t)).any() else None,
                              acc_ok=float((ar <= np.maximum(G2_ACC, at))[M["type"][ii] == t].mean())) for t in np.unique(M["type"][ii])}
    res["pass_"] = bool(ins >= G2_LINE and acc_ok >= G2_LINE)
    np.savez(O / "select" / f"g2_{a.pool}.npz", names=nm, okp=okp, m_twin=mt, m_rec=mr, a_rec=ar, a_twin=at)
    dump(RES / f"g2_{a.pool}.json", res)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["pass_"] else 2)


def _g3_scene(arg):
    s, rows, pas_of = arg
    H = np.load(ROWS_H / f"{s}.npz")
    hk = {int(k): i for i, k in enumerate(H["k"])}
    Hp, Ha = H["poses_model"], H["anchor"]
    Z = {p: np.load(O / "rows" / p / f"{s}.npz") for p in set(pas_of.values())}
    Zk = {p: {int(k): i for i, k in enumerate(z["k"])} for p, z in Z.items()}
    Zd = {p: (z["poses_model"], z["anchor"]) for p, z in Z.items()}
    path = G.Path(H["log_t_us"], H["log_xyyaw"])
    out = []
    for r in rows:
        p, k = pas_of[r["name"]], r["k"]
        pm, pa = Zd[p]
        i, j = Zk[p][k], hk[k]
        a_ = pm[i].astype(np.float64)
        b_ = Hp[j].astype(np.float64)
        wa, wb = from_rig(a_, pa[i]), from_rig(b_, Ha[j])
        la, lb = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[[[0, 0]], a_[:, :2]], axis=0), axis=1))][-1], \
            np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[[[0, 0]], b_[:, :2]], axis=0), axis=1))][-1]
        _, e_end = path.project(wa[7, :2])
        _, e_ego = path.project(pa[i][:2])
        out.append(dict(ego_diff=float(np.abs(a_[:, :2] - b_[:, :2]).max()), world_end=float(np.hypot(*(wa[7, :2] - wb[7, :2]))),
                        ratio=float(la / lb) if lb > 1.0 else np.nan, e_end=float(e_end), e_ego=float(e_ego)))
    return out


def cmd_g3(a):
    passes = a.pas.split(",")
    M, by, pas_of = _bysc(passes, a.pool)
    with ProcessPoolExecutor(a.jobs) as ex:
        R = [r for rr in ex.map(_g3_scene, [(s, rr, pas_of) for s, rr in by.items()], chunksize=2) for r in rr]
    r1 = json.loads(Path(a.r1).read_text())
    ed, we, ra = (np.array([r[k] for r in R]) for k in ("ego_diff", "world_end", "ratio"))
    ee, eg = np.array([r["e_end"] for r in R]), np.array([r["e_ego"] for r in R])
    big = np.abs(eg) >= 0.5
    res = dict(pool=a.pool, rows=len(R), r1=dict(model=r1["model"], wrong=r1["wrong"], pass_=r1["pass_"]),
               ego_frame_diff_median=float(np.median(ed)), world_end_median=float(np.median(we)), world_end_p90=float(np.percentile(we, 90)),
               len_ratio_median=float(np.nanmedian(ra)), len_ratio_p10_p90=[float(np.nanpercentile(ra, 10)), float(np.nanpercentile(ra, 90))],
               ego_offset_ge05_rows=int(big.sum()), recovering_share=float((np.abs(ee[big]) < np.abs(eg[big])).mean()) if big.any() else None,
               end_offset_over_ego_median=float(np.median(np.abs(ee[big]) / np.abs(eg[big]))) if big.any() else None)
    res["pass_"] = bool(r1["pass_"] and 0.8 <= res["len_ratio_median"] <= 1.25 and res["world_end_median"] <= 3.0)
    dump(RES / f"g3_{a.pool}.json", res)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["pass_"] else 2)


def cmd_gates(a):
    J = {g: json.loads((RES / f).read_text()) for g, f in (("G1", "g1.json"), ("G2", f"g2_{a.pool}.json"), ("G3", f"g3_{a.pool}.json")) if (RES / f).exists()}
    L = ["# Recovery rows: probe gates (plans/2026-10-21-recovery-rows.md)", ""]
    g1, g2, g3 = J.get("G1", {}), J.get("G2", {}), J.get("G3", {})
    if g1:
        L += [f"- **G1 zero perturbation = on1_PAIH** ({g1['scenes']} probe scenes, {g1['rows']} rows): row files equal {g1['files_equal']}, ego {g1['ego_equal']}, "
              f"label {g1['fut_equal']}, front tokens {g1['front_equal_share']:.4f}, teacher {g1['teacher_equal']} -> **{'pass' if g1['pass_'] else 'FAIL'}**"]
    if g2:
        L += [f"- **G2 label in raster / lateral acceleration** ({g2['rows']} rows): inside {g2['inside_share']:.4f} of {g2['rows_conditioned']} rows whose twin keeps its own "
              f"label inside (all raster rows {g2['inside_share_all_raster']:.4f}, plain twins {g2['twin_inside_share_all_raster']:.4f}); a_lat <= max(4, twin) "
              f"{g2['acc_ok_share']:.4f} (<= 4: {g2['acc_le4_share']:.4f}, twins {g2['twin_acc_le4_share']:.4f}; p50 / p90 / p99 {g2['a_rec_pct']}) "
              f"-> **{'pass' if g2['pass_'] else 'FAIL'}**"]
    if g3:
        L += [f"- **G3 render sanity** ({g3['rows']} rows): R1 median {g3['r1']['model']['median_m']:.4f} m, over {g3['r1']['model']['share_over']:.3f}, wrong model "
              f"{g3['r1']['wrong']['share_over']:.3f} ({'pass' if g3['r1']['pass_'] else 'FAIL'}); NTV0 plan vs its plan on the plain twin: 4 s length ratio median "
              f"{g3['len_ratio_median']:.3f}, world 4 s endpoint distance median {g3['world_end_median']:.2f} m (p90 {g3['world_end_p90']:.2f}), ego-frame max diff median "
              f"{g3['ego_frame_diff_median']:.2f} m; plan endpoint closer to the log than the ego on {g3['recovering_share']} of {g3['ego_offset_ge05_rows']} rows with "
              f"|offset| >= 0.5 m -> **{'pass' if g3['pass_'] else 'FAIL'}**"]
    ok = all(J.get(g, {}).get("pass_") for g in ("G1", "G2", "G3"))
    L += ["", f"**Probe gates: {'all pass' if ok else 'not all pass'}**"]
    (RES / "gates_probe.md").write_text("\n".join(L) + "\n")
    dump(RES / "gates_probe.json", dict(pass_=ok, **J))
    print("\n".join(L))
    raise SystemExit(0 if ok else 2)


# ---------------------------------------------------------------- weights / smoke / rows / verdict
def cmd_weights(a):
    B = np.load(W5 / "R5LC.npz")
    T = np.load(CR / a.pool / "tab.npz")
    names, n0 = T["names"].astype(str), len(B["names"])
    assert np.array_equal(names[:n0], B["names"].astype(str)), f"{a.pool}: prefix != R5LC's rows"
    part = T["pool"].astype(str)
    tail = np.isin(part[n0:], ["reca", "recb", "recc"])
    assert tail.all(), f"{a.pool}: tail parts {sorted(set(part[n0:]))}"
    nt = len(names) - n0
    w = np.zeros(len(names))
    if a.k == 0:
        w[:n0] = B["w"]
    else:
        w[:n0] = B["w"].astype(np.float64) * 64 / (64 + a.k)
        w[n0:] = a.k / (64 + a.k) / nt
    grp = np.concatenate([B["grp"].astype(str), np.full(nt, "rec")])
    assert abs(w.sum() - 1) < 1e-9, w.sum()
    gs = defaultdict(float)
    for g_, x in zip(grp, w):
        gs[str(g_)] += x
    out = D / "runs/round6/rec/w" / f"{a.out}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=names, w=w, grp=grp)
    rec = dict(base="R5LC", pool=a.pool, k_on=64, k_rec=a.k, k_all=64 + a.k, rec_rows=int(nt), rec_parts=dict(Counter(part[n0:].tolist())),
               rec_per_batch=a.k, group_share={k: float(v) for k, v in sorted(gs.items())}, out=str(out))
    dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_smoke(a):
    import r5
    r5.RES = RES
    r5.cmd_smoke(argparse.Namespace(tag=a.tag, arm="A", wfile=a.wfile, k_s3=a.k, suffix="", index=""))


def cmd_rows(a):
    passes = a.pas.split(",")
    M = _meta(passes)
    have = np.concatenate([np.load(CR / f"on1_{q}" / "tab.npz")["names"].astype(str) for q in a.pool.split(",")])
    keep = np.isin(M["names"].astype(str), have)
    M = {k: v[keep] for k, v in M.items()}
    d0 = M["d"] == 0
    err = np.abs(M["e_real"] - M["E"])
    rec = dict(pool=a.pool, rows=int(keep.sum()), episodes=int(d0.sum()), scenes=int(len(set(M["scene"].tolist()))),
               by_type=dict(Counter(M["type"][d0].tolist())), by_cls_rows=dict(Counter(M["cls"].tolist())), by_cls_episodes=dict(Counter(M["cls"][d0].tolist())),
               by_e0=dict(Counter(np.abs(M["ep_e0"][d0]).astype(str).tolist())), by_psi=dict(Counter(np.abs(M["ep_psi"][d0]).astype(str).tolist())),
               v0_pct={q: float(np.percentile(M["v0"][d0], q)) for q in (10, 50, 90)},
               realised_e_err_median=float(np.median(err)), realised_e_err_p90=float(np.percentile(err, 90)),
               realised_over_intended_median_d0=float(np.median(M["e_real"][d0 & (np.abs(M["E"]) >= 0.4)] / M["E"][d0 & (np.abs(M["E"]) >= 0.4)]))
               if (d0 & (np.abs(M["E"]) >= 0.4)).any() else None,
               dpsi_err_median_deg=float(np.degrees(np.median(np.abs(M["dpsi_real"] - np.arctan(M["D"]))))))
    vb = np.digitize(M["v0"][d0], [5, 10, 20])
    rec["by_speed_bin"] = {b: dict(episodes=int((vb == i).sum()), mean_abs_e0=float(np.abs(M["ep_e0"][d0][vb == i]).mean()) if (vb == i).any() else None,
                                   mean_abs_psi=float(np.abs(M["ep_psi"][d0][vb == i]).mean()) if (vb == i).any() else None)
                           for i, b in enumerate(("2-5", "5-10", "10-20", ">=20"))}
    for p in passes:
        rec[f"select_{p}"] = json.loads((O / "select" / f"{p}.json").read_text())
    L = [f"# Recovery rows: pools {a.pool}", "", f"{rec['rows']} rows, {rec['episodes']} episodes (rows k0 - 4 .. k0 + 4), {rec['scenes']} scenes.", "",
         "| | " + " | ".join(CLS) + " |", "|:--|--:|--:|--:|",
         "| episodes | " + " | ".join(str(rec["by_cls_episodes"].get(c, 0)) for c in CLS) + " |",
         "| rows | " + " | ".join(str(rec["by_cls_rows"].get(c, 0)) for c in CLS) + " |", "",
         f"Episodes by type: {rec['by_type']}; |e0| (m): {rec['by_e0']}; |psi0| (deg): {rec['by_psi']}.", "",
         "| log speed at t0 (m/s) | episodes | mean abs e0 (m) | mean abs psi0 (deg) |", "|:--|--:|--:|--:|"]
    L += [f"| {b} | {v['episodes']} | {v['mean_abs_e0'] if v['mean_abs_e0'] is None else round(v['mean_abs_e0'], 2)} | "
          f"{v['mean_abs_psi'] if v['mean_abs_psi'] is None else round(v['mean_abs_psi'], 1)} |" for b, v in rec["by_speed_bin"].items()]
    L += ["", f"Realised vs intended offset (the MPC tracking the perturbed reference): |e_real - E| median {rec['realised_e_err_median']:.3f} m, p90 "
          f"{rec['realised_e_err_p90']:.3f} m; at t0 with |E| >= 0.4 m e_real / E median {rec['realised_over_intended_median_d0']}; heading error median "
          f"{rec['dpsi_err_median_deg']:.2f} deg.", "",
          "Dropped at select: " + "; ".join(f"{p}: {rec[f'select_{p}']['dropped']}" for p in passes)]
    (RES / "rows.md").write_text("\n".join(L) + "\n")
    dump(RES / "rows.json", rec)
    print("\n".join(L))


def cmd_verdict(a):
    J = json.loads(Path(a.json).read_text())
    Q = json.loads(Path(a.pcs).read_text())
    R, P = J["reads"], J["pairs"]
    lat = lambda k: R[k]["z_offroad"] + R[k]["z_zero-other"]  # noqa: E731
    cap = lat("NTV0-ag") + 3
    rows, ok = [], True
    for s in ("0", "1"):
        k, q = f"REC-s{s}-ag", Q[f"R6REC-s{s}+AG"]
        d = P[f"{k}:NTV0-ag"]["mean"]
        b, l_, r_ = d["mean"] >= 0, lat(k) <= cap, q["diff_ci"][1] >= 0
        ok &= b and l_ and r_
        rows.append(f"| {k} | {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'yes' if b else 'no'} | {lat(k)} (offroad {R[k]['z_offroad']}, "
                    f"zero-other {R[k]['z_zero-other']}) | {'yes' if l_ else 'no'} | {q['diff_median']:+.0f} [{q['diff_ci'][0]:+.0f}, {q['diff_ci'][1]:+.0f}] | "
                    f"{'yes' if r_ else 'no'} |")
    L = ["", "## Registered rule (plans/2026-10-21-recovery-rows.md)", "",
         f"adopt-candidate = both REC seeds: mean - NTV0 >= 0 (paired point estimate), offroad + zero-other zeros <= NTV0's + 3 = {cap}, local PCS ruler "
         "difference to NTV0 with a 95 % CI upper bound >= 0.", "",
         "| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | ruler - NTV0 [95 % CI] | CI upper >= 0 |", "|:--|--:|:-:|--:|:-:|--:|:-:|",
         *rows, "", f"**REC: {'adopt-candidate' if ok else 'not an adopt-candidate'}.**"]
    with open(a.md, "a") as f:
        f.write("\n".join(L) + "\n")
    J["verdict"] = dict(adopt_candidate=bool(ok), cap=cap)
    Path(a.json).write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    nj = max(1, (os.cpu_count() or 8) // 2)
    try:
        from jevdrive.common import n_cpus
        nj = n_cpus()
    except Exception:
        pass
    p = sp.add_parser("sched"); p.add_argument("--pass", dest="pas", required=True, choices=("A", "B", "C")); p.add_argument("--jobs", type=int, default=nj)
    p = sp.add_parser("select"); p.add_argument("--pass", dest="pas", required=True, choices=("A", "B", "C", "Z")); p.add_argument("--prefix", default="")
    p.add_argument("--jobs", type=int, default=nj)
    p = sp.add_parser("g1"); p.add_argument("--pool", required=True)
    p = sp.add_parser("g2"); p.add_argument("--pool", required=True); p.add_argument("--pass", dest="pas", required=True); p.add_argument("--jobs", type=int, default=nj)
    p = sp.add_parser("g3"); p.add_argument("--pool", required=True); p.add_argument("--pass", dest="pas", required=True); p.add_argument("--r1", required=True)
    p.add_argument("--jobs", type=int, default=nj)
    p = sp.add_parser("gates"); p.add_argument("--pool", required=True)
    p = sp.add_parser("weights"); p.add_argument("--pool", required=True); p.add_argument("--k", type=int, required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("smoke"); p.add_argument("--tag", required=True); p.add_argument("--wfile", default="R6REC"); p.add_argument("--k", type=int, default=8)
    p = sp.add_parser("rows"); p.add_argument("--pool", required=True); p.add_argument("--pass", dest="pas", required=True)
    p = sp.add_parser("verdict"); p.add_argument("--json", required=True); p.add_argument("--md", required=True); p.add_argument("--pcs", required=True)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
