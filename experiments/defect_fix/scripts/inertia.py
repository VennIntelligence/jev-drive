#!/usr/bin/env python3
"""INERTIA (plans/2026-10-20-inertia-prereg.md, results/inertia/): does the plan's stop / turn / open-road speed follow the fed ego speed
or the cue? No training, no rollout. Box, repo root.

  $TR inertia.py recede                (CPU) supplement 1: the planned stop station of A0's stored 10 Hz plans along the approach of the
                                       d381 overrun scenes and the stop controls -> $O/recede.parquet, results/inertia/recede.{csv,json}
  $TR inertia.py cf [--every 5]        (GPU pool job) counterfactual forwards on the stored P160 own-state rows (RSG-L18-s0, PAI practice
                                       scenes): frames re-encoded from the row files, the fed row and the logged route memory -> $O/cf.parquet,
                                       $O/rows.parquet, $O/gate.json
  $TR inertia.py report                (CPU) the registered tables and verdicts -> results/inertia/report.{md,json}

Arms (the prereg): R_f (fed speed history x f: vx vy, ax, the 3 past poses re-spaced along the own history polyline, warp_loop.respace),
Z (the 12 speed entries of vhist zeroed), W_f (the 8 slots taken from the rendered frames at t0 + f x SLOT_T: 0.1 / 0.3 s spacing for
f = 0.5 / 1.5, exact at 10 Hz), B_f = W_f + R_f, F = R_1.0 (re-encoded from the frames).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/alpasim/lib"), str(REPO / "lib"), str(REPO / "experiments/route_follow/scripts")]
import stoppt as SP  # noqa: E402
import warp_loop as WL  # noqa: E402

DATA = SP.DATA
O0 = DATA / "runs/defect_fix/inertia"
O = O0 / ("smoke" if os.environ.get("O_SMOKE") else "")
RES = REPO / "experiments/defect_fix/results/inertia"
ROWS = DATA / "runs/practice_pai/rows/PPAI160"
GAP = DATA / "runs/route_follow/pai_gap"
TAGS = ("NTV0-L18-s0", "RB0-F-s0", "RSG-L18-s0")
RF = (0.5, 0.6, 0.8, 1.2, 1.4, 1.5)
WF = (0.5, 1.5)
ARMS = ("F",) + tuple(f"R{f}" for f in RF) + ("Z",) + tuple(f"{x}{f}" for f in WF for x in "WB")
EGO_SPEED = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18]       # vhist.EGO_SPEED (not imported: vhist reads the environment at import)
SLOT_T = np.round(np.arange(-7, 1) * 0.2, 3)
V_STOP, TOL_US = 0.5, 30_000


# ---------------------------------------------------------------- plan metrics
def metrics(P):
    """poses (..., 8, 3) rear axle at 0.5 .. 4 s -> dict of (...,) arrays: S (stop station or the 4 s arc), has (plan contains a stop),
    A4, v24 (2-4 s mean speed), head20 (deg, plan heading at 20 m of arc; nan if the plan is shorter)."""
    P = np.asarray(P, np.float64)
    xy = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P[..., :2]], -2)
    seg = np.linalg.norm(np.diff(xy, axis=-2), axis=-1)
    c = np.concatenate([np.zeros(seg.shape[:-1] + (1,)), np.cumsum(seg, -1)], -1)
    slow = seg / 0.5 < V_STOP
    has = slow.any(-1)
    i = np.asarray(np.argmax(slow, -1))
    S = np.where(has, np.take_along_axis(c, i[..., None], -1)[..., 0], c[..., -1])
    yaw = np.concatenate([np.zeros(P.shape[:-2] + (1,)), np.unwrap(P[..., 2], axis=-1)], -1)
    flat_c, flat_y = c.reshape(-1, 9), yaw.reshape(-1, 9)
    h = np.array([np.interp(20.0, cc, yy) if cc[-1] >= 20.0 and np.all(np.diff(cc) > 1e-6) else np.nan for cc, yy in zip(flat_c, flat_y)])
    return dict(S=S, has=has, A4=c[..., -1], v24=(c[..., -1] - c[..., 4]) / 2.0, head20=np.degrees(h).reshape(S.shape))


# ---------------------------------------------------------------- supplement 1: receding vs world-fixed planned stop (A0, stored)
def cmd_recede(a):
    import pandas as pd
    import pai_gap as PG
    sc = pd.read_csv(GAP / "scenes.csv").set_index("scene")
    over = sorted(sc.index[sc["sub"] == "cor_route_end"])
    ctrl1 = sorted(sc.index[(sc.h_end_still == True) & (sc.rest_lead == False) & (sc.score >= 1.0)])  # noqa: E712  (supplement 1's filter)
    ctrl = sorted(x.strip() for x in open(a.controls)) if a.controls else []                    # supplement 3: A0 rests within [-8, +4] m of H
    cs = set(ctrl)
    grp_of = lambda s: "overrun" if s in over else "control" if s in cs else "control_v1"  # noqa: E731
    ctrl = sorted(cs | set(ctrl1))
    G = pd.read_parquet(SP.S_ / "gate/a0.parquet")
    G = G[G.scene.isin(over + ctrl)]
    D = PG.drive(SP.A0)
    rows = []
    for s, g in G.groupby("scene"):
        recs = {int(r["k"]): r for r in D.get(s, [])}
        g = g.sort_values("k")
        first = g[np.isfinite(g.rest) & (g.rest <= 60)]
        if not len(first):
            rows.append(dict(scene=s, group=grp_of(s), k=-1))
            continue
        H = float(first.s_e.iloc[0] + first.rest.iloc[0])
        g = g[g.s_e < g.s_end - 0.5]                         # supplement 2: past the log end s_e is clamped there
        reach = g.k[g.s_e >= H]
        if len(reach):
            g = g[g.k <= reach.min()]                         # the approach ends at the first decision at or past H
        for r in g[(g.s_e >= H - 60) & (g.s_e < H + 2)].itertuples():
            d = recs.get(int(r.k))
            if d is None:
                continue
            P = np.asarray(d.get("poses_model") or d["poses"], np.float64)
            m = metrics(P)
            v_end = float(np.linalg.norm(P[-1, :2] - P[-2, :2]) / 0.5)
            rows.append(dict(scene=s, group=grp_of(s), k=int(r.k), v0=float(r.v0), s_e=float(r.s_e), H=H,
                             has=bool(m["has"]), stop_st=float(r.s_e + m["S"]) if m["has"] else np.nan, A4=float(m["A4"]),
                             E=float(r.s_e + m["A4"]), v_end=v_end))
    df = pd.DataFrame(rows)
    O.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(O / "recede.parquet")
    per = []
    for (s, grp), g in df[df.k >= 0].groupby(["scene", "group"]):
        g = g.sort_values("k")
        r = dict(scene=s, group=grp, n=len(g), share=float(g.has.mean()), H=float(g.H.iloc[0]))
        st = g[g.has]
        if len(g) < 10:
            r["cls"] = "too short"
        elif r["share"] < 0.10:
            r["cls"] = "never"
        elif len(st) >= 5 and np.ptp(st.s_e) >= 5:
            b = float(np.polyfit(st.s_e, st.stop_st, 1)[0])
            r |= dict(slope=b, cls="receding" if b >= 0.7 else "fixed" if b <= 0.3 else "mixed")
        else:
            r["cls"] = "not judged"
        if len(st):
            last = st.k.max()
            after = g[(g.k > last) & (g.s_e < r["H"])]
            r |= dict(ahead_med=float((st.stop_st - st.s_e).median()), rel_H_med=float((st.stop_st - r["H"]).median()),
                      letgo=bool((~after.has).sum() >= 5), letgo_dist=float(r["H"] - st[st.k == last].s_e.iloc[0]))
        ap = g[(g.s_e >= r["H"] - 40) & (g.s_e < r["H"])]           # supplement 2, exploratory: the plan's 4 s end station
        if len(ap) >= 10 and np.ptp(ap.s_e) >= 5:
            r |= dict(E_slope=float(np.polyfit(ap.s_e, ap.E, 1)[0]), E_minus_H_med=float((ap.E - r["H"]).median()), v_end_med=float(ap.v_end.median()),
                      v0_med=float(ap.v0.median()))
            r["E_cls"] = "end recedes" if r["E_slope"] >= 0.7 else "end pinned" if r["E_slope"] <= 0.3 else "between"
        per.append(r)
    P = pd.DataFrame(per).sort_values(["group", "scene"])
    P.to_csv(RES / "recede.csv", index=False, float_format="%.3f")
    out = {grp: dict(scenes=int(len(g)), **{c: int((g.cls == c).sum()) for c in ("receding", "fixed", "mixed", "never", "not judged", "too short")},
                     letgo=int(g.letgo.fillna(False).sum()),
                     explore={c: int((g.get("E_cls", pd.Series(dtype=str)) == c).sum()) for c in ("end recedes", "end pinned", "between")})
           for grp, g in P.groupby("group")}
    d2 = df[df.k >= 0].assign(dH=lambda x: x.H - x.s_e, EmH=lambda x: x.E - x.H)
    d2["bin"] = pd.cut(d2.dH, [0, 5, 10, 20, 30, 40, 60], include_lowest=True).astype(str)
    out["bins"] = {grp: x.groupby("bin").agg(n=("k", "size"), v0=("v0", "median"), A4=("A4", "median"), E_minus_H=("EmH", "median"),
                                             v_end=("v_end", "median")).round(2).reset_index().to_dict("records") for grp, x in d2.groupby("group")}
    # per-scene series (decision index, ego station, planned stop station, H) for the overrun scenes
    ser = {s: g.sort_values("k")[["k", "s_e", "stop_st", "E", "v_end", "H"]].round(2).values.tolist() for s, g in df[df.k >= 0].groupby("scene")}
    json.dump(dict(summary=out, series=ser), open(RES / "recede.json", "w"), indent=0)
    print(P.to_string(index=False))
    print(json.dumps(out, indent=1))


def gate_ok(a, b):
    """supplement 2's G0 for one row: every coordinate within max(0.06 m, 2 fp16 steps at its magnitude)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    step = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(b), 1e-3))) - 10)
    return bool((np.abs(a - b) <= np.maximum(0.06, 2 * step)).all())


# ---------------------------------------------------------------- cf: rows and labels
def _scene_rows(job):
    """One P160 row file -> the rows kept (k = 0 mod every, the prereg filters) with labels, window inputs and slot frame indices."""
    f, every = job
    zz = np.load(f, allow_pickle=True)
    z = {k: zz[k] for k in ("scene", "session", "run", "k", "t0_us", "frame_ts", "slot", "valid", "ego_in", "v0", "e0", "after_event", "after_end",
                            "lht", "cam_t", "log_t_us", "log_s", "log_xyyaw", "s_ego", "path", "path_ok", "box", "box_ok", "has_stop", "stop_s", "wl_s",
                            "poses_model")}
    ft = z["frame_ts"].astype(np.int64)
    T, S = z["log_t_us"] * 1e-6, np.maximum.accumulate(z["log_s"].astype(np.float64))
    yaw = np.unwrap(z["log_xyyaw"][:, 2])
    hd = lambda s0, d: float(np.degrees(np.interp(s0 + d, S, yaw) - np.interp(s0, S, yaw)))  # noqa: E731
    out = []
    for n, k in enumerate(z["k"]):
        v0 = float(z["v0"][n])
        if k % every or z["after_event"][n] or z["after_end"][n] or (z["slot"][n] < 0).any() or not z["valid"][n].all() or v0 < 2.0 \
                or not abs(float(np.nan_to_num(z["e0"][n], nan=99))) <= 3.0:
            continue
        t0 = int(z["t0_us"][n])
        sl = {1.0: z["slot"][n].astype(int)}
        for fw in WF:
            tt = t0 + np.round(fw * SLOT_T * 1e6).astype(np.int64)
            j = np.abs(ft[None] - tt[:, None]).argmin(1)
            sl[fw] = j if (np.abs(ft[j] - tt) <= TOL_US).all() else None
        Pp, pok = z["path"][n].astype(np.float64), z["path_ok"][n]
        Pp = np.where(pok[:, None], Pp, 1e6)
        B, bok = z["box"][n, :, :, :2].reshape(-1, 2).astype(np.float64), z["box_ok"][n].reshape(-1)
        s_e = float(z["s_ego"][n])
        lab = SP.label_row(t0 * 1e-6, s_e, T, S, v0, lambda rest: SP.lead_flag(rest, Pp, 20, B, bok))
        ws = np.concatenate([np.where(z["has_stop"], z["stop_s"][n], np.nan), z["wl_s"][n]]).astype(np.float64)
        line = bool(np.isfinite(lab["rest"]) and (np.abs(ws - lab["rest"] + 5) <= 10).any())
        obj40 = SP.lead_flag(0.0, Pp, 20, B, bok) or SP.lead_flag(20.0, Pp, 20, B, bok)
        hs = lab["hs8"]
        out.append(dict(scene=str(z["scene"]), session=str(z["session"]), run=str(z["run"]), k=int(k), t0=t0, v0=v0, lht=bool(z["lht"]),
                        cam_t=z["cam_t"].astype(np.float64), ego=z["ego_in"][n].astype(np.float32), poses_log=z["poses_model"][n].astype(np.float32),
                        sl=sl, rest=float(lab["rest"]), lead=bool(lab["lead"]), line=line, obj40=bool(obj40),
                        h40=hd(s_e, 40.0), h60=hd(s_e, 60.0), hv24=float((hs[7] - hs[3]) / 2.0) if np.isfinite(hs).all() else np.nan, file=f))
    return out


def window(r):
    rest, v0 = r["rest"], r["v0"]
    stop = np.isfinite(rest) and 8.0 <= rest <= 40.0 and not r["lead"]
    ns = (not np.isfinite(rest) or rest > 8.0 * v0) and not r["obj40"]
    turn = abs(r["h40"]) >= 30.0 and (not np.isfinite(rest) or rest > 40.0) and not r["obj40"]
    return "STOP" if stop else "TURN" if turn else "OPEN" if ns and abs(r["h60"]) < 10.0 else "NS" if ns else ""


def scale_row(ego, f):
    """The fed row with its speed history x f (the prereg R_f)."""
    if f == 1.0:
        return ego.copy()
    P = ego[8:20].reshape(4, 3).astype(np.float64) * [10.0, 10.0, 1.0]
    a = np.cumsum(np.linalg.norm(np.diff(P[::-1, :2], axis=0), axis=1))
    Pp = WL.respace(P, f * a)
    r = ego.copy()
    r[4:6] = ego[4:6] * f
    r[6] = ego[6] * f
    r[8:17] = (Pp[:3] / [10.0, 10.0, 1.0]).reshape(9)
    return r.astype(np.float32)


def rmem_of(runs):
    """{(session, k): rmem (200,)} from the rollouts' drive.jsonl."""
    out = {}
    for run in runs:
        for line in open(Path(run) / "sim/driver-logs/drive.jsonl"):
            if '"rmem"' not in line:
                continue
            d = json.loads(line)
            if d.get("kind") == "drive" and d.get("infer"):
                out[(d["session"], int(d["k"]))] = np.asarray(d["rmem"], np.float32)
    return out


def cmd_cf(a):
    import pandas as pd
    import torch
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.run import Run
    O.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "inertia-cf", config=vars(a)) as run:
        files = sorted(glob.glob(str(Path(a.rows) / "*.npz")))[:a.limit or None]
        with ProcessPoolExecutor(a.workers) as ex:
            R = [r for out in ex.map(_scene_rows, [(f, a.every) for f in files], chunksize=1) for r in out]
        for r in R:
            r["win"] = window(r)
        R = [r for r in R if r["win"]]
        rm = rmem_of(sorted({r["run"] for r in R}))
        R = [r for r in R if (r["session"], r["k"]) in rm]
        run.info(f"{len(R)} rows in {len({r['scene'] for r in R})} scenes: " + str(pd.Series([r["win"] for r in R]).value_counts().to_dict()))
        tags = tuple(a.tags.split(","))
        assert a.gate_tag in tags, (a.gate_tag, tags)
        cores = {t: SC.Core(t, "cuda", "backwarp") for t in tags}
        enc_core = cores[tags[0]]
        net, A = enc_core.model.net, enc_core.A

        def encode(frames, slots):
            """frames (n, 2, 6, 128, 256) uint8; slots (8,) indices -> H (8, 32, 512) on the card (prev of slot 0 = zeros, as pai_core.plan)."""
            cur = torch.from_numpy(frames[slots]).to(enc_core.dev)
            prev = torch.cat([torch.zeros_like(cur[:1]), cur[:-1]])
            with torch.no_grad():
                return net.run_batched(A.vision_feeds(prev, cur), ["view_39"])["view_39"].reshape(8, *A.H_SHAPE)

        recs, meta = [], []
        by = defaultdict(list)
        for i, r in enumerate(R):
            by[r["file"]].append(i)
        for nf, (f, idx) in enumerate(sorted(by.items())):
            fr = np.load(f, allow_pickle=True)["frames"]
            for i in idx:
                r = R[i]
                Hs = {fw: encode(fr, r["sl"][fw]) for fw in (1.0,) + WF if r["sl"][fw] is not None}
                feeds = {"F": (1.0, r["ego"])} | {f"R{f}": (1.0, scale_row(r["ego"], f)) for f in RF}
                ez = r["ego"].copy()
                ez[EGO_SPEED] = 0.0
                feeds["Z"] = (1.0, ez)
                for fw in WF:
                    if fw in Hs:
                        feeds[f"W{fw}"] = (fw, r["ego"])
                        feeds[f"B{fw}"] = (fw, scale_row(r["ego"], fw))
                names = list(feeds)
                H = torch.stack([Hs[feeds[n][0]] for n in names])
                tc = torch.tensor([[0.0, 1.0] if r["lht"] else [1.0, 0.0]] * len(names), device=enc_core.dev)
                for t, core in cores.items():
                    row = np.stack([np.r_[feeds[n][1], rm[(r["session"], r["k"])]] if core.rm else feeds[n][1] for n in names]).astype(np.float32)
                    with torch.no_grad():
                        o = core.model(H, torch.from_numpy(row).to(core.dev), tc)   # P0: no adapter, the row is not read
                    mu = (o[0] if isinstance(o, tuple) else o).float()[:, core.pi].reshape(len(names), 33, 15).cpu().numpy()
                    for n, m in zip(names, mu):
                        p = I.to_rear(m[:, 0:3], m[:, 11], I.T_IDXS, r["cam_t"][:2], Z.T_OUT, "lever")[:, :3]
                        recs.append(dict(i=i, tag=t, arm=n, P=np.asarray(p, np.float32).ravel()))
                meta.append({k: r[k] for k in ("scene", "k", "t0", "v0", "win", "rest", "lead", "line", "obj40", "h40", "h60", "hv24")}
                            | dict(i=i, w_ok=all(r["sl"][fw] is not None for fw in WF), gate=float(np.abs(
                                np.asarray(recs[-len(names) * len(tags) + len(names) * tags.index(a.gate_tag) + names.index("F")]["P"]).reshape(8, 3)[:, :2]
                                - r["poses_log"][:, :2]).max()), gate2=bool(gate_ok(
                                np.asarray(recs[-len(names) * len(tags) + len(names) * tags.index(a.gate_tag) + names.index("F")]["P"]).reshape(8, 3)[:, :2],
                                r["poses_log"][:, :2]))))
            if nf % 20 == 0:
                run.info(f"{nf + 1} / {len(by)} scenes, {len(meta)} rows")
        M = pd.DataFrame(meta)
        C = pd.DataFrame(recs)
        Pm = np.stack(C.P.values).reshape(-1, 8, 3)
        for k, v in metrics(Pm).items():
            C[k] = v
        C.drop(columns="P").to_parquet(O / "cf.parquet")
        np.save(O / "cf_poses.npy", Pm.astype(np.float32))
        M.to_parquet(O / "rows.parquet")
        g = dict(gate_tag=a.gate_tag, rows_dir=str(a.rows), rows=len(M), gate_p99=float(M.gate.quantile(0.99)), gate_max=float(M.gate.max()), gate2_share=float(M.gate2.mean()),
                 pass_=bool(M.gate2.mean() >= 0.99 and M.gate.max() <= 0.15), w_ok=float(M.w_ok.mean()))   # supplement 2's G0
        json.dump(g, open(O / "gate.json", "w"), indent=1)
        run.summary.update(g)
        run.info(f"gate {g}")



# ---------------------------------------------------------------- stage 1 (supplement 3): scene list and the parity check
S1 = O0 / "s1"
A0R = DATA / "runs/defect_fix/pai_accgap/NTV0-L18-s0"


def cmd_select(a):
    """The stage-1 scene list: d381 overrun (cor_route_end) + straight-at-turn, stop controls (A0 1.0, the human stops, no lead at the first
    approach decision, A0 rests >= 2 s within [-8, +4] m of the human's stop point), turn controls (A0 1.0, stratum turn, |turn| >= 45 deg)."""
    import pandas as pd
    import pai_gap as PG
    sc = pd.read_csv(GAP / "scenes.csv").set_index("scene")
    G = pd.read_parquet(SP.S_ / "gate/a0.parquet")
    D = PG.drive(A0R)
    stop = []
    for s in sc.index[sc.score >= 1.0]:
        g = G[G.scene == s].sort_values("k")
        g = g[g.s_e < g.s_end - 0.5]
        f = g[np.isfinite(g.rest) & (g.rest <= 60)]
        if not len(f) or s not in D or bool(f.lead.iloc[0]):
            continue
        H = float(f.s_e.iloc[0] + f.rest.iloc[0])
        v = np.array([np.hypot(*r["ego"][4:6]) * 10 for r in D[s]])
        k = np.array([r["k"] for r in D[s]])
        se = dict(zip(g.k, g.s_e))
        for i in np.flatnonzero((k >= int(f.k.iloc[0])) & (v < 0.5)):
            if len(v[i:i + 20]) == 20 and (v[i:i + 20] < 0.5).all():
                prev = [se[x] for x in k[:i] if x in se]
                if prev and -8.0 <= prev[-1] - H <= 4.0:
                    stop.append(s)
                break
    groups = {"overrun": sorted(sc.index[sc["sub"] == "cor_route_end"]), "turn_fail": sorted(sc.index[sc["sub"] == "cor_straight_at_turn"]),
              "stop_ctrl": sorted(stop), "turn_ctrl": sorted(sc.index[(sc.score >= 1.0) & (sc.stratum == "turn") & (sc.turn_deg.abs() >= 45)])}
    S1.mkdir(parents=True, exist_ok=True)
    rows = [(s, g) for g, v in groups.items() for s in v]
    pd.DataFrame(rows, columns=["scene", "group"]).to_csv(RES / "scenes_s1.csv", index=False)
    (S1 / "scenes.txt").write_text("".join(f"{s}\n" for s, _ in rows))
    (S1 / "stop_ctrl.txt").write_text("".join(f"{s}\n" for s in groups["stop_ctrl"]))
    print({g: len(v) for g, v in groups.items()}, len(rows))


def cmd_parity(a):
    """A re-roll against A0 on its scenes: score and the model plan per decision (k) -> stdout JSON; exit 2 unless every scene passes
    (same score within 1e-9 and the plan within 0.15 m over the first 50 policy decisions)."""
    import pai_gap as PG
    x, _ = PG.summaries(A0R)
    y, _ = PG.summaries(Path(a.read))
    Dx, Dy = PG.drive(A0R), PG.drive(Path(a.read))
    out, ok = {}, True
    for s in sorted(set(y) & set(x)):
        kx = {int(r["k"]): np.asarray(r.get("poses_model") or r["poses"]) for r in Dx.get(s, [])}
        ky = {int(r["k"]): np.asarray(r.get("poses_model") or r["poses"]) for r in Dy.get(s, [])}
        ks = sorted(set(kx) & set(ky))
        d = [float(np.abs(kx[k][:, :2] - ky[k][:, :2]).max()) for k in ks]
        first = next((k for k, e in zip(ks, d) if e > 0.15), None)
        sx, sy = float(x[s]["score"] or 0), float(y[s]["score"] or 0)
        pas = abs(sx - sy) < 1e-9 and (first is None or first >= (ks[0] + 50 if ks else 0))
        ok &= pas
        out[s] = dict(score_a0=sx, score=sy, decisions=len(ks), max_d=max(d) if d else None, first_k_over=first, pass_=pas)
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok and out else 2)


# ---------------------------------------------------------------- report
def boot(fn, groups, B=2000, seed=0):
    return WL.boot(fn, groups, B, seed)


def cmd_report(a):
    import pandas as pd
    g = json.load(open(O / "gate.json"))
    assert g["pass_"], f"G0 failed: {g}"
    M = pd.read_parquet(O / "rows.parquet").set_index("i")
    C = pd.read_parquet(O / "cf.parquet")
    W = {c: C.pivot_table(index=["tag", "i"], columns="arm", values=c) for c in ("S", "has", "A4", "v24", "head20")}
    out, lines = dict(gate=g, cells=[]), []
    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
    TG = [t for t in TAGS + ("P0",) if t in set(C.tag)] + sorted(set(C.tag) - set(TAGS) - {"P0"})
    for tag in TG:
        for win in ("STOP", "STOP-line", "NS", "TURN", "OPEN"):
            m = M[M.win.eq("STOP") & M.line] if win == "STOP-line" else M[M.win.eq(win)]
            if not len(m):
                continue
            ids, sc, v0 = m.index.values, m.scene.values, m.v0.values
            wok = m.w_ok.values
            X = {c: W[c].loc[tag].reindex(ids) for c in W}
            cell = dict(tag=tag, win=win, rows=len(m), scenes=int(m.scene.nunique()), rows_w=int(wok.sum()), scenes_w=int(m.scene[wok].nunique()))
            cell["judge"] = cell["scenes"] >= 15
            S, H, A4, v24, h20 = (X[c] for c in ("S", "has", "A4", "v24", "head20"))

            def lin(D, lo, hi, scale, mask=None):
                d = (D[hi] - D[lo]).values * scale
                mk = np.isfinite(d) if mask is None else (mask & np.isfinite(d))
                return lambda i: float(np.mean(d[i][mk[i]])) if mk[i].any() else np.nan
            # per +20 %: R central difference over 0.8 / 1.2; W / B over 0.5 / 1.5 x 0.2; R at 0.5 / 1.5 for the channel share
            cell["dS_R"] = boot(lin(S, "R0.8", "R1.2", 0.5), sc)
            cell["dS_R50"] = boot(lin(S, "R0.5", "R1.5", 0.2, wok), sc)
            for x in "WB":
                cell[f"dS_{x}"] = boot(lin(S, f"{x}0.5", f"{x}1.5", 0.2, wok), sc)
            cell["dHas_R"] = boot(lin(H.astype(float), "R0.8", "R1.2", 0.5), sc)
            cell["dHas_B"] = boot(lin(H.astype(float), "B0.5", "B1.5", 0.2, wok), sc)
            cell["has_F"] = float(H["F"].mean())
            cell["has_R"] = {f: float(H[f"R{f}" if f != 1.0 else "F"].mean()) for f in (0.6, 0.8, 1.0, 1.2, 1.4)}
            cell["has_Z"] = float(H["Z"].mean())
            cell["S_F_minus_rest"] = float(np.median(S["F"].values - m.rest.values)) if win.startswith("STOP") else None
            cell["S_R"] = {f: float(np.mean(S[f"R{f}" if f != 1.0 else "F"])) for f in (0.6, 0.8, 1.0, 1.2, 1.4)}
            cell["S_Z_minus_F"] = float(np.mean(S["Z"] - S["F"]))
            den = lambda lo, hi: 4.0 * (hi - lo) * v0  # noqa: E731
            for x, lo, hi, mk in (("R", 0.8, 1.2, None), ("W", 0.5, 1.5, wok), ("B", 0.5, 1.5, wok), ("R50", 0.5, 1.5, wok)):
                xx = "R" if x == "R50" else x
                num = (A4[f"{xx}{hi}"] - A4[f"{xx}{lo}"]).values
                dd = den(lo, hi)
                k_ = np.isfinite(num) if mk is None else (mk & np.isfinite(num))
                cell[f"G_{x}"] = boot(lambda i: float(num[i][k_[i]].sum() / dd[i][k_[i]].sum()), sc)
            for x, lo, hi, mk in (("R", 0.8, 1.2, None), ("B", 0.5, 1.5, wok)):
                num = (v24[f"{x}{hi}"] - v24[f"{x}{lo}"]).values
                dd = (hi - lo) * v0
                k_ = np.isfinite(num) if mk is None else (mk & np.isfinite(num))
                cell[f"slope_v24_{x}"] = boot(lambda i: float(num[i][k_[i]].sum() / dd[i][k_[i]].sum()), sc)
            sgn = np.sign(m.h40.values)
            for x, lo, hi, sc_, mk in (("R", "R0.8", "R1.2", 0.5, None), ("B", "B0.5", "B1.5", 0.2, wok)):
                d = (h20[hi] - h20[lo]).values * sc_ * sgn
                k_ = np.isfinite(d) if mk is None else (mk & np.isfinite(d))
                cell[f"dHead20_{x}"] = boot(lambda i: float(np.mean(d[i][k_[i]])) if k_[i].any() else np.nan, sc)
            cell["v24_F_minus_h"] = float(np.nanmedian(v24["F"].values - m.hv24.values))
            out["cells"].append(cell)
    # verdicts (prereg)
    V = {}
    cells = {(c["tag"], c["win"]): c for c in out["cells"]}
    for tag in TG:
        st, ns, tu, op = (cells.get((tag, w)) for w in ("STOP", "NS", "TURN", "OPEN"))
        v = {}
        if st:
            useB = st["rows_w"] >= 0.6 * st["rows"]
            dS = (st["dS_B"] if useB else st["dS_R"])[0]
            dH = -(st["dHas_B"] if useB else st["dHas_R"])[0]
            v["STOP_channel"] = "B" if useB else "R"
            v["STOP"] = ("underpowered " if not st["judge"] else "") + ("cue-locked" if dS < 1 and dH < 0.10 else "inertia" if dS > 3 or dH >= 0.25 else "mixed")
            if ns:
                v["STOP_aux_ratio"] = st["G_B" if useB else "G_R"][0] / ns["G_B" if useB else "G_R"][0]
        if tu and op:
            useB = tu["rows_w"] >= 0.6 * tu["rows"] and op["rows_w"] >= 0.6 * op["rows"]
            x = "B" if useB else "R"
            ratio = tu[f"G_{x}"][0] / op[f"G_{x}"][0]
            dh = tu[f"dHead20_{x}"][0]
            v["TURN_channel"], v["TURN_ratio"], v["TURN_dhead20"] = x, ratio, dh
            v["TURN"] = ("underpowered " if not tu["judge"] else "") + ("inertia" if ratio >= 0.7 or dh <= -2 else "cue-locked" if ratio <= 0.3 and abs(dh) < 1 else "mixed")
        if op:
            x = "B" if op["rows_w"] >= 0.6 * op["rows"] else "R"
            sl = op[f"slope_v24_{x}"][0]
            v["OPEN_channel"], v["OPEN"] = x, ("underpowered " if not op["judge"] else "") + ("copy" if sl >= 0.7 else "anchored" if sl <= 0.3 else "mixed")
        for c in [c for c in out["cells"] if c["tag"] == tag]:
            key = "dS" if c["win"].startswith("STOP") else "G"
            r_, b_ = (c[f"{key}_R50"] if key == "dS" else c["G_R50"])[0], c[f"{key}_B"][0]
            v[f"share_R_{c['win']}"] = r_ / b_ if b_ and np.isfinite(b_) and abs(b_) > 1e-6 else np.nan
        V[tag] = v
    out["verdicts"] = V
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / f"report{a.suffix}.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    lines.append(f"G0 ({g.get('gate_tag', 'RSG-L18-s0')} F against the logged plan): p99 {g['gate_p99']:.4f} m, max {g['gate_max']:.4f} m, within 2 fp16 steps {g['gate2_share']:.3f}, rows {g['rows']}, W/B usable {g['w_ok']:.2f}\n")
    lines.append("| model | window | rows (scenes) | W/B rows | has stop F | dS_R per +20 % | dS_W | dS_B | d(has) R / B per +20 % | G_R | G_W | G_B | v24 slope R / B | dHead20 R / B |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for c in out["cells"]:
        lines.append(f"| {c['tag']} | {c['win']} | {c['rows']} ({c['scenes']}) | {c['rows_w']} ({c['scenes_w']}) | {c['has_F']:.2f} | {f3(c['dS_R'])} | {f3(c['dS_W'])} | "
                     f"{f3(c['dS_B'])} | {c['dHas_R'][0]:+.3f} / {c['dHas_B'][0]:+.3f} | {f3(c['G_R'])} | {f3(c['G_W'])} | {f3(c['G_B'])} | "
                     f"{c['slope_v24_R'][0]:.2f} / {c['slope_v24_B'][0]:.2f} | {c['dHead20_R'][0]:+.2f} / {c['dHead20_B'][0]:+.2f} |")
    lines.append("\nStop station by input-speed scale (R arm, mean m; has-stop share), Z = speed entries zeroed:\n")
    for c in out["cells"]:
        if c["win"].startswith("STOP") or c["win"] == "OPEN":
            lines.append(f"- {c['tag']} {c['win']}: S " + ", ".join(f"{f}: {v:.1f}" for f, v in c["S_R"].items()) + f"; has " +
                         ", ".join(f"{f}: {v:.2f}" for f, v in c["has_R"].items()) + f"; Z: has {c['has_Z']:.2f}, S - S_F {c['S_Z_minus_F']:+.1f}" +
                         (f"; S_F - rest median {c['S_F_minus_rest']:+.1f} m" if c["S_F_minus_rest"] is not None else "") + f"; v24_F - human {c['v24_F_minus_h']:+.2f}")
    lines.append("\nVerdicts (prereg rules):\n")
    for t, v in V.items():
        lines.append(f"- {t}: " + "; ".join(f"{k} {x:.2f}" if isinstance(x, float) else f"{k} {x}" for k, x in v.items()))
    (RES / f"report{a.suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("recede")
    r.add_argument("--controls", default="", help="file of stop-control scene ids (supplement 3)")
    c = sp.add_parser("cf")
    c.add_argument("--rows", default=str(ROWS), help="row files (onp_rows output dir)")
    c.add_argument("--gate-tag", default="RSG-L18-s0", help="the policy whose logged plan G0 compares against")
    c.add_argument("--name", default="", help="output subdir of $O (default: P160 at $O itself)")
    c.add_argument("--tags", default=",".join(TAGS), help="models (comma list; the first encodes the frames)")
    c.add_argument("--every", type=int, default=5)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--limit", type=int, default=0)
    sp.add_parser("select")
    pa = sp.add_parser("parity")
    pa.add_argument("--read", required=True)
    rp = sp.add_parser("report")
    rp.add_argument("--name", default="")
    a = p.parse_args()
    global O
    if getattr(a, "name", ""):
        O = O0 / a.name
    a.suffix = f"_{a.name}" if getattr(a, "name", "") else ""
    {"recede": cmd_recede, "cf": cmd_cf, "report": cmd_report, "select": cmd_select, "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    main()
