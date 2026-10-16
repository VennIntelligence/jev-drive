#!/usr/bin/env python3
"""WAIT-TRAIN prereg 补记 E (diagnostics after decision 335; nothing here trains or writes a registered verdict file). Box, AlpaSim venv
(numpy / pandas / matplotlib / PIL), CPU, after `wt.sh explore-e`. Inputs under runs/wait_cap/train/explore_e/; outputs to --out (default
explore_e/out/): *.json, *.md and figs/*.png, copied into experiments/wait_cap/results/wait_train/explore_e/ on the Mac.

  static   the public static scene (human never moves, lead 1.6 m ahead) per decision for every 补记 D / E run: served plan 2 s / 4 s arc, lateral
           end point, measured speed (the ego row fed at t0), the speed history fed to the adapter (JEV_VHIST), the evaluator's distance to the
           nearest obstacle, collision; the c1_replay plans of the same logged messages with JEV_VHIST off / zero; the open-loop plans of the
           same log on its real camera frames (waitcap_test rows) -> static.{json,md}, figs/static_traces.png, figs/static_frames.png
  xreplay  (op-train, GPU) every stage's plan on the exact model inputs P0 saw in its own closed-loop static-scene run (SH30_DUMP: the 8 slot
           frames, valid mask, measured ego row per decision; P0 stopped there, so from k = 3 the frames and the ego row are a standing car's):
           the measured row, and the row with the speed history zeroed (vhist.py zero's entries); a route-memory checkpoint gets the route memory
           its own static run logged at the same decision (drive.jsonl `rmem`). P0 on its own inputs reproduces its logged plan -> xreplay.json
  stages   one row per stage: open-loop wait W (decision 326 / pilot_gate), static scene, PAI stop17 zeros, held 460 @18 -> stages.{json,md}
  offroad  the J2P-only offroad scenes on held @18 (offroad in WTJ2P, not in WTCP), classified by the rules below -> offroad.{json,md},
           figs/offroad_<i>.png (road areas, log path, both drives, J2P plans before the event; speed by time)

Offroad classes (fixed before any per-scene output was looked at; k_off = J2P's first decision with the evaluator's offroad flag):
  T  turn not completed: the log path turns >= 30 deg within 30 m after J2P's station at k_off - 2, and at k_off J2P's heading lags the log
     path's heading at its station by >= 10 deg in the turn's direction (under-turned), or its offset from the log path is >= 1 m to the turn's
     outside.
  S  slow-speed curvature tracking: not T; J2P's speed is <= 0.8 x WTCP's at the same decision, averaged over k_off - 4 .. k_off, and the
     served plan asks for less distance than the ego is covering (2 s arc < 0.8 x 2 s x measured speed on >= 2 of those decisions) while the
     ego leaves the path sideways (|offset| grows >= 0.5 m over those decisions) - the speed deficit turned into steering (d285 / d307).
  O  other: described one by one.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path("/root/autodl-tmp/ujs")
O = DATA / "runs/wait_cap/train"
XE, XD = O / "explore_e", O / "explore"
STATIC = "2021.09.16.15.12.03_veh-42_01037_01434-b5b8d20688385790"
STATIC_LOG = "2021.09.16.15.12.03_veh-42_01037_01434"
STOP17 = "054b5901 095cf563 127c4f85 2d10a08d 35e082a4 3fe2bac8 44df49fe 4cb7f7ac 4ecc114d 559d4ad6 68c3be44 7d994ed3 9f8acd00 ba40be62 bc309fd6 cbd87234 fc9fde87".split()
PAI_REF = DATA / "runs/route_follow/pai_g10/acc"
STAGES = ("P0", "P2-F-s0", "RB0-F-s0", "RSG-L18-s0", "WTCP-L18-s0", "WTJ2P-L18-s0", "WTJ5P-L18-s0")
T0, DT = 17_000, 500_000
sys.path.insert(0, str(HERE.parents[2] / "research"))
import plot_style as PS  # noqa: E402  (research/plot_style.py: CVPR widths, Okabe-Ito)
PAL = [PS.PALETTE[c] for c in ("blue", "vermillion", "green", "orange", "purple", "sky_blue", "black")]
INK2 = PS.BASELINE


def arc(P, n=8):
    P = np.asarray(P, float)[:n, :2]
    return float(np.hypot(*P[0]) + np.hypot(*np.diff(P, axis=0).T).sum())


def run_of(lab):
    for d in (XE / f"static/{lab}/run", XD / f"sub/{lab}/run"):
        if (d / "aggregate/results-summary.json").exists():
            return d
    return None


def load_run(d):
    """A static-scene run dir -> dict(rec (drive.jsonl drive records by k), metrics {name: (t, value) array}, summary)."""
    import polars as pl                                       # c1_extract's read: the parquet's timestamps are wider than 64 bits
    rec = [json.loads(l) for l in open(d / "driver-logs/drive.jsonl")]
    rec = sorted((r for r in rec if r["kind"] == "drive" and r["scene"] == STATIC), key=lambda r: r["k"])
    m = pl.read_parquet(glob.glob(str(d / f"rollouts/{STATIC}/*/metrics.parquet"))[0]).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
    M = {n[0]: np.array([g["timestamps_us"].to_list(), g["values"].to_list()], np.float64).T for n, g in m.group_by("name")}
    summ = next(r for r in json.loads((d / "aggregate/results-summary.json").read_text())["rollouts"] if r["clipgt_id"] == STATIC)
    return dict(rec=rec, metrics=M, summary=summ)


def series(o, name, t):
    m = o["metrics"].get(name)
    if m is None or not len(m):
        return np.nan
    i = np.flatnonzero(m[:, 0] == t)
    return float(m[i[0], 1]) if len(i) else np.nan


def metrics(dirs):
    """scene -> (score, progress_rel, at-fault collision, offroad, dist travelled) of the first rollout per scene (wt.py metrics + distance)."""
    M = {}
    for D in dirs:
        f = Path(D) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                m = r.get("metrics") or {}
                M.setdefault(r["clipgt_id"], (float(r.get("score") or 0.0), float(m.get("progress_rel", np.nan)), float(m.get("collision_at_fault", np.nan)),
                                              float(m.get("offroad", np.nan)), float(m.get("dist_traveled_m", np.nan))))
    return M


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PS.apply()
    return plt


def save_png(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, pil_kwargs={"optimize": True, "compress_level": 9})
    kb = Path(path).stat().st_size >> 10
    print(path, kb, "KB", "(over 500 KB)" if kb > 500 else "")


# ---------------------------------------------------------------- static scene
def static_rows(o, tag):
    rows = []
    for r in o["rec"]:
        t0 = int(r["t0"])
        P = np.asarray(r["poses"], float)
        e = r["ego"]
        fed = r.get("vhist")
        rows.append(dict(k=int(r["k"]), t=(t0 - T0) / 1e6, n_keys=int(r["n_keys"]), v=10 * float(e[4]), v_fed=None if fed is None else float(fed["v_fed"]),
                         hist15=float(np.hypot(*np.asarray(r["hist"])[0][:2])), arc2=arc(P, 4), arc4=arc(P), y4=float(P[-1, 1]),
                         gap=series(o, "min_distance_to_obstacle_m", t0), coll=series(o, "collision_any", t0)))
    return rows


def replay(tag, mode):
    f = XE / f"replay/{tag}/replay_{mode}.pkl"
    if not f.exists():
        return None
    R = pickle.load(open(f, "rb")).get(STATIC)
    return None if R is None else {int(k): v["run"] for k, v in R["plans"].items()}


def openloop_rows():
    """The same log's waitcap_test rows on its real camera frames (human history), per stage: plan 4 s arc."""
    z = np.load(DATA / "runs/wait_cap/select/rows.npz")
    i = np.flatnonzero(z["log"] == STATIC_LOG)
    out = dict(frame=z["frame"][i].tolist(), cell=z["cell"][i].tolist(), v0=z["v0"][i].tolist(), d4=z["d4"][i].tolist(), lead_gap=z["lead_gap"][i].tolist(), arc4={})
    for t in STAGES:
        p = np.load(DATA / f"runs/wait_cap/plans/{t}.npz")
        pos = {n: j for j, n in enumerate(p["names"].tolist())}
        out["arc4"][t] = [arc(p["plan"][pos[n]]) if n in pos else None for n in z["token"][i].tolist()]
    return out


def cmd_static(a):
    out = Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    res = {}
    labs = sorted({Path(x).parent.name for x in glob.glob(str(XE / "static/*/run"))} | {Path(x).parent.name for x in glob.glob(str(XD / "sub/*/run"))})
    for lab in labs:
        o = load_run(run_of(lab))
        tag = lab.removesuffix("-vhzero")
        rows = static_rows(o, tag)
        m = o["summary"].get("metrics") or {}
        ct = o["metrics"].get("collision_any")
        t_coll = None if ct is None or not (ct[:, 1] > 0).any() else float((ct[ct[:, 1] > 0][0, 0] - T0) / 1e6)
        rp = {}
        if not lab.endswith("-vhzero"):
            for mode in ("off", "zero"):
                R = replay(tag, mode)
                if R is not None:
                    rp[mode] = {k: dict(arc2=arc(P, 4), arc4=arc(P), y4=float(np.asarray(P)[-1, 1])) for k, P in sorted(R.items())}
            R = replay(tag, "off")
            if R is not None:                                 # the reproduction check: replayed plan vs the logged one
                rp["off_max_abs_diff_m"] = float(max(np.abs(np.asarray(R[int(r["k"])]) - np.asarray(r["poses"])).max() for r in o["rec"] if int(r["k"]) in R))
        res[lab] = dict(tag=tag, vhist="zero" if lab.endswith("-vhzero") else "off", score=float(o["summary"].get("score") or 0.0),
                        dist=float(m.get("dist_traveled_m", np.nan)), t_coll=t_coll, min_gap=float(np.nanmin([r["gap"] for r in rows])),
                        rows=rows, replay=rp)
    ol = openloop_rows()
    (out / "static.json").write_text(json.dumps(dict(runs=res, openloop=ol), indent=1, default=float))
    order = [x for x in STAGES if x in res] + [x for x in res if x not in STAGES]
    L = ["# Static public scene, per decision (补记 E)", "",
         f"Scene `{STATIC}`. arc = served plan arc length (m) over 2 / 4 s; v = measured speed (m/s) in the ego row at t0; gap = the evaluator's "
         "min distance to an obstacle (m) at t0; replay = the same logged messages through the driver again with JEV_VHIST off / zero (open loop).", "",
         "| run | history fed | score | distance (m) | first contact (s) | min gap (m) |", "|---|---|--:|--:|--:|--:|"]
    for lab in order:
        r = res[lab]
        L.append(f"| {r['tag']} | {r['vhist']} | {r['score']:.2f} | {r['dist']:.2f} | {r['t_coll'] if r['t_coll'] is not None else '-'} | {r['min_gap']:.2f} |")
    for lab in order:
        r = res[lab]
        L += ["", f"## {lab}", "", "| k | keys | v | arc2 | arc4 | y4 | gap | replay off arc4 | replay zero arc4 |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for x in r["rows"]:
            ro, rz = (r["replay"].get(m, {}).get(x["k"], {}).get("arc4") for m in ("off", "zero"))
            L.append(f"| {x['k']} | {x['n_keys']} | {x['v']:.2f} | {x['arc2']:.2f} | {x['arc4']:.2f} | {x['y4']:.2f} | {x['gap']:.2f} | "
                     + " | ".join("" if v is None else f"{v:.2f}" for v in (ro, rz)) + " |")
        if "off_max_abs_diff_m" in r["replay"]:
            L.append(f"\nreplay reproduction: max |replayed - logged pose| = {r['replay']['off_max_abs_diff_m']:.4f} m")
    L += ["", "## Open loop on the same log's real frames (waitcap_test rows, human history; plan 4 s arc, m)", "",
          "| frame | cell | v0 | human d4 | lead gap | " + " | ".join(STAGES) + " |", "|--:|---|--:|--:|--:|" + "--:|" * len(STAGES)]
    for j in range(len(ol["frame"])):
        if ol["cell"][j] in ("W_L", "S_L", "G_L"):
            L.append(f"| {ol['frame'][j]} | {ol['cell'][j]} | {ol['v0'][j]:.2f} | {ol['d4'][j]:.2f} | {ol['lead_gap'][j]:.2f} | "
                     + " | ".join("-" if ol["arc4"][t][j] is None else f"{ol['arc4'][t][j]:.2f}" for t in STAGES) + " |")
    (out / "static.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))
    static_figs(res, order, out)


def static_figs(res, order, out):
    plt = plt_setup()
    fig, ax = plt.subplots(1, 3, figsize=(PS.DOUBLE_COLUMN_IN, 2.5))
    plain = [x for x in order if not x.endswith("-vhzero")]
    for i, lab in enumerate(plain):
        c = PAL[i % len(PAL)]
        for zl, ls in ((lab, "-"), (lab + "-vhzero", ":")):
            if zl not in res:
                continue
            R = res[zl]["rows"]
            t = [x["t"] for x in R]
            ax[0].plot(t, [x["arc4"] for x in R], ls, color=c, label=lab if zl == lab else None)
            ax[1].plot(t, [x["v"] for x in R], ls, color=c)
            ax[2].plot(t, [x["gap"] for x in R], ls, color=c)
    ax[0].axhline(1.0, color=INK2, lw=1, ls="--")
    ax[0].set(ylabel="plan 4 s arc (m)", xlabel="time (s)")
    ax[1].set(ylabel="measured speed (m/s)", xlabel="time (s)")
    ax[2].set(ylabel="gap to nearest obstacle (m)", xlabel="time (s)")
    ax[0].plot([], [], ":", color=INK2, label="history zeroed")
    ax[0].legend(fontsize=6, loc="upper left", handlelength=1.6)
    fig.tight_layout(pad=0.3)
    save_png(fig, out / "figs/static_traces.png")
    plt.close(fig)
    from PIL import Image
    import io
    labs = [x for x in ("P0", "WTJ2P-L18-s0-vhzero") if (XE / f"static/{x}/run/driver-logs/dump/s00_k0.npz").exists()]
    if labs:                                                  # the CAM_F0 JPEG the driver received at t0 (SH30_DUMP)
        fig, ax = plt.subplots(len(labs), 4, figsize=(PS.DOUBLE_COLUMN_IN, 1.05 * len(labs)), squeeze=False)
        for r, lab in enumerate(labs):
            for c, k in enumerate((0, 3, 6, 9)):
                z = np.load(XE / f"static/{lab}/run/driver-logs/dump/s00_k{k}.npz")
                ax[r, c].imshow(Image.open(io.BytesIO(z["jpeg"].tobytes())))
                ax[r, c].text(0.02, 0.97, f"{lab.replace('-vhzero', ' (history zeroed)')}, t = {k * 0.5:.1f} s", transform=ax[r, c].transAxes,
                              va="top", fontsize=5.5, color="white")
                ax[r, c].axis("off")
        fig.subplots_adjust(0, 0, 1, 1, 0.01, 0.02)
        save_png(fig, out / "figs/static_frames.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------- cross-stage replay on P0's closed-loop inputs (GPU)
def cmd_xreplay(a):
    import torch
    sys.path[:0] = [str(HERE.parents[2] / "experiments/alpasim/lib")]
    import sh30_core as SC
    import vhist as VH
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    dumps = sorted(glob.glob(str(XE / "static/P0/run/driver-logs/dump/s00_k*.npz")), key=lambda f: int(f.split("_k")[-1][:-4]))
    Dz = [dict(np.load(f)) for f in dumps]
    res = dict(source="P0 static run dumps", k=list(range(len(Dz))), v=[10 * float(d["ego"][4]) for d in Dz], tags={})
    for tag in STAGES:
        core = SC.Core(tag, "cuda", "backwarp")
        rm = {}
        if core.rm:                                           # the route memory this checkpoint's own static run logged at the same decision
            src = next((x for x in (XE / f"static/{tag}/run", XD / f"sub/{tag}/run") if (x / "driver-logs/drive.jsonl").exists()), None)
            for l in open(src / "driver-logs/drive.jsonl"):
                r = json.loads(l)
                if r["kind"] == "drive" and "rmem" in r:
                    rm[int(r["k"])] = np.asarray(r["rmem"], np.float32)
        out = {}
        for mode in ("measured", "zero"):
            arcs = []
            for k, d in enumerate(Dz):
                ego = d["ego"].astype(np.float32).copy()
                if mode == "zero":
                    ego[VH.EGO_SPEED] = 0.0
                row = ego if not core.rm else np.r_[ego, rm.get(k, np.zeros(core.rm_dim, np.float32))]
                cur, valid = d["cur"], d["valid"]
                prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
                with torch.no_grad():
                    pp, cc = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
                    H = core.model.net.run_batched(core.A.vision_feeds(pp, cc), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)
                    tc = torch.tensor([[1.0, 0.0]], device=core.dev)
                    o = core.model(H, torch.from_numpy(row[None]).to(core.dev), tc)
                    o = (o[0] if isinstance(o, tuple) else o).float()
                mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
                P = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, d["cam_t"][:2].astype(np.float64), Z.T_OUT, "lever")
                arcs.append(dict(arc2=arc(P, 4), arc4=arc(P), y4=float(P[-1, 1]), dpose_vs_logged=float(np.abs(P - d["poses"]).max()) if tag == "P0" else None))
            out[mode] = arcs
        res["tags"][tag] = out
        print(tag, "measured", [round(x["arc4"], 2) for x in out["measured"]], "zero", [round(x["arc4"], 2) for x in out["zero"]], flush=True)
        del core
        torch.cuda.empty_cache()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "xreplay.json").write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- stages
def cmd_stages(a):
    sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
    import wt
    out = Path(a.out)
    full = {wt.pai_full(p): p for p in STOP17}
    st = json.loads((out / "static.json").read_text())["runs"]
    res = {}
    for t in STAGES:
        held = metrics(glob.glob(f"{DATA}/runs/route_mem/read/{t}/c*/run"))
        pd_ = XD / f"paisub/{t}" if t.startswith("WT") else XE / f"paisub/{t}"
        pai = metrics(glob.glob(f"{pd_}/runs/*/sim"))
        H = np.array(list(held.values())) if held else np.zeros((0, 5))
        try:
            W = wt.cap_read(t)[0]["wait_W"]
        except Exception as e:                                  # noqa: BLE001
            W = None
            print(t, "cap_read", e)
        s = st.get(t) or {}
        res[t] = dict(wait_W=W, static=dict(score=s.get("score"), dist=s.get("dist"), t_coll=s.get("t_coll"), min_gap=s.get("min_gap")),
                      static_vhzero=(st.get(t + "-vhzero") or {}).get("dist"),
                      stop17=dict(zeros=sum(pai[x][0] == 0 for x in full if x in pai), n=sum(x in pai for x in full)),
                      held=dict(scenes=len(H), zeros=int((H[:, 0] == 0).sum()), mean=float(H[:, 0].mean()) if len(H) else None,
                                offroad=int(np.nansum(H[:, 3] > 0)), at_fault=int(np.nansum(H[:, 2] > 0)), dist=float(np.nanmean(H[:, 4])) if len(H) else None))
    ref = metrics(glob.glob(f"{PAI_REF}/runs/*/sim"))
    res["RSG-L18-s0 archive"] = dict(stop17=dict(zeros=sum(ref[x][0] == 0 for x in full if x in ref), n=sum(x in ref for x in full)))
    (out / "stages.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["| stage | open-loop wait W | static: distance (m) / contact (s) | static, history zeroed: distance (m) | PAI stop17 zeros | held 460 @18: scenes | zeros | mean | offroad | at-fault | mean distance (m) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    for t in STAGES:
        r = res[t]
        s = r["static"]
        L.append(f"| {t} | {f(r['wait_W'])} | {f(s['dist'], 2)} / {f(s['t_coll'], 1)} | {f(r['static_vhzero'], 2)} | {r['stop17']['zeros']} / {r['stop17']['n']} | "
                 f"{r['held']['scenes']} | {r['held']['zeros']} | {f(r['held']['mean'])} | {r['held']['offroad']} | {r['held']['at_fault']} | {f(r['held']['dist'], 1)} |")
    a_ = res["RSG-L18-s0 archive"]["stop17"]
    L.append(f"| RSG-L18-s0 archive (PAI config, d328) | | | | {a_['zeros']} / {a_['n']} | | | | | | |")
    (out / "stages.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- offroad
def cmd_offroad(a):
    import pandas as pd
    out = Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    F = XE / "offroad"
    sc = json.loads((F / "scenes.json").read_text())["j_only"]
    C = sorted(glob.glob(str(DATA / "runs/cap_audit/collect-held/*/dec.parquet")))[-1]
    D = pd.read_parquet(C)
    S = pd.read_parquet(Path(C).parent / "scene.parquet").set_index("scene")
    Lg = pd.read_parquet(Path(C).parent / "log.parquet")
    logs, maps = {}, {}
    for f in glob.glob(str(F / "logs_c*.pkl")):
        logs |= pickle.load(open(f, "rb"))
    for f in glob.glob(str(F / "map_c*.pkl")):
        maps |= pickle.load(open(f, "rb"))
    res = {}
    for s in sc:
        J = D[(D.label == "WTJ2P-L18-s0") & (D.scene == s)].sort_values("k").set_index("k")
        Cc = D[(D.label == "WTCP-L18-s0") & (D.scene == s)].sort_values("k").set_index("k")
        o = logs.get(s)
        if not len(J) or o is None:
            res[s] = dict(cls="?", why="no collect rows or no logs")
            continue
        off = J.index[(J.offroad.fillna(0) > 0)]
        k_off = int(off[0]) if len(off) else int(J.index[-1])
        # the log path's heading along station, the turn ahead of J2P at k_off - 2
        eg = o["logged"][0]["traj"]                                         # logged ego (t, x, y, yaw)
        Pg = eg[:, 1:3]
        Sg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Pg, axis=0).T))]
        Yg = np.unwrap(eg[:, 3])
        kk = max(k_off - 2, int(J.index[0]))
        s0 = float(J.s.get(kk, J.s.iloc[0]))
        yaw_at = lambda x: float(np.interp(x, Sg, Yg))                     # noqa: E731
        ahead = [yaw_at(s0 + d) - yaw_at(s0) for d in np.arange(0, 30.5, 1.0)]
        turn = float(ahead[int(np.argmax(np.abs(ahead)))])
        sg = np.sign(turn) if abs(turn) > 1e-6 else 1.0
        jr = J.loc[k_off]
        lag = float(jr.head_path)                                         # degrees, ego heading minus the log path's (left +)
        T = abs(np.degrees(turn)) >= 30 and (lag * sg <= -10 or float(jr.lat) * sg <= -1.0)
        win = [k for k in range(k_off - 4, k_off + 1) if k in J.index]
        vr = float(np.nanmean([J.v[k] for k in win]) / max(np.nanmean([Cc.v[k] for k in win if k in Cc.index]), 1e-3)) if win else np.nan
        recs = {int(r["k"]): r for r in o.get("rec", [])}
        short = sum(1 for k in win if k in recs and arc(recs[k]["poses"], 4) < 0.8 * 2.0 * 10 * float(recs[k]["ego"][4]))
        dlat = float(abs(J.lat[win[-1]]) - abs(J.lat[win[0]])) if len(win) > 1 else 0.0
        Sx = (not T) and vr <= 0.8 and short >= 2 and dlat >= 0.5
        cls = "T" if T else "S" if Sx else "O"
        lr = Lg[Lg.scene == s].set_index("k")
        res[s] = dict(cls=cls, k_off=k_off, turn_deg=float(np.degrees(turn)), head_lag_deg=lag, lat=float(jr.lat), v_ratio=vr, plan_short=short, dlat=dlat,
                      v_J=[float(J.v.get(k, np.nan)) for k in range(1, 19)], v_C=[float(Cc.v.get(k, np.nan)) for k in range(1, 19)],
                      v_log=[float(lr.v.get(k, np.nan)) for k in range(1, 19)], junc=bool(jr.junc) if "junc" in jr else None,
                      dpsi_log=float(S.dpsi_log.get(s, np.nan)), margin=float(jr.margin) if "margin" in jr else None,
                      s_J=float(jr.s), s_C=float(Cc.s.get(k_off, np.nan)), lat_C=float(Cc.lat.get(k_off, np.nan)))
    (out / "offroad.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["| # | scene | class | k_off | log turn ahead (deg) | J2P heading vs path (deg) | J2P offset (m) | speed J2P / CP (k_off-4..k_off) | short plans | offset growth (m) | station J2P / CP at k_off (m) |",
         "|--:|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for i, (s, r) in enumerate(sorted(res.items(), key=lambda x: (x[1]["cls"], x[0]))):
        if r["cls"] == "?":
            L.append(f"| {i} | {s} | ? | | | | | | | | |")
            continue
        L.append(f"| {i} | `{s}` | {r['cls']} | {r['k_off']} | {r['turn_deg']:.0f} | {r['head_lag_deg']:.1f} | {r['lat']:.2f} | {r['v_ratio']:.2f} | {r['plan_short']} | "
                 f"{r['dlat']:.2f} | {r['s_J']:.1f} / {r['s_C']:.1f} |")
    cnt = {c: sum(r["cls"] == c for r in res.values()) for c in ("T", "S", "O", "?")}
    L.append(f"\ncounts: {cnt}")
    (out / "offroad.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    offroad_figs(res, logs, maps, D, out)


def offroad_figs(res, logs, maps, D, out):
    plt = plt_setup()
    for i, (s, r) in enumerate(sorted(res.items(), key=lambda x: (x[1]["cls"], x[0]))):
        if r["cls"] == "?" or s not in maps:
            continue
        o = logs[s]
        fig, ax = plt.subplots(1, 2, figsize=(PS.DOUBLE_COLUMN_IN, 2.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
        for ar in maps[s]["areas"]:
            ax[0].fill(*ar["ext"].T, color="#ecebe7", lw=0)
            for h in ar["holes"]:
                ax[0].fill(*h.T, color="white", lw=0)
        for ln in maps[s]["lanes"]:
            for e in (ln["left"], ln["right"]):
                if e is not None:
                    ax[0].plot(*e.T, color="#c9c8c3", lw=0.6)
        eg = o["logged"][0]["traj"]
        ax[0].plot(eg[:, 1], eg[:, 2], color=INK2, lw=1, ls="--", label="log (human)")
        for lab, c in (("WTJ2P-L18-s0", PAL[1]), ("WTCP-L18-s0", PAL[0])):
            X = D[(D.label == lab) & (D.scene == s)].sort_values("k")
            ax[0].plot(X.x, X.y, color=c, label=lab, marker="o", ms=1.8)
            if lab == "WTJ2P-L18-s0":
                ko = X[X.k == r["k_off"]]
                if len(ko):
                    ax[0].plot(ko.x, ko.y, "x", color="black", ms=6, mew=1.2, label=f"J2P offroad k={r['k_off']}")
        recs = {int(x["k"]): x for x in o.get("rec", [])}
        for k in range(max(r["k_off"] - 4, 0), r["k_off"] + 1):
            if k in recs:
                an, P = np.asarray(recs[k]["anchor"], float), np.asarray(recs[k]["poses"], float)
                c_, s_ = np.cos(an[2]), np.sin(an[2])
                ax[0].plot(an[0] + c_ * P[:, 0] - s_ * P[:, 1], an[1] + s_ * P[:, 0] + c_ * P[:, 1], color=PAL[1], lw=1, alpha=0.5)
        X = D[(D.label == "WTJ2P-L18-s0") & (D.scene == s)]
        cx, cy = float(X.x.median()), float(X.y.median())
        R_ = max(25.0, float(np.ptp(X.x)), float(np.ptp(X.y))) * 0.75
        ax[0].set(xlim=(cx - R_, cx + R_), ylim=(cy - R_, cy + R_), aspect="equal", xlabel="x (m)", ylabel="y (m)")
        ax[0].text(0.02, 0.98, f"class {r['cls']}; thin: J2P plans, k_off-4..k_off", transform=ax[0].transAxes, va="top", fontsize=6)
        ax[0].legend(fontsize=5.5, loc="lower right")
        k = np.arange(1, 19) * 0.5
        ax[1].plot(k, r["v_log"], color=INK2, ls="--", lw=1, label="log")
        ax[1].plot(k, r["v_C"], color=PAL[0], label="WTCP-L18-s0")
        ax[1].plot(k, r["v_J"], color=PAL[1], label="WTJ2P-L18-s0")
        ax[1].axvline(r["k_off"] * 0.5, color="black", lw=0.8, ls=":")
        ax[1].set(ylabel="speed (m/s)", xlabel="time (s)")
        ax[1].legend(fontsize=6)
        fig.tight_layout(pad=0.3)
        save_png(fig, out / f"figs/offroad_{i:02d}_{r['cls']}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("static", "xreplay", "stages", "offroad"))
    ap.add_argument("--out", default=str(XE / "out"))
    a = ap.parse_args()
    dict(static=cmd_static, xreplay=cmd_xreplay, stages=cmd_stages, offroad=cmd_offroad)[a.cmd](a)
