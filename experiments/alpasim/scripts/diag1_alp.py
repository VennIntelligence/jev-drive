#!/usr/bin/env python3
"""Lane DIAG1 (experiments/op_parity/plans/2026-10-09-diag1-prereg.md, domains A-nu / A-pai): per-decision tables of the AlpaSim rollout
decisions lane COL1 extracted: the served checkpoint's plan against the shipped policy weights (no adapter) on the same vision tokens.
No simulator, no training; simulator state and logged futures are labels only. COL1's files are read, not changed.

  replay  (envs/op-train, one GPU: a pool job) col1_lead.py's replay again (logged driver messages -> the real driver class), with more
          captured per decision on the hooked (H, ego, tc): ft (the served checkpoint's own output), p0 (pp_train.load_pmodel("P0")),
          nobias (inputs_on=False), gate (model.gate = 0.5), posecv (the 4-pose history in `ego` replaced by a straight constant-speed
          history at the fed vx: pp_wod_diag.edit "posecv"; the vision tokens keep the driven history). Every output is decoded as the
          driver decodes its own (33 camera-frame points -> rear axle through the lever arm, 0.5 .. 4 s) -> $OUT/replay/<group>.pkl
  nuplan  (envs/op-train, CPU) replay + COL1's logs (cases.pkl, lead/ctrl_logs.pkl) -> $OUT/nuplan.npz
  slim    (Tokyo box, numpy only, read-only on COL1's files) x/<run>/replay/*.npz without the model frames -> one npz per run
  pai     (envs/op-train, CPU) the slimmed PAI replays + their logs.pkl ($OUT/pai_src/<run>.npz, <run>_logs.pkl) -> $OUT/pai.npz

$OUT = $DATA_DIR/runs/op_parity/diag1/alp. Columns: README.txt next to the tables (written by `nuplan` / `pai`).
Labels (both tracks, from the rollout's own log): fut_roll / fut_log (N, 8, 3) = the rollout's realized / the scene's recorded ego rear-axle
track at +0.5 .. +4 s, each in its own rear-axle frame at the decision time (NaN beyond the track's end); roll_arc / log_arc (N, 4) =
their polyline arc through the 0.5 s points at 0.5 / 1 / 2 / 4 s (the construction of a plan's arc); v0_sim = the controller trace's
speed at the decision (pose-difference speed where the trace has not started), v0_pose = pose-difference speed; sim_gap / sim_closing =
col1_pai.lead_label (nearest actor box overlapping the straight-ahead corridor of ego width + 0.5 m within 80 m; NaN = none).
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = DATA / "runs/op_parity/diag1/alp"
PX, PY, PYAW = slice(8, 20, 3), slice(9, 20, 3), slice(10, 20, 3)      # ego feature layout: lib/parity_adapter.ego_features
T_HIST = np.array([-1.5, -1.0, -0.5, 0.0], np.float32)               # op_interp.T_KEY: the history times of every driver here
VARS = ("ft", "p0", "nobias", "gate", "posecv")
H_ARC = (0, 1, 3, 7)                                                   # 0.5 / 1 / 2 / 4 s on the 0.5 s grid


# ---------------------------------------------------------------- replay (GPU)
def cmd_replay(a):
    import torch
    import col1_lead as CL
    import sh30_driver as D
    import pp_train as T
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    spec = json.loads((CL.LD / "spec.json").read_text())
    (OUT / "replay").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    p0 = T.load_pmodel("P0", dev)
    pb, ctx, th = D.egodriver_pb2, CL.Ctx(), torch.from_numpy(T_HIST).to(dev)
    for g in (a.groups.split(",") if a.groups else [g for g in spec if g != "extra"]):
        sp = spec[g]
        if (OUT / "replay" / f"{g}.pkl").exists() and not a.force:
            continue
        if sp["driver"] == "sh30":
            core = D.C.Core(sp["tag"], "cuda", "backwarp")
            drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        else:
            import ap2_driver as AD
            core = AD.AC.Core(sp["tag"], "cuda", "")
            drv = AD.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        model, sl, cap, last = core.model, core.model.net.slices, {}, {}
        hook = model.register_forward_hook(lambda m, args, out: cap.update(H=args[0], ego=args[1], tc=args[2], out=out))
        orig = core.plan

        def plan(*x, _o=orig, **k):
            last["o"] = _o(*x, **k)
            return last["o"]
        core.plan = plan
        stored = pickle.load(open(CL.LD / "replay" / f"{g}.pkl", "rb"))
        out, gate = {}, dict(n=0, hook=0.0, stored=0.0, n_stored_gt_1mm=0)
        for i, scene in enumerate(sp["scenes"]):
            R, uuid = {}, None
            for kind, raw in pickle.load(open(CL.LD / "msgs" / g / f"{scene}.pkl", "rb")):
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
                    req = pb.DriveRequest.FromString(raw)
                    drv.drive(req, ctx)
                    s, o = drv.sessions[uuid], last["o"]
                    H, ego, tc, ft = cap["H"], cap["ego"], cap["tc"], cap["out"]          # the served call (later calls overwrite cap)
                    cam_t = np.asarray(s.cam["t"], np.float64)
                    with torch.no_grad():
                        V = {"ft": ft, "p0": p0(H, ego, tc), "nobias": model(H, ego, tc, inputs_on=False)}
                        model.gate = 0.5
                        V["gate"] = model(H, ego, tc)
                        model.gate = 0.0
                        e = ego.clone()
                        e[:, PX], e[:, PY], e[:, PYAW] = e[:, 4:5] * th[None].to(e.dtype), 0, 0
                        V["posecv"] = model(H, e, tc)
                    V = {n: v.float()[0].cpu().numpy() for n, v in V.items()}
                    r = dict(now=int(req.time_now_us), poses=o["poses"], n_slots=int(o["valid"].sum()), cmd=int(np.argmax(s.cmd)), cam_t=cam_t,
                             ego=ego.float()[0].cpu().numpy()[:20], hist=o["hist"], tc=tc.float()[0].cpu().numpy())
                    for n, v in V.items():
                        mu = v[core.pi].reshape(33, 15)
                        r[f"plan_{n}"] = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")      # sh30_core.Core.plan's export
                        if n in ("ft", "p0"):
                            r[f"mu_{n}"], r[f"lead_{n}"], r[f"lp_{n}"] = mu.astype(np.float16), v[sl["lead"]], v[sl["lead_prob"]]
                    es = float(np.abs(r["poses"] - stored[scene][len(R.get("now", []))]["poses"]).max())
                    gate["n"] += 1
                    gate["hook"] = max(gate["hook"], float(np.abs(r["plan_ft"] - r["poses"]).max()))
                    gate["stored"], gate["n_stored_gt_1mm"] = max(gate["stored"], es), gate["n_stored_gt_1mm"] + int(es > 1e-3)
                    for n, v in r.items():
                        R.setdefault(n, []).append(v)
            drv.sessions.pop(uuid, None)
            out[scene] = {n: np.array(v) for n, v in R.items()}
            if i % 25 == 0:
                print(g, i, len(sp["scenes"]), scene, flush=True)
        pickle.dump(dict(tag=sp["tag"], driver=sp["driver"], route=bool(getattr(core, "route", False)), cold=core.cold, gate=gate, scenes=out),
                    open(OUT / "replay" / f"{g}.pkl", "wb"), protocol=4)
        print(f"replay {g}: {len(out)} scenes, {gate['n']} decisions; GATE max |hook FT decode - driver poses| = {gate['hook']:.2e} m, "
              f"max |poses - stored COL1 replay poses| = {gate['stored']:.2e} m ({gate['n_stored_gt_1mm']} decisions > 1e-3)", flush=True)
        hook.remove()
        del core, drv, model
        torch.cuda.empty_cache()
    (OUT / "replay" / "DONE").touch()


# ---------------------------------------------------------------- labels (shared by both tracks)
def labels(o, t):
    """One rollout record (c1_extract.one_log) and decision times t (us) -> label columns (module docstring)."""
    import col1_lib as L
    import col1_pai as CP
    t = np.asarray(t, float)
    off, ego = L.center_off(o), o["actors"]["EGO"]
    gt = next(x["traj"] for x in o["logged"] if x["id"] == "EGO")
    dt = 0.5e6 * np.arange(9)

    def fut(tr):
        out = np.full((len(t), 9, 3), np.nan)
        for i, ti in enumerate(t):
            if not tr[0, 0] - 2e4 <= ti <= tr[-1, 0]:
                continue
            p = L.to_rig(L.interp(tr, ti + dt), off)
            out[i] = np.c_[L.into(p[0], p[:, :2]), L.wrap(p[:, 2] - p[0, 2])]
            out[i, ti + dt > tr[-1, 0] + 2e4] = np.nan
        return out

    def arc(f):
        return np.cumsum(np.hypot(*np.diff(f[:, :, :2], axis=1).transpose(2, 0, 1)), 1)[:, list(H_ARC)]
    fr, fl = fut(ego), fut(gt)
    vp = L.speed(ego, t)
    c = o.get("ctrl")
    vs = vp.copy()
    if c is not None and len(c):
        cols = list(o["ctrl_cols"])
        c = c[np.argsort(c[:, 0])]
        inside = (t >= c[0, 0] - 6e4) & (t <= c[-1, 0] + 6e4)
        vs[inside] = np.interp(t[inside], c[:, 0], np.hypot(c[:, cols.index("vx")], c[:, cols.index("vy")]))
    g, cl = CP.lead_label(o, t, off)
    return dict(fut_roll=fr[:, 1:], fut_log=fl[:, 1:], roll_arc=arc(fr), log_arc=arc(fl), v0_sim=vs, v0_pose=vp, sim_gap=g,
                sim_closing=np.where(np.isnan(g), np.nan, cl), log_v0=L.speed(gt, t))


def stack(rows):
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        out[k] = np.concatenate(v) if isinstance(v[0], np.ndarray) and v[0].ndim else np.array(v)
    return out


def describe(T, extra):
    n = len(T["now_us"])
    by = Counter(zip(T["group"].tolist(), T["tag"].tolist()))
    sc = {k: len(set(T["scene"][(T["group"] == k[0])].tolist())) for k in by}
    dv = np.abs(T["v0"] - T["v0_sim"])
    lines = [f"rows {n}; per group (tag): " + ", ".join(f"{g} ({t}): {c} decisions / {sc[g, t]} scenes" for (g, t), c in sorted(by.items())),
             f"|v0 - v0_sim| median {np.nanmedian(dv):.3f} m/s, p95 {np.nanpercentile(dv, 95):.3f}, max {np.nanmax(dv):.3f}; "
             f"|v0 - v0_pose| median {np.nanmedian(np.abs(T['v0'] - T['v0_pose'])):.3f}",
             "arrays: " + ", ".join(f"{k} {T[k].shape}" for k in T)] + extra
    return lines


# ---------------------------------------------------------------- nuplan table
def cmd_nuplan(a):
    import col1_lead as CL
    C, N = pickle.load(open(CL.O / "cases.pkl", "rb")), pickle.load(open(CL.LD / "ctrl_logs.pkl", "rb"))
    rows, gates = [], []
    for f in sorted((OUT / "replay").glob("*.pkl")):
        g, R = f.stem, pickle.load(open(f, "rb"))
        gates.append(f"  {g}: hook-vs-driver {R['gate']['hook']:.2e} m, vs stored COL1 replay {R['gate']['stored']:.2e} m "
                     f"({R['gate']['n_stored_gt_1mm']} of {R['gate']['n']} > 1e-3), cold rule {R['cold']}, route arm {R['route']}")
        for scene, r in R["scenes"].items():
            o = N[scene] if g == "ctrl" else C[g, scene]
            n = len(r["now"])
            end = {x["k"]: x["poses"][-1][:2] for x in o["rec"]}
            row = dict(group=np.full(n, g), scene=np.full(n, scene), tag=np.full(n, R["tag"]), driver=np.full(n, R["driver"]), k=np.arange(n),
                       now_us=r["now"], cmd=r["cmd"], n_slots=r["n_slots"], v0=r["ego"][:, 4] * 10.0, ego=r["ego"], hist=r["hist"],
                       cam_x=r["cam_t"][:, 0], plan_run=r["poses"], **{f"plan_{v}": r[f"plan_{v}"] for v in VARS},
                       mu_ft=r["mu_ft"], mu_p0=r["mu_p0"], lead_p0=r["lead_p0"], lp_p0=r["lp_p0"], lead_ft=r["lead_ft"], lp_ft=r["lp_ft"],
                       repro_err=np.array([np.hypot(*(r["poses"][k, -1, :2] - np.array(end[k]))) if k in end else np.nan for k in range(n)]),
                       collision_any=np.full(n, bool(o["summary"]["metrics"].get("collision_any")) if "metrics" in o["summary"] else
                                             bool(o["summary"].get("collision_any"))),
                       **labels(o, r["now"]))
            rows.append(row)
    T = stack(rows)
    np.savez_compressed(OUT / "nuplan.npz", **T)
    e = T["ego"]
    extra = ["gates (replay):"] + gates + [
        f"repro_err (replay plan end point vs the run's logged plan): median {np.nanmedian(T['repro_err']):.4f} m, > 0.05 m in {(T['repro_err'] > 0.05).sum()} rows "
        f"(col1_lead read drops those; kept here, filter on repro_err)",
        "ego feature check per driver (mean | std | share exactly 0) of vx/10, vy/10, ax/3, ay/3:"]
    for d in sorted(set(T["driver"].tolist())):
        m = T["driver"] == d
        extra.append(f"  {d} (n {m.sum()}): " + "; ".join(f"{e[m, j].mean():+.3f} | {e[m, j].std():.3f} | {(e[m, j] == 0).mean():.2f}" for j in (4, 5, 6, 7))
                     + f"; present {e[m, 0].mean():.2f}; cmd L/S/R {e[m, 1:4].mean(0).round(2).tolist()}; n_slots {dict(Counter(T['n_slots'][m].tolist()))}"
                     + f"; fed vx vs -hist x(-0.5 s) / 0.5: median abs diff {np.median(np.abs(e[m, 4] * 10 + e[m, 14] * 10 / 0.5)):.3f} m/s")
    txt = ["nuplan.npz: AlpaSim nuPlan-track decisions (COL1 collision cases per driver tag + `ctrl` = P2H10-F-s0 control scenes), "
           "scripts/diag1_alp.py replay + nuplan"] + describe(T, extra)
    (OUT / "README_nuplan.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


# ---------------------------------------------------------------- PAI
def cmd_slim(a):
    """Tokyo box: <src>/<run>/replay/*.npz -> <dst>/<run>.npz (all per-decision arrays but the model frames, + scene)."""
    src, dst = Path(a.src), Path(a.dst)
    for d in sorted(src.glob("*/replay")):
        fs = sorted(d.glob("*.npz"))
        n_msgs = len(list((d.parent / "msgs").glob("*.pkl")))
        if not fs or len(fs) < n_msgs:
            print(d.parent.name, "replay incomplete:", len(fs), "of", n_msgs, "scenes; skipped")
            continue
        R = {}
        for f in fs:
            z = np.load(f)
            for k in z.files:
                if k not in ("frames", "frame_k"):
                    R.setdefault(k, []).append(z[k])
            R.setdefault("scene", []).append(np.full(len(z["now"]), f.stem))
        np.savez_compressed(dst / f"{d.parent.name}.npz", **{k: np.concatenate(v) for k, v in R.items()})
        print(d.parent.name, len(fs), "scenes", sum(len(v) for v in R["now"]), "decisions")


def cmd_pai(a):
    import col1_lib as L
    arm = lambda run: "vcont" if run.startswith("v1") else "baseline"  # noqa: E731  (col1_pai_chain*.sh: v1* = DRV_PY=col1_pai_driver.py)
    rows, notes = [], []
    for f in sorted((OUT / "pai_src").glob("*.npz")):
        if f.stem.endswith("_logs"):
            continue
        run, z, logs = f.stem, np.load(f), L.load(OUT / "pai_src" / f"{f.stem}_logs.pkl")
        for scene in sorted(set(z["scene"].tolist())):
            m, o = z["scene"] == scene, logs[scene]
            n = int(m.sum())
            rec = {r["now"]: r for r in o["rec"] if r.get("infer")}
            now = z["now"][m]
            has = np.array([t in rec for t in now.tolist()])
            ego = np.array([rec[t]["ego"] if t in rec else [np.nan] * 20 for t in now.tolist()], np.float32)
            logged = np.array([rec[t]["poses"] if t in rec else np.full((8, 3), np.nan) for t in now.tolist()], np.float32)
            served = np.array([rec[t].get("poses_model", rec[t]["poses"]) if t in rec else np.full((8, 3), np.nan) for t in now.tolist()], np.float32)
            row = dict(group=np.full(n, run), arm=np.full(n, arm(run)), scene=np.full(n, scene), tag=np.full(n, a.tag), k=np.arange(n), now_us=now,
                       t0_us=z["t0"][m], cmd=z["cmd"][m], n_slots=np.array([rec[t]["n_real"] if t in rec else -1 for t in now.tolist()]),
                       v0=ego[:, 4] * 10.0, ego=ego, hist=np.array([rec[t]["hist"] if t in rec else np.full((4, 3), np.nan) for t in now.tolist()], np.float32),
                       cam_x=np.full(n, float(o["start"]["cam_t"][0])), plan_run=z["run"][m], plan_ft=z["ft"][m], plan_ftS=z["ftS"][m], plan_p0=z["p0"][m],
                       plan_logged=logged, mu_ft=z["mu_ft"][m], mu_p0=z["mu_p0"][m], lead_p0=z["lead_p0"][m], lp_p0=z["lp_p0"][m], lead_ft=z["lead_ft"][m],
                       lp_ft=z["lp_ft"][m], repro_err=np.hypot(*(z["run"][m][:, -1, :2] - served[:, -1, :2]).T),
                       **labels(o, z["t0"][m]))
            rows.append(row)
            if not has.all():
                notes.append(f"  {run} {scene}: {int((~has).sum())} replay decisions without a logged inference record (ego NaN)")
    T = stack(rows)
    np.savez_compressed(OUT / "pai.npz", **T)
    e = T["ego"]
    ok = ~np.isnan(e[:, 0])
    extra = [f"gate: max |ft (re-decoded from the hooked tokens) - run (the driver's returned poses in the replay)| = {np.abs(T['plan_ft'] - T['plan_run']).max():.2e} m; "
             f"repro_err (replay plan end point vs the run's logged model plan): median {np.nanmedian(T['repro_err']):.4f} m, > 0.05 m in {(T['repro_err'] > 0.05).sum()} rows",
             "ego = the run's own drive.jsonl record of the decision (rounded to 1e-5), matched on now_us; the replay did not store it",
             f"ego feature check (n {ok.sum()}; mean | std | share exactly 0) of vx/10, vy/10, ax/3, ay/3: "
             + "; ".join(f"{e[ok, j].mean():+.3f} | {e[ok, j].std():.3f} | {(e[ok, j] == 0).mean():.2f}" for j in (4, 5, 6, 7))
             + f"; cmd L/S/R {e[ok, 1:4].mean(0).round(2).tolist()}; n_slots {dict(Counter(T['n_slots'].tolist()))}"
             + f"; fed vx vs -hist x(-0.5 s) / 0.5: median abs diff {np.median(np.abs(e[ok, 4] * 10 + e[ok, 14] * 10 / 0.5)):.3f} m/s"] + notes
    txt = [f"pai.npz: AlpaSim PAI-track decisions (10 Hz), served checkpoint {a.tag}; group = COL1 run under x/, arm = baseline (pai_driver.py) | "
           "vcont (col1_pai_driver.py, speed-continuity re-timing of the served trajectory); scripts/diag1_alp.py slim + pai"] + describe(T, extra)
    (OUT / "README_pai.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("replay", cmd_replay), ("nuplan", cmd_nuplan), ("slim", cmd_slim), ("pai", cmd_pai)):
        p = sub.add_parser(n)
        p.add_argument("--groups"), p.add_argument("--force", action="store_true"), p.add_argument("--src"), p.add_argument("--dst")
        p.add_argument("--tag", default="P2H10-F-s0")
        p.set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
