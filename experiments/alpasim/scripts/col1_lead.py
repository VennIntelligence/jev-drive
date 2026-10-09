#!/usr/bin/env python3
"""Lane COL1, nuPlan half: the frozen model's own lead outputs on the collision rollouts and on control rollouts, against the simulator's gap
(plans/2026-10-09-col1-lead-prereg.md; simulator state is a label only). State in $DATA_DIR/runs/alpasim/col1/lead/.

  msgs    (AlpaSim venv) the logged driver-side messages of every collision case (cases.pkl of col1_nuplan.py) and of the control scenes
          (ctrl/manifest.json: P2H10-F-s0 re-run with logs kept; the scenes of the first two public shards) -> msgs/<group>/<scene>.pkl
          and spec.json {group: {driver, tag, scenes, frames}}; --frames F adds scenes (one `group scene` per line) whose model frames are kept
  maps    (AlpaSim venv) privileged map polygons of the clip scenes -> maps.pkl (drawn in the clips only)
  replay  (envs/op-train, one GPU: a pool job) every decision of every scene again through the real driver class; per decision the plan,
          the served checkpoint's lead / lead_prob (FT) and the shipped policy's (P0 = load_pmodel("P0"), no adapter) on the same vision
          tokens; the model frames of the newest slot for the collision cases -> replay/<group>.pkl
  read    (AlpaSim venv) labels, the registered quantities and lines -> --out DIR / nuplan_lead.md, nuplan_lead.csv

Signals as registered: p = sigmoid(lead_prob[0]); d = lead[:72].reshape(3, 6, 4)[0, 0, 0] minus the camera-to-front-bumper distance;
v_l = [0, 0, 2]. Labels: g = the struck object's nearest corner along the ego heading minus the front bumper (controls: the nearest object
whose box overlaps the straight-ahead corridor of ego width + 0.5 m within 80 m); c = closing speed along the ego heading; a_need =
c_+^2 / (2 max(g - 0.5 c_+ - 1, 0.1)).
"""
import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RA = DATA / "runs/alpasim"
O = RA / "col1"
LD = O / "lead"
DRV = {"P2H10-F": "sh30", "APY10m10-AB": "ap2", "AP2H10-AB": "ap2"}
FRONT, CAM_X = 1.461 + 5.176 / 2, 1.786                     # front bumper and CAM_F0 ahead of the rear axle
CAM_TO_BUMPER = FRONT - CAM_X
BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
GRID = [(p, a) for p in (0.3, 0.5, 0.7, 0.9) for a in (1.0, 1.5, 2.5)]


# ---------------------------------------------------------------- msgs
def cmd_msgs(a):
    import c1_extract as X
    C = pickle.load(open(O / "cases.pkl", "rb"))
    spec, jobs = {}, []
    for (drv, scene), o in sorted(C.items()):
        d = LD / "msgs" / drv
        d.mkdir(parents=True, exist_ok=True)
        spec.setdefault(drv, dict(driver=DRV[drv.rsplit("-s", 1)[0]], tag=drv, scenes=[], frames=[]))
        spec[drv]["scenes"].append(scene), spec[drv]["frames"].append(scene)
        jobs.append((o["run"], scene, str(d)))
    ctrl = json.loads((O / "ctrl/manifest.json").read_text())["P2H10-F-s0"] if (O / "ctrl/DONE").exists() else []     # controls join when their re-run is done
    sh = dict(l.split("\t") for l in (RA / "c0b/lists/shards.tsv").read_text().splitlines() if l.strip())
    d = LD / "msgs/ctrl"
    d.mkdir(parents=True, exist_ok=True)
    if ctrl:
        spec["ctrl"] = dict(driver="sh30", tag="P2H10-F-s0", scenes=[], frames=[], runs={})
    for run in ctrl:
        for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]:
            s = r["clipgt_id"]
            if sh.get(s) in ("part001", "part002") and X.glob.glob(f"{run}/rollouts/{s}/*/rollout.asl"):
                spec["ctrl"]["scenes"].append(s)
                spec["ctrl"]["runs"][s] = run
                jobs.append((run, s, str(d)))
    for line in (Path(a.frames).read_text().splitlines() if a.frames else []):    # extra scenes whose model frames the clips need
        g, s = line.split()[:2]
        if g == "extra":                                  # a control rollout of the P2H10-F-s0 re-run, any of its 700 scenes
            run = next(r for r in ctrl if X.glob.glob(f"{r}/rollouts/{s}/*/rollout.asl"))
            (LD / "msgs/extra").mkdir(parents=True, exist_ok=True)
            spec.setdefault("extra", dict(driver="sh30", tag="P2H10-F-s0", scenes=[], frames=[], runs={}))
            if s not in spec["extra"]["scenes"]:
                spec["extra"]["scenes"].append(s), spec["extra"]["frames"].append(s)
                spec["extra"]["runs"][s] = run
                jobs.append((run, s, str(LD / "msgs/extra")))
    jobs = [j for j in jobs if not (Path(j[2]) / f"{j[1]}.pkl").exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        n = len(list(ex.map(X.one_msgs, jobs)))
    (LD / "spec.json").write_text(json.dumps(spec))
    print("msgs", n, "new;", {k: len(v["scenes"]) for k, v in spec.items()})


def cmd_maps(a):
    """Privileged map polygons (c1_extract map) of the clip scenes of --frames (`set scene` per line) -> lead/maps.pkl; clips only."""
    import c1_extract as X
    spec, C = json.loads((LD / "spec.json").read_text()), pickle.load(open(O / "cases.pkl", "rb"))
    by, out = defaultdict(set), (pickle.load(open(LD / "maps.pkl", "rb")) if (LD / "maps.pkl").exists() else {})
    for line in Path(a.frames).read_text().splitlines():
        g, s = line.split()[:2]
        if s not in out:
            by[C[g, s]["run"] if (g, s) in C else spec[g]["runs"][s]].add(s)
    for run, sc in by.items():
        with tempfile.TemporaryDirectory() as d:
            Path(d, "s.txt").write_text("\n".join(sorted(sc)))
            X.cmd_map(argparse.Namespace(run=run, scenes=f"{d}/s.txt", out=f"{d}/m.pkl", radius=90.0))
            out |= pickle.load(open(f"{d}/m.pkl", "rb"))
    pickle.dump(out, open(LD / "maps.pkl", "wb"), protocol=4)
    print("maps", len(out))


# ---------------------------------------------------------------- replay (GPU)
class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a):
    import torch
    import sh30_driver as D
    import pp_train as T
    spec = json.loads((LD / "spec.json").read_text())
    (LD / "replay").mkdir(exist_ok=True)
    p0 = T.load_pmodel("P0", torch.device("cuda"))
    pb, ctx = D.egodriver_pb2, Ctx()
    for g, sp in spec.items():
        if (LD / "replay" / f"{g}.pkl").exists() and not a.force:
            continue
        if sp["driver"] == "sh30":
            core = D.C.Core(sp["tag"], "cuda", "backwarp")
            drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        else:
            import ap2_driver as AD
            core = AD.AC.Core(sp["tag"], "cuda", "")
            drv = AD.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        sl, cap, last = core.model.net.slices, {}, {}
        core.model.register_forward_hook(lambda m, args, out: cap.update(H=args[0], ego=args[1], tc=args[2], out=out))
        orig = core.plan

        def plan(*x, _o=orig, **k):
            last["o"] = _o(*x, **k)
            return last["o"]
        core.plan = plan
        out = {}
        for i, scene in enumerate(sp["scenes"]):
            rec, uuid = [], None
            for kind, raw in pickle.load(open(LD / "msgs" / g / f"{scene}.pkl", "rb")):
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
                    with torch.no_grad():
                        o0 = p0(cap["H"], cap["ego"], cap["tc"]).float()[0]
                    ft = cap["out"].float()[0]
                    r = dict(now=int(req.time_now_us), poses=last["o"]["poses"].copy(), n_slots=int(last["o"]["valid"].sum()),
                             cmd=int(np.argmax(drv.sessions[uuid].cmd)), cam_x=float(drv.sessions[uuid].cam["t"][0]))
                    for name, v in (("FT", ft), ("P0", o0)):
                        r[name] = dict(lead=v[sl["lead"]].cpu().numpy(), lead_prob=v[sl["lead_prob"]].cpu().numpy())
                    if scene in sp["frames"]:
                        r["frame"] = last["o"]["cur"][7].copy()
                    rec.append(r)
            drv.sessions.pop(uuid, None)
            out[scene] = rec
            if i % 25 == 0:
                print(g, i, len(sp["scenes"]), scene, flush=True)
        pickle.dump(out, open(LD / "replay" / f"{g}.pkl", "wb"), protocol=4)
        print("replay", g, len(out), "scenes", flush=True)
        del core, drv
        torch.cuda.empty_cache()
    (LD / "REPLAY_DONE").touch()


# ---------------------------------------------------------------- read
def vel(L, tr, t, dt=0.2e6):
    a, b = L.interp_pose(tr, t - dt), L.interp_pose(tr, t + dt)
    t0, t1 = np.clip([t - dt, t + dt], tr[0, 0], tr[-1, 0])
    return (b[:2] - a[:2]) / max((t1 - t0) * 1e-6, 1e-3)


def label(L, o, t, aid=None):
    """-> (g, c, v_e, a_need, object id) at sim time t; aid None = the nearest object in the straight-ahead corridor (controls)."""
    from shapely.geometry import box as rect
    e = L.ego(o)
    pe = L.interp_pose(e, t)
    u = np.array([np.cos(pe[2]), np.sin(pe[2])])
    rear = pe[:2] - L.CENTER * u
    ve = vel(L, e, t)
    best = None
    for k in ([aid] if aid else [k for k in o["actors"] if k != "EGO"]):
        tr = o["actors"][k]
        if not (tr[0, 0] <= t <= tr[-1, 0]):
            continue
        p = L.interp_pose(tr, t)
        cs = (np.array(L.box(*p, *o["size"][k][:2]).exterior.coords)[:4] - rear) @ L.rot(pe[2])
        if aid:
            g = float(cs[:, 0].min() - FRONT)
        else:
            from shapely.geometry import Polygon
            it = Polygon(cs).intersection(rect(FRONT, -(L.EGO_W + 0.5) / 2, FRONT + 80, (L.EGO_W + 0.5) / 2))
            if it.is_empty:
                continue
            g = float(it.bounds[0] - FRONT)
        if best is None or g < best[0]:
            best = (g, float((ve - vel(L, tr, t)) @ u), k)
    v_e = float(np.hypot(*ve))
    if best is None:
        return None, None, v_e, 0.0, None
    g, c, k = best
    cp = max(c, 0.0)
    return g, c, v_e, cp * cp / (2 * max(g - 0.5 * cp - 1.0, 0.1)), k


def signals(r, src):
    ld = r[src]["lead"][:72].reshape(3, 6, 4)
    return float(1 / (1 + np.exp(-r[src]["lead_prob"][0]))), float(ld[0, 0, 0] - CAM_TO_BUMPER), float(ld[0, 0, 2])


def trig(p, d, vl, ve, ps, as_):
    return bool(p >= ps and ((max(ve - vl, 0.0) ** 2 / (2 * max(d - 4.0, 0.5)) >= as_) or (ve < 0.5 and d < 6.0)))


def boot(x, n=10000):
    x = np.asarray(x, float)
    if not len(x):
        return (float("nan"),) * 3
    m = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f3(b):
    return f"{b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]"


def cmd_read(a):
    import c1_extract as X
    import c1_lib as L
    spec = json.loads((LD / "spec.json").read_text())
    C = pickle.load(open(O / "cases.pkl", "rb"))
    tax = {(r["driver"], r["scene"]): r for r in csv.DictReader(open(Path(a.out) / "nuplan_cases.csv"))}
    cf = LD / "ctrl_logs.pkl"
    if not cf.exists():
        with ProcessPoolExecutor(a.jobs) as ex:
            N = dict(ex.map(X.one_log, [(spec["ctrl"]["runs"][s], s) for s in spec["ctrl"]["scenes"]], chunksize=4))
        for run in set(spec["ctrl"]["runs"].values()):
            S = {r["clipgt_id"]: r for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]}
            recs = defaultdict(list)
            for line in open(f"{run}/driver-logs/drive.jsonl"):
                if '"drive"' in line:
                    r = json.loads(line)
                    if r["kind"] == "drive" and r["scene"] in N:
                        recs[r["scene"]].append((r["k"], r["poses"][-1][:2]))
            for s, o in N.items():
                if spec["ctrl"]["runs"][s] == run:
                    o["summary"], o["rec"] = S[s], [dict(k=k, poses=[p]) for k, p in sorted(recs[s])]
        pickle.dump(N, open(cf, "wb"), protocol=4)
    N = pickle.load(open(cf, "rb"))
    rows, dropped, total = [], defaultdict(int), defaultdict(int)
    spec.pop("extra", None)                               # clip-only rollouts
    for g, sp in spec.items():
        R = pickle.load(open(LD / "replay" / f"{g}.pkl", "rb"))
        for scene, recs in R.items():
            if g == "ctrl":
                o = N[scene]
                if o["summary"]["metrics"].get("collision_any"):
                    continue
                grp, aid, t_ev, kind = "N", None, None, ""
            else:
                o, tx = C[g, scene], tax[g, scene]
                grp, aid, t_ev, kind = "C", tx["obj"], L.first_event(o, "collision_at_fault"), tx["kind"]
            run_end = {r["k"]: r["poses"][-1][:2] for r in o["rec"]}
            for k, r in enumerate(recs):
                total[g] += 1
                err = float(np.hypot(*(r["poses"][-1][:2] - np.array(run_end[k])))) if k in run_end else float("nan")
                if not err <= 0.05:
                    dropped[g] += 1
                    continue
                gg, c, ve, an, ob = label(L, o, r["now"], aid)
                row = dict(group=grp, driver=sp["tag"], set=g, scene=scene, kind=kind, k=k, now=round(r["now"] * 1e-6, 3), t_event=None if t_ev is None else round(t_ev * 1e-6, 2),
                           before=bool(t_ev is None or r["now"] < t_ev), g=gg, c=c, v_e=ve, a_need=an, obj=ob, repro_err=round(err, 4), n_slots=r["n_slots"])
                for src in ("P0", "FT"):
                    row[f"p_{src}"], row[f"d_{src}"], row[f"vl_{src}"] = signals(r, src)
                rows.append(row)
    out = Path(a.out)
    with open(out / "nuplan_lead.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    by = defaultdict(list)
    for r in rows:
        by[r["set"], r["scene"]].append(r)
    CA = ("lead stopped", "slow lead")
    recipes = [("P2H10-F (registered C)", ["P2H10-F-s0", "P2H10-F-s1"]), ("APY10m10-AB (separate)", ["APY10m10-AB-s0", "APY10m10-AB-s1"]),
               ("AP2H10-AB (separate, 700 scenes)", ["AP2H10-AB-s0", "AP2H10-AB-s1"])]
    Nr = [v for (g, s), v in by.items() if g == "ctrl"]
    md = ["# COL1: the frozen model's lead outputs on the nuPlan collision rollouts (generated by scripts/col1_lead.py read)", "",
          "Pre-registration: [plans/2026-10-09-col1-lead-prereg.md](../../plans/2026-10-09-col1-lead-prereg.md). P0 = shipped policy weights without the adapter bias on the same vision tokens "
          "(primary); FT = the served checkpoint's own output vector. Per-decision rows: `nuplan_lead.csv`.", "",
          "## Replay check (4 s end point within 0.05 m of the run's plan)", "", "| set | decisions | reproduced | dropped |", "|:--|--:|--:|--:|"]
    for g in spec:
        md.append(f"| {g} | {total[g]} | {total[g] - dropped[g]} | {dropped[g]} |")
    md += ["", f"Controls N: {len(Nr)} P2H10-F-s0 rollouts without any collision flag on part001 + part002 (re-run of the original chunk lists, logs kept), "
               f"{sum(len(v) for v in Nr)} decisions; an object in the straight-ahead corridor at {sum(r['obj'] is not None for v in Nr for r in v)} of them.", ""]

    def seen(v, src):
        """First decision of the final approach that is a hit with a_need <= 4 -> its lead time, else None."""
        for r in v:
            if r["before"] and r["g"] is not None and r[f"p_{src}"] >= 0.5 and abs(r[f"d_{src}"] - r["g"]) <= max(2.0, 0.3 * r["g"]) and r["a_need"] <= 4.0:
                return r["t_event"] - r["now"]
        return None
    md += ["## 1. Seen in time (class A: lead stopped + slow lead)", "", "| recipe | C_A rollouts (scenes) | S_time P0 | S_time FT | lead time of the first seen-in-time decision, P0 (s) | line L1 |", "|:--|--:|--:|--:|:--|:--|"]
    for name, tags in recipes:
        ca = [v for (g, s), v in by.items() if g in tags and v[0]["kind"] in CA]
        if not ca:
            md.append(f"| {name} | 0 | | | | no class-A collision |")
            continue
        s0, s1 = [seen(v, "P0") for v in ca], [seen(v, "FT") for v in ca]
        b0, b1 = boot([x is not None for x in s0]), boot([x is not None for x in s1])
        line = "descriptive only (< 8 rollouts)" if len(ca) < 8 else ">= 0.60: sees the lead in time" if b0[0] >= 0.6 else "<= 0.30: does not" if b0[0] <= 0.3 else "in between: partial"
        md.append(f"| {name} | {len(ca)} ({len({v[0]['scene'] for v in ca})}) | {sum(x is not None for x in s0)} / {len(ca)} = {f3(b0)} | {sum(x is not None for x in s1)} / {len(ca)} = {f3(b1)} | "
                  f"{', '.join(f'{x:.1f}' for x in s0 if x is not None) or 'none'} | {line} |")
    for title, sel in (("class-A collision decisions before impact, all recipes", lambda r: r["group"] == "C" and r["kind"] in CA and r["before"]),
                       ("all P2H10-F collision decisions before impact (struck object)", lambda r: r["group"] == "C" and r["driver"].startswith("P2H10") and r["before"]),
                       ("control decisions with an object in the corridor", lambda r: r["group"] == "N" and r["obj"] is not None)):
        md += ["", f"p and d - g against the true gap g, {title}:", "", "| g bin m | decisions | median p P0 | share p >= 0.5 P0 | median d - g P0 (p >= 0.5) | hit share P0 | median p FT | share p >= 0.5 FT | median d - g FT (p >= 0.5) | hit share FT |",
               "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for lo, hi in BINS:
            v = [r for r in rows if sel(r) and r["g"] is not None and lo <= max(r["g"], 0) < hi]
            if not v:
                md.append(f"| {lo}-{hi if hi < 1e8 else ''} | 0 | | | | | | | | |")
                continue
            cells = []
            for src in ("P0", "FT"):
                p, dg = np.array([r[f"p_{src}"] for r in v]), np.array([r[f"d_{src}"] - r["g"] for r in v])
                hit = (p >= 0.5) & (np.abs(dg) <= np.maximum(2.0, 0.3 * np.array([r["g"] for r in v])))
                cells += [f"{np.median(p):.2f}", f"{(p >= 0.5).mean():.2f}", f"{np.median(dg[p >= 0.5]):+.1f}" if (p >= 0.5).any() else "", f"{hit.mean():.2f}"]
            md.append(f"| {lo}-{hi if hi < 1e8 else ''} | {len(v)} | " + " | ".join(cells) + " |")

    def guard(v, src, ps, as_):
        return [trig(r[f"p_{src}"], r[f"d_{src}"], r[f"vl_{src}"], r["v_e"], ps, as_) for r in v]

    def ceiling(Cr, src, ps, as_):
        turned, late = [], []
        for v in Cr:
            v = [r for r in v if r["before"]]
            t = guard(v, src, ps, as_)
            turned.append(any(x and r["a_need"] <= 4.0 for x, r in zip(t, v)))
            late.append(any(t) and not turned[-1])
        fa_d, fa_s, sup, nd = 0, [], 0, 0
        for v in Nr:
            t = guard(v, src, ps, as_)
            fa = [x and (r["obj"] is None or r["a_need"] <= 0.5) for x, r in zip(t, v)]
            fa_d += sum(fa)
            sup += sum(x and not y for x, y in zip(t, fa))
            nd += len(v)
            ks = [r["k"] for r, y in zip(v, fa) if y]
            fa_s.append(any(k + 1 in ks for k in ks))
        return turned, late, fa_d, nd, fa_s, sup
    md += ["", "## 2. Guard ceiling, rule G(p*, a*) on model outputs and ego speed", "",
           "Turned = a rollout of C with a trigger at a decision before impact whose label has a_need <= 4 m/s^2 (upper bound). False-alarm scene = a control rollout with false triggers "
           "(a_need <= 0.5 m/s^2 or no object in the corridor) at 2 consecutive decisions. Supported = control triggers with a_need > 0.5.", "",
           "| C | source | p* | a* | turned / C | too late | false-alarm decisions | false-alarm scenes / N | supported control triggers | line L2 |", "|:--|:--|--:|--:|:--|--:|:--|:--|--:|:--|"]
    for name, tags in recipes:
        Cr = [v for (g, s), v in by.items() if g in tags]
        for src in ("P0", "FT"):
            for ps, as_ in [(0.5, 1.5)] + ([x for x in GRID if x != (0.5, 1.5)] if name.startswith("P2H10") else []):
                turned, late, fa_d, nd, fa_s, sup = ceiling(Cr, src, ps, as_)
                bt, bf = boot(turned), boot(fa_s)
                prim = (ps, as_) == (0.5, 1.5)
                line = ("pass" if bt[0] >= 0.4 and bf[0] <= 0.15 else "not as a rule") if prim and src == "P0" and name.startswith("P2H10") else "primary point, descriptive" if prim else "post hoc grid"
                md.append(f"| {name} | {src} | {ps} | {as_} | {sum(turned)} / {len(Cr)} = {f3(bt)} | {sum(late)} | {fa_d} / {nd} = {fa_d / max(nd, 1):.3f} | {sum(fa_s)} / {len(fa_s)} = {f3(bf)} | {sup} | {line} |")
    md += ["", "## Turned rollouts at the primary point, by kind (P0)", "", "| recipe | kind | rollouts | turned | trigger too late | no trigger |", "|:--|:--|--:|--:|--:|--:|"]
    for name, tags in recipes:
        kd = defaultdict(list)
        for (g, s), v in by.items():
            if g in tags:
                kd[v[0]["kind"]].append(v)
        for k, Cr in sorted(kd.items()):
            turned, late = ceiling(Cr, "P0", 0.5, 1.5)[:2]
            md.append(f"| {name} | {k} | {len(Cr)} | {sum(turned)} | {sum(late)} | {len(Cr) - sum(turned) - sum(late)} |")
    md += ["", "## Per collision rollout (P0; decisions before impact, oldest first)", "", "| driver | scene | kind | t impact | g m | a_need | p | d m | v_l | v_e | trigger (0.5, 1.5) |", "|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|"]
    j = lambda xs, f="{:.1f}": " ".join("-" if x is None else f.format(x) for x in xs)  # noqa: E731
    for (g, s), v in sorted(by.items()):
        if g == "ctrl":
            continue
        v = [r for r in v if r["before"]]
        md.append(f"| {g} | {s[-16:]} | {v[0]['kind'] if v else ''} | {v[0]['t_event'] if v else ''} | {j([r['g'] for r in v])} | {j([r['a_need'] for r in v])} | {j([r['p_P0'] for r in v], '{:.2f}')} | "
                  f"{j([r['d_P0'] for r in v])} | {j([r['vl_P0'] for r in v])} | {j([r['v_e'] for r in v])} | {''.join('x' if t else '.' for t in guard(v, 'P0', 0.5, 1.5))} |")
    (out / "nuplan_lead.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("msgs", cmd_msgs), ("maps", cmd_maps), ("replay", cmd_replay), ("read", cmd_read)):
        p = sub.add_parser(n)
        p.add_argument("--out"), p.add_argument("--frames"), p.add_argument("--jobs", type=int, default=16), p.add_argument("--force", action="store_true")
        p.set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
