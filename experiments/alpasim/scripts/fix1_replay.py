#!/usr/bin/env python3
"""Lane FIX1 open-loop design check (plans/2026-10-09-fix1-prereg.md, "Design reads"): the serving switches of lib/serve_fix.py on the
driver-side messages lane COL1 extracted from existing nuPlan rollouts (col1/lead/msgs: the at-fault collision rollouts and 198 control
rollouts of P2H10-F-s0). Every decision goes through the real driver class with the switches off (the cores must reproduce COL1's replay)
and the three arms are computed beside it on the same model output; the ego does not react, so this shows when a switch would act,
not what it does to a score.

  replay  (envs/op-train, one GPU: a pool job)  -> $DATA_DIR/runs/alpasim/fix1/replay/<group>.pkl
  read    (any python with numpy)  labels from results/collisions/nuplan_lead.csv -> --out DIR / replay.md, replay.csv
"""
import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
LD = DATA / "runs/alpasim/col1/lead"
O = DATA / "runs/alpasim/fix1/replay"
ARMS = {"a": (1.0, False), "b": (0.0, True), "ab": (1.0, True)}


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


class Tap:
    """Stands in for a session's Serve: runs the three arms on the model output and leaves the returned poses alone."""

    def __init__(self, FX, front):
        self.arms, self.rows, self.front = {k: FX.Serve(v, l, front) for k, (v, l) in ARMS.items()}, [], front

    def __call__(self, o, v0, a0, t_us):
        r = dict(t_us=t_us, v0=v0, a0=a0, poses=o["poses"].copy(), vm=float(o["mu"][0, 3]), lead=o["lead"][:72].copy(), lead_prob=o["lead_prob"].copy())
        for k, f in self.arms.items():
            t = time.perf_counter()
            p, info = f(dict(o), v0, a0, t_us)
            r[k] = dict(poses=p, info=info, ms=1e3 * (time.perf_counter() - t))
        self.rows.append(r)
        return o["poses"], {}


def cmd_replay(a):
    import sh30_driver as D
    FX = D.FX
    assert not FX.ON, "run with the switches off: the arms are computed beside the unchanged driver"
    spec = json.loads((LD / "spec.json").read_text())
    O.mkdir(parents=True, exist_ok=True)
    pb, ctx = D.egodriver_pb2, Ctx()
    for g in a.groups:
        sp = spec[g]
        if (O / f"{g}.pkl").exists() and not a.force:
            continue
        if sp["driver"] == "sh30":
            core = D.C.Core(sp["tag"], "cuda", "backwarp")
            drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        else:
            import ap2_driver as AD
            core = AD.AC.Core(sp["tag"], "cuda", "")
            drv = AD.Driver(core, Path(tempfile.mkdtemp()), 0, False)
        core.lead_out = True
        ref = pickle.load(open(LD / "replay" / f"{g}.pkl", "rb"))
        out, err = {}, 0.0
        for i, scene in enumerate(sp["scenes"][:a.limit]):
            uuid = tap = None
            for kind, raw in pickle.load(open(LD / "msgs" / g / f"{scene}.pkl", "rb")):
                if kind == "driver_session_request":
                    req = pb.DriveSessionRequest.FromString(raw)
                    uuid = req.session_uuid
                    drv.start_session(req, ctx)
                    s = drv.sessions[uuid]
                    tap = s.fix = Tap(FX, FX.cam_to_front(req.rollout_spec.vehicle, s.cam["t"][0]))
                elif kind == "driver_camera_image":
                    drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                elif kind == "driver_ego_trajectory":
                    drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                elif kind == "route_request":
                    drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                elif kind == "driver_request":
                    drv.drive(pb.DriveRequest.FromString(raw), ctx)
            drv.sessions.pop(uuid, None)
            out[scene] = dict(front=tap.front, rows=tap.rows)
            err = max([err] + [float(np.abs(r["poses"] - q["poses"]).max()) for r, q in zip(tap.rows, ref[scene])])
            if i % 25 == 0:
                print(g, i, len(sp["scenes"]), scene, "max |plan - COL1 replay|", err, flush=True)
        pickle.dump(out, open(O / f"{g}.pkl", "wb"), protocol=4)
        print("replay", g, len(out), "scenes; max |plan - COL1 replay| m", err, flush=True)
        del core, drv
    (O / "REPLAY_DONE").touch()


def cmd_read(a):
    lab = {}
    for r in csv.DictReader(open(a.labels)):
        lab[r["set"], r["scene"], int(r["k"])] = r
    rows = []
    for f in sorted(O.glob("*.pkl")):
        g = f.stem
        for scene, d in pickle.load(open(f, "rb")).items():
            for k, r in enumerate(d["rows"]):
                L = lab.get((g, scene, k), {})
                fl = lambda x: float(x) if x not in ("", None) else np.nan  # noqa: E731
                p8 = np.concatenate([[0.0], np.hypot(*np.diff(np.concatenate([np.zeros((1, 2)), r["poses"][:, :2]]), axis=0).T)]).cumsum()
                row = dict(group=L.get("group", ""), set=g, scene=scene, kind=L.get("kind", ""), k=k, before=L.get("before", ""), g=fl(L.get("g")),
                           c=fl(L.get("c")), a_need=fl(L.get("a_need")), v0=r["v0"], a0=r["a0"], vm=r["vm"], vp0=p8[1] / 0.5, s2=p8[4], front=d["front"])
                for arm in ARMS:
                    q, info = r[arm]["poses"], r[arm]["info"]
                    s = np.concatenate([[0.0], np.hypot(*np.diff(np.concatenate([np.zeros((1, 2)), q[:, :2]]), axis=0).T)]).cumsum()
                    row[f"s2_{arm}"], row[f"ms_{arm}"] = s[4], r[arm]["ms"]
                    if "lead" in info:
                        li = info["lead"]
                        row.update({f"p_{arm}": li["p"][0], f"d_{arm}": li["d"] if li["d"] is not None else np.nan, f"vl_{arm}": li["vl"] if li["vl"] is not None else np.nan,
                                    f"amp_{arm}": li["a_mpc"], f"ae_{arm}": li["a_e2e"], f"cut_{arm}": li["cut"], f"src_{arm}": li["src"]})
                rows.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    A = lambda rs, k: np.array([r.get(k, np.nan) for r in rs], float)  # noqa: E731
    md = ["# FIX1 open-loop design check on COL1's nuPlan replay set (generated by scripts/fix1_replay.py read)", "",
          "The ego does not react: this shows when a switch acts on logged decisions, not what it does to a score. Rows: `replay.csv`.", ""]
    ctrl = [r for r in rows if r["set"] == "ctrl"]
    md += ["## (a) how far the plan's first-segment speed is from the ego's speed (control rollouts, P2H10-F-s0)", "",
           "| ego speed m/s | decisions | median vp0 / v0 | p10 | p90 | median vp0 - v0 m/s | median model v_ego - v0 m/s | median 2 s arc change with (a) m | p10 | p90 |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, 16), (16, 40)):
        rs = [r for r in ctrl if lo <= r["v0"] < hi]
        if rs:
            v0, vp, vm, d2 = A(rs, "v0"), A(rs, "vp0"), A(rs, "vm"), A(rs, "s2_a") - A(rs, "s2")
            q = vp / np.maximum(v0, 0.1)
            md.append(f"| {lo}-{hi} | {len(rs)} | {np.median(q):.2f} | {np.percentile(q, 10):.2f} | {np.percentile(q, 90):.2f} | {np.median(vp - v0):+.2f} | {np.median(vm - v0):+.2f} | "
                      f"{np.median(d2):+.2f} | {np.percentile(d2, 10):+.2f} | {np.percentile(d2, 90):+.2f} |")
    md += ["", "## (b) the lead limit on control rollouts (no collision flag)", "",
           "A decision is *cut* when the limit removes at least 0.5 m from the first 2 s of the served arc. False = the label has no object in the corridor or a_need <= 0.5 m/s^2.", "",
           "| arm | decisions | lead present | limit is the lower candidate | cut >= 0.5 m | of them false | scenes with >= 2 consecutive cut decisions | of them all false | median cut m (cut decisions) |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in ("b", "ab"):
        cut, p, src = A(ctrl, f"cut_{arm}"), A(ctrl, f"p_{arm}"), np.array([r.get(f"src_{arm}") == "mpc" for r in ctrl])
        false = ~(A(ctrl, "a_need") > 0.5)
        sc = {}
        for r, c_, f_ in zip(ctrl, cut >= 0.5, false):
            sc.setdefault(r["scene"], []).append((r["k"], c_, f_))
        run = lambda v: any(x[1] and y[1] for x, y in zip(sorted(v), sorted(v)[1:]))  # noqa: E731
        hit = [s for s, v in sc.items() if run(v)]
        md.append(f"| {arm} | {len(ctrl)} | {(p > 0.5).sum()} | {src.sum()} | {(cut >= 0.5).sum()} | {((cut >= 0.5) & false).sum()} | {len(hit)} / {len(sc)} | "
                  f"{sum(all(f_ for _, c_, f_ in sc[s] if c_) for s in hit)} | {np.median(cut[cut >= 0.5]) if (cut >= 0.5).any() else float('nan'):.2f} |")
    md += ["", "## (b) the lead limit on the at-fault collision rollouts", "",
           "| set | scene | kind | decisions before impact | first cut decision (k, s before impact, a_need) | cut m at the last 3 decisions before impact | lead p / d / true gap at the last decision |", "|:--|:--|:--|--:|:--|:--|:--|"]
    C = {}
    for r in rows:
        if r["group"] == "C" and r["before"] == "True":
            C.setdefault((r["set"], r["scene"], r["kind"]), []).append(r)
    turned = {}
    for (st, scene, kind), rs in sorted(C.items()):
        rs = sorted(rs, key=lambda r: r["k"])
        first = next((r for r in rs if r.get("cut_ab", 0) >= 0.5), None)
        n = len(rs)
        turned.setdefault(st, []).append(first is not None and first["a_need"] <= 4)
        md.append(f"| {st} | {scene[-16:]} | {kind} | {n} | " + (f"k {first['k']}, {0.5 * (n - 1 - rs.index(first)) + 0.5:.1f} s, {first['a_need']:.1f}" if first else "none") +
                  f" | {' / '.join('%.1f' % r.get('cut_ab', 0) for r in rs[-3:])} | {rs[-1].get('p_ab', float('nan')):.2f} / {rs[-1].get('d_ab', float('nan')):.1f} / {rs[-1]['g']:.1f} |")
    md += ["", "Rollouts with a cut decision before impact at which a stop was still possible (a_need <= 4): " +
           ", ".join(f"{k} {sum(v)} / {len(v)}" for k, v in sorted(turned.items())), "",
           "## Cost per decision (CPU, ms)", "", "| arm | median | p90 | max |", "|:--|--:|--:|--:|"]
    for arm in ARMS:
        m = A(rows, f"ms_{arm}")
        md.append(f"| {arm} | {np.median(m):.2f} | {np.percentile(m, 90):.2f} | {m.max():.2f} |")
    (out / "replay.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--groups", nargs="+", default=["ctrl", "P2H10-F-s0", "P2H10-F-s1", "APY10m10-AB-s0"])
    p.add_argument("--limit", type=int, default=10**9)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--labels", default=str(HERE.parent / "results/collisions/nuplan_lead.csv"))
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"replay": cmd_replay, "read": cmd_read}[a.cmd](a)
