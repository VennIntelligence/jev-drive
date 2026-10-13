#!/usr/bin/env python3
"""SPIN-FIX decision-by-decision trace of one rollout (numpy only, either box): controller rows (speed, side slip, |v wz|, acceleration, MPC
steering command u and steering state) and, at each inference, the plan (2 s arc deficit, 1-2 s speed, the largest segment curvature reached
by 2 s and its lateral demand v^2 k at the logged speed, the plan's 2.5 s lookahead), the gate record served (ratio, open, full's kpk / amin)
and the served reference's lateral demand over 0.5-2 s. With --mode, each record is also rebuilt by track_ref.build (logged v0, measured
acceleration from the controller row as a0) to check that the logged gate value is reproduced and to try a candidate (--cfg).
  sf_trace.py <root> <scene8> [--from T] [--to T] [--mode gate] [--cfg k=v,...] [--rows]"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "../../alpasim/lib"))
from sf_detect import detect
from sf_mid import demand
import track_ref as TR


def load(root, sc):
    for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
        for ro in json.load(open(os.path.join(sim, "aggregate", "results-summary.json")))["rollouts"]:
            if ro["clipgt_id"][7:7 + len(sc)] != sc:
                continue
            rows = list(csv.DictReader(open(os.path.join(sim, "controller", f"alpasim_controller_{ro['rollout_id']}.csv"))))
            C = {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}
            R = [r for r in map(json.loads, open(os.path.join(sim, "..", "driver", "drive.jsonl")))
                 if r["kind"] == "drive" and r.get("infer") and r["session"] == ro["rollout_id"]]
            return ro, C, R
    raise SystemExit(f"{sc}: not found under {root}")


def plan_facts(poses, v0):
    p = np.vstack([[0, 0, 0], np.asarray(poses)])
    seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
    path = TR.Path(poses)
    kk = np.abs(np.diff(path.hv) / np.maximum(np.diff(path.hs), 1e-6))
    s2 = min(seg[:4].sum(), path.s[-2])                                  # arc the plan covers by 2 s / 2.5 s
    s25 = min(seg[:5].sum(), path.s[-2])
    k2 = kk[path.hs[:-1] < max(s2, 0.1)].max(initial=0.0)
    k25 = kk[path.hs[:-1] < max(s25, 0.1)].max(initial=0.0)
    lat = np.hypot(*p[1:5, :2].T)                                        # lateral offset of the 0.5 .. 2 s points
    return dict(def2=2 * v0 - seg[:4].sum(), v12=seg[2:4].sum(), k2=k2, k25=k25, y05=p[1, 1], y1=p[2, 1], y2=p[4, 1], vp=seg / 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("scene")
    ap.add_argument("--from", dest="t_from", type=float); ap.add_argument("--to", dest="t_to", type=float)
    ap.add_argument("--mode", default=""); ap.add_argument("--cfg", default="")
    ap.add_argument("--rows", action="store_true", help="also print every controller row")
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (kv.split("=") for kv in filter(None, a.cfg.split(",")))}
    ro, C, R = load(a.root, a.scene)
    ts = C["timestamp_us"]; t = (ts - ts[0]) / 1e6 + 0.1
    d = detect(t, C["vx"], C["vy"], C["wz"], C["front_steering_angle"])
    m = ro.get("metrics") or {}
    print(f"{ro['clipgt_id']} score {float(ro.get('score') or 0):.3f} spin {d['kind'] or '-'} t_on {d['t_first']} t_end {t[-1]:.1f} "
          f"metrics {{{', '.join(f'{k}: {v:.3g}' for k, v in m.items() if isinstance(v, (int, float)) and v)}}}")
    t1 = d["t_first"] or t[-1]
    lo, hi = a.t_from if a.t_from is not None else t1 - 3.0, a.t_to if a.t_to is not None else t1 + 0.6
    ev = []
    for r in R:
        tr_ = (r["now"] - ts[0]) / 1e6 + 0.1
        if lo <= tr_ <= hi:
            i = int(np.abs(t - tr_).argmin())
            tk = r.get("track", {})
            v0 = tk.get("v0", r.get("fix", {}).get("v0", 0.0))
            f = plan_facts(r["poses"], v0)
            ps = np.vstack([[0, 0, 0], np.asarray(r.get("poses_track", r["poses"]))])
            A_srv = demand(np.arange(9) * 0.5, ps[:, 0], ps[:, 1], ps[:, 2])[0]
            e = dict(t=tr_, kind="I", v0=v0, a0=C["acceleration"][i], gate=tk.get("gate", np.nan), open=tk.get("open", -1),
                     kpk=tk.get("kpk", np.nan), amin=tk.get("amin", np.nan), A_srv=A_srv, **f)
            if a.mode:
                (tt, x, y, h), info = TR.build(np.asarray(r["poses"]), v0, 0.0, mode=a.mode, a0=float(C["acceleration"][i]), **kw)
                e.update(g_re=info.get("gate", np.nan), o_re=info.get("open", -1), A_re=demand(tt, x, y, h)[0])
            ev.append(e)
    hdr = (f"{'t':>5} {'v0':>5} {'a0':>5} {'def2':>5} {'v12':>5} {'k2':>6} {'v2k2':>5} {'k25':>6} {'v2k25':>5} {'y.5':>5} {'y1':>5} {'y2':>5} "
           f"{'gate':>5} {'op':>2} {'kpk':>6} {'amin':>5} {'A_srv':>5}" + (f" {'g_re':>5} {'o_re':>4} {'A_re':>5}" if a.mode else "") + "   plan v (0.5 s seg)")
    print(hdr)
    for e in ev:
        print(f"{e['t']:5.1f} {e['v0']:5.2f} {e['a0']:5.2f} {e['def2']:5.2f} {e['v12']:5.2f} {e['k2']:6.3f} {e['v0'] ** 2 * e['k2']:5.1f} {e['k25']:6.3f} "
              f"{e['v0'] ** 2 * e['k25']:5.1f} {e['y05']:5.2f} {e['y1']:5.2f} {e['y2']:5.2f} {e['gate']:5.2f} {e['open']:2d} {e['kpk']:6.3f} "
              f"{e['amin']:5.1f} {e['A_srv']:5.1f}" + (f" {e['g_re']:5.2f} {e['o_re']:4d} {e['A_re']:5.1f}" if a.mode else "")
              + "   " + " ".join(f"{x:4.1f}" for x in e["vp"]))
    if a.rows:
        print(f"{'t':>5} {'vx':>5} {'vy':>5} {'|vwz|':>5} {'acc':>5} {'u_st':>6} {'delta':>6} {'u_lon':>6}")
        for i in np.where((t >= lo) & (t <= hi))[0]:
            print(f"{t[i]:5.1f} {C['vx'][i]:5.2f} {C['vy'][i]:5.2f} {abs(C['vx'][i] * C['wz'][i]):5.2f} {C['acceleration'][i]:5.2f} "
                  f"{C['u_steering_angle'][i]:6.3f} {C['front_steering_angle'][i]:6.3f} {C['u_longitudinal_actuation'][i]:6.2f}")


if __name__ == "__main__":
    main()
