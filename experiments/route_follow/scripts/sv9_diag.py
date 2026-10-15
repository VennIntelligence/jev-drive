#!/usr/bin/env python3
"""SPIN v9 step 2 (plans/2026-10-15-spin-v9-prereg.md): the handoff decisions of stored PAI reads, numpy only (Tokyo host python3).
Per rollout: sf_detect's spin (handoff / fast / slow), rf_pai281's handoff spin (|vx wz| > 6 in (1.7, 3.2] s), and for the decisions
k = 17 .. 17 + N (k 17 = the first one the policy controls after the 1.7 s force-GT handoff): fed speed (ego row), plan 2 s arc length and its
ratio to 2 x fed speed, plan lateral offset / heading at 2 s, and track_ref's v8 rule (JEV_TRACK=gate) replayed offline on the logged plan
with the controller row's speed and acceleration (gate ratio, open, the re-timed reference's 2 s arc). Controller: the steering command
u_steering_angle at the first rows after the handoff, the onset of the first |vx wz| > 6 after 1.7 s and the decision in effect then.
  sv9_diag.py --run app1x=<root> rsg=<root> [--n 10] [--scenes list] --csv out.csv --md out.md"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "../../steer_long/scripts"), os.path.join(here, "../../alpasim/lib")]
from sf_detect import detect
import track_ref as TR

K0 = 17


def rollouts(root, keep):
    """{scene: dict(score, kind, det, hs, rows (dict of arrays), recs {k: record})}."""
    out = {}
    for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
        f = os.path.join(sim, "aggregate", "results-summary.json")
        if not os.path.exists(f):
            continue
        ros = [r for r in json.load(open(f))["rollouts"] if (keep is None or r["clipgt_id"] in keep) and r.get("metrics")]
        if not ros:
            continue
        R = {}
        for line in open(os.path.join(sim, "..", "driver", "drive.jsonl")):
            if '"infer": true' in line:
                r = json.loads(line)
                R.setdefault(r["session"], {})[r["k"]] = r
        for r in ros:
            c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
            if not os.path.exists(c) or r["rollout_id"] not in R:
                continue
            rows = list(csv.DictReader(open(c)))
            g = {k: np.array([float(x[k]) for x in rows]) for k in ("timestamp_us", "vx", "vy", "wz", "u_steering_angle", "front_steering_angle", "acceleration")}
            g["t"] = (g["timestamp_us"] - g["timestamp_us"][0]) / 1e6 + 0.1
            m = r["metrics"]
            kind = "" if r["score"] else ("collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else "corridor" if m.get("left_corridor_laterally") else "other")
            w = (g["t"] > 1.7) & (g["t"] <= 3.2)
            out[r["clipgt_id"]] = dict(score=float(r["score"] or 0), kind=kind, rows=g, recs=R[r["rollout_id"]],
                                      det=detect(g["t"], g["vx"], g["vy"], g["wz"], g["front_steering_angle"]),
                                      hs=bool(w.any() and np.abs(g["vx"] * g["wz"])[w].max() > 6))
    return out


def dec(rec, g):
    """One decision -> plan / speed / v8 replay facts."""
    P = np.asarray(rec["poses"], float)
    seg = np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P[:, :2]], axis=0).T)
    vf = float(10 * np.hypot(rec["ego"][4], rec["ego"][5]))
    i = int(np.abs(g["timestamp_us"] - rec["t0"]).argmin())
    v0, a0 = float(np.hypot(g["vx"][i], g["vy"][i])), float(g["acceleration"][i])
    (t, x, y, h), inf = TR.build(P, v0, 0.0, mode="gate", a0=a0)
    (_, xt, yt, _), _ = TR.build(P, v0, 0.0, mode="time", a0=a0)
    arc = lambda X, Y, j: float(np.hypot(np.diff(X[:j + 1]), np.diff(Y[:j + 1])).sum())  # noqa: E731
    M = np.asarray(rec.get("poses_model", rec["poses"]), float)
    vm = np.hypot(*np.diff(np.r_[[[0.0, 0.0]], M[:2, :2]], axis=0).T) / 0.5          # the model plan's speed over 0-0.5 / 0.5-1 s
    lead = ((rec.get("fix") or {}).get("lead") or {}).get("cut", 0.0)
    return dict(k=rec["k"], vm1=float(vm[1]), dv=vf - float(vm[1]), lead_cut=float(lead or 0.0), rel=round((rec["now"] - g["timestamp_us"][0]) / 1e6 + 0.1, 2), vf=vf, v0=v0, a0=a0,
                arc2=float(seg[:4].sum()), ratio=float(seg[:4].sum() / max(2 * vf, 1e-3)), y2=float(P[3, 1]), h2=float(np.degrees(P[3, 2])),
                y1=float(P[1, 1]), v8_open=int(inf["open"]), v8_gate=float(inf["gate"]), v8_pp=inf.get("pp"), v8_arc2=arc(x, y, 20),
                v8_y2=float(y[20]), time_arc2=arc(xt, yt, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--scenes")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    keep = set(open(a.scenes).read().split()) if a.scenes else None
    rows, L = [], []
    for lr in a.run:
        lab, root = lr.split("=", 1)
        D = rollouts(root, keep)
        for sc, d in sorted(D.items()):
            g, det = d["rows"], d["det"]
            ds = [dec(d["recs"][k], g) for k in range(K0, K0 + a.n + 1) if k in d["recs"]]
            if not ds:
                continue
            A = np.abs(g["vx"] * g["wz"])
            on = np.flatnonzero((g["t"] > 1.7) & (A > 6))
            t_on = float(g["t"][on[0]]) if len(on) else None
            k_on = max((x["k"] for x in ds if t_on is not None and x["rel"] <= t_on + 1e-6), default=None)
            h = (g["t"] > 1.7) & (g["t"] <= 2.2)
            d0 = ds[0]
            rows.append(dict(label=lab, scene=sc[7:15], score=d["score"], kind=d["kind"], hspin=int(d["hs"]), spin=det["kind"],
                             t_on=t_on, k_on=k_on, v_hand=d0["v0"], u_peak05=float(np.abs(g["u_steering_angle"][h]).max()) if h.any() else None,
                             ratio17=d0["ratio"], min_ratio=min(x["ratio"] for x in ds), arc17=d0["arc2"], vf17=d0["vf"], y2_17=d0["y2"], h2_17=d0["h2"],
                             v8_open_any=int(any(x["v8_open"] for x in ds)), v8_open17=d0["v8_open"], v8_gate17=d0["v8_gate"], v8_gate_max=max(x["v8_gate"] for x in ds),
                             v8_arc17=d0["v8_arc2"], time_arc17=d0["time_arc2"],
                             on_ratio=next((x["ratio"] for x in ds if x["k"] == k_on), None), on_y2=next((x["y2"] for x in ds if x["k"] == k_on), None),
                             on_v8_gate=next((x["v8_gate"] for x in ds if x["k"] == k_on), None),
                             vm1_17=d0["vm1"], dv17=d0["dv"], lead17=d0["lead_cut"],
                             first_low=next((x["k"] for x in ds if x["ratio"] < 0.7), None)))
    keys = list(rows[0])
    with open(a.csv, "w") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    q = lambda v: "[" + ", ".join(f"{x:.2f}" for x in np.quantile(v, (0.1, 0.5, 0.9))) + "]" if len(v) else "-"  # noqa: E731
    L += [f"# Handoff decisions (k {K0} .. {K0 + a.n}), sv9_diag.py", "", "Quantiles are 10 / 50 / 90 %. ratio = plan 2 s arc / (2 x fed speed).", "",
          "| label | group | n | v at handoff m/s | fed - model plan speed (0.5-1 s) k17 m/s | ratio k17 | min ratio k17..k27 | plan y 2 s m (k17) | v8 gate ratio k17 | v8 opens in window | u peak 1.7-2.2 s rad | ratio < 0.7 in window |",
          "|:--|:--|--:|:--|:--|:--|:--|:--|:--|--:|:--|--:|"]
    for lab in dict.fromkeys(r["label"] for r in rows):
        for gname, sel in (("handoff spin", lambda r: r["hspin"]), ("any sf spin", lambda r: r["spin"] != ""), ("no spin", lambda r: r["spin"] == ""),
                           ("no spin, v >= 15", lambda r: r["spin"] == "" and r["v_hand"] >= 15)):
            R = [r for r in rows if r["label"] == lab and sel(r)]
            if not R:
                continue
            c = lambda k: np.array([r[k] for r in R if r[k] is not None], float)  # noqa: E731
            L.append(f"| {lab} | {gname} | {len(R)} | {q(c('v_hand'))} | {q(c('dv17'))} | {q(c('ratio17'))} | {q(c('min_ratio'))} | {q(np.abs(c('y2_17')))} | "
                     f"{q(c('v8_gate17'))} | {int(c('v8_open_any').sum())} | {q(c('u_peak05'))} | {sum(r['first_low'] is not None for r in R)} |")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
