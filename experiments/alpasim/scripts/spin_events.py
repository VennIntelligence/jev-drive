#!/usr/bin/env python3
"""Spin / loss-of-control events in stored PAI reads (CPU, numpy only; results/spin_events.md).

Per rollout: controller CSV (0.1 s rows: vx, vy, wz, MPC steering command u_steering_angle, accel command u_longitudinal_actuation, roll
from the quaternion) joined to the driver's decision records (drive.jsonl: the trajectory actually handed to the controller, `poses_track`
when the gate built one, else `poses`; a decision at `now` governs controller rows with ts in (now, now + 0.1 s]).

  yaw excess = |wz - v * kappa_plan|, kappa_plan = plan heading at 1 s / plan arc to 1 s (the served path's mean curvature over its first
               second, decision governing the row)
  event      = v >= 8 m/s and (yaw excess >= 0.3 rad/s for >= 2 rows, or sideslip |vy| / vx >= 0.15); after an event the next 3 s are
               skipped. Rows in the first 0.2 s are ignored (initial state coercion).
  sf_detect  = experiments/steer_long/scripts/sf_detect.py (the spin rule of decisions 291 / 309), reported alongside.
Per event: onset t, v, peak |wz| (sign: + left) and peak MPC steering over the next 1.5 s, min accel command over [-0.5, +1] s, the plan of
the decision governing the row 0.3 s before onset (decel over 0-1 s and 0-2 s from its 0.5 s segment speeds against the fed speed, mean
curvature over 1 s and max over 2 s, plan speed reaching < 0.5 m/s inside 4 s), max |roll|, and the outcome: score, zero / partial class
and event time from the pai_fix facts CSV when given (else results-summary failure_reason).

Also writes <csv>_rollouts.csv: every rollout with an event, an sf_detect spin or body roll > 5 deg (roll onset, speed, signed peak roll /
pitch / wz over 2 s, z, MPC steering, time spent over 5 deg).

  spin_events.py --read LABEL=ROOT[=FACTS.csv] ... --csv events.csv --md summary.md
"""
import argparse, csv, glob, json, os, sys

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "steer_long", "scripts"), "experiments/steer_long/scripts"]
from sf_detect import detect  # noqa: E402

V_MIN, EXC, SLIP, N_EXC, SKIP = 8.0, 0.3, 0.15, 2, 3.0


def plan_feats(rec):
    P = np.asarray(rec.get("poses_track") or rec["poses"], float)
    v0 = float((rec.get("fix") or {}).get("v0") or 10.0 * np.hypot(rec["ego"][4], rec["ego"][5]))
    xy = np.r_[[[0.0, 0.0]], P[:, :2]]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    vs = seg / 0.5
    yaw = np.unwrap(np.r_[0.0, P[:, 2]])
    k = np.diff(yaw)[:4] / np.maximum(seg[:4], 0.5)
    lead = (rec.get("fix") or {}).get("acc") or {}
    return dict(v0=v0, a1=(vs[1] - v0) / 0.75, a2=(vs[3] - v0) / 1.75, kap1=float(yaw[2] / max(seg[:2].sum(), 1.0)),
                kmax2=float(np.abs(k).max()), y2=float(P[3, 1]), stop=bool((vs < 0.5).any()), won=lead.get("won", ""))


def roll_of(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))


def runs(n, m):
    out, c = np.zeros_like(m, bool), 0
    for i, x in enumerate(m):
        c = c + 1 if x else 0
        if c >= n:
            out[i - n + 1] = True
    return out


def rollout_events(g, dec):
    t = (g["timestamp_us"] - g["timestamp_us"][0]) / 1e6 + 0.1
    v = np.hypot(g["vx"], g["vy"])
    nows = np.array(sorted(dec))
    idx = np.searchsorted(nows, g["timestamp_us"], side="left") - 1
    feats = {n: plan_feats(dec[n]) for n in nows}
    kap = np.array([feats[nows[i]]["kap1"] if i >= 0 else 0.0 for i in idx])
    exc = np.abs(g["wz"] - v * kap)
    slip = np.abs(g["vy"]) / np.maximum(g["vx"], 1e-3)
    ok = (t > 0.2) & (v >= V_MIN)
    cand = (runs(N_EXC, ok & (exc >= EXC)) | (ok & (slip >= SLIP)))
    roll = np.abs(roll_of(g["qx"], g["qy"], g["qz"], g["qw"]))
    ev, last = [], -1e9
    for i in np.where(cand)[0]:
        if t[i] < last + SKIP:
            continue
        last = t[i]
        w = (t >= t[i]) & (t <= t[i] + 1.5)
        wa = (t >= t[i] - 0.5) & (t <= t[i] + 1.0)
        j = np.where(w)[0][np.argmax(np.abs(g["wz"][w]))]
        jb = max(0, i - 3)
        pf = feats[nows[idx[jb]]] if idx[jb] >= 0 else {}
        ev.append(dict(t=round(float(t[i]), 1), v=round(float(v[i]), 1), wz_pk=round(float(g["wz"][j]), 2),
                       dir="left" if g["wz"][j] > 0 else "right", exc_pk=round(float(exc[w].max()), 2),
                       slip_pk=round(float(slip[w].max()), 2), ust_pk=round(float(np.abs(g["u_steering_angle"][w]).max()), 3),
                       uacc_min=round(float(g["u_longitudinal_actuation"][wa].min()), 2),
                       acc_pre=round(float(g["acceleration"][jb]), 2), roll_pk=round(float(np.degrees(roll[w].max())), 1),
                       handoff=bool(t[i] <= 3.2), **{("p_" + k): (round(x, 3) if isinstance(x, float) else x) for k, x in pf.items()}))
    sf = detect(t, g["vx"], g["vy"], g["wz"], g["front_steering_angle"])
    rs = np.degrees(roll_of(g["qx"], g["qy"], g["qz"], g["qw"]))
    pitch = np.degrees(np.arcsin(np.clip(2 * (g["qw"] * g["qy"] - g["qz"] * g["qx"]), -1, 1)))
    ir = np.where(np.abs(rs) > 5)[0]
    ro = {}
    if len(ir):
        i = ir[0]
        w = (t >= t[i]) & (t <= t[i] + 2.0)
        ro = dict(roll_t=round(float(t[i]), 1), roll_v=round(float(v[i]), 1), roll_signed=round(float(rs[w][np.argmax(np.abs(rs[w]))]), 1),
                  roll_pitch=round(float(pitch[w][np.argmax(np.abs(pitch[w]))]), 1), roll_z=round(float(g["z"][i]), 2),
                  roll_wz=round(float(g["wz"][w][np.argmax(np.abs(g["wz"][w]))]), 2), roll_ust=round(float(np.abs(g["u_steering_angle"][w]).max()), 3),
                  roll_dur=round(float(0.1 * (np.abs(rs) > 5).sum()), 1))
    return ev, sf, float(np.degrees(roll.max())), ro


def load_facts(f):
    if not f or not os.path.exists(f):
        return {}
    return {r["scene"]: r for r in csv.DictReader(open(f))}


def read(label, root, facts):
    F = load_facts(facts)
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "aggregate", "results-summary.json"), recursive=True)):
        sim = os.path.dirname(os.path.dirname(f))
        dj = [x for x in (os.path.join(os.path.dirname(sim), "driver", "drive.jsonl"), os.path.join(sim, "driver-logs", "drive.jsonl"))
              if os.path.exists(x)]
        if not dj:
            continue
        dj = dj[0]
        D = {}
        for line in open(dj):
            if '"infer": true' in line:
                r = json.loads(line)
                D.setdefault(r["scene"], {})[int(r["now"])] = r
        for r in json.load(open(f))["rollouts"]:
            c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
            sc = r["clipgt_id"]
            if not os.path.exists(c) or sc not in D:
                continue
            rows = list(csv.DictReader(open(c)))
            g = {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}
            ev, sf, rollmax, ro = rollout_events(g, D[sc])
            fa = F.get(sc, {})
            score = float(fa.get("score") or r.get("score") or 0)
            cls = fa.get("fclass") or ("ok" if score >= 0.98 else (r.get("failure_reason") or "partial"))
            t_ev = float(fa["t_ev_s"]) if fa.get("t_ev_s") else None
            base = dict(read=label, scene=sc[7:15], score=round(score, 3), cls=cls, t_fail=t_ev, sf=sf["kind"],
                        sf_t=sf["t_first"], roll_max=round(rollmax, 1), **ro)
            out.append(dict(base, events=ev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", nargs="+", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    allrows, summ, roll_rows = [], [], []
    for spec in a.read:
        p = spec.split("=")
        label, root, facts = p[0], p[1], (p[2] if len(p) > 2 else "")
        R = read(label, root, facts)
        roll_rows += [{k: x for k, x in r.items() if k != "events"} | dict(n_ev=len(r["events"])) for r in R if r["sf"] or r["events"] or r["roll_max"] > 5]
        for r in R:
            for e in r["events"]:
                allrows.append({**{k: x for k, x in r.items() if k != "events"}, **e})
        n = len(R)
        S = [r for r in R if r["events"]]
        fast = [r for r in S if any(e["v"] >= 15 for e in r["events"])]
        pre = lambda r: r["t_fail"] is None or r["events"][0]["t"] <= r["t_fail"] + 0.05  # noqa: E731
        zs = [r for r in S if r["score"] == 0 and pre(r)]
        ps = [r for r in S if 0 < r["score"] < 0.98]
        brake = sum(1 for r in S for e in r["events"] if e["uacc_min"] <= -4)
        straight = sum(1 for r in S for e in r["events"] if abs(e.get("p_kap1", 1)) < 0.005)
        summ.append(dict(read=label, n=n, mean=np.mean([r["score"] for r in R]) if R else 0, zeros=sum(r["score"] == 0 for r in R),
                         ev=sum(len(r["events"]) for r in S), scenes=len(S), scenes15=len(fast), sf=sum(bool(r["sf"]) for r in R),
                         z_pre=len(zs), loss_z=len(zs), part=len(ps), loss_p=sum(1 - r["score"] for r in ps),
                         brake=brake, straight=straight, left=sum(1 for r in S for e in r["events"] if e["dir"] == "left"),
                         roll10=sum(r["roll_max"] > 10 for r in R)))
        print(label, summ[-1], flush=True)
    keys = list(dict.fromkeys(k for r in allrows for k in r))
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(allrows)
    keys = list(dict.fromkeys(k for r in roll_rows for k in r))
    with open(a.csv.replace(".csv", "_rollouts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(roll_rows)
    L = ["| read | rollouts | mean | zeros | events | scenes | scenes >= 15 m/s | sf_detect spins | zeros, event first | partials with event "
         "(loss) | events braking <= -4 | events on straight plan | left | roll > 10 deg |", "|" + ":--|" + "--:|" * 13]
    for s in summ:
        L.append(f"| {s['read']} | {s['n']} | {s['mean']:.3f} | {s['zeros']} | {s['ev']} | {s['scenes']} | {s['scenes15']} | {s['sf']} | "
                 f"{s['z_pre']} | {s['part']} ({s['loss_p']:.2f}) | {s['brake']} | {s['straight']} | {s['left']} | {s['roll10']} |")
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
