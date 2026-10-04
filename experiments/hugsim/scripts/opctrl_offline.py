"""Offline readings for plans/2026-10-05-op-control-stack-prereg.md from the open-loop replay (opctrl_replay.py) of the 64 exam-base runs.

  signs     openpilot curvature (+ right?) against the plan's 1 s direction phi1 (+ right) and HUGSIM's steer against its heading change
  transfer  model side: desired curvature per degree of phi1 (instantaneous heading rate per step, v * kappa * 0.25 s, per degree), by speed bin
  c         the same readout as lowspeed_ctrl.md (heading realised over the next 0.25 s step per degree of phi1, regression through the origin,
            |phi1| < 15 deg, |plan 1 s point| >= 0.3 m, cluster bootstrap over runs) for (a) the logged PR #57 iLQR run, (b) openpilot's path
            (lib/op_ctrl.OpLateral on the replayed action curvature and the logged speed); c2 = the heading over the next two steps
  growth    linear launch loop of decisions 100 / 111: phi1_k = G * (theta_k - theta_{k-6}); per-step growth z = spectral radius, iLQR (no delay,
            c per bin) against openpilot's path (one-step delay, transfer per bin), G = 5.85 (HUGSIM spin step 1), 2.27 (step 2), 9.34 (WOD step 1)
    python opctrl_offline.py <replay_dir> <out_dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
import op_ctrl as OC  # noqa: E402

BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (0.0, 3.0)]
RNG = np.random.default_rng(0)


def load(d):
    runs = []
    for f in sorted(Path(d).glob("*.json")):
        r = json.load(open(f))
        n = len(r["k_act"])
        r["th"] = np.degrees(np.unwrap(np.array(r["theta"][:n])))
        r["v"] = np.array(r["v"][:n])
        st = OC.OpLateral()
        r["k_real"] = np.array([st.step(k, v, 0.25)[0] for k, v in zip(r["k_act"], r["v"])])    # mean realised curvature over step j -> j+1
        pl = np.array([p[1] if len(p) > 1 else [0.0, 0.0] for p in r["plan_logged"]], float)
        r["phi1_log"] = np.degrees(np.arctan2(pl[:, 0], pl[:, 1]))
        r["ok_log"] = np.linalg.norm(pl, axis=1) >= 0.3
        r["ok"] = np.linalg.norm(np.array(r["p1"], float), axis=1) >= 0.3
        runs.append(r)
    return runs


def reg(rows):
    """rows: list of (run, x, y); slope through the origin and a run-cluster bootstrap CI."""
    if len(rows) < 5:
        return None
    runs = sorted({r for r, _, _ in rows})
    by = {k: np.array([(x, y) for r, x, y in rows if r == k]) for k in runs}
    def s(keys):
        a = np.concatenate([by[k] for k in keys])
        return float((a[:, 0] * a[:, 1]).sum() / max((a[:, 0] ** 2).sum(), 1e-12))
    bs = [s(RNG.choice(runs, len(runs))) for _ in range(1000)]
    return dict(c=round(s(runs), 4), lo=round(float(np.percentile(bs, 2.5)), 4), hi=round(float(np.percentile(bs, 97.5)), 4), n=len(rows), runs=len(runs))


def table(runs, x_of, y_of, ok_of):
    out = {}
    for lo, hi in BINS:
        rows = []
        for r in runs:
            for j in range(len(r["v"]) - 2):
                x = x_of(r, j)
                if ok_of(r, j) and abs(x) < 15 and lo <= r["v"][j] < hi:
                    y = y_of(r, j)
                    if y is not None and np.isfinite(y):
                        rows.append((r["scenario"], x, y))
        out[f"{lo:g}-{hi:g}"] = reg(rows)
    return out


def growth(cs, G, delay):
    """Growth per step of theta_{k+1} = theta_k + c * G * (theta_{k-d} - theta_{k-d-6}) (decision 100 / 111 window kernel zwin; d = delay steps)."""
    n = 8 + delay
    A = np.zeros((n, n))
    A[0, 0] += 1.0
    A[0, delay] += cs * G
    A[0, delay + 6] -= cs * G
    A[1:, :-1] = np.eye(n - 1)
    ev = np.linalg.eigvals(A)
    ev = np.delete(ev, np.argmin(np.abs(ev - 1.0)))         # a held heading offset is neutral (lambda = 1 always exists)
    return float(np.max(np.abs(ev)))


def main(d, out):
    runs = load(d)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R = {"n_runs": len(runs), "n_steps": int(sum(len(r["v"]) for r in runs))}
    # reproduction of the logged plans by the CPU replay of the mp4 frames
    e = np.concatenate([np.abs(r["lat25_err"]) for r in runs])
    dphi = np.concatenate([np.abs(np.array(r["phi1"]) - r["phi1_log"])[r["ok"] & r["ok_log"]] for r in runs])
    R["replay"] = dict(lat25_err_median=round(float(np.median(e)), 3), lat25_err_p90=round(float(np.percentile(e, 90)), 3),
                       phi1_absdiff_median=round(float(np.median(dphi)), 3), phi1_absdiff_p90=round(float(np.percentile(dphi, 90)), 3))
    # signs
    m = [(np.sign(k), np.sign(p)) for r in runs for k, p, ok in zip(r["k_act"], r["phi1"], r["ok"]) if ok and abs(p) > 0.5 and abs(p) < 15]
    dth = [(np.sign(s), np.sign(r["th"][j + 1] - r["th"][j])) for r in runs for j, s in enumerate(r["steer"][:-1])
           if abs(s) > 1e-3 and r["v"][j + 1] > 0.5]
    mf = [(np.sign(k), np.sign(p)) for r in runs for k, p, ok, v in zip(r["k_act"], r["phi1"], r["ok"], r["v"]) if ok and 0.5 < abs(p) < 15 and v >= 3]
    R["signs_v3plus"] = dict(kact_vs_phi1_agree=round(float(np.mean([a == b for a, b in mf])), 3), n=len(mf))
    R["signs"] = dict(kact_vs_phi1_agree=round(float(np.mean([a == b for a, b in m])), 3), n1=len(m),
                      steer_vs_dtheta_agree=round(float(np.mean([a == b for a, b in dth])), 3), n2=len(dth))
    # model side: instantaneous heading per step per degree of phi1
    R["transfer_act"] = table(runs, lambda r, j: r["phi1"][j], lambda r, j: math.degrees(r["v"][j] * r["k_act"][j] * 0.25), lambda r, j: r["ok"][j])
    R["transfer_plan"] = table(runs, lambda r, j: r["phi1"][j], lambda r, j: math.degrees(r["v"][j] * r["k_plan"][j] * 0.25), lambda r, j: r["ok"][j])
    # c: logged iLQR vs openpilot path (open loop on the same inputs)
    R["c_ilqr_logged"] = table(runs, lambda r, j: r["phi1_log"][j], lambda r, j: r["th"][j + 1] - r["th"][j], lambda r, j: r["ok_log"][j])
    R["c2_ilqr_logged"] = table(runs, lambda r, j: r["phi1_log"][j], lambda r, j: r["th"][j + 2] - r["th"][j], lambda r, j: r["ok_log"][j])
    op_dth = lambda r, j, k=0: math.degrees(r["v"][j + k + 1] * r["k_real"][j + k] * 0.25)  # noqa: E731
    R["c_op"] = table(runs, lambda r, j: r["phi1"][j], lambda r, j: op_dth(r, j), lambda r, j: r["ok"][j])
    R["c2_op"] = table(runs, lambda r, j: r["phi1"][j], lambda r, j: op_dth(r, j) + op_dth(r, j, 1), lambda r, j: r["ok"][j])
    # loop growth
    gr = {}
    for b in ("0-1", "1-2", "2-3"):
        ci, ct = R["c_ilqr_logged"][b], R["transfer_act"][b]
        for G in (2.27, 5.85, 9.34):
            gr[f"{b}|G{G}"] = dict(ilqr=round(growth(ci["c"], G, 0), 3) if ci else None, op=round(growth(max(ct["c"], 0.0), G, 1), 3) if ct else None,
                                   op_ci_hi=round(growth(ct["hi"], G, 1), 3) if ct else None)
    R["growth"] = gr
    R["growth_check_d111"] = dict(c=0.19, G=9.34, z=round(growth(0.19, 9.34, 0), 3), z_c_018=round(growth(0.018, 9.34, 0), 3))
    json.dump(R, open(out / "offline.json", "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
