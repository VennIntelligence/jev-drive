"""Figure for results/op_control_stack.md: raw plan vs openpilot's command vs realised heading, one spin launch and one normal turn.
Per scene (columns): top, heading rates in deg/s - the rate a constant-curvature arc through the plan's 1 s point implies (2 * phi1 per s, raw plan),
openpilot's desired curvature x v (the command), and the realised v * tan(steer) / L; bottom, heading since the start, exam base (PR#57 iLQR) vs opctrl.
    python opctrl_fig.py <out.png> <scene_label> <base_zs_steps.jsonl> <opctrl_zs_steps.jsonl> [<label> <base> <opctrl>]
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

L = 2.7


def load(f):
    recs = [json.loads(x) for x in open(f) if '"step"' in x]
    t = np.array([r["t"] for r in recs])
    th = np.degrees(np.unwrap([r["theta"] for r in recs]))
    v = np.array([r["v"] for r in recs])
    st = np.array([r["steer"] for r in recs])
    phi = []
    for r in recs:
        p = np.array(r["plan"], float)
        phi.append(math.degrees(math.atan2(p[1, 0], p[1, 1])) if len(p) > 1 and np.linalg.norm(p[1]) >= 0.3 else np.nan)
    k = np.array([r.get("kappa") if r.get("kappa") is not None else np.nan for r in recs], float)
    return dict(t=t, th=th - th[0], v=v, rate=np.degrees(v * np.tan(st) / L), phi=np.array(phi), cmd=np.degrees(k * v))


def main():
    out, rest = sys.argv[1], sys.argv[2:]
    cases = [rest[i:i + 3] for i in range(0, len(rest), 3)]
    fig, ax = plt.subplots(2, len(cases), figsize=(6.2 * len(cases), 6.4), sharex="col", squeeze=False)
    for j, (lab, fb, fo) in enumerate(cases):
        b, o = load(fb), load(fo)
        a = ax[0, j]
        a.plot(o["t"], 2 * o["phi"], color="#9aa5b1", lw=1.2, label="raw plan (opctrl run): 2 x phi1")
        a.plot(o["t"], o["cmd"], color="#2a6fdb", lw=1.6, label="openpilot command: kappa_des x v")
        a.plot(o["t"], o["rate"], color="#e07b00", lw=1.6, ls="--", label="realised (opctrl)")
        a.plot(b["t"], b["rate"], color="#c0392b", lw=1.0, alpha=0.7, label="realised (base, iLQR on the plan)")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylim(-40, 40)
        a.set_title(lab, fontsize=10)
        a.set_ylabel("heading rate (deg/s, + right)")
        a.legend(fontsize=7, loc="upper left")
        a = ax[1, j]
        a.plot(b["t"], b["th"], color="#c0392b", lw=1.6, label="base (PR#57 iLQR)")
        a.plot(o["t"], o["th"], color="#2a6fdb", lw=1.6, label="opctrl (openpilot lateral path)")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylabel("heading since start (deg)")
        a.set_xlabel("simulator time (s)")
        a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
