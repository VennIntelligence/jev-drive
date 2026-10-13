#!/usr/bin/env python
"""Steer command vs time around the turn onset (rsg_turns.py's car onset), -2 .. +6 s, one panel per route, one line per arm.
usage: steer_plot.py <out.png> <label>=<run dir> [<label>=<run dir> ...]   (carla env: numpy + matplotlib or cv2)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsg_turns import read  # noqa: E402

out, arms = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")   # Okabe-Ito, up to 6 arms (was 3: a 4th arm was dropped)
routes = sorted({p.parent.name for _, d in arms for p in Path(d).glob("attempts/*/1")})
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axs = plt.subplots(len(routes), 1, figsize=(9, 2.6 * len(routes)), sharex=True)
axs = np.atleast_1d(axs)
for ax, rid in zip(axs, routes):
    for (label, d), col in zip(arms, COLORS):
        att = Path(d) / "attempts" / rid / "1"
        if not (att / "ticks.jsonl").exists():
            continue
        r = read(att)
        t0 = r.get("car_onset_t")
        tk = [json.loads(l) for l in open(att / "ticks.jsonl")]
        t = np.array([x["t"] for x in tk])
        st = np.array([x["steer"] for x in tk])
        if t0 is None:
            ax.plot([], [], color=col, label="%s: no turn onset" % label)
            continue
        m = (t >= t0 - 2) & (t <= t0 + 6)
        ax.plot(t[m] - t0, st[m], color=col, label="%s (onset %+.1f m vs entry, DS %s)" % (label, r["car_onset_m"], r.get("DS")))
        ev = r.get("event")
        if ev and t0 - 2 <= ev["t"] <= t0 + 6:
            ax.axvline(ev["t"] - t0, color=col, ls=":", lw=1)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_ylabel("steer (CARLA, right +)")
    ax.set_title("route %s (%s turn)" % (rid, r.get("turn", "?")), fontsize=10)
    ax.legend(fontsize=8, loc="best")
axs[-1].set_xlabel("time from the car's turn onset (s); dotted = first event")
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
