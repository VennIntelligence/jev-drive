"""Launch behaviour of the three arms (same-day base, decision 118 opctrl, opctrl_long). Left: share of the 64 runs standing (v < 0.3 m/s) by step, among runs still going.
Middle: median speed of the runs still going. Right: one launch (scene-0013-medium-00 is a base spin; scene-0167-easy-00 completes in the base): speed and the action head's
acceleration (model units) of opctrl_long against the base's speed.
    python opctrl_long_fig.py <opctrl_long/closed> <opctrl/closed> <out.png>   (box, envs/hugsim python; base3 is read from <opctrl/closed>)"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive.bench.compat import trace_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

L, O, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ARMS = [("base (same day)", O, "cinque-fixed-base3", "tab:gray"), ("opctrl (d118)", O, "cinque-opctrl", "tab:blue"), ("opctrl_long", L, "cinque-opctrl-long", "tab:red")]


def runs(root, tag):
    d = {}
    for r in csv.DictReader(open(root / "results.csv")):
        if r["tag"] == tag and r["end"] != "crash":
            d[r["scenario"]] = [x for x in map(json.loads, open(trace_dir(r, root) / "zs_steps.jsonl")) if "step" in x]
    return d


N = 120
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
data = {}
for name, root, tag, c in ARMS:
    R = data[name] = runs(root, tag)
    still, med = [], []
    for k in range(N):
        v = [r[k]["v"] for r in R.values() if len(r) > k]
        still.append(np.mean(np.array(v) < 0.3) if len(v) >= 8 else np.nan)
        med.append(np.median(v) if len(v) >= 8 else np.nan)
    ax[0].plot(np.arange(N) * 0.25, still, c=c, label=name)
    ax[1].plot(np.arange(N) * 0.25, med, c=c, label=name)
ax[0].set(xlabel="time (s)", ylabel="share standing (v < 0.3 m/s), runs still going", title="Standing share")
ax[1].set(xlabel="time (s)", ylabel="median speed (m/s)", title="Median speed")
ax[0].legend()
sc = "scene-0167-easy-00"
for name, root, tag, c in ARMS:
    r = data[name].get(sc)
    if r:
        ax[2].plot(np.arange(min(N, len(r))) * 0.25, [x["v"] for x in r[:N]], c=c, label=f"{name} v")
r = data["opctrl_long"].get(sc)
if r and "accel" in r[0]:
    ax[2].plot(np.arange(min(N, len(r))) * 0.25, [x["accel"] for x in r[:N]], c="k", ls=":", label="opctrl_long action accel (model units)")
ax[2].set(xlabel="time (s)", ylabel="m/s, m/s^2", title=sc)
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=130)
