"""Render statistics of a carla_pairs_render run dir: poses, ticks, rate, settle ticks, Large-Map load and teleport cost, lag check.

  python carla_pairs_stats.py <render dir>
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LARGE = ("Town11", "Town12", "Town13", "Town15")


def main():
    rows, loads = [], []
    for f in sorted(Path(sys.argv[1]).glob("render_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            (loads if r.get("event") == "load" else rows).append(r)
    ok = [r for r in rows if "error" not in r]
    print("poses", len(ok), "errors", len(rows) - len(ok), "unsettled", sum(bool(r.get("unsettled")) for r in ok))
    ticks = np.array([r["renders"] for r in ok])
    wall = np.array([r["wall_settle"] + r["wall_frames"] for r in ok])
    print("renders per pose mean %.1f; wall per pose mean %.2f s (p50 %.2f p90 %.2f); render+pack rate %.1f ticks/s per server"
          % (ticks.mean(), wall.mean(), np.median(wall), np.percentile(wall, 90), ticks.sum() / wall.sum()))
    by = defaultdict(list)
    for r in ok:
        by["large" if r["town"] in LARGE else "small"].append(r)
    for k, v in by.items():
        st = np.array([r["settle_ticks"] for r in v])
        ws = np.array([r["wall_settle"] for r in v])
        wf = np.array([r["wall_frames"] for r in v])
        print("%s maps: %d poses, settle ticks mean %.1f p90 %d max %d; settle wall mean %.2f s p90 %.2f max %.2f; frames wall mean %.2f s"
              % (k, len(v), st.mean(), np.percentile(st, 90), st.max(), ws.mean(), np.percentile(ws, 90), ws.max(), wf.mean()))
    first = [r["wall_settle"] for r in ok if r["settle_ticks"] > 4]
    print("poses with > 4 settle ticks: %d (%.0f%%)" % (len(first), 100 * len(first) / max(len(ok), 1)))
    for k in ("small", "large"):
        w = [l["wall"] for l in loads if (l["town"] in LARGE) == (k == "large")]
        if w:
            print("town load (%s): n %d mean %.1f s max %.1f s" % (k, len(w), np.mean(w), np.max(w)))
    lag = [x for r in ok for x in r.get("lag_diff", [])]
    mv = [x for r in ok for x in r.get("move_diff", [])]
    if lag:
        print("lag check: diff between a frame and the next tick without moving: mean %.3f max %.3f; between consecutive poses: mean %.3f p10 %.3f"
              % (np.mean(lag), np.max(lag), np.mean(mv), np.percentile(mv, 10)))
    print("mean luminance of the t0 road frame: %.1f (min %.1f max %.1f)" % (np.mean([r["mean_y"] for r in ok]), min(r["mean_y"] for r in ok),
                                                                         max(r["mean_y"] for r in ok)))


if __name__ == "__main__":
    main()
