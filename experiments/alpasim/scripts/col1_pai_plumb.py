"""COL1, PAI track: plumbing checks of what the driver was fed, from col1_pai_extract.py's logs.pkl. One row per scene.
  col1_pai_plumb.py <logs.pkl> [...]            (any python with numpy; simulator state is the reference, a label only)

Columns: fed speed / acceleration against finite differences of the simulated ego pose; frame latency (now - t0) and decisions with a
synthesised slot; the route's first waypoint against the logged path (lateral distance: the route's frame and source); the command fed
(left / straight / right) against the same rule applied to the logged path from the logged pose; tracking error 0.5 s after a decision
(executed pose against the plan's 0.5 s pose); time of the first decision after the start of the rollout.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import col1_lib as L  # noqa: E402

print("| scene | decisions | v fed - sim, median / p95 abs m/s | a fed - sim, median / p95 abs m/s^2 | latency max ms | cold decisions | "
      "route wp0 to logged path, median / p95 abs m | route wp0 distance m | cmd fed L / S / R | same rule on the log L / S / R | "
      "tracking error at 0.5 s, p95 lat / lon m | first decision s | zero flag |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
tot = np.zeros((2, 3), int)
for f in sys.argv[1:]:
    for s, x in sorted(L.load(f).items()):
        off, ego, gt = L.center_off(x), x["actors"]["EGO"], x["logged"][0]["traj"]
        rig_gt = np.c_[gt[:, 0], L.to_rig(gt, off)]
        rec = [r for r in x.get("rec", []) if r.get("infer")]
        if not rec:
            continue
        now, t0 = (np.array([r[k] for r in rec]) for k in ("now", "t0"))
        e = np.array([r["ego"] for r in rec])
        rigs = L.to_rig(L.interp(ego, t0), off)
        vfd, vfed = L.speed(ego, t0), np.hypot(e[:, 4], e[:, 5]) * 10
        afd = np.interp(t0, ego[:, 0], np.gradient(L.speed(ego, ego[:, 0]), ego[:, 0] * 1e-6))
        d_route, cmd_log, wpd = [], [], []
        for r, rg in zip(rec, rigs):
            if not r["route0"]:
                d_route.append(np.nan), cmd_log.append(-1), wpd.append(np.nan)
                continue
            wp = np.array(r["route0"])
            d_route.append(L.lat_to_path(L.into(rg, rig_gt[:, 1:3]), wp[None])[0, 0])
            wpd.append(np.hypot(*wp))
            g0 = L.to_rig(L.interp(gt, r["t0"]), off)[0]                       # the rule on the logged path from the logged pose
            q = L.into(g0, rig_gt[rig_gt[:, 0] > r["t0"], 1:3])
            j = np.flatnonzero(np.hypot(*q.T) >= np.hypot(*wp)) if len(q) else []
            cmd_log.append(-1 if not len(j) else 0 if q[j[0], 1] > 2 else 2 if q[j[0], 1] < -2 else 1)
        d_route, cmd, cmd_log = np.array(d_route), np.array([r["cmd"] for r in rec]), np.array(cmd_log)
        ex = L.to_rig(L.interp(ego, t0 + 500_000), off)
        te = np.array([L.into(rg, q[:2])[0] - np.array(r["poses"])[0][:2] for r, rg, q in zip(rec, rigs, ex)])[: max(len(rec) - 6, 1)]
        c1, c2 = [int((cmd == i).sum()) for i in range(3)], [int((cmd_log == i).sum()) for i in range(3)]
        tot += np.array([c1, c2])
        print(f"| {s[7:15]} | {len(rec)} | {np.median(vfed - vfd):+.2f} / {np.percentile(np.abs(vfed - vfd), 95):.2f} | "
              f"{np.median(e[:, 6] * 3 - afd):+.2f} / {np.percentile(np.abs(e[:, 6] * 3 - afd), 95):.2f} | {(now - t0).max() / 1e3:.0f} | "
              f"{sum(r['n_real'] < 8 for r in rec)} | {np.nanmedian(np.abs(d_route)):.2f} / {np.nanpercentile(np.abs(d_route), 95):.2f} | "
              f"{np.nanmedian(wpd):.0f} | {c1[0]} / {c1[1]} / {c1[2]} | {c2[0]} / {c2[1]} / {c2[2]} | "
              f"{np.percentile(np.abs(te[:, 1]), 95):.2f} / {np.percentile(np.abs(te[:, 0]), 95):.2f} | {(rec[0]['now'] - ego[0, 0]) * 1e-6:.2f} | "
              f"{x['summary'].get('failure_reason') or ''} |")
print(f"\ncommands fed L / S / R: {tot[0].tolist()}; the same rule on the logged path from the logged pose: {tot[1].tolist()}")
