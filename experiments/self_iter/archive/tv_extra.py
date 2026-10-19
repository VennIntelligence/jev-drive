"""TV-AUDIT one-off (2026-10-19): extra census stats on runs/self_iter/tv_census/frames.npz (new logs, distance to navtrain tokens, lead-moving /
ego-standing frames, red-line no-lead events) -> stdout, saved as results/tv_census_extra.json. Run via the pool in the navsim2 venv."""
import json, sys, numpy as np
sys.path[:0] = ["/root/autodl-tmp/ujs/jev-drive"]
from jevdrive.data import splits
z = np.load("/root/autodl-tmp/ujs/runs/self_iter/tv_census/frames.npz")
tok, log, fr = z["token"].astype(str), z["log"].astype(str), z["frame"].astype(int)
cell = z["cell"].astype(str); v0, d4, lv, lg = z["v0"], z["d4"], z["lead_v"], z["lead_gap"]
nav = np.isin(tok, list(splits.load("navsim/navtrain").members))
tr = np.isin(log, list(splits.load("navsim/tv-clean-train-logs").members))
navlogs = set(log[nav]); out = {}
out["logs_eligible"] = len(set(log)); out["logs_with_navtrain"] = len(navlogs)
m = tr & ~nav; out["train_new_frames"] = int(m.sum())
out["train_new_frames_in_logs_without_navtrain"] = int((m & ~np.isin(log, list(navlogs))).sum())
# distance (frames, 2 Hz) to the nearest navtrain token of the same log
key = {}
for l, f in zip(log[nav], fr[nav]): key.setdefault(l, []).append(f)
key = {l: np.sort(v) for l, v in key.items()}
dist = np.full(len(tok), 10**6)
for i in np.flatnonzero(m):
    a = key.get(log[i])
    if a is not None:
        j = np.searchsorted(a, fr[i]); c = [abs(a[k] - fr[i]) for k in (j - 1, j) if 0 <= k < len(a)]; dist[i] = min(c)
out["train_new_dist_to_navtrain_frames"] = {k: int((m & sel).sum()) for k, sel in (("<=2 (1 s)", dist <= 2), ("3-8 (1.5-4 s)", (dist > 2) & (dist <= 8)), (">8 (> 4 s)", dist > 8))}
stand = (v0 < 0.3) & (d4 < 1.0)
# lead already moving, ego still standing (the timing cue before a launch)
lm = (v0 < 0.3) & (lg <= 15) & (lv >= 0.5)
def ev(mask):
    o = np.lexsort((fr, log)); l, f, mm = log[o], fr[o], mask[o]; i = np.flatnonzero(mm)
    return int(np.r_[True, (l[i][1:] != l[i][:-1]) | (f[i][1:] - f[i][:-1] > 1)].sum()) if len(i) else 0
for nm, mk in (("navtrain", nav), ("train_new", m)):
    out[f"{nm}_lead_moving_ego_standing"] = dict(frames=int((mk & lm).sum()), events=ev(mk & lm))
    out[f"{nm}_lead_moving_ego_standing_human_holds_4s"] = int((mk & lm & stand).sum())
# red-line no-lead standing events: does the human launch within the log right after (green switch seen)?
ty, dF, st0, conn, la = z["ln_type"], z["ln_dF"], z["ln_st0"], z["ln_conn"], z["lead_any"]
red = (dF >= -2) & (dF <= 10) & (ty == 2) & (st0 == 1) & conn & (la > dF + 2) & stand
o = np.lexsort((fr, log)); pos = {(l, f): k for k, (l, f) in enumerate(zip(log, fr))}
evs = []; cur = None
for k in o:
    if red[k] and tr[k] and not nav[k]:
        if cur and cur[0] == log[k] and fr[k] - cur[2] <= 1: cur[2] = fr[k]
        else:
            if cur: evs.append(cur)
            cur = [log[k], fr[k], fr[k]]
if cur: evs.append(cur)
dur = np.array([(e[2] - e[1] + 1) * 0.5 for e in evs])
went = sum(1 for e in evs if any(pos.get((e[0], e[2] + d)) is not None and cell[pos[(e[0], e[2] + d)]] in ("G_L", "G_N") for d in (1, 2, 3)))
out["red_nolead_events_train"] = dict(n=len(evs), launch_after=went, dur_s_p50=float(np.median(dur)), dur_s_p90=float(np.percentile(dur, 90)), logs=len({e[0] for e in evs}), drives=len({"_".join(e[0].split("_")[:2]) for e in evs}))
print(json.dumps(out, indent=1))
