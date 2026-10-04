"""Counts of the route-polyline material: samples, turns >= 25 deg within the horizon by angle bin, sanity checks. CPU only.

  envs/jevdrive on the box:  python experiments/op_route_cmd/scripts/route_report.py   (writes results/counts.json and results/counts_tables.md)

Reads $DATA_DIR/processed/op_route_cmd/{navtrain,wod}/route.npz (route_nav.py, route_wod.py).
"turn >= 25 deg" = the first complete turn (curvature > 0.02 /m merged over 8 m gaps, net heading change >= 25 deg) that starts > 2 m ahead and ends
inside the available path (<= 150 m); a turn the path ends in the middle of is not counted. A road bend of R < 50 m counts as a turn in WOD (no map);
in navtrain `turn_junction` says that the turn lies mostly inside a nuPlan lane connector / intersection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib")]
import route_poly as RP  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "processed" / "op_route_cmd"
RES = REPO / "experiments" / "op_route_cmd" / "results"
INV = data_dir() / "processed" / "op_common_cause" / "pair_inventory" / "navtrain_frames.parquet"


def load(name):
    z = np.load(ROOT / name / "route.npz")
    return {k: z[k] for k in z.files}


def block(z, m, label):
    """Counts for the rows `m` of one source."""
    d = dict(label=label, samples=int(m.sum()), clusters=int(len(set(z["cluster"][m]))))
    td = z["turn_deg"]
    has = m & ~np.isnan(td)
    d["full_150m"] = int((m & z["pmask"].all(1)).sum())
    d["plen_ge_50m"] = int((m & (z["plen"] >= 50)).sum())
    d["in_turn_now"] = int((m & z["in_turn"]).sum())
    d["turn_ge25"] = int(has.sum())
    d["turn_ge25_clusters"] = int(len(set(z["cluster"][has])))
    d["turn_ge25_s_le30"] = int((has & (z["turn_s"] <= 30)).sum())
    d["turn_ge25_s_le60"] = int((has & (z["turn_s"] <= 60)).sum())
    d["turn_ge25_left"] = int((has & (td > 0)).sum())
    d["turn_ge25_right"] = int((has & (td < 0)).sum())
    b = RP.turn_bin(td)
    d["by_bin"] = {n: dict(all=int((has & (b == i)).sum()), left=int((has & (b == i) & (td > 0)).sum()), right=int((has & (b == i) & (td < 0)).sum()))
                   for i, n in enumerate(RP.TURN_BIN_NAMES)}
    d["multi_turn_samples"] = int((m & (z["n_turn"] >= 2)).sum())
    if (z["jct_s"] >= 0).any():
        d["turn_ge25_junction"] = int((has & z["turn_junction"]).sum())
        d["turn_ge25_bend_only"] = int((has & ~z["turn_junction"]).sum())
        d["jct_ahead_on_path"] = int((m & ~np.isnan(z["jct_s"])).sum())
    return d


def events(cluster, frame, m, gap):
    """Number of runs of consecutive labelled frames (same cluster, frame step <= gap) among the rows m: one run ~ one turn seen from all its approach frames."""
    c, f = cluster[m], frame[m]
    o = np.lexsort((f, c))
    c, f = c[o], f[o]
    return int(1 + ((c[1:] != c[:-1]) | (f[1:] - f[:-1] > gap)).sum()) if len(c) else 0


def wod_chain_check(n=30000, seed=0):
    """Independent check of the link composition: the chained position at t = 5.0 s against the logged future[19] of the same frame."""
    sys.path.insert(0, str(Path(__file__).parent))
    import route_wod as W
    from jevdrive import waymo as WM
    df = WM.load_index().reset_index(drop=True)
    past, fut = WM.load_ego()
    sel = (df.split.isin(["train", "val"]) & df.has_future).to_numpy()
    key = {(s, f): i for i, (s, f) in enumerate(zip(df.sequence, df.frame))}
    nxt = np.array([key.get((s, f + W.LINK), -1) for s, f in zip(df.sequence, df.frame)])
    nxt[~sel] = -1
    th, t, ok, rms = W.fit_links(past, fut, nxt >= 0, nxt)
    rows = np.random.default_rng(seed).permutation(np.flatnonzero(ok))[:n]
    err = []
    for (p, _), r in zip(W.chain_paths(fut, nxt, ok, th, t, rows), rows):
        err.append(np.hypot(*(p[20] - fut[r, 19, :2])))
    err = np.array(err)
    return dict(links_candidate=int((nxt >= 0).sum()), links_ok=int(ok.sum()), rms_q=np.nanpercentile(rms, [50, 90, 99, 99.9]).round(3).tolist(),
                pos_err_5s_q=np.percentile(err, [50, 90, 99, 100]).round(3).tolist(), n=len(err))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    nav, wod = load("navtrain"), load("wod")
    out["navtrain"] = block(nav, np.ones(len(nav["id"]), bool), "navtrain (all tokens)")
    # decision-93 junction segments
    inv = pd.read_parquet(INV).set_index("token").loc[nav["id"]]
    fr_nav = inv.frame_idx.to_numpy()
    out["navtrain"]["turn_events"] = events(nav["cluster"], fr_nav, ~np.isnan(nav["turn_deg"]), 2)
    pe = (inv.status.to_numpy() == "branch") & inv.classes.str.contains(",").to_numpy() & (inv.taken.to_numpy() != "")
    out["navtrain_pair_eligible"] = block(nav, pe, "navtrain, decision-93 pair-eligible frames (branch, >= 2 exit classes, taken known)")
    seg = inv[pe].groupby(["log", "node"]).ngroups
    out["navtrain_pair_eligible"]["segments"] = int(seg)
    out["navtrain_pair_eligible"]["turn_events"] = events(nav["cluster"], fr_nav, pe & ~np.isnan(nav["turn_deg"]), 2)
    # sanity: map-derived taken class vs the sign of the hindsight turn
    taken, td, ts = nav["taken_cls"], nav["turn_deg"], nav["turn_s"]
    near = pe & ~np.isnan(td) & (ts <= nav["jct_dist"] + 40)
    lr = near & np.isin(taken, ["left", "right"])
    out["sanity_taken_vs_turn"] = dict(
        left_right_taken_with_turn=int(lr.sum()),
        sign_agrees=float(((taken[lr] == "left") == (td[lr] > 0)).mean()),
        straight_taken=int((pe & (taken == "straight")).sum()),
        straight_taken_without_turn_ge25_within_60m=float((np.isnan(td) | (ts > 60))[pe & (taken == "straight")].mean()),
        left_right_taken_total=int((pe & np.isin(taken, ["left", "right"])).sum()))
    for sp in ("train", "val"):
        m = wod["split"] == sp
        out[f"wod_{sp}"] = block(wod, m, f"WOD-E2E {sp} (frames with a logged future)")
        fr = np.array([int(i.rsplit("-", 1)[1]) for i in wod["id"]])
        out[f"wod_{sp}"]["turn_events"] = events(wod["cluster"], fr, m & ~np.isnan(wod["turn_deg"]), 3)
    # WOD intent vs turn sign, by distance to the turn
    has = ~np.isnan(wod["turn_deg"])
    out["wod_intent_vs_turn_by_dist"] = {nm: {c: [int((m_ & (wod["cmd"] == c) & (wod["turn_deg"] > 0)).sum()), int((m_ & (wod["cmd"] == c) & (wod["turn_deg"] < 0)).sum())]
                                              for c in np.unique(wod["cmd"])} for nm, m_ in (("s<=15", has & (wod["turn_s"] <= 15)), ("15<s<=50", has & (wod["turn_s"] > 15) & (wod["turn_s"] <= 50)), ("s>50", has & (wod["turn_s"] > 50)))}
    out["wod_intent_vs_turn"] = {c: dict(n=int((has & (wod["cmd"] == c)).sum()), left=int((has & (wod["cmd"] == c) & (wod["turn_deg"] > 0)).sum()),
                                         right=int((has & (wod["cmd"] == c) & (wod["turn_deg"] < 0)).sum())) for c in np.unique(wod["cmd"])}
    for k in ("plen", "dur"):
        if k in wod:
            out[f"wod_{k}_quantiles"] = np.percentile(wod[k], [5, 25, 50, 75, 95]).round(1).tolist()
    out["nav_plen_quantiles"] = np.percentile(nav["plen"], [5, 25, 50, 75, 95]).round(1).tolist()
    out["wod_chain_check"] = wod_chain_check()
    json.dump(out, open(RES / "counts.json", "w"), indent=1)
    # tables
    L = ["| source | samples | clusters | turn >= 25 deg (<= 150 m) | left / right | <= 30 m | <= 60 m | 25-45 | 45-75 | 75-105 | 105-135 | >= 135 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("navtrain", "navtrain_pair_eligible", "wod_train", "wod_val"):
        d = out[k]
        L.append(f"| {d['label']} | {d['samples']:,} | {d['clusters']:,} | {d['turn_ge25']:,} | {d['turn_ge25_left']:,} / {d['turn_ge25_right']:,} | {d['turn_ge25_s_le30']:,} | "
                 f"{d['turn_ge25_s_le60']:,} | " + " | ".join(f"{d['by_bin'][n]['all']:,}" for n in RP.TURN_BIN_NAMES) + " |")
    (RES / "counts_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("sanity", "wod_chain", "wod_intent", "nav_plen", "wod_plen", "wod_dur"))}, indent=1))
    print(json.dumps({k: out[k] for k in ("navtrain", "navtrain_pair_eligible")}, indent=1))


if __name__ == "__main__":
    main()
