"""A small sample of negative route polylines from navtrain (generator: lib/route_neg.py) + a sheet for the visual sanity check.

  envs/navsim2 on the box (nuPlan map API for the tier of N1 and for the map underlay of the sheet; CPU only):
  NUPLAN_MAPS_ROOT=... NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=... python experiments/op_route_cmd/scripts/route_neg_sample.py

Candidates (navtrain frames of route.npz; all train membership, <= 2 per log):
  N1 exit   status branch (inventory), the ego lane lacks an exit class X in {left, right, straight}; tier A = no lane of the ego roadblock has X,
            tier B = another lane has it (the "nonexistent" exit exists for another lane: may be a legal route, drop or keep as weak)
  N2 side   no_junction_30m, no connector / intersection on the 150 m driven path, v >= 3 m/s, full 150 m
  N3 wrong  v >= 3 m/s, not in a junction, path >= 100 m (the oncoming side by `map_location`: Singapore drives on the left)
  N4 uturn  as N2
Output: results/negatives_sample.npz (+ .csv) and figs/negatives_sample.png (grey = map lanes, green = logged route, red = the negative).
"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(Path(__file__).parent), str(REPO / "experiments" / "op_common_cause" / "scripts")]
import pair_inv_navtrain as PI  # noqa: E402  (nuPlan map helpers of the decision-93 inventory)
import route_neg as RN  # noqa: E402
import route_nav as NV  # noqa: E402
import route_poly as RP  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

RES, FIGS = REPO / "experiments/op_route_cmd/results", REPO / "experiments/op_route_cmd/figs"
PLAN = {"N1_exit": 60, "N2_side": 40, "N3_wrong": 30, "N4_uturn": 30}
PER_LOG = 2


def pick(z, inv, rng):
    st, v, full = inv.status.to_numpy(), z["v0"], z["pmask"].all(1)
    nojct = np.isnan(z["jct_s"])
    cand = {"N1_exit": np.flatnonzero((st == "branch") & (z["plen"] >= inv.dist.fillna(0).to_numpy() + 30) & (v >= 1.0)),
            "N2_side": np.flatnonzero((st == "no_junction_30m") & nojct & full & (v >= 3.0)),
            "N3_wrong": np.flatnonzero((st != "in_junction") & (z["plen"] >= 100) & (v >= 3.0)),
            "N4_uturn": np.flatnonzero((st == "no_junction_30m") & nojct & full & (v >= 3.0))}
    out, used = [], {}
    for kind, n in PLAN.items():
        k = 0
        for i in rng.permutation(cand[kind]):
            key = (kind, z["cluster"][i])
            if used.get(key, 0) >= PER_LOG:
                continue
            used[key] = used.get(key, 0) + 1
            out.append((kind, int(i)))
            k += 1
            if k >= n * (2 if kind == "N1_exit" else 1):     # N1: draw extra, the tier split is taken after the map check
                break
    return out


def work(log, jobs, z, inv):
    fr = pickle.load(open(os.path.join(os.environ["OPENSCENE_DATA_ROOT"], "navsim_logs", "trainval", log + ".pkl"), "rb"))
    xs = np.array([(f["ego2global_translation"][0], f["ego2global_translation"][1], NV._quat_yaw(f["ego2global_rotation"])) for f in fr])
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xs[:, :2], axis=0).T))]
    m = PI.get_map(fr[0]["map_location"])
    lht = fr[0]["map_location"].startswith("sg-")
    res = []
    for kind, i in jobs:
        fi = int(inv.frame_idx.iloc[i])
        loc = NV.ego_path(xs, cum, fi)
        pos = RP.hindsight(loc, want_path=True)
        P = pos.pop("path")
        if P is None:
            continue
        rng = np.random.default_rng(1000 + i)
        kw, tier = {}, ""
        if kind == "N1_exit":
            from nuplan.common.actor_state.state_representation import Point2D
            st, lane = PI.find_lane(m, Point2D(float(xs[fi, 0]), float(xs[fi, 1])), xs[fi, 2])
            if lane is None:
                continue
            have = set(inv.classes.iloc[i].split(","))
            miss = [c for c in ("left", "right", "straight") if c not in have]
            if not miss:
                continue
            kw["missing"] = str(rng.choice(miss))
            kw["s_branch"] = float(inv.dist.iloc[i])
            tier = "A" if kw["missing"] not in PI.rb_classes(m, lane) else "B"
        neg = RN.make(kind, P, rng, lht=lht, **kw)
        if neg is None:
            continue
        # map underlay for the sheet: lane / connector polygons within 70 m, in the ego frame
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        objs = m.get_proximal_map_objects(Point2D(float(xs[fi, 0]), float(xs[fi, 1])), 70.0, [L.LANE, L.LANE_CONNECTOR])
        c, s = np.cos(xs[fi, 2]), np.sin(xs[fi, 2])
        polys = []
        for layer in objs.values():
            for o in layer:
                x, y = (np.array(a) for a in o.polygon.exterior.xy)
                polys.append(np.stack([c * (x - xs[fi, 0]) + s * (y - xs[fi, 1]), -s * (x - xs[fi, 0]) + c * (y - xs[fi, 1])], -1).astype(np.float32))
        res.append(dict(i=i, kind=kind, tier=tier, neg=neg, pos=pos, polys=polys, lht=lht))
    return res


def main():
    z = {k: v for k, v in np.load(data_dir() / "processed/op_route_cmd/navtrain/route.npz").items()}
    inv = pd.read_parquet(data_dir() / "processed/op_common_cause/pair_inventory/navtrain_frames.parquet").set_index("token").loc[z["id"]].reset_index()
    rng = np.random.default_rng(0)
    sel = pick(z, inv, rng)
    by_log = {}
    for kind, i in sel:
        by_log.setdefault(z["cluster"][i], []).append((kind, i))
    res = []
    for log, jobs in sorted(by_log.items()):
        res += work(log, jobs, z, inv)
    # tier balance of N1 (30 A / 30 B when available), then the plan sizes
    keep, cnt = [], {}
    for r in sorted(res, key=lambda r: r["i"]):
        key = (r["kind"], r["tier"])
        lim = 30 if r["kind"] == "N1_exit" else PLAN[r["kind"]]
        if cnt.get(key, 0) < lim:
            cnt[key] = cnt.get(key, 0) + 1
            keep.append(r)
    res = keep
    RES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    flag = lambda r: r["kind"] != "N1_exit" or r["tier"] == "B"  # noqa: E731  needs a human / visual check
    tab = pd.DataFrame([dict(id=z["id"][r["i"]], log=z["cluster"][r["i"]], kind=r["kind"], tier=r["tier"], missing=r["neg"]["missing"], s_turn=round(r["neg"]["s_turn"], 1),
                             angle=round(r["neg"]["angle"], 0), radius=round(abs(r["neg"]["radius"]), 1), v0=round(float(z["v0"][r["i"]]), 1),
                             plen_pos=round(r["pos"]["plen"], 0), plen_neg=round(r["neg"]["plen"], 0), needs_visual=flag(r), target="follow_logged_path") for r in res])
    tab.to_csv(RES / "negatives_sample.csv", index=False)
    np.savez(RES / "negatives_sample.npz", id=tab.id.to_numpy().astype(str), kind=tab.kind.to_numpy().astype(str), tier=tab.tier.to_numpy().astype(str),
             poly=np.stack([r["neg"]["poly"] for r in res]), pmask=np.stack([r["neg"]["pmask"] for r in res]),
             poly_pos=np.stack([r["pos"]["poly"] for r in res]), pmask_pos=np.stack([r["pos"]["pmask"] for r in res]), needs_visual=tab.needs_visual.to_numpy())
    print(tab.groupby(["kind", "tier"]).size())
    # sheet: 4 kinds x 5 examples (+ N1 tier A / B on the first rows)
    rows = [("N1_exit", "A"), ("N1_exit", "B"), ("N2_side", ""), ("N3_wrong", ""), ("N4_uturn", "")]
    fig, axs = plt.subplots(len(rows), 5, figsize=(20, 4.2 * len(rows)))
    for ri, (kind, tier) in enumerate(rows):
        rs = [r for r in res if r["kind"] == kind and r["tier"] == tier][:5]
        for ci in range(5):
            ax = axs[ri, ci]
            ax.axis("off")
            if ci >= len(rs):
                continue
            r = rs[ci]
            for pg in r["polys"]:
                ax.fill(-pg[:, 1], pg[:, 0], color="#e6e6e6", lw=0.3, ec="#bbbbbb", zorder=0)
            for h, col, lab in ((r["pos"], "#1b9e77", "logged"), (r["neg"], "#d62728", "negative")):
                p = h["poly"][h["pmask"]]
                ax.plot(-p[:, 1], p[:, 0], "o-", color=col, ms=3, lw=2, label=lab, zorder=3)
            ax.plot(0, 0, "k^", ms=9, zorder=5)
            ax.set_xlim(-70, 70)
            ax.set_ylim(-20, 120)
            ax.set_aspect("equal")
            ax.set_title(f"{kind}{(' tier ' + tier) if tier else ''}  {r['neg']['missing']}  {z['id'][r['i']][:6]}  v={z['v0'][r['i']]:.1f}", fontsize=9)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, fontsize=11)
    fig.suptitle("Negative routes (red) vs the logged route (green); grey = nuPlan lanes / connectors (not a model input); x = right, y = forward", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "negatives_sample.png", dpi=70)


if __name__ == "__main__":
    main()
