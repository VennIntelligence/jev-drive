"""Negative route polylines for training: generate at scale over navtrain, screen every row against the nuPlan map (lib/route_neg.py), write the sidecar
and a spot-check sheet.

  envs/navsim2 on the box (nuPlan map API; CPU only):
  export NUPLAN_MAPS_ROOT=$DATA_DIR/datasets/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 OPENSCENE_DATA_ROOT=$DATA_DIR/datasets/navsim
  python experiments/op_route_cmd/scripts/route_neg_sample.py [--max-logs N]        # build: sidecar + counts
  python experiments/op_route_cmd/scripts/route_neg_sample.py --sheet                # sheets from the sidecar (survivors / rejects / borderline)

Candidates (navtrain frames of route.npz, split navsim/navtrain, one draw per frame and kind, rng seeded by (row, kind) so every row is reproducible):
  N1 exit   status branch (decision-93 inventory), the ego lane lacks an exit class X in {left, right, straight}; tier A = no lane of the ego roadblock has X,
            tier B = another lane has it -> emitted as `lane_change_needed`, never used
  N2 side / N4 uturn   status no_junction_30m or branch (not in a junction), v >= 3 m/s, path >= 90 / 60 m (the map screen, not a junction-free path, decides)
  N3 wrong  same status, v >= 3 m/s, path >= 100 m (oncoming side by `map_location`: Singapore drives on the left)
Screens (lib/route_neg.screen): see results/negatives.md. N1 tier-A "straight" rows are capped at STRAIGHT_SHARE of the N1 rows used.
Output (DATA_DIR/processed/op_route_cmd/navtrain/):
  route_neg.npz      rows with use=True only, keyed by `id` (token) + `kind` (a token can carry several kinds): poly / pmask (negative, K=16 vertices at 10 m, ego frame),
                     poly_pos / pmask_pos (the logged route of the frame), tier, missing, flags (trivial_straight, lane_change_ok=False), screen numbers, margin
  route_neg_aux.npz  every other row with its label (lane_change_needed, rej_*, cap_straight), same fields
Target of a negative in training: the logged path (hindsight) or the original model's plan when distilling.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(Path(__file__).parent), str(REPO / "experiments" / "op_common_cause" / "scripts")]
import pair_inv_navtrain as PI  # noqa: E402  (nuPlan map helpers of the decision-93 inventory)
import route_neg as RN  # noqa: E402
import route_nav as NV  # noqa: E402
import route_poly as RP  # noqa: E402
from jevdrive import par  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run  # noqa: E402

RES, FIGS = REPO / "experiments/op_route_cmd/results", REPO / "experiments/op_route_cmd/figs"
STRAIGHT_SHARE = 0.15        # cap of tier-A "missing straight" rows (ramps / loops: trivially easy) in the N1 rows used
NUMS = ("clear_m", "same", "opp", "cross", "area", "off")
SEED_CAP = 7
TRIES = 4                    # draws per (frame, kind) for N2 / N3 / N4; the first that passes the screen is kept, else the last is stored as a reject


def out_dir():
    return data_dir() / "processed/op_route_cmd/navtrain"


def candidates(z, inv):
    st, v = inv.status.to_numpy(), z["v0"]
    ok = np.isin(st, ("no_junction_30m", "branch"))        # in_junction / off_map / heading_mismatch frames: no reliable lane context
    return {"N1_exit": (st == "branch") & (z["plen"] >= inv.dist.fillna(0).to_numpy() + 30) & (v >= 1.0),
            "N2_side": ok & (z["plen"] >= 90) & (v >= 3.0),      # a junction on the path is fine: the exit-leg screen rejects a turn onto a real road
            "N3_wrong": ok & (z["plen"] >= 100) & (v >= 3.0),
            "N4_uturn": ok & (z["plen"] >= 60) & (v >= 3.0)}


_ras = {}


def raster(loc):
    if loc not in _ras:
        _ras[loc] = RN.DrivableRaster(out_dir() / "drivable_raster" / f"{loc}.npy")
    return _ras[loc]


def build_rasters():
    d = out_dir() / "drivable_raster"
    d.mkdir(exist_ok=True)
    for loc in ("sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood"):
        if not (d / f"{loc}.npy").exists():
            RN.DrivableRaster.build(PI.get_map(loc), d / f"{loc}.npy")


def load_log(log):
    fr = pickle.load(open(os.path.join(os.environ["OPENSCENE_DATA_ROOT"], "navsim_logs", "trainval", log + ".pkl"), "rb"))
    xs = np.array([(f["ego2global_translation"][0], f["ego2global_translation"][1], NV._quat_yaw(f["ego2global_rotation"])) for f in fr])
    return fr[0]["map_location"], xs, np.r_[0.0, np.cumsum(np.hypot(*np.diff(xs[:, :2], axis=0).T))]


def build_one(m, xs, cum, lht, i, fi, kind, dist=None, classes="", attempt=0):
    """Negative + logged route + tier of one (row i, kind) from the log poses; None when it cannot be built. Deterministic."""
    loc = NV.ego_path(xs, cum, fi)
    pos = RP.hindsight(loc, want_path=True)
    P = pos.pop("path")
    if P is None:
        return None
    rng = np.random.default_rng([i, RN.KINDS.index(kind), attempt])
    kw, tier = {}, ""
    if kind == "N1_exit":
        from nuplan.common.actor_state.state_representation import Point2D
        _, lane = PI.find_lane(m, Point2D(float(xs[fi, 0]), float(xs[fi, 1])), xs[fi, 2])
        miss = [c for c in ("left", "right", "straight") if c not in set(classes.split(","))]
        if lane is None or not miss:
            return None
        kw["missing"], kw["s_branch"] = str(rng.choice(miss)), float(dist)
        tier = "A" if kw["missing"] not in PI.rb_classes(m, lane) else "B"
    neg = RN.make(kind, P, rng, lht=lht, **kw)
    return None if neg is None else (neg, pos, tier)


def work(job):
    log, rows = job                                  # rows: list of (i, fi, kind, dist, classes)
    loc, xs, cum = load_log(log)
    m, lht, out, ras = PI.get_map(loc), loc.startswith("sg-"), [], raster(loc)
    for i, fi, kind, dist, classes in rows:
        try:
            out.append(one(m, xs, cum, lht, ras, i, fi, kind, dist, classes))
        except Exception as e:  # a row that breaks the map query is dropped, with the reason counted
            import traceback
            out.append(dict(i=i, kind=kind, label="rej_error", err=repr(e)[:80], tb=traceback.format_exc()[-600:]))
    return out


def one(m, xs, cum, lht, ras, i, fi, kind, dist, classes):
    for att in range(1 if kind == "N1_exit" else TRIES):           # N2 / N3 / N4: redraw turn position / side / offset until the screen passes
        b = build_one(m, xs, cum, lht, i, fi, kind, dist, classes, att)
        if b is None:
            return dict(i=i, kind=kind, label="rej_not_built")
        neg, pos, tier = b
        use, label, info = RN.screen(m, tuple(xs[fi]), neg, tier, ras)
        if use:
            break
    return dict(i=i, kind=kind, tier=tier, label=label, use=use, attempt=att, missing=neg["missing"], s_turn=neg["s_turn"], angle=neg["angle"], radius=abs(neg["radius"]),
                poly=neg["poly"], pmask=neg["pmask"], poly_pos=pos["poly"], pmask_pos=pos["pmask"], plen_neg=neg["plen"], **info)


def margin(r):
    """Normalised distance of a used row to its decision threshold (small = borderline)."""
    if r["kind"] == "N1_exit":
        return np.nan
    if r["kind"] in ("N2_side", "N4_uturn"):
        return (min(r["clear_m"], 30.0) - RN.CLEAR_M) / RN.CLEAR_M
    return min((RN.N3_SAME_MAX - r["same"]) / RN.N3_SAME_MAX, (r["opp"] + r["off"] - RN.N3_BAD_MIN) / RN.N3_BAD_MIN)


def build(a):
    with Run("op_route_cmd", "route_neg", config=vars(a)) as run:
        tr = splits.load("navsim/navtrain")
        run.use_split(tr)
        z = {k: v for k, v in np.load(out_dir() / "route.npz").items()}
        inv = pd.read_parquet(data_dir() / "processed/op_common_cause/pair_inventory/navtrain_frames.parquet").set_index("token").loc[z["id"]].reset_index()
        keep = tr.mask(z["id"])
        cand = {k: c & keep for k, c in candidates(z, inv).items()}
        by_log = {}
        for kind, c in cand.items():
            for i in np.flatnonzero(c):
                by_log.setdefault(z["cluster"][i], []).append((int(i), int(inv.frame_idx.iloc[i]), kind, float(inv.dist.fillna(0).iloc[i]), inv.classes.iloc[i]))
        build_rasters()
        jobs = sorted(by_log.items())[: a.max_logs]
        run.info("candidates: %s; %d logs", {k: int(c.sum()) for k, c in cand.items()}, len(jobs))
        res = par.pmap(work, jobs, run=run, desc="logs")
        res.raise_if_failed()
        rows = [r for part in res.values for r in part]
        # cap the tier-A straight rows of N1
        ok_n1 = [r for r in rows if r["kind"] == "N1_exit" and r["label"] == "ok"]
        st = [r for r in ok_n1 if r["missing"] == "straight"]
        lr = len(ok_n1) - len(st)
        keep_s = set(np.random.default_rng(SEED_CAP).permutation(len(st))[: int(lr * STRAIGHT_SHARE / (1 - STRAIGHT_SHARE))].tolist())
        for j, r in enumerate(st):
            if j not in keep_s:
                r["use"], r["label"] = False, "cap_straight"
        df = pd.DataFrame(rows)
        if "err" in df and df.err.notna().any():
            print(df.err.value_counts().head(), df.tb.dropna().iloc[0])
        df["id"], df["log"], df["v0"] = z["id"][df.i], z["cluster"][df.i], z["v0"][df.i]
        df["tier"] = df["tier"].fillna("")
        df["trivial_straight"] = (df.kind == "N1_exit") & (df.missing == "straight")
        df["margin"] = [margin(r) if r.get("use") == True else np.nan for r in df.to_dict("records")]  # noqa: E712
        write(df, out_dir())
        rep = df.assign(missing=df.missing.fillna("-")).groupby(["kind", "tier", "label"]).size().unstack(fill_value=0)
        print(rep.to_string())
        rep.to_csv(RES / "negatives_counts.csv")
        run.summary.update(rows=len(df), used=int(df.use.eq(True).sum()))


def write(df, d):
    def pack(x):
        n = len(x)
        z = dict(id=x.id.to_numpy().astype(str), kind=x.kind.to_numpy().astype(str), tier=x.tier.to_numpy().astype(str), label=x.label.to_numpy().astype(str),
                 log=x.log.to_numpy().astype(str), row=x.i.to_numpy().astype(np.int64), missing=x.missing.fillna("").to_numpy().astype(str),
                 poly=np.stack(x.poly.to_list()) if n else np.zeros((0, 16, 2), np.float32), pmask=np.stack(x.pmask.to_list()) if n else np.zeros((0, 16), bool),
                 poly_pos=np.stack(x.poly_pos.to_list()) if n else np.zeros((0, 16, 2), np.float32), pmask_pos=np.stack(x.pmask_pos.to_list()) if n else np.zeros((0, 16), bool),
                 trivial_straight=x.trivial_straight.to_numpy().astype(bool), margin=x.margin.to_numpy().astype(np.float32), v0=x.v0.to_numpy().astype(np.float32))
        for c in ("s_turn", "angle", "radius", "attempt") + NUMS:
            z[c] = x[c].to_numpy().astype(np.float32) if c in x else np.full(n, np.nan, np.float32)
        return z
    ok = df[df.use.eq(True)]
    np.savez(d / "route_neg.npz", **pack(ok))
    np.savez(d / "route_neg_aux.npz", **pack(df[df.poly.notna() & ~df.use.eq(True)]))


# ------------------------------------------------------------------------------------------------ sheets
def underlay(m, pose, rad=75.0):
    from shapely.geometry import box
    objs = RN._near(m, pose[:2], rad, ("LANE", "LANE_CONNECTOR", "INTERSECTION", "CARPARK_AREA"))
    c, s = np.cos(pose[2]), np.sin(pose[2])
    out = []
    for q, v in objs.items():
        for o in v:
            geoms = o.polygon.geoms if hasattr(o.polygon, "geoms") else [o.polygon]
            for g in geoms:
                x, y = (np.array(t) for t in g.exterior.xy)
                out.append((q.name, np.stack([c * (x - pose[0]) + s * (y - pose[1]), -s * (x - pose[0]) + c * (y - pose[1])], -1)))
    return out


COL = dict(DRIVABLE_AREA="#dbe9f6", INTERSECTION="#f2ecc8", CARPARK_AREA="#f6d9b8", PUDO="#f6d9b8", LANE="#d0d0d0", LANE_CONNECTOR="#c4c4c4")


def panel(ax, m, m_loc, xs, cum, lht, r, title):
    b = build_one(m, xs, cum, lht, int(r["row"]), int(r["fi"]), r["kind"], r["dist"], r["classes"], int(r["attempt"]))
    if b is None:
        ax.axis("off")
        return
    neg, pos, _ = b
    pose = tuple(xs[int(r["fi"])])
    ras = raster(m_loc)
    gx, gy = np.meshgrid(np.arange(-70, 70.1, 0.5), np.arange(-20, 120.1, 0.5))
    gp = RN.to_global(np.stack([gy.ravel(), -gx.ravel()], -1), pose)         # plot x = right = -y_ego
    ax.imshow(np.where(ras.inside(gp).reshape(gx.shape), 1.0, np.nan), extent=(-70, 70, -20, 120), origin="lower", cmap="Blues", vmin=0, vmax=2.2, zorder=0, interpolation="nearest")
    for q in ("INTERSECTION", "CARPARK_AREA", "LANE", "LANE_CONNECTOR"):
        for n, pg in underlay(m, pose):
            if n == q:
                ax.fill(-pg[:, 1], pg[:, 0], color=COL[q], lw=0.3, ec="#9a9a9a" if q.startswith("LANE") else "none", zorder={"DRIVABLE_AREA": 0, "INTERSECTION": 1, "CARPARK_AREA": 1, "PUDO": 1}.get(q, 2))
    d = neg["dense"]
    s = RN._arc(d)
    k = (s >= neg["s_free"] + RN.LEG_FROM) & (s <= neg["s_free"] + RN.LEG_FROM + RN.LEG_LEN)
    pp = pos["poly"][pos["pmask"]]
    ax.plot(-pp[:, 1], pp[:, 0], "o-", color="#1b9e77", ms=3, lw=2, zorder=4)
    ax.plot(-d[:, 1], d[:, 0], "-", color="#d62728", lw=2, zorder=5)
    ax.plot(-d[k, 1], d[k, 0], "-", color="#7a0000", lw=4, zorder=6, alpha=0.8)
    ax.plot(0, 0, "k^", ms=9, zorder=7)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-20, 120)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.5)


def sheet(path, rows, titles, ncol, z, inv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr = -(-len(rows) // ncol)
    fig, axs = plt.subplots(nr, ncol, figsize=(3.6 * ncol, 3.9 * nr), squeeze=False)
    cache = {}
    for ax, r, t in zip(axs.ravel(), rows, titles):
        if r["log"] not in cache:
            cache = {r["log"]: load_log(r["log"])}
        loc, xs, cum = cache[r["log"]]
        panel(ax, PI.get_map(loc), loc, xs, cum, loc.startswith("sg-"), r, t)
    for ax in axs.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("red = negative (dark red = screened part), green = logged route; grey = lanes / connectors, blue = drivable raster, yellow = intersection, orange = carpark; "
                 "x = right, y = forward", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=60)
    plt.close(fig)


def sheets(a):
    d = out_dir()
    z = {k: v for k, v in np.load(d / "route.npz").items()}
    inv = pd.read_parquet(data_dir() / "processed/op_common_cause/pair_inventory/navtrain_frames.parquet").set_index("token").loc[z["id"]].reset_index()
    tok2row = {t: j for j, t in enumerate(z["id"])}
    U, X = np.load(d / "route_neg.npz"), np.load(d / "route_neg_aux.npz")
    rng = np.random.default_rng(5)

    def table(Z):
        t = pd.DataFrame({k: Z[k] for k in Z.files if Z[k].ndim == 1})
        t["fi"] = inv.frame_idx.to_numpy()[t.row]
        t["dist"], t["classes"] = inv.dist.fillna(0).to_numpy()[t.row], inv.classes.to_numpy()[t.row]
        return t
    u, x = table(U), table(X)

    def desc(r):
        nums = " ".join(f"{k}={r[k]:.2f}" for k in RN_NUMS if np.isfinite(r[k]))
        return f"{r['kind']}{'-' + r['tier'] if r['tier'] else ''} {r['missing']}  {r['label']}\n{nums}  v={r['v0']:.1f}  {r['id'][:6]}"
    RN_NUMS = NUMS
    # survivors: 25 spread over the kinds (N1 only left / right and straight at its capped share, as in the set)
    quota = {"N1_exit": 5, "N2_side": 7, "N3_wrong": 6, "N4_uturn": 7}
    s_rows = pd.concat([u[u.kind == k].sample(min(n, (u.kind == k).sum()), random_state=1) for k, n in quota.items()])
    rej = x[x.label.str.startswith("rej_") | (x.label == "cap_straight")]
    rj = pd.concat([g.sample(min(len(g), 3), random_state=2) for _, g in rej.groupby("label")])
    rj = rj.sample(min(10, len(rj)), random_state=3)
    sheet(FIGS / "negatives_screened_sheet.png", list(s_rows.to_dict("records")) + list(rj.to_dict("records")),
          ["USE " + desc(r) for r in s_rows.to_dict("records")] + ["REJECT " + desc(r) for r in rj.to_dict("records")], 5, z, inv)
    # borderline: the used rows closest to their threshold, per kind
    bq = {"N2_side": 4, "N4_uturn": 3, "N3_wrong": 3}
    b = pd.concat([g.nsmallest(max(len(g) // 10, n * 3), "margin").sample(n, random_state=4) for k, n in bq.items() for g in [u[(u.kind == k) & u.margin.notna()]]])
    a1 = u[(u.kind == "N1_exit") & (u.clear_m < 20)]
    b = pd.concat([b, a1.sample(min(2, len(a1)), random_state=4)])
    sheet(FIGS / "negatives_borderline_sheet.png", list(b.to_dict("records")), ["BORDERLINE " + desc(r) for r in b.to_dict("records")], 4, z, inv)
    b.drop(columns=["poly", "pmask"], errors="ignore").to_csv(RES / "negatives_borderline.csv", index=False)
    print("sheets written;", len(s_rows), "survivors,", len(rj), "rejects,", len(b), "borderline")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-logs", type=int, default=None)
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    sheets(a) if a.sheet else build(a)
