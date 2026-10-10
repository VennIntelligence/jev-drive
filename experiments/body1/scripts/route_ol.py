"""BODY1 route tube, open loop (prereg Amendment 7): how far does the student's OWN plan go from the logged path of the same state, by turn
bucket, turn side and state family? Reads the plan dumps of prog_ol.py (`states` writes ol_<name>.parquet + ol_<name>_plans.npy for any set
of checkpoints), adds the logged future of every state from the cache tabs, no GPU and no simulator. Only navtrain hold / validation logs and
navtest tokens are read; no AlpaSim scene.

Per state and checkpoint (lib/route.py; distance of the 8 plan poses, 0.5 .. 4 s, to the logged path extended by a ray at each end):
  dmax   largest distance within 4 s (m)            leave_R  dmax > R for R in 2 / 3 / 4 m (4 m = the board's `max_dist_to_gt_trajectory`)
  out4   signed lateral at 4 s, + = outside of the logged turn (left of the path on a right turn)        exc_B  mean over poses of relu(d - B)
Tables: results/route/ol_<name>.csv (every subset x checkpoint) and ol_<name>_pairs.csv (new - ref on the same states, jevdrive.stats.paired
by log; pairs as prog_ol.py's --new / --ref). Subsets: state family x turn bucket (|logged 4 s heading change|; right / left by its sign).

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/route_ol.py --name a6_full_hold --set hold --new P2H10S-F-s0 P2H10S-F-s1 --ref P2H10-F-s0 P2H10-F-s1
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import route as RT  # noqa: E402

B.FAMS["bd4"] = "bd4_"
RS, BANDS = (2.0, 3.0, 4.0), (1.5, 2.0, 2.5, 3.0)


def logged(M, a, split):
    """Logged future (n, 8, 3) of the dump's states, in dump order (prog_ol.py's selection)."""
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    if a.set == "navtest":
        t = dict(np.load(B.cache_root() / "lb_navtest" / "tab.npz"))
        ag = np.load(data_dir() / "runs/op_parity/agent_labels/navtest-k32.npz")
        sd = np.load(data_dir() / "runs/op_probe/labels/navtest.npz")
        rr = np.flatnonzero(ag["ok"] & sd["ok"])
        names, F = t["names"][rr], t["fut"][rr]
    else:
        hold = splits.load(split)
        names, F = [], []
        for f in a.fams:
            for k in a.shards:
                t = B.tab(f, k)
                rr = np.flatnonzero(hold.mask(t["log"]))
                names.append(t["names"][rr]), F.append(t["fut"][rr])
        names, F = np.concatenate(names), np.concatenate(F)
    assert len(names) == len(M) and (names == M.name.to_numpy()).all(), "dump and tabs disagree"
    return F


def main(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("body1", f"route-ol-{a.name}", config=vars(a)) as run:
        split = {"hold": B.HOLD, "val": B.VAL, "navtest": "navsim/navtest"}[a.set]
        run.use_split(splits.load(split))
        M = pd.read_parquet(B.root() / "prog" / f"ol_{a.name}.parquet")
        tags = [c.split("|")[1] for c in M.columns if c.startswith("arc4|")]
        P = np.load(B.root() / "prog" / f"ol_{a.name}_plans.npy")
        assert P.shape[:2] == (len(tags), len(M))
        F = logged(M, a, split)
        off = M[["dy", "dpsi"]].to_numpy(np.float64)
        ok = ~np.isnan(F[:, :, 0]).any(1)
        turn = RT.turn_deg(F, off)
        side = -np.sign(turn)                                                        # + lateral (left) is outside on a right turn
        Q = {}
        for m, tag in enumerate(tags):
            d, lat = RT.path_dist_np(P[m], F, off)
            so = side[:, None] * lat                                                 # signed lateral of every pose, + = outside of the logged turn
            Q[tag] = dict(dmax=d.max(1), out4=so[:, -1], **{f"leave_{r:g}": (d.max(1) > r).astype(float) for r in RS},
                          **{f"wide_{r:g}": (so.max(1) > r).astype(float) for r in (1.0,) + RS}, **{f"cut_{r:g}": (so.min(1) < -r).astype(float) for r in (1.0,) + RS},
                          **{f"exc_{b:g}": np.maximum(d - b, 0).mean(1) for b in BANDS}, **{f"pos_{b:g}": (d.max(1) > b).astype(float) for b in BANDS})
        at = np.abs(turn)
        buckets = [("all", np.ones(len(M), bool)), ("< 20 deg", at < 20), ("20-45 deg", (at >= 20) & (at <= 45)), ("> 45 deg", at > 45),
                   ("> 45 deg right", turn < -45), ("> 45 deg left", turn > 45), ("48-70 deg right", (turn <= -48) & (turn >= -70))]
        fams = [("pooled", np.ones(len(M), bool))] + [(f, (M.fam == f).to_numpy()) for f in M.fam.unique() if M.fam.nunique() > 1] + \
               [("off-track", (M.fam != "log").to_numpy())] * (M.fam.nunique() > 1)
        logs = M.log.to_numpy()
        new, ref = list(a.new), list(a.ref) * (len(a.new) if len(a.ref) == 1 else 1)
        rows, prs = [], []
        for fn, fm in fams:
            for bn, bm in buckets:
                m = fm & bm & ok
                if m.sum() < 20:
                    continue
                for tag in tags:
                    q = Q[tag]
                    rows.append(dict(fam=fn, bucket=bn, tag=tag, n=int(m.sum()), logs=int(len(np.unique(logs[m]))), dmax_mean=q["dmax"][m].mean(),
                                     dmax_p90=np.quantile(q["dmax"][m], 0.9), dmax_p99=np.quantile(q["dmax"][m], 0.99), out4_mean=q["out4"][m].mean(),
                                     **{k: q[k][m].mean() for k in q if k[:3] in ("lea", "exc", "pos", "wid", "cut")}))
                for x, y in zip(new, ref):
                    for k in ("dmax", "out4", "leave_2", "leave_3", "leave_4", "wide_1", "wide_2", "wide_3", "wide_4", "cut_2", "exc_2", "exc_3"):
                        r = stats.paired(Q[x][k][m], Q[y][k][m], groups=logs[m])
                        prs.append(dict(fam=fn, bucket=bn, new=x, ref=y, stat=k, n=int(m.sum()), new_v=r["mean_a"], ref_v=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"]))
        out = B.REPO / a.out
        out.mkdir(parents=True, exist_ok=True)
        D, R = pd.DataFrame(rows), pd.DataFrame(prs)
        D.to_csv(out / f"ol_{a.name}.csv", index=False, float_format="%.5f")
        R.to_csv(out / f"ol_{a.name}_pairs.csv", index=False, float_format="%.5f")
        # the gate's number (Amendment 7 item 3): leave rate on the turn rows over 45 deg, each tag, and new - ref
        gate = {t: {bn: {k: float(D[(D.fam == "pooled") & (D.bucket == bn) & (D.tag == t)][k].iloc[0]) for k in ("n", "leave_2", "leave_3", "leave_4", "wide_1", "wide_2", "wide_3", "wide_4", "dmax_mean", "out4_mean")}
                    for bn in ("> 45 deg", "all")} for t in tags}
        (out / f"ol_{a.name}.json").write_text(json.dumps(dict(name=a.name, set=a.set, tags=tags, new=new, ref=ref, n=len(M), gate=gate), indent=1) + "\n")
        pd.set_option("display.width", 250)
        show = D[D.bucket.isin(["all", "> 45 deg", "> 45 deg right", "> 45 deg left"])]
        run.info("\n" + show[["fam", "bucket", "tag", "n", "dmax_mean", "out4_mean", "leave_2", "leave_3", "leave_4", "wide_1", "wide_2", "wide_3", "wide_4", "cut_2", "cut_3", "exc_2", "exc_3"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        run.info("\n" + R[(R.bucket == "> 45 deg") & R.stat.isin(["out4", "leave_4", "wide_1", "wide_2", "wide_3", "wide_4"])].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        run.summary.update(n=len(M), tags=tags)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="prog_ol.py dump name (ol_<name>.parquet)")
    ap.add_argument("--set", default="hold", choices=["hold", "val", "navtest"])
    ap.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1", "bd4"])
    ap.add_argument("--shards", type=int, nargs="+", default=list(range(B.NSH)))
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/body1/results/route")
    main(ap.parse_args())
