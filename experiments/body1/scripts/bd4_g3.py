"""BODY1 arm 4.3, offline gate G3 (a) / (b) (prereg Amendment 4 item 5 and note vii): contact rates of the student's OWN plan, new checkpoint
against a reference checkpoint on the same states, truth by lib/sweep.py. Open loop, no simulator.

  hold     states of navsim/body1-hold-logs in the families log / ot1 / yr1 / bd4 (the logs no new term was trained on)
  val      the same on navsim/body1-val-logs (Amendment 5 item 3: the part of the train logs kept out of the hinge-only rows; the weight of
           the hinge-only rows is selected here and hold logs are not opened by this set)
  navtest  on-log navtest tokens (lb_navtest, on the checkpoints' own front protocol; agent labels navtest-k32, NAVSIM raster navtest; the scorer-layer raster if it was built)
Rates per state: agent = counted contact of the 4 s sweep (rear-end contacts by a faster object excluded, as the row labels);
boundary = minimum footprint margin < -0.20 m with no contact at t = 0, on the NAVSIM raster (carries the line) and on item C's raster (`road`,
reported; a road-and-lane raster without car parks, not the scorer's road area: Amendment 5 note of 2026-10-10). Difference new - ref with jevdrive.stats.paired(groups=log); relative fall = (ref - new) / ref. Tables per state family, user class
(taxonomy classes with Amendment 1 item 1, contact_head.classes), the > 45 deg bucket (|logged 4 s heading change|) and speed bin -> results/loss/g3_<name>.{csv,json}.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/bd4_g3.py --name pilot --new P2H10B-P-s0 --ref P2H10-P-s0 P2H10-F-s0 --shards 2 3
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

B.FAMS["bd4"] = "bd4_"

def plans(dirs, rows_of, tags, dev, bench=False):
    """Own plans (len(tags), n, 8, 3) of the selected rows of each cache dir, in the state's own frame. Every dir is read on the front
    protocol the checkpoints were trained on (jevdrive.bench's rule, `parity_frames`). Until 2026-10-10 the board dirs (lb_*) were read on
    GIMM frames whatever the checkpoint: for warp-trained checkpoints the navtest plans of those dumps are off-protocol (ADE to the log
    about 1.5 m against 0.57 m on warp; results/navtest_warp.md). bench=True: a tag whose plan file jevdrive.bench has archived for
    lb_navtest takes those plans instead of a forward."""
    import torch
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive.bench.models import parity_frames
    fr = {parity_frames(t) for t in tags}
    assert len(fr) == 1, f"checkpoints of different front protocols in one call: {fr}"
    fr = fr.pop()
    W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
    out = []
    for d, rr in zip(dirs, rows_of):
        o, todo = np.zeros((len(tags), len(rr), 8, 3), np.float32), list(range(len(tags)))
        if bench and d == "lb_navtest":
            import turn_oracle as TO
            names = np.load(B.cache_root() / d / "tab.npz")["names"][rr]
            for m, t in enumerate(tags):
                f = _pl.Path(TO.pf(f"{t}@{fr}"))
                if f.exists():
                    z = np.load(f)
                    pos = {x: i for i, x in enumerate(z["tokens"].tolist())}
                    o[m] = z["poses"][[pos[x] for x in names]]
                    todo.remove(m)
            print(f"{d}: bench plans for {len(tags) - len(todo)} tags, forward ({fr}) for {[tags[m] for m in todo]}", flush=True)
        if todo:
            models = {m: T.load_pmodel(tags[m], dev) for m in todo}
            S = T.Store([d], dev, need_side=False, frames=fr, host=True)
            pi = torch.as_tensor(S.pi, device=dev)
            with torch.no_grad():
                for i in range(0, len(rr), 256):
                    r = torch.as_tensor(rr[i:i + 256], device=dev)
                    f = S.front[r]
                    for m, model in models.items():
                        p = model(f, S.ego[r], S.tc[r], None, None).float()[:, pi].view(-1, 33, 15)
                        o[m, i:i + len(r)] = torch.stack(T.rear(p, S.cam_x[r], W), -1).cpu().numpy()
        out.append(o)
    return np.concatenate(out, 1)


def main(a):
    import pandas as pd
    import torch
    import sweep as SW
    from jevdrive import stats
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags = [a.new] + a.ref
    with Run("body1", f"g3-{a.name}", config=vars(a)) as run:
        if a.set in ("hold", "val"):
            hold = splits.load(B.HOLD if a.set == "hold" else B.VAL)
            run.use_split(hold)
            L = B.labels()
            road = np.load(data_dir() / "runs/body1/labels_road/navtrain_s01234567891011.npz", mmap_mode=None)["sdf"]
            tax = np.load(B.root() / "taxonomy" / "tax.npz")
            dirs, rows_of, meta = [], [], []
            for f in a.fams:
                for k in a.shards:
                    t = B.tab(f, k)
                    rr = np.flatnonzero(hold.mask(t["log"]))
                    g, off = B.state_index(f, k, t)
                    dirs.append(B.cdir(f, k)), rows_of.append(rr)
                    meta.append(pd.DataFrame(dict(fam=f, grow=g[rr], dy=off[rr, 0], dpsi=off[rr, 1], log=t["log"][rr], v0=t["speed"][rr])))
            M = pd.concat(meta, ignore_index=True)
            g = M.grow.to_numpy()
            box, val, cls, sdf, sdf_r = L["box"][g], L["valid"][g], L["cls"][g], L["sdf"][g], road[g]
            from contact_head import classes                                        # user class with Amendment 1 item 1 (the tax file's `pc` predates it)
            M["cls"], M["dyaw"] = np.array(B.CLS)[classes(tax)[0][g]], np.abs(tax["dyaw"][g])
        else:
            run.use_split(splits.load("navsim/navtest"))
            t = dict(np.load(B.cache_root() / "lb_navtest" / "tab.npz"))
            ag = np.load(data_dir() / "runs/op_parity/agent_labels/navtest-k32.npz")
            sd = np.load(data_dir() / "runs/op_probe/labels/navtest.npz")
            assert (ag["tokens"] == t["names"]).all() and (sd["tokens"] == t["names"]).all()
            rr = np.flatnonzero(ag["ok"] & sd["ok"])
            dirs, rows_of = ["lb_navtest"], [rr]
            box, val, cls, sdf = ag["box"][rr], ag["valid"][rr], ag["cls"][rr], sd["sdf"][rr]
            fr = data_dir() / "runs/body1/labels_road/navtest.npz"
            sdf_r = np.load(fr)["sdf"][rr] if fr.exists() else None
            fut = np.nan_to_num(t["fut"][rr][:, :, 2].astype(np.float64))
            M = pd.DataFrame(dict(fam="navtest", dy=0.0, dpsi=0.0, log=t["log"][rr], v0=t["speed"][rr], cls="all",
                                  dyaw=np.abs(np.degrees(np.unwrap(fut, axis=1)[:, -1]))))
        P = plans(dirs, rows_of, tags, dev, a.bench_plans)                          # (tags, n, 8, 3)
        off = M[["dy", "dpsi"]].to_numpy(np.float32)
        R = {}
        for m, tag in enumerate(tags):
            lab = SW.labels(P[m][:, None], off, box, val, cls, sdf)
            R[tag] = dict(agent=lab["a_hit"][:, 0].astype(float), bnd=((lab["b_margin"][:, 0] < -0.20) & ~lab["b_t0"][:, 0]).astype(float))
            if sdf_r is not None:
                br = SW.boundary_labels(SW.dense(P[m][:, None], off[:, None]), sdf_r)
                R[tag]["road"] = ((br["margin"][:, 0] < -0.20) & ~br["t0"][:, 0]).astype(float)
            R[tag]["sum"] = R[tag]["agent"] + R[tag]["bnd"]
        logs = M.log.to_numpy()
        subs = [("pooled", np.ones(len(M), bool))] + [(f"fam {f}", (M.fam == f).to_numpy()) for f in M.fam.unique()] + \
               [(f"class {c}", (M.cls == c).to_numpy()) for c in sorted(M.cls.unique()) if c != "all"] + [("> 45 deg", (M.dyaw > 45).to_numpy()),
               ("v < 1", (M.v0 < 1).to_numpy()), ("v 1-3", ((M.v0 >= 1) & (M.v0 <= 3)).to_numpy()), ("v > 3", (M.v0 > 3).to_numpy())]
        rows = []
        for ref in a.ref:
            for name, m in subs:
                if not m.any():
                    continue
                for q in R[a.new]:
                    r = stats.paired(R[a.new][q][m], R[ref][q][m], groups=logs[m])
                    rows.append(dict(ref=ref, subset=name, rate=q, n=int(m.sum()), logs=int(len(np.unique(logs[m]))), new=r["mean_a"], base=r["mean_b"], diff=r["mean"],
                                     lo=r["lo"], hi=r["hi"], rel_fall=(r["mean_b"] - r["mean_a"]) / r["mean_b"] if r["mean_b"] > 0 else np.nan))
        D = pd.DataFrame(rows)
        OUT = B.REPO / a.out
        OUT.mkdir(parents=True, exist_ok=True)
        D.to_csv(OUT / f"g3_{a.name}.csv", index=False, float_format="%.5f")
        pooled = {ref: {q: D[(D.ref == ref) & (D.subset == "pooled") & (D.rate == q)].iloc[0].to_dict() for q in R[a.new]} for ref in a.ref}
        verdict = {ref: dict(agent_lower=bool(p["agent"]["new"] < p["agent"]["base"]), bnd_lower=bool(p["bnd"]["new"] < p["bnd"]["base"]),
                             agent_30=bool(p["agent"]["rel_fall"] >= 0.30 and p["agent"]["hi"] < 0), bnd_25=bool(p["bnd"]["rel_fall"] >= 0.25 and p["bnd"]["hi"] < 0), bnd_30=bool(p["bnd"]["rel_fall"] >= 0.30 and p["bnd"]["hi"] < 0),
                             no_rise_sum_falls=bool(p["agent"]["diff"] <= 0 and p["bnd"]["diff"] <= 0 and p["sum"]["diff"] < 0)) for ref, p in pooled.items()}
        (OUT / f"g3_{a.name}.json").write_text(json.dumps(dict(new=a.new, refs=a.ref, set=a.set, fams=a.fams, shards=a.shards, n=len(M), pooled=pooled, verdict=verdict),
                                                          indent=1, default=float) + "\n")
        run.info("\n" + D[D.subset.isin(["pooled"]) | D.subset.str.startswith("fam")].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        run.info(json.dumps(verdict))
        run.summary.update(n=len(M), verdict=verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--set", default="hold", choices=["hold", "val", "navtest"])
    ap.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1", "bd4"])
    ap.add_argument("--shards", type=int, nargs="+", default=list(range(B.NSH)))
    ap.add_argument("--out", default="experiments/body1/results/loss", help="table directory, relative to the repo")
    ap.add_argument("--bench-plans", action="store_true", help="navtest: plans from jevdrive.bench's archived plan files where they exist")
    main(ap.parse_args())
