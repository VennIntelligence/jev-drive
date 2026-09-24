"""WOD-E2E zero-shot exam, project-venv side (todos/2026-09-24-zeroshot-exam/wod-e2e.md).

  sets                   freeze the pre-registered frame sets -> $DATA_DIR/processed/wod_zeroshot/sets.{npz,json}
  fetch                  8-camera records of frames f-3..f from the raw GCS val shards (needs gcloud + Clash)
  packages               Alpamayo input packages (JPEGs + calibration, no protobuf needed to read) + op_calib.json
  views --set check      render the adapter views (Alpamayo's four cameras, openpilot's two model frames) as PNGs
  score                  RFS / ADE with bootstrap CIs from the model runners' prediction files

The model runners live in scripts/wod_zeroshot_alpamayo.py and scripts/wod_zeroshot_openpilot.py (own venvs).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

SETS = ("rater", "extra", "check")


def targets(sets, which):
    return [str(n) for w in which for n in sets[w]["name"]]


def cmd_sets(a, log):
    log.info(f"sets: {Z.build_sets(a.seed)}")


def cmd_fetch(a, log):
    sets, (spans, ordinal) = Z.load_sets(), Z.load_spans()
    want = {}
    for n in targets(sets, a.set):
        for hn in Z.history_names(n, 3):
            want.setdefault(spans[hn][0], {})[ordinal[hn]] = hn
    log.info(f"{sum(map(len, want.values()))} records from {len(want)} raw shards")
    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy") or "http://127.0.0.1:7890"
    r = Z.fetch_records(want, a.route, proxy, a.workers, log=log.info)
    log.event("fetch", **r)
    log.info(f"fetched {r}")


def cmd_packages(a, log):
    from concurrent.futures import ProcessPoolExecutor
    sets = Z.load_sets()
    names = targets(sets, a.set)
    t0 = time.time()
    with ProcessPoolExecutor(min(16, Z.n_cpus())) as ex:
        list(ex.map(Z.write_package, names, chunksize=8))
    p = Z.write_op_calib(targets(sets, SETS))
    log.info(f"{len(names)} packages in {time.time() - t0:.0f} s; {p}")


def alp_views(pkg, scale=0.3):
    """Four Alpamayo views (numpy path, for looking only) of the newest frame in a package."""
    from PIL import Image
    import io
    z = np.load(pkg)
    cal = Z.read_calib(z, Z.ALP_SRC)
    imgs = [np.asarray(Image.open(io.BytesIO(z[f"jpg_3_{c}"].tobytes())).convert("RGB")) for c in Z.ALP_SRC]
    out, cover = [], []
    for v in G.PAI_ORDER:
        rays = G.ftheta_rays(np, G.PAI_RIG[v], scale)
        src, U, V = G.choose_sources(np, rays, {c: cal[c] for c in Z.ALP_SRC})
        out.append(G.render_np(src, U, V, imgs))
        cover.append(float((src >= 0).mean()))
    return out, cover


def op_views(name, spans, op_calib):
    """openpilot road / wide model frames (RGB, for looking) of one frame from the slim shard."""
    from PIL import Image
    import io
    from jevdrive import waymo as W
    sp = spans[name]
    imgs = []
    with open(W.shard_dir() / sp[0], "rb") as f:
        for k in range(3):
            f.seek(sp[1 + 2 * k])
            imgs.append(np.asarray(Image.open(io.BytesIO(f.read(sp[2 + 2 * k]))).convert("RGB")))
    cal = {int(c): d for c, d in op_calib[name.rsplit("-", 1)[0]].items()}
    out = []
    for k in ("road", "wide"):
        rays = G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H)
        src, U, V = G.choose_sources(np, rays, {c: cal[c] for c in Z.OP_SRC})
        out.append(G.render_np(src, U, V, imgs, nearest=True))
    return out


def cmd_views(a, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets, (spans, _) = Z.load_sets(), Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    covers = []
    for n in targets(sets, a.set)[: a.limit]:
        views, cover = alp_views(Z.root("packages") / f"{n}.npz")
        covers.append(cover)
        road, wide = op_views(n, spans, op_calib)
        fig, ax = plt.subplots(2, 3, figsize=(15, 5.5))
        for i, (img, t) in enumerate(zip(views + [road, wide],
                                         [f"{v} cover {c:.3f}" for v, c in zip(G.PAI_ORDER, cover)] + ["op road", "op wide"])):
            ax.flat[i].imshow(img)
            ax.flat[i].set_title(t, fontsize=8)
            ax.flat[i].axis("off")
        fig.suptitle(n, fontsize=8)
        fig.tight_layout()
        fig.savefig(log.dir / f"views_{n}.png", dpi=110)
        plt.close(fig)
        log.info(f"{n}: cover {np.round(cover, 3)}")
    c = np.array(covers)
    log.event("coverage", views=list(G.PAI_ORDER), mean=c.mean(0).tolist(), min=c.min(0).tolist())
    log.info(f"mean coverage {dict(zip(G.PAI_ORDER, np.round(c.mean(0), 4)))}")


# ---------------------------------------------------------------- scoring

def _alp(name_list, variant):
    d = Z.root("preds", f"alpamayo_{variant}")
    if not all((d / f"{n}.npz").exists() for n in name_list):
        return None, None
    xyz = np.stack([np.load(d / f"{n}.npz")["xyz"] for n in name_list])           # (n, 6, 64, 3)
    cot = [json.loads(str(np.load(d / f"{n}.npz")["cot"])) for n in name_list]
    return Z.resample(Z.ALP_T, xyz[..., :2]), cot                                   # (n, 6, 20, 2)


def _op(name_list, model):
    d = Z.root("preds", f"op_{model}")
    if not all((d / f"{n}.npz").exists() for n in name_list):
        return None
    return np.stack([np.load(d / f"{n}.npz")["wod"] for n in name_list])


def _strat_idx(strata, B, rng):
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    return [g[rng.integers(0, len(g), (B, len(g)))] for g in groups]


def cmd_score(a, log):
    import pandas as pd
    from jevdrive import waymo as W
    S = Z.load_sets()
    r = S["rater"]
    names, past, fut = [str(x) for x in r["name"]], r["past"], r["future"]
    traj, sc, cl = r["traj"], r["scores"], r["cluster"].astype(str)
    speed, log_xy = W.init_speed(past), fut[..., :2]
    best = traj[np.arange(len(traj)), sc.argmax(1)]
    base = W.baselines(past)
    cands = {"rater_best": best[:, None], "rater_worst": traj[np.arange(len(traj)), sc.argmin(1)][:, None],
             "logged_future": log_xy[:, None], "cv": base["cv"][:, None], "zero": base["zero"][:, None]}
    kind = {k: "baseline" for k in cands}
    cots = {}
    for v in ("nav", "nonav"):
        p, cot = _alp(names, v)
        if p is not None:
            cands[f"alpamayo_{v}"], kind[f"alpamayo_{v}"], cots[v] = p, "alpamayo", cot
    for m in ("small", "cinque", "lebowski"):
        p = _op(names, m)
        if p is not None:
            cands[f"op_{m}"], kind[f"op_{m}"] = p[:, None], "openpilot"
    # per frame, per candidate: RFS, inside, ADE
    per = {}
    for k, p in cands.items():
        K = p.shape[1]
        rfs = np.stack([W.rater_feedback_score(p[:, j], traj, sc, speed, details=True)[0] for j in range(K)], 1)
        ins = np.stack([W.rater_feedback_score(p[:, j], traj, sc, speed, details=True)[1][:, 0] for j in range(K)], 1)
        d_best = np.linalg.norm(p - best[:, None], axis=-1)                       # (n, K, 20)
        d_log = np.linalg.norm(p - log_xy[:, None], axis=-1)
        per[k] = dict(rfs=rfs, ins=ins, ade3=d_best[..., :12].mean(-1), ade5=d_best.mean(-1), ade5log=d_log.mean(-1),
                      lon5=(p[..., -1, 0] - log_xy[:, None, -1, 0]), lat5=(p[..., -1, 1] - log_xy[:, None, -1, 1]))
    # rows: how one trajectory per frame is chosen
    rows = {}
    for k, q in per.items():
        if q["rfs"].shape[1] == 1:
            rows[k] = {m: v[:, 0] for m, v in q.items()}
            continue
        p = cands[k]
        med = np.array([Z.medoid(t) for t in p])
        rows[f"{k} | E[1 sample]"] = {m: v.mean(1) for m, v in q.items()}
        rows[f"{k} | sample 0"] = {m: v[:, 0] for m, v in q.items()}
        rows[f"{k} | medoid-of-6"] = {m: v[np.arange(len(v)), med] for m, v in q.items()}
        o = q["rfs"].argmax(1)
        rows[f"{k} | ORACLE best-of-6 RFS"] = {m: v[np.arange(len(v)), o] for m, v in q.items()}
        o = q["ade5"].argmin(1)
        rows[f"{k} | ORACLE minADE_6"] = {m: v[np.arange(len(v)), o] for m, v in q.items()}
    B, rng = a.boot, np.random.default_rng(0)
    sidx = _strat_idx(cl, B, rng)
    fidx = rng.integers(0, len(names), (B, len(names)))

    def cmean_boot(x):  # (B,) cluster-mean replicates
        return np.mean([x[g].mean(1) for g in sidx], 0)

    def cmean(x):
        return float(pd.Series(x).groupby(cl).mean().mean())
    out = []
    for k, q in rows.items():
        rb = cmean_boot(q["rfs"])
        row = {"row": k, "n": len(names), "rfs": cmean(q["rfs"]), "rfs_lo": np.percentile(rb, 2.5),
               "rfs_hi": np.percentile(rb, 97.5), "rfs_frame": q["rfs"].mean(),
               "rfs_frame_lo": np.percentile(q["rfs"][fidx].mean(1), 2.5),
               "rfs_frame_hi": np.percentile(q["rfs"][fidx].mean(1), 97.5),
               "in_trust": q["ins"].mean(), "floored": float((q["rfs"] <= W.RFS_FLOOR + 1e-9).mean()),
               "ade3_rater": q["ade3"].mean(), "ade5_rater": q["ade5"].mean(),
               "ade5_rater_lo": np.percentile(q["ade5"][fidx].mean(1), 2.5),
               "ade5_rater_hi": np.percentile(q["ade5"][fidx].mean(1), 97.5), "ade5_log": q["ade5log"].mean(),
               "lon5_bias": q["lon5"].mean(), "lat5_absmean": np.abs(q["lat5"]).mean()}
        for ref in ("cv", "logged_future"):
            d = rb - cmean_boot(rows[ref]["rfs"])
            row[f"d_{ref}"], row[f"d_{ref}_lo"], row[f"d_{ref}_hi"] = (cmean(q["rfs"]) - cmean(rows[ref]["rfs"]),
                                                                      np.percentile(d, 2.5), np.percentile(d, 97.5))
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(log.dir / "results.csv", index=False)
    clus = pd.DataFrame({k: pd.Series(q["rfs"]).groupby(cl).mean() for k, q in rows.items()}).T
    clus.loc["n"] = pd.Series(cl).value_counts()
    clus.to_csv(log.dir / "clusters.csv")
    it = r["intent"]
    grp = np.where(it == 1, "straight", "turn")
    intent = pd.DataFrame({k: pd.Series(q["rfs"]).groupby(grp).mean() for k, q in rows.items()}).T
    intent.loc["n"] = pd.Series(grp).value_counts()
    intent.to_csv(log.dir / "intent.csv")
    # paired nav - nonav by intent group, frame-mean bootstrap
    if "alpamayo_nav | E[1 sample]" in rows and "alpamayo_nonav | E[1 sample]" in rows:
        d = rows["alpamayo_nav | E[1 sample]"]["rfs"] - rows["alpamayo_nonav | E[1 sample]"]["rfs"]
        nv = []
        for g in ("straight", "turn", "all"):
            m = np.ones(len(d), bool) if g == "all" else grp == g
            bs = d[m][rng.integers(0, m.sum(), (B, m.sum()))].mean(1)
            nv.append({"group": g, "n": int(m.sum()), "d_rfs_frame": d[m].mean(), "lo": np.percentile(bs, 2.5),
                       "hi": np.percentile(bs, 97.5)})
        pd.DataFrame(nv).to_csv(log.dir / "nav_effect.csv", index=False)
    # ADE-extra: ADE vs the logged future, bootstrap by sequence
    e = S["extra"]
    en = [str(x) for x in e["name"]]
    elog, eb = e["future"][..., :2], W.baselines(e["past"])
    ex = {"cv": eb["cv"], "ctra": eb["ctra"], "zero": eb["zero"]}
    p, _ = _alp(en, "nav")
    if p is not None:
        ex["alpamayo_nav | E[1 sample]"] = p
    for m in ("small", "cinque", "lebowski"):
        q = _op(en, m)
        if q is not None:
            ex[f"op_{m}"] = q
    seqs = e["sequence"].astype(str)
    useq = np.unique(seqs)
    pos = [np.flatnonzero(seqs == s) for s in useq]
    sb = rng.integers(0, len(useq), (min(B, 2000), len(useq)))
    erows = []
    for k, p in ex.items():
        d = np.linalg.norm(p - (elog[:, None] if p.ndim == 4 else elog), axis=-1)
        d = d.mean(1) if p.ndim == 4 else d                    # E over the 6 samples
        a3, a5 = d[:, :12].mean(1), d.mean(1)
        per_seq = np.array([a5[i].mean() for i in pos])
        bs = per_seq[sb].mean(1)
        erows.append({"row": k, "n": len(en), "ade3_log": a3.mean(), "ade5_log": a5.mean(),
                      "ade5_lo": np.percentile(bs, 2.5), "ade5_hi": np.percentile(bs, 97.5)})
    pd.DataFrame(erows).to_csv(log.dir / "extra.csv", index=False)
    np.savez_compressed(log.dir / "per_frame.npz", names=np.array(names), cluster=cl, intent=it, traj=traj,
                        scores=sc, logged=log_xy, speed=speed,
                        **{f"pred/{k}": v for k, v in cands.items()}, **{f"rfs/{k}": q["rfs"] for k, q in per.items()})
    (log.dir / "cot.json").write_text(json.dumps({v: dict(zip(names, c)) for v, c in cots.items()}))
    pd.set_option("display.width", 250)
    log.info("\n" + res[["row", "rfs", "rfs_lo", "rfs_hi", "rfs_frame", "in_trust", "floored", "ade3_rater",
                         "ade5_rater", "ade5_log", "d_cv", "d_cv_lo", "d_cv_hi", "lon5_bias"]].round(3).to_string())
    log.info("\n" + pd.DataFrame(erows).round(3).to_string())
    log.info(f"results -> {log.dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--boot", type=int, default=10000)
    for name in ("fetch", "packages", "views"):
        p = sub.add_parser(name)
        p.add_argument("--set", nargs="+", default=list(SETS), choices=SETS)
        if name == "fetch":
            p.add_argument("--route", default="proxy", choices=("proxy", "direct"))
            p.add_argument("--workers", type=int, default=96)
        if name == "views":
            p.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    log = RunLog("wod_zeroshot", a.cmd)
    log.event("start", args=vars(a))
    globals()[f"cmd_{a.cmd}"](a, log)
    log.event("end")


if __name__ == "__main__":
    main()
