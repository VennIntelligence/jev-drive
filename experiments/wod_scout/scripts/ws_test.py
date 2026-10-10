"""WODSCOUT Q1 / A1 + A3: are the 1 505 WOD-E2E test submission frames distinguishable from the 479 val rater frames on what we can measure
(plans/2026-10-10-wodscout-prereg.md)? Test inputs and our own stored test plans only; no test label exists or is reconstructed.

  python experiments/wod_scout/scripts/ws_test.py [--sheets 12]            (jevdrive env, CPU, box, a few minutes)

A1  per-frame covariates (ego state, intent, luma of the target FRONT frame, shipped lead output, WLG plan shape), two-sample tests with Holm
    correction, and a classifier two-sample test (logistic and gradient boosting, 5-fold, permutation null). The classifier is a statistical test only.
A3  a small look: N random test and N random val frames (seed 0) as sheets for the pre-registered rubric.

Tables -> experiments/wod_scout/results/q1_spotlight/, sheets -> experiments/wod_scout/figs/q1_look_*.webp.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_P = _R / "experiments/op_parity/scripts"
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_P), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, io, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT, FIG = _R / "experiments/wod_scout/results/q1_spotlight", _R / "experiments/wod_scout/figs"
TAGS = ("WLG-full-s0", "WLG-full-s1")
B, NPERM = 4000, 200
_IDX = {}


def _index():
    if not _IDX:
        from jevdrive import waymo as W
        df = W.load_index()
        _IDX.update(df=df, key={nm: i for i, nm in enumerate(W.frame_names(df))}, dir=W.shard_dir())
    return _IDX


def jpeg(nm, cam="front"):
    from PIL import Image
    I = _index()
    rw = I["df"].iloc[I["key"][nm]]
    with open(I["dir"] / rw.shard, "rb") as fh:
        fh.seek(int(rw[f"{cam}_off"]))
        return Image.open(io.BytesIO(fh.read(int(rw[f"{cam}_len"]))))


def luma(nm):
    im = jpeg(nm)
    im.draft("L", (im.width // 8, im.height // 8))
    return float(np.asarray(im.convert("L"), np.float32).mean())


def covariates(names, past, intent, run):
    from jevdrive import par
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from ws_val import arc, heading_change
    own = [np.stack([np.load(Z.root("preds", f"op_cinque_{t}") / f"{n}.npz")["wod"] for n in names]).astype(np.float64)[..., :2] for t in TAGS]
    ens = np.mean(own, 0)
    lead = np.stack([np.asarray(np.load(Z.root("preds", "op_cinque") / f"{n}.npz")["lead_prob"]).reshape(-1)[:3] for n in names])
    lum = par.pmap(luma, list(names), run=run, workers=24)
    lum.raise_if_failed()
    hd = heading_change(ens)
    d5 = np.linalg.norm(ens[:, -1], axis=-1)
    return pd.DataFrame({"name": names, "v0": W.init_speed(past), "ax": past[:, -1, 4], "ay": past[:, -1, 5], "intent_left": (intent == 2).astype(float),
                         "intent_right": (intent == 3).astype(float), "luma": np.array(lum.values), "lead_prob_0": lead[:, 0], "lead_prob_1": lead[:, 1], "lead_prob_2": lead[:, 2],
                         "plan_d5": d5, "plan_arc5": arc(ens)[:, -1], "plan_abs_y5": np.abs(ens[:, -1, 1]), "plan_abs_heading": np.abs(hd),
                         "plan_seed_gap5": np.linalg.norm(own[0][:, -1] - own[1][:, -1], axis=-1)}), ens


def main(a):
    from scipy.stats import chi2_contingency, ks_2samp
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from jevdrive import stats
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.data import splits
    from jevdrive.run import Run
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with Run("wod_scout", "test-shift", seed=0, config=vars(a)) as run:
        for s in ("wod/val", "wod/test"):
            run.use_split(splits.load(s))
        S = Z.load_sets()
        I = _index()
        past_all, _ = W.load_ego()
        part = {}
        for sp_, nm in (("val", S["rater"]["name"].astype(str)), ("test", S["test"]["name"].astype(str))):
            rows = np.array([I["key"][n] for n in nm])
            part[sp_] = covariates(nm, past_all[rows].astype(np.float64), I["df"].intent.to_numpy()[rows], run)
        dv, dt = part["val"][0], part["test"][0]
        dv["cluster"] = S["rater"]["cluster"].astype(str)
        D = pd.concat([dv.assign(split="val"), dt.assign(split="test")], ignore_index=True)
        D.to_csv(OUT / "covariates.csv", index=False)
        binary = {"stopped (v0 < 0.5)": lambda d: d.v0 < 0.5, "slow (0.5-5)": lambda d: (d.v0 >= 0.5) & (d.v0 < 5), "mid (5-12)": lambda d: (d.v0 >= 5) & (d.v0 < 12),
                  "fast (>= 12)": lambda d: d.v0 >= 12, "intent left": lambda d: d.intent_left > 0, "intent right": lambda d: d.intent_right > 0,
                  "night (frame luma < 50)": lambda d: d.luma < 50, "dusk band (50 <= luma < 120)": lambda d: (d.luma >= 50) & (d.luma < 120),
                  "lead_prob > 0.5": lambda d: d.lead_prob_0 > 0.5, "plan hold (d5 < 1 m)": lambda d: d.plan_d5 < 1, "plan creep (1 <= d5 < 5 m)": lambda d: (d.plan_d5 >= 1) & (d.plan_d5 < 5),
                  "plan turn (heading >= 25 deg)": lambda d: d.plan_abs_heading >= 25, "plan lane shift (|y5| 1-4 m, heading < 15 deg)":
                  lambda d: (d.plan_abs_y5 >= 1) & (d.plan_abs_y5 < 4) & (d.plan_abs_heading < 15), "braking (ax < -1)": lambda d: d.ax < -1, "accelerating (ax > 1)": lambda d: d.ax > 1}
        cont = ["v0", "ax", "luma", "lead_prob_0", "plan_d5", "plan_abs_y5", "plan_abs_heading", "plan_seed_gap5"]
        rng = np.random.default_rng(0)
        iv, it = rng.integers(len(dv), size=(B, len(dv))), rng.integers(len(dt), size=(B, len(dt)))
        rows = []
        for k, f in binary.items():
            xv, xt = f(dv).to_numpy(float), f(dt).to_numpy(float)
            bs = xt[it].mean(1) - xv[iv].mean(1)
            p = chi2_contingency([[xv.sum(), len(xv) - xv.sum()], [xt.sum(), len(xt) - xt.sum()]])[1]
            rows.append({"covariate": k, "kind": "share", "val": xv.mean(), "test": xt.mean(), "test - val": xt.mean() - xv.mean(), "lo": np.percentile(bs, 2.5),
                         "hi": np.percentile(bs, 97.5), "p": p})
        for k in cont:
            xv, xt = dv[k].to_numpy(float), dt[k].to_numpy(float)
            bs = np.median(xt[it], 1) - np.median(xv[iv], 1)
            rows.append({"covariate": k, "kind": "median", "val": np.median(xv), "test": np.median(xt), "test - val": np.median(xt) - np.median(xv),
                         "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "p": ks_2samp(xv, xt).pvalue})
        T = pd.DataFrame(rows)
        o = np.argsort(T.p.to_numpy())
        holm = np.empty(len(T))
        run_max = 0.0
        for r_, j in enumerate(o):
            run_max = max(run_max, min(1.0, (len(T) - r_) * T.p.iloc[j]))
            holm[j] = run_max
        T["p Holm"] = holm
        T["differs (Holm < 0.05)"] = T["p Holm"] < 0.05
        stats.write_table(T.to_dict("records"), OUT / "shift")
        run.info("shift:\n%s", T.to_string(float_format=lambda v: f"{v:.3f}"))

        # classifier two-sample test (val = 0, test = 1); one frame per sequence in both sets, so frames are the units
        feats = [c for c in dv.columns if c not in ("name", "cluster")]
        X, y = D[feats].to_numpy(float), (D.split == "test").to_numpy(int)
        models = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                  "gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, class_weight="balanced", random_state=0)}
        crow, score = [], {}
        for nm, mk in models.items():
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pr = cross_val_predict(mk(), X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, pr)
            null = []
            for k in range(NPERM if nm == "logistic" else NPERM // 4):
                yp = np.random.default_rng(100 + k).permutation(y)
                null.append(roc_auc_score(yp, cross_val_predict(mk(), X, yp, cv=cv, method="predict_proba")[:, 1]))
            null = np.array(null)
            crow.append({"classifier": nm, "features": len(feats), "AUC (5-fold, out of fold)": auc, "null mean": null.mean(), "null 97.5 %": np.percentile(null, 97.5),
                         "permutations": len(null), "p": (1 + (null >= auc).sum()) / (1 + len(null)), "separable (p < 0.05 and AUC >= 0.60)": bool(auc >= 0.60 and
                         (1 + (null >= auc).sum()) / (1 + len(null)) < 0.05)})
            score[nm] = pr
        stats.write_table(crow, OUT / "c2st")
        run.info("c2st:\n%s", pd.DataFrame(crow).to_string(float_format=lambda v: f"{v:.3f}"))
        D["c2st_logistic"], D["c2st_gb"] = score["logistic"], score["gradient boosting"]
        D.to_csv(OUT / "covariates.csv", index=False)
        # val share by cluster of the same covariates (to see which val clusters look most like the test surplus, descriptive)
        byc = dv.assign(stopped=dv.v0 < 0.5, night=dv.luma < 50, lead=dv.lead_prob_0 > 0.5, hold=dv.plan_d5 < 1, turn=dv.plan_abs_heading >= 25) \
            .groupby("cluster")[["stopped", "night", "lead", "hold", "turn", "v0", "luma"]].mean().reset_index()
        byc["n"] = dv.groupby("cluster").size().to_numpy()
        stats.write_table(byc.to_dict("records"), OUT / "val_clusters_covariates")
        if a.sheets:
            r2 = np.random.default_rng(0)
            pick = [("test", dt.name[i], part["test"][1][i], dt.v0[i], "") for i in np.sort(r2.choice(len(dt), a.sheets, replace=False))] + \
                   [("val", dv.name[i], part["val"][1][i], dv.v0[i], dv.cluster[i]) for i in np.sort(r2.choice(len(dv), a.sheets, replace=False))]
            look(pick)
            pd.DataFrame([{"sheet_row": j + 1, "split": p[0], "name": p[1], "v0": p[3], "val cluster": p[4]} for j, p in enumerate(pick)]).to_csv(OUT / "look_frames.csv", index=False)
        run.summary.update(auc={r_["classifier"]: r_["AUC (5-fold, out of fold)"] for r_ in crow}, n_differs=int(T["differs (Holm < 0.05)"].sum()))
        (OUT / "shift_meta.json").write_text(json.dumps(dict(n_val=len(dv), n_test=len(dt), features=feats, B=B), indent=1))


def look(pick, per=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    for s0 in range(0, len(pick), per):
        blk = pick[s0:s0 + per]
        fig, axs = plt.subplots(len(blk), 2, figsize=(12.5, 3.3 * len(blk)), gridspec_kw=dict(width_ratios=[3.2, 1]))
        for j, ((ax_i, ax_b), (sp_, nm, plan, v0, c)) in enumerate(zip(np.atleast_2d(axs), blk)):
            ims = [jpeg(nm, cam).convert("RGB") for cam in ("front_left", "front", "front_right")]
            h = min(i.height for i in ims)
            ax_i.imshow(np.concatenate([np.asarray(i.resize((int(i.width * h / i.height), h)))[int(h * 0.22):int(h * 0.82)] for i in ims], 1))
            ax_i.axis("off")
            ax_i.set_title(f"row {s0 + j + 1}  {sp_}  {nm}  |  v0 {v0:.1f} m/s  {('| ' + c) if c else ''}", fontsize=8.5, loc="left")
            ax_b.plot(-np.r_[0, plan[:, 1]], np.r_[0, plan[:, 0]], color="#2a78d6", lw=2)
            ax_b.plot(0, 0, "^", ms=7, color="#0b0b0b")
            half = max(4.0, np.abs(plan[:, 1]).max() * 1.2 + 1)
            ax_b.set_xlim(-half, half)
            ax_b.set_ylim(-1, max(6.0, plan[:, 0].max() * 1.08 + 1))
            ax_b.grid(color="#e4e3df", lw=0.5)
            ax_b.tick_params(labelsize=7)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, format="png")
        plt.close(fig)
        Image.open(buf).convert("RGB").save(FIG / f"q1_look_sheet{s0 // per + 1}.webp", quality=72, method=6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=12)
    main(ap.parse_args())
