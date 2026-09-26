"""Night queue 3, lane D, Q4b: NAVSIM-protocol compatibility check of the NAVSIM-trained heads on P5 v1 BA
(todos/2026-09-26-night-queue-3.md, Q4b and the [D] Q4b entry, written before any Q4b number).

  features  scripts/nq3_d/q4b_openpilot.py (envs/openpilot): `temporal` of every P5 obs row under the NAVSIM input
            protocol -> processed/carla_p5v1_ba/nq3_q4b_navsim_protocol/<model>.npz
  hydra     night2_n3.fit unchanged per model x seed, P5 rows carrying the NAVSIM-protocol features (navtrain
            statistics, the N3 ego construction); NAVSIM `ridge_late` zero-shot on the same rows (sanity, description)
  compat    decision 53's check unchanged (night2_n3.compat's statistic): top-10 overlap of Hydra's selected anchors on
            the P5 null frames vs navtest, seed 0 decides at 30 %; the N3 (5 Hz protocol) numbers beside
  exam      only when comparable: Hydra per seed on P5 (p5_exam.exam + reactivity_mc.criteria against the seed's P5
            prior, N3 [B] 09:58 (5)), NAVSIM `ridge_late` / `cls_late` alongside, and decision 53's criterion 1

Run on the box: P5_SET=carla_p5v1_ba python -m jevdrive.nq3_q4b <step>
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
SEEDS = (0, 1, 2)
FEAT = "processed/carla_p5v1_ba/nq3_q4b_navsim_protocol"
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "nq3" / "q4"


def out(*p) -> Path:
    d = data_dir() / "runs" / "nq3" / "q4b"
    (d / Path(*p)).parent.mkdir(parents=True, exist_ok=True)
    return d / Path(*p)


def p5_sets(model: str, p5: dict) -> dict:
    """night2_n3.extra_sets' P5 entry with the NAVSIM-protocol `temporal` in place of op_streams_vis."""
    from .night2_n3 import nav_ego
    z = np.load(data_dir() / FEAT / f"{model}.npz")
    r = p5["rows"]
    names = p5["t"].frame_name.to_numpy()[r]
    at = pd.Series(np.arange(len(z["name"])), index=z["name"]).reindex(names)
    assert at.notna().all(), f"{int(at.isna().sum())} P5 rows without NAVSIM-protocol features"
    return {"p5": (nav_ego(p5["past"][r], p5["t"].intent.to_numpy()[r]), z["temporal"][at.astype(int).to_numpy()].astype(np.float32), names)}


def hydra(rl, models=MODELS, seeds=SEEDS):
    from . import elicit_e6 as E6, navsim_heads as H, night2_n3 as N3
    p5 = N3.p5_data()
    orig = N3.extra_sets
    sets = {m: p5_sets(m, p5) for m in models}
    try:
        for m in models:
            N3.extra_sets = lambda model, m=m: sets[m]
            for s in seeds:
                N3.fit(rl, m, s)
                z = np.load(rl.dir / f"sel_{m}_s{s}.npz", allow_pickle=True)
                ref = np.load(N3._sel_files()[(m, s)], allow_pickle=True)
                same = float((z["navtest_hydra"] == ref["navtest_hydra"]).mean())
                rl.event("q4b_hydra_check", model=m, seed=s, navtest_same_anchor=same)
                rl.log.info("%s s%d: refit Hydra navtest selection = stored N3 on %.4f of tokens", m, s, same)
                assert same >= 0.99, "Hydra refit does not reproduce the stored N3 selection"
                (out("sel") / "x").parent.mkdir(parents=True, exist_ok=True)
                (out("sel") / f"sel_{m}_s{s}.npz").write_bytes((rl.dir / f"sel_{m}_s{s}.npz").read_bytes())
    finally:
        N3.extra_sets = orig
    # NAVSIM ridge_late zero-shot on the same rows (N3 navridge's recipe; description only)
    tr = E6._navtrain(True)
    folds = H._group_folds(tr["log"], H.FOLDS, seed=0)
    Y = torch.as_tensor(tr["fut"].reshape(len(tr["fut"]), -1), device=H.DEV)
    res = {}
    for m in models:
        ego, feat, names = sets[m]["p5"]
        Xe, Xe5 = H._std(tr["ego"], ego)
        base, oof_e, _ = H.fit_ridge(Xe, Y, {"p5": Xe5}, folds)
        Xf, Xf5 = H._std(tr[m], feat)
        r, _, _ = H.fit_ridge(Xf, Y - oof_e, {"p5": Xf5}, folds)
        res[f"p5_{m}"], res["p5_keys"] = (base["p5"] + r["p5"]).reshape(-1, 8, 3).cpu().numpy(), names
    np.savez_compressed(out("navridge.npz"), **res)


def compat(rl) -> pd.DataFrame:
    from . import night2_n3 as N3
    p5n = N3._null_frames(pd.read_parquet(data_dir() / "processed/carla_p5v1_ba/null.parquet"))
    top = lambda sel: set(pd.Series(sel).value_counts().index[:10])  # noqa: E731
    rows = []
    for f in sorted(out("sel").glob("sel_*_s*.npz")):
        m, s = f.stem.split("_")[1], int(f.stem.split("_s")[-1])
        for proto, z in (("navsim 2 Hz (Q4b)", np.load(f, allow_pickle=True)),
                         ("P5 5 Hz (N3)", np.load(N3._sel_files()[(m, s)], allow_pickle=True))):
            keys = pd.Series(np.arange(len(z["p5_keys"])), index=z["p5_keys"].astype(str))
            r = keys.reindex(p5n.astype(str)).dropna().astype(int).to_numpy()
            rows.append({"model": m, "seed": s, "protocol": proto, "null_frames": len(r),
                         "top10_overlap": len(top(z["p5_hydra"][r]) & top(z["navtest_hydra"])) / 10,
                         "cls_late_top10_overlap": len(top(z["p5_clsref"][r]) & top(z["navtest_clsref"])) / 10,
                         "distinct_anchors_null": int(len(np.unique(z["p5_hydra"][r]))),
                         "dac_sel_diff": float(z["p5_dac_sel"][r].mean() - z["navtest_dac_sel"].mean()),
                         "comparable": len(top(z["p5_hydra"][r]) & top(z["navtest_hydra"])) / 10 >= 0.3})
    t = pd.DataFrame(rows)
    q = t[t.protocol.str.startswith("navsim")]
    verdict = {m: bool(q[(q.model == m) & (q.seed == 0)].comparable.iloc[0]) for m in q.model.unique()}
    for dst in (out(), RESULTS):
        dst.mkdir(parents=True, exist_ok=True)
        t.to_csv(dst / "q4b_compat.csv", index=False, float_format="%.4f")
        (dst / "q4b_verdict.json").write_text(json.dumps({"comparable_seed0": verdict}, indent=1))
    rl.log.info("compatibility\n%s\nseed-0 verdict %s", t.to_markdown(index=False, floatfmt=".3f"), verdict)
    return t


def exam(rl, models=MODELS, seeds=SEEDS):
    """Hydra's P5 flips under the NAVSIM protocol (only for the models judged comparable) with decision 53's criterion 1."""
    from . import night2_n3 as N3, p5_exam as E, reactivity_mc as MC
    ok = json.loads(out("q4b_verdict.json").read_text())["comparable_seed0"]
    models = [m for m in models if ok.get(m)]
    if not models:
        rl.log.info("no model comparable: Hydra's P5 cells stay 「不可比」 (nothing to examine)")
        (out("exam.skipped")).write_text("not comparable\n")
        return
    p5 = N3.p5_data()
    t, n, rows = p5["t"], len(p5["t"]), p5["rows"]
    nr = np.load(out("navridge.npz"), allow_pickle=True)
    mc = {s: np.load(data_dir() / N3.MC_RUNS[s] / "preds_obs.npz") for s in seeds}
    P5 = {}

    def put(name, v):
        a = np.full((n, 20, 2), np.nan, np.float32)
        a[rows] = v
        P5[name] = a
    for m in models:
        put(f"ridge_late NAV (2 Hz) [{m}]", N3._grid(nr[f"p5_{m}"][N3._rows_of(nr["p5_keys"], t.frame_name.to_numpy()[rows])]))
        for s in seeds:
            put(f"prior s{s} [{m}]", mc[s][f"prior [{m}]"])
            z = np.load(out("sel", f"sel_{m}_s{s}.npz"), allow_pickle=True)
            r5 = N3._rows_of(z["p5_keys"], t.frame_name.to_numpy()[rows])
            put(f"Hydra s{s} (2 Hz) [{m}]", N3._grid(z["anchors"][z["p5_hydra"][r5]]))
            put(f"cls_late NAV s{s} (2 Hz) [{m}]", N3._grid(z["anchors"][z["p5_clsref"][r5]]))
    oo, nn = E.deltas(p5["obs"], p5["null"], t, P5)
    res = E.exam(oo, nn, p5["pairs"], list(P5))
    crit = []
    for m in models:
        for s in seeds:
            arms = [k for k in P5 if k.endswith(f"[{m}]") and not k.startswith("prior") and (f"s{s} " in k or " s" not in k)]
            c = MC.criteria(res, arms, f"prior s{s} [{m}]").assign(model=m, seed=s)
            pe = float(MC.criteria(res, [f"prior s{s} [{m}]"], f"prior s{s} [{m}]").ped_flip.iloc[0])
            c["prior_ped_point"] = pe
            h = c.arm == f"Hydra s{s} (2 Hz) [{m}]"
            c.loc[h, "criterion1"] = np.where(c.loc[h, "ped_hi"] < pe, "榜单 head 压掉反应", "同一水平")
            crit.append(c)
    crit = pd.concat(crit)
    for dst in (out(), RESULTS):
        crit.to_csv(dst / "q4b_p5_criteria.csv", index=False, float_format="%.4f")
    res["flips"].to_csv(out("q4b_p5_flip_rates.csv"), index=False)
    rl.log.info("P5 criteria (NAVSIM protocol)\n%s", crit.to_markdown(index=False, floatfmt=".3f"))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("hydra", "compat", "exam"))
    a = ap.parse_args()
    rl = RunLog("nq3", f"q4b-{a.step}")
    {"hydra": lambda: hydra(rl), "compat": lambda: compat(rl), "exam": lambda: exam(rl)}[a.step]()
    rl.close()


if __name__ == "__main__":
    main()
