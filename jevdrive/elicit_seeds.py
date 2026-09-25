"""Overnight queue item 2 ([SEEDS], todos/2026-09-26-overnight-queue.md): seed-0 reproduction checks and the 3-seed
tables of the M-C, NAVSIM, WOD and Q9b heads (seed 0 = the stored run; seeds 1, 2 from scripts/seeds_overnight.sh).

    python -m jevdrive.elicit_seeds check     # the seed-0 re-runs against the stored runs
    python -m jevdrive.elicit_seeds report    # tables into research/results/elicitation/seeds/
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
R = data_dir() / "runs"
OUT = Path(__file__).resolve().parents[1] / "research/results/elicitation/seeds"
MC_STORED = {"ba": "reactivity/mc-carla_p5v1_ba/20260925-233126", "pdm": "reactivity/mc-carla_p5v1_pdm/20260925-230424"}
NAV_STORED = "navsim_zs/heads/20260925-232810"
WOD_STORED = "drive_backbones/heads_train/20260925-110819"
Q9B_STORED = {"ba": "fusion_diag/q9b-fit-v1-ba/20260926-000240", "pdm": "fusion_diag/q9b-fit-v1-pdm/20260926-003949"}
NAV_ARMS = ("ridge_late_cinque_temporal", "cls_late_cinque_temporal", "ridge_late_lebowski_temporal",
            "cls_late_lebowski_temporal", "ridge_ego", "cls_ego_K1024")


def latest(pattern: str) -> Path | None:
    fs = sorted(glob.glob(str(R / pattern)))
    return Path(fs[-1]) if fs else None


def _npz_diff(a: Path, b: Path) -> float:
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    d = 0.0
    for k in za.files:
        if za[k].dtype.kind == "f":
            d = max(d, float(np.nanmax(np.abs(za[k] - zb[k]))))
        else:
            assert (za[k] == zb[k]).all(), k
    return d


def check() -> pd.DataFrame:
    rows = []
    new = latest("reactivity/mc-carla_p5v1_ba-seed0/*")
    rows.append({"what": "M-C BA preds_obs", "max_abs_diff_m": _npz_diff(new / "preds_obs.npz", R / MC_STORED["ba"] / "preds_obs.npz"),
                 "criteria_equal": pd.read_csv(new / "criteria.csv").round(10).equals(pd.read_csv(R / MC_STORED["ba"] / "criteria.csv").round(10))})
    new = latest("navsim_zs/heads-s0/*")
    for f in sorted((R / NAV_STORED).glob("*.npz")):
        rows.append({"what": f"NAVSIM {f.stem}", "max_abs_diff_m": _npz_diff(new / f.name, f)})
    new = latest("drive_backbones/heads_train-seed0/*")
    rows.append({"what": "WOD heads preds", "max_abs_diff_m": _npz_diff(new / "p3drive_heads_preds_dir0.npz",
                                                                        R / WOD_STORED / "p3drive_heads_preds_dir0.npz")})
    t = pd.DataFrame(rows)
    log.info("seed-0 reproduction\n%s", t.to_markdown(index=False, floatfmt=".3g"))
    return t


def _summ(df: pd.DataFrame, keys: list, hw_col: str = "halfwidth") -> pd.DataFrame:
    """One row per key: the three seeds' values, mean, range, seed 0's CI half-width, range < half-width."""
    out = []
    for k, g in df.groupby(keys, sort=False):
        v = g.set_index("seed")["value"]
        hw = float(g.set_index("seed")[hw_col].get(0, np.nan))
        out.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                    **{f"s{s}": v.get(s, np.nan) for s in (0, 1, 2)}, "mean": v.mean(), "range": v.max() - v.min(),
                    "ci_halfwidth_s0": hw, "range_lt_halfwidth": bool(v.max() - v.min() < hw) if len(v) == 3 else None})
    return pd.DataFrame(out)


def mc() -> pd.DataFrame:
    rows = []
    for st in ("ba", "pdm"):
        for s in (0, 1, 2):
            d = R / MC_STORED[st] if s == 0 else latest(f"reactivity/mc-carla_p5v1_{st}-seed{s}/*")
            if d is None:
                continue
            c = pd.read_csv(d / "criteria.csv")
            for _, r in c.iterrows():
                arm, model = r.arm.rsplit(" [", 1)
                base = {"set": st, "model": model.rstrip("]"), "arm": arm, "seed": s}
                rows += [{**base, "metric": "ped_flip", "value": r.ped_flip, "halfwidth": (r.ped_hi - r.ped_lo) / 2},
                         {**base, "metric": "cutin_delta_vs_prior", "value": r.cutin_delta_vs_prior,
                          "halfwidth": (r.cutin_hi - r.cutin_lo) / 2},
                         {**base, "metric": "null_ff_oos", "value": r.null_ff_oos, "halfwidth": np.nan},
                         {**base, "metric": "pass", "value": float(r["pass"]), "halfwidth": np.nan}]
    return _summ(pd.DataFrame(rows), ["set", "model", "arm", "metric"])


def _scores(ver: str, split: str, name: str):
    fs = sorted(glob.glob(str(R / "navsim/eval" / f"{ver}_{split}_{name}" / "*" / "*.csv")))
    return pd.read_csv(fs[-1]) if fs else None


def nav() -> pd.DataFrame:
    rows, tok = [], {}
    rng = np.random.default_rng(0)
    for s in (0, 1, 2):
        for arm in NAV_ARMS:
            name = f"heads_{arm}" if s == 0 else f"heads_s{s}_{arm}"
            for ver, metric in (("v1", "PDMS"), ("v2", "EPDMS")):
                df = _scores(ver, "navtest", name)
                if df is None:
                    continue
                df = df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)]
                sc = df.set_index("token")["score"].astype(float)
                tok[(s, arm, metric)] = sc
                v = sc.to_numpy()
                bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
                rows.append({"arm": arm, "metric": metric, "seed": s, "value": 100 * v.mean(),
                             "halfwidth": 100 * (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2})
            df = _scores("v2", "navhard_two_stage", name)
            if df is not None:
                summ = df[df["token"].str.startswith("extended_pdm_score")].set_index("token")["score"]
                rows.append({"arm": arm, "metric": "navhard EPDMS", "seed": s,
                             "value": 100 * summ.get("extended_pdm_score_combined", np.nan), "halfwidth": np.nan})
        for late, ego in (("ridge_late_cinque_temporal", "ridge_ego"), ("ridge_late_lebowski_temporal", "ridge_ego"),
                          ("cls_late_cinque_temporal", "cls_ego_K1024"), ("cls_late_lebowski_temporal", "cls_ego_K1024")):
            for metric in ("PDMS", "EPDMS"):
                if (s, late, metric) in tok and (s, ego, metric) in tok:
                    x, y = tok[(s, late, metric)].align(tok[(s, ego, metric)], join="inner")
                    d = (x - y).to_numpy()
                    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
                    rows.append({"arm": f"{late} - {ego}", "metric": metric, "seed": s, "value": 100 * d.mean(),
                                 "halfwidth": 100 * (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2})
    return _summ(pd.DataFrame(rows), ["arm", "metric"])


def wod() -> pd.DataFrame:
    rows = []
    for s in (0, 1, 2):
        d = R / WOD_STORED if s == 0 else latest(f"drive_backbones/heads_train-seed{s}/*")
        if d is None:
            continue
        a = pd.read_csv(d / "heads_rfs_arms_p3drive_heads.csv")
        rows += [{"row": f"RFS cluster mean: {r.arm}", "seed": s, "value": r.rfs_cluster_mean, "halfwidth": np.nan}
                 for r in a.itertuples()]
        p = pd.read_csv(d / "heads_rfs_paired_p3drive_heads.csv")
        want = [(f"cls_late op-{m} temporal", "cls ego K1024") for m in ("cinque", "lebowski")] + \
               [(f"A ridge_late op-{m} temporal", "ridge ego") for m in ("cinque", "lebowski")] + [("cls ego K1024", "ridge ego")]
        rows += [{"row": f"RFS paired: {r.arm} - {r.vs}", "seed": s, "value": r.delta, "halfwidth": r.halfwidth}
                 for r in p.itertuples() if (r.arm, r.vs) in want]
        j = pd.read_csv(d / "rejudge_p3drive_heads.csv")
        j = j[(j.judge == "ADE vs log") & (j.scope == "s_ego deciles 1-9") & j.subset.isin(("pre_onset", "all"))]
        rows += [{"row": f"ADE dec1-9 {r.subset} delta vs ridge ego: {r.arm}", "seed": s, "value": r.delta,
                  "halfwidth": r.halfwidth} for r in j.itertuples()]
    return _summ(pd.DataFrame(rows), ["row"])


def q9b() -> pd.DataFrame:
    rows = []
    for st in ("ba", "pdm"):
        c0 = pd.read_csv(R / Q9B_STORED[st] / "criteria.csv")
        for _, r in c0.iterrows():
            arm, model = r.arm.rsplit(" [", 1)
            if arm.startswith("Q9b pair grid s"):
                rows.append({"set": st, "model": model.rstrip("]"), "arm": "Q9b pair grid", "seed": int(arm[-1]),
                             "value": r.ped_flip, "halfwidth": (r.ped_hi - r.ped_lo) / 2, "null": r.null_ff_oos})
            elif arm == "Q9b hard grid":
                rows.append({"set": st, "model": model.rstrip("]"), "arm": arm, "seed": 0, "value": r.ped_flip,
                             "halfwidth": (r.ped_hi - r.ped_lo) / 2, "null": r.null_ff_oos})
        for s in (1, 2):
            d = latest(f"fusion_diag/q9b-fit-v1-{st}-hardseed{s}/*")
            if d is None:
                continue
            c = pd.read_csv(d / "criteria.csv")
            for _, r in c.iterrows():
                arm, model = r.arm.rsplit(" [", 1)
                if arm == "Q9b hard grid":
                    rows.append({"set": st, "model": model.rstrip("]"), "arm": arm, "seed": s, "value": r.ped_flip,
                                 "halfwidth": (r.ped_hi - r.ped_lo) / 2, "null": r.null_ff_oos})
                elif arm.startswith("Q9b pair grid s"):   # the re-run's pair arms: reproduction of the stored ones
                    old = c0[c0.arm == r.arm].iloc[0]
                    log.info("Q9b %s %s re-run (hard seed %d): ped flip %.4f vs stored %.4f", st, r.arm, s, r.ped_flip, old.ped_flip)
    return _summ(pd.DataFrame(rows), ["set", "model", "arm"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "check":
        check().to_csv(OUT / "seed0_reproduction.csv", index=False)
        return
    for name, f in (("mc", mc), ("navsim", nav), ("wod", wod), ("q9b", q9b)):
        try:
            t = f()
        except Exception as e:           # a part whose runs are not there yet
            log.warning("%s: %s", name, e)
            continue
        t.to_csv(OUT / f"{name}_seeds.csv", index=False, float_format="%.4f")
        log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
