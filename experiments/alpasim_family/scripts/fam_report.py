"""FAMILY: the sensitivity table (plans/2026-10-12-family-prereg.md) from the collected per-scene / per-decision tables.

  fam_report.py <collect dir> [<collect dir> ...] --out DIR      (Mac, project venv)

Collect dirs come from proxy_collect.py on the box (per_scene.csv, per_scene_k.parquet). Labels: our arms are compared with A0, LTF arms
(label starting with L) with L0, APY arms (APY_*) with APY_A0. G0 / G1 compare A0 / L0 with the stored YR10m10-F-s0 (tmp/aud1_drift) and LTF anchor tables on the
same scenes. CIs: scene bootstrap (10 000 draws); the log-clustered one is reported next to it (8 logs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
VEH = pd.read_csv(ROOT / "experiments/alpasim_family/results/asset_calib_public.csv")[["scene", "vehicle"]]
B = 10_000


def boot(d: pd.DataFrame, f, groups=None, seed=0):
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(d), (B, len(d)))
        return np.percentile([f(d.iloc[i]) for i in idx], [2.5, 97.5])
    g = d.groupby(groups).indices
    keys = list(g)
    out = []
    for _ in range(B // 10):
        pick = rng.choice(len(keys), len(keys))
        out.append(f(d.iloc[np.concatenate([g[keys[i]] for i in pick])]))
    return np.percentile(out, [2.5, 97.5])


def summary(s: pd.DataFrame, k: pd.DataFrame) -> dict:
    gt = s.dist_to_gt_trajectory
    ev = s.offroad_or_collision_at_fault.sum()
    m = k.groupby("k").dist_gt.mean()
    sl = np.polyfit(m.loc[4:10].index, m.loc[4:10].values, 1)[0]
    return dict(n=len(s), score=s.score.mean(), gt=gt.mean(), ge4=(gt >= 4).mean(), events=int(ev), km=s.dist_traveled_m.sum() / max(ev, 1) / 1e3,
                zeros=int((s.score == 0).sum()), coll=int((s.collision_at_fault > 0).sum()), offroad=int((s.offroad > 0).sum()),
                corridor=int((s.left_corridor_laterally > 0).sum()), d4=m.get(4), d7=m.get(7), d10=m.get(10), slope=sl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = pd.concat([pd.read_csv(Path(d) / "per_scene.csv") for d in a.dirs]).drop_duplicates(["driver", "scene"])
    K = pd.concat([pd.read_parquet(Path(d) / "per_scene_k.parquet") for d in a.dirs]).drop_duplicates(["driver", "scene", "k"])
    S = S.merge(VEH, on="scene")
    S["log"] = S.scene.str.rsplit("-", n=1).str[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    def ctl_of(x):
        return "L0" if x.startswith("L") else "APY_A0" if x.startswith("APY_") else "A0"
    for lab in sorted(S.driver.unique(), key=lambda x: (ctl_of(x), x != ctl_of(x), x)):
        ctl = ctl_of(lab)
        s, c = S[S.driver == lab].set_index("scene"), S[S.driver == ctl].set_index("scene")
        sc = s.index.intersection(c.index)
        s, c = s.loc[sc], c.loc[sc]
        r = {"arm": lab, "control": ctl, **summary(s.reset_index(), K[(K.driver == lab) & K.scene.isin(sc)])}
        if lab != ctl:
            p = pd.DataFrame({"ds": s.score - c.score, "a": s.dist_to_gt_trajectory, "b": c.dist_to_gt_trajectory, "log": s.log})
            r["dscore"] = p.ds.mean()
            r["dscore_lo"], r["dscore_hi"] = boot(p, lambda x: x.ds.mean())
            r["dscore_log_lo"], r["dscore_log_hi"] = boot(p.reset_index(drop=True), lambda x: x.ds.mean(), "log")
            r["gt_ratio"] = p.a.mean() / p.b.mean()
            r["gt_ratio_lo"], r["gt_ratio_hi"] = boot(p, lambda x: x.a.mean() / x.b.mean())
            kk = K[K.k == 10].merge(VEH, on="scene")
            v = kk[kk.driver == lab].groupby("vehicle").dist_gt.mean() - kk[kk.driver == ctl].groupby("vehicle").dist_gt.mean()
            r["veh_up"] = int((v > 0).sum())
            r["scenes_changed"] = int((np.abs(p.ds) > 1e-6).sum() + 0)
            r["plan_identical"] = bool((np.abs(p.a - p.b) < 1e-9).all())
            r["reproduces_ab"] = bool(-0.20 <= r["dscore"] <= -0.06 and 1.4 <= r["gt_ratio"] <= 2.4)
            r["uniform_c"] = r["veh_up"] >= 12
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["arm", "n", "score", "dscore", "dscore_lo", "dscore_hi", "gt", "gt_ratio", "gt_ratio_lo", "gt_ratio_hi", "ge4", "events", "zeros",
            "d4", "d7", "d10", "slope", "veh_up", "reproduces_ab"]
    print(T[[c for c in cols if c in T]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
