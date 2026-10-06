"""View contrasts stratified by logged heading change (the opj binning of the joint diagnosis), from stored per-token outputs only (CPU, no model runs).

  1 P3 - P2            op_parity full run, seed means      (side + rear cameras added)
  2 V - F              unfreeze pilot, seed means          (1.40 m virtual camera vs frozen W frames; V/F also differ in the warp road height)
  3 P0 vh140 - P0 W    shipped model, zero-shot camera height change (decision 145)
  4 P3 side masked - P3  side cameras masked at test
  5 decoders (op_probe, 2 152 scored tokens, weighted back to navtest): V = Cinque vision, Cf = WA front-only, Ca = WA all-view encoder; Cf - V, Ca - Cf
  6 geometry proxy: share of the logged 4 s path outside a camera half-FOV (bearing from the front axle)

  $DATA_DIR/envs/op-train/bin/python experiments/op_probe/scripts/opj_view_by_turn.py [--out DIR]   (box)  -> view_by_turn.{md,csv}; the md is copied to results/ on the Mac
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
EV = D / "runs/navsim/eval"
B, SEED = 4000, 0
EDGES = [0, 5, 20, 45, 400]


def cboot(V, g, w=None, B=B, seed=SEED):
    """Cluster bootstrap (resample logs) of weighted column means. V (n, k) -> (k, 3) [mean, lo, hi]."""
    V = np.asarray(V, float)
    w = np.ones(len(V)) if w is None else np.asarray(w, float)
    ok = np.isfinite(V)
    codes, uniq = pd.factorize(pd.Series(np.asarray(g)))
    nu = len(uniq)
    S = np.stack([np.bincount(codes, np.where(ok[:, j], V[:, j] * w, 0), nu) for j in range(V.shape[1])], 1)
    N = np.stack([np.bincount(codes, ok[:, j] * w, nu) for j in range(V.shape[1])], 1)
    idx = np.random.default_rng(seed).integers(nu, size=(B, nu))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S[idx].sum(1) / N[idx].sum(1)
        full = S.sum(0) / N.sum(0)
    return np.stack([full, np.nanquantile(r, 0.025, 0), np.nanquantile(r, 0.975, 0)], 1)


def load_arm(frames, m):
    fs = sorted(EV.glob(f"v2_navtest_opi_lb_navtest_{frames}-cinque_PP{m}__base/*/*.csv"))
    if not fs:
        raise FileNotFoundError(f"{frames} {m}")
    df = pd.read_csv(fs[-1])
    t = df[~df.token.astype(str).str.startswith(("average", "extended_pdm_score"))].set_index("token")
    return t


def arm(frames, ms, d):
    ts = [load_arm(frames, m).reindex(d.index) for m in ms]
    sc = np.mean([t.score.values for t in ts], 0)
    fd = np.mean([(t.drivable_area_compliance.values < 1 - 1e-9).astype(float) for t in ts], 0)
    return 100 * sc, 100 * fd


def strata(d):
    turn = d.dyaw.abs()
    out = [("all", "all tokens", np.ones(len(d), bool))]
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        lab = f"{lo}-{hi} deg" if hi < 400 else f"> {lo} deg"
        m = ((turn >= lo) & (turn < hi)).values
        out.append(("|heading change|", lab, m))
    for lo, hi in zip(EDGES[1:-1], EDGES[2:]):
        lab = f"{lo}-{hi} deg" if hi < 400 else f"> {lo} deg"
        m = ((turn >= lo) & (turn < hi)).values
        out.append(("left", lab, m & (d.dyaw > 0).values))
        out.append(("right", lab, m & (d.dyaw < 0).values))
    return out


f = lambda m, k=2: f"{m[0]:+.{k}f} [{m[1]:+.{k}f}, {m[2]:+.{k}f}]"  # noqa: E731
f0 = lambda m, k=2: f"{m[0]:.{k}f}"  # noqa: E731


def navtest_table(d, A, B_, name):
    """A, B_ = (score, failDAC) arrays x100; contrast A - B_ per stratum."""
    rows = []
    for grp, lab, m in strata(d):
        s = d[m]
        g = s.log.values
        r = cboot(np.stack([A[0][m], B_[0][m], A[0][m] - B_[0][m], A[1][m], B_[1][m], A[1][m] - B_[1][m]], 1), g)
        rows.append({"contrast": name, "group": grp, "stratum": lab, "n": int(m.sum()), "logs": int(s.log.nunique()),
                     "EPDMS A": f0(r[0]), "EPDMS B": f0(r[1]), "dEPDMS A-B": f(r[2]), "DACfail% A": f0(r[3]), "DACfail% B": f0(r[4]), "dDACfail pp A-B": f(r[5])})
    return rows


def decoder_table(d, dec, obj):
    q = dec[dec.obj == obj]
    st = {s: q[q.stage == s].set_index("token") for s in ("E", "P2-V", "WA-Cf", "WA-Ca")}
    toks = st["E"].index
    dd = d.set_index("token").loc[toks]
    # navtest weights: F / R / FF tokens weight 1, the 1 500 random both-pass tokens carry n_PP / 1500
    w = np.where(dd.pp1500.values, d.set_PP.sum() / d.pp1500.sum(), 1.0)
    P = {s: (st[s].loc[toks, "DAC"].values < 1 - 1e-9).astype(float) * 100 for s in st}
    S = {s: st[s].loc[toks, "score"].values * 100 for s in st}
    rows = []
    for grp, lab, m in strata(dd.reset_index()):
        g = dd.log.values[m]
        V = np.stack([P["E"][m], P["P2-V"][m], P["WA-Cf"][m], P["WA-Ca"][m], P["WA-Cf"][m] - P["P2-V"][m], P["WA-Ca"][m] - P["WA-Cf"][m],
                      S["P2-V"][m], S["WA-Cf"][m], S["WA-Ca"][m], S["WA-Cf"][m] - S["P2-V"][m], S["WA-Ca"][m] - S["WA-Cf"][m]], 1)
        r = cboot(V, g, w[m])
        rows.append({"decoder objective": obj, "group": grp, "stratum": lab, "n scored": int(m.sum()), "n navtest-equivalent": int(round(w[m].sum())),
                     "DACfail% ego-only": f0(r[0]), "DACfail% V": f0(r[1]), "DACfail% Cf": f0(r[2]), "DACfail% Ca": f0(r[3]),
                     "dDACfail Cf-V pp": f(r[4]), "dDACfail Ca-Cf pp": f(r[5]),
                     "score V": f0(r[6]), "score Cf": f0(r[7]), "score Ca": f0(r[8]), "dscore Cf-V": f(r[9]), "dscore Ca-Cf": f(r[10])})
    return rows


def fov_table(d, fut):
    """Share of the logged 4 s path (8 poses at 0.5 s) whose bearing from the front axle exceeds a half-FOV; per token mean, then bin mean."""
    b = np.degrees(np.abs(np.arctan2(fut[:, :, 1], fut[:, :, 0] - 1.5)))      # camera ~ 1.5 m ahead of the rear axle (t0 frame x forward)
    rows = []
    for grp, lab, m in strata(d):
        if grp == "|heading change|" or grp == "all":
            r = {"stratum": lab, "n": int(m.sum())}
            for h in (30, 45, 60, 75):
                r[f"path share outside +-{h} deg"] = f"{100 * (b[m] > h).mean():.1f}%"
                r[f"tokens with end pose outside +-{h}"] = f"{100 * (b[m][:, -1] > h).mean():.1f}%"
            rows.append(r)
    return rows


def md(rows):
    df = pd.DataFrame(rows)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D / "runs/op_probe/view_by_turn"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(D / "runs/op_probe/joint/navtest_tokens.parquet").set_index("token")
    d0 = d.copy()
    arms = {"P2": arm("warp", ["P2-F-s0", "P2-F-s1"], d), "P3": arm("warp", ["P3-F-s0", "P3-F-s1"], d),
            "P3 nosd": arm("warp", ["P3-F-s0_noside", "P3-F-s1_noside"], d), "F": arm("warp", ["UF-F-s0", "UF-F-s1"], d),
            "V": arm("vh140", ["UF-V-s0", "UF-V-s1"], d), "P0 W": arm("warp", ["P0"], d), "P0 vh140": arm("vh140", ["P0"], d)}
    for k, (s, _) in arms.items():
        print(k, np.nanmean(s), np.isnan(s).sum())
    dn = d.reset_index()
    CON = [("1 P3 - P2 (side + rear cameras added)", "P3", "P2"), ("2 V - F (1.40 m virtual camera, pilot)", "V", "F"),
           ("3 P0 at 1.40 m - P0 at W frames (zero-shot height)", "P0 vh140", "P0 W"), ("4 P3 side masked - P3 (test-time ablation)", "P3 nosd", "P3")]
    rows = [r for n, x, y in CON for r in navtest_table(dn, arms[x], arms[y], n)]
    dec = pd.read_parquet(D / "runs/op_probe/joint/decoders.parquet")
    drows = [r for o in ("hinge10", "imit") for r in decoder_table(dn, dec, o)]
    tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
    pos = {t: i for i, t in enumerate(tab["names"].tolist())}
    fut = np.stack([tab["fut"][pos[t]] for t in d.index])
    frows = fov_table(dn, fut)
    pd.DataFrame(rows).to_csv(out / "view_by_turn_navtest.csv", index=False)
    pd.DataFrame(drows).to_csv(out / "view_by_turn_decoders.csv", index=False)
    pd.DataFrame(frows).to_csv(out / "view_by_turn_fov.csv", index=False)
    (out / "tables.md").write_text("\n\n".join([f"### {n}\n\n" + md([r for r in rows if r['contrast'] == n]) for n, _, _ in CON]
                                               + ["### decoders\n\n" + md(drows), "### fov\n\n" + md(frows)]) + "\n")
    print((out / "tables.md").read_text())
