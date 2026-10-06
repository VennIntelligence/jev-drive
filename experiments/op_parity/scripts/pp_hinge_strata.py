"""Hinge lane, stratified readout (same binning as experiments/op_probe/scripts/opj_figs.py / opj_build.py, joint diagnosis):
navtest by logged manoeuvre (straight / curve 8-20 / left > 20 / right > 20) and by |heading change| bins (0-5-20-45-400 deg), EPDMS and DAC fail %,
P2 vs hinge vs WA-JEPA with cluster-bootstrap contrasts; HUGSIM HD on straight (< 30 deg) vs turning (>= 30 deg) routes (route turn = range of the
unwrapped route yaw, runs/op_parity/hugsim/routes.json). Run on the box after the chain (needs results/hugsim_hinge/extract.csv):
  $DATA_DIR/envs/op-train/bin/python experiments/op_parity/scripts/pp_hinge_strata.py --hinge P2H10  -> results/hinge_strata.{md,csv}
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

B, SEED = 2000, 0
TURN_BINS = [0, 5, 20, 45, 400]
MAN = (("straight", "straight"), ("curve", "curve 8-20"), ("left turn", "left turn > 20"), ("right turn", "right turn > 20"))


def cboot(V, g, B=B, seed=SEED):   # identical to opj_figs.cboot
    V = np.asarray(V, float)
    V = V[:, None] if V.ndim == 1 else V
    ok = np.isfinite(V)
    codes, uniq = pd.factorize(pd.Series(np.asarray(g)))
    nu = len(uniq)
    S = np.stack([np.bincount(codes, np.where(ok[:, j], V[:, j], 0), nu) for j in range(V.shape[1])], 1)
    N = np.stack([np.bincount(codes, ok[:, j].astype(float), nu) for j in range(V.shape[1])], 1)
    idx = np.random.default_rng(seed).integers(nu, size=(B, nu))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S[idx].sum(1) / N[idx].sum(1)
        full = S.sum(0) / N.sum(0)
    return np.stack([full, np.nanquantile(r, 0.025, 0), np.nanquantile(r, 0.975, 0)], 1)


f = lambda m: f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]"  # noqa: E731
f0 = lambda m: f"{m[0]:.2f}"  # noqa: E731


def navtest(H):
    import pp_eval as E
    E.DATA, E.FRAMES = "lb_navtest", "warp"
    d = pd.read_parquet(data_dir() / "runs/op_probe/joint/navtest_tokens.parquet").set_index("token")
    sc = {m: E.read_csv(E.eval_csv(f"{H}-F-s{s}"))[0] for s, m in ((0, "H0"), (1, "H1"))}
    d["H_score"] = (sc["H0"].score.reindex(d.index) + sc["H1"].score.reindex(d.index)) / 2
    d["H_fDAC"] = sum((sc[k].drivable_area_compliance.reindex(d.index) < 1 - 1e-9).astype(float) for k in sc) / 2
    d["P2_fDAC"] = (d.P2s0_DAC < 1 - 1e-9).astype(float) / 2 + (d.P2s1_DAC < 1 - 1e-9).astype(float) / 2
    d["WA_fDAC"] = (d.WA_DAC < 1 - 1e-9).astype(float)
    d["P2_score"] = (d.P2s0_score + d.P2s1_score) / 2
    d = d[d.H_score.notna()].copy()
    d["turn"] = d.dyaw.abs()
    strata = [("all", "all tokens", d.index == d.index)]
    strata += [("manoeuvre", lab, (d.maneuver == k).values) for k, lab in MAN]
    strata += [("|heading change|", f"{lo}-{hi} deg" if hi < 400 else f"> {lo} deg", ((d.turn >= lo) & (d.turn < hi)).values)
               for lo, hi in zip(TURN_BINS[:-1], TURN_BINS[1:])]
    rows = []
    for grp, lab, m in strata:
        s = d[m]
        g = s.log.values
        V = np.stack([100 * s.P2_score, 100 * s.H_score, 100 * s.WA_score, 100 * (s.H_score - s.P2_score), 100 * (s.H_score - s.WA_score),
                      100 * s.P2_fDAC, 100 * s.H_fDAC, 100 * s.WA_fDAC, 100 * (s.H_fDAC - s.P2_fDAC), 100 * (s.H_fDAC - s.WA_fDAC)], 1)
        r = cboot(V, g)
        rows.append({"group": grp, "stratum": lab, "n": len(s), "EPDMS P2": f0(r[0]), "EPDMS hinge": f0(r[1]), "EPDMS WA": f0(r[2]),
                     "hinge - P2": f(r[3]), "hinge - WA": f(r[4]), "DAC fail % P2": f0(r[5]), "DAC fail % hinge": f0(r[6]), "DAC fail % WA": f0(r[7]),
                     "dDACfail hinge - P2 (pp)": f(r[8]), "dDACfail hinge - WA (pp)": f(r[9])})
    return pd.DataFrame(rows)


def hugsim(H):
    out = []
    routes = json.load(open(data_dir() / "runs/op_parity/hugsim/routes.json"))
    turn = {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in routes.items()}
    HR = _R / "experiments/hugsim/results"
    W = pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    ex = pd.read_csv(_R / "experiments/op_parity/results/hugsim_hinge/extract.csv")
    for preset in ("exam", "spec"):
        hd = lambda arms: pd.concat([ex[(ex.preset == preset) & (ex.arm == a)].set_index("scenario").hdscore for a in arms], axis=1).mean(1, skipna=False).reindex(W.index)  # noqa: E731
        df = pd.DataFrame({"P2": hd(["P2-F-s0", "P2-F-s1"]), "H": hd([f"{H}-F-s0", f"{H}-F-s1"]), "WA": W.hdscore, "turn": W.scene.map(turn)}).dropna()
        for lab, m in (("all 64", df.turn >= 0), ("straight route (< 30 deg)", df.turn < 30), ("turning route (>= 30 deg)", df.turn >= 30)):
            s = df[m]
            r = cboot(np.stack([s.P2, s.H, s.WA, s.H - s.P2, s.H - s.WA, s.P2 - s.WA], 1), s.index.values)
            out.append({"preset": preset, "stratum": lab, "n": len(s), "HD P2": f0(r[0]), "HD hinge": f0(r[1]), "HD WA": f0(r[2]),
                        "hinge - P2": f(r[3]), "hinge - WA": f(r[4]), "P2 - WA": f(r[5])})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hinge", default="P2H10")
    a = ap.parse_args()
    nt, hg = navtest(a.hinge), hugsim(a.hinge)
    out = _R / "experiments/op_parity/results"
    nt.to_csv(out / "hinge_strata_navtest.csv", index=False), hg.to_csv(out / "hinge_strata_hugsim.csv", index=False)
    md = lambda df: "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])  # noqa: E731
    (out / "hinge_strata.md").write_text(
        "# Hinge lane, stratified (opj binning)\n\nnavtest (EPDMS x 100, DAC fail % of tokens, seed means, 95% cluster bootstrap over logs, B 2000; bins as in op_probe/results/joint):\n\n"
        + md(nt) + "\n\nHUGSIM 64 HD x 1 (seed means, bootstrap over scenarios, B 2000; turning route = unwrapped route yaw range >= 30 deg):\n\n" + md(hg) + "\n")
    print(md(nt)), print(md(hg))
