"""Turn-training lane (plans/2026-10-06-turn-train-prereg.md): pilot-scale arms H / T1 / T2 / T3 (tags HP / T1P / T2P / T3P, -F-s0 / -F-s1), navtest
stratified by the op_probe joint bins, paired cluster bootstrap over logs, vs H and vs WA-JEPA.

  gate    [--arms T1P T2P T3P]   per-arm overall guard (EPDMS vs H >= -0.3) and the closure of the > 20 deg EPDMS gap to WA-JEPA -> $DATA_DIR/runs/op_parity/turn/gate.json
  report  [--arms ...]           results/turn-train_navtest.{csv,md}; with results/hugsim_turn/extract.csv also the HUGSIM turning-route table
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402
from pp_hinge_strata import MAN, TURN_BINS, cboot, f, f0  # noqa: E402

OUT = _R / "experiments/op_parity/results"
RUN = data_dir() / "runs/op_parity/turn"
GUARD, CLOSE = -0.3, 0.5
NAMES = {"HP": "H", "T1P": "T1", "T2P": "T2", "T3P": "T3", "UF-F": "F"}


def load(arms):
    import pp_eval as E
    E.DATA, E.FRAMES = "lb_navtest", "warp"
    d = pd.read_parquet(data_dir() / "runs/op_probe/joint/navtest_tokens.parquet").set_index("token")
    d["turn"] = d.dyaw.abs()
    for a in arms:                                           # seed-mean EPDMS and DAC-fail indicator per arm
        tags = [f"{a}-s{s}" if a == "UF-F" else f"{a}-F-s{s}" for s in (0, 1)]
        sc = [E.read_csv(E.eval_csv(t))[0] for t in tags]
        d[f"{a}_score"] = sum(x.score.reindex(d.index) for x in sc) / 2
        d[f"{a}_fDAC"] = sum((x.drivable_area_compliance.reindex(d.index) < 1 - 1e-9).astype(float) for x in sc) / 2
    d["WA_fDAC"] = (d.WA_DAC < 1 - 1e-9).astype(float)
    return d[d[[f"{a}_score" for a in arms]].notna().all(1)].copy()


def strata(d):
    out = [("all", "all tokens", d.index == d.index)]
    out += [("manoeuvre", lab, (d.maneuver == k).values) for k, lab in MAN]
    out += [("|heading change|", f"{lo}-{hi} deg" if hi < 400 else f"> {lo} deg", ((d.turn >= lo) & (d.turn < hi)).values)
            for lo, hi in zip(TURN_BINS[:-1], TURN_BINS[1:])]
    out += [("|heading change|", "> 20 deg (pooled)", (d.turn >= 20).values)]
    return out


def table(d, arms):
    rows = []
    for grp, lab, m in strata(d):
        s = d[m]
        for a in arms:
            V = np.stack([100 * s.WA_score, 100 * s.HP_score, 100 * s[f"{a}_score"], 100 * (s[f"{a}_score"] - s.HP_score), 100 * (s[f"{a}_score"] - s.WA_score),
                          100 * s.WA_fDAC, 100 * s.HP_fDAC, 100 * s[f"{a}_fDAC"], 100 * (s[f"{a}_fDAC"] - s.HP_fDAC), 100 * (s[f"{a}_fDAC"] - s.WA_fDAC)], 1)
            r = cboot(V, s.log.values)
            clo = (r[3, 0] / (r[0, 0] - r[1, 0])) if grp != "all" and r[0, 0] - r[1, 0] > 0.5 else np.nan
            rows.append({"group": grp, "stratum": lab, "n": len(s), "arm": NAMES.get(a, a), "EPDMS WA": f0(r[0]), "EPDMS H": f0(r[1]), "EPDMS arm": f0(r[2]),
                         "arm - H": f(r[3]), "arm - WA": f(r[4]), "closure of H->WA gap": "" if np.isnan(clo) else f"{clo:+.2f}",
                         "DAC fail % WA": f0(r[5]), "DAC fail % H": f0(r[6]), "DAC fail % arm": f0(r[7]), "dDACfail arm - H (pp)": f(r[8]), "dDACfail arm - WA (pp)": f(r[9])})
    return pd.DataFrame(rows)


def cmd_gate(a):
    d = load(["HP", *a.arms])
    s = d[(d.turn >= 20).values]
    res = {}
    for t in a.arms:
        r = cboot(np.stack([100 * d[f"{t}_score"] - 100 * d.HP_score, 100 * s[f"{t}_score"].reindex(d.index) - 100 * s.HP_score.reindex(d.index)], 1), d.log.values)
        gap = 100 * (s.WA_score.mean() - s.HP_score.mean())
        gain = 100 * (s[f"{t}_score"].mean() - s.HP_score.mean())
        res[t] = {"overall_vs_H": float(r[0, 0]), "gap20_H_to_WA": float(gap), "gain20_vs_H": float(gain), "closure20": float(gain / gap),
                  "guard_ok": bool(r[0, 0] >= GUARD), "closes_half": bool(gain / gap >= CLOSE), "hugsim": bool(r[0, 0] >= GUARD and gain / gap >= CLOSE)}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def hugsim(arms):
    p = OUT / "hugsim_turn/extract.csv"
    if not p.exists():
        return None
    routes = json.load(open(data_dir() / "runs/op_parity/hugsim/routes.json"))
    turn = {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in routes.items()}
    W = pd.read_csv(_R / "experiments/hugsim/results/hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    ex = pd.read_csv(p)
    out = []
    have = [a for a in ["HP", *arms] if ((ex.arm == f"{a}-F-s0").any())]
    for preset in ("exam", "spec"):
        hd = lambda a: pd.concat([ex[(ex.preset == preset) & (ex.arm == f"{a}-F-s{s}")].set_index("scenario").hdscore for s in (0, 1)], axis=1).mean(1, skipna=False).reindex(W.index)  # noqa: E731
        df = pd.DataFrame({a: hd(a) for a in have} | {"WA": W.hdscore, "turn": W.scene.map(turn)}).dropna()
        for lab, m in (("all 64", df.turn >= 0), ("straight route (< 30 deg)", df.turn < 30), ("turning route (>= 30 deg)", df.turn >= 30)):
            s = df[m]
            for a in have:
                if a == "HP":
                    continue
                r = cboot(np.stack([s.WA, s.HP, s[a], s[a] - s.HP, s[a] - s.WA], 1), s.index.values)
                out.append({"preset": preset, "stratum": lab, "n": len(s), "arm": NAMES[a], "HD WA": f0(r[0]), "HD H": f0(r[1]), "HD arm": f0(r[2]),
                            "arm - H": f(r[3]), "arm - WA": f(r[4])})
    return pd.DataFrame(out)


def cmd_report(a):
    d = load(["HP", *a.arms, "UF-F"])
    arms = list(a.arms)
    nt = table(d, arms)
    nt.to_csv(OUT / "turn-train_navtest.csv", index=False)
    base = []
    for grp, lab, m in strata(d):
        s = d[m]
        r = cboot(np.stack([100 * s.WA_score, 100 * s["UF-F_score"], 100 * s.HP_score, 100 * (s.HP_score - s["UF-F_score"]), 100 * s.WA_fDAC, 100 * s["UF-F_fDAC"], 100 * s.HP_fDAC], 1), s.log.values)
        base.append({"group": grp, "stratum": lab, "n": len(s), "EPDMS F (P2, no hinge)": f0(r[1]), "EPDMS H": f0(r[2]), "EPDMS WA": f0(r[0]), "H - F": f(r[3]),
                     "DAC fail % F": f0(r[5]), "DAC fail % H": f0(r[6]), "DAC fail % WA": f0(r[4])})
    hg = hugsim(arms)
    txt = ("# Turn-training lane, stratified navtest (pilot scale)\n\nEPDMS x 100, DAC fail % of tokens, seed means (2 seeds), 95% cluster bootstrap over the 136 navtest logs "
           "(B 2000), bins as op_probe/results/joint. closure = (arm - H) / (WA - H) in that stratum (shown where WA - H > 0.5).\n\nReference arms:\n\n" + md(pd.DataFrame(base)) + "\n\n"
           "Arms vs H and vs WA-JEPA:\n\n" + md(nt) + "\n")
    if hg is not None:
        txt += "\nHUGSIM 64 HD x 1 (seed means, bootstrap over scenarios; turning route = unwrapped route yaw range >= 30 deg):\n\n" + md(hg) + "\n"
        hg.to_csv(OUT / "turn-train_hugsim.csv", index=False)
    (OUT / "turn-train_navtest.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "report"])
    ap.add_argument("--arms", nargs="+", default=["T1P", "T2P"])
    a = ap.parse_args()
    {"gate": cmd_gate, "report": cmd_report}[a.cmd](a)
