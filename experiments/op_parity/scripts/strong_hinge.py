"""op_parity strong-hinge (plans/2026-10-08-strong-hinge-prereg.md): pilot gate.

  gate --new SHP-F-s0 --ref RH0-F-s0   navtest paired (log-cluster bootstrap) EPDMS all / straight; exit 2 = stop (EPDMS gain < +0.3 or straight EPDMS < -0.2)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/strong_hinge"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"


def cmd_gate(a):
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    tab = np.load(TAB)
    tok, log = tab["names"], tab["log"]
    psi = np.abs(np.degrees(np.arctan2(np.sin(tab["fut"][:, 7, 2]), np.cos(tab["fut"][:, 7, 2]))))
    U = {k: BT.load("navtest", s)[0].reindex(tok) for k, s in (("new", a.new), ("ref", a.ref))}
    assert not U["new"].score.isna().any() and not U["ref"].score.isna().any(), "missing navtest tokens"
    res = {}
    for name, m in (("all", np.ones(len(tok), bool)), ("S5", psi < 5), ("T20", psi > 20), ("T45", psi > 45)):
        for col in ("score", "EP", "DAC"):
            x = U["new"][col].to_numpy()[m] * 100
            y = U["ref"][col].to_numpy()[m] * 100
            if col == "DAC":
                x, y = (U["new"][col].to_numpy()[m] < 1) * 100.0, (U["ref"][col].to_numpy()[m] < 1) * 100.0
            r = stats.paired(x, y, groups=log[m], n_boot=4000)
            res[f"{col} | {name}"] = dict(new=r["mean_a"], ref=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"], n=int(m.sum()))
    g, s5 = res["score | all"], res["score | S5"]
    stop = bool(g["diff"] < 0.3 or s5["diff"] < -0.2)
    res["gate"] = dict(rule="stop iff EPDMS gain < +0.3 or straight (S5) EPDMS diff < -0.2", epdms_gain=g["diff"], straight_diff=s5["diff"], stop=stop,
                       new=a.new, ref=a.ref)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(2 if stop else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--new", default="SHP-F-s0")
    p.add_argument("--ref", default="RH0-F-s0")
    a = ap.parse_args()
    {"gate": cmd_gate}[a.cmd](a)
