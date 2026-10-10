"""Lane SH30S was paused by the user before any training, 2026-10-10: no pre-registration was written, no arm was trained, this reader was
never run. What exists: the first-look tables of the stored checkpoints in results/sh30s/ (SH30-F against P2H10-F and P2H10S-F, validation logs).

BODY1 SH30S (planned pre-registration plans/2026-10-10-sh30s-prereg.md, not written): navtest pilot read of the body-lesson ingredients on the SH30 recipe. CPU, stored bench units only.

  nav --name pilot --new SH30S-P-s0 SH30A-P-s0 --ref SH30-P-s0 --gate
        every new tag against the reference on the 12 146 navtest tokens (paired, log-cluster bootstrap): EPDMS / EP / DAC failure % on
        all / S5 (< 5 deg) / T20 (> 20 deg) / T45 (> 45 deg; |logged 4 s heading change|, strong_hinge.py's strata), the trainers' dev ADE.
        --gate: an arm is eligible iff EPDMS diff >= --min-gain and S5 EPDMS diff >= --min-straight and dev ADE <= ref + --max-dade (point
        estimates); selected = the eligible arm with the largest EPDMS diff. Exit 2 = no eligible arm (stop).
  nav --name calib --new P2H10S-P-s0 --ref P2H10-P-s0      the same table without a gate (what the pilot scale shows of a known full-scale gain)
-> experiments/body1/results/sh30s/<name>.{json,md}

  .venv/bin/python experiments/body1/scripts/sh30s_gate.py nav --name pilot --new SH30S-P-s0 SH30A-P-s0 --ref SH30-P-s0 --gate
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

OUT = _R / "experiments/body1/results/sh30s"


def dev_ade(tag):
    """Last dev ADE (m) logged by the trainer of `tag`."""
    from jevdrive.common import data_dir
    d = sorted((data_dir() / "runs/op_parity" / f"train-{tag}").iterdir())[-1]
    v = [json.loads(ln) for ln in open(d / "events.jsonl")]
    v = [e["value"] for e in v if e.get("kind") == "scalar" and e.get("tag") == "dev/ade"]
    return float(v[-1]) if v else float("nan")


def cmd_nav(a):
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    from jevdrive.common import data_dir
    tab = np.load(data_dir() / "runs/op_parity/cache/lb_navtest/tab.npz")
    tok, log = tab["names"], tab["log"]
    psi = np.abs(np.degrees(np.arctan2(np.sin(tab["fut"][:, 7, 2]), np.cos(tab["fut"][:, 7, 2]))))
    U = {t: BT.load("navtest", t)[0].reindex(tok) for t in a.new + [a.ref]}
    for t, u in U.items():
        assert not u.score.isna().any(), f"missing navtest tokens of {t}"
    strata = (("all", np.ones(len(tok), bool)), ("S5", psi < 5), ("T20", psi > 20), ("T45", psi > 45))
    res = dict(ref=a.ref, dev_ade={t: dev_ade(t) for t in a.new + [a.ref]}, arms={})
    for t in a.new:
        r = {}
        for name, m in strata:
            for col in ("score", "EP", "DAC"):
                f = (lambda u: (u[col].to_numpy()[m] < 1) * 100.0) if col == "DAC" else (lambda u: u[col].to_numpy()[m] * 100)
                p = stats.paired(f(U[t]), f(U[a.ref]), groups=log[m], n_boot=4000)
                r[f"{'DAC fail %' if col == 'DAC' else 'EPDMS' if col == 'score' else col} | {name}"] = dict(
                    new=p["mean_a"], ref=p["mean_b"], diff=p["mean"], lo=p["lo"], hi=p["hi"], n=int(m.sum()))
        res["arms"][t] = r
    rc = 0
    if a.gate:
        el = {}
        for t in a.new:
            g, s5, da = res["arms"][t]["EPDMS | all"]["diff"], res["arms"][t]["EPDMS | S5"]["diff"], res["dev_ade"][t] - res["dev_ade"][a.ref]
            el[t] = dict(epdms_gain=g, straight_diff=s5, dev_ade_diff=da, eligible=bool(g >= a.min_gain and s5 >= a.min_straight and da <= a.max_dade))
        ok = [t for t in a.new if el[t]["eligible"]]
        sel = max(ok, key=lambda t: el[t]["epdms_gain"]) if ok else ""
        res["gate"] = dict(rule=f"eligible iff EPDMS diff >= {a.min_gain:+.2f} and S5 EPDMS diff >= {a.min_straight:+.2f} and dev ADE <= ref + {a.max_dade} m; "
                                "selected = eligible arm with the largest EPDMS diff", arms=el, selected=sel, stop=not ok)
        rc = 0 if ok else 2
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    keys = list(next(iter(res["arms"].values())))
    L = [f"navtest, 12 146 tokens, new - {a.ref}, paired log-cluster bootstrap (B 4000); values x 100\n",
         "| read-out | n | " + a.ref + " | " + " | ".join(a.new) + " |", "|:--|--:|:--|" + ":--|" * len(a.new)]
    for k in keys:
        c = [res["arms"][t][k] for t in a.new]
        L.append(f"| {k} | {c[0]['n']} | {c[0]['ref']:.2f} | " + " | ".join(f"{x['new']:.2f} ({x['diff']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}])" for x in c) + " |")
    L.append(f"| dev ADE (m) | | {res['dev_ade'][a.ref]:.4f} | " + " | ".join(f"{res['dev_ade'][t]:.4f}" for t in a.new) + " |")
    if a.gate:
        L.append("\n```json\n" + json.dumps(res["gate"], indent=1, default=float) + "\n```")
    (OUT / f"{a.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    raise SystemExit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("nav")
    p.add_argument("--name", required=True)
    p.add_argument("--new", nargs="+", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--min-gain", type=float, default=0.15)
    p.add_argument("--min-straight", type=float, default=-0.20)
    p.add_argument("--max-dade", type=float, default=0.01)
    a = ap.parse_args()
    {"nav": cmd_nav}[a.cmd](a)
