"""op_parity Stage B report (plans/2026-10-06-parity-prereg.md, addendum 1): train protocol {N, W, G} x arm {P1, P2} x seed {0, 1}.

Reads the devkit CSVs and pose files written by pp_eval.py (`--frames` = keys / warp / gimm on lb_navtest = the matched, deployable protocol;
lb_hq_navtestX with real frames: `real` for W / G models, `keys` (the real 2 Hz keys) for N models) and writes results/stageB*.md / .csv:
  cells     per (protocol, arm, seed) and seed mean: EPDMS + subscores, ADE vs log, plan speed ratio, on full navtest (matched)
  effects   per-token seed-mean contrasts, bootstrap over navtest logs: protocol main effects (G - N, W - N on P2 and on P1), the interaction
            (P2 - P1)_X - (P2 - P1)_G, the seed spread (s1 - s0), and per model the test-protocol effect on the 1 499 real-frame tokens
            (real-frame test - matched test, same tokens)
  decision  the pre-registered rule with its guards (speed ratio within 5 % of G, no EP / DAC trade)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import json  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

PROTO = {"N": "keys", "W": "warp", "G": "gimm"}
REAL = {"N": "keys", "W": "real", "G": "real"}
ARMS, SEEDS = ("P1", "P2"), (0, 1)
SUBS = {"EP": "ego_progress", "DAC": "drivable_area_compliance", "NC": "no_at_fault_collisions", "TTC": "time_to_collision_within_bound",
        "EC": "two_frame_extended_comfort", "LK": "lane_keeping", "DDC": "driving_direction_compliance"}


def tag(p, arm, s):
    return f"{arm}-{p}-s{s}"


def load(data, frames, model):
    import pp_eval as E
    E.DATA, E.FRAMES = data, frames
    f = E.eval_csv(model)
    if f is None:
        return None
    t, _ = E.read_csv(f)
    pf = data_dir() / "runs" / "op_lb" / data / "preds" / f"{E.stem(model).replace('@', '-')}__base.npz"
    z = np.load(pf)
    return t, dict(zip(z["tokens"].tolist(), z["poses"]))


def main():
    import pandas as pd
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    fz = np.load(Z.root("index") / "navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    log = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    M, R = {}, {}                                                       # matched navtest, real-frame subset
    for p in PROTO:
        for arm in ARMS:
            for s in SEEDS:
                M[p, arm, s] = load("lb_navtest", PROTO[p], tag(p, arm, s))
                R[p, arm, s] = load("lb_hq_navtestX", REAL[p], tag(p, arm, s))
        M[p, "P0", 0] = load("lb_navtest", PROTO[p], "P0")
    miss = [k for k, v in {**M, **{("R",) + k: v for k, v in R.items()}}.items() if v is None]
    toks = sorted(set.intersection(*[set(v[0].index) for v in M.values() if v is not None]))
    g = np.array([log[t] for t in toks])
    F = np.stack([fut[t] for t in toks])
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    Lg = plen(F)
    mv = Lg > 2.0

    def per_tok(v, col="score"):
        return 100 * v[0].loc[toks, col].to_numpy(float)

    def geo(v):
        P = np.stack([v[1][t] for t in toks])
        return np.linalg.norm(P[:, :, :2] - F[:, :, :2], axis=-1).mean(1), plen(P) / np.maximum(Lg, 1e-6)

    def mean_seeds(p, arm, col="score"):
        xs = [per_tok(M[p, arm, s], col) for s in SEEDS if M.get((p, arm, s)) is not None]
        return np.mean(xs, 0) if xs else None

    rows = []
    for (p, arm, s), v in M.items():
        if v is None:
            continue
        ade, sr = geo(v)
        r = {"protocol": p, "arm": arm, "seed": s, "EPDMS": per_tok(v).mean()} | {k: float(np.nanmean(per_tok(v, c))) for k, c in SUBS.items()}
        rows.append(r | {"ade_vs_log": ade.mean(), "speed_ratio_med": float(np.median(sr[mv]))})
    cells = pd.DataFrame(rows).sort_values(["protocol", "arm", "seed"])
    out = _R / "experiments" / "op_parity" / "results"
    stats.write_table(cells.to_dict("records"), out / "stageB_cells", floatfmt=".2f",
                      note=f"matched protocol on navtest ({len(toks)} tokens), v2 EPDMS x 100; P0 = shipped Cinque under that protocol")

    eff = []

    def add(name, a, b, groups=g):
        if a is None or b is None:
            return None
        r = stats.paired(a, b, groups=groups)
        eff.append({"contrast": name, "mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "n": r["n"], "units": r["units"]})
        return r
    for arm in ARMS:
        for p in ("N", "W"):
            add(f"{arm}: G - {p} (EPDMS)", mean_seeds("G", arm), mean_seeds(p, arm))
    for p in ("N", "W"):
        a = [mean_seeds(p, "P2"), mean_seeds(p, "P1"), mean_seeds("G", "P2"), mean_seeds("G", "P1")]
        if all(x is not None for x in a):
            add(f"interaction (P2 - P1)_{p} - (P2 - P1)_G", a[0] - a[1], a[2] - a[3])
    for p in PROTO:
        add(f"P2 - P1 under {p}", mean_seeds(p, "P2"), mean_seeds(p, "P1"))
        for arm in ARMS:
            if M.get((p, arm, 1)) is not None and M.get((p, arm, 0)) is not None:
                add(f"seed spread {arm} {p}: s1 - s0", per_tok(M[p, arm, 1]), per_tok(M[p, arm, 0]))
    # test-protocol effect: same model, real frames vs its matched protocol, on the real-frame subset
    for p in PROTO:
        for arm in ARMS:
            xs, ys, gs = [], [], None
            for s in SEEDS:
                r, m = R.get((p, arm, s)), M.get((p, arm, s))
                if r is None or m is None:
                    continue
                sub = sorted(set(r[0].index) & set(m[0].index))
                xs.append(100 * r[0].loc[sub, "score"].to_numpy(float)), ys.append(100 * m[0].loc[sub, "score"].to_numpy(float))
                gs = np.array([log[t] for t in sub])
            if xs:
                add(f"{arm} {p}: real-frame test - matched test (1 499 tokens)", np.mean(xs, 0), np.mean(ys, 0), gs)
    stats.write_table(eff, out / "stageB_effects", floatfmt=".2f", note="per-token EPDMS x 100, seed means; cluster bootstrap over navtest logs, B 10000")

    # ---- decision (P2, matched protocol)
    G = mean_seeds("G", "P2")
    srG = np.median(np.mean([geo(M["G", "P2", s])[1] for s in SEEDS if M.get(("G", "P2", s))], 0)[mv]) if G is not None else np.nan
    dec = {"missing": [str(k) for k in miss]}
    choice = "G"
    for p in ("N", "W"):
        X = mean_seeds(p, "P2")
        if X is None or G is None:
            dec[p] = "missing"
            continue
        d = stats.paired(X, G, groups=g)
        dep, ddac = (stats.paired(mean_seeds(p, "P2", c), mean_seeds("G", "P2", c), groups=g) for c in (SUBS["EP"], SUBS["DAC"]))
        sr = np.median(np.mean([geo(M[p, "P2", s])[1] for s in SEEDS if M.get((p, "P2", s))], 0)[mv])
        trade = (np.sign(dep["mean"]) != np.sign(ddac["mean"])) and (dep["lo"] > 0 or dep["hi"] < 0) and (ddac["lo"] > 0 or ddac["hi"] < 0)
        ok = (G.mean() - X.mean() <= 1.0) and abs(sr / srG - 1) <= 0.05 and not trade
        dec[p] = {"X_minus_G": d, "EP": dep, "DAC": ddac, "speed_ratio": sr, "speed_ratio_G": srG, "ep_dac_trade": bool(trade), "passes": bool(ok)}
        if ok and choice == "G":
            choice = p
    dec["choice"] = choice
    (out / "stageB_decision.json").write_text(json.dumps(dec, indent=1, default=float))
    print(cells.to_string()), print(pd.DataFrame(eff).to_string()), print("choice", choice, "missing", miss)


if __name__ == "__main__":
    main()
