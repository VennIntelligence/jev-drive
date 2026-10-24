#!/usr/bin/env python3
"""ARENA-V2 summary (plans/2026-10-25-arena-v2-check.md): do the arena held re-reads of the private-board anchors, scored with the old and
the fixed (1b024f0) corridor scorer, order the anchors as the old / new board does? Box, repo root, .venv python (pandas, numpy).

  arena_v2.py sum [--out DIR]    $X/v2/<tag>/{old,new}/rollouts.csv (arena_v2.sh v2) -> DIR/{summary.md, summary.json, rows_old.csv.gz,
                                 rows_new.csv.gz (arena.py collect columns tag / id / score / type, for pcs_local.py score --subset)}
Valid windows: every tag has a score and no tag route-failed there (prereg). CIs: 95 %, bootstrap over logs (2 000).
"""
import argparse
import glob
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/alpasim_proxy/arena"
X = W / "v2read"
REPO = Path(__file__).resolve().parents[3]
RES = REPO / "experiments/alpasim_proxy/results"
J = DATA / "runs/defect_fix/jt/arena"
# tag -> (label, board submission id prefix, stored arena read glob (old scorer, ASLs pruned), pcs_local tag)
TAGS = {"G0014": ("garage", "1a7f9e1a", W / "reads/G0014", "G0014"),
        "starter": ("go-straight", "60321101", W / "reads/starter", "starter"),
        "YR10m10-F-s0": ("YR", "cdf145f9", W / "reads/YR10m10-F-s0", "YR10m10-F-s0"),
        "DD-sample": ("DD", "85d42611", W / "reads/DD-sample", "DD-sample"),
        "LTF-sample": ("LTF", "f6b127cc", W / "reads/LTF-sample", "LTF-sample"),
        "APY10m10-AB-s0": ("APY", "94d2e447", W / "reads/APY10m10-AB-s0", "APY10m10-AB-s0"),
        "JTN-L18-s1-native": ("JTN-s1", "91f5b902", J / "JTN-L18-s1@off", "JTN-L18-s1@off"),
        "JTN-L18-s1-stop3": ("JTN-s1+stop3", "a3836ef6", J / "JTN-L18-s1@stop3", "JTN-L18-s1@stop3")}
ANCH6 = ["G0014", "starter", "YR10m10-F-s0", "DD-sample", "LTF-sample", "APY10m10-AB-s0"]
BOARD = {"old": "leaderboard_nuplan_20261019T1349Z.json", "new": "leaderboard_nuplan_20261025.json"}


def board(v):
    L = json.loads((RES / BOARD[v]).read_text())["leaderboard"]
    by = {r["submission_id"][:8]: r for r in L}
    return {t: (by[b]["average_scene_score"], by[b]["policy_capability_score"]) for t, (_, b, _, _) in TAGS.items()}


def tau_b(x, y):
    c = d = tx = ty = 0
    for i, j in combinations(range(len(x)), 2):
        a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if a == 0 and b == 0:
            continue
        if a == 0:
            tx += 1
        elif b == 0:
            ty += 1
        elif a == b:
            c += 1
        else:
            d += 1
    return (c - d) / np.sqrt((c + d + tx) * (c + d + ty))


def boot(diff, logs, n=2000, seed=0):
    """mean of diff with a log-clustered bootstrap CI."""
    g = pd.DataFrame(dict(d=diff, l=logs)).groupby("l").d.agg(["sum", "count"])
    s, k = g["sum"].to_numpy(), g["count"].to_numpy()
    r = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    m = s[r].sum(1) / k[r].sum(1)
    return float(diff.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def stored(path):
    rows = []
    for f in glob.glob(str(path / "c*/run/aggregate/results-summary.json")):
        for r in json.loads(Path(f).read_text())["rollouts"]:
            rows.append(dict(clip=r["clipgt_id"], score=float(r.get("score") or 0.0)))
    return pd.DataFrame(rows).drop_duplicates("clip").set_index("clip").score if rows else pd.Series(dtype=float)


def cmd_sum(a):
    out = Path(a.out or X / "report"); out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(RES / "arena/set.csv.gz").set_index("id")
    D, have = {}, []
    for t in TAGS:
        for v in ("old", "new"):
            f = X / "v2" / t / v / "rollouts.csv"
            if f.exists():
                D[t, v] = pd.read_csv(f).drop_duplicates("clip").set_index("clip")
        if (t, "new") in D:
            have.append(t)
    bad = set().union(*(set(d.index[d.failure_reason.fillna("").str.startswith("Waypoint")]) for d in D.values()))
    ids = sorted(set.intersection(*(set(D[t, "new"].index) for t in have)) - bad)
    logs = pd.Series(ids).str.rsplit("-", n=1).str[0].to_numpy()
    strat = S.stratum.reindex(ids).to_numpy()
    B = {v: board(v) for v in BOARD}
    sc = {(t, v): D[t, v].score.reindex(ids).fillna(0.0).to_numpy(float) for t in have for v in ("old", "new") if (t, v) in D}
    rec = dict(windows=len(ids), route_fail_dropped=len(bad), chunks=sorted(D[have[0], "new"].chunk.unique().tolist()), tags={})
    L = [f"# ARENA-V2 readout ({len(ids)} valid held windows, chunks {', '.join(rec['chunks'])}; {len(bad)} route-fail dropped)", ""]
    L += ["| subject | old scorer mean | new scorer mean | new - old | zeros old / new | windows up / down | old reproduces stored | "
          "stored read (d355 / jt) mean, same windows | re-read - stored [CI] | same score share | board old | board new |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t in have:
        lab = TAGS[t][0]; o = sc.get((t, "old")); n = sc[t, "new"]
        rep = (X / "v2" / t / "old_rc.txt").read_text().strip() if (X / "v2" / t / "old_rc.txt").exists() else "?"
        st = stored(TAGS[t][2]).reindex(ids)
        m = st.notna().to_numpy()
        rr = boot(o[m] - st.to_numpy()[m], logs[m]) if o is not None and m.sum() else (np.nan,) * 3
        same = float((np.abs(o[m] - st.to_numpy()[m]) < 1e-6).mean()) if o is not None and m.sum() else np.nan
        r = dict(old=float(o.mean()), new=float(n.mean()), zeros_old=int((o <= 0).sum()), zeros_new=int((n <= 0).sum()),
                 up=int((n > o + 1e-9).sum()), down=int((n < o - 1e-9).sum()), old_rc=rep, stored_n=int(m.sum()),
                 stored_mean=float(st[m].mean()) if m.sum() else None, reread_minus_stored=rr, same_share=same,
                 board_old=B["old"][t], board_new=B["new"][t],
                 by_stratum={s: dict(n=int((strat == s).sum()), old=float(o[strat == s].mean()), new=float(n[strat == s].mean()))
                             for s in ("static", "launch", "braking", "turn", "cruise", "other") if (strat == s).any()})
        rec["tags"][t] = r
        L.append(f"| {lab} | {r['old']:.4f} | {r['new']:.4f} | {r['new'] - r['old']:+.4f} | {r['zeros_old']} / {r['zeros_new']} | {r['up']} / {r['down']} | "
                 f"{rep} | {r['stored_mean'] if r['stored_mean'] is None else round(r['stored_mean'], 4)} ({m.sum()}) | "
                 f"{rr[0]:+.4f} [{rr[1]:+.4f}, {rr[2]:+.4f}] | {same:.3f} | {B['old'][t][0]:.4f} | {B['new'][t][0]:.4f} |")
    L += ["", "Order against the board (mean scene score). tau-b; slope / intercept / r of the arena mean on the board mean.", "",
          "| scorer vs board | set | tau-b | slope | intercept | r | arena order |", "|---|---|--:|--:|--:|--:|---|"]
    rec["order"] = {}
    for v, bv in (("old", "old"), ("new", "new"), ("old", "new"), ("new", "old")):
        for nm, ts in (("6 anchors", [t for t in ANCH6 if t in have]), ("8 subjects", have)):
            if any((t, v) not in sc for t in ts):
                continue
            x = np.array([sc[t, v].mean() for t in ts]); y = np.array([B[bv][t][0] for t in ts])
            k, b0 = np.polyfit(y, x, 1); rr_ = np.corrcoef(x, y)[0, 1]; tb = tau_b(x, y)
            order = " > ".join(TAGS[t][0] for t in sorted(ts, key=lambda t: -sc[t, v].mean()))
            rec["order"][f"{v}-scorer/{bv}-board/{nm}"] = dict(tau_b=tb, slope=k, intercept=b0, r=rr_, order=order)
            L.append(f"| {v} scorer vs {bv} board | {nm} | {tb:+.3f} | {k:.2f} | {b0:+.3f} | {rr_:.3f} | {order} |")
    L += ["", "Board order (new): " + " > ".join(TAGS[t][0] for t in sorted(have, key=lambda t: -B["new"][t][0])),
          "Board order (old): " + " > ".join(TAGS[t][0] for t in sorted(have, key=lambda t: -B["old"][t][0])), "",
          "Pairs (arena paired difference, A - B, [CI]; board difference new / old):", "",
          "| A - B | old scorer | new scorer | board old | board new |", "|---|--:|--:|--:|--:|"]
    rec["pairs"] = {}
    for a_, b_ in combinations(sorted(have, key=lambda t: -B["new"][t][0]), 2):
        po = boot(sc[a_, "old"] - sc[b_, "old"], logs); pn = boot(sc[a_, "new"] - sc[b_, "new"], logs)
        rec["pairs"][f"{a_}-{b_}"] = dict(old=po, new=pn)
        L.append(f"| {TAGS[a_][0]} - {TAGS[b_][0]} | {po[0]:+.4f} [{po[1]:+.4f}, {po[2]:+.4f}] | {pn[0]:+.4f} [{pn[1]:+.4f}, {pn[2]:+.4f}] | "
                 f"{B['old'][a_][0] - B['old'][b_][0]:+.4f} | {B['new'][a_][0] - B['new'][b_][0]:+.4f} |")
    L += ["", "New - old by stratum (windows; mean old -> new):", "", "| subject | " + " | ".join(
        f"{s} ({int((strat == s).sum())})" for s in ("static", "launch", "braking", "turn", "cruise", "other")) + " |", "|---|" + "--:|" * 6]
    for t in have:
        bs = rec["tags"][t]["by_stratum"]
        L.append(f"| {TAGS[t][0]} | " + " | ".join(f"{bs[s]['old']:.3f} -> {bs[s]['new']:.3f}" if s in bs else "-" for s in
                                                   ("static", "launch", "braking", "turn", "cruise", "other")) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(rec, indent=1, default=float))
    for v in ("old", "new"):
        rows = [dict(tag=TAGS[t][3], id=i, score=s_, type="zero" if s_ <= 0 else "ok") for t in have if (t, v) in sc for i, s_ in zip(ids, sc[t, v])]
        pd.DataFrame(rows).to_csv(out / f"rows_{v}.csv.gz", index=False)
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("sum"); q.add_argument("--out", default="")
    a = p.parse_args()
    dict(sum=cmd_sum)[a.cmd](a)


if __name__ == "__main__":
    main()
