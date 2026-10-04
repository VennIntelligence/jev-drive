#!/usr/bin/env python
"""Guard line `navhard`: official NAVSIM v2 two-stage EPDMS of the candidate vs shipped Cinque, plus decision 110's early-turn set.

  .venv/bin/python experiments/op_guard/scripts/line_navhard.py --candidate it_dw3-s0 --mode subset --gpu 0 --cpus 0-24

Both modes score all 5 912 navhard_two_stage tokens: the official score combines every stage-2 token with its stage-1 scene
through the devkit's scene mapping, so a token subset is not the official score (and the full board fits the subset budget).
Readouts (rule: no drop = candidate paired value >= shipped):
  official    combined EPDMS; stage 1 / stage 2 as diagnostics. CIs: paired bootstrap over the 225 scene-mapping groups of the
              per-token devkit harness (nav_harness.py: the devkit's pdm_score + aggregation, whose mean is the official number).
              `full` mode also runs the devkit's own script (navsim_zs_score.sh) and reports its number (harness - official in the
              note); `subset` mode runs only the harness on all leased cores (the official script on the other half doubles the CPU
              time: 17 min on 8 cores vs the harness 12.5 min) and uses an existing official CSV of the candidate if there is one.
  early-turn  tokens whose PDM reference heading change within 2 s is >= 5 deg (results/navhard_early_turn.csv, frozen from
              decision 110's set: stage 1 231 / 450, stage 2 66 %); mean per-token EPDMS (stage-2 tokens unweighted, as
              decision 110), stages pooled for the rule, each stage as a diagnostic; CI clustered by mapping group.
Work files: $DATA_DIR/runs/op_guard/<candidate>/nav/navhard/ (mode-independent). --force redoes this candidate only.
"""
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guardlib as G  # noqa: E402
import navlib as N  # noqa: E402

LINE = "navhard"
BOARD = "navhard"


def split_cpus(cpus: str):
    """Two disjoint halves of the cpu list (official run || per-token harness)."""
    if not cpus:
        from jevdrive.common import n_cpus
        n = n_cpus()
        return "", "", max(1, n // 2), max(1, n - n // 2)
    ids = []
    for p in cpus.split(","):
        lo, _, hi = p.partition("-")
        ids += list(range(int(lo), int(hi or lo) + 1))
    h = len(ids) // 2
    return ",".join(map(str, ids[:h])), ",".join(map(str, ids[h:])), h, len(ids) - h


def chain(c, a, force, logd):
    p = N.plans(c, BOARD, a.gpu, a.cpus, a.procs, force, logd)
    redo = force or not p["hit"]
    N.export(c, BOARD, redo, logd)
    work = G.data_dir() / "runs" / "op_guard" / c["name"] / "nav" / BOARD
    if a.mode == "subset":
        h = N.harness(c, a.cpus, redo, work, logd)
        o = N.official_cached(c, BOARD) if not redo else None
        return dict(plans=p, harness=h, official=o, work=str(work))
    c1, c2, _, _ = split_cpus(a.cpus)
    res, err = {}, []

    def go(k, f):
        try:
            res[k] = f()
        except BaseException as e:      # noqa: BLE001  (SystemExit of a failed step included)
            err.append(f"{k}: {e}")
    ts = [threading.Thread(target=go, args=("official", lambda: N.official(c, BOARD, c1, redo, logd))),
          threading.Thread(target=go, args=("harness", lambda: N.harness(c, c2, redo, work, logd)))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        raise SystemExit("; ".join(err))
    return dict(plans=p, **res, work=str(work))


def main():
    ap = G.line_args(LINE, __doc__)
    ap.add_argument("--procs", type=int, default=N.PROCS, help="op_lb rollout shards on the card")
    a = ap.parse_args()
    t0 = time.time()
    cand = G.resolve(a.candidate)
    out = G.run_dir(cand["name"], a.mode)
    if G.done(out, LINE) and not a.force:
        N.say(f"{LINE}: cached {out}/lines/{LINE}.json")
        return
    logd = out / "logs" / LINE
    prov = {}
    if cand["name"] != G.SHIPPED:
        prov[G.SHIPPED] = chain(G.resolve(G.SHIPPED), a, False, logd)
    prov[cand["name"]] = chain(cand, a, a.force, logd)
    from jevdrive import stats
    pc, ps = prov[cand["name"]], prov[G.SHIPPED]
    off = all(p.get("official") for p in (pc, ps))      # both have the devkit script's CSV: report its numbers
    oc, os_ = ((N.v2_summary(pc["official"]["csv"]), N.v2_summary(ps["official"]["csv"])) if off else (pc["harness"], ps["harness"]))
    gc, gs = (pd.read_csv(Path(p["work"]) / "harness_groups.csv").set_index("group") for p in (pc, ps))
    tc, ts = (pd.read_csv(Path(p["work"]) / "harness_tokens.csv").set_index("token") for p in (pc, ps))
    chk = {n: round(p["harness"]["combined"] - N.v2_summary(p["official"]["csv"])["combined"], 4) for n, p in prov.items() if p.get("official")}
    src = ("official script" if off else "devkit harness (official aggregation; no official CSV in subset mode)") + \
        ("; harness - official combined: " + ", ".join(f"{n} {v:+.4f}" for n, v in chk.items()) if chk else "")
    rows = []
    for k, label, rule in (("combined", "EPDMS two-stage (5912 tokens)", True), ("stage1", "EPDMS stage 1", False), ("stage2", "EPDMS stage 2", False)):
        r = stats.paired(gc[k].to_numpy(), gs[k].to_numpy())
        note = (f"{src}; CI paired over {r['units']} mapping groups" if rule else "diagnostic (no rule); CI over mapping groups")
        rows.append(G.row(LINE, label, oc[k], os_[k], rule=G.LINES[LINE][2] if rule else "", ok=G.at_least(oc[k], os_[k], 0.0) if rule else None,
                          ci=N.ci(r), note=note, harness_delta=r["mean"]))
    E = pd.read_csv(N.EARLY).set_index("token")
    for st, label, rule in ((None, "EPDMS early-turn set (stages pooled)", True), (1, "EPDMS early-turn stage 1", False), (2, "EPDMS early-turn stage 2", False)):
        tok = E.index if st is None else E.index[E.stage == st]
        r = stats.paired(100 * tc.loc[tok, "score"].to_numpy(), 100 * ts.loc[tok, "score"].to_numpy(), groups=ts.loc[tok, "group"].to_numpy())
        note = (f"{r['n']} tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by {r['units']} mapping groups"
                + ("" if rule else "; diagnostic (no rule)"))
        rows.append(G.row(LINE, label, r["mean_a"], r["mean_b"], rule=G.LINES[LINE][2] if rule else "", ok=G.at_least(r["mean_a"], r["mean_b"], 0.0) if rule else None,
                          ci=N.ci(r), note=note, n=r["n"]))
    G.write_line(out, LINE, cand["name"], a.mode, rows, t0, provenance=prov)
    for r in rows:
        N.say(f"{LINE}: {r['metric']}: {r['value']:.3f} vs {r['ref']:.3f} delta {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] pass {r['pass']}")


if __name__ == "__main__":
    main()
