#!/usr/bin/env python
"""Guard line `navtest`: official NAVSIM v1 PDMS of the candidate, paired per token against shipped Cinque on the same tokens.

  .venv/bin/python experiments/op_guard/scripts/line_navtest.py --candidate it_dw3-s0 --mode subset --gpu 0 --cpus 0-24

Modes: `subset` = jevdrive.data.splits navsim/op-guard-navtest-sub (log x driving-command x shipped-score stratified draw,
nav_subset.py; tracking study in results/navtest_subset_study_*.md), `full` = all 12 146 navtest tokens.
Rule: candidate PDMS >= shipped PDMS - 0.3 on the paired mean (same tokens); the CI is the log-cluster paired bootstrap
(jevdrive.stats.paired, groups = log). Chain and cache rules: navlib.py.
Subset choice: 0.33 of navtest = the largest fraction whose rollout fits the ~35 min nav budget next to navhard; on the 26 arms
already scored in full whose delta is within 1.5 PDMS of shipped, the frozen subset's delta is off the full delta by 0.06 mean,
0.18 max (results/navtest_subset_freeze.json; 50-seed study: results/navtest_subset_study_log-cmd-sbin.md).
Cold runtime (2026-10-05, one RTX 6000D card, 25 cores, 4 rollout shards): subset 6.5 min (it_dw3-s0) to 9.4 min (shipped, card
shared with a training job; includes the one-time 26 s subset frame copy), of which ~1 min export + scoring; full 19.5 min
(rollout 17.0, export 0.5, scoring 2.0). The reference (shipped) is computed once and cached.
Resumable: every step skips what exists; --force redoes this candidate's rollout and scoring (never shipped's, unless the
candidate is shipped).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guardlib as G  # noqa: E402
import navlib as N  # noqa: E402

LINE = "navtest"
SUBS = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "comfort": "C"}


def chain(c, board, a, force, logd):
    p = N.plans(c, board, a.gpu, a.cpus, a.procs, force, logd)
    N.export(c, board, force or not p["hit"], logd)
    o = N.official(c, board, a.cpus, force or not p["hit"], logd)
    return dict(plans=p, official=o)


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
    board = "navtest_sub" if a.mode == "subset" else "navtest"
    logd = out / "logs" / LINE
    ship = G.resolve(G.SHIPPED)
    prov = {}
    if cand["name"] != G.SHIPPED:
        prov[G.SHIPPED] = chain(ship, board, a, False, logd)
    prov[cand["name"]] = chain(cand, board, a, a.force, logd)
    from jevdrive import stats
    x, y = N.v1_tokens(prov[cand["name"]]["official"]["csv"]), N.v1_tokens(prov[G.SHIPPED]["official"]["csv"])
    want = N.names(board)
    idx = [t for t in want if t in x.index and t in y.index]
    r = stats.paired(100 * x.loc[idx, "score"].to_numpy(), 100 * y.loc[idx, "score"].to_numpy(), groups=N.navtest_logs(idx))
    subs = {k: 100 * float((x.loc[idx, c] - y.loc[idx, c]).mean()) for c, k in SUBS.items()}
    note = (f"{len(idx)} / {len(want)} tokens valid in both; paired mean delta, 95% CI log-cluster bootstrap over {r['units']} logs; "
            "sub-score deltas (pp) " + " ".join(f"{k} {v:+.2f}" for k, v in subs.items()))
    rows = [G.row(LINE, f"PDMS v1 ({'subset' if a.mode == 'subset' else 'full'}, {len(idx)} tokens)", r["mean_a"], r["mean_b"], rule=G.LINES[LINE][2],
                  ok=G.at_least(r["mean_a"], r["mean_b"], 0.3), ci=N.ci(r), note=note, n=len(idx), n_expected=len(want), sub_deltas=subs)]
    from jevdrive.data import splits
    prov["split"] = splits.load(N.SUB_SPLIT).id if a.mode == "subset" else splits.load("navsim/navtest").id
    G.write_line(out, LINE, cand["name"], a.mode, rows, t0, provenance=prov)
    N.say(f"{LINE}: {rows[0]['metric']} {r['mean_a']:.3f} vs shipped {r['mean_b']:.3f}, delta {stats.fmt(r)}, pass {rows[0]['pass']}")


if __name__ == "__main__":
    main()
