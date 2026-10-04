"""Guard line `b2d_ds`: Bench2Drive driving score on decision 107's 19 val routes (vlm_arb DEV + TGT) under op_arb.sh arm `spec`
(zones on, turn desire on, open-loop-aligned camera (1.59, 0, 1.86), clip + 0.2 s delay), candidate ONNX via SRV_ONNX.
full = 19 routes x seeds 2, 3 (decision 107's seeds); subset = 19 routes x seed 2 (all routes kept because the decision-107 drop came
from two routes, 37969 / 27297; one seed halves the cost; seed 2 is also the b2d_turns seed). No shipped `spec` run at this camera
existed, so the shipped candidate is run like any other (no cache).
Rule "DS no drop", made noise-aware: paired mean DS difference over (seed, route) runs >= -tol, tol = 2.3 x sqrt(209 / n): decision
38's 95% band of the difference of two independent evaluations of one checkpoint (+-2.3 DS over 209 routes, i.e. route-level SD 11.6),
scaled to n paired runs (subset n = 19: 7.6 DS; full n = 38: 5.4 DS). The CI is jevdrive.stats.paired with routes as clusters.

  python experiments/op_guard/scripts/line_b2d_ds.py --candidate it_dw3-s0 [--mode subset|full] [--cl-lines b2d_turns,b2d_ds]
      [--cards 2] [--stage smoke|all] [--collect-only] [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

LINE = "b2d_ds"


def collect(cand, mode, force):
    want = [(s, r) for s in C.DS_SEEDS[mode] for r in C.DS_ROUTES]
    me = C.ds_rows(cand, mode)
    ref = me if cand == G.SHIPPED else C.ds_rows(G.SHIPPED, mode)
    both = [k for k in want if k in me and k in ref]
    have_ref = len(both) == len(want)
    ks = both if have_ref else [k for k in want if k in me]
    tol = C.ds_tol(len(want))
    rows = []
    v = sum(me[k]["DS"] for k in ks) / len(ks) if ks else None
    d, ci = C.paired_ci([me[k]["DS"] for k in both], [ref[k]["DS"] for k in both], [k[1] for k in both]) if have_ref else (None, None)
    rv = v - d if d is not None else None
    rows.append(G.row("ds", "DS mean (paired over seed x route)", v, rv, "delta >= -%.1f (no drop beyond decision-38 noise)" % tol,
                      G.at_least(v, rv, tol), ci=ci, note="n %d, paired diff %s" % (len(ks), "%+.2f" % d if d is not None else "-"), tol=tol))
    for rid, metric, key in (("rc", "RC mean", "RC"), ("collisions", "collisions (sum)", "collisions"), ("red_light", "red lights (sum)", "red_light"),
                             ("completed", "routes completed", None)):
        f = (lambda r: r["status"] == "Completed") if key is None else (lambda r, key=key: r[key])
        agg = (lambda d: sum(f(d[k]) for k in ks) / len(ks) if ks else None) if rid == "rc" else (lambda d: sum(f(d[k]) for k in ks) if ks else None)
        rows.append(G.row(rid, metric, agg(me), agg(ref) if have_ref else None, "reported", None))
    missing = [k for k in want if k not in me]
    prov = dict(routes=C.DS_ROUTES, seeds=list(C.DS_SEEDS[mode]), arm="op_arb.sh spec (zones on, desire on, open-loop camera)", tol=tol,
                missing=["s%d-%s" % k for k in missing], missing_ref=["s%d-%s" % k for k in want if k not in ref],
                per_run={"s%d-%s" % k: {x: me[k][x] for x in ("DS", "RC", "status", "collisions", "red_light", "attempt", "wall_s") if x in me[k]} for k in me},
                unit_wall_s=round(sum(r.get("wall_s") or 0 for r in me.values()), 1), cache_hits=[])
    return rows, prov, "ok" if not missing else "error"


if __name__ == "__main__":
    sys.exit(C.main_line(LINE, __doc__, collect))
