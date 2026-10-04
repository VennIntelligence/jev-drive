"""Guard line `hugsim`: HUGSIM closed loop under interface preset `spec` (docs/openpilot-interface.md), candidate ONNX served.

subset = decision 124's 11 representative scenes (experiments/leaderboard_audit/scripts/unified_hugsim_small.txt), full = all 64
(experiments/hugsim/scripts/derot_all64.txt); one run per scene (decision 124: HUGSIM is near-deterministic, 10-11 of 11 scenes
reproduce). Readouts per scene: spin (spin_analysis.analyse, decision 118 definition: heading error vs the route >= 60 deg), end class
(zs_run: complete / max_steps = stuck / fg / bg collision / off_route), HD-Score. Rule: spins not up vs shipped; stuck, completes and HD
(paired over scenes, jevdrive.stats bootstrap) are reported without a line ("stuck reduction is the goal").
Shipped subset cache: $DATA_DIR/runs/unified/hugsim tag uni-d118 (preset opctrl_d118 = spec), used when every scene's interface.json
is the spec resolution; --force reruns it.

  python experiments/op_guard/scripts/line_hugsim.py --candidate it_dw3-s0 [--mode subset|full] [--cl-lines hugsim,b2d_turns,b2d_ds]
      [--cards 2] [--stage smoke|all] [--collect-only] [--force]          (blocking; lease: OP_GUARD_LANE or its own, see cllib.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

LINE = "hugsim"


def collect(cand, mode, force):
    want = [Path(s).stem for s in C.HUGSIM_SETS[mode].read_text().split()]
    me = C.hugsim_scenes(cand, mode, force)
    ref = me if cand == G.SHIPPED else C.hugsim_scenes(G.SHIPPED, mode)
    both = [s for s in want if s in me and s in ref]
    have_ref = len(both) == len(want)

    def cnt(d, f, ks):
        return sum(f(d[s]) for s in ks)
    rows = []
    for rid, metric, f in (("spins", "spins (heading err >= 60 deg)", lambda r: r["spin"]), ("stuck", "stuck (end max_steps)", lambda r: r["end"] == "max_steps"),
                           ("complete", "completes", lambda r: r["end"] == "complete")):
        v = cnt(me, f, both if have_ref else [s for s in want if s in me])
        rv = cnt(ref, f, both) if have_ref else None
        if rid == "spins":
            rows.append(G.row(rid, metric, v, rv, "not up vs shipped", G.at_most(v, rv, 0), note="%d scenes" % len(both)))
        else:
            rows.append(G.row(rid, metric, v, rv, "reported (stuck: reduction is the goal)" if rid == "stuck" else "reported", None))
    ks = both if have_ref else [s for s in want if s in me]
    hd = sum(me[s]["hd"] for s in ks) / max(len(ks), 1)
    d, ci = C.paired_ci([me[s]["hd"] for s in both], [ref[s]["hd"] for s in both]) if have_ref else (None, None)
    rows.append(G.row("hd", "HD-Score mean (paired over scenes)", hd, (hd - d) if d is not None else None, "reported", None, ci=ci,
                      note="paired mean diff %+.3f" % d if d is not None else "no shipped reference"))
    missing = [s for s in want if s not in me]
    prov = dict(scenes=C.HUGSIM_SETS[mode].name, n=len(want), missing=missing, missing_ref=[s for s in want if s not in ref],
                per_scene={s: me[s] for s in want if s in me}, cache_hits=sorted({r["src"] for r in me.values() if r["src"] != "run"}),
                unit_wall_s=round(sum(r["wall_s"] for r in me.values() if r["src"] == "run"), 1))
    return rows, prov, "ok" if not missing else "error"


if __name__ == "__main__":
    sys.exit(C.main_line(LINE, __doc__, collect))
