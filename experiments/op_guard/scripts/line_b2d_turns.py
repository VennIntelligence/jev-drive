"""Guard line `b2d_turns`: decision 127's 25 junction turns (13 choice + 12 forced on 20 Bench2Drive val routes, seed 2) with the
zones off: op_arb.sh arm `spec` at the open-loop-aligned camera (1.59, 0, 1.86) with DRIVE_ARGS '"zones": false, "div_m": 1e9' = the
rig122 lane's arm `olnz`, candidate ONNX via SRV_ONNX. Per-turn readouts are junction_rig122_report.score_attempt's (took the
intended branch = came within 5 m of the dense exit; leaves lane = peak cross-track > 1.75 m; collisions in the turn window = entry
to 15 s after it + 3 s); the turn set is the 25 labelled turns of results/junction_rig122_per_turn.csv.
Primary readout: took-exit rate (choice, forced; paired vs shipped with a route-cluster bootstrap); the only pass cell is
"collisions in turn windows not up vs shipped" (tmp plan section 4: took-exit up, collisions not up).
subset = full = the 25 turns (one seed; closed loop is not bitwise deterministic, single turns flip: decision 127).
Shipped cache: $DATA_DIR/runs/rig122/arms/olnz-s2-k* (decision 127's olnz), used when the unit config is arm olnz exactly; --force reruns it.

  python experiments/op_guard/scripts/line_b2d_turns.py --candidate it_dw3-s0 [--cl-lines b2d_turns,b2d_ds] [--cards 2]
      [--stage smoke|all] [--collect-only] [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cllib as C  # noqa: E402
import guardlib as G  # noqa: E402

LINE = "b2d_turns"


def collect(cand, mode, force):
    keys = C.turn_keys()
    me, cols, routes, src = C.turn_rows(cand, mode, force)
    rme, rcols, rroutes, _ = (me, cols, routes, src) if cand == G.SHIPPED else C.turn_rows(G.SHIPPED, mode)
    ix = {(r["route"], r["turn"]): r for r in me}
    rx = {(r["route"], r["turn"]): r for r in rme}
    both = [k for k in keys if k in ix and k in rx]
    have_ref = len(both) == len(keys)
    ks = both if have_ref else [k for k in keys if k in ix]
    rows = []
    for gid, gname, sel in (("took_choice", "took exit, choice turns", lambda r: r["forced"] == 0),
                            ("took_forced", "took exit, forced turns", lambda r: r["forced"] == 1), ("took_all", "took exit, all turns", lambda r: True)):
        kk = [k for k in ks if sel(ix[k])]
        y = [float(ix[k]["branch"] == "yes") for k in kk]
        yr = [float(rx[k]["branch"] == "yes") for k in kk] if have_ref else []
        d, ci = C.paired_ci(y, yr, [k[0] for k in kk]) if have_ref else (None, None)
        rows.append(G.row(gid, gname, sum(y) / max(len(y), 1), sum(yr) / max(len(yr), 1) if have_ref else None, "primary readout (up is the goal)", None, ci=ci,
                          note="%d / %d%s" % (sum(y), len(y), " vs %d / %d" % (sum(yr), len(yr)) if have_ref else "")))
    ent = [k for k in ks if ix[k]["entered"]]
    rent = [k for k in ks if have_ref and rx[k]["entered"]]
    lv = sum(ix[k]["leaves"] for k in ent) / max(len(ent), 1)
    rlv = sum(rx[k]["leaves"] for k in rent) / max(len(rent), 1) if have_ref else None
    rows.append(G.row("leaves_lane", "leaves lane, share of entered turns", lv, rlv, "reported", None, note="%d entered" % len(ent)))
    cw = sum(ix[k]["coll"] for k in ks)
    rcw = sum(rx[k]["coll"] for k in ks) if have_ref else None
    rows.append(G.row("coll_window", "collisions in turn windows", cw, rcw, "not up vs shipped", G.at_most(cw, rcw, 0), note="%d turns" % len(ks)))
    rs = [r for r in routes if r in rroutes] if have_ref else list(routes)
    hits = sum(c["kind"].startswith("hit") for c in cols if c["route"] in rs)
    rhits = sum(c["kind"].startswith("hit") for c in rcols if c["route"] in rs) if have_ref else None
    rows.append(G.row("coll_run", "collisions, whole runs (20 routes)", hits, rhits, "reported", None))
    ds = sum(routes[r]["ds"] for r in rs) / max(len(rs), 1)
    rds = sum(rroutes[r]["ds"] for r in rs) / max(len(rs), 1) if have_ref else None
    rows.append(G.row("route_ds", "route DS mean, zones off", ds, rds, "reported", None))
    missing = [r for r in C.TURN_ROUTES if r not in routes]
    prov = dict(turns=len(keys), routes=len(C.TURN_ROUTES), seed=C.TURN_SEED, arm="op_arb.sh spec, zones off (= rig122 olnz)", missing_routes=missing,
                per_turn=me, collisions=cols, per_route=routes, attempts=src,
                cache_hits=sorted({str(Path(s).parents[2]) for s in src.values() if str(C.CACHE_TURNS) in s}),
                unit_wall_s=round(sum(v["wall_s"] or 0 for r, v in routes.items() if str(C.CACHE_TURNS) not in src[r]), 1))
    return rows, prov, "ok" if not missing else "error"


if __name__ == "__main__":
    sys.exit(C.main_line(LINE, __doc__, collect))
