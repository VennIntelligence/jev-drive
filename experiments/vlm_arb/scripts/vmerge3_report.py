"""Report of the vmerge3 lane (plan 2026-10-04-vmerge3.md).

  python vmerge3_report.py report   -> $DATA_DIR/runs/vlm_arb/vmerge3/results/{vmerge3.md, vmerge3_runs.csv, vmerge3_paired.csv,
                                       vmerge3_bypass.csv, vmerge3_release.csv}
  python vmerge3_report.py smoke    the two debug units: bypass activations, perception vs truth, cross answers (stdout)

Primary read (registered): seeds 0 and 1, paired per route as vmerge2_report.py (route bootstrap, 2000 draws, seed 0). Arms: drive and
vmerge2 are the logged runs (drive seeds 2, 3 = vmerge2 batch reruns). Lines: obstacle-route vmerge3 - drive >= +20.7 DS; vehicle collisions
per run of vmerge3 <= drive's on the same seeds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_v2_report as v2  # noqa: E402
import vlm_vred_report as vr  # noqa: E402
import vmerge2_report as m2  # noqa: E402
from vlm_arb_checks import vlm_rows  # noqa: E402
from vlm_arb_common import LIGHT_ROUTES, OBS_ROUTES, ROUTES, RUN, SIGN_ROUTES, jsonl, route_row, unit_dir  # noqa: E402

OUT = RUN / "vmerge3/results"
ARMS3 = ("vmerge3", "vm3priv", "vm3norel")
SEEDS4 = (0, 1, 2, 3)
BYP_LINE, L_P95 = 41.43 / 2, 400.0


def collect(arms):
    rows = []
    for arm in arms:
        for s in SEEDS4:
            for rid in ROUTES:
                d = m2.run_dir(arm, s, rid)
                r = route_row(d, rid)
                if r:
                    rows.append(dict(arm=arm, seed=s, unit=d.name, **r))
    return pd.DataFrame(rows)


def truth_obstacles():
    """Per obstacle route: the privileged blocker extent and adjacent-lane offset that vmerge2's pbyp3 logged (route arc, m)."""
    out = {}
    for rid in OBS_ROUTES:
        for s in SEEDS4:
            d = m2.run_dir("vmerge2", s, rid)
            r = route_row(d, rid)
            if not r:
                continue
            for p in jsonl(Path(r["attempt"]) / "plans.jsonl"):
                st = (p.get("pc") or {}).get("bypass_state")
                if st:
                    out[rid] = dict(t_start_s=st["start_s"], t_end_s=st["end_s"], t_offset=st["offset"])
                    break
            if rid in out:
                break
    return out


def bypass_rows(arm, seeds, truth):
    """Per run: the perceived bypass (perc.jsonl) against the logged truth; gap refusals; pull-out start; contacts."""
    rows = []
    for s in seeds:
        for rid, _, k, r in vr.attempts(arm, s):
            a = Path(r["attempt"])
            pf = a / "perc.jsonl"
            P = jsonl(pf) if pf.exists() else []
            st = [p for p in P if p.get("state")]
            first = st[0] if st else None
            held = sum(1 for p in P if p.get("state") and p.get("gap") is False)
            plans = [p for p in jsonl(a / "plans.jsonl")] if st else []
            on = [p for p in plans if (p.get("pc") or {}).get("bypass")]
            started = [p for p in plans if (p.get("pc") or {}).get("shift_frac", 0) >= 0.05]
            tr = truth.get(rid, {})
            last = st[-1]["state"] if st else {}
            rows.append(dict(route=rid, seed=s, obstacle=rid in OBS_ROUTES, activated=bool(st), t_act=first["t"] if first else None,
                             side=(first or {}).get("state", {}).get("side"), offset=(first or {}).get("state", {}).get("offset"),
                             t_offset=tr.get("t_offset"), start_s=last.get("start_s"), end_s=last.get("end_s"),
                             t_start_s=tr.get("t_start_s"), t_end_s=tr.get("t_end_s"), gap_hold_snaps=held,
                             t_pullout=started[0]["t"] if started else None, path_on_snaps=len(on),
                             gap_reasons=";".join(sorted({str(p.get("why")).split(" ")[0] for p in P if p.get("gap") is False}))))
    return pd.DataFrame(rows)


def release_rows(arm, seeds):
    rows = []
    for s in seeds:
        for rid, _, k, r in vr.attempts(arm, s):
            ans, st, head = vlm_rows(r["attempt"])
            C = [d for d in ans if d["ans"].get("kind") == "cross"]
            last = st[-1] if st else {}
            rows.append(dict(route=rid, seed=s, n_cross=len(C), n_cross_pos=sum(d["ans"].get("Q_cross") == "vehicle_crossing" for d in C),
                             cusum_releases=last.get("n_rel", 0), waits=last.get("n_cwait", 0), timeouts=last.get("n_ctime", 0),
                             reholds=last.get("n_rehold", 0), s_waiting=0.5 * sum(1 for x in st if x.get("rel_wait")),
                             r6_snaps=sum(1 for x in st if "R6" in x.get("rules", []))))
    return pd.DataFrame(rows)


def latency(arm, seed):
    rows = []
    for rid, s, k, r in vr.attempts(arm, seed):
        ans, _, _ = vlm_rows(r["attempt"])
        rows += [dict(ok=bool(d["ans"].get("ok")), lat=d["ans"].get("latency_ms", np.nan), kind=d["ans"].get("kind", "light")) for d in ans]
    return pd.DataFrame(rows)


def smoke(_):
    truth = truth_obstacles()
    for name, rid in (("dbg-vm3-s0-25169", "25169"), ("dbg-vm3-s0-334", "334")):
        d = RUN / "arms" / ("v2-" + name)
        r = route_row(d, rid)
        print("==", name, None if not r else {k: r[k] for k in ("DS", "RC", "collisions_vehicle", "collisions_layout", "red_light", "vehicle_blocked", "crash")})
        if not r:
            continue
        a = Path(r["attempt"])
        P = jsonl(a / "perc.jsonl") if (a / "perc.jsonl").exists() else []
        st = [p for p in P if p.get("state")]
        print("perc snaps", len(P), "with state", len(st), "first", st[0] if st else None)
        ans, sts, _ = vlm_rows(a)
        C = [x for x in ans if x["ans"].get("kind") == "cross"]
        print("answers", len(ans), "ok", sum(bool(x["ans"].get("ok")) for x in ans), "cross", len(C),
              "cross p", [round(x["ans"]["Q_cross_p"]["vehicle_crossing"], 2) for x in C if x["ans"].get("ok")][:40])
        print("last state", sts[-1] if sts else None)
    print("truth", truth)


def report(_):
    OUT.mkdir(parents=True, exist_ok=True)
    arms = ["drive", "vmerge2"] + list(ARMS3)
    df = collect(arms)
    df.to_csv(OUT / "vmerge3_runs.csv", index=False)
    other = [r for r in ROUTES if r not in LIGHT_ROUTES + SIGN_ROUTES + OBS_ROUTES]
    sets = [("all", ROUTES), ("obstacle", OBS_ROUTES), ("light", LIGHT_ROUTES), ("stop sign", SIGN_ROUTES), ("other", other)]
    cols = ["DS", "RC", "red_light", "stop_infraction", "collisions_vehicle", "collisions", "vehicle_blocked"]
    con = [("vmerge3", "drive"), ("vmerge3", "vmerge2"), ("vmerge3", "vm3priv"), ("vmerge3", "vm3norel"), ("vm3priv", "drive"),
           ("vm3norel", "drive"), ("vmerge2", "drive")]
    d01 = df[df.seed.isin((0, 1))]
    P1 = v2.pair_rows(d01, con, sets, cols)
    P1.insert(0, "seeds", "0,1")
    seeds_all = sorted(set(df[df.arm == "vmerge3"].seed))
    dA = df[df.seed.isin(seeds_all)]
    PA = v2.pair_rows(dA, con, sets, cols) if len(seeds_all) > 2 else None
    if PA is not None:
        PA.insert(0, "seeds", ",".join(map(str, seeds_all)))
    pd.concat([P1] + ([PA] if PA is not None else [])).to_csv(OUT / "vmerge3_paired.csv", index=False)
    inf = df.groupby(["arm"])[["red_light", "stop_infraction", "collisions_vehicle", "collisions_layout", "collisions_pedestrian",
                               "outside_route_lanes", "vehicle_blocked", "route_timeout", "scenario_timeouts"]].sum().reindex(arms)
    # the two lines
    obs = P1[(P1.contrast == "vmerge3 - drive") & (P1.routes == "obstacle") & (P1.metric == "DS")].iloc[0]
    cpr = d01.groupby("arm").collisions_vehicle.mean()
    line_b = obs["est"] >= BYP_LINE
    line_c = cpr.get("vmerge3", np.nan) <= cpr.get("drive", np.nan)
    truth = truth_obstacles()
    B = pd.concat([bypass_rows(a, seeds_all, truth).assign(arm=a) for a in ("vmerge3", "vm3norel")], ignore_index=True)
    B.to_csv(OUT / "vmerge3_bypass.csv", index=False)
    Rl = pd.concat([release_rows(a, seeds_all).assign(arm=a) for a in ("vmerge3", "vm3priv")], ignore_index=True)
    Rl.to_csv(OUT / "vmerge3_release.csv", index=False)
    lat = pd.concat([latency("vmerge3", s).assign(seed=s) for s in seeds_all], ignore_index=True)
    lok = lat[lat.ok]
    lat_line = "n %d, answered %.1f%%, p50 / p95 / p99 %.0f / %.0f / %.0f ms (cross %d requests, p95 %.0f ms)" % (
        len(lat), 100 * lat.ok.mean(), v2.pct(lok.lat, 50), v2.pct(lok.lat, 95), v2.pct(lok.lat, 99), (lat.kind == "cross").sum(),
        v2.pct(lok[lok.kind == "cross"].lat, 95))
    missing = []
    for a in ARMS3:
        for s in seeds_all:
            miss = [r for r in ROUTES if not ((df.arm == a) & (df.seed == s) & (df.route == r)).any()]
            if miss:
                missing.append("%s seed %d: %d of 19 routes missing (%s)" % (a, s, len(miss), " ".join(miss)))
    per_route = df[df.arm.isin(["drive", "vmerge2"] + list(ARMS3))].pivot_table(index="route", columns=["arm", "seed"], values="DS", aggfunc="mean").round(1)
    D = ["# vmerge3: the bypass without privileged input + a cross-traffic release check (19 routes, diagnostic)", "",
         "Plan: [../plans/2026-10-04-vmerge3.md](../plans/2026-10-04-vmerge3.md). `drive` and `vmerge2` are the logged runs of the earlier batches "
         "(drive seeds 2, 3 = the reruns of the vmerge2 batch). Arms: `vmerge3` = vmerge2 + perceived bypass + release check; `vm3priv` = vmerge2 + "
         "release check (privileged bypass); `vm3norel` = vmerge2 + perceived bypass.", "",
         "## Missing runs", "", ("\n".join("- " + m for m in missing) if missing else "none: every registered run finished"), "",
         "## Registered lines (seeds 0 and 1)", "",
         "- Bypass: vmerge3 - drive on the 4 obstacle routes = %s DS; line >= +%.1f: **%s**" % (v2.fmt(obs.to_dict()), BYP_LINE, "yes" if line_b else "no"),
         "- Collisions: vehicle collisions per run vmerge3 %.3f (%d / %d) vs drive %.3f (%d / %d): **%s**" % (
             cpr.get("vmerge3", np.nan), d01[d01.arm == "vmerge3"].collisions_vehicle.sum(), (d01.arm == "vmerge3").sum(),
             cpr.get("drive", np.nan), d01[d01.arm == "drive"].collisions_vehicle.sum(), (d01.arm == "drive").sum(), "yes" if line_c else "no"), "",
         "Vehicle collisions per run, seeds 0 and 1: " + ", ".join("%s %.3f" % kv for kv in cpr.reindex(arms).items()), "",
         "## Arms, seeds 0 and 1", "", v2.arm_table(d01, arms).to_markdown(), "",
         "## Arms, every finished seed", "", v2.arm_table(df, arms).to_markdown(), "",
         "## Infractions by type (official, summed over runs, every finished seed)", "", inf.to_markdown(), "",
         "## Paired differences, seeds 0 and 1 (registered primary; mean over routes [95% route-cluster CI])", ""]
    D += v2.pair_md(P1, cols, cols)
    if PA is not None:
        D += ["", "## Paired differences, seeds %s" % ",".join(map(str, seeds_all)), ""] + v2.pair_md(PA, cols, cols)
    D += ["", v2.NOISE, "", "## Latency (vmerge3, every answered request)", "", lat_line, "",
          "## Perceived bypass against the logged truth (vmerge3 and vm3norel; truth = vmerge2's privileged pbyp3 extent / offset on the same route)", "",
          B.round(2).to_markdown(index=False), "",
          "## Release check (vmerge3 and vm3priv): cross questions, waits, timeouts, R6 re-holds", "", Rl.round(2).to_markdown(index=False), "",
          "## DS per route and seed", "", per_route.to_markdown(), ""]
    (OUT / "vmerge3.md").write_text("\n".join(D))
    print("\n".join(D))


if __name__ == "__main__":
    {"report": report, "smoke": smoke}[sys.argv[1]](sys.argv[2:])
