"""Gate and report of the vmerge lane (plan 2026-10-03-vmerge.md).

  python vmerge_report.py few      seed-0 units: finished, no crash, answered >= 98%, latency p95 <= L + 10 ms, > TTL <= 1% -> gates/vmerge_few.json
  python vmerge_report.py report [abl ...]   vmerge.md + CSVs -> $DATA_DIR/runs/vlm_arb/vmerge/results/ (ablation arms vm<abl> added)

Conventions of results/report.md: official infractions; paired differences per (route, seed) averaged over seeds, resampling routes (2000
draws, seed 0), 95% percentile intervals. `drive` and `vred` are the logged runs of the earlier batches (not rerun). Diagnostic read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_v2_report as v2  # noqa: E402
import vlm_vred_report as vr  # noqa: E402
from vlm_arb_checks import vlm_rows  # noqa: E402
from vlm_arb_common import LIGHT_ROUTES, OBS_ROUTES, ROUTES, RUN, SEEDS, SIGN_ROUTES, jsonl, route_row, write_json  # noqa: E402

OUT = RUN / "vmerge/results"
L_REG, LAT_TOL_MS, TTL_MS = 0.35, 10.0, 2500.0
ARM = "vmerge"


def answers(seed):
    """Every answered request (light and sign) of the vmerge units of one seed, with truth and trigger columns."""
    rows = []
    for rid, s, k, r in vr.attempts(ARM, seed):
        ans, st, head = vlm_rows(r["attempt"])
        for d in ans:
            g, a = d["gt"], d["ans"]
            rows.append(dict(route=rid, seed=s, t_q=d["t_q"], ok=bool(a.get("ok")), kind=a.get("kind", "light"), lat=a.get("latency_ms", np.nan),
                             ans=a.get("Q_sign") if a.get("kind") == "sign" else a.get("Q_light"), tl=g.get("tl"), tl_dist=g.get("tl_dist"),
                             stop_dist=g.get("stop_dist"), why=g.get("why", ""), det_l=(g.get("det") or [np.nan, np.nan])[0],
                             det_s=(g.get("det") or [np.nan, np.nan])[1]))
    return pd.DataFrame(rows)


def few(_):
    rows, crashes = [], 0
    for rid in ROUTES:
        r = route_row(v2.run_dir(ARM, 0, rid), rid)
        rows.append(dict(route=rid, finished=bool(r), crash=bool(r and r["crash"]), DS=r["DS"] if r else None))
        crashes += bool(r and r["crash"])
    an = answers(0)
    lat = an.lat[an.ok].to_numpy()
    lat_ok = bool(len(lat) and v2.pct(lat, 95) <= 1e3 * L_REG + LAT_TOL_MS and np.mean(lat > TTL_MS) <= 0.01 and an.ok.mean() >= 0.98)
    df = pd.DataFrame(rows)
    gate = dict(passed=bool(df.finished.all() and crashes == 0 and lat_ok), latency_ok=lat_ok, p50_ms=v2.pct(lat, 50), p95_ms=v2.pct(lat, 95),
                over_L=float(np.mean(lat > 1e3 * L_REG)) if len(lat) else None, answered=float(an.ok.mean()) if len(an) else None,
                n_answers=int(len(an)), crashes=crashes, routes=rows)
    write_json(RUN / "gates/vmerge_few.json", gate)
    print(df.to_string())
    print({k: v for k, v in gate.items() if k != "routes"})


def components(seed):
    """Per run: ask reasons, trigger recall on light approaches, R2 holds / cusum releases (with the truth light at the release),
    R3 holds, sign answers, bypass activations and suppressions."""
    out = []
    for rid, s, k, r in vr.attempts(ARM, seed):
        a = Path(r["attempt"])
        ans, st, head = vlm_rows(a)
        L = [d for d in ans if d["ans"].get("kind", "light") == "light"]
        S = [d for d in ans if d["ans"].get("kind") == "sign"]
        appr = [d for d in L if d["gt"].get("tl") in (0, 1, 2) and d["gt"].get("tl_dist") is not None and 0 <= d["gt"]["tl_dist"] < 40]
        rel = sorted({x["rel_t"] for x in st if x.get("rel_t") is not None})
        false_rel = 0
        for tr in rel:                                      # the truth light at the release: the latest answered request before it
            prev = [d for d in L if d["t_q"] <= tr]
            if prev and prev[-1]["gt"].get("tl") in (1, 2) and (prev[-1]["gt"].get("tl_dist") or 99) < 50:
                false_rel += 1
        why = pd.Series([d["gt"].get("why", "") for d in L])
        n_r2 = sum(1 for p, q in zip(st, st[1:]) if "R2" in q.get("rules", []) and "R2" not in p.get("rules", []))
        n_r3 = sum(1 for p, q in zip(st, st[1:]) if "R3" in q.get("rules", []) and "R3" not in p.get("rules", []))
        priv = jsonl(a / "privileged.jsonl") if (a / "privileged.jsonl").exists() else []
        byp = [p for p in priv if p["pc"].get("bypass")]
        sup = pd.Series([p["pc"].get("suppressed") for p in priv if p["pc"].get("suppressed")])
        out.append(dict(route=rid, seed=s, n_light=len(L), n_sign=len(S), ask_det_only=int((why == "det").sum()),
                        ask_window=int(why.str.contains("window").sum()), ask_hold=int(why.str.contains("hold").sum()),
                        appr_frames=len(appr), appr_det_fired=int(sum(1 for d in appr if "det" in d["gt"].get("why", ""))),
                        r2_holds=n_r2, cusum_releases=len(rel), false_releases=false_rel, r3_holds=n_r3,
                        sign_yes=int(sum(d["ans"].get("Q_sign") == "stop_sign_for_ego" for d in S)),
                        bypass_snapshots=len(byp), byp_first_t=byp[0]["t"] if byp else None,
                        byp_suppressed=";".join("%s:%d" % kv for kv in sup.value_counts().items())))
    return pd.DataFrame(out)


def report(abl):
    OUT.mkdir(parents=True, exist_ok=True)
    abl = ["vm" + a for a in abl]
    arms = ["drive", "vred", ARM] + abl
    df = v2.collect(arms)
    df.to_csv(OUT / "vmerge_runs.csv", index=False)
    other = [r for r in ROUTES if r not in LIGHT_ROUTES + SIGN_ROUTES + OBS_ROUTES]
    sets = [("all", ROUTES), ("light", LIGHT_ROUTES), ("stop sign", SIGN_ROUTES), ("obstacle", OBS_ROUTES), ("other", other)]
    cols = ["DS", "RC", "red_light", "stop_infraction", "collisions", "vehicle_blocked"]
    con = [(ARM, "drive"), (ARM, "vred"), ("vred", "drive")] + [(ARM, a) for a in abl if a != "vmj"]
    if "vmj" in abl:                                   # the follow-up arm runs on the light routes + 17280 only
        con += [("vmj", ARM), ("vmj", "vred"), ("vmj", "drive")]
    P = v2.pair_rows(df, con, sets, cols)
    P.to_csv(OUT / "vmerge_paired.csv", index=False)
    inf = df.groupby("arm")[["red_light", "stop_infraction", "collisions_vehicle", "collisions_layout", "collisions_pedestrian",
                             "outside_route_lanes", "vehicle_blocked", "route_timeout", "scenario_timeouts", "min_speed_infractions"]].sum().reindex(arms)
    C = pd.concat([components(s) for s in SEEDS], ignore_index=True)
    C.to_csv(OUT / "vmerge_components.csv", index=False)
    an = pd.concat([answers(s) for s in SEEDS], ignore_index=True)
    lat = an.lat[an.ok].to_numpy()
    per_route = df.pivot_table(index="route", columns="arm", values="DS", aggfunc="mean").reindex(columns=arms).round(1)
    D = ["# vmerge: the merged Bench2Drive arm (19 routes x 2 traffic seeds, diagnostic)", "",
         "Plan: [../plans/2026-10-03-vmerge.md](../plans/2026-10-03-vmerge.md). `drive` and `vred` are the logged runs of the earlier batches.", "",
         "## Arms", "", v2.arm_table(df, arms).to_markdown(), "", "## Infractions by type (official, summed over runs)", "", inf.to_markdown(), "",
         "## Paired differences (mean over routes [95% route-cluster CI])", ""]
    D += v2.pair_md(P, cols, cols) + ["", v2.NOISE, "", "## Components (from the vmerge logs, summed over runs)", "",
                                      C.drop(columns=["route", "seed", "byp_first_t", "byp_suppressed"]).sum().to_frame("sum").T.to_markdown(index=False), "",
                                      "Per run:", "", C.to_markdown(index=False), "",
                                      "## Latency (all answered requests, light and sign)", "",
                                      "n %d, answered %.1f%%, p50 / p95 / p99 %.0f / %.0f / %.0f ms, over L %.1f%%, over TTL %.2f%%; sign requests %d" % (
                                          len(an), 100 * an.ok.mean(), v2.pct(lat, 50), v2.pct(lat, 95), v2.pct(lat, 99), 100 * np.mean(lat > 1e3 * L_REG),
                                          100 * np.mean(lat > TTL_MS), int((an.kind == "sign").sum())), "",
                                      "## DS per route (mean over seeds)", "", per_route.to_markdown(), ""]
    (OUT / "vmerge.md").write_text("\n".join(D))
    print("\n".join(D))


if __name__ == "__main__":
    {"few": few, "report": report}[sys.argv[1]](sys.argv[2:])
