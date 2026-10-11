"""Log-consistent margin pilot (plans/2026-10-11-logmargin-pilot-prereg.md): G0 and the pilot's three lines, collected from the readers' files.
Reads no checkpoint and runs no model (CPU).

Inputs under $DATA_DIR/runs/body1/logmargin/: ident_off.json (bd4_train.py ident, G0.a), g0-<tag>/g0.json (bd4_train.py --logmargin, G0.b / G0.c),
reads/g3_lm_{hold,val}.{json,csv} (bd4_g3.py), reads/ol_lm_{navtest_w,hold}.{json,csv} (route_ol.py); $DATA_DIR/runs/body1/prog/ol_lm_{navtest_w,
hold}.parquet (prog_ol.py; arc ratios by shape_gate.arcs); the trainers' events (dev ADE, training scalars). -> report.{json,md} in the same directory.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/logmargin_report.py
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import argparse  # noqa: E402
import json  # noqa: E402

import b1 as B  # noqa: E402

TAG, REF, SP = "P2H10SM-P-s0", "P2H10-P-s0", "P2H10S-P-s0"
SPREAD = 12                                         # W2 seed spread of the base on warp frames, full scale (decision 234: 84 / 79 / 87 / 91)
S_FALL = dict(agent=0.3712121212, bnd=0.3381294964)  # P2H10S-P-s0 against P2H10-P-s0, hold pooled (results/loss/g3_a6_hold_S.json)
KEEP, ADE_TOL, ARC = 0.5, 0.005, (0.995, 1.005)
SCAL = ["imit", "agent", "agent_pos", "imit/agent_pos", "agent_ho", "agent_ho_pos", "road_ho", "road_ho_pos", "fam/ot1/road_pos", "fam/yr1/road_pos",
        "fam/bd4/road_pos", "fam/bd4/agent_pos", "hinge"]


def J(f):
    return json.loads(_pl.Path(f).read_text())


def main(a):
    import pandas as pd
    import route_gate as RG
    import shape_gate as SG
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    OD = data_dir() / "runs/body1/logmargin"
    R = OD / "reads"
    with Run("body1", "logmargin-report", config=vars(a)) as run:
        res = dict(tag=TAG, ref=REF, s_pilot=SP)
        res["ident"] = J(OD / "ident_off.json")
        res["g0"] = J(OD / f"g0-{TAG}" / "g0.json")
        N, H = J(R / "ol_lm_navtest_w.json")["gate"], J(R / "ol_lm_hold.json")["gate"]
        b = "> 45 deg"
        nav = {t: N[t]["pooled"][b] for t in (TAG, SP, REF)}
        hold = {t: dict(log=H[t]["log"][b], pooled=H[t]["pooled"][b]) for t in (TAG, SP, REF)}
        archived = J(B.REPO / "experiments/body1/results/navtest_warp/route/ol_a6_pilot_navtest_w.json")["gate"]
        Dn = pd.read_csv(R / "ol_lm_navtest_w.csv")
        side = {f"{t} {s}": Dn[(Dn.fam == "pooled") & (Dn.bucket == f"> 45 deg {s}") & (Dn.tag == t)].iloc[0][["n", "wide_2", "leave_4", "out4_mean"]].to_dict()
                for t in (TAG, SP, REF) for s in ("right", "left")}
        g3 = {s: J(R / f"g3_lm_{s}.json")["pooled"] for s in ("hold", "val")}
        G3 = {s: pd.read_csv(R / f"g3_lm_{s}.csv") for s in ("hold", "val")}
        arcs = {"hold": SG.arcs("lm_hold", [(TAG, REF), (SP, REF)], REF), "navtest": SG.arcs("lm_navtest_w", [(TAG, REF), (SP, REF)], REF)}
        dev = {t: SG.dev_of(t) for t in (TAG, SP, REF)}
        scal = {t: RG.last_scalars(t, SCAL) for t in (TAG, SP)}
        hb = g3["hold"][REF]
        arc_h = arcs["hold"][(TAG, "all")]
        L = {"i W2": bool(nav[TAG]["W2"] <= nav[REF]["W2"] + SPREAD),
             "ii agent": bool(hb["agent"]["rel_fall"] >= KEEP * S_FALL["agent"]),
             "ii boundary": bool(hb["bnd"]["rel_fall"] >= KEEP * S_FALL["bnd"]),
             "iii dev ADE": bool(dev[TAG]["ade"] <= dev[SP]["ade"] + ADE_TOL),
             "iii arc": bool(ARC[0] <= arc_h[0] <= ARC[1])}
        li, lii, liii = L["i W2"], L["ii agent"] and L["ii boundary"], L["iii dev ADE"] and L["iii arc"]
        verdict = ("all three lines met: candidate S' for the next joint run" if li and lii and liii else
                   "; ".join(([] if li else ["(i) missed: the fixed margin is not the drift mechanism (at pilot scale)"]) +
                             ([] if lii else ["(ii) missed: log-consistent margins remove the lesson"]) +
                             ([] if liii else ["(iii) missed: not a candidate (imitation accuracy or speed profile)"])))
        res |= dict(navtest=nav, navtest_side=side, navtest_archived={t: archived[t]["pooled"][b] for t in (SP, REF)}, hold_route=hold,
                    rates={s: {r: {q: g3[s][r][q] for q in ("agent", "bnd", "road", "sum") if q in g3[s][r]} for r in g3[s]} for s in g3},
                    arcs={s: {f"{k[0]} | {k[1]}": v for k, v in arcs[s].items()} for s in arcs}, dev=dev, scalars=scal, lines=L,
                    line_values=dict(W2_limit=nav[REF]["W2"] + SPREAD, agent_fall_min=KEEP * S_FALL["agent"], bnd_fall_min=KEEP * S_FALL["bnd"],
                                     agent_fall_min_if_402=KEEP * 0.402, dev_ade_max=dev[SP]["ade"] + ADE_TOL, arc_band=ARC), verdict=verdict)
        (OD / "report.json").write_text(json.dumps(res, indent=1, default=float) + "\n")

        f = lambda v: f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"  # noqa: E731
        ok = lambda k: "met" if L[k] else "**NOT met**"  # noqa: E731
        rr = lambda p, q: f"{p[q]['new']:.4f} against {p[q]['base']:.4f}: fall {100 * p[q]['rel_fall']:.1f} %, {p[q]['diff']:+.4f} [{p[q]['lo']:+.4f}, {p[q]['hi']:+.4f}]"  # noqa: E731
        M = [f"# Log-consistent margin pilot `{TAG}`: report (generated by logmargin_report.py)", "",
             f"G0.a identity: {'bit for bit' if res['ident']['identical'] else '**NOT identical**'} ({res['ident']['scalars_compared']} scalars; max weight difference "
             f"{res['ident']['max_abs_net_weight_diff']:.1e} / adapter {res['ident']['max_abs_adapter_weight_diff']:.1e}). G0 gate: {'met' if res['g0']['gate'] else '**NOT met**'}.", "",
             "| Line | value | limit | verdict |", "|:--|:--|:--|:-:|",
             f"| (i) W2, navtest turn tokens > 45 deg (n = {nav[TAG]['n']}, warp frames) | {nav[TAG]['W2']} (base pilot {nav[REF]['W2']}, S pilot {nav[SP]['W2']}) | <= {nav[REF]['W2']} + {SPREAD} | {ok('i W2')} |",
             f"| (ii) own-plan agent-contact rate, hold pooled (n = {hb['agent']['n']}) | {rr(hb, 'agent')} | fall >= {100 * KEEP * S_FALL['agent']:.2f} % | {ok('ii agent')} |",
             f"| (ii) own-plan boundary rate, hold pooled | {rr(hb, 'bnd')} | fall >= {100 * KEEP * S_FALL['bnd']:.2f} % | {ok('ii boundary')} |",
             f"| (iii) dev ADE at step 3 000 (m) | {dev[TAG]['ade']:.4f} (S pilot {dev[SP]['ade']:.4f}, base pilot {dev[REF]['ade']:.4f}) | <= {dev[SP]['ade'] + ADE_TOL:.4f} | {ok('iii dev ADE')} |",
             f"| (iii) 4 s arc ratio against the base pilot, hold pooled (n = {arc_h[3]}) | {f(arc_h)} | in [{ARC[0]}, {ARC[1]}] | {ok('iii arc')} |",
             "", f"**Verdict: {verdict}.**", "",
             "| Read (reported) | this pilot | S pilot | base pilot |", "|:--|:--|:--|:--|",
             f"| navtest > 45 deg: W2 / leaves the 4 m tube / mean lateral at 4 s (m, + = outside) | {nav[TAG]['W2']} / {nav[TAG]['L4']} / {nav[TAG]['out4_mean']:+.3f} | "
             f"{nav[SP]['W2']} / {nav[SP]['L4']} / {nav[SP]['out4_mean']:+.3f} | {nav[REF]['W2']} / {nav[REF]['L4']} / {nav[REF]['out4_mean']:+.3f} |",
             f"| the same, archived read (bench plans) | | {archived[SP]['pooled'][b]['W2']} / {archived[SP]['pooled'][b]['L4']} / {archived[SP]['pooled'][b]['out4_mean']:+.3f} | "
             f"{archived[REF]['pooled'][b]['W2']} / {archived[REF]['pooled'][b]['L4']} / {archived[REF]['pooled'][b]['out4_mean']:+.3f} |"]
        for s in ("right", "left"):
            q = [side[f"{t} {s}"] for t in (TAG, SP, REF)]
            M.append(f"| navtest > 45 deg {s} (n = {int(q[0]['n'])}): W2 / tube exits / mean lateral at 4 s | " + " | ".join(
                f"{round(x['wide_2'] * x['n'])} / {round(x['leave_4'] * x['n'])} / {x['out4_mean']:+.3f}" for x in q) + " |")
        M.append(f"| hold on-log turn rows (n = {hold[TAG]['log']['n']}): W2 | " + " | ".join(str(hold[t]['log']['W2']) for t in (TAG, SP, REF)) + " |")
        M.append(f"| hold turn rows, four families (n = {hold[TAG]['pooled']['n']}): W2 / tube exits / mean lateral at 4 s | " + " | ".join(
            f"{hold[t]['pooled']['W2']} / {hold[t]['pooled']['L4']} / {hold[t]['pooled']['out4_mean']:+.3f}" for t in (TAG, SP, REF)) + " |")
        for s in ("hold", "val"):
            for q, lab in (("agent", "agent-contact"), ("bnd", "boundary (NAVSIM raster)"), ("road", "boundary (road-and-lane raster)")):
                p = g3[s][REF]
                if q in p:
                    M.append(f"| {s}: own-plan {lab} rate (n = {p[q]['n']}) | {p[q]['new']:.4f} (fall {100 * p[q]['rel_fall']:.1f} % against the base; against the S pilot "
                             f"{g3[s][SP][q]['diff']:+.4f} [{g3[s][SP][q]['lo']:+.4f}, {g3[s][SP][q]['hi']:+.4f}]) | | {p[q]['base']:.4f} |")
        for s, keys in (("hold", ("all", "open", "lead", "fam log", "fam ot1", "fam yr1", "fam bd4", "> 45 deg")), ("navtest", ("all", "open", "lead", "> 45 deg"))):
            for k in keys:
                if (TAG, k) in arcs[s]:
                    M.append(f"| 4 s arc ratio against the base pilot, {s} {k} (n = {arcs[s][(TAG, k)][3]}) | {f(arcs[s][(TAG, k)])} | {f(arcs[s][(SP, k)])} | 1 |")
        M.append(f"| dev ADE / dev_drift_off | {dev[TAG]['ade']:.4f} / {dev[TAG].get('drift_off', float('nan')):.3f} | {dev[SP]['ade']:.4f} / "
                 f"{dev[SP].get('drift_off', float('nan')):.3f} | {dev[REF]['ade']:.4f} / {dev[REF].get('drift_off', float('nan')):.3f} |")
        for k in SCAL:
            M.append(f"| training scalar `{k}`, mean of steps 2 701-3 000 | {scal[TAG][k]:.4f} | {scal[SP][k]:.4f} | |")
        for s in ("hold", "val"):
            D = G3[s]
            for sub in ("fam log", "fam ot1", "fam yr1", "fam bd4", "> 45 deg"):
                for q in ("agent", "bnd"):
                    x = D[(D.ref == REF) & (D.subset == sub) & (D.rate == q)]
                    if len(x):
                        x = x.iloc[0]
                        M.append(f"| {s} {sub}: {q} rate (n = {int(x.n)}) | {x.new:.4f} (fall {100 * x.rel_fall:.1f} %) | | {x.base:.4f} |")
        (OD / "report.md").write_text("\n".join(M) + "\n")
        run.info("\n" + "\n".join(M))
        run.summary.update(lines=L, verdict=verdict)


if __name__ == "__main__":
    main(argparse.ArgumentParser().parse_args())
