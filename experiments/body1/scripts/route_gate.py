"""BODY1 arm 4.3, prereg Amendment 7 (route hinge, `P2H10R`): band selection, pilot gate and G3, collected from the readers' files. Reads no
checkpoint and runs no model (CPU).

  select  item 2: the two band pilots on the validation part of the train logs -> results/route/select.{json,md}
  pilot   item 3: Amendment 6's pilot lines + the corridor line (C-a, C-b, C-c) for --arm against P2H10-P-s0, with `P2H10S-P-s0` (positive
          control) and "A on on-log rows only" (negative control) from the look's files -> results/route/pilot_gate.{json,md}
  g3      item 3: (a) to (e) + the corridor line, P2H10R-F-s{0..3} against P2H10-F of the same seed -> results/route/g3.{json,md}
Inputs: results/loss/g3_<name>.json (bd4_g3.py), results/route/ol_<name>.json (route_ol.py; counts), $DATA_DIR/runs/body1/prog/ol_<name>.parquet
(prog_ol.py; arc ratios recomputed by shape_gate.arcs), yr_probe_<name>.json (ot3_rows.py probe), the trainers' events.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/route_gate.py select
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import shape_gate as SG  # noqa: E402

OUT = SRC = B.REPO / "experiments/body1/results/route"                 # --out moves the tables; the hold-log inputs stay in SRC
NAV = ""                                                               # --nav: suffix of the navtest dumps and tables (`_w` = warp frames, results/navtest_warp.md)
LOSS = B.REPO / "experiments/body1/results/loss"
REFP, CANDS = "P2H10-P-s0", {1.5: "P2H10R-Pb15-s0", 2.5: "P2H10R-Pb25-s0"}
TOL = 5                                                                # rows / tokens at full scale (item 3)


def J(f):
    return json.loads(_pl.Path(f).read_text())


def last_scalars(tag, keys, lo=2700, hi=3000):
    """Mean of the logged 25-step means of `loss/<key>` over steps (lo, hi] of the newest run of the tag."""
    from jevdrive.common import data_dir
    d = sorted((data_dir() / "runs/op_parity" / f"train-{tag}").iterdir())[-1]
    acc = {k: [] for k in keys}
    for ln in open(d / "events.jsonl"):
        e = json.loads(ln)
        if e.get("kind") == "scalar" and e["tag"].startswith("loss/") and e["tag"][5:] in acc and lo < e["step"] <= hi:
            acc[e["tag"][5:]].append(e["value"])
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


def corridor(name_nav, name_hold, new, ref, tol_log, tol_pool):
    """Corridor line of item 3 for one (new, ref) pair -> dict of counts and verdicts."""
    N, H = J(OUT / f"ol_{name_nav}{NAV}.json")["gate"], J(SRC / f"ol_{name_hold}.json")["gate"]
    b = "> 45 deg"
    q = dict(Ca=(N[new]["pooled"][b]["W2"], N[ref]["pooled"][b]["W2"], TOL), Cb=(H[new]["log"][b]["W2"], H[ref]["log"][b]["W2"], tol_log),
             Cc_navtest=(N[new]["pooled"][b]["L4"], N[ref]["pooled"][b]["L4"], TOL), Cc_hold=(H[new]["pooled"][b]["L4"], H[ref]["pooled"][b]["L4"], tol_pool))
    return dict(counts={k: dict(new=v[0], ref=v[1], tol=v[2]) for k, v in q.items()}, lines={k: bool(v[0] <= v[1] + v[2]) for k, v in q.items()},
                n=dict(navtest=N[new]["pooled"][b]["n"], hold_log=H[new]["log"][b]["n"], hold_pooled=H[new]["pooled"][b]["n"]),
                out4=dict(navtest=(N[new]["pooled"][b]["out4_mean"], N[ref]["pooled"][b]["out4_mean"]), hold=(H[new]["pooled"][b]["out4_mean"], H[ref]["pooled"][b]["out4_mean"])))


def cmd_select(a):
    from jevdrive.run import Run
    with Run("body1", "route-gate-select", config=vars(a)) as run:
        V = J(OUT / "ol_a7_val.json")["gate"]
        dref = SG.dev_of(REFP)
        res = {}
        for band, tag in CANDS.items():
            g = J(LOSS / f"g3_a7_val_b{int(band * 10)}.json")["pooled"][REFP]
            dv = SG.dev_of(tag)
            sh = last_scalars(tag, ["imit/route_pos", "route_ho_pos", "route", "route_ho"])
            L = dict(dev_ade=bool(dv["ade"] <= dref["ade"] + 0.01), drift=bool(dv["drift_off"] <= 0.30), agent_30=bool(g["agent"]["rel_fall"] >= 0.30),
                     bnd_25=bool(g["bnd"]["rel_fall"] >= 0.25), tube_imit=bool(sh["imit/route_pos"] <= 0.05), tube_ho=bool(sh["route_ho_pos"] <= 0.20))
            res[tag] = dict(band=band, dev=dv, agent=g["agent"], bnd=g["bnd"], shares=sh, W2=V[tag]["pooled"]["> 45 deg"]["W2"], n=V[tag]["pooled"]["> 45 deg"]["n"],
                            W2_log=V[tag]["log"]["> 45 deg"]["W2"], out4=V[tag]["pooled"]["> 45 deg"]["out4_mean"], lines=L, eligible=bool(all(L.values())))
        el = sorted((r["W2"], -r["band"], t) for t, r in res.items() if r["eligible"])
        pick = None
        if el:
            pick = el[0][2]
            if len(el) == 2 and abs(el[0][0] - el[1][0]) < 2:              # tie: the wider tube
                pick = CANDS[2.5]
        ctx = {t: dict(W2=V[t]["pooled"]["> 45 deg"]["W2"], W2_log=V[t]["log"]["> 45 deg"]["W2"], out4=V[t]["pooled"]["> 45 deg"]["out4_mean"]) for t in V if t not in res}
        (OUT / "select.json").write_text(json.dumps(dict(ref=REFP, ref_dev=dref, candidates=res, context=ctx, selected=pick, band=res[pick]["band"] if pick else None), indent=1, default=float) + "\n")
        Lm = ["| Validation part, shards s2 + s3 | " + " | ".join(f"`{t}` (B = {r['band']} m)" for t, r in res.items()) + " | " + " | ".join(f"`{t}`" for t in ctx) + " |",
              "|:--|" + "--:|" * (len(res) + len(ctx))]
        row = lambda lab, fn, cf=lambda t: "": Lm.append(f"| {lab} | " + " | ".join(fn(r) for r in res.values()) + " | " + " | ".join(cf(t) for t in ctx) + " |")  # noqa: E731
        row("dev ADE at step 3 000 (m; limit switch-off + 0.01)", lambda r: f"{r['dev']['ade']:.4f}", lambda t: f"{dref['ade']:.4f}" if t == REFP else "")
        row("dev_drift_off (limit 0.30)", lambda r: f"{r['dev']['drift_off']:.3f}")
        row("own-plan agent-contact rate, relative fall (>= 30 %)", lambda r: f"{r['agent']['new']:.4f}, {100 * r['agent']['rel_fall']:.1f} %")
        row("own-plan boundary rate, relative fall (>= 25 %)", lambda r: f"{r['bnd']['new']:.4f}, {100 * r['bnd']['rel_fall']:.1f} %")
        row("route hinge non-zero, imitation rows, steps 2 701-3 000 (<= 5 %)", lambda r: f"{100 * r['shares']['imit/route_pos']:.2f} %")
        row("route hinge non-zero, hinge-only rows (<= 20 %)", lambda r: f"{100 * r['shares']['route_ho_pos']:.2f} %")
        row("eligible", lambda r: "yes" if r["eligible"] else "**no**")
        row(f"W2, turn rows > 45 deg, four families (n = {next(iter(res.values()))['n']})", lambda r: str(r["W2"]), lambda t: str(ctx[t]["W2"]))
        row("W2, on-log turn rows (reported)", lambda r: str(r["W2_log"]), lambda t: str(ctx[t]["W2_log"]))
        row("mean lateral at 4 s, + = outside (m; reported)", lambda r: f"{r['out4']:+.3f}", lambda t: f"{ctx[t]['out4']:+.3f}")
        Lm.append(f"\nSelected: {'`' + pick + '`' if pick else 'none (the variant ends)'}")
        (OUT / "select.md").write_text("\n".join(Lm) + "\n")
        run.info("\n" + "\n".join(Lm))
        run.summary.update(selected=pick)


def lines_a6(g, dv, dref, yr, yref, A):
    ag, bn = g["agent"], g["bnd"]
    return dict(agent_30=bool(ag["rel_fall"] >= 0.30 and ag["hi"] < 0), bnd_25=bool(bn["rel_fall"] >= 0.25 and bn["hi"] < 0), dev_ade=bool(dv["ade"] <= dref["ade"] + 0.01),
                slope=bool(yr[0] <= yref[0] + 0.05), arc_pooled=bool(A["all"][0] >= 0.995), arc_open=bool(A["open"][0] >= 0.995))


def cmd_pilot(a):
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    with Run("body1", "route-gate-pilot", config=vars(a)) as run:
        arm = a.arm
        g = J(LOSS / "g3_a7_hold_R.json")["pooled"][REFP]
        A = {k[1]: v for k, v in SG.arcs("a7_pilot_hold", [(arm, REFP)], REFP).items()}
        Nv = {k[1]: v for k, v in SG.arcs("a7_pilot_navtest" + NAV, [(arm, REFP)], REFP).items()}
        yr = J(data_dir() / "runs/alpasim/ot3/results/yr_probe_b43a7.json")["models"]
        dv, dref = SG.dev_of(arm), SG.dev_of(REFP)
        L = lines_a6(g, dv, dref, yr[arm]["alpha_05"], yr[REFP]["alpha_05"], A)
        C = corridor("a7_pilot_navtest", "a7_pilot_hold", arm, REFP, 1, 1)
        ctrl = {t: corridor("a6_pilot_navtest", "a6_pilot_hold", t, REFP, 1, 1) for t in ("P2H10S-P-s0", "P2H10B-P-Aon-s0")}
        L |= C["lines"]
        met = bool(all(L.values()))
        (OUT / "pilot_gate.json").write_text(json.dumps(dict(arm=arm, ref=REFP, agent=g["agent"], bnd=g["bnd"], road=g.get("road"), dev=dv, ref_dev=dref, alpha_05=yr[arm]["alpha_05"],
                                                              ref_alpha_05=yr[REFP]["alpha_05"], arc_hold=A, arc_navtest=Nv, corridor=C, controls=ctrl, lines=L, gate_met=met), indent=1, default=float) + "\n")
        f = lambda v: f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"  # noqa: E731
        c = lambda r, k: f"{r['counts'][k]['new']} against {r['counts'][k]['ref']} ({'met' if r['lines'][k] else 'NOT met'})"  # noqa: E731
        Lm = [f"| Pilot gate (hold logs and navtest, shards s2 + s3) | `{arm}` against `{REFP}` | line | verdict |", "|:--|:--|:--|:-:|"]
        v = lambda k: "met" if L[k] else "**NOT met**"  # noqa: E731
        for q, lab, k, ln in (("agent", "own-plan agent-contact rate", "agent_30", ">= 30 %, interval excluding 0"), ("bnd", "own-plan boundary rate (NAVSIM raster)", "bnd_25", ">= 25 %, interval excluding 0")):
            Lm.append(f"| {lab} | {g[q]['new']:.4f} against {g[q]['base']:.4f}: {100 * g[q]['rel_fall']:.1f} %, {g[q]['diff']:+.4f} [{g[q]['lo']:+.4f}, {g[q]['hi']:+.4f}] | {ln} | {v(k)} |")
        Lm.append(f"| dev ADE at step 3 000 (m) | {dv['ade']:.4f} against {dref['ade']:.4f} | <= + 0.01 | {v('dev_ade')} |")
        Lm.append(f"| continuation slope alpha_05 | {yr[arm]['alpha_05'][0]:.3f} against {yr[REFP]['alpha_05'][0]:.3f} | <= + 0.05 | {v('slope')} |")
        Lm.append(f"| 4 s arc ratio, hold states pooled (n = {A['all'][3]}) | {f(A['all'])} | >= 0.995 | {v('arc_pooled')} |")
        Lm.append(f"| 4 s arc ratio, hold open states (n = {A['open'][3]}) | {f(A['open'])} | >= 0.995 | {v('arc_open')} |")
        for k, lab in (("Ca", f"(C-a) W2, navtest turn tokens > 45 deg (n = {C['n']['navtest']})"), ("Cb", f"(C-b) W2, on-log hold turn rows (n = {C['n']['hold_log']})"),
                       ("Cc_navtest", "(C-c) leaves the 4 m tube, navtest turn tokens"), ("Cc_hold", f"(C-c) leaves the 4 m tube, hold turn rows, four families (n = {C['n']['hold_pooled']})")):
            Lm.append(f"| {lab} | {C['counts'][k]['new']} against {C['counts'][k]['ref']} | <= reference + {C['counts'][k]['tol']} | {v(k)} |")
            Lm.append(f"| - control `P2H10S-P-s0` (must fail C-a, C-b) / \"A on on-log rows only\" (must pass) | {c(ctrl['P2H10S-P-s0'], k)} / {c(ctrl['P2H10B-P-Aon-s0'], k)} | | |")
        Lm.append(f"| (reported) mean lateral at 4 s on turn rows, + = outside (m): navtest / hold | {C['out4']['navtest'][0]:+.3f} against {C['out4']['navtest'][1]:+.3f} / {C['out4']['hold'][0]:+.3f} against {C['out4']['hold'][1]:+.3f} | | |")
        Lm.append(f"| (reported) 4 s arc ratio, navtest pooled / open | {f(Nv['all'])} / {f(Nv['open'])} | | |")
        Lm.append(f"\nPilot gate: {'met' if met else 'NOT met'}")
        (OUT / "pilot_gate.md").write_text("\n".join(Lm) + "\n")
        run.info("\n" + "\n".join(Lm))
        run.summary.update(gate=met, **L)


def cmd_g3(a):
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    with Run("body1", "route-gate-g3", config=vars(a)) as run:
        sd = a.seeds
        new, ref = [f"P2H10R-F-s{i}" for i in sd], [f"P2H10-F-s{i}" for i in sd]
        Nv = SG.arcs("a7_full_navtest", list(zip(new, ref)), "P2H10-F-s0")
        Hd = SG.arcs("a7_full_hold", list(zip(new, ref)), "P2H10-F-s0")
        yr = J(data_dir() / "runs/alpasim/ot3/results/yr_probe_b43a7full.json")["models"]
        res = {}
        for i, x, y in zip(sd, new, ref):
            ga, gb = J(LOSS / f"g3_a7_full_hold_s{i}.json"), J(LOSS / f"g3_a7_full_navtest_s{i}.json")
            pa, pb = ga["pooled"][y], gb["pooled"][y]
            dv = SG.dev_of(x)
            C = corridor("a7_full_navtest", "a7_full_hold", x, y, TOL, TOL)
            arc = {"navtest pooled >= 0.995": Nv[(x, "all")][0] >= 0.995, "navtest pooled lower bound >= 0.990": Nv[(x, "all")][1] >= 0.990, "navtest open >= 0.995": Nv[(x, "open")][0] >= 0.995,
                   "navtest lead >= 0.990": Nv[(x, "lead")][0] >= 0.990, "hold pooled >= 0.990": Hd[(x, "all")][0] >= 0.990} | {f"hold {f} >= 0.980": Hd[(x, f"fam {f}")][0] >= 0.980 for f in SG.FAMS}
            L = dict(a=bool(ga["verdict"][y]["agent_30"] and ga["verdict"][y]["bnd_30"]), b=bool(gb["verdict"][y]["no_rise_sum_falls"]),
                     c=bool(yr[x]["alpha_05"][0] <= yr[y]["alpha_05"][0] + 0.05), drift=bool(dv["drift_off"] <= 0.30), e=bool(all(arc.values())), **C["lines"])
            res[x] = dict(ref=y, hold=dict(agent=pa["agent"], bnd=pa["bnd"]), navtest=dict(agent=pb["agent"], bnd=pb["bnd"], sum=pb["sum"]), alpha_05=(yr[x]["alpha_05"], yr[y]["alpha_05"]), dev=dv,
                          arc={k: bool(v) for k, v in arc.items()}, arc_v=dict(navtest_all=Nv[(x, "all")], navtest_open=Nv[(x, "open")], navtest_lead=Nv[(x, "lead")], hold_all=Hd[(x, "all")],
                                                                              **{f"hold_{f}": Hd[(x, f"fam {f}")] for f in SG.FAMS}), corridor=C, lines=L, met=bool(all(L.values())))
        met = bool(all(r["met"] for r in res.values()))
        (OUT / "g3.json").write_text(json.dumps(dict(seeds=sd, tags=res, met_without_d=met, note="(d) navtest EPDMS and the turn oracle are read by bd4_g3d.py / jevdrive.bench"), indent=1, default=float) + "\n")
        Lm = ["| G3 item | " + " | ".join(f"seed {i}" for i in sd) + " | line |", "|:--|" + ":--|" * (len(sd) + 1)]
        row = lambda lab, fn, ln="": Lm.append(f"| {lab} | " + " | ".join(fn(res[x]) for x in new) + f" | {ln} |")  # noqa: E731
        ok = lambda r, k: "" if r["lines"][k] else " **NOT met**"  # noqa: E731
        row("(a) hold logs: agent-contact rate, relative fall", lambda r: f"{100 * r['hold']['agent']['rel_fall']:.1f} % [{r['hold']['agent']['lo']:+.4f}, {r['hold']['agent']['hi']:+.4f}]" + ok(r, "a"), ">= 30 %, interval excluding 0")
        row("(a) hold logs: boundary rate, relative fall", lambda r: f"{100 * r['hold']['bnd']['rel_fall']:.1f} % [{r['hold']['bnd']['lo']:+.4f}, {r['hold']['bnd']['hi']:+.4f}]", ">= 30 %, interval excluding 0")
        nt = lambda r, q: f"{round(r['navtest'][q]['new'] * r['navtest'][q]['n'])} against {round(r['navtest'][q]['base'] * r['navtest'][q]['n'])}"  # noqa: E731
        row("(b) navtest on-log: agent-contact tokens", lambda r: nt(r, "agent") + ok(r, "b"), "does not rise")
        row("(b) navtest on-log: boundary tokens", lambda r: nt(r, "bnd"), "does not rise")
        row("(b) navtest on-log: sum, difference [95 %]", lambda r: f"{r['navtest']['sum']['diff']:+.5f} [{r['navtest']['sum']['lo']:+.5f}, {r['navtest']['sum']['hi']:+.5f}]", "falls")
        row("(c) continuation slope", lambda r: f"{r['alpha_05'][0][0]:.3f} against {r['alpha_05'][1][0]:.3f}" + ok(r, "c"), "<= base + 0.05")
        row("dev_drift_off", lambda r: f"{r['dev']['drift_off']:.3f}" + ok(r, "drift"), "<= 0.30")
        f = lambda v: f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"  # noqa: E731
        for k, lab, ln in (("navtest_all", "(e) arc ratio, navtest pooled", ">= 0.995, lower bound >= 0.990"), ("navtest_open", "(e) navtest open", ">= 0.995"), ("navtest_lead", "(e) navtest lead", ">= 0.990"),
                           ("hold_all", "(e) hold pooled", ">= 0.990")) + tuple((f"hold_{x}", f"(e) hold `{x}`", ">= 0.980") for x in SG.FAMS):
            row(lab, lambda r: f(r["arc_v"][k]), ln)
        row("(e) all arc lines", lambda r: "met" if r["lines"]["e"] else "**NOT met**")
        for k, lab in (("Ca", "(C-a) W2, navtest turn tokens > 45 deg"), ("Cb", "(C-b) W2, on-log hold turn rows"), ("Cc_navtest", "(C-c) 4 m tube, navtest turn tokens"), ("Cc_hold", "(C-c) 4 m tube, hold turn rows, four families")):
            row(lab, lambda r: f"{r['corridor']['counts'][k]['new']} against {r['corridor']['counts'][k]['ref']}" + ok(r, k), f"<= base + {TOL}")
        row("(reported) mean lateral at 4 s on turn rows, navtest (m)", lambda r: f"{r['corridor']['out4']['navtest'][0]:+.3f} against {r['corridor']['out4']['navtest'][1]:+.3f}")
        row("**all items above**", lambda r: "**met**" if r["met"] else "**NOT met**")
        Lm.append(f"\nG3 without (d): {'met' if met else 'NOT met'}")
        (OUT / "g3.md").write_text("\n".join(Lm) + "\n")
        run.info("\n" + "\n".join(Lm))
        run.summary.update(g3=met)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "pilot", "g3"])
    ap.add_argument("--arm", default="", help="pilot: the selected pilot tag")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--nav", default="", help="suffix of the navtest dump / table names (_w: the dumps on warp frames)")
    ap.add_argument("--out", default="", help="table directory instead of results/route (the navtest tables of route_ol.py are read from it)")
    a = ap.parse_args()
    NAV = SG.NAV = a.nav
    if a.out:
        OUT = B.REPO / a.out
    {"select": cmd_select, "pilot": cmd_pilot, "g3": cmd_g3}[a.cmd](a)
