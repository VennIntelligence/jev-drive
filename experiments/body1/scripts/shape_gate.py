"""BODY1 arm 4.3, prereg Amendment 6: the pilot gate (item 5, notes (b), (c)) and G3 (e) (item 6, note (d)) of the shape-only arm, collected
from the readers' files. Reads no checkpoint and runs no model (CPU).

  pilot   contact lines, dev ADE and slope of Amendment 5 item 4 + the arc lines, P2H10S-P-s0 against the switch-off pilot P2H10-P-s0; the
          positive control P2H10B-Pw3-s0 and the two drop-one ablations next to it (they gate nothing) -> results/shape/pilot_gate.{json,md}
  g3e     arc lines of G3 (e), P2H10S-F-s{0,1} against P2H10-F-s{0,1}, with P2H10B-F's values from the diagnosis files -> results/shape/g3e.{json,md}
Inputs: results/loss/g3_<name>.json (bd4_g3.py), $DATA_DIR/runs/body1/prog/ol_<name>.parquet (prog_ol.py states; ratios are recomputed here
from the per-state arcs so that no rounded table decides a line), $DATA_DIR/runs/alpasim/ot3/results/yr_probe_<name>.json, the trainers' events.

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/shape_gate.py pilot
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

OUT = B.REPO / "experiments/body1/results/shape"
LOSS = B.REPO / "experiments/body1/results/loss"
ARM, REFP, CTRL, NOA, AON = "P2H10S-P-s0", "P2H10-P-s0", "P2H10B-Pw3-s0", "P2H10B-Pw3-noA-s0", "P2H10B-P-Aon-s0"
G3P = {ARM: "a6_hold_S", CTRL: "a5_hold_w3", NOA: "a6_hold_noA", AON: "a6_hold_Aon"}
FAMS = ("log", "ot1", "yr1", "bd4")


def arcs(name, pairs, ref0):
    """{(pair new tag, subset): (ratio, lo, hi, n)} from the per-state parquet of prog_ol.py; groups from the plan of ref0 (prog_ol.group)."""
    import pandas as pd
    import prog_ol as PO
    from jevdrive import stats
    M = pd.read_parquet(B.root() / "prog" / f"ol_{name}.parquet")
    g, logs = PO.group(M, ref0), M.log.to_numpy()
    subs = [("all", np.ones(len(M), bool))] + [(q, g == q) for q in PO.GROUPS] + [(f"fam {f}", (M.fam == f).to_numpy()) for f in M.fam.unique()]
    subs += [("> 45 deg", (M.dyaw > 45).to_numpy()), ("v < 1", (M.v0 < 1).to_numpy())]
    subs += [(f"fam {f} | {q}", ((M.fam == f).to_numpy() & (g == q))) for f in M.fam.unique() for q in ("open", "lead")] if M.fam.nunique() > 1 else []
    out = {}
    for x, y in pairs:
        for lab, m in subs:
            if m.sum() < 20:
                continue
            r = stats.paired(M[f"arc4|{x}"].to_numpy()[m], M[f"arc4|{y}"].to_numpy()[m], groups=logs[m])
            b = r["mean_b"]
            out[(x, lab)] = (r["mean_a"] / b, 1 + r["lo"] / b, 1 + r["hi"] / b, int(m.sum()))
    return out


def dev_of(tag):
    from jevdrive.common import data_dir
    d = sorted((data_dir() / "runs/op_parity" / f"train-{tag}").iterdir())[-1]
    o = {}
    for ln in open(d / "events.jsonl"):
        e = json.loads(ln)
        if e.get("kind") == "scalar" and e["tag"].startswith("dev/"):
            o[e["tag"][4:]] = e["value"]
    return o


def cmd_pilot(a):
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    with Run("body1", "shape-gate-pilot", config=vars(a)) as run:
        tags = [ARM, CTRL, NOA, AON]
        g3 = {t: json.loads((LOSS / f"g3_{G3P[t]}.json").read_text())["pooled"][REFP] for t in tags}
        A = arcs("a6_pilot_hold", [(t, REFP) for t in tags], REFP)
        N = arcs("a6_pilot_navtest", [(t, REFP) for t in tags], REFP)
        yr = json.loads((data_dir() / "runs/alpasim/ot3/results/yr_probe_b43a6.json").read_text())["models"]
        dv = {t: dev_of(t) for t in tags + [REFP]}
        res = {}
        for t in tags:
            ag, bn = g3[t]["agent"], g3[t]["bnd"]
            L = dict(agent_30=bool(ag["rel_fall"] >= 0.30 and ag["hi"] < 0), bnd_25=bool(bn["rel_fall"] >= 0.25 and bn["hi"] < 0),
                     dev_ade=bool(dv[t]["ade"] <= dv[REFP]["ade"] + 0.01), slope=bool(yr[t]["alpha_05"][0] <= yr[REFP]["alpha_05"][0] + 0.05),
                     arc_pooled=bool(A[(t, "all")][0] >= 0.995), arc_open=bool(A[(t, "open")][0] >= 0.995))
            res[t] = dict(agent=ag, bnd=bn, road=g3[t].get("road"), dev=dv[t], alpha_05=yr[t]["alpha_05"], lines=L, contact_lines=bool(L["agent_30"] and L["bnd_25"]),
                          arc_lines=bool(L["arc_pooled"] and L["arc_open"]), gate=bool(all(L.values())),
                          arc_hold={k[1]: v for k, v in A.items() if k[0] == t}, arc_navtest={k[1]: v for k, v in N.items() if k[0] == t})
        ref = dict(dev=dv[REFP], alpha_05=yr[REFP]["alpha_05"])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "pilot_gate.json").write_text(json.dumps(dict(arm=ARM, ref=REFP, control=CTRL, ablations=[NOA, AON], reference=ref, tags=res,
                                                              gate_met=res[ARM]["gate"]), indent=1, default=float) + "\n")
        nm = {ARM: "shape-only arm", CTRL: "all terms, full gradient (positive control)", NOA: "B + C without A", AON: "A on on-log rows only"}
        f = lambda v: f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"  # noqa: E731
        Lm = ["| | " + " | ".join(f"`{t}` ({nm[t]})" for t in tags) + f" | `{REFP}` |", "|:--|" + "--:|" * (len(tags) + 1)]
        row = lambda lab, fn, r="": Lm.append(f"| {lab} | " + " | ".join(fn(t) for t in tags) + f" | {r} |")  # noqa: E731
        for q, lab in (("agent", "agent-contact rate"), ("bnd", "boundary rate (NAVSIM raster)"), ("road", "boundary rate (item C's raster)")):
            row(f"own-plan {lab}: new, relative fall, difference [95 %]",
                lambda t: f"{res[t][q]['new']:.4f}, {100 * res[t][q]['rel_fall']:.1f} %, {res[t][q]['diff']:+.4f} [{res[t][q]['lo']:+.4f}, {res[t][q]['hi']:+.4f}]", f"{res[ARM][q]['base']:.4f}")
        row("dev ADE at step 3 000 (m)", lambda t: f"{dv[t]['ade']:.4f}", f"{dv[REFP]['ade']:.4f}")
        row("dev_drift_off", lambda t: f"{dv[t]['drift_off']:.3f}", f"{dv[REFP]['drift_off']:.3f}")
        row("continuation slope alpha_05", lambda t: "{:.3f} [{:.3f}, {:.3f}]".format(*yr[t]["alpha_05"]), "{:.3f} [{:.3f}, {:.3f}]".format(*yr[REFP]["alpha_05"]))
        for lab in ["all"] + ["open", "lead", "near obj", "near edge", "contact"] + [f"fam {x}" for x in FAMS] + ["fam bd4 | open", "> 45 deg", "v < 1"]:
            if (ARM, lab) in A:
                row(f"4 s arc ratio, hold: {lab} (n = {A[(ARM, lab)][3]})", lambda t: f(A[(t, lab)]), "1")
        for lab in ["all", "open", "lead", "near obj", "near edge", "contact", "> 45 deg", "v < 1"]:
            if (ARM, lab) in N:
                row(f"4 s arc ratio, navtest: {lab} (n = {N[(ARM, lab)][3]})", lambda t: f(N[(t, lab)]), "1")
        for q in res[ARM]["lines"]:
            row(f"line `{q}`", lambda t: "met" if res[t]["lines"][q] else "NOT met")
        row("**gate**", lambda t: "**met**" if res[t]["gate"] else "**NOT met**")
        (OUT / "pilot_gate.md").write_text("\n".join(Lm) + "\n")
        run.info("\n" + "\n".join(Lm))
        run.summary.update(gate=res[ARM]["gate"], **res[ARM]["lines"])


def cmd_g3e(a):
    from jevdrive.run import Run
    with Run("body1", "shape-gate-g3e", config=vars(a)) as run:
        sd = a.seeds                                                  # note (h): seeds 2, 3 are a supplementary read (groups still from P2H10-F-s0's plan)
        new, ref, old = tuple(f"P2H10S-F-s{i}" for i in sd), tuple(f"P2H10-F-s{i}" for i in sd), ("P2H10B-F-s0", "P2H10B-F-s1")
        Nv = arcs(f"{a.name}_navtest", list(zip(new, ref)) + [(ref[1], ref[0])], "P2H10-F-s0")
        Hd = arcs(f"{a.name}_hold", list(zip(new, ref)) + [(ref[1], ref[0])], "P2H10-F-s0")
        r01 = ("P2H10-F-s0", "P2H10-F-s1")
        No, Ho = arcs("navtest", list(zip(old, r01)), r01[0]), arcs("hold", list(zip(old, r01)), r01[0])
        spec = [("navtest on-log pooled >= 0.995", Nv, "all", 0.995, None), ("navtest on-log pooled, lower bound >= 0.990", Nv, "all", None, 0.990),
                ("navtest open >= 0.995", Nv, "open", 0.995, None), ("navtest lead >= 0.990", Nv, "lead", 0.990, None), ("hold pooled >= 0.990", Hd, "all", 0.990, None)]
        spec += [(f"hold `{x}` >= 0.980", Hd, f"fam {x}", 0.980, None) for x in FAMS]
        res, Lm = {}, [f"| G3 (e) line | seed {sd[0]} | seed {sd[1]} | `P2H10B-F` seed 0 / 1 | base seed 1 / seed 0 | verdict |", "|:--|:--|:--|:--|:--|:-:|"]
        f = lambda v: f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"  # noqa: E731
        for lab, T, sub, pt, lb in spec:
            v = [T[(t, sub)] for t in new]
            ok = [bool(x[0] >= pt) if pt is not None else bool(x[1] >= lb) for x in v]
            O = No if T is Nv else Ho
            res[lab] = dict(s0=v[0], s1=v[1], met=ok, old=[O[(t, sub)][0] for t in old], floor=T[(ref[1], sub)][0])
            Lm.append(f"| {lab} | {f(v[0])} | {f(v[1])} | {O[(old[0], sub)][0]:.4f} / {O[(old[1], sub)][0]:.4f} | {T[(ref[1], sub)][0]:.4f} | {'met' if all(ok) else '**NOT met**'} |")
        extra = {}
        for T, nm in ((Nv, "navtest"), (Hd, "hold")):
            for sub in ("near obj", "near edge", "contact", "> 45 deg", "v < 1", "fam bd4 | open", "fam bd4 | lead"):
                if (new[0], sub) in T:
                    extra[f"{nm}: {sub}"] = [T[(t, sub)] for t in new]
                    Lm.append(f"| (reported) {nm}: {sub} | {f(T[(new[0], sub)])} | {f(T[(new[1], sub)])} | | {T[(ref[1], sub)][0]:.4f} | |")
        met = all(all(r["met"]) for r in res.values())
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{a.outname}.json").write_text(json.dumps(dict(lines=res, reported=extra, met=met), indent=1, default=float) + "\n")
        (OUT / f"{a.outname}.md").write_text("\n".join(Lm) + f"\n\nG3 (e): {'met' if met else 'NOT met'}\n")
        run.info("\n" + "\n".join(Lm) + f"\nG3 (e): {'met' if met else 'NOT met'}")
        run.summary.update(g3e=met)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "g3e"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--name", default="a6_full", help="prog_ol.py file stem without the set")
    ap.add_argument("--outname", default="g3e")
    a = ap.parse_args()
    {"pilot": cmd_pilot, "g3e": cmd_g3e}[a.cmd](a)
