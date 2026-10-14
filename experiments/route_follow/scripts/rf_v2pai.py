#!/usr/bin/env python3
"""Route v2 on PAI 281 (Tokyo) and the turn-gain switch diagnostic: served-memory check, per-scene turn reads, paired reports. Plain python3 +
numpy (Tokyo), stored run roots read, never written. Runner: rf_v2pai.sh.

  served --run <pai_eval root | pai_run dir> [--order 0|1]   every logged decision's `rmem` against rmem.Tracker rebuilt from the session's
         logged route records (ts, pose, waypoints) at the logged anchor: max |served - offline|, mask agreement (offline rebuild of the
         served memory). --order (default: the width of the logged rmem, 203 = ordered): for the ordered layout also where it puts the memory's
         start (-s_e, progress anchor) against the projection layout's (-s_geo): median / p90 |difference|, decisions >= 5 m apart, and
         the first valid slot of either layout.
  turns  --pkl LABEL=<rf_pai_follow extract pkl>[:<pkl label>] ... --scenes <file> [--tg LABEL=<run root>] --out <json>
         per (scene, label): rf_pai_follow.one (strict turn on the label's own AlpaSim route: entry / exit, completed, outcome), the zero kind
         and the ego's route station at the zero event relative to the entry, rf_shape.track_shape / speeds on [entry - 40, exit] and the
         decision 293 shape gate (rf_p4.gate_of with the human topo class of plan_strict36.json, results/p3_branch/shape.json; checked on
         completed turns), and with --tg the turn-gain switch's decisions (drive.jsonl `tg`): count on, first on (route station vs entry).
  tgcheck --run <root> --ref <identity root>   the turn-gain switch as logged: on / off recomputed with turn_gain.decide from the logged rmem
         and plan (5 / 4-decimal logs: near-threshold flips are listed), served = branch_poses(plan, g) on every on-decision, and every plan
         before a session's first on-decision equal to the identity's at the same k (the switch changes nothing until it fires)
  table  --turns <json> --a <identity label> --b <label> [--strict <file>] [--exam <file>] --md <md>   paired per-scene table and counts
         (completions on the strict turns, on the exam turn scenes; new zeros on the non-turn part = b zero, a not, the event before the entry
         or the scene has no strict turn; offroad ones among them: the turn-gain stop rule).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]
sys.path[:0] = [str(HERE), str(R / "experiments/route_mem/lib")]
RES = HERE.parent / "results"


def drive_recs(root):
    """{session: (scene, [route records], [inference records])} of every driver log under a pai_eval root or a pai_run dir."""
    out = {}
    fs = glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")) + glob.glob(os.path.join(root, "driver", "drive.jsonl"))
    for f in fs:
        for line in open(f):
            if '"route"' not in line and '"infer": true' not in line:
                continue
            r = json.loads(line)
            s = out.setdefault(r["session"], [r.get("scene", ""), [], []])
            if r.get("kind") == "route":
                s[1].append(r)
            elif r.get("kind") == "drive" and r.get("infer"):
                s[0] = r.get("scene") or s[0]
                s[2].append(r)
    return out


def served(a):
    import rmem as RM
    D = drive_recs(a.run)
    n = nd = nm = 0
    dmax, gap, first = 0.0, [], []
    for se, (sc, routes, recs) in sorted(D.items(), key=lambda x: x[1][0]):
        recs = [r for r in recs if "rmem" in r]
        if not recs:
            continue
        order = (len(recs[0]["rmem"]) == RM.DIM_O) if a.order is None else bool(a.order)
        tr, pose = RM.Tracker(order=order), {int(r["ts"]): np.array(r["pose"], float) for r in routes}
        for r in routes:
            tr.add(int(r["ts"]), r["wp"])
        dm, mm = 0.0, 0
        for r in recs:
            t0, an = int(r["t0"]), np.array(r["anchor"], float)
            f = tr.features(t0, pose.get, an)
            s = np.asarray(r["rmem"], float)
            dm = max(dm, float(np.abs(s - f).max()))
            Ps, Po = RM.pts(s), RM.pts(f)
            mm += int(((Ps[:, 0] > 0.5) != (Po[:, 0] > 0.5)).any())
            if order:                                                   # where the memory starts: progress anchor vs projection
                B, Q, last = [], [], None
                for ts in sorted(tr.routes):
                    if ts > t0 or ts not in pose:
                        continue
                    wp = tr.routes[ts][np.isfinite(tr.routes[ts]).all(1)]
                    Q.append(pose[ts][:2])
                    if len(wp):
                        B.append(RM.to_local(pose[ts], wp))
                        last = pose[ts]
                if B:
                    info = {}
                    RM.memory_o(B, an, last, RM.travelled(Q, an), info)
                    gap.append(info["s_geo"] - info["s_e"])
                    v = Ps[:, 0] > 0.5
                    vg = RM.window(RM.in_ego(an, RM.merge(B)))[:, 0] > 0.5
                    first.append((float(RM.OFFS[v.argmax()]) if v.any() else np.nan, float(RM.OFFS[vg.argmax()]) if vg.any() else np.nan))
        n, nd, nm, dmax = n + 1, nd + len(recs), nm + mm, max(dmax, dm)
        print(f"{sc[7:15]} session {se[:8]}: {len(recs)} decisions, layout {'ordered' if order else 'old'} ({len(recs[0]['rmem'])}), "
              f"max |served - offline| {dm:.2e}, mask differs at {mm}")
    print(f"served memory = offline rebuild: {n} sessions, {nd} decisions, max |diff| {dmax:.2e} (logged rounding 1e-5), mask differs at {nm}")
    if gap:
        g, F = np.abs(np.array(gap)), np.array(first)
        both = np.isfinite(F).all(1)
        print(f"ordered start vs projection start ({len(g)} decisions): median |diff| {np.median(g):.2f} m, p90 {np.quantile(g, 0.9):.2f} m, "
              f">= 5 m at {int((g >= 5).sum())}; first valid slot ordered / projection differ at {int((F[both, 0] != F[both, 1]).sum())} "
              f"(ordered further at {int((F[both, 0] > F[both, 1]).sum())})")


def tg_stats(root):
    """{scene: {n_on, n_dec, k_first, t0_first}} from drive.jsonl `tg` (turn_gain.py)."""
    out = {}
    for se, (sc, _, recs) in drive_recs(root).items():
        on = [r for r in recs if (r.get("tg") or {}).get("on")]
        out[sc] = dict(n_on=len(on), n_dec=len(recs), n_tg=sum("tg" in r for r in recs), k_first=on[0]["k"] if on else None,
                       t0_first=on[0]["t0"] if on else None)
    return out


def turns(a):
    import rf_pai_follow as F
    import rf_p4 as P4
    import rf_shape as SH
    want = [x.split()[0] for x in open(a.scenes) if x.strip()]
    J = json.loads((RES / "p3_branch" / "shape.json").read_text())
    topo = {k: v["topo"] for k, v in json.loads((RES / "turn_gallery" / "plan_strict36.json").read_text())["scenes"].items()}
    tg = {k: tg_stats(v) for k, v in (x.split("=", 1) for x in a.tg)}
    rows = []
    for spec in a.pkl:
        lab, rest = spec.split("=", 1)
        f, src = (rest.rsplit(":", 1) + [lab])[:2] if ":" in rest else (rest, lab)
        D = {sc["scene"]: sc for sc in pickle.load(open(f, "rb"))["scenes"]}
        for s in want:
            o = D.get(s, {}).get("rolls", {}).get(src)
            if o is None or o.get("ctrl") is None:
                rows.append(dict(label=lab, scene=s, missing=True))
                continue
            r, _, _ = F.one(D[s], lab, o, zeros=False)
            C = o["ctrl"]
            Rr = F.dedup(o["route"])
            Sr = F.stations(Rr)
            P = np.c_[C["x"], C["y"]]
            t = np.asarray(C["t"], float)
            st, _ = F.track(Rr, Sr, P)
            tz = o["tz"] if o["tz"] is not None and not o["score"] else np.inf
            iz = int(np.searchsorted(t, tz)) if np.isfinite(tz) else None
            r.update(missing=False, st_zero=float(st[min(iz, len(st) - 1)]) if iz is not None else None, topo=topo.get(s))
            if r.get("turn"):
                shp = SH.track_shape(t, P, C["vx"], st, r["entry"] - SH.WIN_PRE, r["exit"], tz)
                spd = SH.speeds(t, st, C["vx"], r["entry"], r["exit"], tz)
                gt = P4.gate_of(topo.get(s, "other"), J)
                gk = dict(kmax=bool(shp["kmax"] <= gt["kmax"]), dk=bool(shp["dk"] <= gt["dk"]), alat=bool(shp["alat"] <= gt["alat"]),
                          v_entry=bool(np.isfinite(spd["v_0"]) and spd["v_0"] <= gt["v_entry"]))
                r.update(kmax=shp["kmax"], dk=shp["dk"], alat=shp["alat"], v_0=spd["v_0"], gate=gk, shape_ok=bool(r["completed"] and all(gk.values())),
                         z_rel_entry=(r["st_zero"] - r["entry"]) if r["st_zero"] is not None else None)
            if lab in tg:
                q = tg[lab].get(s, {})
                r["tg"] = q
                if q.get("t0_first") is not None and r.get("turn"):
                    r["tg_first_rel_entry"] = float(st[min(int(np.searchsorted(t, q["t0_first"])), len(st) - 1)] - r["entry"])
            rows.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()})
    Path(a.out).write_text(json.dumps(dict(scenes=want, rows=rows), indent=1, default=float))
    print(len(rows), "rows ->", a.out)


def tgcheck(a):
    sys.path.insert(0, str(R / "experiments/alpasim/lib"))
    import turn_gain as TG
    ref = {sc: {r["k"]: r["poses"] for r in recs} for sc, _, recs in drive_recs(a.ref).values()}
    n = flips = 0
    dmax = pre_max = 0.0
    pre_n = 0
    for se, (sc, _, recs) in sorted(drive_recs(a.run).items(), key=lambda x: x[1][0]):
        recs = [r for r in recs if "tg" in r]
        k_on = next((r["k"] for r in recs if r["tg"]["on"]), None)
        for r in recs:
            n += 1
            d = TG.decide(r.get("rmem"), np.array(r["poses"]))
            if d["on"] != r["tg"]["on"]:
                flips += 1
                print(f"  flip {sc[7:15]} k {r['k']}: logged {r['tg']}, recomputed {d}")
            if r["tg"]["on"]:
                dmax = max(dmax, float(np.abs(TG.branch_poses(np.array(r["poses"]), r["tg"]["g"]) - np.array(r["poses_branch"])).max()))
            if (k_on is None or r["k"] <= k_on) and r["k"] in ref.get(sc, {}):
                pre_n += 1
                pre_max = max(pre_max, float(np.abs(np.array(r["poses"]) - np.array(ref[sc][r["k"]])).max()))
        print(f"{sc[7:15]}: {len(recs)} decisions, switch on at {sum(r['tg']['on'] for r in recs)}, first k {k_on}")
    print(f"tgcheck: {n} decisions, on/off recomputed differs at {flips}; served vs branch_poses(plan, g) max |diff| {dmax:.1e} (4-decimal log); "
          f"plans up to the first on-decision vs identity: {pre_n} compared, max |diff| {pre_max:.1e}")


def fmt(x, f="{:+.1f}"):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def table(a):
    T = json.loads(Path(a.turns).read_text())
    by = {(r["label"], r["scene"]): r for r in T["rows"]}
    strict = [x.split()[0] for x in open(a.strict) if x.strip()] if a.strict else T["scenes"]
    exam = {x.split()[0] for x in open(a.exam) if x.strip()} if a.exam else set()
    A, B = a.a, a.b
    L = [f"# {B} vs {A}: per scene", "", f"Scenes {len(T['scenes'])}; strict turns = {len(strict)} (the identity's strict turns); exam turn scenes {len(exam)}.", ""]
    hdr = "| scene | set | side | deg | " + f"{A} | {B} | {B} where (m rel. entry) | {B} v_0 | {B} shape gate | switch on / dec (first, m rel. entry) |"
    L += [hdr, "|:--|:--|:--|--:|:--|:--|:--|--:|:--|:--|"]
    cnt = dict(a_done=0, b_done=0, a_shape=0, b_shape=0, gained=[], lost=[], new_zero_nonturn=[], new_offroad_pre=[], a_zero=0, b_zero=0)
    for s in T["scenes"]:
        ra, rb = by.get((A, s), {}), by.get((B, s), {})
        if ra.get("missing", True) or rb.get("missing", True):
            L.append(f"| {s[7:15]} | missing {'a' if ra.get('missing', True) else 'b'} | | | | | | | | |")
            continue
        st = ("strict" if s in strict else "") + ("+exam" if s in exam else "")
        oa = (ra.get("outcome") or "no turn") + (f" ({ra['kind']})" if ra.get("kind") else "")
        ob = (rb.get("outcome") or "no turn") + (f" ({rb['kind']})" if rb.get("kind") else "")
        za, zb = ra["score"] == 0, rb["score"] == 0
        cnt["a_zero"] += za
        cnt["b_zero"] += zb
        ent = ra.get("entry") if ra.get("turn") else rb.get("entry")
        zrel = (rb["st_zero"] - ent) if (zb and rb.get("st_zero") is not None and ent is not None) else None
        if s in strict:
            cnt["a_done"] += bool(ra.get("completed"))
            cnt["b_done"] += bool(rb.get("completed"))
            cnt["a_shape"] += bool(ra.get("shape_ok"))
            cnt["b_shape"] += bool(rb.get("shape_ok"))
            if rb.get("completed") and not ra.get("completed"):
                cnt["gained"].append(s[7:15])
            if ra.get("completed") and not rb.get("completed"):
                cnt["lost"].append(s[7:15])
        if zb and not za and (ent is None or zrel is None or zrel < 0):
            cnt["new_zero_nonturn"].append(s[7:15])
            if rb.get("kind") == "offroad":
                cnt["new_offroad_pre"].append(s[7:15])
        q = rb.get("tg") or {}
        sw = f"{q.get('n_on', '')} / {q.get('n_tg', '')}" + (f" ({fmt(rb.get('tg_first_rel_entry'))})" if rb.get("tg_first_rel_entry") is not None else "")
        g = rb.get("gate")
        gs = "" if not g else ("pass" if rb.get("shape_ok") else ("not completed" if not rb.get("completed") else "fail: " + ",".join(k for k, v in g.items() if not v)))
        L.append(f"| {s[7:15]} | {st} | {ra.get('side') or rb.get('side') or ''} | {fmt(ra.get('turn_deg') or rb.get('turn_deg'), '{:+.0f}')} | {oa} | {ob} | "
                 f"{fmt(zrel)} | {fmt(rb.get('v_0'), '{:.1f}')} | {gs} | {sw} |")
    ex = [s for s in T["scenes"] if s in exam and not by.get((A, s), {}).get("missing", True) and not by.get((B, s), {}).get("missing", True)]
    exd = lambda lab: sum(bool(by.get((lab, s), {}).get("completed")) for s in ex)  # noqa: E731
    exturn = sum(bool(by.get((A, s), {}).get("turn")) for s in ex)
    sa = np.array([by.get((A, s), {}).get("score", np.nan) for s in T["scenes"]], float)
    sb = np.array([by.get((B, s), {}).get("score", np.nan) for s in T["scenes"]], float)
    m = np.isfinite(sa) & np.isfinite(sb)
    x = sb[m] - sa[m]
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1) if len(x) else np.array([np.nan])
    S = ["", "## Counts", "",
         f"- strict turns completed: {A} {cnt['a_done']} / {len(strict)}, {B} {cnt['b_done']} / {len(strict)}; gained {len(cnt['gained'])} "
         f"({', '.join(cnt['gained'])}), lost {len(cnt['lost'])} ({', '.join(cnt['lost'])})",
         f"- decision 293 shape gate (completed and kappa / dkappa / a_lat / entry speed within the class gate): {A} {cnt['a_shape']}, {B} {cnt['b_shape']}",
         f"- exam turn scenes: {len(ex)} ({exturn} with a strict turn on the identity's route): completed {A} {exd(A)}, {B} {exd(B)}",
         f"- zeros over the {int(m.sum())} scenes: {A} {cnt['a_zero']}, {B} {cnt['b_zero']}; mean score {A} {np.nanmean(sa):.4f}, {B} {np.nanmean(sb):.4f}, "
         f"paired {x.mean():+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]",
         f"- new zeros on the non-turn part ({B} zero, {A} not, event before the entry or no strict turn): {len(cnt['new_zero_nonturn'])} "
         f"({', '.join(cnt['new_zero_nonturn'])}); offroad among them (stop rule >= 3): {len(cnt['new_offroad_pre'])}"]
    Path(a.md).write_text("\n".join(L[:4] + S + [""] + L[4:]) + "\n")
    J = {k: v for k, v in cnt.items()}
    J.update(strict_n=len(strict), exam_n=len(ex), exam_done_a=exd(A), exam_done_b=exd(B), mean_a=float(np.nanmean(sa)), mean_b=float(np.nanmean(sb)),
             paired=float(x.mean()) if len(x) else None)
    Path(a.md).with_suffix(".json").write_text(json.dumps(J, indent=1))
    print("\n".join(S))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("served")
    s.add_argument("--run", required=True)
    s.add_argument("--order", type=int, default=None)
    s = sp.add_parser("turns")
    s.add_argument("--pkl", nargs="+", required=True)
    s.add_argument("--scenes", required=True)
    s.add_argument("--tg", nargs="*", default=[])
    s.add_argument("--out", required=True)
    s = sp.add_parser("tgcheck")
    s.add_argument("--run", required=True)
    s.add_argument("--ref", required=True)
    s = sp.add_parser("table")
    s.add_argument("--turns", required=True)
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--strict", default="")
    s.add_argument("--exam", default="")
    s.add_argument("--md", required=True)
    a = ap.parse_args()
    {"served": served, "turns": turns, "tgcheck": tgcheck, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
