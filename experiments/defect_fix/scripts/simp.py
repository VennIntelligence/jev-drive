#!/usr/bin/env python3
"""SIMP (block B, plans/2026-10-20-lon-anchor-prereg.md supplement 2): the offline gate of the merged S-row hinges (lib/nr1_simp.py). Box, repo
root, a GPU pool job (the trainer's own setup builds every raster and box tensor).

  $TR simp.py gate -- <rm_train train args of a SIMP arm>   rm_train's setup (on_train.setup + nr1_lat load / attach, captured; no training),
        then on the S rows: the human's path (tab `fut`, the imitation target from the visited state) and the stored plans (lon_anchor
        labels_v2.npz `poses`) through region / clear and the four trainer terms they replace (road hinge, lat_onc, agent hinge, lat_side)
        -> results/lon_anchor/simp_gate.{json,md}; exit 2 when a registered line fails
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts"), str(REPO / "experiments/op_adapt_r2/lib"), str(REPO / "experiments/op_parity/scripts"),
                str(REPO / "experiments/alpasim/lib"), str(REPO / "experiments/alpasim/scripts"), str(REPO / "experiments/body1/lib"),
                str(REPO / "experiments/onpol_nuplan/lib"), str(REPO / "experiments/onpol_nuplan/scripts"), str(REPO / "experiments/route_mem/scripts"),
                str(REPO / "experiments/defect_fix/lib"), str(HERE)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RES = REPO / "experiments/defect_fix/results/lon_anchor"
LINE_REGION, LINE_CLEAR, ACT = 0.02, 0.02, 0.05      # supplement 2: human-path active share <= 2 % each; active = max per-step violation > 0.05 m


class _Stop(Exception):
    pass


def capture(argv):
    """Run rm_train's __main__ with argv up to on_train.setup's return -> (ns, pack, attach kwargs)."""
    import on_train as OT
    import nr1_lat
    CAP = {}
    setup0, attach0 = OT.setup, nr1_lat.attach

    def setup(a, dev):
        CAP["ns"], CAP["a"] = setup0(a, dev), a
        raise _Stop

    def attach(pack, **k):
        CAP["pack"], CAP["k"] = pack, k
        return attach0(pack, **k)
    OT.setup, nr1_lat.attach = setup, attach
    sys.argv = [str(REPO / "experiments/route_mem/scripts/rm_train.py")] + argv
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except _Stop:
        pass
    OT.setup, nr1_lat.attach = setup0, attach0
    assert "pack" in CAP and "ns" in CAP, "setup did not reach nr1_lat.attach (SIMP flags on?)"
    return CAP["ns"], CAP["pack"], CAP["k"]


def evaluate(poses, rows, ns, pack, on0, bs=2048):
    """Plans (n, 8, 3) of pool rows -> dict of per-row max-step violations (n,): region, its road / onc parts, clear, its lead part, and the
    replaced trainer terms: road (Losses43: relu(0.3 - sdf) at the footprint corners), onc (lat_onc), agent (agent hinge 0.3 / side 0),
    side (lat_side 0.4)."""
    import torch
    import nr1_lat
    import nr1_simp
    from loss43 import shape_only
    dev = pack["dev"]
    keys = ("region", "region_road", "region_onc", "clear", "clear_lead", "old_road", "old_onc", "old_agent", "old_side")
    out = {k: np.zeros(len(rows), np.float32) for k in keys}
    ah = pack["ah"]
    for i in range(0, len(rows), bs):
        r = torch.as_tensor(rows[i:i + bs], device=dev)
        P = torch.as_tensor(np.nan_to_num(poses[i:i + bs]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            v, vr, vo = nr1_simp.region_steps(P, r, pack, parts=True)
            out["region"][i:i + bs], out["region_road"][i:i + bs], out["region_onc"][i:i + bs] = (t.amax(1).cpu().numpy() for t in (v, vr, vo))
            cok = pack["clear_ok"][r]
            vc, vl = nr1_simp.clear_steps(P[..., 0], P[..., 1], P[..., 2], r, ah, parts=True)
            out["clear"][i:i + bs], out["clear_lead"][i:i + bs] = (vc.amax(1) * cok).cpu().numpy(), (vl.amax(1) * cok).cpu().numpy()
            xs, ys = shape_only(P[..., 0], P[..., 1], P[..., 2])
            rk = pack["road_ok"][r]
            m = torch.relu(nr1_simp.ROAD_M - pack["road"].margins(xs, ys, P[..., 2], r + on0))
            out["old_road"][i:i + bs] = (m.amax(1) * rk).cpu().numpy()
            if "onc" in pack:
                d = nr1_lat.dense(P, pack["M"])
                out["old_onc"][i:i + bs] = (torch.relu(-nr1_lat.region_m(d, r, pack, "onc")).amax(1) * pack["ok"][r]).cpu().numpy()
                out["old_side"][i:i + bs] = ((0.4 - nr1_lat.side_dist(d, r, ah)).clamp_min(0).amax(1) * ah.ok[r]).cpu().numpy()
            out["old_agent"][i:i + bs] = (ns.agent2.per_step(xs, ys, P[..., 2], r + on0).amax(1) * cok).cpu().numpy()
    return out


def cmd_gate(argv):
    import torch
    ns, pack, k = capture(argv)
    on0 = k["on0"]
    a_ = argv[argv.index("--on") + 1]
    tab = np.load(DATA / "runs/op_parity/cache" / a_ / "tab.npz")
    part = tab["pool"].astype(str)
    S = pack["self"].cpu().numpy()
    src = np.where(part == "cp", "pai", "nuplan")
    lab = np.load(DATA / "runs/defect_fix/lon_anchor/labels_v2.npz")
    assert np.array_equal(lab["names"], tab["names"]), "labels_v2 rows != the pool's"
    fut = tab["fut"].astype(np.float32)
    hum_rows = np.flatnonzero(S & np.isfinite(fut).all((1, 2)))
    st_rows = np.flatnonzero(S & np.isfinite(lab["poses"]).all((1, 2)))
    H = evaluate(fut[hum_rows], hum_rows, ns, pack, on0)
    P = evaluate(lab["poses"][st_rows], st_rows, ns, pack, on0)
    rk, ck = pack["road_ok"].cpu().numpy(), pack["clear_ok"].cpu().numpy()
    oc = pack["ok"].cpu().numpy() if "onc" in pack else np.zeros_like(rk)
    okm = dict(region=rk | oc, region_road=rk, region_onc=oc, clear=ck, clear_lead=ck, old_road=rk, old_onc=oc, old_agent=ck,
               old_side=pack["ah"].ok.cpu().numpy())

    def share(D, rows, key, s_):
        m = okm[key][rows] & (src[rows] == s_ if s_ != "all" else True)
        return dict(rows=int(m.sum()), active=float((D[key][m] > ACT).mean()) if m.any() else 0.0)
    rep = {nm: {key: {s_: share(D, rows, key, s_) for s_ in ("all", "nuplan", "pai")} for key in D}
           for nm, D, rows in (("human", H, hum_rows), ("stored", P, st_rows))}
    g = dict(region=rep["human"]["region"]["all"]["active"], clear=rep["human"]["clear"]["all"]["active"], line_region=LINE_REGION, line_clear=LINE_CLEAR)
    g["pass_"] = bool(g["region"] <= LINE_REGION and g["clear"] <= LINE_CLEAR)
    res = dict(gate=g, argv=argv, attach={kk: (int(v.sum()) if torch.is_tensor(v) else None) for kk, v in pack.items() if kk in ("self", "road_ok", "clear_ok")},
               tables=rep, note="active = max per-step violation > 0.05 m; human = tab fut (the imitation target from the visited state); stored = "
               "nuPlan rows' served plans, cp rows' NTV0-L18-s0 model plans; clear_lead = the lead part (its along-heading gradient may only shorten "
               "the plan): on the stored plans this is the brake-gradient trigger rate")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "simp_gate.json").write_text(json.dumps(res, indent=1))
    pc = lambda x: f"{100 * x:.2f} %"  # noqa: E731
    L = [f"# SIMP offline gate (supplement 2): {'PASS' if g['pass_'] else 'FAIL'}", "",
         f"Human path (tab fut) as the plan: region active {pc(g['region'])} (line <= {pc(LINE_REGION)}), clear active {pc(g['clear'])} (line <= {pc(LINE_CLEAR)}).",
         "", "| plans | term | all rows | all active | nuPlan rows | nuPlan active | PAI rows | PAI active |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for nm in ("human", "stored"):
        for key, v in rep[nm].items():
            L.append(f"| {nm} | {key} | {v['all']['rows']} | {pc(v['all']['active'])} | {v['nuplan']['rows']} | {pc(v['nuplan']['active'])} | "
                     f"{v['pai']['rows']} | {pc(v['pai']['active'])} |")
    L += ["", res["note"]]
    (RES / "simp_gate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not g["pass_"]:
        raise SystemExit(2)


LB_RISE, LB_FOL = 0.10, 0.05                         # supplement 3: BRK clear rise vs the control >= 0.10 (CI low > 0); FOL change within 0.05


def cmd_leadread(arm, ctl, ref):
    """lead_break.py cf --name simp on RB0-F's own PAI crash states -> the supplement 3 rule: the share of plans that stay behind the lead
    over 4 s (as served, arm F) on BRK, arm vs control (and vs the SIMP arm without flattening), FOL held -> results/lon_anchor/simp_leadbreak.*"""
    import pandas as pd
    import inertia as IN
    O = DATA / "runs/defect_fix/lead_break/simp"
    g = json.load(open(O / "gate.json"))
    M = pd.read_parquet(O / "rows.parquet").set_index("i")
    C = pd.read_parquet(O / "cf.parquet")
    C["clear"] = np.where(np.isfinite(C.margin), (C.margin >= 0).astype(float), np.nan)
    W = C.pivot_table(index="i", columns=["tag", "arm"], values="clear", dropna=False)
    out, L = dict(gate=g, arm=arm, ctl=ctl, ref=ref), [f"# SIMP-F lead-break read (supplement 3; lead_break.py cf on RB0-F-s0's PAI crash states)", "",
                                                       f"G0: {'PASS' if g['pass_'] else 'FAIL'} (p99 {g['gate_p99']:.4f} m, rows {g['rows']})", ""]
    p3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
    for win in ("BRK", "FOL"):
        m = M[M.win == win]
        ids, sc = m.index.values, m.scene.values
        X = W.reindex(ids)
        col = lambda t, x="F": X[(t, x)].values if (t, x) in X.columns else np.full(len(ids), np.nan)  # noqa: E731
        cell = dict(rows=len(m), scenes=int(m.scene.nunique()), clear={})
        for t, x in [(t, "F") for t in sorted({c[0] for c in X.columns})] + [("RB0-F-s0", "OFF")]:
            v = col(t, x)
            k = np.isfinite(v)
            cell["clear"][f"{t}/{x}"] = IN.boot(lambda i: float(np.nanmean(v[i][k[i]])), sc)
        def pair(t1, t0):
            u, v = col(t1), col(t0)
            k = np.isfinite(u) & np.isfinite(v)
            return IN.boot(lambda i: float(np.mean(u[i][k[i]] - v[i][k[i]])), sc)
        cell["arm-ctl"], cell["arm-ref"], cell["ref-ctl"] = pair(arm, ctl), pair(arm, ref), pair(ref, ctl)
        off, c0 = cell["clear"]["RB0-F-s0/OFF"][0], cell["clear"][f"{ctl}/F"][0]
        cell["gap_closed"] = float(cell["arm-ctl"][0] / max(off - c0, 1e-6))
        out[win] = cell
        L += [f"## {win}: {len(m)} decisions in {cell['scenes']} scenes", "", "| model / arm | stays behind the lead (95 % scene CI) |", "|:--|--:|"]
        L += [f"| {k} | {p3(v)} |" for k, v in cell["clear"].items()]
        L += ["", f"- {arm} - {ctl}: {p3(cell['arm-ctl'])}; {arm} - {ref}: {p3(cell['arm-ref'])}; {ref} - {ctl}: {p3(cell['ref-ctl'])}; "
              f"share of the gap to RB0-F OFF closed: {cell['gap_closed']:.2f}", ""]
    b, f = out["BRK"]["arm-ctl"], out["FOL"]["arm-ctl"]
    out["pass_"] = bool(g["pass_"] and b[0] >= LB_RISE and b[1] > 0 and abs(f[0]) <= LB_FOL)
    L += [f"Rule: BRK {arm} - {ctl} >= +{LB_RISE} with CI low > 0, and |FOL {arm} - {ctl}| <= {LB_FOL}: {'PASS' if out['pass_'] else 'FAIL'}"]
    (RES / "simp_leadbreak.json").write_text(json.dumps(out, indent=1, default=float))
    (RES / "simp_leadbreak.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "leadread":
        cmd_leadread(*sys.argv[2:5])
        raise SystemExit(0)
    assert sys.argv[1] == "gate" and sys.argv[2] == "--", "usage: simp.py gate -- <rm_train train args> | leadread ARM CTL REF"
    cmd_gate(sys.argv[3:])
