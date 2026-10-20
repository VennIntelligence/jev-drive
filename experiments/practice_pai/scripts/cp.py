#!/usr/bin/env python3
"""CURATED-PAI (plans/2026-10-20-curated-pilot-prereg.md): split, roll-stage checks, the pre-training firing count (G0), STOPPT labels and draw
weights on the merged pool, and the closed-loop gate.

  split    (Mac, repo root) registers physical_ai_av/curated-pilot@v1 from results/curated/scenes570.tsv; checks: inside curated-train, disjoint
           from curated-val (= test281 + practice-pai) and test281
  stage    --roots R.. --n N --min-ok K --out FILE [--n3 N3 --done 11 --budget-gb 100 --free-floor-gb 450]   (box) roll stage check: >= K of
           the stage's N scenes scored, frame dumps and rollout.asl present, disk per scene; exit 2 (and $O/VERDICT = STOP ..) on failure.
           With --n3: the stage-3 scene count (<= N3, >= 150) and the window W (chunks holding rollouts at once) that keep the pilot's net
           growth within the budget and the free space above the floor -> FILE with suffix .n3 ("n3 W")
  labels   --pool P --rows DIR --scenes TSV --workers N --out DIR   (box) STOPPT labels of the pool's curated PAI rows (defect_fix stoppt.py's
           PAI rule, rows prefix `cp`) -> <pool>/stoppt.npz (ok only on cp rows: the term never touches the nuPlan rows); per class: rows, STOPPT
           qualifying / firing on the stored plans, agent hinge firing on the stored plans -> DIR/fire.{json,md}; G0 -> exit 2 + VERDICT on failure
  weights  --pool P --base W0 --arm pilot|control --out NAME   (box) JT00S's draw on the merged pool: pilot = the nuPlan S share halved, the other
           half uniform over the cp rows outside the held scenes; control = JT00S with every cp row at weight 0
  gate     --pilot R --control R --ref R --out DIR   (box) the registered lines on the whole PAI 281 -> DIR/gate.{json,md}, DIR/VERDICT, $O/VERDICT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/defect_fix/scripts"), str(R_ / "experiments/defect_fix/lib"),
                str(R_ / "experiments/wait_cap/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = D / "runs/practice_pai/curated"
CR = D / "runs/op_parity/cache"
DS = "physical_ai_av"
TSV = R_ / "experiments/practice_pai/results/curated/scenes570.tsv"
G0_STOP_SHARE, G0_FIRES = 0.066, 2000          # prereg G0: 2 x d386's PAI practice-pool share (848 / 25 634 = 3.3 %), 2.4 x its 848 fires
PREFIX = "cp"


def scenes(p=TSV):
    return list(csv.DictReader(open(p), delimiter="\t"))


def verdict(text):
    O.mkdir(parents=True, exist_ok=True)
    (O / "VERDICT").write_text(text + "\n")
    print(text)


def cmd_split(a):
    from jevdrive.data import splits
    S = scenes()
    ids = [r["scene_id"] for r in S]
    tr, va, te = (splits.load(f"{DS}/{n}") for n in ("curated-train", "curated-val", "test281"))
    assert set(ids) <= set(tr), f"{len(set(ids) - set(tr))} scenes outside curated-train"
    cp = splits.define(DS, "curated-pilot", ids, "scene", "experiments/practice_pai/results/curated/scenes570.tsv: the curated-train NuRec "
                       "scenes with a usdz on the GPU box on 2026-10-20 (ranks 1-570 of the box's nurec_select.csv at download time), classes from "
                       "experiments/pai_av/results/nurec_select.csv (pav.py select)", status="frozen", used_by=["experiments/practice_pai"],
                       notes="CURATED-PAI pilot roll scenes; exam exclusion of pav.py select (same drive within 2 min of a test281 / val160 scene, "
                       "same road); exam_gap_s inf for all 570; PAI-AV drive-group exclusion not applied (Tokyo labels pending)")
    splits.check_disjoint(cp, va)
    splits.check_disjoint(cp, te)
    print(json.dumps(dict(id=cp.id, n=len(cp), classes=Counter(r["cls"] for r in S)), indent=1))


def du_gb(p):
    out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()
    return int(out[0]) / 1e9 if out else 0.0


def cmd_stage(a):
    import pai281 as P
    n_ok, n_dump, n_asl, bad, size = 0, 0, 0, [], 0.0
    want = set()
    for r in a.roots:
        want |= set(" ".join(p.read_text() for p in sorted(Path(r, "chunks").glob("*.txt"))).split())
        J, bj = P.load(r)
        sc = {s: float(J[s].get("score") or 0.0) for s in J if s not in bj}
        n_ok += len(set(sc) & want)
        bad += sorted(want - set(sc))
        n_dump += len(list(Path(r).glob("runs/*/sim/driver-logs/dump/*")))
        n_asl += len(list(Path(r).glob("runs/*/sim/**/rollout.asl")))
        size += du_gb(r)
    per = size / max(len(want), 1)
    rec = dict(roots=a.roots, scenes=len(want), scored=n_ok, missing=bad, dump_entries=n_dump, asl=n_asl, gb=round(size, 2), gb_per_scene=round(per, 3))
    ok = n_ok >= a.min_ok and n_dump > 0 and n_asl > 0 and len(want) == a.n
    if a.n3:        # size stage 3: kept products final(N) = merged pool 12 GB + rows / cache / pool 0.094 GB a scene (d367: 15 GB / 160 scenes),
        # transient = W chunks of 32 scenes holding rollout.asl + dumps; final + transient <= min(budget, free - floor)
        free = shutil.disk_usage(D).free / 1e9
        cap = min(a.budget_gb, free - a.free_floor_gb)
        fin = lambda n: 12.0 + 0.094 * n  # noqa: E731
        chunk = 32 * max(per, 1e-3)
        N, W = a.done + a.n3, int((cap - fin(a.done + a.n3)) // chunk)
        if W < 2:
            W, N = 2, int((cap - 2 * chunk - 12.0) / 0.094)
        n3 = min(a.n3, N - a.done)
        rec |= dict(free_gb=round(free, 1), cap_gb=round(cap, 1), n3=n3, window=W, projected_peak_gb=round(fin(a.done + n3) + W * chunk, 1))
        ok = ok and n3 >= 150
    rec["pass_"] = bool(ok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    if not ok:
        verdict(f"STOP roll stage check {Path(a.out).name}: {json.dumps({k: v for k, v in rec.items() if k != 'missing'})}")
        raise SystemExit(2)
    if a.n3:
        Path(a.out).with_suffix(".n3").write_text(f"{rec['n3']} {rec['window']}\n")


def agent_fire(pool, names, poses):
    """S's agent hinge (margin 0.3, side 0, the trainer's function) on the stored plans -> fires (n,) bool (rows without labels: False)."""
    import torch
    from agent_hinge import AgentHinge
    H = AgentHinge(CR / pool / "agent.npz", names, "cpu", 0.3, 0.0)
    out = np.zeros(len(names), bool)
    P = torch.as_tensor(np.nan_to_num(poses), dtype=torch.float32)
    for i in range(0, len(names), 4096):
        r = torch.arange(i, min(i + 4096, len(names)))
        ok = H.ok[r]
        if ok.any():
            v = H.per_step(P[r][ok, :, 0], P[r][ok, :, 1], P[r][ok, :, 2], r[ok]).amax(1) > 0
            out[r[ok].numpy()] = v.numpy()
    return out


def cmd_labels(a):
    import stoppt as SP
    tb = np.load(CR / a.pool / "tab.npz")
    names, part, speed, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["speed"], tb["log"].astype(str)
    m = part == PREFIX
    i = np.flatnonzero(m)
    res = SP.pai_rows(names[i], speed[i], Path(a.rows), a.workers)
    A = SP.pack(res, len(i))
    rest = np.full(len(names), np.inf, np.float32)
    rest[i] = np.where(np.isfinite(A["rest"]), A["rest"], np.inf)
    ok = np.zeros(len(names), bool)
    ok[i] = A["ok"]
    why = np.full(len(names), "not_cp", "U16")
    why[i] = A["why"]
    np.savez(CR / a.pool / "stoppt.npz", names=tb["names"], rest=rest, v0=speed.astype(np.float32), ok=ok, why=why)
    poses = np.stack([r["poses"] for r in res]).astype(np.float32)
    fire, _ = SP.score(poses, A["rest"], speed[i], A["ok"])
    fire = fire.astype(bool) & A["ok"]
    own = [n.split(":", 2)[2] for n in names[i]]                       # the PAI pool's own names "on:pai:<scene>:<k>"
    afire = agent_fire(a.src_pool, np.array(own), poses)
    cls = {r["scene_id"]: r["cls"] for r in scenes(a.scenes)}
    sc = logs[i]
    held = np.array([zlib.crc32(s.encode()) % 100 < 15 for s in sc])
    turn = np.nan_to_num(A["turn8"]) > 30
    C = np.array([cls.get(s, "?") for s in sc])
    tab = {}
    for c in ["all"] + sorted(set(C)):
        k = (C == c) if c != "all" else np.ones(len(C), bool)
        kd = k & ~held
        tab[c] = dict(scenes=len(set(sc[k])), rows=int(k.sum()), rows_drawn=int(kd.sum()), stoppt_ok=int((A["ok"] & k).sum()),
                      stoppt_fire=int((fire & k).sum()), stoppt_fire_drawn=int((fire & kd).sum()), stoppt_fire_share=float((fire & k).sum() / max(k.sum(), 1)),
                      agent_fire=int((afire & k).sum()), agent_fire_share=float((afire & k).sum() / max(k.sum(), 1)),
                      turn_rows=int((turn & k).sum()), turn_rows_any_fire=int((turn & k & (fire | afire)).sum()),
                      why=dict(Counter(A["why"][k].tolist()).most_common()))
    st = tab.get("stop_nolead", dict(stoppt_fire_share=0.0))
    g0 = dict(stop_share=st["stoppt_fire_share"], line_share=G0_STOP_SHARE, fires_drawn=tab["all"]["stoppt_fire_drawn"], line_fires=G0_FIRES)
    g0["pass_"] = bool(g0["stop_share"] >= G0_STOP_SHARE and g0["fires_drawn"] >= G0_FIRES)
    rec = dict(pool=a.pool, cp_rows=int(m.sum()), held_scenes=int(len(set(sc[held]))), G0=g0, by_class=tab,
               reference_d386_practice_pool=dict(rows=25634, stoppt_ok=3138, stoppt_fire=848, share=0.0331))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fire.json").write_text(json.dumps(rec, indent=1))
    L = [f"# CURATED-PAI firing count before training (G0 {'PASS' if g0['pass_'] else 'FAIL'})", "",
         f"Pool {a.pool}, {int(m.sum())} curated PAI S rows (prefix `{PREFIX}`), NTV0-L18-s0's stored plans (`poses_model`). "
         "STOPPT = defect_fix stoppt.py's PAI rule and the trainer's function (nr1_free); agent = S's agent hinge (margin 0.3, side 0). "
         "drawn = outside the 15 % held scenes (rm_train --on-hold-pct 15).", "",
         "| class | scenes | rows | drawn | STOPPT ok | STOPPT fires | share | fires drawn | agent fires | share | turn rows (turn8 > 30 deg) | turn rows any fire |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for c, t in tab.items():
        L.append(f"| {c} | {t['scenes']} | {t['rows']} | {t['rows_drawn']} | {t['stoppt_ok']} | {t['stoppt_fire']} | {100 * t['stoppt_fire_share']:.1f} % | "
                 f"{t['stoppt_fire_drawn']} | {t['agent_fire']} | {100 * t['agent_fire_share']:.1f} % | {t['turn_rows']} | {t['turn_rows_any_fire']} |")
    L += ["", f"- G0: stop_nolead STOPPT fire share {100 * g0['stop_share']:.2f} % (line >= {100 * G0_STOP_SHARE:.1f} %), drawn STOPPT fires "
          f"{g0['fires_drawn']} (line >= {G0_FIRES}): {'PASS' if g0['pass_'] else 'FAIL'}",
          "- reference (d386): practice pool (RSG-L18-s0 states, 160 scenes) 848 / 25 634 = 3.3 %",
          "- turn rows: no term of this recipe reads the PAI road (no raster); only the agent hinge can fire there (reported, not gated)"]
    (out / "fire.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not g0["pass_"]:
        verdict(f"STOP G0 (firing count before training): {json.dumps(g0)}; see {out / 'fire.md'}")
        raise SystemExit(2)


def cmd_weights(a):
    tb = np.load(CR / a.pool / "tab.npz")
    names, part, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["log"].astype(str)
    w0 = np.load(D / "runs/defect_fix/w" / f"{a.base}.npz")
    n0 = len(w0["names"])
    assert np.array_equal(w0["names"].astype(str), names[:n0]) and (part[n0:] == PREFIX).all(), "merged pool != on1_JT + cp rows"
    w = np.zeros(len(names))
    grp = np.r_[w0["grp"].astype(str), np.full(len(names) - n0, "P")].astype("U12")
    w[:n0] = w0["w"]
    P = np.arange(n0, len(names))
    held = np.array([zlib.crc32(g.encode()) % 100 < 15 for g in logs[P]])     # = rm_train --on-hold-pct 15
    grp[P[held]] = "P_held"
    if a.arm == "pilot":
        S0 = np.isin(grp[:n0], ["S", "S_held"])
        fs = float(w[:n0][S0].sum())
        w[:n0][S0] *= 0.5
        w[P[~held]] = 0.5 * fs / (~held).sum()
    w /= w.sum()
    np.savez(D / "runs/defect_fix/w" / f"{a.out}.npz", names=tb["names"], w=w, grp=grp)
    rec = dict(pool=a.pool, base=a.base, arm=a.arm, rows=len(names), share={g: float(w[grp == g].sum()) for g in np.unique(grp)},
               per_batch_32={g: float(w[grp == g].sum() * 32) for g in np.unique(grp)}, rows_by_group={g: int((grp == g).sum()) for g in np.unique(grp)})
    if a.arm == "control":
        assert np.allclose(w[:n0], w0["w"] / w0["w"].sum(), rtol=0, atol=1e-15), "control: JT00S's weights changed"
    (O / f"w_{a.out}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_gate(a):
    import pai281 as P
    from jevdrive import stats
    (J, bj), (K, bk), (C, bc) = (P.load(r) for r in (a.pilot, a.control, a.ref))
    want = set()
    for f in sorted(Path(a.ref, "chunks").glob("*.txt")):
        want |= set(f.read_text().split())
    sc = lambda X, S: np.array([float(X[s].get("score") or 0.0) for s in S])  # noqa: E731
    S = sorted((want & set(J) & set(K)) - set(bj) - set(bk))
    sj, sk = sc(J, S), sc(K, S)
    p = stats.paired(sj, sk, groups=np.array(S))
    Sr = sorted((set(S) & set(C)) - set(bc))
    rep = {lab: dict(stats.paired(sc(X, Sr), sc(C, Sr), groups=np.array(Sr)), zeros=int((sc(X, Sr) == 0).sum()), ref_zeros=int((sc(C, Sr) == 0).sum()),
                     n=len(Sr)) for lab, X in (("pilot", J), ("control", K))}
    zj, zk = int((sj == 0).sum()), int((sk == 0).sum())
    l1 = p["mean"] >= 0.010
    l2 = rep["pilot"]["mean"] >= 0.0
    l3 = zj <= zk + 2 and rep["pilot"]["zeros"] <= rep["pilot"]["ref_zeros"] + 2
    v = "PASS" if l1 and l2 and l3 else "FAIL"
    G = dict(verdict=v, scenes_in_exam=len(want), scenes=len(S), rpc_excluded=dict(pilot=sorted(bj), control=sorted(bk), ref=sorted(bc)),
             pilot=dict(mean=float(sj.mean()), zeros=zj), control=dict(mean=float(sk.mean()), zeros=zk), paired_pilot_minus_control=p, vs_ntv0=rep,
             L1=f"pilot - control {p['mean']:+.4f} >= +0.010: {l1}", L2=f"pilot - stored NTV0+ACC+GAP {rep['pilot']['mean']:+.4f} >= 0: {l2}",
             L3=f"zeros pilot {zj} <= control {zk} + 2 and <= stored {rep['pilot']['ref_zeros']} + 2: {l3}")
    md = [f"# CURATED-PAI gate: {v}", "", f"PAI 281 (NTV0 + ACC + GAP's 8 chunk lists, {len(want)} scenes); scored by both arms without a driver RPC "
          f"error: {len(S)}. Scene-clustered bootstrap 95 % CI (jevdrive.stats, 10 000 draws, seed 0).", "",
          "| arm | mean | zeros |", "|:--|--:|--:|", f"| pilot (+ curated PAI rows + STOPPT) | {sj.mean():.4f} | {zj} |",
          f"| control (same command, PAI rows weight 0) | {sk.mean():.4f} | {zk} |", "",
          f"- L1 {G['L1']} (CI {stats.fmt(p, '+.4f')})", f"- L2 {G['L2']} (CI {stats.fmt(rep['pilot'], '+.4f')}, n {rep['pilot']['n']})", f"- L3 {G['L3']}",
          f"- reported: control - stored NTV0+ACC+GAP {stats.fmt(rep['control'], '+.4f')}, zeros {rep['control']['zeros']} vs {rep['control']['ref_zeros']}",
          "", "Per-class table: pai281_full.md; stop-point overrun zeros: overrun.json; PCS ruler: on the Mac (pai_pcs_local.py), not gated."]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(G, indent=1, default=float))
    (out / "gate.md").write_text("\n".join(md) + "\n")
    (out / "VERDICT").write_text(v + "\n")
    verdict(f"{v} (gate {out / 'gate.md'})")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("split")
    p = sp.add_parser("stage")
    p.add_argument("--roots", nargs="+", required=True), p.add_argument("--n", type=int, required=True), p.add_argument("--min-ok", type=int, required=True)
    p.add_argument("--out", required=True), p.add_argument("--n3", type=int, default=0), p.add_argument("--done", type=int, default=11)
    p.add_argument("--budget-gb", type=float, default=100.0), p.add_argument("--free-floor-gb", type=float, default=450.0)
    p = sp.add_parser("labels")
    for k in ("--pool", "--src-pool", "--rows", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--scenes", default=str(TSV)), p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("weights")
    p.add_argument("--pool", required=True), p.add_argument("--base", default="JT00S"), p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True, choices=("pilot", "control"))
    p = sp.add_parser("gate")
    for k in ("--pilot", "--control", "--ref", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
