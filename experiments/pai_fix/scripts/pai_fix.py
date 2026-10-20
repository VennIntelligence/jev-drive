#!/usr/bin/env python3
"""PAI-FIX: the targeted PAI exam (stop overshoot + route turns, small launch / lead guard). Plan: ../plans/2026-10-20-pai-fix-prereg.md.

  facts   (box, repo .venv)  pai_fix.py facts --pkl <rf_tax pkl> --runs <read root>... --out <csv>
          per scene: score, failure class (decision 381's rules: rf_tax.classify, pai281.ptype), the human's facts from the log
          (pai_gap.human / rest_cue: handoff speed, stop, ends still, lead at the rest point, turn), driver RPC flag
  select  (Mac)  pai_fix.py select --facts <val160 facts csv>... --out experiments/pai_fix/scenes
          the exam lists from the registered rules (selection rules below), written once and committed before any exam read
  report  pai_fix.py report --read NAME=<facts csv> ... [--ref NAME] --out <md>      per-subset metrics of exam reads (paired against --ref)
  calib   (Mac)  pai_fix.py calib --inv tmp/pai0/inv.json --gap <pai_gap scenes.csv> --out <dir>
          stored full PAI 281 reads: does the 281-derived exam (the same rules on test281) rank checkpoints like the full read; noise floor

Failure classes of a scene (one each): overshoot = zero, corridor past the end of the log while the human ends still (cor_route_end);
route-turn = zero, corridor at a route turn (cor_straight_at_turn / cor_turn_under / cor_turn_over / cor_wrong_exit); offroad = zero offroad
(junction / curve / straight); AF-front; side = AF-side / AF-rear; zero-other (drift, lane change, other); partials (0 < s < 0.98): rear-hit,
slow (slow / creep / no-launch), hit. Launch scenes (the human is below 1 m/s at the handoff and the log moves > 5 m): launched = the ego
covers >= 2 m and >= 30 % of the log's distance.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/defect_fix/scripts"),
                str(REPO / "experiments/wait_cap/scripts")]

TURN_SUB = ("cor_straight_at_turn", "cor_turn_under", "cor_turn_over", "cor_wrong_exit")
ZCLS = ["overshoot", "route-turn", "offroad", "AF-front", "side", "zero-other"]
PCLS = ["rear-hit", "slow", "hit"]
OK = 0.98


def fclass(score, ptype, primary):
    if score >= OK:
        return "ok"
    if score > 0:
        return {"creep": "slow", "no-launch": "slow"}.get(ptype, ptype if ptype in PCLS else "hit")
    if primary == "cor_route_end":
        return "overshoot"
    if primary in TURN_SUB:
        return "route-turn"
    if ptype.startswith("AF-front"):
        return "AF-front"
    if ptype.startswith("AF-"):
        return "side"
    if "offroad" in ptype:
        return "offroad"
    return "zero-other"


# ---------------------------------------------------------------- facts (box)
def _facts_job(job):
    import pai_gap as PG
    s, o = job
    out, _ = PG.scene_job((s, o, [], None))
    m = o["metrics"] or {}
    out.update(dist=float(m.get("dist_traveled_m") or 0.0), gt_dist=float(m.get("gt_dist_traveled_m") or 0.0))
    return out


def cmd_facts(a):
    import pai_gap as PG
    bad = set()
    for r in a.runs:
        bad |= PG.summaries(r)[1]
    A = {o["scene"]: o for o in pickle.load(open(a.pkl, "rb"))["rolls"] if "ctrl" in o}
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(_facts_job, sorted(A.items()), chunksize=4))
    T = pd.DataFrame(rows).set_index("scene")
    T["rpc"] = T.index.isin(bad)
    T["fclass"] = [fclass(s, p, q) for s, p, q in zip(T.score, T.ptype, T.primary.fillna(""))]
    T.to_csv(a.out)
    print(len(T), "scenes,", int(T.rpc.sum()), "RPC ->", a.out, dict(Counter(T.fclass)))


# ---------------------------------------------------------------- shared metrics
def launch_mask(F):
    return (F.v_hand < 1.0) & (F.gt_dist > 5.0)


def launched(F):
    return (F.dist >= 2.0) & (F.dist >= 0.3 * F.gt_dist)


def metrics(F):
    """F: facts rows of one read on one subset -> dict of the registered metrics."""
    c = Counter(F.fclass)
    L = launch_mask(F)
    return dict(n=len(F), mean=float(F.score.mean()) if len(F) else np.nan, zeros=int((F.score == 0).sum()),
                **{f"z_{k}": c[k] for k in ZCLS}, **{f"p_{k}": c[k] for k in PCLS},
                launch=f"{int((L & launched(F)).sum())}/{int(L.sum())}")


# ---------------------------------------------------------------- selection rules (registered in the prereg)
N_GUARD = 6                         # scenes per guard class


def target_rules(H, nurec):
    """H: the human's facts per scene (h_end_still, rest_lead, v_hand); nurec: exam_nurec.csv rows (turn_deg / turn_vmed over 6.4 s).
    -> Series 'stop' / 'stopL' / 'turn' / '' per scene. STOP: the human is moving at the handoff (>= 1 m/s) and ends the clip still with
    no logged vehicle within 15 m ahead of its rest point (decision 381's overshoot population); STOP-L: the same with a vehicle there.
    TURN: neither, and the human turns >= 25 deg within 6.4 s while moving (median speed > 1 m/s over that window; pav.py's
    turn_mid | turn_sharp)."""
    N = nurec.reindex(H.index)
    st = H.h_end_still.astype(bool) & (H.v_hand >= 1.0)
    lead = H.rest_lead.astype(bool)
    turn = ~st & (N.turn_vmed > 1.0) & (N.turn_deg >= 25.0)
    return pd.Series(np.where(st & ~lead, "stop", np.where(st, "stopL", np.where(turn, "turn", ""))), index=H.index)


def crc(s):
    import zlib
    return zlib.crc32(s.encode())


def guard_rules(H, tr, passed):
    """Guard candidates (not a target scene, scored >= 0.98 by every family read): G-LAUNCH = the human is below 1 m/s at the handoff and
    the log moves > 5 m; G-LEAD = moving at the handoff, not ending still, a logged vehicle in the human's path within 30 m on >= 30 % of
    its moving 0.5 s steps (pai_gap.human lead_share). N_GUARD each, by crc32 of the scene id."""
    free = (tr == "") & passed
    L = launch_mask(H) & free
    D = (H.v_hand >= 1.0) & ~H.h_end_still.astype(bool) & (H.lead_share >= 0.3) & free & ~L
    pick = lambda m: sorted(H.index[m], key=crc)[:N_GUARD]  # noqa: E731
    return pick(L), pick(D)


def cmd_select(a):
    NU = pd.read_csv(a.nurec).set_index("scene_id")
    F = {Path(f).stem: pd.read_csv(f).set_index("scene") for f in a.facts}
    H = F[Path(a.human).stem] if a.human else next(iter(F.values()))
    cm = pd.read_csv(a.clip_map).set_index("scene_id")
    H = H[cm.group.reindex(H.index) == "val160"]
    tr = target_rules(H, NU)
    S = pd.DataFrame({k: x.score.reindex(H.index) for k, x in F.items()})
    passed = (S >= OK).all(axis=1)
    gl, gd = guard_rules(H, tr, passed)
    sub = pd.Series("", index=H.index, dtype=object)
    sub[tr == "stop"], sub[tr == "stopL"], sub[tr == "turn"] = "STOP", "STOP-L", "TURN"
    sub[gl], sub[gd] = "G-LAUNCH", "G-LEAD"
    V = H.loc[sub != "", ["v_hand", "h_end_still", "rest_lead", "rest_lead_gap", "lead_share", "stratum"]].assign(subset=sub[sub != ""])
    V["turn6"], V["turn6_vmed"] = NU.turn_deg.reindex(V.index), NU.turn_vmed.reindex(V.index)
    V["group"], V["family_mean"], V["family_fail"] = "val160", S.mean(axis=1).reindex(V.index), [
        ",".join(sorted({F[k].fclass.get(s, "") for k in F} - {"ok", ""})) for s in V.index]
    # T281: the scenes A0 (NTV0-L18-s0 + ACC + GAP, d366 / d381) zeroes as overshoot or at a turn / junction / curve; selection-biased
    G = pd.read_csv(a.gap).set_index("scene")
    sb = G["sub"].fillna("")
    t_stop, t_turn = G.index[sb == "cor_route_end"], G.index[sb.str.startswith(("cor_straight", "cor_turn", "cor_wrong", "off_junc", "off_curve"))]
    T = pd.DataFrame(index=list(t_stop) + list(t_turn)).assign(subset=["T281-STOP"] * len(t_stop) + ["T281-TURN"] * len(t_turn), group="test281")
    T["a0_sub"] = sb.reindex(T.index)
    for c in ("v_hand", "h_end_still", "rest_lead", "rest_lead_gap", "lead_share", "stratum"):
        T[c] = G[c].reindex(T.index)
    X = pd.concat([V, T])
    # chunks: val160 (v) and T281 (t) apart, scenes in crc32 order, split into near-equal chunks of <= CHUNK; s000 = 1-scene smoke (val160 STOP)
    X["chunk"] = ""
    for g, p in (("val160", "v"), ("test281", "t")):
        ids = sorted(X.index[X.group == g], key=crc)
        k = -(-len(ids) // a.chunk)
        for i, part in enumerate(np.array_split(np.array(ids, dtype=object), k)):
            X.loc[list(part), "chunk"] = f"{p}{i:03d}"
    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    for c, idx in X.groupby("chunk").groups.items():
        (out / "chunks" / f"{c}.txt").write_text("\n".join(sorted(idx, key=crc)) + "\n")
    smoke = sorted(X.index[X.subset == "STOP"], key=crc)[0]
    (out / "chunks" / "s000.txt").write_text(smoke + "\n")
    X.index.name = "scene"
    X.sort_values(["group", "subset"]).to_csv(out / "exam.csv", float_format="%.3f")
    (out / "exam_scenes.txt").write_text("\n".join(sorted(X.index)) + "\n")
    print(X.groupby(["group", "subset"]).size().to_string(), "\n", X.groupby("chunk").size().to_dict(), "smoke", smoke)


def cmd_drives(a):
    """Curated-train scenes on the same drive as an exam val160 scene (pav.py's rule: log session clock within 600 s and the start-to-start
    distance drivable at 30 m/s + 0.3 km) or on the same road (a track point within 10 m) -> train_exclude.csv."""
    cm = pd.read_csv(a.clip_map)
    ex = [s for s in open(a.exam).read().split() if s in set(cm.scene_id[cm.group == "val160"])]
    tr = cm.scene_id[cm.group == "train1761"].tolist()
    def ld(s):
        f = Path(a.meta) / f"{s}.json"
        if not f.exists():
            return None
        m = json.load(open(f))
        return dict(ss=((m.get("time_range") or {}).get("start", np.nan)) * 1e-6, la=np.asarray(m["lat"], float), lo=np.asarray(m["lon"], float))
    E = {s: ld(s) for s in ex}
    miss = [s for s, v in E.items() if v is None]
    E = {s: v for s, v in E.items() if v is not None}
    names = list(E)
    km = lambda la1, lo1, la2, lo2: 6371 * np.hypot(np.radians(la2 - la1), np.radians(lo2 - lo1) * np.cos(np.radians(la1)))  # noqa: E731
    Es = np.array([E[n]["ss"] for n in names])
    El, Eo = np.array([E[n]["la"][0] for n in names]), np.array([E[n]["lo"][0] for n in names])
    rows = []
    for s in tr:
        r = ld(s)
        if r is None:
            rows.append(dict(scene_id=s, why="no meta"))
            continue
        d = km(r["la"][0], r["lo"][0], El, Eo)
        ds = np.abs(Es - r["ss"])
        hit = (ds < 600) & (d < 0.3 + 0.03 * ds)
        for j, n in enumerate(names):                 # one row per (train scene, exam scene) pair
            place = bool(d[j] < 5.0 and (km(r["la"][::10, None], r["lo"][::10, None], E[n]["la"][None, ::10], E[n]["lo"][None, ::10]) < 0.010).any())
            if hit[j] or place:
                rows.append(dict(scene_id=s, exam_scene=n, why="+".join((["session"] if hit[j] else []) + (["same_place"] if place else [])),
                                 gap_s=float(ds[j]) if hit[j] else np.nan))
    X = pd.DataFrame(rows)
    X.to_csv(a.out, index=False, float_format="%.1f")
    print(len(ex), "exam val160 scenes,", len(miss), "without meta;", X.scene_id.nunique(), "curated-train scenes flagged in", len(X), "pairs:",
          X.why.value_counts().to_dict() if len(X) else {})


# ---------------------------------------------------------------- calibration on stored PAI 281 reads (Mac)
B = "/root/autodl-tmp/ujs/runs/"
CAL = {  # full PAI 281 reads of checkpoints under ACC + GAP (the PAI submission serve), CHUNK table of the stored reads
    "RSG-s0": "route_follow/acc_gap/read/s2", "JTN-s0": "defect_fix/pai_accgap/JTN-L18-s0", "JTN-s1": "defect_fix/pai_accgap/JTN-L18-s1",
    "NTV0-s0": "defect_fix/pai_accgap/NTV0-L18-s0", "NTV0-s1": "defect_fix/pai_accgap/NTV0-L18-s1", "NTVX-s0": "defect_fix/pai_accgap/NTVX-L18-s0",
    "NTVXS-s0": "defect_fix/pai_accgap/NTVXS-L18-s0", "BNC2K-s1": "defect_fix/pai_accgap/BNC2K-L18-s1", "JTXS-s1": "defect_fix/pai_accgap/JTXS-L18-s1",
    "B-s0": "defect_fix/stand_b/pai281/STANDFMX-L18-s0", "B-s1": "defect_fix/stand_b/pai281/STANDFMX-L18-s1",
    "B2-s0": "defect_fix/stand_b/pai281/STANDFMXH-L18-s0", "B2-s1": "defect_fix/stand_b/pai281/STANDFMXH-L18-s1",
    "CPAIC1K": "practice_pai/curated/read/CPAIC1K-L18-s0", "CPAIP1K": "practice_pai/curated/read/CPAIP1K-L18-s0"}
EXTRA = {"NTV0-s0 rerun (R0)": "alpasim_proxy/pai_partials/R0", "NTV0-s0-hl": "defect_fix/pai_accgap/NTV0-L18-s0-hl",
         "NTV0-s0 native (N0)": "route_follow/pai_input_loop/N0", "JTN-s0 native": "defect_fix/pai_nohold/JTN-L18-s0",
         "RSG-s0 native (g10)": "route_follow/pai_g10/g10", "RSG-s0 ACC": "route_follow/pai_g10/acc"}
SEEDS = [("NTV0-s0", "NTV0-s1"), ("JTN-s0", "JTN-s1"), ("B-s0", "B-s1"), ("B2-s0", "B2-s1")]
NATIVE = [("NTV0-s0", "NTV0-s0 native (N0)"), ("JTN-s0", "JTN-s0 native"), ("RSG-s0", "RSG-s0 native (g10)")]


def inv_frame(inv, root):
    """inv.json entry -> facts-like frame (score, fclass without the corridor sub-type, gt / dist)."""
    v = inv[B + root]
    R = v["R"]
    F = pd.DataFrame({s: dict(score=x["score"], ptype=x["t"], gt_dist=float(x["m"].get("gt_dist_traveled_m") or 0),
                              dist=float(x["m"].get("dist_traveled_m") or 0)) for s, x in R.items()}).T
    F["score"] = F.score.astype(float)
    F[["gt_dist", "dist"]] = F[["gt_dist", "dist"]].astype(float)
    F["rpc"] = F.index.isin(v["bad"])
    return F


def cmd_calib(a):
    from scipy.stats import kendalltau, spearmanr
    from jevdrive import stats
    inv = json.load(open(a.inv))
    G = pd.read_csv(a.gap).set_index("scene")
    NU = pd.read_csv(a.nurec).set_index("scene_id")
    tr = target_rules(G, NU)
    A0sub = G["sub"].fillna("")
    a0fail = A0sub.map(lambda x: "stop" if x == "cor_route_end" else "turn" if x.startswith(("cor_straight", "cor_turn", "off_junc", "off_curve")) else "")
    reads = {k: inv_frame(inv, r) for k, r in {**CAL, **EXTRA}.items()}
    common = sorted(set.intersection(*[set(F.index[~F.rpc]) for F in reads.values()]) & set(G.index))
    S = pd.DataFrame({k: F.score.reindex(common) for k, F in reads.items()})
    sub = tr.reindex(common)
    fl = a0fail.reindex(common)
    sets = {"STOP": sub == "stop", "STOP-L": sub == "stopL", "TURN": sub == "turn", "TARGET (STOP + STOP-L + TURN)": sub != "",
            "complement (rest of 281)": sub == "",
            "T281 (A0's 46 overshoot / turn zeros)": fl != "", "full": pd.Series(True, index=common)}
    L = ["# PAI-FIX calibration on stored PAI 281 reads (generated by pai_fix.py calib)", "",
         f"Scenes clean (no driver RPC error) in all {len(reads)} reads: {len(common)} of 281. 281-derived exam = the registered rules on test281: "
         f"STOP {int(sets['STOP'].sum())}, STOP-L {int(sets['STOP-L'].sum())}, TURN {int(sets['TURN'].sum())}. T281 = the 46 scenes A0 (NTV0-s0 + ACC + GAP) zeroes as overshoot (23) or "
         f"turn / junction (23); on the common set {int(sets['T281 (A0' + chr(39) + 's 46 overshoot / turn zeros)'].sum())}.", "",
         "## Means per read (ACC + GAP checkpoints, then serve / rerun rows)", "",
         "| read | " + " | ".join(sets) + " |", "|---|" + "--:|" * len(sets)]
    for k in S:
        L.append(f"| {k} | " + " | ".join(f"{S.loc[m, k].mean():.3f}" for m in sets.values()) + " |")
    cal = list(CAL)
    full = S.loc[:, cal].mean()
    L += ["", f"## Rank agreement over the {len(cal)} ACC + GAP checkpoints", "",
          "| subset | Spearman vs full | Kendall vs full | Spearman vs complement | pair sign agreement with full (pairs with abs(full diff) >= 0.01) |",
          "|---|--:|--:|--:|--:|"]
    comp = S.loc[sets["complement (rest of 281)"], cal].mean()
    for name, m in sets.items():
        if name == "full":
            continue
        x = S.loc[m, cal].mean()
        agree = [np.sign(x[i] - x[j]) == np.sign(full[i] - full[j]) for n, i in enumerate(cal) for j in cal[n + 1:] if abs(full[i] - full[j]) >= 0.01]
        L.append(f"| {name} | {spearmanr(x, full)[0]:+.2f} | {kendalltau(x, full)[0]:+.2f} | {spearmanr(x, comp)[0]:+.2f} | "
                 f"{np.mean(agree):.2f} ({sum(agree)}/{len(agree)}) |")
    L += ["", "## Noise floor: seed pairs and the A0 rerun (paired diff, scene bootstrap 95 % CI; zero flips = scenes zero in one, scored in the other)", "",
          "| pair | subset | n | diff | zero flips | scenes that differ by > 0.01 |", "|---|---|--:|--:|--:|--:|"]
    for x, y in SEEDS + [("NTV0-s0", "NTV0-s0 rerun (R0)")]:
        for name in ("STOP", "STOP-L", "TURN", "TARGET (STOP + STOP-L + TURN)", "T281 (A0's 46 overshoot / turn zeros)", "full"):
            m = sets[name]
            u, v = S.loc[m, x].values, S.loc[m, y].values
            p = stats.paired(u, v)
            L.append(f"| {x} - {y} | {name} | {int(m.sum())} | {stats.fmt(p, '+.3f')} | {int(((u == 0) != (v == 0)).sum())} | {int((abs(u - v) > 0.01).sum())} |")
    # T281 column: how many of A0's zeros does another read score (> 0) -> the chance level of "fixes"
    L += ["", "## T281: of A0's overshoot / turn zeros, how many does each read score > 0 (fixed) and how many A0-scored target scenes it zeroes", "",
          "| read | overshoot fixed (of A0's) | turn fixed (of A0's) | new zeros on TARGET (A0 scored) |", "|---|--:|--:|--:|"]
    a0 = S["NTV0-s0"]
    for k in S:
        fs = int(((fl == "stop") & (S[k] > 0)).sum()), int(((fl == "turn") & (S[k] > 0)).sum())
        nz = int((sets["TARGET (STOP + STOP-L + TURN)"] & (a0 > 0) & (S[k] == 0)).sum())
        L.append(f"| {k} | {fs[0]}/{int((fl == 'stop').sum())} | {fs[1]}/{int((fl == 'turn').sum())} | {nz} |")
    # native vs ACC + GAP on the target subsets
    L += ["", "## Native (VC + LEAD) against ACC + GAP, same checkpoint, on the 281-derived subsets", "",
          "| checkpoint | subset | ACC + GAP | native | zeros AG / native | same zero scenes |", "|---|---|--:|--:|--:|--:|"]
    for ag, nat in NATIVE:
        for name in ("STOP", "STOP-L", "TURN", "complement (rest of 281)"):
            m = sets[name]
            za, zn = S.loc[m, ag] == 0, S.loc[m, nat] == 0
            L.append(f"| {ag} | {name} | {S.loc[m, ag].mean():.3f} | {S.loc[m, nat].mean():.3f} | {int(za.sum())} / {int(zn.sum())} | {int((za & zn).sum())} |")
    nat_rank = {name: (S.loc[sets[name], [n for _, n in NATIVE]].mean().rank().tolist(), S.loc[sets[name], [g for g, _ in NATIVE]].mean().rank().tolist())
                for name in ("STOP", "STOP-L", "TURN", "TARGET (STOP + STOP-L + TURN)", "full")}
    L += ["", "Rank of (NTV0, JTN, RSG) native / ACC + GAP per subset: " + "; ".join(f"{k} {v[0]} / {v[1]}" for k, v in nat_rank.items())]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.md").write_text("\n".join(L) + "\n")
    S.assign(subset=sub, a0_fail=fl).to_csv(out / "calib_scores.csv.gz", float_format="%.4f")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("facts")
    f.add_argument("--pkl", required=True), f.add_argument("--runs", nargs="+", required=True), f.add_argument("--out", required=True)
    c = sp.add_parser("calib")
    c.add_argument("--inv", default="tmp/pai0/inv.json"), c.add_argument("--gap", required=True)
    c.add_argument("--nurec", default=str(REPO / "experiments/pai_av/results/exam_nurec.csv")), c.add_argument("--out", required=True)
    s = sp.add_parser("select")
    s.add_argument("--facts", nargs="+", required=True), s.add_argument("--human", default=""), s.add_argument("--gap", required=True)
    s.add_argument("--nurec", default=str(REPO / "experiments/pai_av/results/exam_nurec.csv"))
    s.add_argument("--clip-map", default=str(REPO / "experiments/pai_av/results/clip_map.csv"))
    s.add_argument("--chunk", type=int, default=32), s.add_argument("--out", required=True)
    d = sp.add_parser("drives")
    d.add_argument("--exam", required=True), d.add_argument("--meta", required=True), d.add_argument("--out", required=True)
    d.add_argument("--clip-map", default=str(REPO / "experiments/pai_av/results/clip_map.csv"))
    a = ap.parse_args()
    {"facts": cmd_facts, "calib": cmd_calib, "select": cmd_select, "drives": cmd_drives}[a.cmd](a)


if __name__ == "__main__":
    main()
