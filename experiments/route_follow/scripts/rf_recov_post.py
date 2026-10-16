#!/usr/bin/env python3
"""RECOVERY-ENVELOPE post-training lines (prereg experiments/route_follow/plans/2026-10-16-recovery-envelope-prereg.md, section
全量读数与线). Helper of rf_recov_post.sh; box, repo root.

  missing  (stdlib)   rf_recov_post.py missing --scenes <dir> --cache <all-usdzs> --chunks <chunk files> [--tsv OUT]
      scenes of the chunk lists whose usdz is not in the scene cache; --tsv writes them in pai_fetch.sh's format (stage 1). Prints the count.
  lines    (AlpaSim venv: numpy + pandas)   rf_recov_post.py lines --arm NAME --kind R|L|RL --tags T0 [T1] --dir DIR --strict FILE [--note ..]
      from DIR/held18/confirm.json, DIR/confirm/confirm.json, DIR/follow/{follow.json,scenes.csv}, DIR/pai_londiag.json, DIR/pai_turns.json,
      DIR/pai_track.json -> DIR/lines.md, DIR/lines.json. Missing inputs leave their lines `pending`.

Registered lines (main session, copied verbatim into the prereg): vs RSG-L18 2-seed means on held 452 paired, log-clustered bootstrap:
  H1 harmless: zeros <= +3, at-fault <= +3, GT <= +0.03 m (points; CIs reported);
  H2 turning: held 24 turns completion seed mean >= +3 for R and RL; PAI 281 (seed 0, vs RSG-L18-s0-gate): strict 36 completion >= 10
  (reference 4), all-281 zeros <= reference + 3; L / RL: PAI cor_route_end <= reference - 3 and rear-ended <= reference + 3;
  public guard: score CI low > -0.01, zeros <= +5. Adopt an arm only if H1 + guard pass and its own H2 line passes.
Reading of "its own H2 line" used here (stated in lines.md): R = held turns + PAI turning (strict 36 and PAI zeros); L = the L / RL PAI lines
(cor_route_end and rear-ended) + PAI zeros; RL = all of them. Every line is also reported on its own.
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

TOKYO_REF = dict(label="RSG-L18-s0-gate (Tokyo, results/pai281/RSG-L18-s0-gate.md)", zeros=71, strict36=4, rear_ended_collision_rear=72, mean=0.5860)
B, SEED = 10_000, 0


def missing(a):
    have = set(os.listdir(a.cache))
    rows = {}
    for f in ("sim_scenes.csv", "sim_scenes_2604.csv"):
        for r in csv.DictReader(open(os.path.join(a.scenes, f))):
            rows.setdefault(r["scene_id"], []).append(r)
    want = list(dict.fromkeys(s for c in a.chunks for s in open(c).read().split()))
    miss = [s for s in want if not any(r["uuid"] + ".usdz" in have for r in rows.get(s, []))]
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("scene_id\tuuid\tpath\trev\ta\tb\tc\tstage\n")
            for s in miss:
                for r in rows.get(s, [])[-1:]:                      # the newest catalogue row (no scene of the 281 is in both)
                    f.write(f"{s}\t{r['uuid']}\t{r['path']}\t{r['hf_revision']}\t-\t-\t-\t1\n")
    print(len(miss))


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def n_scenes(md, st):
    m = re.search(rf"## {st}: (\d+) scenes, (\d+) logs", Path(md).read_text()) if Path(md).exists() else None
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def h2_held(d, arm_tags, ref_tags):
    """Turn completion counts per label from rf_follow (follow.json outcomes) and the paired seed-mean difference on the turn scenes common to
    every label (scenes.csv), log-clustered bootstrap of the summed difference."""
    import numpy as np
    import pandas as pd
    J = jload(d / "follow/follow.json")
    if J is None or not (d / "follow/scenes.csv").exists():
        return None
    lab = lambda t: f"{t}@18"  # noqa: E731
    if any(lab(t) not in J["labels"] for t in arm_tags + ref_tags):
        return None
    cnt = {t: dict(completed=J["labels"][lab(t)]["outcomes"]["completed"], turns=J["labels"][lab(t)]["turns"], outcomes=J["labels"][lab(t)]["outcomes"])
           for t in arm_tags + ref_tags}
    S = pd.read_csv(d / "follow/scenes.csv")
    S = S[S.label.isin([lab(t) for t in arm_tags + ref_tags])]
    T = S[S.turn.astype(str).str.lower().isin(("true", "1", "1.0"))]
    common = set.intersection(*[set(T[T.label == lab(t)].scene) for t in arm_tags + ref_tags])
    T = T[T.scene.isin(common)].assign(c=lambda x: x.completed.astype(str).str.lower().isin(("true", "1", "1.0")).astype(float))
    P = T.pivot(index="scene", columns="label", values="c")
    diff = P[[lab(t) for t in arm_tags]].mean(1) - P[[lab(t) for t in ref_tags]].mean(1)
    logs = T.drop_duplicates("scene").set_index("scene").log.reindex(diff.index)
    g = diff.groupby(logs.to_numpy()).sum().to_numpy()
    bs = g[np.random.default_rng(SEED).integers(0, len(g), (B, len(g)))].sum(1)
    pt = float(np.mean([cnt[t]["completed"] for t in arm_tags]) - np.mean([cnt[t]["completed"] for t in ref_tags]))
    return dict(counts=cnt, seed_mean_diff=pt, paired=[float(diff.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                common_turns=len(common), logs=len(g))


def lines(a):
    d = Path(a.dir)
    arm, kind, tags = a.arm, a.kind, a.tags
    two = len(tags) == 2
    ref_tags = ["RSG-L18-s0", "RSG-L18-s1"]
    pk = f"{arm} - RSG-L18"
    held = (jload(d / "held18/confirm.json") or {}).get("held18", {}).get(pk)
    guard = (jload(d / "confirm/confirm.json") or {}).get("all10", {}).get(pk)
    nh, lh = n_scenes(d / "held18/confirm.md", "held18")
    ng, lg = n_scenes(d / "confirm/confirm.md", "all10")
    h2 = h2_held(d, tags, ref_tags)
    pl = jload(d / "pai_londiag.json")
    pt = jload(d / "pai_turns.json")
    tr = jload(d / "pai_track.json") or {}
    strict = [x.split()[0] for x in open(a.strict) if x.strip()]

    def pai_strict():
        if pt is None:
            return None
        by = {(r["label"], r["scene"]): r for r in pt["rows"]}
        out = {}
        for lab in ("arm", "ref"):
            rows = [by.get((lab, s), {}) for s in strict]
            out[lab] = dict(completed=sum(bool(r.get("completed")) for r in rows), missing=sum(r.get("missing", True) for r in rows))
        return out

    ps = pai_strict()
    same_path = bool(tr.get("arm") and tr.get("arm") == tr.get("ref"))
    L = {}
    # H1 harmless (held 452 paired, seed means; points, CIs reported)
    if held:
        L["H1_zeros"] = dict(v=held["zero"], line="<= +3 (point)", ok=held["zero"][0] <= 3)
        L["H1_at_fault"] = dict(v=held["z_atf"], line="<= +3 (point)", ok=held["z_atf"][0] <= 3)
        L["H1_gt"] = dict(v=held["gt"], line="<= +0.03 m (point)", ok=held["gt"][0] <= 0.03)
        L["H1_score"] = dict(v=held["score"], line="reported", ok=None)
    if guard:
        L["guard_score"] = dict(v=guard["score"], line="CI low > -0.01", ok=guard["score"][1] > -0.01)
        L["guard_zeros"] = dict(v=guard["zero"], line="<= +5 (point)", ok=guard["zero"][0] <= 5)
    if h2:
        L["H2_held_turns"] = dict(v=[h2["seed_mean_diff"], h2["paired"][1], h2["paired"][2]], line=">= +3 (seed-mean completed turns)",
                                  ok=h2["seed_mean_diff"] >= 3)
    if pl and same_path:
        n, r = pl["new"], pl["ref_counts"]
        L["PAI_zeros"] = dict(v=[n["zeros"], r["zeros"]], line="<= ref + 3", ok=n["zeros"] <= r["zeros"] + 3)
        m3 = 0 if kind == "J" else 3                                     # JOINT-DAGGER registers cor_route_end <= reference
        L["PAI_cor_route_end"] = dict(v=[n["cor_route_end"], r["cor_route_end"]], line=f"<= ref{' - 3' if m3 else ''}", ok=n["cor_route_end"] <= r["cor_route_end"] - m3)
        L["PAI_rear_ended"] = dict(v=[n["rear_ended"], r["rear_ended"]], line="<= ref + 3", ok=n["rear_ended"] <= r["rear_ended"] + 3)
    if pl and same_path and kind == "S":                              # STACK (plans/2026-10-16-stack-prereg.md 4): fewer crashes, at-fault = collision zeros
        n, r = pl["new"], pl["ref_counts"]
        L["PAI_S_crash"] = dict(v=[n["zeros"], r["zeros"], n["coll_zeros"], r["coll_zeros"]], line="zeros <= ref - 5 OR at-fault (collision zeros) <= ref - 3",
                                ok=n["zeros"] <= r["zeros"] - 5 or n["coll_zeros"] <= r["coll_zeros"] - 3)
    if ps and same_path and kind == "S":
        L["PAI_S_strict"] = dict(v=[ps["arm"]["completed"], ps["ref"]["completed"]], line=">= reference", ok=ps["arm"]["completed"] >= ps["ref"]["completed"])
    if ps and same_path:
        L["PAI_strict36"] = dict(v=[ps["arm"]["completed"], ps["ref"]["completed"]], line=">= 10 (registered reference 4)", ok=ps["arm"]["completed"] >= 10)

    def all_ok(keys):
        v = [L[k]["ok"] if k in L else None for k in keys]
        return None if any(x is None for x in v) else all(v)

    H1, G = all_ok(["H1_zeros", "H1_at_fault", "H1_gt"]), all_ok(["guard_score", "guard_zeros"])
    own_keys = dict(R=["H2_held_turns", "PAI_strict36", "PAI_zeros"], L=["PAI_cor_route_end", "PAI_rear_ended", "PAI_zeros"],
                    RL=["H2_held_turns", "PAI_strict36", "PAI_zeros", "PAI_cor_route_end", "PAI_rear_ended"],
                    J=["H2_held_turns", "PAI_strict36", "PAI_zeros", "PAI_cor_route_end"], S=["PAI_S_crash", "PAI_S_strict"])[kind]
    OWN = all_ok(own_keys)
    if not two:                                                        # the registered lines are 2-seed means: no verdict on seed 0 alone
        verdict = "PENDING (seed 1 not read)" if (H1 is not False and G is not False and OWN is not False) else "NOT ADOPTED (seed 0 already fails)"
    elif None in (H1, G, OWN):
        verdict = "NOT ADOPTED" if False in (H1, G, OWN) else "PENDING"
    else:
        verdict = "ADOPT" if H1 and G and OWN else "NOT ADOPTED"
    ok = lambda x: {None: "pending", True: "PASS", False: "FAIL"}[x]  # noqa: E731
    f = lambda v, p=1: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
    md = [f"# {dict(J='JOINT-DAGGER', S='STACK').get(kind, 'RECOVERY-ENVELOPE')} {arm} ({kind}): registered lines vs RSG-L18", "",
          f"Tags {', '.join(tags)}{'' if two else ' (seed 1 pending: seed-0 numbers against the RSG-L18 2-seed mean)'}; reference RSG-L18-s0 / -s1 "
          f"(stored reads). Held: {nh} paired scenes / {lh} logs; public: {ng} scenes / {lg} logs; log-clustered 95 % CI, {B} draws, seed {SEED}.", "",
          f"**Verdict: {verdict}** (H1 {ok(H1)}, guard {ok(G)}, own lines {ok(OWN)}: {', '.join(own_keys)})", "",
          "| line | value [95 % CI] | registered | result |", "|:--|:--|:--|:--|"]
    rowfmt = dict(H1_zeros=lambda v: f(v), H1_at_fault=lambda v: f(v), H1_gt=lambda v: f(v, 3), H1_score=lambda v: f(v, 4), guard_score=lambda v: f(v, 4),
                  guard_zeros=lambda v: f(v), H2_held_turns=lambda v: f(v),
                  PAI_zeros=lambda v: f"{v[0]} vs {v[1]} (d {v[0] - v[1]:+d})", PAI_cor_route_end=lambda v: f"{v[0]} vs {v[1]} (d {v[0] - v[1]:+d})",
                  PAI_rear_ended=lambda v: f"{v[0]} vs {v[1]} (d {v[0] - v[1]:+d})", PAI_strict36=lambda v: f"{v[0]} / 36 vs {v[1]} / 36",
                  PAI_S_crash=lambda v: f"zeros {v[0]} vs {v[1]} (d {v[0] - v[1]:+d}), at-fault {v[2]} vs {v[3]} (d {v[2] - v[3]:+d})",
                  PAI_S_strict=lambda v: f"{v[0]} / 36 vs {v[1]} / 36")
    names = dict(H1_zeros="H1 held zeros d", H1_at_fault="H1 held at-fault zeros d", H1_gt="H1 held GT distance d (m)", H1_score="held mean score d",
                 guard_score="guard public mean score d", guard_zeros="guard public zeros d", H2_held_turns="H2 held turns completed d (of ~24)",
                 PAI_zeros="PAI all-281 zeros, arm vs ref", PAI_cor_route_end="PAI cor_route_end, arm vs ref", PAI_rear_ended="PAI rear-ended, arm vs ref",
                 PAI_strict36="PAI strict 36 completed, arm vs ref", PAI_S_crash="STACK: PAI zeros / at-fault, arm vs ref",
                 PAI_S_strict="STACK: PAI strict 36, arm vs ref")
    for k in names:
        if k in L:
            md.append(f"| {names[k]} | {rowfmt[k](L[k]['v'])} | {L[k]['line']} | {ok(L[k]['ok']) if L[k]['ok'] is not None else '-'} |")
        else:
            md.append(f"| {names[k]} | pending | | pending |")
    if h2:
        md += ["", "Held turns (rf_follow, 452 x 18 decisions): " + "; ".join(f"{t} {c['completed']} / {c['turns']}" for t, c in h2["counts"].items())
               + f"; paired on the {h2['common_turns']} turn scenes common to every label ({h2['logs']} logs): {f(h2['paired'])}."]
    if pl or ps:
        md += ["", f"PAI 281, seed 0 ({tags[0]}) against the same-host reference RSG-L18-s0 read on the box with the identical path (gate serve, gains 20, "
               f"JEV_VCONT 1.0, JEV_LEAD 1, the Tokyo reference's 8 chunk lists). Serving path logged: arm {tr.get('arm') or '?'}, ref {tr.get('ref') or '?'}"
               + ("" if same_path else " -> **paths differ or unknown: PAI lines not judged**") + "."]
        if pl:
            n, r = pl["new"], pl["ref_counts"]
            md.append(f"rf_londiag pai ({n['scenes']} paired scenes): mean {n['mean']:.4f} vs {r['mean']:.4f}, zeros {n['zeros']} vs {r['zeros']}, collision zeros "
                      f"{n['coll_zeros']} vs {r['coll_zeros']}, cor_route_end {n['cor_route_end']} vs {r['cor_route_end']}, rear-ended {n['rear_ended']} vs "
                      f"{r['rear_ended']}, exam 72 mean d {pl['exam72_mean_diff'][0]:+.4f} [{pl['exam72_mean_diff'][1]:+.4f}, {pl['exam72_mean_diff'][2]:+.4f}].")
        if ps:
            md.append(f"Strict 36 (rf_pai_follow turn rule, rf_v2pai.py turns): arm {ps['arm']['completed']} (missing {ps['arm']['missing']}), "
                      f"box ref {ps['ref']['completed']} (missing {ps['ref']['missing']}).")
        md.append(f"Tokyo reference for context only (other host, not paired): {TOKYO_REF['label']}: zeros {TOKYO_REF['zeros']}, strict 36 completed "
                  f"{TOKYO_REF['strict36']}, rear-ended (collision_rear) {TOKYO_REF['rear_ended_collision_rear']}, mean {TOKYO_REF['mean']}.")
    md += ["", "Lines (prereg 全量读数与线, verbatim): H1 zeros <= +3, at-fault <= +3, GT <= +0.03 m (points; CIs reported); H2 held 24 turns completion seed "
           "mean >= +3 for R and RL; PAI 281 (seed 0): strict 36 completion >= 10 (reference 4), all-281 zeros <= reference + 3; L / RL: PAI cor_route_end "
           "<= reference - 3 and rear-ended <= reference + 3; public guard: score CI low > -0.01, zeros <= +5. Adopt only if H1 + guard pass and the arm's "
           "own H2 line passes. Own lines as read here: R = held turns + PAI strict 36 + PAI zeros; L = PAI cor_route_end + rear-ended + PAI zeros; "
           "RL = all five. PAI reference = RSG-L18-s0 read on the box (box and Tokyo PAI reads are deterministic per host but not equal, "
           "results/p4_self/box_det.md); the strict-36 threshold stays the registered absolute 10.",
           f"Sources: {d}/held18, confirm, follow, pai_londiag.json, pai_turns.json." + (f" {a.note}" if a.note else "")]
    if kind == "S":                                                    # STACK's own registration (plans/2026-10-16-stack-prereg.md 4)
        md[-2] = ("Lines (STACK prereg, verbatim): H1 zeros <= +3, at-fault <= +3, GT <= +0.03 m; guard score CI low > -0.01, zeros <= +5; PAI zeros <= "
                  "reference - 5 OR at-fault <= reference - 3; strict turns not below reference. At-fault on PAI = rf_londiag collision zeros. Reads staged.")
    if kind == "J":                                                    # JOINT-DAGGER's own registration (plans/2026-10-16-joint-dagger-prereg.md)
        md[-2] = ("Lines (JOINT-DAGGER prereg, verbatim): H1 zeros <= +3, at-fault <= +3, GT <= +0.03 m; H2 held 24 turns seed mean >= +3; PAI strict 36 "
                  ">= 10 (reference 4), PAI zeros <= reference + 3, cor_route_end <= reference; guard score CI low > -0.01, zeros <= +5. Reads staged: "
                  "public guard and PAI only after H1 passes on held. Own lines = H2 held turns + PAI strict 36 + PAI zeros + cor_route_end.")
    res = dict(arm=arm, kind=kind, tags=tags, two_seeds=two, verdict=verdict, H1=H1, guard=G, own=OWN, own_keys=own_keys, lines=L,
               held_scenes=nh, public_scenes=ng, h2=h2, pai_londiag=pl, pai_strict=ps, pai_track=tr, tokyo_ref=TOKYO_REF)
    (d / "lines.md").write_text("\n".join(md) + "\n")
    (d / "lines.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("\n".join(md))


def track(a):
    """JEV_TRACK path the drivers of a PAI read logged ("serving:" line; unset = gate since decision 309; several = a/b)."""
    p = set()
    for f in glob.glob(os.path.join(a.read, "runs", "*", "sim", "native-logs", "driver.log")):
        for line in open(f, errors="replace"):
            if "serving: JEV_VCONT" in line:
                w = line.split()
                p.add(next((w[i + 1] for i in range(len(w) - 1) if w[i] == "JEV_TRACK"), "0"))
    print("/".join(sorted(p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("missing")
    m.add_argument("--scenes", required=True)
    m.add_argument("--cache", required=True)
    m.add_argument("--chunks", nargs="+", required=True)
    m.add_argument("--tsv", default="")
    q = sp.add_parser("lines")
    q.add_argument("--arm", required=True)
    q.add_argument("--kind", required=True, choices=("R", "L", "RL", "J", "S"))
    q.add_argument("--tags", nargs="+", required=True)
    q.add_argument("--dir", required=True)
    q.add_argument("--strict", required=True)
    q.add_argument("--note", default="")
    t = sp.add_parser("track")
    t.add_argument("--read", required=True)
    a = ap.parse_args()
    dict(missing=missing, lines=lines, track=track)[a.cmd](a)


if __name__ == "__main__":
    main()
