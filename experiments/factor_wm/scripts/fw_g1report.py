"""factor_wm G1 readouts as registered (plans/2026-10-05-stage1-prereg.md sections 5 and 3.3), arms S0 (shipped) / S1 / S2 / S3.

HUGSIM (guard line `hugsim`, full mode = 64 scenes, spec preset, one run): per scene end class, spin, HD, launch stall (max speed over the first
40 steps < 1.6 m/s, decision 138's definition, from zs_steps.jsonl); ego-motion failure = stuck (max_steps) or spin or launch stall or bg collision.
  G1 gate on the frozen 24 scenes (results/g1_scenes.json): fixed = S0 HD < 0.5 and arm HD >= 0.5 and the arm's end is not a collision;
  S3 fixes >= 4 of the 12 ego scenes and >= S2 fixes + 2; of the 8 guard scenes at most 1 drops below HD 0.5; navtest PDMS S3 - S0 >= -0.5.
G1-guard: (a) collisions (fg + bg) over 64 S3 <= S0 + 2; (b) S0-ego-failure scenes that end in a collision under S3 <= half of those S3
  completes with HD >= 0.5; (c) WOD g1s stay clips, false go (ego > 2 m or past +5 m) S3 <= S0 + 0.05; (d) navtest NC S3 - S0 >= -0.5 pp.
Descriptive: per-arm HUGSIM totals, navhard, WOD g0b failure rates (fw_report.reclassify), steer jitter.

  $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_g1report.py --out $DATA_DIR/runs/factor_wm/report/g1
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402
import fw_report as FR  # noqa: E402

ARMS = {"S0": "shipped", "S1": "fw-S1", "S2": "fw-S2", "S3": "fw-S3"}
GUARD = C.data_dir() / "runs" / "op_guard"
SCENES = Path(__file__).resolve().parents[1] / "results" / "g1_scenes.json"


def line(cand, name):
    f = GUARD / cand / "full" / "lines" / f"{name}.json"
    return json.load(open(f)) if f.exists() else None


def hugsim(cand):
    d = line(cand, "hugsim")
    if d is None:
        return None
    out = {}
    for s, r in d["provenance"]["per_scene"].items():
        v, st = [], []
        f = Path(r["run_dir"]) / "zs_steps.jsonl"
        if f.exists():
            for ln in open(f):
                j = json.loads(ln)
                if "step" in j:
                    v.append(float(j.get("v", np.nan)))
                    st.append(float(j.get("steer", np.nan)))
        v = np.array(v)
        stall = bool(len(v) and np.nanmax(v[:40]) < 1.6)
        end = r["end"]
        out[s] = dict(end=end, spin=bool(r["spin"]), hd=float(r["hd"]), stall=stall,
                      ego_fail=bool(end == "max_steps" or r["spin"] or stall or end == "bg_collision"),
                      coll=end in ("fg_collision", "bg_collision"), jitter=float(np.nanmean(np.abs(np.diff(st)))) if len(st) > 2 else np.nan)
    return out


def nav(cand, name):
    d = line(cand, name)
    if d is None:
        return None
    return d["rows"]


def wod(arm):
    out = {}
    for name in ("g0b", "g1s"):
        fs = sorted(glob.glob(str(C.root("roll", f"eval-{arm}") / f"{name}-eval-[0-9]*of*.npz")))
        if not fs:
            continue
        zs = [dict(np.load(f, allow_pickle=True)) for f in fs]
        Z = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
        FR.reclassify(Z)
        if name == "g0b":
            pert = np.array([a.startswith(("kick", "swerve")) for a in Z["arm"]])
            r = {}
            for grp, m in (("perturbed", pert), ("free", Z["arm"] == "free")):
                ev = Z["event"][m]
                r[grp] = dict(n=int(m.sum()), any=float(np.mean(np.isin(ev, ["stall", "heading", "lane"]))),
                              **{k: float(np.mean(ev == k)) for k in ("stall", "heading", "lane", "ahead", "behind")},
                              by_cat={c: {k: int(np.sum((ev == k) & (Z["cat"][m] == c))) for k in ("stall", "heading", "lane")} | {"n": int(np.sum(Z["cat"][m] == c))}
                                      for c in ("launch", "turn", "cruise")})
            out["g0b"] = r
        else:
            dist = np.array([np.sum(vs[1:] + vs[:-1]) * 0.5 * C.DT for vs in Z["vs"]])
            go = (dist > 2.0) | (Z["event"] == "ahead")
            out["g1s"] = dict(n=len(go), false_go=float(np.mean(go)), median_dist_m=float(np.median(dist)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = json.load(open(SCENES))
    H = {arm: hugsim(c) for arm, c in ARMS.items()}
    NT = {arm: nav(c, "navtest") for arm, c in ARMS.items()}
    NH = {arm: nav(c, "navhard") for arm, c in ARMS.items()}
    W = {arm: wod(arm) for arm in ARMS}
    res = dict(hugsim_totals={}, gate={}, guard={}, navtest={}, navhard={}, wod=W)
    for arm, h in H.items():
        if h is None:
            continue
        vals = list(h.values())
        res["hugsim_totals"][arm] = dict(n=len(vals), hd=float(np.mean([x["hd"] for x in vals])), complete=sum(x["end"] == "complete" for x in vals),
                                         stuck=sum(x["end"] == "max_steps" for x in vals), spin=sum(x["spin"] for x in vals), stall=sum(x["stall"] for x in vals),
                                         ego_fail=sum(x["ego_fail"] for x in vals), fg=sum(x["end"] == "fg_collision" for x in vals),
                                         bg=sum(x["end"] == "bg_collision" for x in vals), jitter_median=float(np.nanmedian([x["jitter"] for x in vals])))
    S0 = H.get("S0")
    key = lambda s: s if s in (S0 or {}) else s  # noqa: E731
    for arm in ("S1", "S2", "S3"):
        h = H.get(arm)
        if h is None or S0 is None:
            continue
        fixed = [s for s in sc["ego"] if s in h and S0[s]["hd"] < 0.5 and h[s]["hd"] >= 0.5 and not h[s]["coll"]]
        drop = [s for s in sc["guard"] if s in h and h[s]["hd"] < 0.5]
        to_coll = [s for s in S0 if S0[s]["ego_fail"] and h.get(s, {}).get("coll")]
        to_ok = [s for s in S0 if S0[s]["ego_fail"] and h.get(s, {}).get("hd", 0) >= 0.5 and not h[s]["coll"]]
        res["gate"][arm] = dict(fixed_ego12=len(fixed), fixed=fixed, guard8_drops=len(drop), drops=drop,
                                fg4_hd05=sum(h[s]["hd"] >= 0.5 for s in sc["fg"] if s in h),
                                s0_fail_to_coll=len(to_coll), s0_fail_to_ok=len(to_ok),
                                coll_total=sum(x["coll"] for x in h.values()), coll_total_s0=sum(x["coll"] for x in S0.values()))
    for arm in ARMS:
        if NT[arm]:
            r = NT[arm][0]
            res["navtest"][arm] = dict(value=r["value"], delta=r["delta"], ci=r.get("ci"), sub=r.get("sub_deltas"))
        if NH[arm]:
            res["navhard"][arm] = [dict(metric=r["metric"], value=r["value"], delta=r["delta"], ci=r.get("ci")) for r in NH[arm]]
    g = res["gate"].get("S3")
    if g and "S2" in res["gate"] and "S3" in res["navtest"]:
        nt = res["navtest"]["S3"]["delta"]
        res["G1"] = dict(fixes=g["fixed_ego12"] >= 4, vs_S2=g["fixed_ego12"] >= res["gate"]["S2"]["fixed_ego12"] + 2,
                         guard8=g["guard8_drops"] <= 1, navtest=nt >= -0.5)
        fg0 = W.get("S0", {}).get("g1s", {}).get("false_go")
        fg3 = W.get("S3", {}).get("g1s", {}).get("false_go")
        nc = (res["navtest"]["S3"].get("sub") or {}).get("NC")
        res["G1guard"] = dict(a_collisions=g["coll_total"] <= g["coll_total_s0"] + 2, b_trade=g["s0_fail_to_coll"] <= 0.5 * g["s0_fail_to_ok"],
                              c_false_go=(fg3 is not None and fg0 is not None and fg3 <= fg0 + 0.05), d_nc=(nc is not None and nc >= -0.5))
        res["G1_pass"] = bool(all(res["G1"].values()) and all(res["G1guard"].values()))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1, default=str)
    L = ["# factor_wm G1 readouts (generated by scripts/fw_g1report.py)", "",
         "## HUGSIM 64, spec, one run", "", "| arm | HD mean | complete | stuck | spin | launch stall | ego failure | fg coll | bg coll | steer jitter |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm, t in res["hugsim_totals"].items():
        L.append(f"| {arm} | {t['hd']:.3f} | {t['complete']} | {t['stuck']} | {t['spin']} | {t['stall']} | {t['ego_fail']} | {t['fg']} | {t['bg']} | {t['jitter_median']:.4f} |")
    L += ["", "## G1 gate (24 frozen scenes) and trade-off", "", "| arm | fixed of 12 ego | guard-8 drops | fg-4 HD >= 0.5 | S0 ego fail -> collision | S0 ego fail -> HD >= 0.5 | collisions (S0) |",
          "|:--|--:|--:|--:|--:|--:|--:|"]
    for arm, gg in res["gate"].items():
        L.append(f"| {arm} | {gg['fixed_ego12']} | {gg['guard8_drops']} | {gg['fg4_hd05']} | {gg['s0_fail_to_coll']} | {gg['s0_fail_to_ok']} | {gg['coll_total']} ({gg['coll_total_s0']}) |")
    L += ["", "## NAVSIM", "", "| arm | navtest PDMS | delta [95% CI] | sub-score deltas (pp) |", "|:--|--:|--:|:--|"]
    for arm, n in res["navtest"].items():
        ci = n.get("ci") or [np.nan, np.nan]
        L.append(f"| {arm} | {n['value']:.2f} | {n['delta']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {n.get('sub')} |")
    for arm, rows in res["navhard"].items():
        for r in rows:
            L.append(f"| {arm} navhard | {r['metric']}: {r['value']} | {r['delta']} | |")
    L += ["", "## WOD engine (g0b val 120, plane, closed, 8 s) and stay guard (g1s 40)", "",
          "| arm | perturbed any | stall | heading | lane | free any | g1s false go |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for arm, w in W.items():
        p, f = w.get("g0b", {}).get("perturbed"), w.get("g0b", {}).get("free")
        s = w.get("g1s", {})
        if p:
            L.append(f"| {arm} | {p['any']:.2f} | {p['stall']:.2f} | {p['heading']:.2f} | {p['lane']:.2f} | {f['any']:.2f} | {s.get('false_go', float('nan')):.2f} |")
    if "G1" in res:
        L += ["", f"G1 lines: {res['G1']}", f"G1-guard: {res['G1guard']}", f"**G1 {'pass' if res['G1_pass'] else 'fail'}**"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
