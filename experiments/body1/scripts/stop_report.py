#!/usr/bin/env python3
"""BODY1 arm 4.1 closed-loop read (plans/2026-10-10-body1-prereg.md, section 4.4 + Amendment 2): the stop switch against P2H10-F.

  pilot   --man <pilot manifest> [--base-man <tr1 manifest>]      pilot8: off twice, on once -> per-scene table, body-record coverage
  check   --run <dir>[,<dir>] --base <dir>[,<dir>] --out DIR      the one-chunk checklist of Amendment 2 item 8 (b) + trace figure
  report  --base-man M --man M [M ...] --arm stop2 [--arm stop1] --out DIR      the full read: lines L1-L3, zeros by class, flips, flags,
          latency, the > 45 deg scenes -> report.md, per_scene.csv, flips.csv, flags.csv, stats.json
Manifests are ot2_loop.py's {label: [run dirs]}; arm X uses the labels X-s0 / X-s1, the baseline P2H10-F-s0 / P2H10-F-s1.
Box, envs/op-train python (numpy, pandas, matplotlib). Scores come from aggregate/results-summary.json; flags from the driver's drive.jsonl
(`body`); traces from the controller's per-rollout csv. > 45 deg = |heading change of the logged 4 s future of the scene's navtest token|
(bench TURN_BINS convention; op_parity cache lb_navtest).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
BASE = ("P2H10-F-s0", "P2H10-F-s1")
SHORT = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor", "other": "other", "": ""}


def drives(dirs):
    """-> {scene: [drive records in decision order]} (the first run dir that holds a scene wins, as c0b_report.load_driver) and {scene: (dir, session)}."""
    D, where = {}, {}
    for d in map(Path, dirs):
        cur = defaultdict(list)
        for line in (d / "driver-logs/drive.jsonl").open():
            if '"kind": "drive"' in line:
                x = json.loads(line)
                cur[x["scene"]].append(x)
        for s, v in cur.items():
            if s not in D:
                D[s], where[s] = sorted(v, key=lambda x: x["k"]), (d, v[0]["session"])
    return D, where


def ctrl(d, session):
    f = Path(d) / "controller" / f"alpasim_controller_{session}.csv"
    return np.genfromtxt(f, delimiter=",", names=True) if f.exists() else None


def turns():
    t = np.load(DATA / "runs/op_parity/cache/lb_navtest/tab.npz")
    return dict(zip(t["names"].tolist(), np.degrees(t["fut"][:, -1, 2]).tolist()))


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else float("nan")


def flag_stats(D):
    """Drive records {scene: [...]} -> dict of flag facts (records without `body` count as unflagged)."""
    recs = [x for v in D.values() for x in v]
    b = [x.get("body") for x in recs]
    fl = np.array([bool(y and y["flag"]) for y in b])
    F = [y for y in b if y and y["flag"]]
    sc = {s: sum(bool(x.get("body") and x["body"]["flag"]) for x in v) for s, v in D.items()}
    return dict(decisions=len(recs), with_body=int(sum(y is not None for y in b)), flags=int(fl.sum()), flag_rate=float(fl.mean()) if len(fl) else 0.0, scenes=len(D),
                scenes_flagged=int(sum(n > 0 for n in sc.values())), by_k=[int(sum(bool(x.get("body") and x["body"]["flag"]) for x in recs if x["k"] == k)) for k in range(10)],
                removes_speed=int(sum(y["cut"] > 0.05 for y in F)), at_cap=int(sum(y["a"] >= 5.999 for y in F)), d_median=q([y["d"] for y in F], 0.5),
                d_under_half=int(sum(y["d"] <= 0.5 for y in F)), a_median=q([y["a"] for y in F], 0.5), cut_median=q([y["cut"] for y in F], 0.5),
                v0_median=q([y["v0"] for y in F], 0.5), standing=int(sum(y["v0"] < 0.5 for y in F)), hook_ms_p50=q([y["ms"] for y in b if y], 0.5),
                hook_ms_p90=q([y["ms"] for y in b if y], 0.9), total_ms_p50=q([x["ms"]["total"] for x in recs], 0.5), total_ms_p90=q([x["ms"]["total"] for x in recs], 0.9),
                total_ms_p99=q([x["ms"]["total"] for x in recs], 0.99), total_ms_max=max(x["ms"]["total"] for x in recs)), sc


def summary(Rs, scenes):
    sc = np.array([Rs[s]["score"] for s in scenes])
    zc = np.array([R.zclass(Rs[s]) for s in scenes])
    return sc, zc, dict(n=len(scenes), mean=float(sc.mean()), ones=int((sc >= 1).sum()), zeros=int((sc == 0).sum()), **{SHORT[f]: int((zc == f).sum()) for f in (*R.FAIL, "other")},
                        slow=int(((sc > 0) & (sc < 1)).sum()))


# ---------------------------------------------------------------- pilot
def cmd_pilot(a):
    man = json.loads(Path(a.man).read_text())
    base = json.loads(Path(a.base_man).read_text()) if a.base_man else {}
    Rs = {k: R.load_driver(v)[0] for k, v in man.items()}
    Ds = {k: drives(v)[0] for k, v in man.items()}
    ref = R.load_driver(base[BASE[0]])[0] if base else {}
    scenes = sorted(set.intersection(*(set(r) for r in Rs.values())))
    print("| scene | " + " | ".join(man) + " | TR1 P2H10-F-s0 (in its chunk) | decisions with `body` / flagged (per label) |\n|:--|" + "--:|" * (len(man) + 2))
    for s in scenes:
        cov = "; ".join(f"{sum('body' in x for x in Ds[k].get(s, []))}/{sum(bool(x.get('body') and x['body']['flag']) for x in Ds[k].get(s, []))}" for k in man)
        print(f"| {s[-16:]} | " + " | ".join(f"{Rs[k][s]['score']:.4f} {SHORT[R.zclass(Rs[k][s])]}" for k in man) + f" | {ref[s]['score'] if s in ref else float('nan'):.4f} | {cov} |")
    ks = list(man)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = np.array([Rs[ks[i]][s]["score"] - Rs[ks[j]][s]["score"] for s in scenes])
            pr = np.array([(Rs[ks[i]][s]["score_metrics"].get("progress_clipped_rel") or 0) - (Rs[ks[j]][s]["score_metrics"].get("progress_clipped_rel") or 0) for s in scenes])
            print(f"{ks[i]} - {ks[j]}: scenes with a different score {int((np.abs(d) > 1e-9).sum())} of {len(scenes)}, max |score diff| {np.abs(d).max():.6f}, max |progress diff| {np.abs(pr).max():.6f}")
    for k in man:
        st, _ = flag_stats(Ds[k])
        print(k, json.dumps({x: st[x] for x in ("decisions", "with_body", "flags", "total_ms_p50", "total_ms_p90", "hook_ms_p50", "hook_ms_p90")}))


# ---------------------------------------------------------------- one-chunk checklist
def after_flags(D, where):
    """Per flagged decision that removes speed: (scene, k, v0, v at the next decision, commanded a, min controller acceleration command in the next 0.5 s)."""
    rows, cache = [], {}
    for s, v in D.items():
        for i, x in enumerate(v):
            b = x.get("body")
            if not (b and b["flag"] and b["cut"] > 0.05):
                continue
            if s not in cache:
                cache[s] = ctrl(*where[s])
            c = cache[s]
            v1 = v[i + 1]["body"]["v0"] if i + 1 < len(v) and v[i + 1].get("body") else np.nan
            m = (c["timestamp_us"] > x["now"]) & (c["timestamp_us"] <= x["now"] + 500_000) if c is not None else np.zeros(0, bool)
            rows.append((s, x["k"], b["v0"], v1, b["a"], float(c["u_longitudinal_actuation"][m].min()) if m.any() else np.nan, float(np.abs(c["u_steering_angle"][m]).max()) if m.any() else np.nan))
    return rows


def reversals(u, thr=0.02):
    """Steering reversals: sign changes of the steering-command increments larger than `thr` rad per 0.1 s step."""
    d = np.diff(u)
    d = d[np.abs(d) > thr]
    return int((np.sign(d[1:]) != np.sign(d[:-1])).sum()) if len(d) > 1 else 0


def traces(D, where, Db, whereb, scenes, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, len(scenes), figsize=(3.6 * len(scenes), 6.2), sharex=True, squeeze=False)
    for j, s in enumerate(scenes):
        c, cb = ctrl(*where[s]), ctrl(*whereb[s]) if s in whereb else None
        for i, (col, lab) in enumerate((("vx", "ego speed (m/s)"), ("u_longitudinal_actuation", "acceleration command (m/s^2)"), ("u_steering_angle", "steering command (rad)"))):
            ax = axs[i, j]
            if cb is not None:
                ax.plot(cb["timestamp_us"] * 1e-6, cb[col], color="#888888", lw=1.0, label="switch off (baseline run)")
            ax.plot(c["timestamp_us"] * 1e-6, c[col], color="#0072B2", lw=1.3, label="switch on")
            for x in D[s]:
                if x.get("body") and x["body"]["flag"]:
                    ax.axvline(x["now"] * 1e-6, color="#D55E00", lw=0.8, ls=":" if x["body"]["cut"] <= 0.05 else "-", alpha=0.7)
            ax.set_ylabel(lab) if j == 0 else None
            ax.grid(alpha=0.25)
        axs[0, j].set_title(s[-16:], fontsize=9)
        axs[2, j].set_xlabel("simulated time (s)")
    axs[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(title + " (vertical lines: flagged decisions; dotted = no speed removed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def cmd_check(a):
    import c0b_chain as CH
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run, base = a.run.split(","), a.base.split(",")
    Rr, Rb = R.load_driver(run)[0], R.load_driver(base)[0]
    D, where = drives(run)
    Db, whereb = drives(base)
    want = set(Path(a.scenes).read_text().split()) if a.scenes else set(Rb)
    s_, d_, ie, ip = CH.driver_health([Path(x) for x in run])
    st, per = flag_stats(D)
    sb, _ = flag_stats(Db)
    rows = after_flags(D, where)
    v0, v1, acmd, umin = (np.array([r[i] for r in rows], float) for i in (2, 3, 4, 5))
    ok = np.isfinite(v1)
    slower = float((v1[ok] < v0[ok]).mean()) if ok.any() else float("nan")
    # steering reversals in flagged scenes against the same scenes of the baseline run
    fs = [s for s, n in per.items() if n > 0]
    rev = [(reversals(ctrl(*where[s])["u_steering_angle"]), reversals(ctrl(*whereb[s])["u_steering_angle"]) if s in whereb else np.nan) for s in fs]
    allmin = min(float(ctrl(*where[s])["u_longitudinal_actuation"].min()) for s in fs) if fs else float("nan")
    scenes = sorted(set(Rr) & set(Rb))
    _, _, sr = summary(Rr, scenes)
    _, _, sbs = summary(Rb, scenes)
    chk = [("rollouts complete", f"{len(set(Rr) & want)} / {len(want)}; sessions {s_}, drive calls {d_}, inference errors {ie}, input errors {ip}", len(set(Rr) & want) == len(want) and not ie and not ip),
           ("`drive` total ms p50 / p90 (target <= 100)", f"{st['total_ms_p50']:.1f} / {st['total_ms_p90']:.1f} (p99 {st['total_ms_p99']:.1f}, max {st['total_ms_max']:.0f}); baseline run {sb['total_ms_p50']:.1f} / {sb['total_ms_p90']:.1f}; hook alone {st['hook_ms_p50']:.1f} / {st['hook_ms_p90']:.1f}",
            st["total_ms_p50"] <= 100 and st["total_ms_p90"] <= 100),
           ("flag rate per decision (1-5 %, expected 2-2.6 %)", f"{st['flags']} / {st['decisions']} = {100 * st['flag_rate']:.2f} %; scenes with a flag {st['scenes_flagged']} / {st['scenes']}; `body` on {st['with_body']} decisions; by decision index {st['by_k']}",
            0.01 <= st["flag_rate"] <= 0.05 and st["with_body"] == st["decisions"]),
           ("ego slower 0.5 s after a flagged decision that removes speed (>= 80 %)", f"{int((v1[ok] < v0[ok]).sum())} / {int(ok.sum())} = {100 * slower:.0f} %; median speed change {np.median((v1 - v0)[ok]) if ok.any() else float('nan'):+.2f} m/s for a median commanded {np.median(acmd) if len(acmd) else float('nan'):.2f} m/s^2",
            slower >= 0.8),
           ("acceleration command after a flagged decision never under -8.5 m/s^2", f"min {np.nanmin(umin) if len(umin) else float('nan'):.2f} in the 0.5 s after a flag; min {allmin:.2f} anywhere in flagged scenes", not len(umin) or np.nanmin(umin) >= -8.5),
           ("steering reversals in flagged scenes (switch on / same scenes of the baseline run)", f"sum {sum(r[0] for r in rev)} / {np.nansum([r[1] for r in rev]):.0f}; max per scene {max([r[0] for r in rev], default=0)} / {np.nanmax([r[1] for r in rev]) if rev else float('nan'):.0f}", None)]
    L = ["| check | value | verdict |", "|:--|:--|:--|"] + [f"| {n} | {v} | {'pass' if p else ('FAIL' if p is not None else 'see the traces')} |" for n, v, p in chk]
    L += ["", f"Flagged decisions: {st['flags']}, of them removing speed {st['removes_speed']}, at the 6 m/s^2 cap {st['at_cap']}, ego under 0.5 m/s {st['standing']}; medians: stop point {st['d_median']:.1f} m, "
              f"deceleration {st['a_median']:.2f} m/s^2, metres removed in 2 s {st['cut_median']:.2f}, ego speed {st['v0_median']:.1f} m/s.", "",
          "| run | n | mean | score 1 | zeros | collision | offroad | corridor | slow |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for n, s in (("switch on", sr), ("baseline run", sbs)):
        L.append(f"| {n} | {s['n']} | {s['mean']:.4f} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} |")
    top = [r[0] for r in sorted(rows, key=lambda r: -(r[2] - (r[3] if np.isfinite(r[3]) else r[2])))]
    top = list(dict.fromkeys(top))[:3]
    if top:
        traces(D, where, Db, whereb, top, out / "check_traces.png", "three flagged scenes with the largest speed drop after a flag")
        L += ["", f"Traces: check_traces.png, scenes {', '.join(t[-16:] for t in top)}."]
    (out / "check.md").write_text("\n".join(L) + "\n")
    (out / "check.json").write_text(json.dumps(dict(flags=st, base=sb, checks=[(n, v, p) for n, v, p in chk], run=sr, base_scores=sbs, traces=top), indent=1, default=float))
    print("\n".join(L))
    if any(p is False for _, _, p in chk):
        raise SystemExit("checklist FAILED")


# ---------------------------------------------------------------- the full read
def cmd_report(a):
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.base_man).read_text())
    for m in a.man:                                                     # a label run in several stages (chunk0 first, then the rest) keeps all its run dirs
        for k, v in json.loads(Path(m).read_text()).items():
            man[k] = (man[k] if k in man and k not in BASE else []) + v
    arms = {"base": list(BASE)} | {x: [f"{x}-s0", f"{x}-s1"] for x in a.arm}
    names = [k for v in arms.values() for k in v]
    Rs = {k: R.load_driver(man[k])[0] for k in names}
    Dd = {k: drives(man[k]) for k in names}
    scenes = sorted(set.intersection(*(set(Rs[k]) for k in names)))
    logs = np.array([R.log_of(s) for s in scenes])
    T = turns()
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    gt45 = np.abs(turn) > 45
    S = {k: summary(Rs[k], scenes) for k in names}
    sc, zc = {k: S[k][0] for k in names}, {k: S[k][1] for k in names}
    prog = {k: np.array([(Rs[k][s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for s in scenes]) for k in names}
    met = lambda k, s, m: float((Rs[k][s].get("metrics") or {}).get(m) or 0)  # noqa: E731
    FS = {k: flag_stats(Dd[k][0]) for k in names}
    nfl = {k: np.array([FS[k][1].get(s, 0) for s in scenes]) for k in names}
    st = dict(scenes=len(scenes), logs=len(set(logs)), turn_missing=int(np.isnan(turn).sum()), gt45=int(gt45.sum()), drivers={k: S[k][2] | dict(mean_ci=R.ci(sc[k], logs)) for k in names},
              flags={k: FS[k][0] for k in names})
    L = [f"Scenes common to all runs: {len(scenes)} from {len(set(logs))} logs; logged 4 s turn > 45 deg: {int(gt45.sum())} (token not in lb_navtest: {int(np.isnan(turn).sum())}).", "",
         "## Drivers", "", "| driver | n | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress |",
         "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        s = S[k][2]
        L.append(f"| {k} | {s['n']} | {R.fmt(R.ci(sc[k], logs))} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} | {prog[k].mean():.3f} |")
    rec = {g: np.mean([sc[k] for k in v], 0) for g, v in arms.items()}
    gm = lambda g, f: float(np.mean([S[k][2][f] for k in arms[g]]))  # noqa: E731
    L += ["", "## Recipes (per-scene mean of the two seeds; counts are seed means)", "", "| recipe | seeds | mean [95 % CI, logs] | zeros | at-fault collision | offroad | left corridor | slow |", "|:--|:--|:--|--:|--:|--:|--:|--:|"]
    for g, v in arms.items():
        L.append(f"| {g} | {' / '.join(f'{sc[k].mean():.4f}' for k in v)} | {R.fmt(R.ci(rec[g], logs))} | {gm(g, 'zeros'):g} | {gm(g, 'collision'):g} | {gm(g, 'offroad'):g} | {gm(g, 'corridor'):g} | {gm(g, 'slow'):g} |")
    L += ["", "## Lines (Amendment 2 item 8; the registered configuration is the first arm, any other arm is a labelled secondary)", "",
          "| arm | L1 at-fault collision zeros, seed mean (per seed) arm vs base | L1 | L2 difference [95 % CI by log] | by scene | L2 | L3 slow arm vs 1.1 x base | L3 | all three |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    st["lines"] = {}
    for g in a.arm:
        v, b = arms[g], arms["base"]
        c_arm, c_base = [S[k][2]["collision"] for k in v], [S[k][2]["collision"] for k in b]
        l1 = np.mean(c_arm) < np.mean(c_base) and all(x <= y for x, y in zip(c_arm, c_base))
        p_log, p_sc = stats.paired(rec[g], rec["base"], groups=logs), stats.paired(rec[g], rec["base"])
        l2 = p_log["mean"] >= 0 and p_log["lo"] > -0.005
        l3 = gm(g, "slow") <= 1.1 * gm("base", "slow")
        st["lines"][g] = dict(L1=bool(l1), collisions_arm=c_arm, collisions_base=c_base, L2=bool(l2), diff_log=p_log, diff_scene=p_sc, L3=bool(l3), slow_arm=gm(g, "slow"), slow_base=gm("base", "slow"),
                              per_seed={f"s{i}": stats.paired(sc[v[i]], sc[b[i]], groups=logs) for i in range(2)})
        L.append(f"| {g} | {np.mean(c_arm):g} ({c_arm[0]} / {c_arm[1]}) vs {np.mean(c_base):g} ({c_base[0]} / {c_base[1]}) | {'met' if l1 else 'not met'} | {p_log['mean']:+.4f} [{p_log['lo']:+.4f}, {p_log['hi']:+.4f}] | "
                 f"[{p_sc['lo']:+.4f}, {p_sc['hi']:+.4f}] | {'met' if l2 else 'not met'} | {gm(g, 'slow'):g} vs {1.1 * gm('base', 'slow'):.1f} (base {gm('base', 'slow'):g}) | {'met' if l3 else 'not met'} | **{'yes' if l1 and l2 and l3 else 'no'}** |")
    L += ["", "Per seed (arm seed i against baseline seed i, paired by scene, CI by log): " + "; ".join(
        f"{g} s{i} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" for g in a.arm for i, r in enumerate(st["lines"][g]["per_seed"].values())) + "."]
    # ---- flips and what happened to the baseline zeros
    flips = []
    L += ["", "## Zero / non-zero changes per seed", "", "| arm | seed | base zeros removed (collision / offroad / corridor) | new zeros (collision / offroad / corridor) | new zeros in scenes with a flag | base at-fault collisions: "
          "passed / still collision / other zero | of them with >= 1 flag |", "|:--|:--|:--|:--|--:|:--|:--|"]
    for g in a.arm:
        for i in range(2):
            k, b = arms[g][i], arms["base"][i]
            rem, new = (sc[b] == 0) & (sc[k] > 0), (sc[b] > 0) & (sc[k] == 0)
            bc = zc[b] == "collision_at_fault"
            cnt = lambda m, z: " / ".join(str(int((m & (z == f)).sum())) for f in R.FAIL)  # noqa: E731
            o3 = (int((bc & (sc[k] > 0)).sum()), int((bc & (zc[k] == "collision_at_fault")).sum()), int((bc & (sc[k] == 0) & (zc[k] != "collision_at_fault")).sum()))
            f3 = (int((bc & (sc[k] > 0) & (nfl[k] > 0)).sum()), int((bc & (zc[k] == "collision_at_fault") & (nfl[k] > 0)).sum()), int((bc & (sc[k] == 0) & (zc[k] != "collision_at_fault") & (nfl[k] > 0)).sum()))
            L.append(f"| {g} | s{i} | {int(rem.sum())} ({cnt(rem, zc[b])}) | {int(new.sum())} ({cnt(new, zc[k])}) | {int((new & (nfl[k] > 0)).sum())} | {o3[0]} / {o3[1]} / {o3[2]} | {f3[0]} / {f3[1]} / {f3[2]} |")
            for j in np.flatnonzero(rem | new | bc | (zc[k] == "collision_at_fault")):
                s = scenes[j]
                F = [x["body"] for x in Dd[k][0].get(s, []) if x.get("body") and x["body"]["flag"]]
                ks = [x["k"] for x in Dd[k][0].get(s, []) if x.get("body") and x["body"]["flag"]]
                flips.append(dict(arm=g, seed=i, scene=s, log=logs[j], turn4s_deg=round(float(turn[j]), 1), base_score=round(float(sc[b][j]), 4), base_zero=SHORT[zc[b][j]], arm_score=round(float(sc[k][j]), 4),
                                  arm_zero=SHORT[zc[k][j]], change="removed" if rem[j] else "new" if new[j] else "same", flags=len(F), flag_k="".join(map(str, ks)), base_flags_n=int(nfl[b][j]),
                                  min_d=min((y["d"] for y in F), default=""), max_a=max((y["a"] for y in F), default=""), v0_first=F[0]["v0"] if F else "", cut_sum=round(sum(y["cut"] for y in F), 2) if F else "",
                                  base_progress=round(float(prog[b][j]), 3), arm_progress=round(float(prog[k][j]), 3), arm_collision_rear=met(k, s, "collision_rear"),
                                  arm_collision_front=met(k, s, "collision_front"), arm_collision_lateral=met(k, s, "collision_lateral"), arm_dist_m=round(met(k, s, "dist_traveled_m"), 1),
                                  base_dist_m=round(met(b, s, "dist_traveled_m"), 1), base_min_obstacle_m=round(met(b, s, "min_distance_to_obstacle_m"), 2), arm_min_obstacle_m=round(met(k, s, "min_distance_to_obstacle_m"), 2)))
    if flips:
        with (out / "flips.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(flips[0]))
            w.writeheader(), w.writerows(flips)
    # ---- flags
    L += ["", "## Flags and what a flag costs", "", "| driver | decisions | flagged (%) | scenes with a flag (%) | flags that remove speed | at the 6 m/s^2 cap | ego under 0.5 m/s | median stop point (m) / deceleration / m removed in 2 s | "
          "flags by decision index 0..9 |", "|:--|--:|:--|:--|--:|--:|--:|:--|:--|"]
    for g in a.arm:
        for k in arms[g]:
            f = FS[k][0]
            L.append(f"| {k} | {f['decisions']} | {f['flags']} ({100 * f['flag_rate']:.2f}) | {f['scenes_flagged']} ({100 * f['scenes_flagged'] / f['scenes']:.1f}) | {f['removes_speed']} | {f['at_cap']} | {f['standing']} | "
                     f"{f['d_median']:.1f} / {f['a_median']:.2f} / {f['cut_median']:.2f} | {f['by_k']} |")
    L += ["", "| arm, seed | scenes with a flag: n, mean score arm / base, difference | scenes without a flag: n, mean arm / base, difference | flagged scenes 1.0 -> slow | unflagged 1.0 -> slow | slow -> 1.0 (all) | "
              "mean progress change in flagged scenes |", "|:--|:--|:--|--:|--:|--:|--:|"]
    for g in a.arm:
        for i in range(2):
            k, b = arms[g][i], arms["base"][i]
            f = nfl[k] > 0
            slow = lambda x: (x > 0) & (x < 1)  # noqa: E731
            L.append(f"| {g} s{i} | {int(f.sum())}, {sc[k][f].mean():.4f} / {sc[b][f].mean():.4f}, {(sc[k] - sc[b])[f].mean():+.4f} | {int((~f).sum())}, {sc[k][~f].mean():.4f} / {sc[b][~f].mean():.4f}, "
                     f"{(sc[k] - sc[b])[~f].mean():+.4f} | {int((f & (sc[b] >= 1) & slow(sc[k])).sum())} | {int((~f & (sc[b] >= 1) & slow(sc[k])).sum())} | {int((slow(sc[b]) & (sc[k] >= 1)).sum())} | "
                     f"{(prog[k] - prog[b])[f].mean():+.4f} |")
    # ---- latency
    L += ["", "## Driver latency per `drive` call (ms, whole call inside the driver; target 100)", "", "| driver | p50 | p90 | p99 | max | hook alone p50 / p90 |", "|:--|--:|--:|--:|--:|:--|"]
    for k in names:
        f = FS[k][0]
        L.append(f"| {k} | {f['total_ms_p50']:.1f} | {f['total_ms_p90']:.1f} | {f['total_ms_p99']:.1f} | {f['total_ms_max']:.0f} | " + (f"{f['hook_ms_p50']:.1f} / {f['hook_ms_p90']:.1f} |" if f["with_body"] else "- |"))
    # ---- > 45 deg
    L += ["", f"## Scenes whose logged 4 s future turns > 45 deg ({int(gt45.sum())} scenes, {len(set(logs[gt45]))} logs)", "", "| recipe | mean | zeros (seed mean) | collision / offroad / corridor | flagged scenes (seed mean) | difference to base [95 % CI, logs] |",
          "|:--|--:|--:|:--|--:|:--|"]
    st["gt45"] = {}
    for g, v in arms.items():
        z = [int((sc[k][gt45] == 0).sum()) for k in v]
        cls = " / ".join(f"{np.mean([(zc[k][gt45] == f).sum() for k in v]):g}" for f in R.FAIL)
        d = stats.paired(rec[g][gt45], rec["base"][gt45], groups=logs[gt45]) if g != "base" else None
        st["gt45"][g] = dict(mean=float(rec[g][gt45].mean()), zeros=z, diff=d)
        L.append(f"| {g} | {rec[g][gt45].mean():.4f} | {np.mean(z):g} | {cls} | {np.mean([(nfl[k][gt45] > 0).sum() for k in v]):g} | " + (f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] |" if d else "- |"))
    with (out / "per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "turn4s_deg"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress", "flags")])
        for j, s in enumerate(scenes):
            w.writerow([s, logs[j], round(float(turn[j]), 1)] + [x for k in names for x in (round(float(sc[k][j]), 6), SHORT[zc[k][j]], round(float(prog[k][j]), 4), int(nfl[k][j]))])
    with (out / "flags.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["driver", "scene", "k", "p", "z0", "z1", "v0", "s_reg", "s_step", "d", "D", "a", "cut", "scene_score", "base_score"])
        for g in a.arm:
            for i, k in enumerate(arms[g]):
                for s, v in Dd[k][0].items():
                    for x in v:
                        y = x.get("body")
                        if y and y["flag"] and s in Rs[k]:
                            w.writerow([k, s, x["k"], y["p"], *y["z"], y["v0"], y["s_reg"], y["s_step"], y["d"], y["D"], y["a"], y["cut"], Rs[k][s]["score"], Rs[arms["base"][i]].get(s, {}).get("score", "")])
    (out / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--man", required=True), p.add_argument("--base-man")
    p = sub.add_parser("check")
    p.add_argument("--run", required=True), p.add_argument("--base", required=True), p.add_argument("--out", required=True), p.add_argument("--scenes")
    p = sub.add_parser("report")
    p.add_argument("--base-man", required=True), p.add_argument("--man", nargs="+", required=True), p.add_argument("--arm", action="append", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pilot": cmd_pilot, "check": cmd_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
