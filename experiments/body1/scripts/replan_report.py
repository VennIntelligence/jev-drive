#!/usr/bin/env python3
"""BODY1 arm 4.2 closed-loop read (plans/2026-10-10-body1-prereg.md, section 4.4 + Amendment 3): the re-plan switch against P2H10-F.

  pilot   --man <pilot manifest> [--base-man <tr1 manifest>]      pilot8: off twice, on once -> per-scene table, body-record coverage
  check   --run <dir>[,<dir>] --base <dir>[,<dir>] --out DIR --expect LO,HI     the one-chunk checklist of Amendment 3 + the trace figure
  chunk   --run <dir> --base <dir> --out DIR      one run against the baseline run of the same scenes, descriptive (one seed)
  report  --base-man M --man M [M ...] --out DIR [--arm rp] [--dev chunk0.txt:0]      the full read on both readings: all scenes x seeds, and
          the part never run with a BODY1 switch (everything except the scenes of --dev's list in that seed) -> report.md, per_scene.csv,
          flips.csv, replans.csv, stats.json
Manifests are ot2_loop.py's {label: [run dirs]}; arm X uses the labels X-s0 / X-s1, the baseline P2H10-F-s0 / P2H10-F-s1.
Box, envs/op-train python. Scores come from aggregate/results-summary.json; decisions from the driver's drive.jsonl (`body`, written by
serve_body.replan); traces from the controller's per-rollout csv. > 45 deg = |heading change of the logged 4 s future of the scene's navtest
token| (bench TURN_BINS convention; op_parity cache lb_navtest), the scene-level definition of the stop arm's read.
Taught classes = at-fault collision + offroad + left corridor (the first failing flag of a zero scene).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import c0b_report as R  # noqa: E402
import stop_report as SR  # noqa: E402

BASE, SHORT, q = SR.BASE, SR.SHORT, SR.q
TAUGHT = ("collision", "offroad", "corridor")


def body(x):
    b = x.get("body")
    return b if b and b.get("rp") else None


def rp_stats(D):
    """Drive records {scene: [...]} -> (dict of re-plan facts, {scene: [served a per decision]})."""
    recs = [x for v in D.values() for x in v]
    b = [body(x) for x in recs]
    B = [y for y in b if y]
    rs = np.array([y["reason"] for y in B]) if B else np.zeros(0, str)
    per = {s: [(body(x) or {}).get("a", 0.0) for x in v] for s, v in D.items()}
    both = sum(1 for v in per.values() if any(a > 0 for a in v) and any(a < 0 for a in v))
    flip = 0                                                           # a side change within IDLE decisions: excluded by the rule
    for v in per.values():
        nz = [(i, np.sign(a)) for i, a in enumerate(v) if a]
        flip += sum(1 for (i, s0), (j, s1) in zip(nz, nz[1:]) if s0 != s1 and j - i <= 2)
    A = np.array([y["a"] for y in B if y["reason"] == "replan"])
    tot = [x["ms"]["total"] for x in recs]
    return dict(decisions=len(recs), with_body=len(B), scenes=len(D), replans=int((rs == "replan").sum()), replan_rate=float((rs == "replan").mean()) if len(rs) else 0.0,
                scenes_replanned=int(sum(any(v) for v in per.values())), none_clear=int((rs == "none_clear").sum()), cold_flagged=int(sum(y["reason"] == "cold" and bool(y["flag"]) for y in B)),
                flag_a=int(sum(y["flag"] == "a" for y in B)), flag_b=int(sum(y["flag"] == "b" for y in B)), flag_ab=int(sum(y["flag"] == "ab" for y in B)),
                by_k=[int(sum(y["reason"] == "replan" and y["k"] == k for y in B)) for k in range(10)], by_size={f"{a:g}": int((np.abs(A) == a).sum()) for a in (0.3, 0.6, 0.9, 1.2, 1.5)},
                left=int((A > 0).sum()), right=int((A < 0).sum()), shift_abs_mean=float(np.abs(A).mean()) if len(A) else 0.0, scenes_both_sides=both, side_flips_within_idle=flip,
                hook_ms_p50=q([y["ms"] for y in B], 0.5), hook_ms_p90=q([y["ms"] for y in B], 0.9), total_ms_p50=q(tot, 0.5), total_ms_p90=q(tot, 0.9), total_ms_p99=q(tot, 0.99),
                total_ms_max=max(tot) if tot else float("nan")), per


# ---------------------------------------------------------------- pilot
def cmd_pilot(a):
    man = json.loads(Path(a.man).read_text())
    base = json.loads(Path(a.base_man).read_text()) if a.base_man else {}
    Rs = {k: R.load_driver(v)[0] for k, v in man.items()}
    Ds = {k: SR.drives(v)[0] for k, v in man.items()}
    ref = R.load_driver(base[BASE[0]])[0] if base else {}
    scenes = sorted(set.intersection(*(set(r) for r in Rs.values())))
    print("| scene | " + " | ".join(man) + " | TR1 P2H10-F-s0 (in its chunk) | decisions with a re-plan record / re-planned (per label) |\n|:--|" + "--:|" * (len(man) + 2))
    for s in scenes:
        cov = "; ".join(f"{sum(body(x) is not None for x in Ds[k].get(s, []))}/{sum(bool(body(x) and body(x)['a']) for x in Ds[k].get(s, []))}" for k in man)
        print(f"| {s[-16:]} | " + " | ".join(f"{Rs[k][s]['score']:.4f} {SHORT[R.zclass(Rs[k][s])]}" for k in man) + f" | {ref[s]['score'] if s in ref else float('nan'):.4f} | {cov} |")
    ks = list(man)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = np.array([Rs[ks[i]][s]["score"] - Rs[ks[j]][s]["score"] for s in scenes])
            pr = np.array([(Rs[ks[i]][s]["score_metrics"].get("progress_clipped_rel") or 0) - (Rs[ks[j]][s]["score_metrics"].get("progress_clipped_rel") or 0) for s in scenes])
            print(f"{ks[i]} - {ks[j]}: scenes with a different score {int((np.abs(d) > 1e-9).sum())} of {len(scenes)}, max |score diff| {np.abs(d).max():.6f}, max |progress diff| {np.abs(pr).max():.6f}")
    for k in man:
        st, _ = rp_stats(Ds[k])
        print(k, json.dumps({x: st[x] for x in ("decisions", "with_body", "replans", "none_clear", "flag_a", "flag_b", "flag_ab", "total_ms_p50", "total_ms_p90", "hook_ms_p50", "hook_ms_p90")}))


# ---------------------------------------------------------------- one-chunk checklist
def yaw_of(c):
    return np.unwrap(2 * np.arctan2(c["qz"], c["qw"]))


def lat_to(cb, c):
    """Signed lateral offset (m, + = left) of every sample of trace c from the path of trace cb."""
    P, h = np.c_[cb["x"], cb["y"]], yaw_of(cb)
    out = np.empty(len(c))
    for i, p in enumerate(np.c_[c["x"], c["y"]]):
        j = int(np.argmin(((P - p) ** 2).sum(1)))
        out[i] = -np.sin(h[j]) * (p[0] - P[j, 0]) + np.cos(h[j]) * (p[1] - P[j, 1])
    return out


def traces(D, where, whereb, scenes, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, len(scenes), figsize=(3.6 * len(scenes), 7.6), sharex=True, squeeze=False)
    for j, s in enumerate(scenes):
        c, cb = SR.ctrl(*where[s]), SR.ctrl(*whereb[s]) if s in whereb else None
        t = c["timestamp_us"] * 1e-6
        ks, av = [x["now"] * 1e-6 for x in D[s]], [(body(x) or {}).get("a", 0.0) for x in D[s]]
        axs[0, j].bar(ks, av, width=0.4, color=["#D55E00" if a else "#bbbbbb" for a in av])
        axs[0, j].axhline(0, color="k", lw=0.5)
        if cb is not None:
            n = min(len(c), len(cb))
            axs[1, j].plot(t, lat_to(cb, c), color="#0072B2", lw=1.3)
            axs[2, j].plot(t[:n], np.degrees(yaw_of(c)[:n] - yaw_of(cb)[:n]), color="#0072B2", lw=1.3)
            axs[3, j].plot(cb["timestamp_us"] * 1e-6, cb["u_steering_angle"], color="#888888", lw=1.0, label="switch off (baseline run)")
        axs[3, j].plot(t, c["u_steering_angle"], color="#0072B2", lw=1.3, label="switch on")
        for i, lab in enumerate(("served shift at 4 s (m, + left)", "lateral offset from the baseline run's path (m)", "heading minus baseline run (deg)", "steering command (rad)")):
            axs[i, j].set_ylabel(lab, fontsize=7) if j == 0 else None
            axs[i, j].grid(alpha=0.25)
        axs[0, j].set_title(s[-16:], fontsize=9)
        axs[3, j].set_xlabel("simulated time (s)")
    axs[3, 0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def cmd_check(a):
    import c0b_chain as CH
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run, base = a.run.split(","), a.base.split(",")
    lo, hi = (float(x) for x in a.expect.split(","))
    Rr, Rb = R.load_driver(run)[0], R.load_driver(base)[0]
    D, where = SR.drives(run)
    Db, whereb = SR.drives(base)
    want = set(Path(a.scenes).read_text().split()) if a.scenes else set(Rb)
    s_, d_, ie, ip = CH.driver_health([Path(x) for x in run])
    st, per = rp_stats(D)
    sb, _ = rp_stats(Db)
    scenes = sorted(set(Rr) & set(Rb))
    sc, zc, sr = SR.summary(Rr, scenes)
    scb, zcb, sbs = SR.summary(Rb, scenes)
    rp = np.array([any(per.get(s, [])) for s in scenes])
    rem, new = (scb == 0) & (sc > 0), (scb > 0) & (sc == 0)
    fs = [s for s in scenes if any(per.get(s, []))]
    rev = [(SR.reversals(SR.ctrl(*where[s])["u_steering_angle"]), SR.reversals(SR.ctrl(*whereb[s])["u_steering_angle"]) if s in whereb else np.nan) for s in fs]
    # heading against the baseline run at the last decision (a scene-paired proxy; the registered heading item is read against the log on the Mac)
    dy = []
    for s in scenes:
        if s in D and s in Db and len(D[s]) == len(Db[s]) >= 10:
            dy.append((s, np.degrees(R_wrap(D[s][9]["anchor"][2] - Db[s][9]["anchor"][2])), bool(any(per[s]))))
    dyr = np.array([x[1] for x in dy if x[2]])
    chk = [("rollouts complete, no driver error, a re-plan record on every decision", f"{len(set(Rr) & want)} / {len(want)}; sessions {s_}, drive calls {d_}, inference errors {ie}, input errors {ip}; records {st['with_body']} / {st['decisions']}",
            len(set(Rr) & want) == len(want) and not ie and not ip and st["with_body"] == st["decisions"]),
           ("`drive` total ms p50 / p90 (<= 100)", f"{st['total_ms_p50']:.1f} / {st['total_ms_p90']:.1f} (p99 {st['total_ms_p99']:.1f}, max {st['total_ms_max']:.0f}); baseline run {sb['total_ms_p50']:.1f} / {sb['total_ms_p90']:.1f}; hook alone {st['hook_ms_p50']:.1f} / {st['hook_ms_p90']:.1f}",
            st["total_ms_p50"] <= 100 and st["total_ms_p90"] <= 100),
           (f"share of decisions re-planned ({100 * lo:.2f}-{100 * hi:.2f} %)", f"{st['replans']} / {st['decisions']} = {100 * st['replan_rate']:.2f} %; scenes re-planned {st['scenes_replanned']} / {st['scenes']}; by decision index {st['by_k']}; "
            f"by size {st['by_size']}; left / right {st['left']} / {st['right']}; flagged without a clear candidate {st['none_clear']}", lo <= st["replan_rate"] <= hi),
           ("no side change within 2 decisions", f"{st['side_flips_within_idle']} (scenes served both sides at any distance: {st['scenes_both_sides']})", st["side_flips_within_idle"] == 0),
           ("new zeros in re-planned scenes <= zeros removed", f"new zeros {int(new.sum())} (in re-planned scenes {int((new & rp).sum())}: {', '.join(SHORT[zc[i]] for i in np.flatnonzero(new & rp)) or '-'}); zeros removed {int(rem.sum())} "
            f"({', '.join(SHORT[zcb[i]] for i in np.flatnonzero(rem)) or '-'}; in re-planned scenes {int((rem & rp).sum())})", int((new & rp).sum()) <= int(rem.sum())),
           ("steering reversals in re-planned scenes (switch on / same scenes of the baseline run): no scene with >= 3 more", f"sum {sum(r[0] for r in rev)} / {np.nansum([r[1] for r in rev]) if rev else 0:.0f}; max per scene {max([r[0] for r in rev], default=0)} / {np.nanmax([r[1] for r in rev]) if rev else float('nan'):.0f}; "
            f"scenes with >= 3 more than the baseline run: {sum(1 for r in rev if r[0] - (r[1] if np.isfinite(r[1]) else 0) >= 3)}", not any(r[0] - (r[1] if np.isfinite(r[1]) else 0) >= 3 for r in rev)),
           ("heading at decision 9 minus the baseline run's, re-planned scenes (deg)", f"n {len(dyr)}, sd {dyr.std(ddof=1) if len(dyr) > 1 else float('nan'):.2f}, mean abs {np.abs(dyr).mean() if len(dyr) else float('nan'):.2f}, max abs {np.abs(dyr).max() if len(dyr) else float('nan'):.2f}", None)]
    L = ["| check | value | verdict |", "|:--|:--|:--|"] + [f"| {n} | {v} | {'pass' if p else ('FAIL' if p is not None else 'reported')} |" for n, v, p in chk]
    L += ["", "| run | n | mean | score 1 | zeros | collision | offroad | corridor | slow |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for n, s in (("switch on", sr), ("baseline run", sbs)):
        L.append(f"| {n} | {s['n']} | {s['mean']:.4f} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} |")
    d = sc - scb
    L += ["", f"Re-planned scenes: {int(rp.sum())}, mean score {sc[rp].mean() if rp.any() else float('nan'):.4f} against {scb[rp].mean() if rp.any() else float('nan'):.4f}; scenes never re-planned: {int((~rp).sum())}, "
              f"with a different score {int((np.abs(d[~rp]) > 1e-9).sum())}."]
    top = sorted(fs, key=lambda s: -sum(abs(x) for x in per[s]))[:3]
    if top:
        traces(D, where, whereb, top, out / "check_traces.png", "the three scenes with the largest summed served shift")
        L += ["", f"Traces: check_traces.png, scenes {', '.join(t[-16:] for t in top)}."]
    (out / "check.md").write_text("\n".join(L) + "\n")
    (out / "check.json").write_text(json.dumps(dict(replan=st, base=sb, checks=[(n, v, p) for n, v, p in chk], run=sr, base_scores=sbs, traces=top, heading_vs_base=dy), indent=1, default=float))
    print("\n".join(L))
    if any(p is False for _, _, p in chk):
        raise SystemExit("checklist FAILED")


def cmd_chunk(a):
    """Descriptive read of one run against the baseline run of the same scenes (one seed): difference, > 45 deg, every flagged scene, every zero."""
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Ra, Rb = R.load_driver(a.run.split(","))[0], R.load_driver(a.base.split(","))[0]
    D, Db = SR.drives(a.run.split(","))[0], SR.drives(a.base.split(","))[0]
    T = SR.turns()
    scenes = sorted(set(Ra) & set(Rb))
    logs = np.array([R.log_of(s) for s in scenes])
    sa, sb = np.array([Ra[s]["score"] for s in scenes]), np.array([Rb[s]["score"] for s in scenes])
    za, zb = np.array([SHORT[R.zclass(Ra[s])] for s in scenes]), np.array([SHORT[R.zclass(Rb[s])] for s in scenes])
    pa, pb = (np.array([(r[s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for s in scenes]) for r in (Ra, Rb))
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    g45 = np.abs(turn) > 45
    p, ps = stats.paired(sa, sb, groups=logs), stats.paired(sa, sb)
    rows, nrp = [], np.zeros(len(scenes), int)
    for i, s in enumerate(scenes):
        Bd = [body(x) or {} for x in D[s]]
        nrp[i] = sum(bool(y.get("a")) for y in Bd)
        if not any(y.get("flag") and y.get("reason") != "cold" for y in Bd) and sa[i] == sb[i] and sa[i] > 0:
            continue
        both = len(D[s]) > 9 and len(Db.get(s, [])) > 9
        x, y0 = (D[s][9]["anchor"], Db[s][9]["anchor"]) if both else (None, None)
        rows.append(dict(scene=s, log=logs[i], turn4s_deg=round(float(turn[i]), 1), base_score=round(float(sb[i]), 4), base_zero=zb[i], arm_score=round(float(sa[i]), 4), arm_zero=za[i],
                         base_progress=round(float(pb[i]), 3), arm_progress=round(float(pa[i]), 3), replans=int(nrp[i]), served=" ".join(f"{y.get('a', 0):g}" for y in Bd),
                         flags=" ".join(y.get("flag") or "-" for y in Bd), reasons=" ".join(y.get("reason", "?")[:2] for y in Bd), v0_first_replan=next((y["v0"] for y in Bd if y.get("a")), ""),
                         heading_minus_base_k9_deg=round(float(np.degrees(R_wrap(x[2] - y0[2]))), 2) if both else "", dist_to_base_k9_m=round(float(np.hypot(x[0] - y0[0], x[1] - y0[1])), 2) if both else ""))
    with (out / "flagged_scenes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(), w.writerows(rows)
    rp = nrp > 0
    slow = lambda v: int(((v > 0) & (v < 1)).sum())  # noqa: E731
    st = dict(scenes=len(scenes), logs=len(set(logs)), diff_log=p, diff_scene=ps, mean_arm=float(sa.mean()), mean_base=float(sb.mean()), slow_arm=slow(sa), slow_base=slow(sb),
              zeros_arm={c: int((za == c).sum()) for c in TAUGHT}, zeros_base={c: int((zb == c).sum()) for c in TAUGHT}, replanned_scenes=int(rp.sum()),
              replanned_diff_sum=float((sa - sb)[rp].sum()), replanned_better=int(((sa - sb)[rp] > 1e-9).sum()), replanned_worse=int(((sa - sb)[rp] < -1e-9).sum()), replanned_same=int((np.abs(sa - sb)[rp] <= 1e-9).sum()),
              gt45=dict(n=int(g45.sum()), mean_arm=float(sa[g45].mean()), mean_base=float(sb[g45].mean()), zeros_arm=int((sa[g45] == 0).sum()), zeros_base=int((sb[g45] == 0).sum()), replanned=int((rp & g45).sum()),
                        diff=stats.paired(sa[g45], sb[g45], groups=logs[g45])))
    (out / "chunk_stats.json").write_text(json.dumps(st, indent=1, default=float))
    print(json.dumps(st, default=float))
    for r in rows:
        if r["replans"] or r["base_score"] == 0 or r["arm_score"] == 0:
            print(r["scene"][-16:], r["turn4s_deg"], r["base_score"], r["base_zero"], "->", r["arm_score"], r["arm_zero"], "|", r["served"], "|", r["flags"], "| v0", r["v0_first_replan"], "dyaw", r["heading_minus_base_k9_deg"], "d", r["dist_to_base_k9_m"])


def R_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- the full read
def cmd_report(a):
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.base_man).read_text())
    for m in a.man:
        for k, v in json.loads(Path(m).read_text()).items():
            man[k] = (man[k] if k in man and k not in BASE else []) + v
    arm, base = [f"{a.arm}-s0", f"{a.arm}-s1"], list(BASE)
    names = base + arm
    Rs = {k: R.load_driver(man[k])[0] for k in names}
    Dd = {k: SR.drives(man[k]) for k in names}
    scenes = sorted(set.intersection(*(set(Rs[k]) for k in names)))
    n = len(scenes)
    logs = np.array([R.log_of(s) for s in scenes])
    T = SR.turns()
    turn = np.array([T.get(s.rsplit("-", 1)[1], np.nan) for s in scenes])
    gt45 = np.abs(turn) > 45
    S = {k: SR.summary(Rs[k], scenes) for k in names}
    sc, zc = {k: S[k][0] for k in names}, {k: np.array([SHORT[z] for z in S[k][1]]) for k in names}
    prog = {k: np.array([(Rs[k][s].get("score_metrics") or {}).get("progress_clipped_rel") or 0.0 for s in scenes]) for k in names}
    PS = {k: rp_stats(Dd[k][0]) for k in arm}
    nrp = {k: np.array([sum(1 for x in PS[k][1].get(s, []) if x) for s in scenes]) for k in arm}
    dev_f, dev_seed = a.dev.split(":")
    dev = set(Path(dev_f).read_text().split())
    fresh = np.ones((2, n), bool)                                       # (seed, scene): never run with any BODY1 switch before this arm
    fresh[int(dev_seed)] = np.array([s not in dev for s in scenes])
    slow = lambda x: (x > 0) & (x < 1)  # noqa: E731
    taught = lambda z: np.isin(z, TAUGHT)  # noqa: E731
    st = dict(scenes=n, logs=len(set(logs)), gt45=int(gt45.sum()), turn_missing=int(np.isnan(turn).sum()), fresh_pairs=int(fresh.sum()), drivers={k: S[k][2] | dict(mean_ci=R.ci(sc[k], logs)) for k in names},
              replan={k: PS[k][0] for k in arm})
    L = [f"Scenes common to all runs: {n} from {len(set(logs))} logs; logged 4 s turn > 45 deg: {int(gt45.sum())} (token not in lb_navtest: {int(np.isnan(turn).sum())}). "
         f"Never run with a BODY1 switch before: {int(fresh.sum())} of {2 * n} (seed, scene) pairs.", "",
         "## Drivers", "", "| driver | n | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress |", "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k in names:
        s = S[k][2]
        L.append(f"| {k} | {s['n']} | {R.fmt(R.ci(sc[k], logs))} | {s['ones']} | {s['zeros']} | {s['collision']} | {s['offroad']} | {s['corridor']} | {s['slow']} | {prog[k].mean():.3f} |")
    # ---- lines on both readings
    L += ["", "## Lines (Amendment 3; the stricter of the two readings decides)", "",
          "| reading | (seed, scene) pairs | L1 taught-class zeros arm vs base (collision / offroad / corridor), per seed | L1 | L2 mean difference [95 % CI by log] | by scene | L2 | L3 slow arm vs 1.1 x base | L3 | all three |",
          "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|"]
    st["lines"] = {}
    for name, M in (("all", np.ones((2, n), bool)), ("never run with a switch", fresh)):
        w = M.sum(0)
        use = w > 0
        d = np.where(use, sum(np.where(M[i], sc[arm[i]] - sc[base[i]], 0.0) for i in range(2)) / np.maximum(w, 1), 0.0)[use]      # per-scene mean difference over the seeds in the reading
        ta, tb = [int((taught(zc[arm[i]]) & M[i]).sum()) for i in range(2)], [int((taught(zc[base[i]]) & M[i]).sum()) for i in range(2)]
        cls = lambda ks: " / ".join(str(sum(int(((zc[ks[i]] == f) & M[i]).sum()) for i in range(2))) for f in TAUGHT)  # noqa: E731
        l1 = sum(ta) < sum(tb) and all(x <= y for x, y in zip(ta, tb))
        p_log, p_sc = stats.paired(d, np.zeros_like(d), groups=logs[use]), stats.paired(d, np.zeros_like(d))
        l2 = p_log["mean"] >= 0 and p_log["lo"] > -0.005
        sa, sb_ = sum(int((slow(sc[arm[i]]) & M[i]).sum()) for i in range(2)), sum(int((slow(sc[base[i]]) & M[i]).sum()) for i in range(2))
        l3 = sa <= 1.1 * sb_
        st["lines"][name] = dict(pairs=int(M.sum()), scenes=int(use.sum()), L1=bool(l1), taught_arm=ta, taught_base=tb, L2=bool(l2), diff_log=p_log, diff_scene=p_sc, L3=bool(l3), slow_arm=sa, slow_base=sb_,
                                 all=bool(l1 and l2 and l3))
        L.append(f"| {name} | {int(M.sum())} | {sum(ta)} ({cls(arm)}) vs {sum(tb)} ({cls(base)}); s0 {ta[0]} vs {tb[0]}, s1 {ta[1]} vs {tb[1]} | {'met' if l1 else 'not met'} | {p_log['mean']:+.4f} [{p_log['lo']:+.4f}, {p_log['hi']:+.4f}] | "
                 f"[{p_sc['lo']:+.4f}, {p_sc['hi']:+.4f}] | {'met' if l2 else 'not met'} | {sa} vs {1.1 * sb_:.1f} (base {sb_}) | {'met' if l3 else 'not met'} | **{'yes' if l1 and l2 and l3 else 'no'}** |")
    st["verdict"] = bool(all(v["all"] for v in st["lines"].values()))
    ps = [stats.paired(sc[arm[i]], sc[base[i]], groups=logs) for i in range(2)]
    st["per_seed"] = ps
    L += ["", f"Counts are sums over the (seed, scene) pairs of the reading (two-seed mean = half of the `all` row). Verdict: **{'all three lines met on both readings' if st['verdict'] else 'not met'}**. "
          "Per seed (paired by scene, CI by log): " + "; ".join(f"s{i} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" for i, r in enumerate(ps)) + "."]
    # ---- flips
    flips = []
    L += ["", "## Zero / non-zero changes per seed", "", "| seed | base zeros removed (collision / offroad / corridor) | of them in re-planned scenes | new zeros (collision / offroad / corridor) | of them in re-planned scenes | "
          "base taught zeros: passed / still zero | of them re-planned at least once |", "|:--|:--|--:|:--|--:|:--|:--|"]
    for i in range(2):
        k, b = arm[i], base[i]
        rem, new, rp = (sc[b] == 0) & (sc[k] > 0), (sc[b] > 0) & (sc[k] == 0), nrp[k] > 0
        cnt = lambda m, z: " / ".join(str(int((m & (z == f)).sum())) for f in TAUGHT)  # noqa: E731
        bz = taught(zc[b])
        L.append(f"| s{i} | {int(rem.sum())} ({cnt(rem, zc[b])}) | {int((rem & rp).sum())} | {int(new.sum())} ({cnt(new, zc[k])}) | {int((new & rp).sum())} | {int((bz & (sc[k] > 0)).sum())} / {int((bz & (sc[k] == 0)).sum())} | "
                 f"{int((bz & (sc[k] > 0) & rp).sum())} / {int((bz & (sc[k] == 0) & rp).sum())} |")
        for j in np.flatnonzero(rem | new | bz | taught(zc[k])):
            s = scenes[j]
            B = [body(x) for x in Dd[k][0].get(s, [])]
            flips.append(dict(seed=i, scene=s, log=logs[j], fresh=bool(fresh[i, j]), turn4s_deg=round(float(turn[j]), 1), base_score=round(float(sc[b][j]), 4), base_zero=zc[b][j], arm_score=round(float(sc[k][j]), 4),
                              arm_zero=zc[k][j], change="removed" if rem[j] else "new" if new[j] else "same", replans=int(nrp[k][j]), served=" ".join(f"{(y or {}).get('a', 0):g}" for y in B),
                              flags=" ".join((y or {}).get("flag", "") or "-" for y in B), reasons=" ".join((y or {}).get("reason", "?")[:2] for y in B),
                              base_progress=round(float(prog[b][j]), 3), arm_progress=round(float(prog[k][j]), 3)))
    if flips:
        with (out / "flips.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(flips[0]))
            w.writeheader(), w.writerows(flips)
    # ---- re-plans
    L += ["", "## Re-plans", "", "| driver | decisions | re-planned (%) | scenes re-planned (%) | flags agent / boundary / both | flagged, none clear | flagged at the cold start (not acted on) | by size 0.3 / 0.6 / 0.9 / 1.2 / 1.5 m | left / right | "
          "scenes served both sides | by decision index 0..9 |", "|:--|--:|:--|:--|:--|--:|--:|:--|:--|--:|:--|"]
    for k in arm:
        f = PS[k][0]
        L.append(f"| {k} | {f['decisions']} | {f['replans']} ({100 * f['replan_rate']:.2f}) | {f['scenes_replanned']} ({100 * f['scenes_replanned'] / f['scenes']:.1f}) | {f['flag_a']} / {f['flag_b']} / {f['flag_ab']} | {f['none_clear']} | {f['cold_flagged']} | "
                 f"{' / '.join(str(v) for v in f['by_size'].values())} | {f['left']} / {f['right']} | {f['scenes_both_sides']} | {f['by_k']} |")
    L += ["", "| seed | re-planned scenes: n, mean score arm / base, difference | never re-planned: n, scenes with a different score, mean difference | re-planned 1.0 -> slow | re-planned slow -> 1.0 | mean progress change in re-planned scenes |",
          "|:--|:--|:--|--:|--:|--:|"]
    for i in range(2):
        k, b = arm[i], base[i]
        f = nrp[k] > 0
        d = sc[k] - sc[b]
        L.append(f"| s{i} | {int(f.sum())}, {sc[k][f].mean():.4f} / {sc[b][f].mean():.4f}, {d[f].mean():+.4f} | {int((~f).sum())}, {int((np.abs(d[~f]) > 1e-9).sum())}, {d[~f].mean():+.5f} | "
                 f"{int((f & (sc[b] >= 1) & slow(sc[k])).sum())} | {int((f & slow(sc[b]) & (sc[k] >= 1)).sum())} | {(prog[k] - prog[b])[f].mean():+.4f} |")
    # ---- latency
    L += ["", "## Driver latency per `drive` call (ms, whole call inside the driver; target 100)", "", "| driver | p50 | p90 | p99 | max | hook alone p50 / p90 |", "|:--|--:|--:|--:|--:|:--|"]
    for k in names:
        f = PS[k][0] if k in PS else rp_stats(Dd[k][0])[0]
        L.append(f"| {k} | {f['total_ms_p50']:.1f} | {f['total_ms_p90']:.1f} | {f['total_ms_p99']:.1f} | {f['total_ms_max']:.0f} | " + (f"{f['hook_ms_p50']:.1f} / {f['hook_ms_p90']:.1f} |" if f["with_body"] else "- |"))
    # ---- > 45 deg
    rec = {"base": (sc[base[0]] + sc[base[1]]) / 2, a.arm: (sc[arm[0]] + sc[arm[1]]) / 2}
    L += ["", f"## Scenes whose logged 4 s future turns > 45 deg ({int(gt45.sum())} scenes, {len(set(logs[gt45]))} logs)", "", "| recipe | mean | zeros (sum of the two seeds) | collision / offroad / corridor | re-planned scenes (sum) | difference to base [95 % CI, logs] |",
          "|:--|--:|--:|:--|--:|:--|"]
    st["gt45_read"] = {}
    for g, v in (("base", base), (a.arm, arm)):
        z = [int((sc[k][gt45] == 0).sum()) for k in v]
        cls = " / ".join(str(sum(int((zc[k][gt45] == f).sum()) for k in v)) for f in TAUGHT)
        d = stats.paired(rec[g][gt45], rec["base"][gt45], groups=logs[gt45]) if g != "base" else None
        st["gt45_read"][g] = dict(mean=float(rec[g][gt45].mean()), zeros=z, diff=d)
        L.append(f"| {g} | {rec[g][gt45].mean():.4f} | {sum(z)} | {cls} | {sum(int((nrp[k][gt45] > 0).sum()) for k in v) if g != 'base' else '-'} | " + (f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] |" if d else "- |"))
    with (out / "per_scene.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "log", "turn4s_deg", "fresh_s0", "fresh_s1"] + [f"{k}|{c}" for k in names for c in ("score", "zero_class", "progress")] + [f"{k}|replans" for k in arm] + [f"{k}|served" for k in arm])
        for j, s in enumerate(scenes):
            w.writerow([s, logs[j], round(float(turn[j]), 1), int(fresh[0, j]), int(fresh[1, j])] + [x for k in names for x in (round(float(sc[k][j]), 6), zc[k][j], round(float(prog[k][j]), 4))]
                       + [int(nrp[k][j]) for k in arm] + [" ".join(f"{x:g}" for x in PS[k][1].get(s, [])) for k in arm])
    with (out / "replans.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["driver", "scene", "k", "reason", "flag", "a", "side", "za_plan", "zb_plan", "za_served", "zb_served", "v0", "n_candidates", "scene_score", "base_score"])
        for i, k in enumerate(arm):
            for s, v in Dd[k][0].items():
                for x in v:
                    y = body(x)
                    if y and y["flag"] and s in Rs[k]:
                        import serve_body as SB
                        j = int(np.flatnonzero(SB.A_RP == y["a"])[0]) + 1 if y["a"] else 0
                        w.writerow([k, s, x["k"], y["reason"], y["flag"], y["a"], y["side"], y["za"][0], y["zb"][0], y["za"][j], y["zb"][j], y["v0"], sum(y["ok"]), Rs[k][s]["score"], Rs[base[i]].get(s, {}).get("score", "")])
    (out / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--man", required=True), p.add_argument("--base-man")
    p = sub.add_parser("check")
    p.add_argument("--run", required=True), p.add_argument("--base", required=True), p.add_argument("--out", required=True), p.add_argument("--scenes"), p.add_argument("--expect", required=True)
    p = sub.add_parser("chunk")
    p.add_argument("--run", required=True), p.add_argument("--base", required=True), p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--base-man", required=True), p.add_argument("--man", nargs="+", required=True), p.add_argument("--arm", default="rp"), p.add_argument("--out", required=True)
    p.add_argument("--dev", required=True, help="<scene list>:<seed> = the part already read with a BODY1 switch (chunk0 x s0, decision 224)")
    a = ap.parse_args()
    {"pilot": cmd_pilot, "check": cmd_check, "report": cmd_report, "chunk": cmd_chunk}[a.cmd](a)


if __name__ == "__main__":
    main()
