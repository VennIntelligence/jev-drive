"""Readouts of the P2 on-policy lane (plans/2026-10-06-p2-onpolicy-prereg.md) -> $DATA_DIR/runs/factor_wm/p2op/report/ (op-train env).

  check   after seed 0's first collection round: labelled states, events, NaNs of roll/xr1-s0 (exit 1 on failure)
  gate    the seed-0 early-stop gate (prereg section 5) -> gate.json / gate.md; exit 3 = stop the lane
  report  every table: navtest (W, G), navhard two-stage (G, W), HUGSIM 64 (exam, spec), engine (g0b, g1s) -> tables.md, report.json
Arms: P2 = op_parity P2-F-s{0,1}; C = PC-s{0,1} (off-policy control); X = PX-s{0,1} (on-policy). Seed means per unit, paired cluster
bootstraps (jevdrive.stats, B 10 000): navtest over logs, navhard over the stage-1 logs of the 225 groups, HUGSIM over scenarios.
HUGSIM rows come from pp_hugsim_report.py extract full (FULL_OUT = report/hugsim) run by the lane in envs/hugsim.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
D = DD / "runs/factor_wm/p2op"
OUT = D / "report"
ARMS = {"P2": ["P2-F-s0", "P2-F-s1"], "C": ["PC-s0", "PC-s1"], "X": ["PX-s0", "PX-s1"]}
HR = _R / "experiments/hugsim/results"
H12 = [_pl.Path(p).stem for p in open(_R / "experiments/factor_wm/scripts/p2op_hug12.txt").read().split()]
V2 = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound"}


def boot(a, b, groups=None):
    from jevdrive import stats
    r = stats.paired(a, b, groups=groups)
    return {k: float(r[k]) for k in ("mean", "lo", "hi", "mean_a", "mean_b")} | {"n": int(r.get("n", len(a)))}


def f(r, k=2):
    return f"{r['mean']:+.{k}f} [{r['lo']:+.{k}f}, {r['hi']:+.{k}f}]"


# ---------------------------------------------------------------- navtest
def navtest(frames, seeds=(0, 1)):
    """{arm: per-token frame (seed mean of score and subscores, x 100)}, WA-JEPA included; token order common; groups = logs."""
    import pp_eval as E
    E.DATA, E.FRAMES = "lb_navtest", frames
    cols = ["score"] + list(V2.values())
    per = {}
    for arm, ms in ARMS.items():
        got = [E.read_csv(E.eval_csv(m))[0][cols].astype(float) * 100 for i, m in enumerate(ms) if i in seeds and E.eval_csv(m) is not None]
        if len(got) == len(seeds):
            per[arm] = sum(got) / len(got)
    per["WA-JEPA"] = E.read_csv(DD / E.WAJEPA_CSV)[0][cols].astype(float) * 100
    toks = sorted(set.intersection(*[set(t.index) for t in per.values()]))
    tab = np.load(DD / "runs/op_parity/cache/lb_navtest/tab.npz")
    lg = dict(zip(tab["names"].tolist(), tab["log"].tolist()))
    return {k: v.loc[toks] for k, v in per.items()}, np.array([lg[t] for t in toks])


# ---------------------------------------------------------------- navhard
def nh_dir(frames, m):
    if m == "WA-JEPA":
        return DD / "runs/op_parity/navhard/harness/wajepa"
    if m == "S3":
        return DD / "runs/op_guard/fw-S3/nav/navhard"
    if m.startswith("P2-F") or m == "P0":
        return DD / ("runs/op_parity/navhard_gimm/harness" if frames == "gimm" else "runs/op_parity/navhard/harness") / m
    return D / "navhard" / frames / m


def navhard(frames, seeds=(0, 1)):
    import pandas as pd
    from jevdrive import navsim_zs as Z
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navhard_two_stage", slim=True)}
    per, g0 = {}, None
    want = {a: [m for i, m in enumerate(ms) if i in seeds] for a, ms in ARMS.items()} | {"WA-JEPA": ["WA-JEPA"], "S3": ["S3"], "P0": ["P0"]}
    for arm, ms in want.items():
        ds = [nh_dir(frames, m) / "harness_groups.csv" for m in ms]
        if not all(p.exists() for p in ds):
            continue
        ts = [pd.read_csv(p).set_index("group") for p in ds]
        g0 = ts[0] if g0 is None else g0
        assert all((t.orig.values == g0.orig.values).all() for t in ts), "group order differs"
        per[arm] = sum(t[["combined", "stage1", "stage2"]].astype(float) for t in ts) / len(ts)
    return per, np.array([lg.get(o, "?") for o in g0.orig])


# ---------------------------------------------------------------- HUGSIM
def hugsim(scen=None, seeds=(0, 1)):
    """{preset: {arm: per-scenario frame (seed means: hd, stuck, stall, spin, fg, bg, complete)}}, WA-JEPA (one run, both presets)."""
    import pandas as pd
    ex = pd.read_csv(OUT / "hugsim" / "extract.csv")
    ex["stuck"] = (ex.end == "max_steps").astype(float)
    ex["stall"] = (ex.v_max40 < 1.6).astype(float)
    ex["spin"] = ex.spin.astype(bool).astype(float)
    for k, e in (("fg", "fg_collision"), ("bg", "bg_collision"), ("complete", "complete")):
        ex[k] = (ex.end == e).astype(float)
    ex = ex.rename(columns={"hdscore": "hd"})
    cols = ["hd", "stuck", "stall", "spin", "fg", "bg", "complete"]
    W = pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    WX = pd.read_csv(HR / "wajepa_ref/wajepa_extract.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    sc = list(W.index) if scen is None else list(scen)
    w = pd.DataFrame({"hd": W.hdscore, "stuck": (W.end == "max_steps").astype(float), "stall": (WX.v_max40 < 1.6).astype(float),
                      "spin": WX.spin.astype(bool).astype(float), "fg": (W.end == "fg_collision").astype(float),
                      "bg": (W.end == "bg_collision").astype(float), "complete": (W.end == "complete").astype(float)}).reindex(sc)
    out = {}
    for pr in ("exam", "spec"):
        out[pr] = {"WA-JEPA": w}
        for arm, ms in ARMS.items():
            ts = [ex[(ex.preset == pr) & (ex.arm == m)].drop_duplicates("scenario", keep="last").set_index("scenario")[cols].reindex(sc)
                  for i, m in enumerate(ms) if i in seeds]
            if ts and all(t.hd.notna().all() for t in ts):
                out[pr][arm] = sum(ts) / len(ts)
    return out


# ---------------------------------------------------------------- engine (WOD val, plan engine)
def engine(m):
    import fw_common as C
    r = {}
    for name in ("g0b", "g1s"):
        p = C.root("p2op", "roll", f"ev-{m}") / f"{name}-eval-0of1.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        ev, arm, cat = z["event"].astype(str), z["arm"].astype(str), z["cat"].astype(str)
        if name == "g0b":
            pert = arm != "free"
            for k in ("stall", "heading", "lane"):
                r[f"g0b_{k}"] = float(np.mean(ev[pert] == k))
            r["g0b_fail"] = float(np.mean(np.isin(ev[pert], ["stall", "heading", "lane"])))
            r["g0b_launch_stall"] = float(np.mean(ev[pert & (cat == "launch")] == "stall"))
        else:
            moved = z["vs"].sum(1) * C.DT
            r["g1s_false_go"] = float(np.mean((moved > 2.0) | (ev == "ahead")))
    return r


# ---------------------------------------------------------------- commands
def cmd_check(a):
    import fw_common as C
    fs = sorted(C.root("p2op", "roll", "xr1-s0").glob("train-collect-[0-9]*of*.npz"))
    assert len(fs) == 3, f"expected 3 shards, got {len(fs)}"
    n, ok, bad, evs = 0, 0, 0, {}
    for p in fs:
        z = np.load(p, allow_pickle=True)
        tok = np.load(p.with_suffix(".tok.npy"), mmap_mode="r")
        assert tok.shape[0] == len(z["c"]), "token rows != rollouts"
        n += len(z["c"])
        ok += int(z["ok"].sum())
        bad += int((~np.isfinite(z["fut8"][z["ok"]])).sum() + (~np.isfinite(z["ego"])).sum() + (~np.isfinite(tok[:: 50].astype(np.float32))).sum())
        free = z["arm"].astype(str) == "free"
        for cat in np.unique(z["cat"]):
            m = free & (z["cat"].astype(str) == cat)
            e = z["event"][m].astype(str)
            evs.setdefault(str(cat), []).extend(e.tolist())
    rates = {c: {k: round(float(np.mean(np.array(v) == k)), 3) for k in ("stall", "heading", "lane", "behind", "ahead", "")} for c, v in evs.items()}
    res = {"rollouts": n, "labelled_states": ok, "nonfinite": bad, "free_arm_events": rates}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if bad or ok < 20000:
        raise SystemExit("check failed")


def cmd_gate(a):
    lines, res, stop = [], {}, []
    nt, g = navtest("warp", seeds=(0,))
    r = boot(nt["X"].score.values, nt["P2"].score.values, g)
    res["navtest_W_X-P2"] = r
    res["navtest_W_C-P2"] = boot(nt["C"].score.values, nt["P2"].score.values, g)
    res["navtest_W_X-C"] = boot(nt["X"].score.values, nt["C"].score.values, g)
    if r["mean"] < -1.0:
        stop.append(f"(a) navtest X - P2 {r['mean']:+.2f} < -1.0")
    nh, gl = navhard("gimm", seeds=(0,))
    r = boot(nh["X"].combined.values, nh["P2"].combined.values, gl)
    res["navhard_G_X-P2"] = r
    res["navhard_G_X-C"] = boot(nh["X"].combined.values, nh["C"].combined.values, gl)
    if r["mean"] < -3.0:
        stop.append(f"(d) navhard G X - P2 {r['mean']:+.2f} < -3.0")
    hs = hugsim(H12, seeds=(0,))
    for pr, arms in hs.items():
        x, p2, c = arms["X"], arms["P2"], arms["C"]
        for k in ("stall", "stuck"):
            res[f"hug12_{pr}_{k}"] = {"X": float(x[k].sum()), "P2": float(p2[k].sum()), "C": float(c[k].sum())}
            if x[k].sum() > p2[k].sum() + 2:
                stop.append(f"(b) HUGSIM-12 {pr} {k} X {x[k].sum():.0f} > P2 {p2[k].sum():.0f} + 2")
        d = float(x.hd.mean() - p2.hd.mean())
        res[f"hug12_{pr}_hd"] = {"X": float(x.hd.mean()), "P2": float(p2.hd.mean()), "C": float(c.hd.mean()), "X-P2": d}
        if d < -0.10:
            stop.append(f"(c) HUGSIM-12 {pr} HD X - P2 {d:+.3f} < -0.10")
    res["engine"] = {m: engine(m) for m in ("P2-F-s0", "PC-s0", "PX-s0")}
    res["stop"] = stop
    res["pass"] = not stop
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(res, indent=1))
    lines += ["# Gate (seed 0)", "", f"verdict: {'PASS' if not stop else 'STOP'}", ""] + [f"- {s}" for s in stop]
    lines += ["", "| read | X - P2 | C - P2 | X - C |", "|:--|:--|:--|:--|",
              f"| navtest W EPDMS | {f(res['navtest_W_X-P2'])} | {f(res['navtest_W_C-P2'])} | {f(res['navtest_W_X-C'])} |",
              f"| navhard G combined | {f(res['navhard_G_X-P2'])} | | {f(res['navhard_G_X-C'])} |"]
    for pr in ("exam", "spec"):
        h = res[f"hug12_{pr}_hd"]
        lines.append(f"| HUGSIM-12 {pr} HD (X / P2 / C) | {h['X']:.3f} / {h['P2']:.3f} / {h['C']:.3f} | | |")
        for k in ("stall", "stuck"):
            v = res[f"hug12_{pr}_{k}"]
            lines.append(f"| HUGSIM-12 {pr} {k} (X / P2 / C) | {v['X']:.0f} / {v['P2']:.0f} / {v['C']:.0f} | | |")
    lines += ["", "engine: " + json.dumps(res["engine"])]
    (OUT / "gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    raise SystemExit(0 if not stop else 3)


def cmd_report(a):
    L, R = [], {}
    P_ = L.append
    pairs = [("X", "C"), ("X", "P2"), ("C", "P2"), ("X", "WA-JEPA"), ("C", "WA-JEPA"), ("P2", "WA-JEPA")]
    # navtest
    for frames in ("warp", "gimm"):
        nt, g = navtest(frames)
        P_(f"## navtest EPDMS, protocol {'W' if frames == 'warp' else 'G'} (seed means; 12 146 tokens; cluster bootstrap over {len(set(g))} logs)")
        P_("")
        P_("| arm | EPDMS | " + " | ".join(V2) + " |")
        P_("|:--|--:|" + "--:|" * len(V2))
        for k, v in nt.items():
            P_(f"| {k} | {v.score.mean():.2f} | " + " | ".join(f"{v[c].mean():.2f}" for c in V2.values()) + " |")
            R[f"navtest_{frames}_{k}"] = {"EPDMS": float(v.score.mean())} | {s: float(v[c].mean()) for s, c in V2.items()}
        P_("")
        P_("| contrast | EPDMS [95% CI] | dNC | dDAC | dEP | dTTC |")
        P_("|:--|:--|--:|--:|--:|--:|")
        for x, y in pairs:
            if x in nt and y in nt:
                r = boot(nt[x].score.values, nt[y].score.values, g)
                R[f"navtest_{frames}_{x}-{y}"] = r
                P_(f"| {x} - {y} | {f(r)} | " + " | ".join(f"{nt[x][c].mean() - nt[y][c].mean():+.2f}" for c in V2.values()) + " |")
        P_("")
    # navhard
    for frames in ("gimm", "warp"):
        nh, gl = navhard(frames)
        P_(f"## navhard two-stage EPDMS, protocol {'G' if frames == 'gimm' else 'W'} (seed means; 225 groups, cluster bootstrap over {len(set(gl))} logs)")
        P_("")
        P_("| arm | combined | stage 1 | stage 2 |")
        P_("|:--|--:|--:|--:|")
        for k, v in nh.items():
            P_(f"| {k} | {v.combined.mean():.2f} | {v.stage1.mean():.2f} | {v.stage2.mean():.2f} |")
            R[f"navhard_{frames}_{k}"] = {c: float(v[c].mean()) for c in ("combined", "stage1", "stage2")}
        P_("")
        P_("| contrast | combined [95% CI] | stage 1 | stage 2 |")
        P_("|:--|:--|:--|:--|")
        for x, y in pairs + [("X", "S3")]:
            if x in nh and y in nh:
                rs = {c: boot(nh[x][c].values, nh[y][c].values, gl) for c in ("combined", "stage1", "stage2")}
                R[f"navhard_{frames}_{x}-{y}"] = rs
                P_(f"| {x} - {y} | {f(rs['combined'])} | {f(rs['stage1'])} | {f(rs['stage2'])} |")
        P_("")
    # HUGSIM
    hs = hugsim()
    for pr, arms in hs.items():
        P_(f"## HUGSIM 64, preset {pr} (seed means of one run per scenario; WA-JEPA one run, shared by both presets)")
        P_("")
        P_("| arm | HD | complete | stuck | launch stall | spin | fg coll | bg coll |")
        P_("|:--|--:|--:|--:|--:|--:|--:|--:|")
        for k, v in arms.items():
            P_(f"| {k} | {v.hd.mean():.3f} | " + " | ".join(f"{v[c].sum():.1f}" for c in ("complete", "stuck", "stall", "spin", "fg", "bg")) + " |")
            R[f"hugsim_{pr}_{k}"] = {"HD": float(v.hd.mean())} | {c: float(v[c].sum()) for c in ("complete", "stuck", "stall", "spin", "fg", "bg")}
        P_("")
        P_("| contrast | HD [95% CI] (unit = scenario) |")
        P_("|:--|:--|")
        for x, y in pairs:
            if x in arms and y in arms:
                r = boot(arms[x].hd.values, arms[y].hd.values)
                R[f"hugsim_{pr}_{x}-{y}"] = r
                P_(f"| {x} - {y} | {f(r, 3)} |")
        P_("")
    # engine
    P_("## Engine (WOD val, plan engine): perturbed-arm failure rates on g0b, false go on g1s")
    P_("")
    P_("| model | g0b fail | stall | heading | lane | launch stall | g1s false go |")
    P_("|:--|--:|--:|--:|--:|--:|--:|")
    for arm, ms in ARMS.items():
        for m in ms:
            e = engine(m)
            if e:
                R[f"engine_{m}"] = e
                P_(f"| {m} | {e['g0b_fail']:.3f} | {e['g0b_stall']:.3f} | {e['g0b_heading']:.3f} | {e['g0b_lane']:.3f} | {e['g0b_launch_stall']:.3f} | {e['g1s_false_go']:.3f} |")
    P_("")
    dev = {m: json.loads((D / "runs" / m / "dev.json").read_text()) for ms in ARMS.values() for m in ms if (D / "runs" / m / "dev.json").exists()}
    R["dev"] = dev
    P_("navtrain dev (inputs on ADE / drift to shipped inputs off, m): " + "; ".join(f"{m} {v['ade']:.3f} / {v['drift_off']:.3f}" for m, v in dev.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    (OUT / "report.json").write_text(json.dumps(R, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "gate", "report"])
    a = ap.parse_args()
    {"check": cmd_check, "gate": cmd_gate, "report": cmd_report}[a.cmd](a)
