"""HUGSIM ax probe (results/hugsim_ax_probe.md): does feeding the sim's acceleration as ax drive P2H's fast entries into sharp turns?

  sets     (any)        scenario sets by the rule in the result file -> results/hugsim_ax_probe_sets.csv
  offline  (box, op-train)  stored HUGSIM ego features of P2H runs through the torch port, ax as fed vs ax = 0 -> results/hugsim_ax_probe_offline.csv
  launch   (box)        prints the two bench commands (unmodified / ax = 0)
  report   (box, .venv) paired readouts of the two arms from the bench run dirs -> results/hugsim_ax_probe_{runs,paired}.csv + markdown

Arms (bench identities, preset spec_plan_smooth, repeat label `axp`): `base` = unmodified; `ax0` = `--opts '{"parity": {"zero_acc": true}}'`
(lib/parity_hugsim.py: ax = ay = 0 in the ego features sent to the bias server; HUGSIM reports ay = 0 already).
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]
RES = REPO / "experiments/op_parity/results"
EVENTS = RES / "four_dirs/hugsim_events.csv"
SHARP = ["scene-0041-medium-00", "scene-570_770-easy-00", "scene-570_770-medium-00", "scene-5980_6180-easy-00", "scene-8440_8640-easy-00"]
ARMS = ["P2H10-F-s0", "P2H10-F-s1"]
PRESET, REPEAT = "spec_plan_smooth", "axp"
OPTS = {"base": None, "ax0": {"parity": {"zero_acc": True}}}
DEV_ONSET = 1.5


def select():
    """Rule (fixed before any run, from the stored P2H `spec_plan_smooth` runs in hugsim_events.csv, 3 repeats x 2 seeds = 6 runs per scenario):
    sharp = the 5 D1 scenarios of four_dirs/hugsim.md; s_ref(scenario) = median over the 6 runs of the route position of the departure onset.
    control = turning == False (route yaw range < 30 deg), all 6 runs end `complete`, one scenario per scene, ranked by the median over the
    6 runs of the peak speed, top 5; s_ref(control) = median of the five sharp s_ref."""
    import gzip
    E = [r for r in csv.DictReader(open(EVENTS)) if r["preset"] == PRESET]
    vmax = defaultdict(float)
    for r in csv.DictReader(gzip.open(RES / "four_dirs/hugsim_steps.csv.gz", "rt")):
        if r["preset"] == PRESET:
            k = (r["arm"], r["rep"], r["scenario"])
            vmax[k] = max(vmax[k], float(r["v"]))
    by = defaultdict(list)
    for r in E:
        by[r["scenario"]].append(r)
    sref = {sc: st.median(float(r["s_onset"]) for r in by[sc]) for sc in SHARP}
    cand = []
    for sc, rs in by.items():
        if rs[0]["turning"] == "False" and len(rs) == 6 and all(r["end"] == "complete" for r in rs):
            cand.append((st.median(vmax[(r["arm"], r["rep"], sc)] for r in rs), sc, rs[0]["scene"], st.median(float(r["s_onset"]) for r in rs)))
    cand.sort(key=lambda c: (-c[0], c[1]))
    ctrl, seen = [], set()
    for v, sc, scene, so in cand:
        if scene not in seen:
            seen.add(scene)
            ctrl.append((sc, v))
    ctrl = ctrl[:5]
    s_ctrl = st.median(sref.values())
    rows = [dict(scenario=sc, set="sharp", s_ref=round(sref[sc], 1), stored_vmax_median="") for sc in SHARP]
    rows += [dict(scenario=sc, set="control", s_ref=round(s_ctrl, 1), stored_vmax_median=round(v, 2)) for sc, v in ctrl]
    return rows, [dict(scenario=sc, stored_vmax_median=round(v, 2)) for v, sc, _, _ in cand]


def cmd_sets(a):
    rows, cand = select()
    with open(RES / "hugsim_ax_probe_sets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("control candidates (ranked, before the one-per-scene cut):", [c["scenario"] for c in cand])


def read_sets():
    return list(csv.DictReader(open(RES / "hugsim_ax_probe_sets.csv")))


def cmd_launch(a):
    sc = ",".join(r["scenario"] for r in read_sets())
    b = ".venv/bin/python -m jevdrive.bench run --model " + " ".join(ARMS) + f" --bench hugsim --preset {PRESET} --repeat {REPEAT} --scenarios {sc}"
    print(b)
    print(b + " --opts '" + json.dumps(OPTS["ax0"]) + "'")


# ------------------------------------------------------------------------------------------------------------------ offline check
def cmd_offline(a):
    """Stored HUGSIM ego features (approach steps of the 5 sharp scenarios, stored P2H10 spec_plan_smooth r0) -> the torch port on navtest
    front tokens of matching speed, ax as fed vs ax = 0; readout: planned speed v13 = (x(3 s) - x(1 s)) / 2 from the plan positions (model clock)."""
    import numpy as np
    import torch
    import pp_train as T
    from jevdrive.bench import tables as BT
    from jevdrive.bench.models import resolve
    from jevdrive.common import data_dir
    dev = torch.device("cuda")
    tidx = 10.0 * (np.arange(33) / 32) ** 2
    ev = {(r["arm"], r["scenario"]): r for r in csv.DictReader(open(EVENTS)) if r["preset"] == PRESET and r["rep"] == "r0"}
    rng = np.random.default_rng(0)
    rows, out = [], []
    for arm in ARMS:
        u, _ = BT.load("hugsim", arm, PRESET)
        egos = []
        for sc in SHARP:
            zs = Path(u.loc[sc, "run_dir"]) / "zs_steps.jsonl"
            recs = [json.loads(x) for x in open(zs)][1:]
            on = int(float(ev[(arm, sc)]["onset"]))
            for r in recs:
                if r["step"] <= on and "parity" in r:
                    egos.append((sc, r["step"], np.asarray(r["parity"]["ego"], np.float32)))
        m = resolve(arm, check=True)
        S = T.Store(["lb_navtest"], dev, need_side=False, frames=m.frames)
        model = T.load_pmodel(m.name, dev)
        sl = model.net.slices
        pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        tab = S.tab
        v_nav = np.hypot(tab["vel"][:, 3, 0], tab["vel"][:, 3, 1])
        E = np.stack([e for _, _, e in egos])
        v_ego = 10 * E[:, 4]
        pick = np.array([rng.choice(np.argsort(np.abs(v_nav - v))[:20]) for v in v_ego])
        E0 = E.copy()
        E0[:, 6:8] = 0

        def plan(ego):
            res = []
            with torch.no_grad():
                for i in range(0, len(E), 128):
                    r = torch.as_tensor(pick[i:i + 128], device=dev)
                    o = model(S.front[r], torch.as_tensor(ego[i:i + 128], device=dev), S.tc[r], None, None).float().cpu().numpy()
                    res.append(o[:, pi].reshape(-1, 33, 15)[:, :, 0])
            x = np.concatenate(res)
            return np.array([(np.interp(3, tidx, p) - np.interp(1, tidx, p)) / 2 for p in x])

        v1, v0 = plan(E), plan(E0)
        for (sc, k, e), a1, b1, bx in zip(egos, v1, v0, 3 * E[:, 6]):
            out.append(dict(arm=arm, scenario=sc, step=k, v_ego=round(10 * float(e[4]), 2), ax_fed=round(float(3 * e[6]), 2), v13=round(float(a1), 3), v13_ax0=round(float(b1), 3)))
        print(arm, len(E), "steps", flush=True)
    with open(RES / "hugsim_ax_probe_offline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, out[0].keys())
        w.writeheader()
        w.writerows(out)
    d = np.array([o["v13_ax0"] - o["v13"] for o in out])
    ax = np.array([o["ax_fed"] for o in out])
    print("n", len(out), "mean ax fed %.2f (p10 %.2f p90 %.2f)" % (ax.mean(), *np.percentile(ax, [10, 90])), "mean d_v13 %.3f" % d.mean())
    for lo, hi, name in [(-9, 0.3, "ax <= 0.3"), (0.3, 1.5, "0.3 < ax <= 1.5"), (1.5, 9, "ax > 1.5")]:
        k = (ax > lo) & (ax <= hi)
        if k.any():
            print(f"{name}: n {k.sum()}, mean d_v13 {d[k].mean():+.3f} m/s, plan v13 orig {np.mean([o['v13'] for o, kk in zip(out, k) if kk]):.2f}")


# ------------------------------------------------------------------------------------------------------------------ report
def run_rows(arm, variant, sets, routes):
    import numpy as np
    from fd_hugsim import Route, load_run
    from jevdrive.bench import hugsim as H
    from jevdrive.bench import tables as BT
    from jevdrive.bench.models import resolve
    key = H.run_key(resolve(arm), PRESET, opts=OPTS[variant], repeat=REPEAT)
    u, src = BT.load("hugsim", "run:" + key)
    if u is None:
        raise SystemExit(f"missing run {key}")
    rows = []
    for s in sets:
        sc = s["scenario"]
        if sc not in u.index:
            print("missing", key, sc)
            continue
        r = u.loc[sc]
        d = Path(r["run_dir"])
        R = load_run(d)
        V, n = R["V"], R["n"]
        route = Route(routes[r["scene"]]["xz"])
        pos, lat = route.project(np.stack([R["X"], R["Y"]], 1))
        end, sref = r["end"], float(s["s_ref"])
        far = np.abs(lat) >= DEV_ONSET
        on = n
        if end in ("bg_collision", "off_route") and far[-1]:
            ok = np.where(~far)[0]
            on = int(ok[-1]) if len(ok) else 0
        hit = np.where(pos >= sref)[0]
        reached = len(hit) > 0
        k = int(hit[0]) if reached else n - 1
        win = (pos >= 10) & (pos <= sref)
        acc = np.diff(V[:n + 1]) / 0.25
        eg = []
        zs = d / "zs_steps.jsonl"
        if zs.exists():
            eg = [json.loads(x)["parity"]["ego"] for x in list(open(zs))[1:] if "parity" in json.loads(x)]
        eg = np.array(eg) if len(eg) else np.zeros((0, 20))
        rows.append(dict(arm=arm, variant=variant, scenario=sc, set=s["set"], s_ref=sref, hd=float(r["hdscore"]), end=end, cls=r["cls"], steps=n,
                         v_ref=float(V[k]), reached=reached, v_onset=float(V[min(on, n)]) if on < n else float("nan"), vmax=float(V.max()),
                         v_win_mean=float(V[:n + 1][win[:n + 1]].mean()) if win.any() else float("nan"),
                         a_sim_win=float(acc[win[:n]].mean()) if win[:n].any() else float("nan"),
                         ax_fed_max_abs=float(np.abs(3 * eg[:, 6]).max()) if len(eg) else float("nan"),
                         ay_fed_max_abs=float(np.abs(3 * eg[:, 7]).max()) if len(eg) else float("nan"),
                         launch_stall=bool(r["launch_stall"]), stuck=bool(r["stuck"]), v_max40=float(r["v_max40"]), run_dir=str(d)))
    return rows


def cmd_report(a):
    import numpy as np
    from jevdrive.common import data_dir
    sets = read_sets()
    routes = json.load(open(data_dir() / "runs/op_parity/hugsim/routes.json"))
    R = []
    for arm in ARMS:
        for v in OPTS:
            R += run_rows(arm, v, sets, routes)
    def w(path, rows):
        with open(path, "w", newline="") as f:
            ww = csv.DictWriter(f, rows[0].keys())
            ww.writeheader()
            ww.writerows(rows)
    w(RES / "hugsim_ax_probe_runs.csv", R)
    idx = {(r["arm"], r["variant"], r["scenario"]): r for r in R}
    P = []
    num = ("hd", "v_ref", "v_onset", "vmax", "v_win_mean", "a_sim_win")
    for s in sets:
        sc = s["scenario"]
        rec = dict(scenario=sc, set=s["set"], s_ref=s["s_ref"])
        for var in OPTS:
            rs = [idx[(arm, var, sc)] for arm in ARMS if (arm, var, sc) in idx]
            for k in num:
                rec[f"{k}_{var}"] = float(np.nanmean([r[k] for r in rs])) if rs else float("nan")
            rec[f"end_{var}"] = "/".join(r["end"] for r in rs)
            rec[f"stall_{var}"] = sum(r["launch_stall"] for r in rs)
            rec[f"stuck_{var}"] = sum(r["stuck"] for r in rs)
            rec[f"reached_{var}"] = sum(r["reached"] for r in rs)
        for k in num:
            rec[f"d_{k}"] = rec[f"{k}_ax0"] - rec[f"{k}_base"]
        P.append(rec)
    w(RES / "hugsim_ax_probe_paired.csv", P)
    # markdown
    f1 = lambda x: "-" if x != x else f"{x:.2f}"  # noqa: E731
    L = ["| scenario | set | s_ref | v_ref base | v_ref ax0 | d v_ref | v_onset base / ax0 | vmax base / ax0 | HD base / ax0 | end base | end ax0 | stall b/a | stuck b/a |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in P:
        L.append(f"| {p['scenario']} | {p['set']} | {p['s_ref']} | {f1(p['v_ref_base'])} | {f1(p['v_ref_ax0'])} | {p['d_v_ref']:+.2f} | {f1(p['v_onset_base'])} / {f1(p['v_onset_ax0'])} | "
                 f"{f1(p['vmax_base'])} / {f1(p['vmax_ax0'])} | {p['hd_base']:.3f} / {p['hd_ax0']:.3f} | {p['end_base']} | {p['end_ax0']} | "
                 f"{p['stall_base']}/{p['stall_ax0']} | {p['stuck_base']}/{p['stuck_ax0']} |")
    print("\n".join(L))
    out = {}
    for S in ("sharp", "control"):
        q = [p for p in P if p["set"] == S]
        out[S] = {k: dict(median=float(np.nanmedian([p[f"d_{k}"] for p in q])), mean=float(np.nanmean([p[f"d_{k}"] for p in q])),
                          n_neg=int(sum(p[f"d_{k}"] < -1.0 for p in q)), n=len(q)) for k in ("v_ref", "vmax", "v_win_mean", "hd")}
    print(json.dumps(out, indent=1))
    (RES / "hugsim_ax_probe_summary.json").write_text(json.dumps(out, indent=1))
    (RES / "hugsim_ax_probe_table.md").write_text("\n".join(L) + "\n")
    # integrity
    ax0 = [r for r in R if r["variant"] == "ax0"]
    base = [r for r in R if r["variant"] == "base"]
    print("ax0 runs max |ax fed| %.4f, max |ay fed| %.4f; base runs max |ax fed| median %.2f" % (
        max(r["ax_fed_max_abs"] for r in ax0), max(r["ay_fed_max_abs"] for r in ax0), float(np.median([r["ax_fed_max_abs"] for r in base]))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    for n, f in (("sets", cmd_sets), ("launch", cmd_launch), ("offline", cmd_offline), ("report", cmd_report)):
        sp.add_parser(n).set_defaults(f=f)
    a = p.parse_args()
    a.f(a)
