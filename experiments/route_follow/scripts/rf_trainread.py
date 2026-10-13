#!/usr/bin/env python3
"""TRAINREAD (user question 2026-10-13): do the policies drive well in closed loop on the scenes they were trained on? Box, AlpaSim's venv.

  rf_trainread.py pick --out DIR [--max-turn 80] [--n-ctl 40] [--seed 0] [--workers N]
      Policy-free pre-screen of every `alpasim_nuplan/onpol-train` scene with rf_follow's own turn rule (route heading change >= 60 deg within
      30 m, found with rf_follow.find_turn inside the log's travel [s0, s_log] to decision 17; the route is the generator's polyline after
      decisions anchored on the LOGGED ego poses instead of a policy's). -> DIR/prescreen.csv (all scenes), DIR/scenes.txt (<= max-turn
      turn scenes, fixed-seed sample when more, then n-ctl non-turn controls), DIR/scenes.csv (id, class). The closed-loop rollout can move the
      anchors a little, so the final turn / non-turn class of a rollout is rf_follow's own (scenes.csv of the follow output).
  rf_trainread.py screen --list FILE --out CSV   the same pre-screen for a scene list (the held 460: policy-free turn class for both sets)
  rf_trainread.py report --follow DIR --train-dirs MODEL=dirs.. --held-dirs MODEL=dirs.. --class-csv train.csv held.csv --out DIR
      Per model, train vs held, by policy-free class (turn / non-turn): mean, zeros by kind; rf_follow turn completion (outcomes incl. "short" =
      too slow), rf_inside inside-cut share. Paired CIs between models on the same scenes (log-clustered, 10 000 draws); train - held is
      unpaired (logs resampled independently in each set). -> DIR/summary.md, summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/route_follow/scripts"), str(R_ / "experiments/route_mem/lib"),
                str(R_ / "experiments/route_mem/scripts"), str(R_ / "experiments/alpasim/scripts"), str(R_ / "experiments/alpasim/lib"),
                str(R_ / "experiments/anchor_offset/scripts"), str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts")]
import numpy as np  # noqa: E402

import rf_follow as F  # noqa: E402


def screen(scene):
    """-> dict(scene, ok, turn, side, deg) from the log's own ego poses."""
    import ev as EV
    from ao_e1 import loader, project, stations
    try:
        tr = loader(str(EV.CFG)).get_data_source(scene).rig.trajectory
        P, Tt, Y = np.asarray(tr.positions, float), np.asarray(tr.timestamps_us, float), np.unwrap(np.asarray(tr.yaws, float))
        recs = [dict(k=k, t0=F.T0 + F.DT * k, anchor=(np.interp(F.T0 + F.DT * k, Tt, P[:, 0]), np.interp(F.T0 + F.DT * k, Tt, P[:, 1]),
                                                       np.interp(F.T0 + F.DT * k, Tt, Y))) for k in range(F.K_END + 1)]
        _, A, _, Rl, log_end = F.regen_full(scene, recs, EV.CFG)
        Rl = Rl[np.r_[True, np.hypot(*np.diff(Rl, axis=0).T) > 1e-6]]
        S = stations(Rl)
        s0 = project(Rl, S, np.asarray(A[0][:2]))[0]
        s_end = project(Rl, S, log_end, s0 - 10.0, s0 + 150.0)[0]
        s, h = F.heading(Rl, S)
        t = F.find_turn(s, h, s0, s_end)
        if t is not None and t[1] + F.PAST <= s_end:
            return dict(scene=scene, ok=True, turn=True, side="L" if t[2] > 0 else "R", deg=float(np.degrees(t[2])))
        return dict(scene=scene, ok=True, turn=False, side="", deg=0.0)
    except Exception as e:  # noqa: BLE001
        return dict(scene=scene, ok=False, turn=False, side="", deg=0.0, err=str(e)[-200:])


def pick(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    sp = splits.load("alpasim_nuplan/onpol-train")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "trainread-pick", config=vars(a)) as run:
        run.use_split(sp)
        res = par.pmap(screen, list(sp.members), run=run, workers=min(a.workers or n_cpus(), 64), desc="screen")
        D = pd.DataFrame([v for v in res.values if v is not None]).sort_values("scene")
        D.to_csv(out / "prescreen.csv", index=False)
        rng = np.random.default_rng(a.seed)
        T, C = D[D.ok & D.turn].scene.to_numpy(), D[D.ok & ~D.turn].scene.to_numpy()
        T = np.sort(rng.choice(T, a.max_turn, replace=False)) if len(T) > a.max_turn else T
        C = np.sort(rng.choice(C, a.n_ctl, replace=False))
        pd.DataFrame(dict(scene=list(T) + list(C), cls=["turn"] * len(T) + ["ctl"] * len(C))).to_csv(out / "scenes.csv", index=False)
        (out / "scenes.txt").write_text("\n".join(list(T) + list(C)) + "\n")
        run.info(f"train scenes {len(sp.members)}: screened ok {int(D.ok.sum())}, turn {int((D.ok & D.turn).sum())}, non-turn {len(C)} candidates -> picked {len(T)} turn + {len(C)} control")
        print(f"picked {len(T)} turn (of {int((D.ok & D.turn).sum())}) + {len(C)} control; errors {int((~D.ok).sum())}")


def screen_list(a):
    import pandas as pd
    from jevdrive import par
    ids = Path(a.list).read_text().split()
    res = par.pmap(screen, ids, workers=a.workers, desc="screen")
    pd.DataFrame([v for v in res.values if v is not None]).sort_values("scene").to_csv(a.out, index=False)


N_BOOT = 10_000
KINDS = ("collision_at_fault", "offroad", "left_corridor_laterally")


def scores(dirs):
    """{scene: (score, zero kind or '')} over the run dirs (a later dir does not override an earlier one)."""
    import json
    out = {}
    for d in dirs.split(","):
        for r in json.loads((Path(d) / "aggregate/results-summary.json").read_text())["rollouts"]:
            if r["clipgt_id"] in out:
                continue
            sc, m = r.get("score") or 0.0, r.get("metrics") or {}
            out[r["clipgt_id"]] = (float(sc), "" if sc > 0 else next((k for k in KINDS if (m.get(k) or 0) > 0), "other"))
    return out


def _cl(x, logs):
    """per-log sums and counts of x (aligned with logs) -> (sum, count) arrays over sorted unique logs."""
    L, inv = np.unique(logs, return_inverse=True)
    return np.bincount(inv, np.asarray(x, float), len(L)), np.bincount(inv, minlength=len(L)).astype(float)


def boot(x, logs, rng):
    """mean of x with a log-clustered 95 % CI (ratio of sums)."""
    s, c = _cl(x, logs)
    i = rng.integers(len(s), size=(N_BOOT, len(s)))
    b = s[i].sum(1) / c[i].sum(1)
    return dict(n=int(len(x)), mean=float(np.mean(x)) if len(x) else None, lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)))


def boot_diff(xa, la, xb, lb, rng, paired):
    """mean(a) - mean(b). paired: xa, xb aligned on the same scenes (one log draw); else independent samples (logs resampled separately)."""
    if paired:
        return boot(np.asarray(xa, float) - np.asarray(xb, float), la, rng)
    sa, ca = _cl(xa, la)
    sb, cb = _cl(xb, lb)
    ia, ib = rng.integers(len(sa), size=(N_BOOT, len(sa))), rng.integers(len(sb), size=(N_BOOT, len(sb)))
    b = sa[ia].sum(1) / ca[ia].sum(1) - sb[ib].sum(1) / cb[ib].sum(1)
    return dict(n=int(len(xa)), n_b=int(len(xb)), mean=float(np.mean(xa) - np.mean(xb)), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975)))


def report(a):
    import json
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cls = {"train": pd.read_csv(a.class_csv[0]).set_index("scene"), "held": pd.read_csv(a.class_csv[1]).set_index("scene")}
    ctl = set(pd.read_csv(a.scenes).query("cls == 'ctl'").scene) if a.scenes else None
    runs = {s: {m: scores(d) for m, d in (x.split("=", 1) for x in dirs)} for s, dirs in (("train", a.train_dirs), ("held", a.held_dirs))}
    models = list(runs["train"])
    Fs = pd.read_csv(Path(a.follow) / "scenes.csv")
    Dx = pd.read_parquet(Path(a.follow) / "decisions.parquet")
    Fs["log"] = Fs.scene.str.rsplit("-", n=1).str[0]
    Fs["completed"] = Fs.completed.fillna(False).astype(bool)
    J = Dx[Dx.junction].merge(Fs[Fs.turn.astype(bool)][["label", "scene", "side"]], on=["label", "scene"])
    J["inw"] = J.lat * np.where(J.side == "L", 1.0, -1.0)
    cut = J.groupby(["label", "scene"]).inw.max().gt(1.0).rename("cut").reset_index()
    Fs = Fs.merge(cut, on=["label", "scene"], how="left")
    R = {"models": {}, "paired": {}, "train_minus_held": {}}
    # class sets: the scenes every model has a rollout on, per set; train non-turn = the fixed control sample only
    sets = {}
    for s in ("train", "held"):
        common = set.intersection(*(set(runs[s][m]) for m in models))
        c = cls[s]
        ok = c[c.ok.astype(bool) & c.index.isin(common)]
        sets[s] = dict(turn=sorted(ok[ok.turn.astype(bool)].index), nonturn=sorted(i for i in ok[~ok.turn.astype(bool)].index if s == "held" or ctl is None or i in ctl))
    for m in models:
        R["models"][m] = {}
        for s in ("train", "held"):
            r = {}
            for k, ids in sets[s].items():
                v = np.array([runs[s][m][i][0] for i in ids])
                kinds = [runs[s][m][i][1] for i in ids]
                lg = [i.rsplit("-", 1)[0] for i in ids]
                r[k] = dict(n=len(ids), mean=boot(v, lg, rng), zeros=int((v == 0).sum()), zero_rate=float((v == 0).mean()),
                            kinds={q: kinds.count(q) for q in (*KINDS, "other")})
            f = Fs[Fs.label == f"{m}@{s}"]
            f = f[f.turn.astype(bool)]
            lab = f.outcome.tolist()
            r["follow"] = dict(turns=len(f), completion=boot(f.completed.astype(float), f.log, rng) if len(f) else None,
                               outcomes={o: lab.count(o) for o in ("completed", "event", "off", "short")},
                               short_share=boot((f.outcome == "short").astype(float), f.log, rng) if len(f) else None,
                               inside_cut=boot(f.cut.astype(float), f.log, rng) if len(f) else None)
            R["models"][m][s] = r
        R["train_minus_held"][m] = {}
        for k in ("turn", "nonturn"):
            A, B = sets["train"][k], sets["held"][k]
            va, vb = [runs["train"][m][i][0] for i in A], [runs["held"][m][i][0] for i in B]
            za, zb = [float(x == 0) for x in va], [float(x == 0) for x in vb]
            la, lb = [i.rsplit("-", 1)[0] for i in A], [i.rsplit("-", 1)[0] for i in B]
            R["train_minus_held"][m][k] = dict(mean=boot_diff(va, la, vb, lb, rng, False), zero_rate=boot_diff(za, la, zb, lb, rng, False))
        fa, fb = (Fs[(Fs.label == f"{m}@{s}") & Fs.turn.astype(bool)] for s in ("train", "held"))
        R["train_minus_held"][m]["completion"] = boot_diff(fa.completed.astype(float), fa.log, fb.completed.astype(float), fb.log, rng, False)
        R["train_minus_held"][m]["short"] = boot_diff((fa.outcome == "short").astype(float), fa.log, (fb.outcome == "short").astype(float), fb.log, rng, False)
        R["train_minus_held"][m]["inside_cut"] = boot_diff(fa.cut.astype(float), fa.log, fb.cut.astype(float), fb.log, rng, False)
    for s in ("train", "held"):                                       # paired model comparisons on the same scenes
        for A, B in (p.split(":") for p in a.pairs):
            q = {}
            for k, ids in sets[s].items():
                lg = [i.rsplit("-", 1)[0] for i in ids]
                va, vb = np.array([runs[s][A][i][0] for i in ids]), np.array([runs[s][B][i][0] for i in ids])
                q[k] = dict(mean=boot_diff(va, lg, vb, lg, rng, True), zero_rate=boot_diff(va == 0, lg, vb == 0, lg, rng, True))
            fa, fb = (Fs[(Fs.label == f"{M}@{s}") & Fs.turn.astype(bool)].set_index("scene") for M in (A, B))
            both = fa.index.intersection(fb.index)
            lg = fa.loc[both].log.to_numpy()
            q["completion"] = boot_diff(fa.loc[both].completed.astype(float), lg, fb.loc[both].completed.astype(float), lg, rng, True) if len(both) else None
            R["paired"][f"{A} - {B} @{s}"] = q
    (out / "summary.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
    ci = lambda d, p=3: "-" if not d else f"{d['mean']:.{p}f} [{d['lo']:.{p}f}, {d['hi']:.{p}f}]"  # noqa: E731
    pc = lambda d: "-" if not d else f"{100 * d['mean']:.1f} [{100 * d['lo']:.0f}, {100 * d['hi']:.0f}]"  # noqa: E731
    pp = lambda d: "-" if not d else f"{100 * d['mean']:+.1f} [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]"  # noqa: E731
    n = {s: {k: len(v) for k, v in sets[s].items()} for s in sets}
    md = ["# Training-scene read (TRAINREAD), 2026-10-13", "",
          "Question: do the policies drive well in closed loop on the scenes they were trained on? Reads @18 decisions, same serving as the held reads "
          f"(rm_chain readlist). Train = `alpasim_nuplan/onpol-train` scenes: turn class = rf_follow's turn rule on the log's travel (policy-free pre-screen, "
          f"all turn scenes or a fixed-seed 80 sample) + {n['train']['nonturn']} fixed-seed non-turn controls. Held = the 460 held reads. Scene counts "
          f"(common to the models) train {n['train']}, held {n['held']}. Class = the policy-free pre-screen, identical across models and sets; "
          "turn completion / outcomes / inside-cut use rf_follow's own per-rollout turn flag (counts below). CIs: log-clustered bootstrap, 10 000 draws.", "",
          "## Train vs held", "",
          "| model | set | turn n | turn mean | turn zeros (col / off / cor / other) | non-turn mean | non-turn zeros | rf_follow turns | completion % | completed / event / off / short | short % | inside-cut % |",
          "|:--|:--|--:|:--|:--|:--|:--|--:|:--|:--|:--|:--|"]
    for m in models:
        for s in ("train", "held"):
            r = R["models"][m][s]
            t, u, f = r["turn"], r["nonturn"], r["follow"]
            k = t["kinds"]
            md.append(f"| {m} | {s} | {t['n']} | {ci(t['mean'])} | {t['zeros']} ({k['collision_at_fault']} / {k['offroad']} / {k['left_corridor_laterally']} / {k['other']}) | "
                      f"{ci(u['mean'])} | {u['zeros']} / {u['n']} | {f['turns']} | {pc(f['completion'])} | {' / '.join(str(v) for v in f['outcomes'].values())} | "
                      f"{pc(f['short_share'])} | {pc(f['inside_cut'])} |")
    md += ["", "Train - held (unpaired: different scenes), turn class unless stated; completion / short / inside-cut in pp:", "",
           "| model | turn mean | turn zero rate (pp) | non-turn mean | non-turn zero rate (pp) | completion | short | inside-cut |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for m in models:
        d = R["train_minus_held"][m]
        md.append(f"| {m} | {ci(d['turn']['mean'])} | {pp(d['turn']['zero_rate'])} | {ci(d['nonturn']['mean'])} | {pp(d['nonturn']['zero_rate'])} | "
                  f"{pp(d['completion'])} | {pp(d['short'])} | {pp(d['inside_cut'])} |")
    md += ["", "Paired model comparisons (same scenes, A - B):", "", "| pair | turn mean | turn zero rate (pp) | non-turn mean | non-turn zero rate (pp) | completion (pp) |", "|:--|:--|:--|:--|:--|:--|"]
    for p, q in R["paired"].items():
        md.append(f"| {p} | {ci(q['turn']['mean'])} | {pp(q['turn']['zero_rate'])} | {ci(q['nonturn']['mean'])} | {pp(q['nonturn']['zero_rate'])} | {pp(q['completion'])} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--out", required=True)
    p.add_argument("--max-turn", type=int, default=80)
    p.add_argument("--n-ctl", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("screen")
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("report")
    p.add_argument("--follow", required=True)
    p.add_argument("--train-dirs", nargs="+", required=True)
    p.add_argument("--held-dirs", nargs="+", required=True)
    p.add_argument("--class-csv", nargs=2, required=True)
    p.add_argument("--scenes", default="")
    p.add_argument("--pairs", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pick": pick, "screen": screen_list, "report": report}[a.cmd](a)
