#!/usr/bin/env python3
"""TRAINREAD (user question 2026-10-13): do the policies drive well in closed loop on the scenes they were trained on? Box, AlpaSim's venv.

  rf_trainread.py pick --out DIR [--max-turn 80] [--n-ctl 40] [--seed 0] [--workers N]
      Policy-free pre-screen of every `alpasim_nuplan/onpol-train` scene with rf_follow's own turn rule (route heading change >= 60 deg within
      30 m, found with rf_follow.find_turn inside the log's travel [s0, s_log] to decision 17; the route is the generator's polyline after
      decisions anchored on the LOGGED ego poses instead of a policy's). -> DIR/prescreen.csv (all scenes), DIR/scenes.txt (<= max-turn
      turn scenes, fixed-seed sample when more, then n-ctl non-turn controls), DIR/scenes.csv (id, class). The closed-loop rollout can move the
      anchors a little, so the final turn / non-turn class of a rollout is rf_follow's own (scenes.csv of the follow output).
  rf_trainread.py report ...  -> see the `report` function.
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--out", required=True)
    p.add_argument("--max-turn", type=int, default=80)
    p.add_argument("--n-ctl", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"pick": pick}[a.cmd](a)
