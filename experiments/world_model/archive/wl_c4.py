"""WL C4: the selection rule as an examinee on the P5 v1 BA open-loop exam (fc65452:todos/2026-09-28-wm-loop.md, "C4 的读法").

Per exam frame of the 20 WL eval base routes: 7 candidates from the frame's Cinque plan and the run's own driven path,
the `main`-arm predictor of each seed rolls them out (source = intervention), that seed's C-learn critic scores them, the
registered rule picks one, and the examinee's output is the picked candidate's commanded speed at 2 s. obs Delta =
x+ frame - x- frame, null Delta = x+ frame - weather-null frame; p5_exam.exam unchanged, pedestrian and cut-in
families scored separately, next to the M-C pair head and the openpilot `ridge_late` prior of decision 42 on the same frames.

  python -m experiments.world_model.archive.wl_c4            (needs runs/wl/model/main/seed{0,1,2}/*/model.pt and the plan streams of every exam run)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.world_model.archive import nq4_w as W
from jevdrive import p5_exam as E
from experiments.world_model.lib import wl as WL
from experiments.world_model.archive import wl_model as M
from jevdrive.common import data_dir, get_logger
from experiments.world_model.lib.wl_traj import ACTIONS, command_actions, world_to_ego

log = get_logger(__name__)
SET = "carla_p5v1_ba"
FLIP_PED_MARGIN, PRIOR_CUTIN_MARGIN = 0.10, 0.05


def exam_tables():
    """obs / null of the WL eval routes (no Light family), and the pairs table restricted the same way."""
    d = data_dir() / "processed" / SET
    ev = set(json.loads(WL.rundir("split.json").read_text())["eval"])
    obs, null = pd.read_parquet(d / "obs.parquet"), pd.read_parquet(d / "null.parquet")
    pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str})
    for t in (obs, null):
        t["base_id"] = t.base_id.astype(str)
    obs = obs[obs.base_id.isin(ev) & (obs.family != "Light")].reset_index(drop=True)
    null = null[null.base_id.isin(ev) & (null.family != "Light")].reset_index(drop=True)
    return obs, null, pairs[pairs.base_id.isin(ev) & (pairs.family != "Light")]


def _rear() -> float:
    return float(json.loads((Path(__file__).resolve().parents[3] / "experiments/b2d_tfv6/results/tfv6-controller/controller-eval/P7.json")
                            .read_text())["rear_axle_offset_m"])


def frame_inputs(frames: list[str]) -> pd.DataFrame:
    """Per frame: v0, the op plan, the route ahead in the frame's ego coordinates (from the run's own poses)."""
    rear, plans = _rear(), WL._op_plans("ba")
    rows, by_run = [], {}
    for fn in frames:
        by_run.setdefault(fn.split("-")[0], []).append(fn)
    for rid, fns in by_run.items():
        p = pd.read_json(WL.attempt_dir("ba", rid) / "pose.jsonl", lines=True).set_index("frame")
        yaw = np.radians(p.yaw.to_numpy())
        xy = np.column_stack((p.x + rear * np.cos(yaw), p.y + rear * np.sin(yaw)))
        pos = {f: i for i, f in enumerate(p.index)}
        for fn in fns:
            i = pos[int(fn.split("-")[1])]
            ahead = xy[i:]
            arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ahead, axis=0).T))]
            keep = np.r_[True, np.diff(np.floor(arc / 0.5)) > 0] & (arc <= 150.0)
            r = world_to_ego(ahead[keep], xy[i], float(p.yaw.iloc[i]))
            rows.append({"fn": fn, "v0": float(np.hypot(p.vx.iloc[i], p.vy.iloc[i])), "op_plan": plans.get(fn), "route_ego": r})
    return pd.DataFrame(rows).set_index("fn")


def candidates_for(fi: pd.DataFrame) -> np.ndarray:
    """(n, 7, 10, 2) commanded (a, omega) of the 7 candidates per frame (same code as the fork data)."""
    out = np.zeros((len(fi), len(ACTIONS), W.FUT, 2), np.float32)
    for k, (fn, r) in enumerate(fi.iterrows()):
        c = command_actions(None if r.op_plan is None else np.asarray(r.op_plan, float), r.route_ego, r.v0)
        out[k] = np.stack([c[a] for a in ACTIONS])
    return out


def _apply(P, k, X):
    kind, w, b = P[k][0], P[k][1].cuda(), P[k][2].cuda()
    return X @ w + b


def prepare_rows(fn_list, rows_of, wm, wz):
    """The W-meta rows the exam windows need (8 history rows per frame), in W order, plus zero padding for the 18-row
    window gather of experiments.world_model.archive.wl_model.Data; returns (meta, z, anchor position of every frame)."""
    need = np.unique(np.concatenate([rows_of[f][1] for f in fn_list]))
    sub = wm.iloc[need].reset_index(drop=True)
    pos = pd.Series(np.arange(len(sub)), index=need)
    z = np.asarray(wz[need])
    pad = W.WIN
    meta = pd.concat([sub, sub.iloc[:pad].assign(a_prev=0.0, w_prev=0.0, v=0.0, a_next=0.0, w_next=0.0)], ignore_index=True)
    meta["src"] = 0
    z = np.concatenate([z, np.zeros((pad, z.shape[1]), z.dtype)])
    return meta, z, np.array([pos[rows_of[f][0]] for f in fn_list])


def rollouts(seed: int, prep, cmds: np.ndarray):
    """Predictions for every frame x candidate with one seed's main-arm model: z0, z5, z10 (float32 numpy) and the
    probe-speed progress."""
    import torch
    meta, z, anchors = prep
    d = M._latest("main", seed)
    ck = torch.load(d / "model.pt", map_location="cuda")
    assert "probes" in ck, f"{d}: checkpoint without probes"
    data = M.Data(meta, z, np.arange(min(len(z) - W.WIN, 4096)))
    data.mu, data.sd = ck["mu"].cuda(), ck["sd"].cuda()
    for lo in range(0, len(z), 8192):
        data.Z[lo: lo + 8192] = ((torch.as_tensor(z[lo: lo + 8192], device="cuda").float() - data.mu) / data.sd).bfloat16()
    model = M.build(z.shape[1]).cuda()
    model.load_state_dict(ck["state"])
    starts = torch.as_tensor(anchors - (W.HIST - 1), device="cuda")
    n, A = len(anchors), len(ACTIONS)
    cm = torch.as_tensor(cmds, device="cuda").reshape(n * A, W.FUT, 2)
    pred = M.predict(model, data, starts.repeat_interleave(A), cm, src=1).reshape(n, A, W.FUT, -1)
    z0 = data.Z[starts + W.HIST - 1].float()
    v_hat = _apply(ck["probes"], "v", pred)                                   # (n, A, 10)
    return {"z0": z0.cpu().numpy(), "z5": pred[:, :, 4].cpu().numpy(), "z10": pred[:, :, 9].cpu().numpy(),
            "prog": (v_hat * W.DT).sum(-1).cpu().numpy()}


def critic_probs(seed: int, ro: dict) -> np.ndarray:
    """The seed's C-learn (fitted on that seed's train-split fork points, experiments.world_model.archive.wl_model's recipe) on the exam rollouts."""
    q, fa = M._long("main", seed)
    n, A = q["z0"].shape[0], len(ACTIONS)
    z0 = np.repeat(q["z0"].astype(np.float32)[:, None], A, 1)
    Xl = np.concatenate([z0, q["z5"].astype(np.float32), q["z10"].astype(np.float32)], -1).reshape(n * A, -1)
    tr = M._truth().set_index(["fork_id", "action"])
    lab = pd.DataFrame({"fork_id": np.repeat(fa.fork_id.to_numpy(), A), "action": np.tile(ACTIONS, n)}).join(tr[["unsafe", "split"]], on=["fork_id", "action"])
    trn = (lab.split == "train").to_numpy() & lab.unsafe.notna().to_numpy()
    y = lab.unsafe.fillna(False).astype(float).to_numpy()
    m = ro["z0"].shape[0]
    Xe = np.concatenate([np.repeat(ro["z0"][:, None], A, 1), ro["z5"], ro["z10"]], -1).reshape(m * A, -1).astype(np.float32)
    return M._mlp_fit(Xl[trn], y[trn], seed, Xe).reshape(m, A)


def select(p: np.ndarray, prog: np.ndarray) -> np.ndarray:
    """The registered rule: among candidates with p < 0.2 the largest progress, else the smallest p."""
    ok = p < M.UNSAFE_P
    return np.where(ok.any(1), np.where(ok, prog, -np.inf).argmax(1), p.argmin(1))


def run() -> dict:
    t0 = time.time()
    obs, null, pairs = exam_tables()
    frames = sorted(set(obs.fn_plus) | set(obs.fn_minus) | set(null.fn_plus) | set(null.fn_null))
    wm = pd.read_parquet(W.wdir("meta.parquet"))
    wz = np.load(W.wdir("z.npy"), mmap_mode="r")
    key = pd.Series(np.arange(len(wm)), index=wm.set.astype(str) + "|" + wm.frame_name)
    # anchors with 7 previous frames of the same run at the 5 Hz spacing
    rows_of, ok = {}, []
    for f in frames:
        i = key.get(f"{SET}|{f}")
        if i is None or i < W.HIST - 1:
            continue
        h = np.arange(i - (W.HIST - 1), i + 1)
        if (wm.adir.to_numpy()[h] == wm.adir.to_numpy()[i]).all() and (np.diff(wm.frame.to_numpy()[h]) == W.TICKS).all():
            rows_of[f] = (int(i), h)
            ok.append(f)
    log.info("exam frames %d, with a full 8-step history %d", len(frames), len(ok))
    fi = frame_inputs(ok)
    cmds = candidates_for(fi)
    out, risk, picks = {}, {}, []
    prep = prepare_rows(ok, rows_of, wm, wz)
    for s in M.SEEDS:
        ro = rollouts(s, prep, cmds)
        p = critic_probs(s, ro)
        sel = select(p, ro["prog"])
        v2 = fi.v0.to_numpy() + cmds[np.arange(len(ok)), sel, :, 0].sum(1) * W.DT
        out[s] = pd.Series(v2, index=ok)
        risk[s] = pd.Series(-p[:, ACTIONS.index("hold")], index=ok)      # post hoc description: the critic's own risk of `hold`, sign-flipped
        pick_s = pd.Series(np.array(ACTIONS)[sel], index=ok)
        if s == M.SEEDS[0]:
            pick_by_seed = {}
        pick_by_seed[s] = pick_s
        picks.append(pd.DataFrame({"fn": ok, "seed": s, "pick": np.array(ACTIONS)[sel], "v2_out": v2, "p_pick": p[np.arange(len(ok)), sel]}))
        log.info("seed %d done, %.0f s", s, time.time() - t0)
    picks = pd.concat(picks)
    have = set(ok)
    ex = {}
    for name, df, a, b in (("obs", obs, "fn_plus", "fn_minus"), ("null", null, "fn_plus", "fn_null")):
        df = df[df[a].isin(have) & df[b].isin(have)].copy()
        for s in M.SEEDS:
            df[f"WL selector s{s}"] = out[s][df[a]].to_numpy() - out[s][df[b]].to_numpy()
            df[f"WL hold-risk s{s}"] = risk[s][df[a]].to_numpy() - risk[s][df[b]].to_numpy()
            df[f"pick_changed s{s}"] = (pick_by_seed[s][df[a]].to_numpy() != pick_by_seed[s][df[b]].to_numpy())
        ex[name] = df
    obs, null = ex["obs"], ex["null"]
    # references on the same frames: decision 42's prior and the M-C pair head, one run per fold seed
    ref_cols = []
    rdirs = [sorted(x for x in (data_dir() / "runs" / "reactivity" / f"mc-carla_p5v1_ba-seed{j}").glob("*") if (x / "obs_scored.parquet").exists())[-1]
             for j in range(3)]
    for j, rdir in enumerate(rdirs):
        o = pd.read_parquet(rdir / "obs_scored.parquet")
        nn = pd.read_parquet(rdir / "null_scored.parquet")
        for col, nm in (("prior [cinque]", f"prior ridge_late f{j}"), ("M-C pair [cinque]", f"M-C pair f{j}")):
            obs[nm] = obs.merge(o[["fn_plus", "fn_minus", col]], on=["fn_plus", "fn_minus"], how="left")[col].to_numpy()
            null[nm] = null.merge(nn[["fn_plus", "fn_null", col]], on=["fn_plus", "fn_null"], how="left")[col].to_numpy()
            ref_cols.append(nm)
    examinees = [f"WL selector s{s}" for s in M.SEEDS] + [f"WL hold-risk s{s}" for s in M.SEEDS] + ref_cols
    res = {}
    rows = []
    for cls, fams in (("ped", WL.PED_FAM), ("cutin", WL.CUTIN_FAM)):
        o, n_, pr = obs[obs.family.isin(fams)], null[null.family.isin(fams)], pairs[pairs.family.isin(fams)]
        r = E.exam(o, n_, pr, examinees)
        fl = r["flips"]
        fl = fl[fl.scope == "pooled"].assign(cls=cls, n_obs=len(o), n_null=len(n_))
        rows.append(fl)
        res[cls] = r
    fl = pd.concat(rows)
    out_dir = WL.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    fl.to_csv(out_dir / "c4_flips.csv", index=False, float_format="%.4f")
    picks.to_csv(out_dir / "c4_picks.csv", index=False, float_format="%.4f")
    # verdicts
    v = {"n_exam_frames": len(ok), "n_frames_dropped_no_history": len(frames) - len(ok),
         "n_obs": {c: int(obs.family.isin(f).sum()) for c, f in (("ped", WL.PED_FAM), ("cutin", WL.CUTIN_FAM))}}
    g = fl.set_index(["cls", "examinee"])
    v["ped"] = {}
    for s in M.SEEDS:
        r = g.loc[("ped", f"WL selector s{s}")]
        v["ped"][f"s{s}"] = {"flip": float(r.flip_rate), "ci": [float(r.flip_lo), float(r.flip_hi)], "null_oos": float(r.false_flip_null_oos),
                             "pass": bool(r.flip_lo > r.false_flip_null_oos + FLIP_PED_MARGIN)}
    v["ped"]["pass"] = all(v["ped"][f"s{s}"]["pass"] for s in M.SEEDS)
    prior = np.mean([g.loc[("cutin", c)].flip_rate for c in ref_cols if c.startswith("prior")])
    v["cutin"] = {"prior_flip_mean": float(prior), "seeds": {}}
    for s in M.SEEDS:
        r = g.loc[("cutin", f"WL selector s{s}")]
        v["cutin"]["seeds"][f"s{s}"] = {"flip": float(r.flip_rate), "ci": [float(r.flip_lo), float(r.flip_hi)],
                                        "pass": bool(r.flip_rate >= prior - PRIOR_CUTIN_MARGIN)}
    v["cutin"]["pass"] = all(x["pass"] for x in v["cutin"]["seeds"].values())
    v["refs"] = {f"{c}": {k: float(g.loc[(cl, c)].flip_rate) for k, cl in (("ped", "ped"), ("cutin", "cutin"))} for c in ref_cols}
    v["pick_changed"] = {nm: {f"s{s}": float(df[f"pick_changed s{s}"].mean()) for s in M.SEEDS} for nm, df in (("obs", obs), ("null", null))}
    v["hold_risk_description"] = {c: {f"s{s}": {"flip": float(g.loc[(c, f"WL hold-risk s{s}")].flip_rate),
                                                "null_oos": float(g.loc[(c, f"WL hold-risk s{s}")].false_flip_null_oos)} for s in M.SEEDS}
                                  for c in ("ped", "cutin")}
    (out_dir / "c4.json").write_text(json.dumps(v, indent=1))
    log.info("C4 done in %.0f s", time.time() - t0)
    return v


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
