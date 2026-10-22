"""NAVSIM open-loop readouts: navtest EPDMS (v2 devkit, the harness that reproduced WA-JEPA's 91.71) and navhard two-stage.

Pipeline per model (every step is the code the lanes used, wrapped):
  prep     parity models only, when the pp_prep token cache of the frame protocol is missing: experiments/op_parity/scripts/pp_prep.py
  plans    the model's 33-step plan at t0 of every token, op_lb plan format (names, plan_pos / vel / yaw, plan_mu / std):
           parity: the torch port on the cached tokens (pp_train.PModel, fp16, batch 128: pp_eval.py plans, one model);
           memory arms (pp_train --mem) also read their front-token bank (runs/op_parity/mem/<kind>/<data>.npy; :noside masks it,
           :mshuf reads the bank row of a token of another log);
           UF-* arms: pp_unfreeze.py plans (pixels); onnx: scripts/op_lb.py run (TensorRT, envs/openpilot; an existing op_lb
           plan file of the same stem is reused); vis_train arms (family vt): experiments/vis_train/scripts/vt.py plans
  poses    parity checkpoints with a trajectory head (Model.thead; lib/traj_head.py): the head's 8 rear-axle poses on the cached tokens,
           written directly as the prediction file; replaces plans + export
  export   experiments/op_openloop/lib/op_interp.py nav-export, adapter `base` (CAM_F0 lever arm to the rear axle, linear
           resampling to 0.5 .. 4 s) -> preds/<stem>__base.npz (tokens, poses (n, 8, 3))
  score    navtest: experiments/zeroshot_openloop/archive/navsim_zs_score.sh score v2 navtest (envs/navsim2, ray, OPENBLAS_CORETYPE
           Haswell) in K shards of whole logs (TOKENS_FILE; the devkit pairs consecutive frames of a log for EC, so shards never
           split a log) -> per-token CSV rows
           navhard: experiments/op_guard/scripts/nav_harness.py (per-token devkit pdm_score + the devkit's two-stage aggregation; the
           mean over the 225 scene-mapping groups is the official number)
  collect  units.csv: navtest one row per token (log, score, 9 sub-scores), navhard one row per group (orig token, log,
           combined / stage1 / stage2); summary.json (EPDMS x 100 and sub-scores)
Plans / preds live under $DATA_DIR/runs/bench/ol/<lb data>/ (op_interp's OPI_ROOT layout, meta.json linked from runs/op_lb).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import runner as R
from .models import REPO, TS_OPTS, Model, agent_cfg, data_dir, resolve
from .sets import NAVSIM, log_shards, navsim_tokens

OL_REL = "bench/ol"                          # OPI_ROOT (relative to $DATA_DIR/runs)
SUBS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance",
        "TLC": "traffic_light_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "LK": "lane_keeping",
        "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
SCORE_SH = REPO / "experiments/zeroshot_openloop/archive/navsim_zs_score.sh"
HARNESS = REPO / "experiments/op_guard/scripts/nav_harness.py"
PP = REPO / "experiments/op_parity/scripts"


def ol_root(data: str) -> Path:
    d = data_dir() / "runs" / OL_REL / data
    (d / "plans").mkdir(parents=True, exist_ok=True)
    m = d / "meta.json"
    if not m.exists():
        m.symlink_to(data_dir() / "runs" / "op_lb" / data / "meta.json")
    return d


def stem(m: Model) -> str:
    return m.spec.replace(":", "_")


def plan_file(m: Model, bench: str) -> Path:
    return data_dir() / "runs" / OL_REL / NAVSIM[bench]["data"] / "plans" / f"{stem(m)}.npz"


def adapter(m: Model) -> str:
    """op_interp nav-export adapter: `base` (lever arm), `lm` for the `:lm` option (base + jevdrive/openpilot/lead_margin.py)."""
    return "lm" if m.opt == "lm" else "base"


def pred_file(m: Model, bench: str) -> Path:
    return data_dir() / "runs" / OL_REL / NAVSIM[bench]["data"] / "preds" / f"{stem(m).replace('@', '-')}__{adapter(m)}.npz"


def cache_dir(data: str, frames: str) -> Path:
    c = data_dir() / "runs" / "op_parity" / "cache"
    return c / (data if frames == "gimm" else f"{data}@{frames}")


# ---------------------------------------------------------------- stages
def stages(m: Model, bench: str, run_dir: Path, shards: int = 0, subset: str = "", procs: int = 0) -> list:
    if m.family == "agent":                                   # third-party NAVSIM agent run as shipped: its own poses stages
        return agent_stages(m, bench, run_dir, shards, subset)
    data = NAVSIM[bench]["data"]
    ol_root(data)                                            # plans/ and the meta.json link that op_interp reads
    if m.family == "vt":                                      # vis_train: its own plans stage, then the parity tail (export, scoring, collect)
        S = [s for s in stages(replace(m, family="parity"), bench, run_dir, shards, subset, procs) if s.name != "prep"]
        S[[s.name for s in S].index("plans")] = R.Stage("plans", R.stage_cmd("jev", "vt-plans", m.spec, bench, plan_file(m, bench)),
                                                        done=str(plan_file(m, bench)), vram=24, cpu=8, ram=32, tries=2)
        return S
    S = []
    ts = m.opt in TS_OPTS
    if ts:                                                    # turn selector: the base model's plans / export (shared, usually finished), then `select`
        S = [s for s in stages(replace(m, opt=""), bench, run_dir, shards, subset, procs) if s.name in ("prep", "plans", "export")]
        S.append(R.Stage("select", R.stage_cmd("op-train", "ts-select", m.spec, bench, run_dir), done=str(pred_file(m, bench)), vram=24, cpu=4, ram=40,
                         after=["export"], tries=2))
    if m.family == "wajepa":
        raise SystemExit(f"{m.name} on {bench}: stored reference only ({m.stored.get(bench)}); its runner is experiments/top10 "
                         "(navhard: experiments/op_parity/scripts/pp_navhard_wajepa.sh)")
    if m.family == "pred":                                    # supplied poses (usage probes): score + collect only
        if not pred_file(m, bench).exists():
            raise SystemExit(f"{m.name}: write the (tokens, poses (n, 8, 3)) prediction file {pred_file(m, bench)} first")
    pf = plan_file(m, bench)
    if ts or m.family == "pred":
        pass
    elif m.family == "parity" and not pf.exists():
        old_stem = f"{m.frames}@cinque_PP{m.name}" + (f"_{m.opt}" if m.opt else "")
        old = data_dir() / "runs/op_lb" / data / "plans" / f"{old_stem}.npz"
        if old.exists():
            _copy_into(old, pf)
    if ts or m.family == "pred":
        pass
    elif m.family == "parity" and not m.unfreeze:
        need = [cache_dir(data, "gimm") / "tab.npz", cache_dir(data, m.frames) / "front.npy"]
        if not all(p.exists() for p in need):
            fr = [] if m.frames == "gimm" else ["--frames", m.frames]
            S.append(R.Stage("prep", [R.py("op-train"), str(PP / "pp_prep.py"), "--data", data, *fr, "--workers", "20"],
                             done=str(cache_dir(data, m.frames) / "front.npy"), vram=20, cpu=22, ram=40))
        if m.thead:
            assert not m.opt, f"{m.spec}: no option is defined for trajectory-head checkpoints"
            S.append(R.Stage("poses", R.stage_cmd("op-train", "thead-poses", m.spec, bench, pred_file(m, bench)), done=str(pred_file(m, bench)),
                             vram=24, cpu=8, ram=24, after=[s.name for s in S], tries=2))
        else:
            S.append(R.Stage("plans", R.stage_cmd("op-train", "parity-plans", m.spec, bench, pf), done=str(pf), vram=24, cpu=8, ram=24,
                             after=[s.name for s in S], tries=2))
    elif m.unfreeze:
        S.append(R.Stage("plans", R.stage_cmd("jev", "unfreeze-plans", m.spec, bench, pf), done=str(pf), vram=60, cpu=20, ram=48, tries=2))
    else:
        S.append(R.Stage("plans", R.stage_cmd("jev", "onnx-plans", m.spec, bench, pf), done=str(pf), vram=24, cpu=16, ram=32, tries=2))
    pr = pred_file(m, bench)
    if not ts and not m.thead and m.family != "pred":
        S.append(R.Stage("export", [R.py("jev"), str(REPO / "experiments/op_openloop/lib/op_interp.py"), "nav-export", "--data", data,
                                    "--adapters", adapter(m), "--plans", stem(m)],
                         done=str(pr), env={"OPI_ROOT": OL_REL}, vram=0.5, cpu=2, ram=8, after=["plans"]))
    last = "select" if ts else "poses" if m.thead else "" if m.family == "pred" else "export"
    if bench == "navtest":
        k = shards or default_shards()
        thr = max(4, min(16, int(R_cores() // max(k, 1)) - 1))
        for i in range(k):
            S.append(R.Stage(f"score{i}of{k}", R.stage_cmd("jev", "navsim-score", m.spec, bench, run_dir, i, k, subset),
                             done=str(run_dir / "score" / f"s{i}of{k}.csv"), vram=0.5, cpu=thr + 1, ram=24,
                             env={"NAVSIM_THREADS": str(thr)}, after=[last] if last else [], tries=2))
        sc = [s.name for s in S if s.name.startswith("score")]
    else:
        n = procs or max(8, min(16, int(R_cores()) - 2))
        S.append(R.Stage("harness", [R.py("navsim2"), str(HARNESS), "--poses", str(pr), "--out", str(run_dir / "harness"), "--procs", str(n)],
                         done=str(run_dir / "harness" / "harness_summary.json"), vram=0.5, cpu=n, ram=32, after=[last] if last else [], tries=2))
        sc = ["harness"]
    S.append(R.Stage("collect", R.stage_cmd("jev", "navsim-collect", m.spec, bench, run_dir, subset), done=str(run_dir / "DONE"),
                     vram=0.5, cpu=2, ram=8, after=sc))
    return S


def R_cores() -> float:
    try:
        from ..cl import box
        return box.probe().cores
    except Exception:
        return float(os.cpu_count() or 8)


def default_shards() -> int:
    """navtest scoring shards: one per ~12 cores of the box's quota (6 on the 75-core box). Small shards backfill a busy box
    (the pool charges the declared cores for a job's first 5 min); whole logs per shard keep the scores identical."""
    return max(1, round(R_cores() / 12))


# ---------------------------------------------------------------- plans
def select_stage(spec: str, bench: str, run_dir: str) -> None:
    """Turn selector (decision 191 N7) on the exported poses of the base SH30 model; writes this model's pred file."""
    _pp_path()
    import turn_selbench as TB
    TB.select(spec, bench, run_dir)


def _pp_path():
    for p in (REPO, REPO / "lib", REPO / "scripts", REPO / "experiments/op_adapt_r2/lib", PP):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def parity_plans(spec: str, bench: str, out: str, batch: int = 128, data: str = "") -> None:
    """pp_eval.py plans for one model (the same loop, batch and fp16 path, so plans are bit-identical), written atomically."""
    import torch
    _pp_path()
    import pp_train as T
    m = resolve(spec, check=True)
    data = data or NAVSIM[bench]["data"]
    dev = torch.device("cuda")
    side_ok = (cache_dir(data, "gimm") / "side.npy").exists()
    S = T.Store([data], dev, need_side=side_ok, frames=m.frames)
    names = S.tab["names"]
    mt = json.loads((data_dir() / "runs" / "op_lb" / data / "meta.json").read_text())
    assert names.tolist() == mt["names"], "pp_prep cache rows differ from op_lb meta"
    model = T.load_pmodel(m.name, dev) if not m.ckpt else _load_ckpt(T, m.ckpt, dev)
    if m.opt in ("sg", "dn"):                                # serving-side variants of the adapter bias (experiments/op_parity/results/stop_gate_xboard.md)
        assert model.adapter is not None and not model.adapter.use_side, f"{m.spec}: sg / dn are defined for ego-only adapters"
        if m.opt == "sg":
            model.gate = SG_THRESHOLD                        # bias = 0 where the fed speed (ego vx) < 0.5 m/s, fixed
        else:
            model.bias_sub = nav_mean_bias(model, S, m.name, dev)
    mem = getattr(model, "mem", None)                        # front-token memory arms (pp_train --mem): bank in tab order; :noside masks it
    M = T.Tokens([T.MEM_ROOT / mem / f"{data}.npy"], dev) if mem else None
    assert m.opt != "mshuf" or M is not None, f"{m.spec}: :mshuf needs a memory arm (pp_train --mem)"
    perm = torch.as_tensor(_derange(S.tab["log"]), device=dev) if m.opt == "mshuf" else None   # :mshuf: the bank row of a token of another log
    sideoff = None                                           # :sideoff: three-view banks (vtr_W*, views CAM_F0 / CAM_L0 / CAM_R0), the side views masked
    if m.opt == "sideoff":
        shp = None if M is None else (M.mms[0].shape if M.host else M.t.shape)
        assert shp is not None and len(shp) == 4 and shp[1] == 3, f"{m.spec}: :sideoff needs a memory arm with a three-view bank (vtr_W*)"
        sideoff = torch.tensor([True, False, False], device=dev).expand(batch, -1)
    sl = model.net.slices
    pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
    ps = np.arange(sl["plan"].start + 495, sl["plan"].start + 990)
    mu, sd = np.zeros((S.n, 33, 15), np.float32), np.zeros((S.n, 33, 15), np.float32)
    # lead head as the HUGSIM server decodes it (jevdrive.openpilot.model.decode; hugsim_zs_server.lead_xv): selection 0 at t = 0
    lp, lx, lv = (np.full(S.n, np.nan, np.float32) for _ in range(3))
    has_lead = "lead" in sl and "lead_prob" in sl
    with torch.no_grad():
        for i in range(0, S.n, batch):
            r = torch.arange(i, min(i + batch, S.n), device=dev)
            mask = torch.zeros(len(r), 1 if M is not None else 3, dtype=torch.bool, device=dev) if m.opt == "noside" else sideoff[:len(r)] if sideoff is not None else None
            side = M[r if perm is None else perm[r]] if M is not None else (S.side[r] if side_ok else None)
            o = model(S.front[r], S.ego[r], S.tc[r], side, mask).float().cpu().numpy()
            mu[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15)
            sd[i:i + len(r)] = np.exp(np.minimum(o[:, ps], 11)).reshape(-1, 33, 15)
            if has_lead:
                ld = o[:, sl["lead"]]
                ld = ld[:, : ld.shape[1] // 2].reshape(-1, 3, 6, 4)
                lp[i:i + len(r)] = 1 / (1 + np.exp(-np.clip(o[:, sl["lead_prob"]][:, 0], -11, None)))
                lx[i:i + len(r)], lv[i:i + len(r)] = ld[:, 0, 0, 0], ld[:, 0, 0, 2]
    _save_plans(out, names=names, plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11], plan_mu=mu, plan_std=sd,
                lead_prob=lp, lead_x=lx, lead_v=lv,
                steps=31, info=json.dumps({"model": f"op_parity {m.spec}", "source": "jevdrive.bench.navsim.parity_plans", "frames": m.frames}))


def _derange(log, seed: int = 0):
    """A permutation p with log[p[i]] != log[i] for every row (experiments/vis_train/scripts/vt.py derange, same draws: one shuffle per data dir)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(log))
    for _ in range(1000):
        bad = np.flatnonzero(log[p] == log)
        if not len(bad):
            return p
        for i in bad:
            j = int(rng.integers(len(log)))
            p[i], p[j] = p[j], p[i]
    raise RuntimeError("no derangement found")


def thead_poses(spec: str, bench: str, out: str, batch: int = 128, data: str = "") -> None:
    """Prediction file (tokens, poses (n, 8, 3) rear axle) of a parity checkpoint with a trajectory head: the head's served plan
    (lib/traj_head.serve: fixed noise row 0 for a flow head) on the pp_prep token cache, same loop and fp16 path as parity_plans."""
    import torch
    _pp_path()
    import pp_train as T
    import traj_head as TH
    m = resolve(spec, check=True)
    assert m.thead, f"{spec}: no thead.pt next to {m.ckpt}"
    data = data or NAVSIM[bench]["data"]
    dev = torch.device("cuda")
    S = T.Store([data], dev, need_side=False, frames=m.frames)
    mt = json.loads((data_dir() / "runs" / "op_lb" / data / "meta.json").read_text())
    assert S.tab["names"].tolist() == mt["names"], "pp_prep cache rows differ from op_lb meta"
    P = TH.serve(_load_ckpt(T, m.ckpt, dev), TH.load(Path(m.ckpt).with_name("thead.pt"), dev), S, batch=batch)[0]
    assert np.isfinite(P).all()
    _save_plans(out, tokens=S.tab["names"].astype(str), poses=P)


SG_THRESHOLD = 0.5


def nav_mean_bias(model, S, tag: str, dev, save: bool = True):
    """(32, 512) fp32 mean adapter bias over all rows of the store (navtest, decision 162 `biasdenav`); saved for the HUGSIM client
    as $DATA_DIR/runs/bench/sg/meanbias-<tag>.npy."""
    import torch
    acc = torch.zeros(32, 512, dtype=torch.float64, device=dev)
    with torch.no_grad():
        for i in range(0, S.n, 512):
            acc += model.adapter(S.ego[i:i + 512], None, None).double().sum(0)
    mb = (acc / S.n).float()
    if save:
        f = data_dir() / "runs" / "bench" / "sg" / f"meanbias-{tag}.npy"
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, mb.cpu().numpy())
    return mb


def _load_ckpt(T, ckpt: str, dev):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mm = T.PModel(ck["model"]["arm"]).to(dev).eval()
    mm.load_state(ck["model"])
    return mm


def _save_plans(out, **arrays) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)


def _copy_into(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, out)


def onnx_plans(spec: str, bench: str, out: str, procs: int = 4) -> None:
    """scripts/op_lb.py run (TensorRT) unless op_lb already holds the plan file of this stem; copied into the bench root."""
    m = resolve(spec, check=True)
    data = NAVSIM[bench]["data"]
    st = f"{m.frames}@{m.base}" + (f"_O{m.name}" if m.onnx else "")
    src = data_dir() / "runs" / "op_lb" / data / "plans" / f"{st}.npz"
    if not src.exists():
        cmd = [R.py("openpilot"), str(REPO / "scripts/op_lb.py"), "run", "--data", data, "--frames", m.frames, "--model", m.base,
               "--procs", str(procs)] + (["--onnx", m.onnx, "--tag", f"O{m.name}"] if m.onnx else [])
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)
    _copy_into(src, Path(out))


def unfreeze_plans(spec: str, bench: str, out: str) -> None:
    """pp_unfreeze.py plans (vision fine-tuned arms read pixels, not the token cache)."""
    m = resolve(spec, check=True)
    data = NAVSIM[bench]["data"]
    src = data_dir() / "runs" / "op_lb" / data / "plans" / f"{m.frames}@cinque_PP{m.name}.npz"
    if not src.exists():
        subprocess.run([R.py("op-train"), str(PP / "pp_unfreeze.py"), "plans", "--data", data, "--frames", m.frames, "--models", m.name],
                       check=True, cwd=REPO)
    _copy_into(src, Path(out))


def vt_plans(spec: str, bench: str, out: str) -> None:
    """experiments/vis_train/scripts/vt.py plans: cached slot tokens + the pixel cache (memory branch / own encoder); :noside / :mshuf / :sideoff."""
    m = resolve(spec, check=True)
    subprocess.run([R.py("op-train"), str(REPO / "experiments/vis_train/scripts/vt.py"), "plans", "--tag", m.name, "--data", NAVSIM[bench]["data"],
                    "--out", str(out), "--mem", {"": "on", "noside": "off", "mshuf": "shuf", "sideoff": "sideoff"}[m.opt]], check=True, cwd=REPO)


# ---------------------------------------------------------------- scoring
def eval_name(m: Model, bench: str, i: int, k: int, subset: str = "") -> str:
    return f"bench_{bench}_{stem(m).replace('@', '-')}" + (f"_{subset.replace('/', '-')}" if subset else "") + f"_s{i}of{k}"


def score_shard(spec: str, bench: str, run_dir: str, i: int, k: int, subset: str = "") -> None:
    """One navtest scoring shard: the devkit on the tokens of whole logs, its per-token CSV copied to <run>/score/s<i>of<k>.csv."""
    m = resolve(spec)
    run_dir = Path(run_dir)
    toks, logs = navsim_tokens(bench, subset)
    mine = set(log_shards(logs, int(k))[int(i)])
    sel = [t for t, lg in zip(toks, logs) if lg in mine]
    d = run_dir / "score"
    d.mkdir(parents=True, exist_ok=True)
    tf = d / f"tokens_s{i}of{k}.txt"
    tf.write_text("\n".join(sel) + "\n")
    name = eval_name(m, bench, i, k, subset)
    R.status(run_dir, f"scoring shard {i}/{k}: {len(sel)} tokens of {len(mine)} logs")
    env = dict(os.environ, TOKENS_FILE=str(tf))
    t0 = time.time()
    with open(d / f"score_s{i}of{k}.log", "w") as log:
        subprocess.run(["bash", str(SCORE_SH), "score", "v2", NAVSIM[bench]["split"], name, str(preds_of(m, bench, run_dir))], check=True,
                       cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    fs = [f for f in (data_dir() / "runs" / "navsim" / "eval" / f"v2_{NAVSIM[bench]['split']}_{name}").glob("*/*.csv")
          if f.stat().st_mtime >= t0 - 5]
    if not fs:
        raise RuntimeError(f"devkit wrote no CSV for {name}; see {d}/score_s{i}of{k}.log")
    t, _ = read_devkit_csv(sorted(fs)[-1])
    miss = set(sel) - set(t.index)
    if miss:
        raise RuntimeError(f"shard {i}/{k}: {len(miss)} of {len(sel)} tokens missing from the devkit CSV")
    _copy_into(sorted(fs)[-1], d / f"s{i}of{k}.csv")


def read_devkit_csv(path):
    """(per-token DataFrame indexed by token, the 'average' row or None) of a devkit v2 CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    tok = df.token.astype(str)
    avg = df[tok.str.startswith("average")]
    t = df[~tok.str.startswith("average") & ~tok.str.startswith("extended_pdm_score")]
    return t.set_index("token"), (avg.iloc[0] if len(avg) else None)


def collect(spec: str, bench: str, run_dir: str, subset: str = "") -> None:
    import pandas as pd
    m = resolve(spec)
    run_dir = Path(run_dir)
    toks, logs = navsim_tokens(bench, subset)
    if bench == "navtest":
        parts = [read_devkit_csv(f)[0] for f in sorted((run_dir / "score").glob("s*of*.csv"))]
        t = pd.concat(parts)
        assert not t.index.duplicated().any(), "a token was scored in two shards"
        lg = dict(zip(toks, logs))
        t = t.loc[[x for x in toks if x in t.index]]
        u = pd.DataFrame({"token": t.index, "log": [lg[x] for x in t.index], "score": t["score"].to_numpy(float)})
        for k, c in SUBS.items():
            u[k] = t[c].to_numpy(float)
        summ = {"n": len(u), "missing": int(len(toks) - len(u)), "EPDMS": 100 * float(u.score.mean())}
        summ |= {k: 100 * float(np.nanmean(u[k])) for k in SUBS}
    else:
        g = pd.read_csv(run_dir / "harness" / "harness_groups.csv")
        lg = dict(zip(toks, logs))
        u = g.assign(log=[lg.get(o, "?") for o in g.orig])
        s = json.loads((run_dir / "harness" / "harness_summary.json").read_text())
        summ = {"n": len(u), **{k: float(s[k]) for k in ("combined", "stage1", "stage2")}}
    if m.family == "agent":
        summ |= agent_v1_summary(m, run_dir)
    u.to_csv(run_dir / "units.csv", index=False)
    summ |= {"model": m.spec, "bench": bench, "subset": subset}
    R.atomic_write(run_dir / "summary.json", json.dumps(summ, indent=1))
    R.status(run_dir, "done " + ", ".join(f"{k} {v:.2f}" for k, v in summ.items() if isinstance(v, float)))
    R.atomic_write(run_dir / "DONE", json.dumps(dict(t=time.strftime("%F %T"), **summ)) + "\n")


# ---------------------------------------------------------------- third-party NAVSIM agents (family `agent`)
V1_SUBS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance",
           "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "C": "comfort"}
AGENT_POSES = REPO / "jevdrive/bench/agent_poses.py"


def preds_of(m: Model, bench: str, run_dir) -> Path:
    """Prediction file a scoring stage reads: <run>/preds.npz for third-party agents (one per run, so a smoke subset never
    overwrites the full run), the shared bench/ol pred file otherwise."""
    return Path(run_dir) / "preds.npz" if m.family == "agent" else pred_file(m, bench)


def agent_env(c: dict) -> dict:
    nav = data_dir() / "datasets" / "navsim"
    return {"NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "NUPLAN_MAPS_ROOT": str(nav / "maps"), "OPENSCENE_DATA_ROOT": str(nav),
            "NAVSIM_EXP_ROOT": str(data_dir() / "runs" / "navsim" / "eval"), "NAVSIM_DEVKIT_ROOT": c["repo"], "PYTHONPATH": c["repo"],
            "OPENBLAS_CORETYPE": "Haswell", "HF_HUB_OFFLINE": "1", **c.get("env_vars", {})}


def agent_cmd(c: dict, tokens: Path, out: Path) -> list:
    """runner `agent`: agent_poses.py with the fork's hydra config + overrides; runner `cmd`: the JSON's own argv template
    ({py} {repo} {tokens} {out}) for models whose shipped inference is not a NAVSIM agent (it must write the same npz)."""
    py = str(data_dir() / "envs" / c["env"] / "bin" / "python")
    if c.get("runner", "agent") == "cmd":
        return [str(x).format(py=py, repo=c["repo"], tokens=tokens, out=out) for x in c["cmd"]]
    return [py, str(AGENT_POSES), "--repo", c["repo"], "--config-name", c.get("config_name", "default_run_pdm_score"),
            *(["--config-dir", c["config_dir"]] if c.get("config_dir") else []), "--tokens", str(tokens), "--out", str(out),
            "--sensors", json.dumps(c.get("sensor_overrides", {})), "--loader-kw", json.dumps(c.get("loader_kwargs", {})),
            "--", *c.get("overrides", [])]


def agent_stages(m: Model, bench: str, run_dir: Path, shards: int = 0, subset: str = "") -> list:
    """poses c<i>of<K> (GPU, the model's env, whole logs per chunk) -> merge -> v2 EPDMS shards + v1.1 PDMS -> collect."""
    if bench != "navtest":
        raise SystemExit(f"{m.name}: third-party agents are registered for navtest only")
    c = agent_cfg(m.name)
    toks, logs = navsim_tokens(bench, subset)
    ul = sorted(set(logs.tolist()))
    k = max(1, min(int(c.get("chunks", 8)), len(ul), -(-len(toks) // int(c.get("min_chunk", 200)))))
    pd_ = Path(run_dir) / "poses"
    pd_.mkdir(parents=True, exist_ok=True)
    lg = dict(zip(toks.tolist(), logs.tolist()))
    S = []
    for i, part in enumerate(log_shards(logs, k)):
        part = set(part)
        tsv, out = pd_ / f"c{i}of{k}.tsv", pd_ / f"c{i}of{k}.npz"
        if not tsv.exists():
            R.atomic_write(tsv, "".join(f"{t}\t{lg[t]}\n" for t in toks.tolist() if lg[t] in part))
        S.append(R.Stage(f"poses{i}of{k}", agent_cmd(c, tsv, out), done=str(out), vram=float(c.get("vram", 16)), cpu=int(c.get("cpu", 6)),
                         ram=float(c.get("ram", 32)), env=agent_env(c), tries=2))
    S.append(R.Stage("merge", R.stage_cmd("jev", "agent-merge", m.spec, bench, run_dir, subset), done=str(preds_of(m, bench, run_dir)),
                     vram=0.5, cpu=2, ram=8, after=[s.name for s in S]))
    n = max(1, min(shards or default_shards(), len(ul), -(-len(toks) // 400)))   # never an empty shard (an empty token filter = no filter)
    thr = max(4, min(16, int(R_cores() // max(n, 1)) - 1))
    for i in range(n):
        S.append(R.Stage(f"score{i}of{n}", R.stage_cmd("jev", "navsim-score", m.spec, bench, run_dir, i, n, subset),
                         done=str(run_dir / "score" / f"s{i}of{n}.csv"), vram=0.5, cpu=thr + 1, ram=24,
                         env={"NAVSIM_THREADS": str(thr)}, after=["merge"], tries=2))
    t1 = max(8, min(24, int(R_cores()) // 4))
    S.append(R.Stage("score_v1", R.stage_cmd("jev", "agent-score-v1", m.spec, bench, run_dir, subset), done=str(run_dir / "v1" / "pdms.csv"),
                     vram=0.5, cpu=t1 + 1, ram=32, env={"NAVSIM_THREADS": str(t1)}, after=["merge"], tries=2))
    S.append(R.Stage("collect", R.stage_cmd("jev", "navsim-collect", m.spec, bench, run_dir, subset), done=str(run_dir / "DONE"),
                     vram=0.5, cpu=2, ram=8, after=[s.name for s in S if s.name.startswith("score")]))
    return S


def agent_merge(spec: str, bench: str, run_dir: str, subset: str = "") -> None:
    """Chunk npz files -> <run>/preds.npz in bench token order; fails on a missing / duplicated token or a non-finite pose."""
    run_dir = Path(run_dir)
    toks, _ = navsim_tokens(bench, subset)
    got, wall = {}, 0.0
    for f in sorted((run_dir / "poses").glob("c*of*.npz")):
        z = np.load(f)
        wall += float(z["wall_s"]) if "wall_s" in z else 0.0
        for t, p in zip(z["tokens"].tolist(), z["poses"]):
            assert t not in got, f"token {t} in two chunks"
            got[t] = p
    miss = [t for t in toks.tolist() if t not in got]
    if miss:
        raise RuntimeError(f"{len(miss)} of {len(toks)} tokens have no poses, e.g. {miss[:3]}")
    P = np.stack([got[t] for t in toks.tolist()]).astype(np.float32)
    assert P.shape[1:] == (8, 3) and np.isfinite(P).all()
    _save_plans(preds_of(resolve(spec), bench, run_dir), tokens=toks.astype(str), poses=P, poses_wall_s=wall)
    R.status(run_dir, f"merged poses of {len(toks)} tokens ({wall / 3600:.2f} GPU-job h)")


def agent_score_v1(spec: str, bench: str, run_dir: str, subset: str = "") -> None:
    """NAVSIM v1.1 PDMS of the run's poses (navsim_zs_score.sh score v1: PrecomputedAgent, envs/navsim1, v1_navtest metric cache);
    a subset is passed as TOKENS_FILE (the full split needs none: a 12 k-token hydra list exceeds the argv limit)."""
    m = resolve(spec)
    run_dir = Path(run_dir)
    d = run_dir / "v1"
    d.mkdir(parents=True, exist_ok=True)
    toks, _ = navsim_tokens(bench, subset)
    env = dict(os.environ)
    if subset:
        (d / "tokens.txt").write_text("\n".join(toks.tolist()) + "\n")
        env["TOKENS_FILE"] = str(d / "tokens.txt")
    name = f"bench_{bench}_{stem(m)}" + (f"_{subset.replace('/', '-')}" if subset else "")
    t0 = time.time()
    with open(d / "score.log", "w") as log:
        subprocess.run(["bash", str(SCORE_SH), "score", "v1", NAVSIM[bench]["split"], name, str(preds_of(m, bench, run_dir))], check=True,
                       cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    fs = [f for f in (data_dir() / "runs" / "navsim" / "eval" / f"v1_{NAVSIM[bench]['split']}_{name}").glob("*/*.csv") if f.stat().st_mtime >= t0 - 5]
    if not fs:
        raise RuntimeError(f"v1 devkit wrote no CSV for {name}; see {d}/score.log")
    import pandas as pd
    t = pd.read_csv(sorted(fs)[-1])
    t = t[~t.token.astype(str).str.startswith("average")]
    miss = set(toks.tolist()) - set(t.token.astype(str))
    if miss:
        raise RuntimeError(f"{len(miss)} of {len(toks)} tokens missing from the v1 CSV")
    _copy_into(sorted(fs)[-1], d / "pdms.csv")


def agent_v1_summary(m: Model, run_dir: Path) -> dict:
    """PDMS (v1.1) next to the published number: mean over tokens (failed tokens count 0, as the devkit's average row) and
    the gate verdict |PDMS - published| <= tolerance (full navtest only; a subset is a smoke read)."""
    import pandas as pd
    c = agent_cfg(m.name)
    t = pd.read_csv(run_dir / "v1" / "pdms.csv")
    t = t[~t.token.astype(str).str.startswith("average")]
    t.to_csv(run_dir / "units_v1.csv", index=False)
    out = {"PDMS_v1": 100 * float(t.score.fillna(0).mean()), "v1_valid": int(t.valid.astype(bool).sum()) if "valid" in t else len(t)}
    out |= {f"v1_{k}": 100 * float(t[col].mean()) for k, col in V1_SUBS.items() if col in t}
    pub = c.get("published", {})
    if pub.get("value") is not None:
        out |= {"published": float(pub["value"]), "published_metric": pub.get("metric", "PDMS v1"), "tolerance": float(c.get("tolerance", 0.5)),
                "delta_vs_published": out["PDMS_v1"] - float(pub["value"])}
    return out
