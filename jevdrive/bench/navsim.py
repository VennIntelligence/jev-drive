"""NAVSIM open-loop readouts: navtest EPDMS (v2 devkit, the harness that reproduced WA-JEPA's 91.71) and navhard two-stage.

Pipeline per model (every step is the code the lanes used, wrapped):
  prep     parity models only, when the pp_prep token cache of the frame protocol is missing: experiments/op_parity/scripts/pp_prep.py
  plans    the model's 33-step plan at t0 of every token, op_lb plan format (names, plan_pos / vel / yaw, plan_mu / std):
           parity: the torch port on the cached tokens (pp_train.PModel, fp16, batch 128: pp_eval.py plans, one model);
           memory arms (pp_train --mem) also read their front-token bank (runs/op_parity/mem/<kind>/<data>.npy; :noside masks it);
           UF-* arms: pp_unfreeze.py plans (pixels); onnx: scripts/op_lb.py run (TensorRT, envs/openpilot; an existing op_lb
           plan file of the same stem is reused)
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
from pathlib import Path

import numpy as np

from . import runner as R
from .models import REPO, Model, data_dir, resolve
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
    data = NAVSIM[bench]["data"]
    ol_root(data)                                            # plans/ and the meta.json link that op_interp reads
    S = []
    if m.family == "wajepa":
        raise SystemExit(f"{m.name} on {bench}: stored reference only ({m.stored.get(bench)}); its runner is experiments/top10 "
                         "(navhard: experiments/op_parity/scripts/pp_navhard_wajepa.sh)")
    pf = plan_file(m, bench)
    if m.family == "parity" and not pf.exists():
        old_stem = f"{m.frames}@cinque_PP{m.name}" + (f"_{m.opt}" if m.opt else "")
        old = data_dir() / "runs/op_lb" / data / "plans" / f"{old_stem}.npz"
        if old.exists():
            _copy_into(old, pf)
    if m.family == "parity" and not m.unfreeze:
        need = [cache_dir(data, "gimm") / "tab.npz", cache_dir(data, m.frames) / "front.npy"]
        if not all(p.exists() for p in need):
            fr = [] if m.frames == "gimm" else ["--frames", m.frames]
            S.append(R.Stage("prep", [R.py("op-train"), str(PP / "pp_prep.py"), "--data", data, *fr, "--workers", "20"],
                             done=str(cache_dir(data, m.frames) / "front.npy"), vram=20, cpu=22, ram=40))
        S.append(R.Stage("plans", R.stage_cmd("op-train", "parity-plans", m.spec, bench, pf), done=str(pf), vram=24, cpu=8, ram=24,
                         after=[s.name for s in S], tries=2))
    elif m.unfreeze:
        S.append(R.Stage("plans", R.stage_cmd("jev", "unfreeze-plans", m.spec, bench, pf), done=str(pf), vram=60, cpu=20, ram=48, tries=2))
    else:
        S.append(R.Stage("plans", R.stage_cmd("jev", "onnx-plans", m.spec, bench, pf), done=str(pf), vram=24, cpu=16, ram=32, tries=2))
    pr = pred_file(m, bench)
    S.append(R.Stage("export", [R.py("jev"), str(REPO / "experiments/op_openloop/lib/op_interp.py"), "nav-export", "--data", data,
                                "--adapters", adapter(m), "--plans", stem(m)],
                     done=str(pr), env={"OPI_ROOT": OL_REL}, vram=0.5, cpu=2, ram=8, after=["plans"]))
    if bench == "navtest":
        k = shards or default_shards()
        thr = max(4, min(16, int(R_cores() // max(k, 1)) - 1))
        for i in range(k):
            S.append(R.Stage(f"score{i}of{k}", R.stage_cmd("jev", "navsim-score", m.spec, bench, run_dir, i, k, subset),
                             done=str(run_dir / "score" / f"s{i}of{k}.csv"), vram=0.5, cpu=thr + 1, ram=24,
                             env={"NAVSIM_THREADS": str(thr)}, after=["export"], tries=2))
        sc = [s.name for s in S if s.name.startswith("score")]
    else:
        n = procs or max(8, min(16, int(R_cores()) - 2))
        S.append(R.Stage("harness", [R.py("navsim2"), str(HARNESS), "--poses", str(pr), "--out", str(run_dir / "harness"), "--procs", str(n)],
                         done=str(run_dir / "harness" / "harness_summary.json"), vram=0.5, cpu=n, ram=32, after=["export"], tries=2))
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
    mem = getattr(model, "mem", None)                        # front-token memory arms (pp_train --mem): bank in tab order; :noside masks it
    M = T.Tokens([T.MEM_ROOT / mem / f"{data}.npy"], dev) if mem else None
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
            mask = torch.zeros(len(r), 1 if M is not None else 3, dtype=torch.bool, device=dev) if m.opt == "noside" else None
            side = M[r] if M is not None else (S.side[r] if side_ok else None)
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
        subprocess.run(["bash", str(SCORE_SH), "score", "v2", NAVSIM[bench]["split"], name, str(pred_file(m, bench))], check=True,
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
    u.to_csv(run_dir / "units.csv", index=False)
    summ |= {"model": m.spec, "bench": bench, "subset": subset}
    R.atomic_write(run_dir / "summary.json", json.dumps(summ, indent=1))
    R.status(run_dir, "done " + ", ".join(f"{k} {v:.2f}" for k, v in summ.items() if isinstance(v, float)))
    R.atomic_write(run_dir / "DONE", json.dumps(dict(t=time.strftime("%F %T"), **summ)) + "\n")
