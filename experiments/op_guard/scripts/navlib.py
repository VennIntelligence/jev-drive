"""NAVSIM guard lines (navtest, navhard): the existing open-loop chain, pointed at guard-owned run dirs.

Chain per (candidate, board), every step reused as shipped:
  1. rollout   scripts/op_lb.py run (envs/openpilot; GIMM history frames, true CAM_F0 height, no command, native arm =
               docs/openpilot-interface.md NAVSIM row) -> $DATA_DIR/runs/op_lb/<data>/plans/<stem>.npz
  2. export    experiments/op_openloop/lib/op_interp.py nav-export (envs/jevdrive, OPI_ROOT=op_lb) -> preds/<stem'>__base.npz
  3. official  experiments/zeroshot_openloop/archive/navsim_zs_score.sh score (the devkits as shipped: v1 PDMS navtest,
               v2 two-stage EPDMS navhard) -> $DATA_DIR/runs/navsim/eval/<ver>_<split>_opg_<data>_<stem'>__base/
  4. navhard   nav_harness.py (envs/navsim2): per-token devkit pdm_score + devkit aggregation (the harness of
               experiments/skill_pack/scripts/hist_align_report.py), per scene-mapping group values for paired CIs.

Guard-owned op_lb data dirs (so --force never touches another experiment's files): g_navtest / g_navhard = symlinks to the
meta.json / gimm.npy / tokens.txt of lb_navtest / lb_navhard; g_navtest_sub = the frozen subset (own meta, gimm rows copied).
Cache: a plan file of the legacy run dirs (lb_navtest / lb_navhard, same stem) is imported instead of a rollout when its
recorded config is the native arm (frames gimm, backend trt, align none, preroll 0, schedule none) and, for an ONNX candidate,
it is newer than the ONNX; provenance (<stem>.guard.json) says `rollout` or `legacy`. --force always rolls out.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guardlib as G  # noqa: E402

REPO = G.REPO
sys.path.insert(0, str(REPO))
D = G.data_dir()
ENVS = D / "envs"
OPLB = D / "runs" / "op_lb"
EVAL = D / "runs" / "navsim" / "eval"
RESULTS = G.GUARD / "results"
SUB_SPLIT = "navsim/op-guard-navtest-sub"          # jevdrive.data.splits (nav_subset.py freeze)
EARLY = RESULTS / "navhard_early_turn.csv"         # decision 110's set, frozen from runs/hugsim-spinattr/navhard_ref.csv
BOARDS = {
    "navtest_sub": dict(src="lb_navtest", data="g_navtest_sub", ver="v1", split="navtest"),
    "navtest": dict(src="lb_navtest", data="g_navtest", ver="v1", split="navtest"),
    "navhard": dict(src="lb_navhard", data="g_navhard", ver="v2", split="navhard_two_stage"),
}
NATIVE = dict(frames="gimm", backend="trt", preroll=0.0, schedule="none")
PROCS = 4                                          # op_lb rollout shards per card (measured: see the line docstrings)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ncpus(cpus: str) -> int:
    if not cpus:
        from jevdrive.common import n_cpus
        return n_cpus()
    n = 0
    for p in cpus.split(","):
        a, _, b = p.partition("-")
        n += int(b or a) - int(a) + 1
    return n


def taskset(cpus: str) -> list:
    return ["taskset", "-c", cpus] if cpus else []


def sh(cmd, log: Path, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    say("$", " ".join(map(str, cmd)))
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}  ({time.strftime('%F %T')})\n")
        f.flush()
        rc = subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env={**os.environ, **(env or {})})
    if rc:
        raise SystemExit(f"rc {rc}: {' '.join(map(str, cmd))} (log {log})")


def stems(cand: dict) -> tuple[str, str]:
    """(plan stem, pred stem): shipped gimm@cinque / gimm-cinque__base, an ONNX candidate gimm@cinque_O<name>."""
    p = "gimm@cinque" + ("" if not cand.get("onnx") else f"_O{cand['name']}")
    return p, p.replace("@", "-") + "__base"


def file_sig(p) -> dict:
    p = Path(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return dict(path=str(p), size=p.stat().st_size, sha256=h.hexdigest())


# ---------------------------------------------------------------- data dirs

def ensure_data(board: str):
    b = BOARDS[board]
    d = OPLB / b["data"]
    (d / "plans").mkdir(parents=True, exist_ok=True)
    if board != "navtest_sub":
        for f in ("meta.json", "gimm.npy", "tokens.txt"):
            if not (d / f).exists():
                (d / f).symlink_to(OPLB / b["src"] / f)
        return d
    from jevdrive.data import splits
    sp = splits.load(SUB_SPLIT)
    stamp = d / "split_id"
    if stamp.is_file() and stamp.read_text().strip() == sp.id:
        return d
    say(f"building {d} for {sp.id} ({sp.n} tokens)")
    for f in ("meta.json", "gimm.npy", "tokens.txt", "split_id"):
        (d / f).unlink(missing_ok=True)
    mt = json.loads((OPLB / b["src"] / "meta.json").read_text())
    keep = set(sp.members)
    rows = np.array([i for i, t in enumerate(mt["names"]) if t in keep])
    assert len(rows) == sp.n, (len(rows), sp.n)
    n = len(mt["names"])
    sub = {k: ([v[i] for i in rows] if isinstance(v, list) and len(v) == n else v) for k, v in mt.items()}
    src = np.load(OPLB / b["src"] / "gimm.npy", mmap_mode="r")
    tmp = d / "gimm.tmp.npy"
    out = np.lib.format.open_memmap(tmp, "w+", src.dtype, (len(rows),) + src.shape[1:])
    for k in range(0, len(rows), 256):
        out[k:k + 256] = src[rows[k:k + 256]]
    out.flush()
    del out
    tmp.replace(d / "gimm.npy")
    (d / "tokens.txt").write_text("\n".join(sub["names"]) + "\n")
    (d / "meta.json").write_text(json.dumps(sub))
    stamp.write_text(sp.id + "\n")
    for f in (d / "plans").glob("*"):          # plans of an older subset version
        f.unlink()
    return d


def names(board: str) -> list:
    return json.loads((OPLB / BOARDS[board]["data"] / "meta.json").read_text())["names"]


# ---------------------------------------------------------------- 1. rollout (or legacy import)

def legacy_ok(f: Path, cand: dict):
    """Why a legacy op_lb plan file may stand in for a rollout of this candidate, or None."""
    if not f.is_file():
        return None
    z = np.load(f)
    i = json.loads(str(z["info"]))
    if any(i.get(k) != v for k, v in NATIVE.items() if k != "preroll") or float(i.get("preroll", -1)) != 0.0 or i.get("align") not in (None, "none") \
            or i.get("vcam") is not None or i.get("model") not in (None, "cinque"):
        return None
    if cand.get("onnx") and f.stat().st_mtime < Path(cand["onnx"]).stat().st_mtime:
        return None
    return dict(file=str(f), mtime=time.strftime("%F %T", time.localtime(f.stat().st_mtime)), info={k: i.get(k) for k in ("frames", "backend", "align", "preroll", "schedule")})


def trt_guard(cand: dict):
    """op_lb keys the TensorRT engine cache by --tag only (trt_cache/cinque-O<name>-trt): refuse an engine dir built from another ONNX."""
    if not cand.get("onnx"):
        return
    d = D / "runs/op_interp/trt_cache" / f"cinque-O{cand['name']}-trt"
    st = d / "guard_onnx.json"
    sig = file_sig(cand["onnx"])
    if st.is_file():
        if json.loads(st.read_text())["sha256"] != sig["sha256"]:
            raise SystemExit(f"{d} holds engines of another ONNX named {cand['name']}: rename the candidate (or move that dir away)")
        return
    eng = [p for p in d.glob("*") if p.is_file()] if d.is_dir() else []
    if eng and min(p.stat().st_mtime for p in eng) < Path(cand["onnx"]).stat().st_mtime:
        raise SystemExit(f"{d} has engines older than {cand['onnx']}: rename the candidate (or move that dir away)")
    d.mkdir(parents=True, exist_ok=True)
    st.write_text(json.dumps(sig))


def plans(cand: dict, board: str, gpu: int, cpus: str, procs: int, force: bool, logd: Path) -> dict:
    b = BOARDS[board]
    d = ensure_data(board)
    ps, _ = stems(cand)
    f, prov = d / "plans" / f"{ps}.npz", d / "plans" / f"{ps}.guard.json"
    if not force and f.is_file() and prov.is_file():
        return dict(json.loads(prov.read_text()), hit=True)
    for x in (f, prov):
        x.unlink(missing_ok=True)
    leg = None if force else legacy_ok(OPLB / b["src"] / "plans" / f"{ps}.npz", cand)
    t0 = time.time()
    if leg:
        z = dict(np.load(leg["file"]))
        want = names(board)
        pos = {t: k for k, t in enumerate(z["names"].tolist())}
        ix = np.array([pos[t] for t in want])
        n = len(z["names"])
        np.savez(f, **{k: (v[ix] if v.ndim and len(v) == n else v) for k, v in z.items()})
        p = dict(source="legacy", legacy=leg)
    else:
        if gpu < 0:
            raise SystemExit(f"{ps} on {board} needs a rollout but --gpu is {gpu} (no card held)")
        trt_guard(cand)
        cmd = taskset(cpus) + [ENVS / "openpilot/bin/python", "scripts/op_lb.py", "run", "--data", b["data"], "--frames", "gimm", "--model", "cinque",
                               "--procs", procs]
        if cand.get("onnx"):
            cmd += ["--onnx", cand["onnx"], "--tag", f"O{cand['name']}"]
        sh(cmd, logd / f"rollout_{board}.log", env=dict(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2"))
        z = np.load(f)
        p = dict(source="rollout", procs=procs, gpu=gpu, ms_per_scene=float(z["ms_per_scene"]))
    p.update(plan=str(f), n=len(names(board)), seconds=round(time.time() - t0, 1), onnx=file_sig(cand["onnx"]) if cand.get("onnx") else None,
             t=time.strftime("%F %T"))
    prov.write_text(json.dumps(p, indent=1))
    return dict(p, hit=False)


# ---------------------------------------------------------------- 2. export, 3. official scoring

def export(cand: dict, board: str, force: bool, logd: Path) -> Path:
    d = OPLB / BOARDS[board]["data"]
    ps, pr = stems(cand)
    out = d / "preds" / f"{pr}.npz"
    if force or not out.is_file() or out.stat().st_mtime < (d / "plans" / f"{ps}.npz").stat().st_mtime:
        sh([ENVS / "jevdrive/bin/python", "experiments/op_openloop/lib/op_interp.py", "nav-export", "--data", BOARDS[board]["data"], "--plans", ps, "--force"],
           logd / f"export_{board}.log", env=dict(OPI_ROOT="op_lb"))
    return out


def eval_name(cand: dict, board: str) -> str:
    b = BOARDS[board]
    return f"{b['ver']}_{b['split']}_opg_{b['data']}_{stems(cand)[1]}"


def official(cand: dict, board: str, cpus: str, force: bool, logd: Path) -> dict:
    """Official devkit score of the candidate's pose file; reruns when the pose file is newer than the CSV. -> {csv, seconds, hit}."""
    b = BOARDS[board]
    preds = OPLB / b["data"] / "preds" / f"{stems(cand)[1]}.npz"
    e = EVAL / eval_name(cand, board)
    fs = sorted(glob.glob(str(e / "*/*.csv")))
    if fs and not force and Path(fs[-1]).stat().st_mtime > preds.stat().st_mtime:
        return dict(csv=fs[-1], hit=True, seconds=None)
    shutil.rmtree(e, ignore_errors=True)
    env = dict(NAVSIM_THREADS=str(max(1, ncpus(cpus))))
    if board == "navtest_sub":
        env["TOKENS_FILE"] = str(OPLB / b["data"] / "tokens.txt")
    t0 = time.time()
    sh(taskset(cpus) + ["experiments/zeroshot_openloop/archive/navsim_zs_score.sh", "score", b["ver"], b["split"], e.name[len(f"{b['ver']}_{b['split']}_"):], preds],
       logd / f"official_{board}.log", env=env)
    fs = sorted(glob.glob(str(e / "*/*.csv")))
    assert fs, f"no CSV under {e}"
    return dict(csv=fs[-1], hit=False, seconds=round(time.time() - t0, 1))


def official_cached(cand: dict, board: str):
    """The candidate's official CSV if one exists and is newer than its pose file, else None (no scoring)."""
    preds = OPLB / BOARDS[board]["data"] / "preds" / f"{stems(cand)[1]}.npz"
    fs = sorted(glob.glob(str(EVAL / eval_name(cand, board) / "*/*.csv")))
    return dict(csv=fs[-1], hit=True, seconds=None) if fs and Path(fs[-1]).stat().st_mtime > preds.stat().st_mtime else None


def v1_tokens(csv: str) -> pd.DataFrame:
    x = pd.read_csv(csv)
    return x[x.valid.astype(bool) & (x.token != "average")].set_index("token")


def v2_summary(csv: str) -> dict:
    x = pd.read_csv(csv).set_index("token")
    return {k: 100 * float(x.loc[f"extended_pdm_score_{s}", "score"]) if f"extended_pdm_score_{s}" in x.index and pd.notna(x.loc[f"extended_pdm_score_{s}", "score"]) else None
            for k, s in (("combined", "combined"), ("stage1", "stage_one"), ("stage2", "stage_two"))}


# ---------------------------------------------------------------- 4. navhard per-token harness

def harness(cand: dict, cpus: str, force: bool, work: Path, logd: Path) -> dict:
    preds = OPLB / BOARDS["navhard"]["data"] / "preds" / f"{stems(cand)[1]}.npz"
    s = work / "harness_summary.json"
    if s.is_file() and not force and s.stat().st_mtime > preds.stat().st_mtime:
        return dict(json.loads(s.read_text()), hit=True)
    t0 = time.time()
    sh(taskset(cpus) + [ENVS / "navsim2/bin/python", G.GUARD / "scripts/nav_harness.py", "--poses", preds, "--out", work, "--procs", max(1, ncpus(cpus))],
       logd / "harness_navhard.log")
    r = json.loads(s.read_text())
    r["seconds"] = round(time.time() - t0, 1)
    s.write_text(json.dumps(r, indent=1))
    return dict(r, hit=False)


# ---------------------------------------------------------------- readouts

def ci(r: dict):
    return [float(r["lo"]), float(r["hi"])]


def navtest_logs(tokens) -> np.ndarray:
    from jevdrive import navsim_zs as Z
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    return np.array([lg[t] for t in tokens])
