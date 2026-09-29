#!/usr/bin/env python
"""openpilot on the NAVSIM leaderboard splits, infrastructure for the navigation arms (research/openpilot-openloop-
integration.md sections 2.5, 7, 9): compact context-rate frame caches and a rollout with a pluggable per-step desire
schedule that keeps the plan's full Gaussian (mu and std) and every output head.
Run root $DATA_DIR/runs/op_lb/<data>/, data = lb_navtest (12 146) | lb_navhard (navhard_two_stage, 5 912, both stages)
| lb_navtrain (seed-0 subset, 1000 tokens per driving command).

  prep    (envs/openpilot) meta.json + tokens.txt; navtrain: the per-command subset and its 4 keyframes rendered ->
          keys.npy. Test splits read their keyframes in place from the navsim_zs frames cache (no copy).
  synth   the 6 context-rate frames t0 - 0.2 k (k = 1, 2, 3, 4, 6, 7; the only non-key frames a Cinque / small output
          at t0 or a Lebowski context step reads) -> <method>.npy (N, 6, 2, 6, 128, 256) uint8. gimm (envs/vfi): one
          worker per GPU, 32-token chunks claimed from a shared queue (<method>.chunks/), batch 8 as the op_interp full
          run; pauses while the card is above --cap-gb. warp (any env with scipy + cv2): CPU pool, same queue.
  run     (envs/openpilot) one model x frames x desire schedules (SCHEDULES), zero state, plan at t0 ->
          plans/<frames>@<model>[.<schedule>].npz: the plan exactly as scripts/op_interp.py run (plan_pos / plan_vel /
          plan_yaw / lead_prob) plus plan_mu / plan_std (33, 15), all non-hidden output heads (heads + info.heads_slices)
          and the desire per step. The plan heads of small / Cinque / Lebowski are one Gaussian (no hypotheses).
          Lebowski gets the 3.3 s ego-motion warp pre-roll by default (section 7), computed on the fly.
  compare two plan files: max / mean plan difference (the equivalence check against op_interp's stored plans).

Export, scoring and report are op_interp's, pointed at this root: OPI_ROOT=op_lb scripts/op_interp.py nav-export /
nav-report and OPI_ROOT=op_lb scripts/op_interp_score.sh (scripts/op_lb_lane.sh chains them).
Non-`none` desire schedules are refused on lb_navtest / lb_navhard without --prereg <id>: those arms wait for the
pre-registration.
"""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

SPLITS = {"lb_navtest": "navtest", "lb_navhard": "navhard_two_stage", "lb_navtrain": "navtrain"}
TEST = {"lb_navtest", "lb_navhard"}
BACKENDS = {"small": "trt-fp32", "cinque": "trt", "lebowski": "trt"}      # the exams' backends
PREROLL = {"small": 0.0, "cinque": 0.0, "lebowski": 3.3}                  # section 7: only Lebowski gets the warm-up
ACTION_T = (0.275, 0.525)
LHT_MAPS = {"sg-one-north"}
_DENSE = I.grid(-1.5)
SYN_T = _DENSE[np.isclose(_DENSE / 0.2, np.round(_DENSE / 0.2)) & ~np.isclose(_DENSE[:, None], I.T_KEY).any(1)]
FRAME = (2, 6, 128, 256)


def root(*p) -> Path:
    d = data_dir() / "runs" / "op_lb" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta(data):
    return json.loads((root(data) / "meta.json").read_text())


class Keys:
    """The four keyframes per token: keys.npy of the run dir, else the split's navsim_zs frames cache by index."""

    def __init__(self, data):
        mt = meta(data)
        own = root(data) / "keys.npy"
        if own.exists():
            self.src, self.ix = np.load(own, mmap_mode="r"), np.arange(len(mt["names"]))
        else:
            from jevdrive import navsim_zs as Z
            self.src = np.load(Z.root("openpilot", mt["split"]) / "frames.npy", mmap_mode="r")
            self.ix = np.asarray(mt["index"])

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, rows):
        if isinstance(rows, slice):
            rows = np.arange(len(self))[rows]
        ix = self.ix[rows]
        if np.ndim(ix) == 0:
            return np.asarray(self.src[ix])
        if len(ix) and (np.diff(ix) == 1).all():                   # contiguous: one sequential read
            return np.asarray(self.src[ix[0]:ix[-1] + 1])
        return np.stack([self.src[k] for k in ix])


# ---------------------------------------------------------------- prep

def cmd_prep(a):
    from jevdrive import navsim_zs as Z
    split = SPLITS[a.data]
    idx = Z.load_index(split, slim=True)
    if split == "navtrain":
        keep = Z.nonav_subset(idx, a.per_cmd, a.seed)       # seed-0 draw, a.per_cmd per command (all when fewer)
        sel = np.array([k for k, e in enumerate(idx) if e["token"] in keep])
    else:
        sel = np.arange(len(idx))
        fj = Z.root("openpilot", split) / "frames.json"
        assert json.loads(fj.read_text())["tokens"] == [e["token"] for e in idx], "frames cache out of sync with the index"
    sub = [idx[k] for k in sel]
    toks = [e["token"] for e in sub]
    R = root(a.data)
    if split == "navtrain" and not (R / "keys.npy").exists():
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        import navsim_zs_openpilot as N
        tmp = R / "keys.tmp.npy"
        out = np.lib.format.open_memmap(tmp, "w+", np.uint8, (len(sub), 4) + FRAME)
        with ProcessPoolExecutor(a.workers) as ex:
            for i, fr in enumerate(tqdm(ex.map(N.render_token, sub, chunksize=8), total=len(sub), desc="render keys")):
                out[i] = fr
        out.flush()
        del out
        tmp.replace(R / "keys.npy")
    (R / "tokens.txt").write_text("\n".join(toks) + "\n")
    cmds = [[int(np.argmax(c)) for c in e["cmd"]] for e in sub]
    (R / "meta.json").write_text(json.dumps({
        "split": split, "names": toks, "index": sel.tolist(),
        "cam": [np.asarray(e["cams"][-1]["CAM_F0"]["t"], float).tolist() for e in sub],
        "pose": [np.asarray(e["pose"], float).tolist() for e in sub], "vel": [np.asarray(e["vel"], float).tolist() for e in sub],
        "speed": [float(np.linalg.norm(e["vel"][-1])) for e in sub], "lht": [e["map"] in LHT_MAPS for e in sub],
        "cmd": [c[-1] for c in cmds], "cmds": cmds, "syn_t": SYN_T.tolist()}))
    print(f"{a.data}: {len(toks)} tokens, commands {np.bincount([c[-1] for c in cmds], minlength=4).tolist()}")


# ---------------------------------------------------------------- synthesis

def _alloc(path: Path, shape):
    with open(path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not path.exists():
            np.lib.format.open_memmap(path, "w+", np.uint8, shape).flush()
    return np.load(path, mmap_mode="r+")


def _claim(q: Path, c: int):
    """Atomic claim of chunk c; a claim whose owner process is gone (crash, kill) is taken over."""
    done, cl = q / f"{c:05d}.done", q / f"{c:05d}.claim"
    if done.exists():
        return False
    try:
        fd = os.open(cl, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        host, pid = cl.read_text().split() if cl.stat().st_size else ("", "0")
        if host != os.uname().nodename or Path(f"/proc/{pid}").exists():
            return False
        cl.unlink(missing_ok=True)
        return _claim(q, c)
    os.write(fd, f"{os.uname().nodename} {os.getpid()}".encode())
    os.close(fd)
    return True


def _vram_gb(gpu: int) -> tuple[float, float]:
    """(total used on the card, used by this process incl. its CUDA context), GB, from nvidia-smi."""
    q = lambda *a: subprocess.check_output(["nvidia-smi", "-i", str(gpu), *a, "--format=csv,noheader,nounits"]).decode()  # noqa: E731
    mine = sum(int(r.split(",")[1]) for r in q("--query-compute-apps=pid,used_memory").splitlines()
               if r.strip() and int(r.split(",")[0]) == os.getpid())
    return int(q("--query-gpu=memory.used")) / 1024, mine / 1024


def _gpu_gate(gpu, cap, need, log):
    """Wait until the card has room: (used by other processes) + need <= cap. Our cache is released while waiting."""
    import torch
    t0 = None
    while True:
        used, mine = _vram_gb(gpu)
        if used - mine + need <= cap:
            if t0 is not None:
                log.info(f"GPU {gpu}: resumed after {time.time() - t0:.0f} s")
                log.event("resume", gpu=gpu, paused_s=time.time() - t0)
            return
        if t0 is None:
            t0 = time.time()
            torch.cuda.empty_cache()
            log.info(f"GPU {gpu}: others use {used - mine:.1f} GB, + {need:.1f} > cap {cap}; paused")
            log.event("pause", gpu=gpu, other_gb=used - mine)
        time.sleep(60)


def _warp_job(args):
    keys, tr_args, cam = args
    return I.synth_cpu(keys, "warp", SYN_T, I.track_navsim(*tr_args), cam)


def cmd_synth(a):
    from jevdrive.runlog import RunLog
    log = RunLog("op_lb", "logs", f"synth_{a.method}" + (f"_gpu{a.gpu}" if a.method == "gimm" else ""))
    log.info(f"args {vars(a)}")
    if a.method == "gimm":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        import torch
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
        model = I.GIMM(data_dir() / "third_party" / "vfi")
        need = a.vram_gb                                  # until the first chunk measured the real peak
    else:
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(a.workers)
    for data in a.data:
        mt = meta(data)
        n = len(mt["names"])
        keys = Keys(data)
        out = _alloc(root(data) / f"{a.method}.npy", (n, len(SYN_T)) + FRAME)
        q = root(data, f"{a.method}.chunks")
        chunks = list(range(0, n, a.chunk))[: a.limit_chunks or None]
        t0, done = time.time(), 0
        for c in chunks + chunks:           # second pass: chunks whose claiming worker died meanwhile
            if not _claim(q, c // a.chunk):
                continue
            tc = time.time()
            k = keys[c:c + a.chunk]
            if a.method == "gimm":
                _gpu_gate(a.gpu, a.cap_gb, need, log)
                syn = I.synth_vfi(k, model, SYN_T, batch=a.batch)
                torch.cuda.synchronize()
                need = min(a.vram_gb, torch.cuda.max_memory_reserved() / 2 ** 30 + 0.7)   # + CUDA context
            else:
                rows = range(c, c + len(k))
                jobs = ((k[i - c], (mt["pose"][i], mt["vel"][i]), mt["cam"][i]) for i in rows)
                syn = np.stack(list(ex.map(_warp_job, jobs, chunksize=4)))
            out[c:c + len(k)] = syn
            out.flush()
            (q / f"{c // a.chunk:05d}.done").touch()
            done += len(k)
            dt = time.time() - tc
            log.info(f"{data} chunk {c // a.chunk}: {len(k)} tokens, {dt / len(k):.3f} s/token"
                     + (f", peak {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB" if a.method == "gimm" else ""))
            log.event("chunk", data=data, chunk=c // a.chunk, n=len(k), s_per_token=dt / len(k))
        left = sum(not (q / f"{c // a.chunk:05d}.done").exists() for c in chunks)
        log.info(f"{data}: {done} tokens here in {time.time() - t0:.0f} s; chunks not done yet (any worker): {left}")
    log.event("end")
    log.close()


# ---------------------------------------------------------------- desire schedules
# name -> f(tok, ts) -> (len(ts),) desire index per step (0 none, 1 turnLeft, 2 turnRight, 3 laneChangeLeft,
# 4 laneChangeRight; log.Desire) or a (len(ts), 8) desire array. tok: the token's meta row (cmds = argmax NAVSIM
# driving_command [left, straight, right, unknown] of the 4 history frames, cmd = cmds[-1], speed, vel, pose, lht,
# token); ts: step times relative to t0 (s, ascending, last = 0). The desire is a held state; OPModel turns it into
# modeld's rising-edge pulse, so a desire held from T on pulses once, at the first step >= T. Register new ones here.

DESIRES = ("none", "turnLeft", "turnRight", "laneChangeLeft", "laneChangeRight", "keepLeft", "keepRight")
T_ON = {"-1.5": -1.5, "-1.0": -1.0, "-0.5": -0.5, "0": 0.0}      # names as the pre-registration (todos/2026-09-29-op-leaderboard.md)
T20 = np.round(np.arange(-30, 1) * 0.05, 3)                  # the 20 Hz steps from -1.5 s to t0


def _held(kind, T):
    """`turn@T` / `lc@T`: when the command at t0 is left / right, the turn (lane-change) desire held on from step time T
    (off before, pre-roll included), else none."""
    base = {"turn": 1, "lc": 3}[kind]

    def f(tok, ts):
        c = tok["cmd"]
        return np.where(ts >= T - 1e-9, base + (c == 2), 0) if c in (0, 2) else np.zeros(len(ts), int)
    return f


def sch_turn_onset(tok, ts, thr_deg=5.0):
    """`turn@onset`: turn desire from the first 20 Hz step >= -1.5 s at which the ego yaw rate of the history track
    (the warp's Hermite / spline track) exceeds 5 deg/s in the t0 command's direction (left = counter-clockwise);
    T = 0.0 when it never does. None for straight / unknown."""
    c = tok["cmd"]
    if c not in (0, 2):
        return np.zeros(len(ts), int)
    yr = I.track_navsim(tok["pose"], tok["vel"])._yaw.derivative()(T20) * (1 if c == 0 else -1)
    hit = np.flatnonzero(yr > np.radians(thr_deg))
    T = T20[hit[0]] if len(hit) else 0.0
    return np.where(ts >= T - 1e-9, 1 + (c == 2), 0)


SCHEDULES = {"none": lambda tok, ts: np.zeros(len(ts), int),
             **{f"turn@{k}": _held("turn", T) for k, T in T_ON.items()}, **{f"lc@{k}": _held("lc", T) for k, T in T_ON.items()},
             "turn@onset": sch_turn_onset}


def sched_tag(name):
    """File-name form of a schedule name (no '@' / '-'): turn@-1.5 -> turn_m1.5."""
    return name.replace("@", "_").replace("-", "m")


def desire_arr(v, n):
    v = np.asarray(v)
    if v.ndim == 2:
        return v.astype(np.float32)
    d = np.zeros((n, 8), np.float32)
    d[np.arange(n), v.astype(int)] = 1
    d[:, 0] = 0
    return d


# ---------------------------------------------------------------- rollout

def _steps(pre, cr):
    """Dense 0.1 s history grid from -1.5 - pre (as op_interp synth), the step -> dense index schedule
    (jevdrive.op_interp.schedule) and the step times; every step's frame source: a key, a synthesized context-rate
    frame (the latest at or before the slot: op_interp's --grid 0.2 hold) or a pre-roll warp at the slot time."""
    dense = I.grid(-1.5 - pre)
    sched = I.schedule(dense, cr)
    dt = 0.2 if cr else 0.05
    ts = np.round(np.arange(-len(sched) + 1, 1) * dt, 3)
    sparse = np.r_[I.T_KEY, SYN_T]
    src = []
    for t in dense[sched]:
        if t < -1.5 - 1e-9:
            src.append(("p", float(t)))
        else:
            j = int(np.flatnonzero(sparse <= t + 1e-9)[np.argmax(sparse[sparse <= t + 1e-9])])
            src.append(("k", j) if j < 4 else ("s", j - 4))
    return ts, src


def plan_stem(frames, model, schedule, pre):
    return f"{frames}@{model}" + ("" if schedule == "none" else f".{sched_tag(schedule)}") + \
        ("" if pre == PREROLL[model] else f"_pre{pre:g}")


def cmd_run(a):
    bad = [s for s in a.schedule if s != "none"]
    if a.data in TEST and bad and not a.prereg:
        sys.exit(f"desire schedules {bad} on {a.data} wait for the pre-registration (pass --prereg <id>)")
    pre = PREROLL[a.model] if a.preroll is None else a.preroll
    stems = [plan_stem(a.frames, a.model, s, pre) for s in a.schedule]
    pdir = root(a.data, "plans_pilot" if a.limit else "plans")
    sfx = f".first{a.limit}" if a.limit else ""
    if a.procs > 1 and not a.shard:
        argv = [sys.executable, __file__] + sys.argv[1:]
        ps = [subprocess.Popen(argv + ["--shard", f"{k}/{a.procs}"]) for k in range(a.procs)]
        for k, p in enumerate(ps):          # a shard that died (rare TensorRT start-up failure) is retried once, alone
            if p.wait() != 0 or not all((pdir / f"{s}{sfx}.part{k}of{a.procs}.npz").exists() for s in stems):
                print(f"shard {k}/{a.procs} failed (rc {p.returncode}); retrying")
                assert subprocess.call(argv + ["--shard", f"{k}/{a.procs}"]) == 0, f"shard {k} failed twice"
        for stem in stems:
            parts = [pdir / f"{stem}{sfx}.part{k}of{a.procs}.npz" for k in range(a.procs)]
            zs = [dict(np.load(f)) for f in parts]
            n = sum(len(z["names"]) for z in zs)
            out = {k: np.concatenate([z[k] for z in zs]) if zs[0][k].ndim and len(zs[0][k]) == len(zs[0]["names"]) else zs[0][k]
                   for k in zs[0]}
            out["ms_per_scene"] = np.mean([z["ms_per_scene"] for z in zs])
            np.savez(pdir / f"{stem}{sfx}.npz", **out)
            for f in parts:
                f.unlink()
            print(f"{stem}{sfx}: {n} scenes, {out['ms_per_scene']:.1f} ms/scene/schedule")
        return
    from jevdrive.openpilot.model import OPModel, decode
    mt = meta(a.data)
    keys = Keys(a.data)
    syn = np.load(root(a.data) / f"{a.frames}.npy", mmap_mode="r")
    cr = a.model == "lebowski"
    backend = a.backend or BACKENDS[a.model]
    m = OPModel(a.model, backend, cache=data_dir() / "runs" / "op_interp" / "trt_cache" / f"{a.model}-{backend}", context_rate=cr)
    ts, src = _steps(pre, cr)
    pre_t = sorted({j for s, j in src if s == "p"})
    n = min(len(mt["names"]), a.limit or 10 ** 9)
    k, K = map(int, (a.shard or "0/1").split("/"))
    rows = np.array_split(np.arange(n), K)[k]
    heads = sorted(((q, s) for q, s in m.slices.items() if q not in ("hidden_state", "pad")), key=lambda x: x[1].start)
    keep = np.concatenate([np.arange(s.start, s.stop) for _, s in heads])
    hs = dict(zip([q for q, _ in heads], np.cumsum([0] + [s.stop - s.start for _, s in heads]).tolist()))
    W = 33 * 15
    assert m.slices["plan"].stop - m.slices["plan"].start == 2 * W, "plan head is not a single Gaussian"
    R = len(rows)
    P = {s: {"plan_pos": np.zeros((R, 33, 3), np.float32), "plan_vel": np.zeros((R, 33, 3), np.float32),
             "plan_yaw": np.zeros((R, 33), np.float32), "lead_prob": np.zeros((R, 3), np.float32),
             "plan_mu": np.zeros((R, 33, 15), np.float32), "plan_std": np.zeros((R, 33, 15), np.float32),
             "heads": np.zeros((R, len(keep)), np.float32), "desire_steps": np.zeros((R, len(ts)), np.int8)}
         for s in a.schedule}
    t0, tg = time.time(), 0.0
    for r, i in enumerate(rows):
        kf = keys[int(i)]
        sf = np.asarray(syn[i])
        pw = {}
        if pre_t:
            fr = I.synth_cpu(kf, "warp", np.array(pre_t), I.track_navsim(mt["pose"][i], mt["vel"][i]), mt["cam"][i])
            pw = dict(zip(pre_t, fr))
        frames = [np.ascontiguousarray(kf[j] if s == "k" else sf[j] if s == "s" else pw[j]) for s, j in src]
        tok = {q: mt[q][i] for q in ("cmds", "cmd", "speed", "vel", "pose", "lht")} | {"token": mt["names"][i]}
        tc = (0, 1) if mt["lht"][i] else (1, 0)
        for s in a.schedule:
            des = desire_arr(SCHEDULES[s](tok, ts), len(ts))
            t = time.perf_counter()
            m.reset()
            for f, d in zip(frames, des):
                raw = m.step(f, desire=d, traffic=tc, action_t=ACTION_T)
            tg += time.perf_counter() - t
            d = decode(raw, m.slices, float(mt["speed"][i]), ACTION_T)
            p = P[s]
            for q in ("plan_pos", "plan_vel", "plan_yaw", "lead_prob"):
                p[q][r] = d[q]
            v = raw[m.slices["plan"]]
            p["plan_mu"][r], p["plan_std"][r] = v[:W].reshape(33, 15), np.exp(np.minimum(v[W:], 11)).reshape(33, 15)
            p["heads"][r] = raw[keep]
            p["desire_steps"][r] = des.argmax(1)
    ms = 1e3 * tg / max(1, R * len(a.schedule))
    for s, stem in zip(a.schedule, stems):
        info = json.dumps({"model": a.model, "backend": backend, "frames": a.frames, "schedule": s, "preroll": pre,
                           "prereg": a.prereg, "step_times": ts.tolist(), "heads_slices": hs, "desires": DESIRES,
                           "plan_std": "exp of the MDN log-std, as openpilot's parse_mdn"})
        out = pdir / (f"{stem}{sfx}" + (f".part{k}of{K}" if a.shard else "") + ".npz")
        np.savez(out, names=np.array(mt["names"])[rows], steps=len(ts), ms_per_scene=ms, info=info, **P[s])
    print(f"shard {a.shard or '0/1'}: {R} scenes x {len(a.schedule)} schedules, {len(ts)} steps, {ms:.1f} ms/scene, "
          f"{time.time() - t0:.0f} s wall")
    if a.shard:                        # ORT / TensorRT teardown can hang for minutes after the outputs are written
        sys.stdout.flush()
        os._exit(0)

# ---------------------------------------------------------------- equivalence check

def cmd_compare(a):
    za, zb = np.load(a.a), np.load(a.b)
    na, nb = za["names"].tolist(), zb["names"].tolist()
    pos = {t: k for k, t in enumerate(nb)}
    common = [t for t in na if t in pos]
    ia, ib = np.array([na.index(t) for t in common]) if len(common) < len(na) else np.arange(len(na)), np.array([pos[t] for t in common])
    e = np.linalg.norm(za["plan_pos"][ia] - zb["plan_pos"][ib], axis=-1)       # (n, 33) m
    per = e.max(1)
    res = {"n": len(common), "max_m": float(e.max()), "mean_m": float(e.mean()), "mean_tokenmax_m": float(per.mean()),
           "p99_tokenmax_m": float(np.percentile(per, 99)), "n_over_0.1m": int((per > 0.1).sum()), "n_over_1m": int((per > 1).sum()),
           "identical": int((per == 0).sum()), "yaw_max_rad": float(np.abs(za["plan_yaw"][ia] - zb["plan_yaw"][ib]).max())}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--data", choices=list(SPLITS), required=True)
    p.add_argument("--per-cmd", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p = sp.add_parser("synth")
    p.add_argument("--data", nargs="+", choices=list(SPLITS), required=True, help="processed in order")
    p.add_argument("--method", choices=("gimm", "warp"), required=True)
    p.add_argument("--gpu", type=int, default=0, help="physical GPU index (gimm)")
    p.add_argument("--vram-gb", type=float, default=12.0, help="hard cap of this process's CUDA memory")
    p.add_argument("--cap-gb", type=float, default=78.0, help="pause while the card's other users + vram-gb exceed this")
    p.add_argument("--chunk", type=int, default=32, help="tokens per claimed chunk; a multiple of 4 keeps GIMM's batch-8 "
                   "composition (4 tokens x 2 views per forward) identical to op_interp's 128-token chunks")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=16, help="warp: CPU processes")
    p.add_argument("--limit-chunks", type=int, default=0, help="staging: only the first N chunks per split")
    p = sp.add_parser("run")
    p.add_argument("--data", choices=list(SPLITS), required=True)
    p.add_argument("--frames", required=True, help="gimm | warp")
    p.add_argument("--model", choices=list(BACKENDS), required=True)
    p.add_argument("--schedule", nargs="+", choices=list(SCHEDULES), default=["none"],
                   help="one plan file per schedule; frames are loaded once per token")
    p.add_argument("--preroll", type=float, default=None, help="s of warp pre-roll (default: 3.3 Lebowski, 0 others)")
    p.add_argument("--backend", default="")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--shard", default="")
    p.add_argument("--limit", type=int, default=0, help="staging: first N tokens -> plans_pilot/")
    p.add_argument("--prereg", default="", help="pre-registration id: required for non-none schedules on test splits")
    p = sp.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", default="")
    a = ap.parse_args()
    {"prep": cmd_prep, "synth": cmd_synth, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)
