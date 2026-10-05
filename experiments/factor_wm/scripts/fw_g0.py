"""factor_wm gate G0 rollouts (op-train env, one card + CPU warp workers; plans/2026-10-05-stage1-prereg.md section 3.1). No training.

  g0a   engine fidelity on g0a: the ego follows the log (replay); frames after t0 are synthesised by an engine from the t0 frame (src anchor,
        decision 123's construction) or from the previous logged frame (src step1); reference = the real logged frames (src log)
  g0b   shipped Cinque closed loop for 8 s on g0b: kinds closed (SPEC lateral + longitudinal) on estar and plane, closedlat (logged speed) on
        estar; arms free, kick +-2 deg, swerve +-0.5 m (moving clips where feasible); replay identity on every 10th clip
Output: $DATA_DIR/runs/factor_wm/roll/<cmd>-<i>of<n>.npz, one row per rollout.

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_g0.py g0b --shard 0/3 [--limit 4]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402
import dg_roll as DR  # noqa: E402

_CL = {}


def _clips(name):
    if name not in _CL:
        _CL[name] = C.Clips(name)
    return _CL[name]


def rel(pose, base):
    d = C.DG.rot(-base[2]) @ (np.asarray(pose[:2]) - base[:2])
    return np.array([d[0], d[1], np.arctan2(np.sin(pose[2] - base[2]), np.cos(pose[2] - base[2]))])


def frame_job(job):
    """(set, clip, step j, engine, src, ego pose in the t0 frame) -> packed frame for state j."""
    name, c, j, engine, src, pose = job
    S = _clips(name)
    P = S.t["pose"][c]
    i = {"log": j, "anchor": 0, "step1": j - 1}[src]
    off = rel(pose, P[i])
    if src == "log":
        off = np.array([np.clip(off[0], -C.DX_CAP, C.DX_CAP), np.clip(off[1], -C.FAIL_DY, C.FAIL_DY), np.clip(off[2], -C.FAIL_PSI, C.FAIL_PSI)])
    D = None if engine == "plane" or S.depth is None else S.depth[c, C.T0 + i]
    return C.warp(engine, S.imgs[c, C.T0 + i], D, S.t["cam"][c], off)


class Runner(DR.Runner):
    def heads(self, T9, tc, v):
        torch = self.torch
        B = T9.shape[0]
        with torch.no_grad():
            o = self.m(T9, torch.ones(B, 9, dtype=torch.bool, device=self.dev), torch.from_numpy(tc).to(self.dev).to(self.dtype))["outputs"]
        o = o.float().cpu().numpy()
        d = self.H(o, v)
        d["acc"] = o[:, self.H.a0 + 1]
        return d

    def run(self, S, specs, batch=48, kl=None):
        torch = self.torch
        kl = kl or S.kl
        res = [None] * len(specs)
        order = sorted(range(len(specs)), key=lambda i: specs[i]["c"])
        for b0 in range(0, len(order), batch):
            ids = order[b0: b0 + batch]
            sp = [specs[i] for i in ids]
            cs = [s["c"] for s in sp]
            uc = sorted(set(cs))
            Th = self.history(S, uc)
            T9 = Th[[uc.index(c) for c in cs]].clone()
            tc = S.t["tc"][cs].astype(np.float32)
            egos = [C.Ego(S.t["pose"][c][: kl + 1], S.t["v"][c], s["kind"], s.get("exo"), float(S.t["k0"][c]), float(S.t["a0"][c]), str(S.t["cat"][c]))
                    for c, s in zip(cs, sp)]
            prev = np.asarray(S.imgs[cs, C.T0])
            rec = [self.heads(T9, tc, np.array([e.v for e in egos]))]
            for j in range(1, kl + 1):
                for b, e in enumerate(egos):
                    e.advance(j, rec[-1]["kappa"][b], rec[-1]["acc"][b])
                jobs = []
                for c, s, e in zip(cs, sp, egos):
                    if e.event is not None:
                        pose = C.compose(S.t["pose"][c][j], np.array([e.clamp_off()]))[0]
                    else:
                        pose = np.r_[e.p, e.th]
                    jobs.append((S.name, c, j, s["engine"], s["src"], pose))
                cur = np.stack(list(self.pool.map(frame_job, jobs, chunksize=1)))
                T9 = torch.cat([T9[:, 1:], self.trunk(prev, cur)[:, None]], 1)
                prev = cur
                rec.append(self.heads(T9, tc, np.array([e.v for e in egos])))
            for b, i in enumerate(ids):
                e = egos[b]
                res[i] = dict({k: np.array([r[k][b] for r in rec]) for k in ("phi1", "kappa", "acc", "y1")}, off=np.array(e.trace),
                              vs=np.array(e.vs), event=e.event or "", t_event=np.nan if e.t_event is None else e.t_event)
            print(f"  {min(b0 + batch, len(order))}/{len(order)} rollouts", flush=True)
        return res


def save(path, specs, res, S):
    out = {k: np.stack([r[k] for r in res]) for k in res[0]}
    meta = {k: np.array([s.get(k) for s in specs]) for k in ("c", "arm", "kind", "engine", "src")}
    out |= meta | dict(clip_id=S.t["id"][meta["c"]], cat=S.t["cat"][meta["c"]], seq=S.t["seq"][meta["c"]])
    np.savez(path.with_suffix(".tmp.npz"), **out)
    path.with_suffix(".tmp.npz").replace(path)


def specs_g0a(S):
    sp = [dict(c=c, kind="replay", engine="plane", src="log", arm="ref") for c in range(S.n)]
    for c in range(S.n):
        for eng in C.ENGINES:
            for src in ("anchor", "step1"):
                sp.append(dict(c=c, kind="replay", engine=eng, src=src, arm=f"{eng}-{src}"))
    return sp


def specs_g0b(S):
    sp = []
    for c in range(S.n):
        arms = [("free", None)] + [(f"kick{d:+g}", C.kick(S.kl, d)) for d in (2.0, -2.0)]
        if S.t["cat"][c] != "launch":
            for o in (0.5, -0.5):
                e = C.swerve(S.kl, o, S.t["v"][c])
                if e is not None:
                    arms.append((f"swerve{o:+g}", e))
        for kind, eng in (("closed", "estar"), ("closed", "plane"), ("closedlat", "estar")):
            sp += [dict(c=c, kind=kind, engine=eng, src="log", arm=a, exo=x) for a, x in arms]
        if c % 10 == 0:
            sp.append(dict(c=c, kind="replay", engine="estar", src="log", arm="replay"))
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["g0a", "g0b"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    from jevdrive.common import n_cpus
    workers = a.workers or int(os.environ.get("CL_WORKERS") or 0) or max(1, len(os.sched_getaffinity(0)) - 2 if hasattr(os, "sched_getaffinity") else n_cpus() - 2)
    R = Runner("shipped", a.dev, workers)
    t0 = time.time()
    S = C.Clips(a.cmd)
    sp = specs_g0a(S) if a.cmd == "g0a" else specs_g0b(S)
    if a.limit:
        sp = [s for s in sp if s["c"] < a.limit]
    i, n = map(int, a.shard.split("/"))
    cs = set(sorted({s["c"] for s in sp})[i::n])
    sp = [s for s in sp if s["c"] in cs]
    tag = f"{a.cmd}{'-smoke' if a.limit else ''}-{i}of{n}"
    out = C.root("roll") / f"{tag}.npz"
    if out.exists():
        print("exists", out)
        DR.bye()
    print(f"{tag}: {len(sp)} rollouts, {len(cs)} clips, {workers} warp workers", flush=True)
    save(out, sp, R.run(S, sp, a.batch), S)
    print(f"{tag}: done in {time.time() - t0:.0f} s -> {out}", flush=True)
    DR.bye()


if __name__ == "__main__":
    main()
