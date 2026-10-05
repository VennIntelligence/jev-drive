"""op_dagger rollout engine runs (op-train env; GPU when leased, CPU fp32 otherwise). See dg_common for the engine.

  check    sf + heldout: (a) replay arm == the batch pipeline on the logged frames (engine identity); (b) port vs decision 123's ONNX phi1 on the
           same sf frames; (c) open yaw +-1 / +-2 deg arms with the anchor-frame source (decision 123's R) and the logged-frame source -> step gains
  collect  DAgger collection on a clip set with a policy: arms free, kick +-, swerve +- (kick of another size where a swerve is infeasible)
  eval     held-out readout arms: open yaw +-2 (gain), closed kick +-2 deg, closed swerve +-0.5 m (moving clips), free
Output: $DATA_DIR/runs/op_dagger/roll/<model>/<set>-<cmd>.npz (one row per rollout: clip, arm, exo, readouts per step, offsets per step, validity).

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/op_dagger/scripts/dg_roll.py collect --model shipped --set train
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_common as C  # noqa: E402

_CL = {}


def _clips(name):
    if name not in _CL:
        _CL[name] = C.Clips(name)
    return _CL[name]


def warp_job(job):
    """(set, clip, frame index, offset or t0-frame pose, source) -> packed frame."""
    name, c, fi, off = job
    S = _clips(name)
    return C.warp(S.imgs[c, fi], S.t["cam"][c], off)


class Runner:
    def __init__(self, model: str, dev: str, workers: int):
        import torch
        self.torch = torch
        self.dev = torch.device(dev)
        self.dtype = torch.float16 if self.dev.type == "cuda" else torch.float32
        self.m = self._load(model)
        self.H = C.Heads(self.m.net)
        self.pool = ProcessPoolExecutor(workers)

    def _load(self, tag):
        torch = self.torch
        from experiments.op_adapt_l.lib import op_adapt_l as L
        import rft
        if tag == "shipped":
            return L.LModel(None, dtype=self.dtype).to(self.dev).eval()
        p = Path(tag) if Path(tag).exists() else C.root("runs", tag) / "ckpt-final.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = rft.RModel(L.LCfg(tag, intent="none"), None, dtype=self.dtype).to(self.dev).eval()
        m.load_state(ck["model"])
        return m

    def trunk(self, prev: np.ndarray, cur: np.ndarray):
        from jevdrive import op_adapt as A
        torch = self.torch
        p, c = (torch.from_numpy(np.ascontiguousarray(x)).to(self.dev) for x in (prev, cur))
        with torch.no_grad():
            return self.m.net.run_batched(A.vision_feeds(p, c), [A.TRUNK_OUT])[A.TRUNK_OUT][:, 0]

    def heads(self, T9, tc, v):
        torch = self.torch
        B = T9.shape[0]
        with torch.no_grad():
            o = self.m(T9, torch.ones(B, 9, dtype=torch.bool, device=self.dev), torch.from_numpy(tc).to(self.dev).to(self.dtype))["outputs"]
        return self.H(o.float().cpu().numpy(), v)

    def history(self, S, cs):
        """(n, 9, 1024, 8, 16) trunks of the logged history pairs of clips cs."""
        cs = list(cs)
        f = np.asarray(S.imgs[cs][:, : C.NH])                                   # (n, 10, ...)
        prev, cur = f[:, :-1].reshape(-1, *f.shape[2:]), f[:, 1:].reshape(-1, *f.shape[2:])
        out = [self.trunk(prev[i:i + 64], cur[i:i + 64]) for i in range(0, len(cur), 64)]
        return self.torch.cat(out).reshape(len(cs), 9, *out[0].shape[1:])

    def rollouts(self, S, specs, src="log", batch=48):
        """specs: dicts (c, kind, exo, arm) -> list of result dicts (one per spec, same order)."""
        torch = self.torch
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
            egos = [C.Ego(S.t["pose"][c], S.t["v"][c], s["kind"], s.get("exo"), float(S.t["k0"][c])) for c, s in zip(cs, sp)]
            prev = np.asarray(S.imgs[cs, C.T0])
            rec = [self.heads(T9, tc, S.t["v"][cs, C.T0])]
            for j in range(1, C.K + 1):
                offs = [e.advance(j, rec[-1]["kappa"][b]) for b, e in enumerate(egos)]
                if src == "log":
                    jobs = [(S.name, c, C.T0 + j, o) for c, o in zip(cs, offs)]
                else:                                                              # decision 123's R: the t0 frame re-projected to the pose
                    jobs = [(S.name, c, C.T0, e.pose_t0()) for c, e in zip(cs, egos)]
                cur = np.stack(list(self.pool.map(warp_job, jobs, chunksize=2)))
                T9 = torch.cat([T9[:, 1:], self.trunk(prev, cur)[:, None]], 1)
                prev = cur
                rec.append(self.heads(T9, tc, S.t["v"][cs, C.T0 + j]))
            for b, i in enumerate(ids):
                tr = np.array(egos[b].trace)
                res[i] = dict({k: np.array([r[k][b] for r in rec]) for k in rec[0]}, off=tr, ok=C.in_cap(tr),
                              kreal=np.r_[egos[b].kreal, np.full(C.K - len(egos[b].kreal), np.nan)])
        return res

    def batch_pipeline(self, S, c, j):
        """Readouts of the logged state j (frames j..j+9) through the trainer's path (op_adapt_h.trunks on 10 images)."""
        torch = self.torch
        from experiments.op_adapt_h.lib import op_adapt_h as H
        x = torch.from_numpy(np.asarray(S.imgs[c, j: j + C.NH])[None]).to(self.dev)
        with torch.no_grad():
            T = H.trunks(self.m.net, x)
        return self.heads(T, S.t["tc"][[c]].astype(np.float32), S.t["v"][[c], C.T0 + j])


def save(path: Path, specs, res, S):
    keys = res[0].keys()
    out = {k: np.stack([r[k] for r in res]) for k in keys}
    out |= dict(c=np.array([s["c"] for s in specs]), arm=np.array([s["arm"] for s in specs]), kind=np.array([s["kind"] for s in specs]),
                exo=np.stack([s.get("exo") if s.get("exo") is not None else np.zeros(C.K + 1) for s in specs]),
                clip_id=S.t["id"][[s["c"] for s in specs]], cat=S.t["cat"][[s["c"] for s in specs]], seq=S.t["seq"][[s["c"] for s in specs]])
    np.savez(path.with_suffix(".tmp.npz"), **out)
    path.with_suffix(".tmp.npz").replace(path)


def open_yaw_specs(S, degs=(1.0, 2.0)):
    return [dict(c=c, kind="open", exo=C.yaw_exo(s * d), arm=f"yaw{s * d:+g}") for c in range(S.n) for d in degs for s in (1, -1)]


def cmd_check(a, R):
    out = {}
    for name in ("sf", "heldout"):
        S = C.Clips(name)
        cs = list(range(S.n)) if name == "sf" else list(range(0, S.n, 12))
        rep = R.rollouts(S, [dict(c=c, kind="replay", arm="replay") for c in cs])
        d = []
        for c, r in zip(cs, rep):
            for j in (0, 3, C.K):
                b = R.batch_pipeline(S, c, j)
                d.append([abs(r["phi1"][j] - b["phi1"][0]), abs(r["kappa"][j] - b["kappa"][0])])
        d = np.array(d)
        out[f"identity_{name}"] = dict(n=len(d), max_dphi1_deg=float(d[:, 0].max()), max_dkappa=float(d[:, 1].max()))
        print(name, out[f"identity_{name}"], flush=True)
        if name == "sf":
            wm = C.data_dir() / "runs" / "wm_vs_reproj" / "op"
            pp, oo = [], []
            for c, r in zip(cs, rep):
                seg, i = S.t["id"][c].split("_")
                f = wm / f"{seg}_{i}.npz"
                if f.exists():
                    z = np.load(f)
                    pp.append(r["phi1"]), oo.append(z["G/phi1"][7:7 + C.K + 1])
            pp, oo = np.concatenate(pp), np.concatenate(oo)
            out["port_vs_onnx_sf"] = dict(n=len(pp), r=float(np.corrcoef(pp, oo)[0, 1]), median_abs_deg=float(np.median(np.abs(pp - oo))))
            print(out["port_vs_onnx_sf"], flush=True)
            for src in ("anchor", "log"):
                sp = open_yaw_specs(S)
                save(C.root("roll", a.model) / f"sf-check-{src}.npz", sp, R.rollouts(S, sp, src=src), S)
    json.dump(out, open(C.root("check") / f"engine-{a.model}.json", "w"), indent=1)


def collect_specs(S, seed):
    rng = np.random.default_rng([seed, 1])
    sp = []
    for c in range(S.n):
        slow = S.t["cat"][c] in ("launch", "low")
        lo, hi = (0.5, 3.0) if slow else (1.0, 4.0)
        sp.append(dict(c=c, kind="closed", exo=None, arm="free"))
        for s in (1, -1):
            sp.append(dict(c=c, kind="closed", exo=C.kick_exo(s * rng.uniform(lo, hi)), arm=f"kick{'+' if s > 0 else '-'}"))
        for s in (1, -1):
            e = C.swerve_exo(s * rng.uniform(0.2, 0.8), S.t["v"][c])
            sp.append(dict(c=c, kind="closed", exo=e if e is not None else C.kick_exo(s * rng.uniform(lo, hi)),
                           arm=f"swerve{'+' if s > 0 else '-'}" if e is not None else f"kick2{'+' if s > 0 else '-'}"))
    return sp


def eval_specs(S):
    sp = open_yaw_specs(S, (2.0,))
    for c in range(S.n):
        sp.append(dict(c=c, kind="replay", exo=None, arm="replay"))
        sp.append(dict(c=c, kind="closed", exo=None, arm="free"))
        for s in (1, -1):
            sp.append(dict(c=c, kind="closed", exo=C.kick_exo(2.0 * s), arm=f"ckick{s * 2:+g}"))
            e = C.swerve_exo(0.5 * s, S.t["v"][c])
            if e is not None:
                sp.append(dict(c=c, kind="closed", exo=e, arm=f"cswerve{s * 0.5:+g}"))
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "collect", "eval"])
    ap.add_argument("--model", default="shipped")
    ap.add_argument("--set", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    import torch
    if a.dev == "cpu":
        torch.set_num_threads(int(os.environ.get("DG_THREADS", "32")))
    R = Runner(a.model, a.dev, a.workers)
    t0 = time.time()
    if a.cmd == "check":
        cmd_check(a, R)
    else:
        S = C.Clips(a.set)
        sp = collect_specs(S, a.seed) if a.cmd == "collect" else eval_specs(S)
        if a.limit:
            sp = [s for s in sp if s["c"] < a.limit]
        i, n = map(int, a.shard.split("/"))
        cs = sorted({s["c"] for s in sp})[i::n]
        sp = [s for s in sp if s["c"] in set(cs)]
        out = C.root("roll", Path(a.model).parent.name if Path(a.model).exists() else a.model) / f"{a.set}-{a.cmd}-{i}of{n}.npz"
        if out.exists():
            print("exists", out)
            os._exit(0)
        save(out, sp, R.rollouts(S, sp, batch=a.batch), S)
        print(f"{a.cmd} {a.set} {a.model}: {len(sp)} rollouts in {time.time() - t0:.0f} s -> {out}", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
