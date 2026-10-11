#!/usr/bin/env python
"""Frame A/B, stage 2: the recorded HUGSIM decisions through the stateless training-side path (the checkpoint's torch port and
sh30_core.Core, both called as they are). experiments/hugsim/plans/2026-10-11-frame-ab-prereg.md, results/frame_ab.md.

Arms per decision n of a scenario (F = the recorded packed frames, one per simulator step):
  a_t   9 raw slots: slot j = the pair (F[n-j-1], F[n-j]), (F[0], F[0]) before the episode (the served queue, decision 249); served ego
        features (model clock); no desire. The torch anchor of the served path
  s8    a_t without the oldest slot
  s8z   s8 with the oldest pair starting from a zero image (the slot structure of the training rows)
  wd    Core.plan on the model clock (mapping M1): keys = F[n-7], F[n-5], F[n-2], F[n] (lib/parity_hugsim.key_steps), poses at model
        -1.5 / -1.0 / -0.5 / 0 s, speed x 1.25; its ego features equal the served ones on warm decisions
  b     Core.plan on the real clock (mapping M2): keys = F[n-6], F[n-4], F[n-2], F[n] at their own poses, nothing dilated
Gates: (ii) Core.plan on navtest tokens against the bench predictions; (iii) the HUGSIM pose convention of the warp, at pixel level.
Writes <out>/torch/<scenario>.npz, <out>/gate_ii.json, <out>/gate_iii.json.

  $DATA_DIR/envs/op-train/bin/python experiments/hugsim/scripts/frame_ab_torch.py --run-dir <bench run dir> --out <dir>
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/alpasim/lib"), str(_R / "experiments/hugsim/scripts")]
import argparse, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import parity_hugsim as PH  # noqa: E402
import sh30_core as C  # noqa: E402
from frame_ab_onnx import load  # noqa: E402
from jevdrive import hugsim_zs as HZ, navsim_zs as NZ, op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.openpilot.interface import HUGSIM_HEIGHT  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

M1_BACK, M2_BACK = (7, 5, 2, 0), (6, 4, 2, 0)        # simulator steps back of the 4 keys, oldest first
DIL, SIM_DT = 1.25, 0.25


def history(S, n):
    """jevdrive.hugsim_zs.History of the first n + 1 recorded steps and the info fields parity_hugsim.ego_inputs reads."""
    h, st = HZ.History(), S["state"]
    h.t, h.pos, h.th, h.v = list(st[:n + 1, 0]), list(st[:n + 1, 1:3]), list(st[:n + 1, 3]), list(st[:n + 1, 4])
    return h, {"ego_velo": st[n, 4], "accelerate": st[n, 5], "command": int(st[n, 6])}


def core_inputs(S, n, back, scale, d):
    """Core.plan arguments of decision n: the keys that exist (oldest first), their rear-axle poses in the frame of step n on a clock
    `scale` times faster than the simulator's (parity_hugsim.ego_inputs: the same poses, velocity and acceleration the bias server gets
    at scale 1.25), per-key body velocity, the yaw rate of the oldest key (cold start only)."""
    h, info = history(S, n)
    ego, pose = PH.ego_inputs(h, info, d, scale)
    st = S["state"]
    have = [i for i, bk in enumerate(back) if n - bk >= 0]
    tq = st[n, 0] + PH.KEYS_S * scale
    v = np.interp(tq, st[:n + 1, 0], st[:n + 1, 4]) * scale
    k0 = n - back[have[0]]
    w = -(st[min(k0 + 1, n), 3] - st[k0, 3]) / (SIM_DT / scale) if n > k0 else 0.0          # left-positive, on the arm's clock
    cmd = np.zeros(4, np.float32)
    cmd[PH.NAV_MAP[int(st[n, 6])]] = 1.0
    return dict(keys=[S["img2"][n - back[i]] for i in have], pose=pose[have], vel=np.c_[v[have], np.zeros(len(have))],
                acc=np.array([st[n, 5] * scale ** 2, 0.0]), cmd=cmd, yaw_rate=float(w)), ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default="P2H10-F-s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-nav", type=int, default=32)
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "torch").mkdir(parents=True, exist_ok=True)
    with Run("hugsim_frame_ab", "torch", seed=a.seed, config=vars(a), resume=a.resume) as run:
        import torch
        from jevdrive import op_adapt as A
        core = C.Core(a.tag)
        core.base_out = True
        model, dev, sl = core.model, core.dev, core.model.net.slices
        tc = torch.tensor([[1.0, 0.0]], device=dev)

        # ---- gate (ii): Core.plan on navtest tokens against the bench predictions of the same checkpoint
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        idx = NZ.load_index("navtest", slim=True)
        R = data_dir() / "runs"
        tab = np.load(R / "op_parity/cache/lb_navtest/tab.npz")
        row = {t: k for k, t in enumerate(tab["names"].tolist())}
        preds = np.load(R / f"bench/ol/lb_navtest/preds/{a.tag}-warp__base.npz")["poses"]
        g2 = []
        for k in run.tqdm(np.random.default_rng(0).permutation(len(idx))[:a.n_nav], desc="gate ii"):
            e = idx[k]
            cam = e["cams"][-1]["CAM_F0"]
            o = core.plan([C.pack(e["cams"][f]["CAM_F0"]["path"], cam) for f in range(4)], np.asarray(e["pose"], float), np.asarray(e["vel"], float),
                          acc=e["acc"][-1], cmd=e["cmd"][-1], cam_t=cam["t"], lht=e["map"] in ("sg-one-north",))
            ref = preds[row[e["token"]]]
            g2.append(dict(token=e["token"], max_m=float(np.abs(o["poses"][:, :2] - ref[:, :2]).max()),
                           nobias_max_m=float(np.abs(o["poses_base"][:, :2] - ref[:, :2]).max())))
        mx, nb = np.array([r["max_m"] for r in g2]), np.array([r["nobias_max_m"] for r in g2])
        G2 = dict(tol_m=0.05, n=len(g2), share_within=float((mx <= 0.05).mean()), max_m=float(mx.max()), median_m=float(np.median(mx)),
                  control_share_over=float((nb > 0.05).mean()), control_median_m=float(np.median(nb)), rows=g2)
        G2["passed"] = bool(G2["share_within"] >= 0.95 and G2["control_share_over"] >= 0.5)
        (out / "gate_ii.json").write_text(json.dumps(G2, indent=1))
        run.info("gate (ii) %s: within %.3f, max %.4f m, control over %.3f", "PASS" if G2["passed"] else "FAIL", G2["share_within"], G2["max_m"], G2["control_share_over"])
        core.base_out = False

        def enc(prev, cur):
            with torch.no_grad():
                p, c = (torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in (prev, cur))
                return torch.cat([model.net.run_batched(A.vision_feeds(p[i:i + 32], c[i:i + 32]), ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                                  for i in range(0, len(p), 32)])

        def policy(front, ego):
            """front (N, n, 32, 512) tokens, ego (N, 20) -> plan mu (N, 33, 15), action mu (N, 2)."""
            res = []
            with torch.no_grad():
                for i in range(0, len(front), 64):
                    res.append(model(front[i:i + 64], torch.from_numpy(ego[i:i + 64]).to(dev), tc.expand(len(front[i:i + 64]), 2)).float())
            o = torch.cat(res)
            ps, ac = sl["plan"], sl["action"]
            return (o[:, ps.start:ps.start + 33 * 15].reshape(-1, 33, 15).cpu().numpy(), o[:, ac.start:ac.start + 2].cpu().numpy())

        g3, sums = [], {}
        for dd in run.tqdm(sorted(Path(a.run_dir).glob("bench/zs/*/zs_frames")), desc="scenarios"):
            S, name = load(dd), dd.parent.name
            F, st, N = S["img2"], S["state"], len(S["img2"])
            d = float(S["setup"]["rear_offset"])
            cam_t = np.array([d, 0.0, HUGSIM_HEIGHT.get(S["setup"]["dataset"], 1.5)])
            prev = np.concatenate([F[:1], F[:-1]])
            T, TZ = enc(prev, F), enc(np.zeros_like(F), F)                 # tokens of (F[k-1], F[k]) and of (zero image, F[k])
            ix = np.maximum(np.arange(N)[:, None] - np.arange(8, -1, -1)[None], 0)        # (N, 9) frame index of slot j = 8 .. 0
            f9 = T[torch.from_numpy(ix).to(dev)]
            f8z = torch.cat([TZ[torch.from_numpy(ix[:, 1]).to(dev)][:, None], f9[:, 2:]], 1)
            res = {}
            for arm, fr in (("a_t", f9), ("s8", f9[:, 1:]), ("s8z", f8z)):
                mu, ac = policy(fr, S["ego"])
                res.update({f"pos_{arm}": mu[:, :, 0:3], f"yaw_{arm}": mu[:, :, 11], f"act_{arm}": ac})
            for arm, back, scale in (("wd", M1_BACK, DIL), ("b", M2_BACK, 1.0)):
                pos, yaw, act = np.zeros((N, 33, 3), np.float32), np.zeros((N, 33), np.float32), np.zeros((N, 2), np.float32)
                nk, ego_d = np.zeros(N, int), np.zeros(N)
                for n in range(N):
                    kw, ego_srv = core_inputs(S, n, back, scale, d)
                    o = core.plan(cam_t=cam_t, lht=False, **kw)
                    with torch.no_grad():                                  # the action head of the same pass (Core.plan returns the plan only)
                        full = model(o["tokens"][None], torch.from_numpy(o["ego"][None]).to(dev), tc).float()[0]
                    assert float((full[core.pi].reshape(33, 15).cpu() - torch.from_numpy(o["mu"])).abs().max()) < 1e-3
                    pos[n], yaw[n], act[n] = o["mu"][:, 0:3], o["mu"][:, 11], full[sl["action"].start:sl["action"].start + 2].cpu().numpy()
                    nk[n], ego_d[n] = len(kw["keys"]), float(np.abs(o["ego"] - (S["ego"][n] if scale != 1.0 else ego_srv)).max())
                res.update({f"pos_{arm}": pos, f"yaw_{arm}": yaw, f"act_{arm}": act, f"nkeys_{arm}": nk, f"egodiff_{arm}": ego_d})
            np.savez(out / "torch" / f"{name}.npz", state=st, **res)
            sums[name] = dict(n=N, wd_ego_max_warm=float(res["egodiff_wd"][res["nkeys_wd"] == 4].max()) if (res["nkeys_wd"] == 4).any() else None)

            # ---- gate (iii): F[n-2] warped along the recorded ego motion to the pose of step n-1, against the real F[n-1]
            for n in range(9, N, 5):
                if st[n - 1, 4] < 3.0:
                    continue
                h, info = history(S, n)
                h.t = list(np.arange(len(h.t)) * 0.5)                      # poses of steps n-3 .. n as "keys 0.5 s apart": rows 1, 2 = steps n-2, n-1
                _, p = PH.ego_inputs(h, dict(info), d, 1.0)
                src, dst = F[n - 2], F[n - 1]
                mae = lambda x: float(np.abs(x[:, :4].astype(np.float32) - dst[:, :4].astype(np.float32)).mean())  # noqa: E731
                flip = lambda q: np.array([q[0], -q[1], -q[2]])  # noqa: E731
                g3.append(dict(scenario=name, n=n, v=float(st[n - 1, 4]), none=mae(src), warp=mae(I.warp_frame(src, cam_t, p[2], p[1])),
                               mirrored=mae(I.warp_frame(src, cam_t, flip(p[2]), flip(p[1]))), reversed=mae(I.warp_frame(src, cam_t, p[1], p[2]))))
            run.info("%s: %s", name, json.dumps(sums[name]))
        G3 = {k: float(np.mean([r[k] for r in g3])) for k in ("none", "warp", "mirrored", "reversed")}
        G3.update(n=len(g3), share_warp_below_none=float(np.mean([r["warp"] < r["none"] for r in g3])),
                  passed=bool(G3["warp"] < min(G3["none"], G3["mirrored"], G3["reversed"])), rows=g3)
        (out / "gate_iii.json").write_text(json.dumps(G3, indent=1))
        (out / "torch_summary.json").write_text(json.dumps(sums, indent=1))
        run.info("gate (iii) %s: %s", "PASS" if G3["passed"] else "FAIL", json.dumps({k: v for k, v in G3.items() if k != "rows"}))
        run.summary.update(gate_ii=G2["passed"], gate_iii=G3["passed"])


if __name__ == "__main__":
    main()
