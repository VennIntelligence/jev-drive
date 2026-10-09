"""M1 offline replay with one input swapped at a time (decision 205): the logged driver-side messages of finished rollouts go back
through the real driver class (c1_replay.py's harness, no simulator), and every chosen decision is planned again with a single input
changed. Open loop: the state of each decision is the logged one.

  m1_replay.py --driver sh30|ap2 --msgs <dir of c1_extract.py msgs> --spec <json> --out <pkl>        (box, envs/op-train, one GPU)

spec: {scene: {"ks": [decisions that get the variants] | "all": true, "frames": bool}}. Every decision is driven once as run (the
session state must evolve as in the run); variants per chosen decision:
  run          as run
  L / S / R    command forced
  mir / mirL   mirrored world: frames flipped left-right, lateral terms of poses / velocity / acceleration / yaw rate negated, command
               L <-> R; the plan is mirrored back. mir keeps the right-hand traffic flag, mirL sets left-hand. A driver without a
               lateral bias of its own returns the as-run plan
  ax0          acceleration feature zeroed
  strF         adapter ego features from a straightened history (poses on the x axis at the driven spacing, yaw 0, lateral velocity 0);
               frames as run
  strW         the reverse: frames warped along the straightened history, ego features as run
  str          both
  w50          frames warped along the history halfway between the driven and the straightened one, ego features as run
  arcL / arcR  strF plus a synthetic constant yaw rate of +/- ARC_W rad/s in the history poses (features only): the plan's response to
               "the ego has been turning", with nothing else changed
  hv0          strF with the history kept but the newest 0.5 s straightened only (pose of t0 - 0.5 s put on the heading of t0)
  navH / navE / navHE   decision 3 only (its four keyframes are the NAVSIM token's): the vision tokens, the ego features, or both
               replaced by the cached NAVSIM ones of that token (op_parity cache lb_navtest: real camera frames, logged ego state)
Also stored at decision 3: per-slot cosine similarity between the vision tokens of the rendered frames and the NAVSIM ones.
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
FORCED = {"L": [1, 0, 0, 0], "S": [0, 1, 0, 0], "R": [0, 0, 1, 0]}
ARC_W = 0.05


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def flip(fr):
    """Packed model frames (..., 6, 128, 256) mirrored left-right: columns reversed, the two Y column phases swapped."""
    return np.ascontiguousarray(fr[..., [2, 3, 0, 1, 4, 5], :, ::-1])


def straight(P, V):
    """History (4, 3), (4, 2) -> the same spacing on the x axis, yaw 0, no lateral velocity."""
    d = np.r_[np.hypot(*np.diff(P[:, :2], axis=0).T), 0.0]
    s = d[::-1].cumsum()[::-1]
    return np.c_[-s, np.zeros(4), np.zeros(4)], np.c_[np.hypot(V[:, 0], V[:, 1]) * np.sign(V[:, 0] + 1e-9), np.zeros(4)]


def arc(P, V, w):
    """Straightened history bent to a constant yaw rate w (left positive)."""
    P, V = straight(P, V)
    s, k = -P[:, 0], w / max(float(V[-1, 0]), 1.0)
    return np.c_[P[:, 0], k * s * s / 2, -k * s], V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, choices=["sh30", "ap2"]), ap.add_argument("--msgs", required=True)
    ap.add_argument("--spec", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import sh30_driver as D
    if a.driver == "sh30":
        core = D.C.Core(os.environ.get("SH30_TAG", "SH30-F-s0"), "cuda", os.environ.get("SH30_COLD", "backwarp"))
        drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
    else:
        import ap2_driver as AD
        core = AD.AC.Core(os.environ.get("AP2_TAG", "AP2-AB-s0"), "cuda", "")
        drv = AD.Driver(core, Path(tempfile.mkdtemp()), 0, False)
    PA, pb = D.C.PA, D.egodriver_pb2
    cr = Path(os.environ["DATA_DIR"]) / "runs/op_parity/cache"
    tab = np.load(cr / "lb_navtest/tab.npz")
    ix = {n: i for i, n in enumerate(tab["names"])}
    nav_ego, nav_H = tab["ego"], np.load(cr / "lb_navtest@warp/front.npy", mmap_mode="r")

    feat, swap, seen = {"fn": None}, {"H": None, "ego": None}, {}
    ego_features = PA.ego_features

    def patched(P, V, acc, cmd, *x, **k):
        if feat["fn"] is None:
            return ego_features(P, V, acc, cmd, *x, **k)
        return feat["fn"](np.asarray(P, np.float64), np.asarray(V, np.float64), acc, cmd)
    PA.ego_features = patched

    def hook(mod, args):
        H, ego, tc = args[:3]
        seen["H"] = H.detach()
        if swap["H"] is not None:
            H = swap["H"].to(H)
        if swap["ego"] is not None:
            ego = torch.from_numpy(swap["ego"][None]).to(ego)
        return (H, ego, tc) + tuple(args[3:])
    core.model.register_forward_pre_hook(hook)

    last, orig = {}, core.plan

    def plan(*x, **k):
        last["x"], last["k"] = x, k
        last["o"] = orig(*x, **k)
        return last["o"]
    core.plan = plan

    def again(keys=None, pose=None, vel=None, acc=None, cmd=None, mirror=False, lht=None, fn=None, H=None, ego=None):
        k0, p0, v0, a0, c0, cam = last["x"]
        kw = dict(last["k"])
        keys, pose, vel = k0 if keys is None else keys, np.array(p0 if pose is None else pose, float), np.array(v0 if vel is None else vel, float)
        acc, cmd, cam = np.array(a0 if acc is None else acc, float), np.array(c0 if cmd is None else cmd, np.float32), np.array(cam, float)
        if mirror:
            keys, cmd = [flip(f) for f in keys], cmd[[2, 1, 0, 3]]
            pose[:, 1:] *= -1
            vel[:, 1] *= -1
            acc[1] *= -1
            cam[1] *= -1
            kw["yaw_rate"] = -kw.get("yaw_rate", 0.0)
        if lht is not None:
            kw["lht"] = lht
        feat["fn"], swap["H"], swap["ego"] = fn, H, ego
        try:
            p = orig(keys, pose, vel, acc, cmd, cam, **kw)["poses"].copy()
        finally:
            feat["fn"], swap["H"], swap["ego"] = None, None, None
        if mirror:
            p[:, 1:] *= -1
        return p

    spec, ctx, out = json.loads(Path(a.spec).read_text()), Ctx(), {}
    for i, (scene, sp) in enumerate(sorted(spec.items())):
        f = Path(a.msgs) / f"{scene}.pkl"
        if not f.exists():
            continue
        rec, k, uuid = dict(plans={}, frames={}, cmd={}, ego={}, cos={}), 0, None
        j = ix.get(scene[-16:])
        for kind, raw in pickle.load(open(f, "rb")):
            if kind == "driver_session_request":
                req = pb.DriveSessionRequest.FromString(raw)
                uuid = req.session_uuid
                drv.start_session(req, ctx)
            elif kind == "driver_camera_image":
                drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
            elif kind == "driver_ego_trajectory":
                drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
            elif kind == "route_request":
                drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
            elif kind == "driver_request":
                drv.drive(pb.DriveRequest.FromString(raw), ctx)
                o = last["o"]
                P = rec["plans"][k] = {"run": o["poses"].copy()}
                rec["cmd"][k], rec["ego"][k] = int(np.argmax(last["x"][4])), o["ego"].copy()
                if sp.get("frames"):
                    rec["frames"][k] = dict(cur=o["cur"][[0, 4, 7]].copy(), valid=o["valid"].copy())
                Hs, e_run, hist = seen["H"].clone(), o["ego"].copy(), o["hist"].copy()
                if sp.get("all") or k in sp.get("ks", []):
                    for name, c in FORCED.items():
                        P[name] = again(cmd=c)
                    P["mir"], P["mirL"] = again(mirror=True), again(mirror=True, lht=True)
                    P["ax0"] = again(acc=[0.0, 0.0])
                    asrun = lambda *_: e_run                                                     # noqa: E731
                    F = lambda g: (lambda P_, V_, acc, cmd: ego_features(*g(P_, V_), acc, cmd))    # noqa: E731
                    P["strF"] = again(fn=F(straight))
                    P["arcL"], P["arcR"] = again(fn=F(lambda p, v: arc(p, v, ARC_W))), again(fn=F(lambda p, v: arc(p, v, -ARC_W)))
                    if len(last["x"][0]) == 4:
                        ps, vs = straight(np.array(last["x"][1], float), np.array(last["x"][2], float))
                        P["strW"], P["str"] = again(pose=ps, vel=vs, fn=asrun), again(pose=ps, vel=vs)
                        p0, v0 = np.array(last["x"][1], float), np.array(last["x"][2], float)
                        P["w50"] = again(pose=0.5 * (p0 + ps), vel=0.5 * (v0 + vs), fn=asrun)

                        def hv0(p, v):
                            p = p.copy()
                            d = np.hypot(*(p[3, :2] - p[2, :2]))
                            sh = np.array([-d, 0.0]) - p[2, :2]
                            p[:3, :2] += sh
                            p[2, 2] = 0.0
                            return p, v
                        P["hv0"] = again(fn=F(hv0))
                if k == 3 and j is not None and len(last["x"][0]) == 4:
                    Hn = torch.from_numpy(np.asarray(nav_H[j])[None]).to(Hs)
                    P["navH"], P["navE"], P["navHE"] = again(H=Hn), again(ego=nav_ego[j]), again(H=Hn, ego=nav_ego[j])
                    rec["cos"] = torch.nn.functional.cosine_similarity(Hs.float(), Hn.float(), dim=-1).mean(-1)[0].cpu().numpy()
                    rec["ego_nav"] = nav_ego[j].copy()
                k += 1
        drv.sessions.pop(uuid, None)
        out[scene] = rec
        if i % 20 == 0:
            print(i, len(spec), scene, flush=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print("replay", a.driver, len(out), "scenes ->", a.out, os.path.getsize(a.out) >> 20, "MB", flush=True)


if __name__ == "__main__":
    main()
