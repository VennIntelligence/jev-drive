"""Real vs render, navhard stage 2: openpilot (Cinque) on the synthetic CAM_F0 frames that sit within 0.5 m / 1 deg of the logged pose and
on the real CAM_F0 at the same timestamp (rvr_nav_pairs.py export), plus the next real log frame (0.5 s later: the adjacent-frame floor).
Single-frame protocol on every side: zero state, the frame held for 1.5 s (31 steps at 20 Hz), heads of the last step. Adapter
navsim_zs.OpenpilotMaps (op_lb's CPU warp of CAM_F0, each frame with its own calibration). Also image statistics of both sides and a
sharpened synthetic arm (unsharp mask fitted here to the real high-frequency share, on the first 40 pairs).
  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/openpilot/bin/python experiments/leaderboard_audit/scripts/rvr_nav_op.py
Writes $DATA_DIR/runs/real_vs_render/nav_heads.npz and nav_imgstats.json."""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).parent)]
from jevdrive import navsim_zs as N  # noqa: E402
import rvr_hugsim_op as H  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = D / "runs/real_vs_render"
SIDES = ("syn", "real", "next", "syn_sharp")


def rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def ycc(x):
    from PIL import Image
    return np.asarray(Image.fromarray(x).convert("YCbCr"))


def stats(x):
    g = H.grey(x[::2, ::2].astype(np.float64))           # 960x540, close to the model's sampling density (1.7 source px per model px)
    return [H.hf_share(g), H.noise_sigma(g)]


def pack_pair(args):
    p, sharp = args
    out, st = {}, {}
    for side in ("syn", "real", "next"):
        c = p[f"{side}_cam"]
        x = rgb(c["path"])
        st[side] = stats(x)
        out[side] = N.OpenpilotMaps(c)(ycc(x))
        if side == "syn" and sharp is not None:
            y = H.apply_fix(x, "sharp", sharp)
            st["syn_sharp"] = stats(y)
            out["syn_sharp"] = N.OpenpilotMaps(c)(ycc(y))
    return out, st


def fit_sharp(pairs):
    real = [stats(rgb(p["real_cam"]["path"]))[0] for p in pairs]
    syn = [rgb(p["syn_cam"]["path"]) for p in pairs]
    best = None
    for sigma in (0.7, 1.0, 1.5, 2.0, 3.0):
        for amount in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
            P = dict(sigma=sigma, amount=amount)
            hf = np.mean([stats(H.apply_fix(x, "sharp", P))[0] for x in syn])
            if best is None or abs(hf - np.mean(real)) < best[0]:
                best = (abs(hf - np.mean(real)), P)
    return best[1]


def main():
    pairs = pickle.load(open(RUN / "nav_pairs.pkl", "rb"))
    sharp = fit_sharp(pairs[:40])
    m = H.model()
    keep, hs = H.keep_layout(m)
    R = {s: [] for s in SIDES}
    ST = {s: [] for s in SIDES}
    with ProcessPoolExecutor(16) as ex:
        for fr, st in ex.map(pack_pair, [(p, sharp) for p in pairs], chunksize=2):
            for s in SIDES:
                m.reset()
                for _ in range(31):
                    raw = m.step(fr[s])
                R[s].append(raw[keep])
                ST[s].append(st[s])
    meta = dict(d=[p["d"] for p in pairs], d_next=[p["d_next"] for p in pairs], v=[p["v"] for p in pairs], k=[p["k"] for p in pairs],
                log=[p["log"] for p in pairs], syn=[p["syn"] for p in pairs])
    np.savez(RUN / "nav_heads.npz", **{s: np.stack(v) for s, v in R.items()}, info=json.dumps(dict(heads_slices=hs, sharp=sharp, **meta)))
    st = {s: dict(zip(("hf", "noise"), np.median(np.array(v), 0).round(4).tolist())) for s, v in ST.items()}
    (RUN / "nav_imgstats.json").write_text(json.dumps(dict(stats=st, sharp=sharp, n=len(pairs)), indent=1))
    print(json.dumps(st), sharp)


if __name__ == "__main__":
    main()
