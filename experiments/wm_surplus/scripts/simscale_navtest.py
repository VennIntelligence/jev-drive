"""navtest poses of the SimScale camera-only NAVSIM models (AlpaSim e2e_challenge samples), run as shipped: the bench `agent`
runner `cmd` for SimScale-LTF / SimScale-DD (jevdrive/bench/agents/SimScale-*.json), env envs/alpasim-ltf (no navsim installed).

The policies take exactly what a NAVSIM AgentInput holds at t0, so this file only builds that input from the OpenScene log
pickles, as navsim's own loader does (navsim/common/dataclasses.py, from_scene_dict_list): CAM_L0 / CAM_F0 / CAM_R0 of the
token's frame (RGB uint8 1920 x 1080, PIL as navsim's Camera loader), command = frame `driving_command` (one-hot, 4),
velocity = `ego_dynamic_state[:2]`, acceleration = `ego_dynamic_state[2:4]` (ego frame). Output: their (8, 3) poses (x forward,
y left, heading at 0.5 .. 4 s), the NAVSIM trajectory convention. SimScale-DD: noise seed 0, noise index from the token
(sha256), so a token's diffusion noise does not depend on chunking. Batches of --batch requests through predict_batch.

    python simscale_navtest.py --model ss-ltf|ss-dd --tokens chunk.tsv --out chunk.npz
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
NAV = D / "datasets/navsim"
E2E = D / "third_party/alpasim/e2e_challenge"
MODELS = {
    "ss-ltf": (E2E / "sample_submission_simscale_navsim_transfuser", "navsim_transfuser_challenge.policy", "LtfPolicy",
               D / "models/alpasim_ltf/ltf_sim_navtest.ckpt"),
    "ss-dd": (E2E / "sample_submission_simscale_navsim_diffusiondrive", "navsim_diffusiondrive_challenge.policy", "DiffusionDrivePolicy",
              D / "models/alpasim_dd/diffusiondrive_sim_navhard.ckpt"),
}


def frames(logs) -> dict:
    out = {}
    for lg in logs:
        for f in pickle.load(open(NAV / "navsim_logs/test" / f"{lg}.pkl", "rb")):
            out[f["token"]] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        return
    rows = [ln.split("\t") for ln in Path(a.tokens).read_text().split("\n") if ln.strip()]
    toks = [r[0] for r in rows]
    F = frames(sorted({r[1] for r in rows}))
    from PIL import Image
    root, mod, cls, ckpt = MODELS[a.model]
    sys.path.insert(0, str(root))
    m = __import__(mod, fromlist=[cls])
    policy = getattr(m, cls)(str(ckpt), device="cuda")            # shipped defaults (fp16 autocast, warm-up)
    t0 = time.time()
    P = []
    for i in range(0, len(toks), a.batch):
        reqs = []
        for t in toks[i:i + a.batch]:
            f = F[t]
            ds = np.asarray(f["ego_dynamic_state"], np.float32)
            imgs = {c: np.array(Image.open(NAV / "sensor_blobs/test" / f["cams"][c]["data_path"])) for c in m.CAMERA_IDS}
            kw = dict(noise_seed=0, noise_index=int(hashlib.sha256(t.encode()).hexdigest()[:7], 16)) if a.model == "ss-dd" else {}
            reqs.append(m.InferenceInput(images=imgs, command_one_hot=np.asarray(f["driving_command"], np.float32),
                                         velocity_xy=ds[:2], acceleration_xy=ds[2:4], **kw))
        P += [np.asarray(p.trajectory, np.float32) for p in policy.predict_batch(reqs)]
        print(f"{time.strftime('%T')} {min(i + a.batch, len(toks))}/{len(toks)} {(time.time() - t0) / min(i + a.batch, len(toks)):.3f} s/token", flush=True)
    P = np.stack(P)
    assert P.shape == (len(toks), 8, 3) and np.isfinite(P).all()
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
    np.savez(tmp, tokens=np.asarray(toks), poses=P, wall_s=time.time() - t0)
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
