#!/usr/bin/env python
"""Cosmos full run worker (fc65452:todos/2026-09-28-cosmos-pilot.md, "full run"; driver experiments/cosmos/lib/cosmos_full.py `run`).
envs/cosmos-transfer, one process per Cosmos slot, the Edge Distilled model loaded once:

  CUDA_VISIBLE_DEVICES=g python experiments/cosmos/archive/cosmos_full_worker.py --root $DATA_DIR/runs/cosmos_full --slot g0 [--keep-npy]
  (--root absolute: importing cosmos_infer changes into the cosmos-transfer2.5 checkout)

Takes READY pairs from <root>/clips/ (oldest first, an O_EXCL claim per pair) and runs the frozen v2 arm G4 on each:
x- with its edgeE control (E4, seed 2025), x- written back as the lossless anchor video, x+ regenerated with its own edgeE
while the latents outside the tight free region are held at x- (guided generation), then the feathered blend (G4b):
x+ = alpha * regenerated + (1 - alpha) * x-. Stored in <root>/pairs/<pair>/: cosmos_{plus,minus}.mp4 and
carla_{plus,minus}.mp4 (H.264 crf 14), gt.npz (with the blend support, see experiments.cosmos.lib.cosmos_full.load_pair), spec.json,
done.json (timings). --keep-npy also writes the raw G4b frames in the pilot layout (<root>/out/G4b/) and keeps the
lossless inputs, so experiments/cosmos/lib/cosmos_eval.py and experiments/cosmos/lib/cosmos_openpilot.py run with COSMOS_ROOT=<root>. Exits when <root>/lane/CONTROLS_DONE exists and nothing is left, when <root>/pairs holds
--target pairs, or on <root>/lane/DRAIN. A pair that raises is marked FAILED in its clip dir and skipped.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/cosmos/archive",)]
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import cosmos_infer as CI  # noqa: E402  (chdirs into the cosmos-transfer2.5 checkout)

DATA = Path(os.environ["DATA_DIR"])
SEED = 2025
H, W = 704, 1280


def ffmpeg() -> str:
    return sorted(glob.glob(str(DATA / "envs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*")))[0]


def write_mp4(path: Path, frames: np.ndarray, crf: int = 0):
    enc = ["-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv444p"] if crf == 0 else \
        ["-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "20",
           "-i", "-", "-c:v", "libx264", *enc, str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def claim(root: Path, pair: str, slot: str) -> bool:
    try:
        fd = os.open(root / "cosmos" / "claims" / pair, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, slot.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def next_pair(root: Path, slot: str):
    ready = sorted(root.glob("clips/*/READY"), key=lambda p: p.stat().st_mtime)
    for r in ready:
        pair = r.parent.name
        if not (root / "pairs" / pair / "done.json").exists() and not (r.parent / "FAILED").exists() and claim(root, pair, slot):
            return pair
    return None


def sample(name: str, spec: dict, video: Path, edge: Path, mask: Path | None = None) -> dict:
    x = {"name": name, "prompt": spec["prompt"], "seed": SEED, "num_steps": 4, "guidance": 3, "video_path": str(video),
         "edge": {"control_path": str(edge), "control_weight": 1.0}}
    if mask is not None:
        x.update(guided_generation_mask=str(mask), guided_generation_step_threshold=99)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--keep-npy", action="store_true")
    ap.add_argument("--start-after", default="", help="wait for this file before loading the model")
    ap.add_argument("--warm", default="", help="file of prompts: fill COSMOS_TE_CACHE with their embeddings and exit")
    a = ap.parse_args()
    root = Path(a.root)
    lane = root / "lane"
    for d in ("cosmos/claims", "pairs", f"cosmos/tmp/{a.slot}"):
        (root / d).mkdir(parents=True, exist_ok=True)
    tmp = root / "cosmos" / "tmp" / a.slot
    CI.patch()
    from cosmos_transfer2.config import InferenceArguments
    from jevdrive.runlog import RunLog
    rl = RunLog(*root.relative_to(DATA / "runs").parts, "cosmos", a.slot)
    rl.info(f"slot {a.slot}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, root {root}")
    while a.start_after and not Path(a.start_after).exists():
        if (lane / "DRAIN").exists():
            return
        time.sleep(60)
    phys = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    peak = CI.NvmlPeak(phys)
    t0 = time.time()
    inf = CI.load_model("edge/distilled", tmp, ["edge"])
    rl.info(f"model loaded in {time.time() - t0:.0f} s")
    rl.event("load", s=time.time() - t0, card_peak_gib=peak.peak / 2**30)
    if a.warm:        # the text encoder goes onto the card for every new prompt (~16 GB); do all of them once, up front
        te = inf.inference_pipeline.model.text_encoder
        for i, pr in enumerate(x for x in Path(a.warm).read_text().splitlines() if x.strip()):
            te.compute_text_embeddings_online({"prompt": [pr]}, "prompt")
            rl.info(f"warm {i}: {pr[:90]}")
        rl.event("warm", n=i + 1, card_peak_gib=peak.peak / 2**30)
        return
    n = 0
    while True:
        if (lane / "DRAIN").exists():
            rl.info("DRAIN: stopping")
            break
        if a.target and len(list(root.glob("pairs/*/done.json"))) >= a.target:
            rl.info(f"target {a.target} reached")
            break
        pair = next_pair(root, a.slot)
        if pair is None:
            if not (lane / "CONTROLS_DONE").exists():
                time.sleep(30)
                continue
            pair = next_pair(root, a.slot)      # READY files written just before CONTROLS_DONE
            if pair is None:
                break
        cd, od = root / "clips" / pair, root / "pairs" / pair
        try:
            t1 = time.time()
            spec = json.loads((cd / "spec.json").read_text())
            recs = []

            def gen(x):
                f = tmp / f"{x['name']}.jsonl"
                f.write_text(json.dumps(x) + "\n")
                s = InferenceArguments.from_files([f])[0][0]
                rec = CI.generate(inf, s, tmp, n + 1, peak)
                recs.append(rec)
                v = np.load(tmp / f"{x['name']}.npy")
                for p in tmp.glob(f"{x['name']}*"):
                    p.unlink()
                return v
            neg = gen(sample(f"{pair}_minus_E4_s{SEED}", spec, cd / "minus" / "rgb.mp4", cd / "minus" / "edgeE.mp4"))
            anchor = tmp / f"{pair}_anchor.mp4"
            write_mp4(anchor, neg)
            reg = gen(sample(f"{pair}_plus_G4_s{SEED}", spec, anchor, cd / "plus" / "edgeE.mp4", cd / "anchor_mask_tight.mp4"))
            anchor.unlink()
            al = np.load(cd / "alpha_tight.npz")["alpha"].astype(np.float32)[..., None]
            plus = np.rint(al * reg.astype(np.float32) + (1 - al) * neg).astype(np.uint8)
            del reg, al
            od.mkdir(parents=True, exist_ok=True)
            part = od / "part"
            part.mkdir(exist_ok=True)
            write_mp4(part / "cosmos_minus.mp4", neg, crf=14)
            write_mp4(part / "cosmos_plus.mp4", plus, crf=14)
            for m in ("plus", "minus"):
                shutil.copy2(cd / f"carla_{m}.mp4", part / f"carla_{m}.mp4")
            for f in ("gt.npz", "gt_boxes.npz", "spec.json"):
                shutil.copy2(cd / f, part / f)
            if a.keep_npy:
                ev = root / "out" / "G4b"             # COSMOS_ROOT=<root> for cosmos_eval / cosmos_openpilot
                ev.mkdir(parents=True, exist_ok=True)
                np.save(ev / f"{pair}_plus_G4b_s{SEED}.npy", plus)
                np.save(ev / f"{pair}_minus_G4b_s{SEED}.npy", neg)
            for p in part.iterdir():
                p.replace(od / p.name)
            part.rmdir()
            done = {"pair": pair, "slot": a.slot, "s_minus": recs[0]["s"], "s_plus": recs[1]["s"],
                    "card_peak_gib": max(r["card_peak_gib"] for r in recs), "wall_s": time.time() - t1, "t": time.time()}
            (od / "done.json").write_text(json.dumps(done))
            if not a.keep_npy:                    # the lossless inputs are regenerable from the CARLA attempts' records
                for f in ("plus/rgb.mp4", "minus/rgb.mp4", "plus/edgeE.mp4", "minus/edgeE.mp4", "anchor_mask_tight.mp4",
                          "alpha_tight.npz", "carla_plus.mp4", "carla_minus.mp4"):
                    (cd / f).unlink(missing_ok=True)
            n += 1
            rl.info(json.dumps(done))
            rl.event("pair", **done)
        except Exception:                          # noqa: BLE001
            err = traceback.format_exc()
            (cd / "FAILED").write_text(err)
            rl.info(f"{pair} FAILED\n{err}")
            rl.event("pair_failed", pair=pair, error=err[-2000:])
            for p in tmp.glob(f"{pair}*"):
                p.unlink()
    rl.event("end", n=n, s=time.time() - t0)


if __name__ == "__main__":
    main()
