#!/usr/bin/env python
"""Cosmos-Transfer2.5 on the pilot clips (todos/2026-09-28-cosmos-pilot.md). Runs in envs/cosmos-transfer from the
cosmos-transfer2.5 checkout ($DATA_DIR/third_party/cosmos-transfer2.5), one model per process, every sample of a spec
file in turn with the model loaded once.

  python scripts/cosmos_infer.py --specs <specs.jsonl> --model edge/distilled --out <dir>

Differences from examples/inference.py, all for the pair comparison:
- checkpoints come from the local ModelScope mirror ($DATA_DIR/models/cosmos/<repo>/<file>), not the HF CLI;
- torch / numpy / python RNGs are re-seeded with the sample seed before every sample: the pipeline draws the fps
  conditioning from torch's global RNG, so without this the two members of a pair would not share it;
- the generated frames are also written raw (uint8 npy, T x H x W x 3), since the mp4 is lossy;
- per-sample wall time (model already loaded), torch peak allocation and the NVML peak of the card, in events.jsonl.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")    # registers edge/distilled
DATA = Path(os.environ["DATA_DIR"])
COSMOS = DATA / "third_party" / "cosmos-transfer2.5"
LOCAL = DATA / "models" / "cosmos"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(COSMOS)
sys.path.insert(0, str(COSMOS))


def local_download(cmd_args: list[str]) -> str:
    """Stand-in for checkpoint_db._hf_download: <repo> --repo-type model --revision <rev> [<file>] [--include ...]."""
    repo, rest = cmd_args[0], cmd_args[1:]
    i = rest.index("--revision") + 2
    fname = rest[i] if i < len(rest) and not rest[i].startswith("--") else None
    base = LOCAL / repo.split("/", 1)[1]
    path = base / fname if fname else base
    # The config registry resolves every experiment's checkpoints at import time, most of them never loaded; a
    # missing one only fails if it is actually read.
    if not path.exists():
        print(f"cosmos_infer: no local copy of {repo} {fname or ''} (fine unless it is loaded)", flush=True)
    return str(path)


class NvmlPeak:
    """Peak used memory of one card, sampled every 50 ms in a thread."""

    def __init__(self, index: int):
        import pynvml
        pynvml.nvmlInit()
        self.h, self.nv, self.peak, self._stop = pynvml.nvmlDeviceGetHandleByIndex(index), pynvml, 0, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.nv.nvmlDeviceGetMemoryInfo(self.h).used)
            time.sleep(0.05)

    def reset(self):
        self.peak = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--model", default="edge/distilled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of sample names")
    a = ap.parse_args()

    from cosmos_transfer2._src.imaginaire.utils import checkpoint_db
    checkpoint_db._hf_download = local_download
    # CheckpointFileHf._download asserts that the returned path exists; route it around that assertion
    checkpoint_db.CheckpointFileHf._download = lambda self: local_download(
        [self.repository, "--repo-type", "model", "--revision", self.revision, self.filename])
    import numpy as np
    import torch
    from cosmos_oss.init import init_environment
    from cosmos_transfer2.config import InferenceArguments, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference
    from jevdrive.runlog import RunLog

    import cosmos_transfer2.inference as CI
    save_orig = CI.save_img_or_video

    def save(video, path, *args, **kw):
        """Also keep the generated frames raw: (C, T, H, W) in [0, 1] -> (T, H, W, C) uint8 npy."""
        save_orig(video, path, *args, **kw)
        if "_control_" not in str(path) and "_mask_" not in str(path):
            v = (video.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            np.save(f"{path}.npy", v)
    CI.save_img_or_video = save

    init_environment()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rl = RunLog("cosmos", "infer", a.model.replace("/", "-"))
    samples, keys = InferenceArguments.from_files([Path(a.specs).resolve()])
    if a.only:
        samples = [s for s in samples if s.name in set(a.only.split(","))]
    todo = [s for s in samples if not (out / f"{s.name}.npy").exists()]
    rl.info(f"{len(samples)} samples in {a.specs}, {len(todo)} to do, model {a.model}, controls {keys}")
    if not todo:
        return
    phys = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    peak = NvmlPeak(phys)
    t0 = time.time()
    setup = SetupArguments(output_dir=out, model=a.model, disable_guardrails=True)
    inf = Control2WorldInference(setup, batch_hint_keys=keys)
    rl.info(f"model loaded in {time.time() - t0:.0f} s, card peak {peak.peak / 2**30:.1f} GiB")
    rl.event("load", s=time.time() - t0, card_peak_gib=peak.peak / 2**30)
    for i, s in enumerate(todo):
        for f in (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
            f(s.seed)
        torch.cuda.reset_peak_memory_stats()
        peak.reset()
        torch.cuda.synchronize()
        t = time.time()
        path = inf._generate_sample(s, out, sample_id=i + 1)        # sample_id > 0: no benchmark bookkeeping
        torch.cuda.synchronize()
        dt = time.time() - t
        rec = {"name": s.name, "s": dt, "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "card_peak_gib": peak.peak / 2**30, "path": path}
        rl.info(json.dumps(rec))
        rl.event("sample", **rec)
    rl.event("end", n=len(todo), s=time.time() - t0)


if __name__ == "__main__":
    main()
