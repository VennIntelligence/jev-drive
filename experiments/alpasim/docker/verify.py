"""Build-time check of the submission image (Dockerfile RUN, CPU only): every driver module imports from the files in the image, every
baked checkpoint reads, and tags.json (tag -> driver family, what serve.py's `auto` uses) is written next to the checkpoints.
A missing code file, package or checkpoint key fails the build instead of the first rollout."""
import json
import os
import sys
from pathlib import Path

RUNS = Path(os.environ["DATA_DIR"]) / "runs/op_parity/runs"
sys.path.insert(0, "/app/jev-drive/experiments/alpasim/lib")
import ap2_driver, ens_driver, sh30_driver  # noqa: E401,E402,F401
import cv2, torch  # noqa: E401,E402,F401
import pp_train as T  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402

assert (A.MODELS_DIR / A.FILES["cinque"]).is_file(), A.MODELS_DIR
tags = {}
for f in sorted(RUNS.glob("*/ckpt-final.pt")):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    assert ck["model"]["arm"] == "P2", (f, ck["model"]["arm"])
    T.proot("runs", f.parent.name)                       # the lookup the drivers do; the directory must already exist (read-only root)
    tags[f.parent.name] = "ap2" if (ck.get("ap2") or {}).get("std", "navsim") == "alpasim" else "sh30"
assert tags, f"no checkpoint under {RUNS}"
(RUNS / "tags.json").write_text(json.dumps(tags, indent=1))
print("verify:", tags, "torch", torch.__version__, "cuda", torch.version.cuda)
