"""Build-time check of the submission image (Dockerfile RUN, CPU only): every driver module imports from the files in the image, every
baked checkpoint reads, and tags.json (tag -> nuPlan driver family, what serve.py's `auto` uses) is written next to the checkpoints.
A checkpoint with the ROUTE-MEM sidecar (route_mem.json, the PAI candidates) is also built the way pai_driver.py serves it (load_branch:
branch attached, weights loaded as the driver loads them (adapter and branch state dict strict), memory layout read), on the CPU. A missing code file, package, sidecar or checkpoint key
fails the build instead of the first rollout."""
import json
import os
import sys
from pathlib import Path

RUNS = Path(os.environ["DATA_DIR"]) / "runs/op_parity/runs"
sys.path.insert(0, "/app/jev-drive/experiments/alpasim/lib")
import ap2_driver, ens_driver, sh30_driver, sh30_hold  # noqa: E401,E402,F401
import pai_driver as PD  # noqa: E402
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
    if (f.parent / "route_mem.json").exists():           # a PAI candidate: the driver's own loader, as main() does it
        RM = PD.load_branch(f.parent.name)
        assert RM is not None, f
        m = T.load_pmodel(f.parent.name, "cpu")
        print("verify: branch", f.parent.name, "dim", RM.dim_of(f.parent.name), "order", RM.head_order(f.parent.name),
              "reflex", bool(getattr(m, "reflex", False)), flush=True)
        del m
assert tags, f"no checkpoint under {RUNS}"
(RUNS / "tags.json").write_text(json.dumps(tags, indent=1))
print("verify:", tags, "torch", torch.__version__, "cuda", torch.version.cuda)
