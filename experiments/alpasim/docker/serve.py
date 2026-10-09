"""Entry point of the submission image: choose the driver family and checkpoint at run time, then become that driver's process.

  JEV_TAG     run tag under $DATA_DIR/runs/op_parity/runs (baked in by build.sh); `tagA+tagB` serves the adapter ensemble
  JEV_DRIVER  sh30 | ap2 | ens | auto (default). auto = the family recorded for the tag in tags.json (verify.py, at build time):
              checkpoints trained on the AlpaSim input standard -> ap2_driver.py, every other op_parity tag -> sh30_driver.py
Everything else (SH30_COLD, AP2_COLD, SH30_MOTION, ALPASIM_DRIVER_*) passes through to the driver unchanged.
"""
import json
import os
import sys
from pathlib import Path

LIB = Path(os.environ.get("JEV_LIB", "/app/jev-drive/experiments/alpasim/lib"))
RUNS = Path(os.environ["DATA_DIR"]) / "runs/op_parity/runs"


def main() -> None:
    tag, fam = os.environ["JEV_TAG"], os.environ.get("JEV_DRIVER", "auto")
    tags = [t for t in tag.replace(",", "+").split("+") if t]
    known = json.loads((RUNS / "tags.json").read_text())
    missing = [t for t in tags if t not in known]
    if missing:
        sys.exit(f"serve: no checkpoint for {missing} in this image; it holds {sorted(known)}")
    if fam == "auto":
        fam = "ens" if len(tags) > 1 else known[tag]
    env = {"sh30": {"SH30_TAG": tag}, "ap2": {"AP2_TAG": tag}, "ens": {"ENS_TAGS": ",".join(tags)}}.get(fam)
    if env is None:
        sys.exit(f"serve: unknown JEV_DRIVER {fam!r} (sh30 | ap2 | ens | auto)")
    print(f"serve: driver {fam}, tag {tag}, replica {os.environ.get('ALPASIM_CONTESTANT_REPLICA_INDEX', '-')}"
          f" of {os.environ.get('ALPASIM_CONTESTANT_REPLICAS', '-')}", flush=True)
    os.environ.update(env)
    os.execv(sys.executable, [sys.executable, str(LIB / f"{fam}_driver.py")])


if __name__ == "__main__":
    main()
