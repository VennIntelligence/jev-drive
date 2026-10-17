"""Entry point of the submission image: choose the driver family and checkpoint at run time, then become that driver's process.

  JEV_TAG     run tag under $DATA_DIR/runs/op_parity/runs (baked in by build.sh); `tagA+tagB` serves the adapter ensemble
  JEV_DRIVER  sh30 | ap2 | ens | pai | auto (default). auto = the nuPlan family recorded for the tag in tags.json (verify.py, at build
              time): checkpoints trained on the AlpaSim input standard -> ap2_driver.py, every other op_parity tag -> sh30_driver.py.
              pai = the PAI-track driver (pai_driver.py, SH30_TAG = the tag); never chosen by auto, a PAI image bakes it in (build.sh TRACK=pai)
  JEV_HOLD    the serve-side static hold (lib/static_hold.py; build.sh HOLD): set and not off, the sh30 family runs lib/sh30_hold.py
              (sh30_driver.py wrapped, the box's `sh30hold` driver) instead of sh30_driver.py; any other family refuses to start
Everything else (SH30_COLD, AP2_COLD, SH30_MOTION, PAI_*, the serving switches JEV_VCONT / JEV_LEAD / JEV_VHIST / JEV_ACC / JEV_GAP /
JEV_TRACK / JEV_HOLD, ALPASIM_DRIVER_*) passes through to the driver unchanged.
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
    if fam == "pai" and len(tags) > 1:
        sys.exit(f"serve: the PAI driver serves one checkpoint, not {tags}")
    env = {"sh30": {"SH30_TAG": tag}, "ap2": {"AP2_TAG": tag}, "ens": {"ENS_TAGS": ",".join(tags)}, "pai": {"SH30_TAG": tag}}.get(fam)
    if env is None:
        sys.exit(f"serve: unknown JEV_DRIVER {fam!r} (sh30 | ap2 | ens | pai | auto)")
    hold = os.environ.get("JEV_HOLD", "").strip().lower() not in ("", "off")
    if hold and fam != "sh30":
        sys.exit(f"serve: JEV_HOLD={os.environ['JEV_HOLD']} is served by the sh30 driver only, not {fam!r}")
    entry = "sh30_hold.py" if hold else f"{fam}_driver.py"
    print(f"serve: driver {fam} ({entry}), tag {tag}, replica {os.environ.get('ALPASIM_CONTESTANT_REPLICA_INDEX', '-')}"
          f" of {os.environ.get('ALPASIM_CONTESTANT_REPLICAS', '-')}; switches "
          + " ".join(f"{k}={os.environ[k]}" for k in ("JEV_VCONT", "JEV_LEAD", "JEV_VHIST", "JEV_ACC", "JEV_GAP", "JEV_TRACK", "JEV_HOLD")
                     if k in os.environ), flush=True)
    os.environ.update(env)
    os.execv(sys.executable, [sys.executable, str(LIB / entry)])


if __name__ == "__main__":
    main()
