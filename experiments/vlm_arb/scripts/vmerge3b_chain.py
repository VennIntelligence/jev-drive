"""Second worker lane for the vmerge3 batch (card 1): a subset of vmerge3_chain's units, same env, same output dirs.

  python -m jevdrive.cl run experiments/vlm_arb/scripts/vmerge3b_chain.py --lane vlm-vmerge3b --workers-per-card 6 \
      --arg stage=all --arg seeds=0,1 --arg ports=8211 --arg only=vm3norel-s1-q0,...

Behaviour is identical to vmerge3_chain (same D8 fixed-latency rule, one Qwen server per card). Only the lane name / root differ;
smoke deps and the report tool are dropped (lane vlm-vmerge3 owns them). Units in `only` must not also be run by the main lane at the same time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vmerge3_chain as m3  # noqa: E402

NAME, ROOT = "vlm-vmerge3b", "vlm_arb_vmerge3b"


def jobs(args):
    keep = set(args["only"].split(","))
    out = [j for j in m3.jobs(args) if j.name in keep]
    for j in out:
        j.deps = ()
    return out
