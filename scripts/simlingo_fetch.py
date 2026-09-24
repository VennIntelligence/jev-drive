"""Fetch the SimLingo release checkpoint and its InternVL2-1B base into the HF cache (see todos/2026-09-25-simlingo-catalogue).

Only what closed-loop eval needs: the hydra config and pytorch_model.pt of RenzKa/simlingo (not the 3.9 GB of
DeepSpeed optimizer shards), plus OpenGVLab/InternVL2-1B, which the agent instantiates with from_pretrained.
Prints the snapshot paths and pinned revisions; the agent's --agent-config is <simlingo snap>/simlingo/checkpoints/
epoch=013.ckpt/pytorch_model.pt, because it reads the config from three directories above the weights.
"""
import argparse
import time
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from jevdrive import hfdl

SIMLINGO = ("RenzKa/simlingo", ("simlingo/.hydra/config.yaml", "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"))
INTERNVL = ("OpenGVLab/InternVL2-1B", ("*",))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--internvl-from-modelscope", action="store_true", help="fetch InternVL2-1B from ModelScope")
    ap.add_argument("--ckpt-url", default="", help="alternative URL for pytorch_model.pt (sha256-checked)")
    ap.add_argument("--streams", type=int, default=1, help="range streams for the 2.6 GB checkpoint (1 = plain GET)")
    args = ap.parse_args()
    if args.ckpt_url:  # pre-fill the checkpoint blob from another host (e.g. huggingface.co behind the proxy)
        e = next(e for e in hfdl.tree(SIMLINGO[0], "simlingo/checkpoints/epoch=013.ckpt") if e.get("lfs"))
        blob = Path(HF_HUB_CACHE) / f"models--{SIMLINGO[0].replace('/', '--')}" / "blobs" / e["lfs"]["oid"]
        t0 = time.monotonic()
        hfdl.download(args.ckpt_url, blob, e["size"], e["lfs"]["oid"], args.streams, chunk=32 << 20, headers={})
        print(f"checkpoint {e['size'] / 1e6:.0f} MB in {time.monotonic() - t0:.0f} s (sha256 ok)", flush=True)
    snap = hfdl.snapshot(*SIMLINGO, streams=args.streams)
    base = hfdl.snapshot(*INTERNVL, streams=8, ms_repo=INTERNVL[0] if args.internvl_from_modelscope else None,
                         absent=("processor_config.json", "preprocessor_config.json", "chat_template.json"))
    print(f"simlingo  {snap}\n  ckpt    {snap}/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt\ninternvl  {base}")
