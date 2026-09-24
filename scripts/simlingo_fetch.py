"""Fetch the SimLingo release checkpoint and its InternVL2-1B base into the HF cache (see todos/2026-09-25-simlingo-catalogue).

Only what closed-loop eval needs: the hydra config and pytorch_model.pt of RenzKa/simlingo (not the 3.9 GB of
DeepSpeed optimizer shards), plus OpenGVLab/InternVL2-1B, which the agent instantiates with from_pretrained.
Prints the snapshot paths and pinned revisions; the agent's --agent-config is <simlingo snap>/simlingo/checkpoints/
epoch=013.ckpt/pytorch_model.pt, because it reads the config from three directories above the weights.
"""
import argparse

from jevdrive import hfdl

SIMLINGO = ("RenzKa/simlingo", ("simlingo/.hydra/config.yaml", "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"))
INTERNVL = ("OpenGVLab/InternVL2-1B", ("*",))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--internvl-from-modelscope", action="store_true", help="fetch InternVL2-1B from ModelScope")
    args = ap.parse_args()
    snap = hfdl.snapshot(*SIMLINGO, streams=1)
    base = hfdl.snapshot(*INTERNVL, streams=8, ms_repo=INTERNVL[0] if args.internvl_from_modelscope else None,
                         absent=("processor_config.json", "preprocessor_config.json", "chat_template.json"))
    print(f"simlingo  {snap}\n  ckpt    {snap}/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt\ninternvl  {base}")
