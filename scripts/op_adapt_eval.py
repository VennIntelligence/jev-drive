"""op-adapt step 3 readouts, feature pass (op-train venv): the original and the adapted Cinque on the cached trunks of
nuScenes val (keyframes), WOD val streams (targets) and P5 v1 BA (targets). Same numeric path for both models
(fp16 frozen parts, the adapted stage 4 from ckpt.pt), so every difference is the adaptation.

  CUDA_VISIBLE_DEVICES=2 python scripts/op_adapt_eval.py --ckpt runs/op_adapt/train-lam1/<ts>/ckpt.pt
Output: <ckpt dir>/eval/<dataset>.npz with, per sample, `key` (token / frame name) and for m in (orig, adapt):
m_temporal (512), m_vision (512), m_plan (33, 15), m_lead (72), m_lead_prob (3); adapt_head_t / adapt_head_v (3).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_data as D  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

AT = (0.275, 0.525)


def streams(ds):
    """(file, sample slots, keys) per cached stream of a dataset."""
    import pandas as pd
    if ds == "nusc":
        lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
        for s in sorted(lab[lab.split == "val"].scene.unique()):
            f = D.root("nusc") / f"{s}.npz"
            z = np.load(f)
            yield f, z["key_slot"], z["tokens"]
    else:
        for f in sorted(D.root(ds).glob("*.npz")):
            if f.stem.endswith(".tmp"):
                continue
            z = np.load(f)
            t = z["targets"]
            yield f, t, z["names"][t]


@torch.no_grad()
def run(nets, heads, f, slots, dev, bs=256):
    z = np.load(f)
    T = torch.as_tensor(z["trunk"]).to(dev)
    c = int(z["stride"])
    off = c * np.arange(-(A.CONTEXT - 1), 1)
    tc = torch.as_tensor(z["traffic"], device=dev)
    sl = nets["orig"].slices
    lead = np.arange(sl["lead"].start, sl["lead"].start + 72)
    lp = np.arange(sl["lead_prob"].start, sl["lead_prob"].stop)
    pidx = A.plan_index(sl)
    out = {}
    for i in range(0, len(slots), bs):
        j = np.asarray(slots[i:i + bs])
        loc = j[:, None] + off[None]
        valid = torch.as_tensor(loc >= 0, device=dev)
        x = T[torch.as_tensor(loc.clip(0), device=dev)]
        t2 = tc[None].expand(len(j), 2)
        for m, net in nets.items():
            o = A.stage4_policy(net, x, AT, t2, valid)
            r = o["outputs"].float()
            res = {"temporal": o["select_4"].float(), "vision": o["mean"].float(), "plan": r[:, pidx].view(-1, 33, 15),
                   "lead": r[:, lead], "lead_prob": r[:, lp]}
            if m == "adapt" and heads is not None:
                ha, hb = heads(o["select_4"], o["tokens"])
                res |= {"head_t": ha.float(), "head_v": hb.float()}
            for k, v in res.items():
                out.setdefault(f"{m}_{k}", []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="+", default=["nusc", "wod", "p5"])
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    dt = {"fp16": torch.float16, "tf32": torch.float32, "bf16": torch.bfloat16}[ck["args"]["dtype"]]
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = ck["args"]["dtype"] == "tf32"
    dev = torch.device("cuda")
    orig = A.load("cinque", dt).to(dev).eval()
    adapt = A.load("cinque", dt, trainable=A.stage4_weights()).to(dev).eval()
    for k, v in ck["stage4"].items():
        adapt.params[k].data.copy_(v)
    heads = A.AuxHeads().to(dev).eval()
    heads.load_state_dict(ck["heads"])
    outdir = Path(a.ckpt).parent / "eval"
    outdir.mkdir(exist_ok=True)
    log = RunLog("op_adapt", "eval")
    log.info(f"ckpt {a.ckpt} -> {outdir}")
    for ds in a.datasets:
        t0, acc, keys = time.time(), {}, []
        for f, slots, k in streams(ds):
            if not len(slots):
                continue
            r = run({"orig": orig, "adapt": adapt}, heads, f, slots, dev)
            for n, v in r.items():
                acc.setdefault(n, []).append(v)
            keys.append(np.asarray(k).astype(str))
        res = {n: np.concatenate(v).astype(np.float32) for n, v in acc.items()}
        np.savez(outdir / f"{ds}.npz", key=np.concatenate(keys), **res)
        log.info(f"{ds}: {len(res['orig_temporal'])} samples in {time.time() - t0:.0f} s")
    log.event("end")


if __name__ == "__main__":
    main()
