"""op_probe reference features: WA-JEPA's released checkpoint (plans/2026-10-06-dac-localize-prereg.md section 1), batched bf16 forward.

  cd $DATA_DIR/third_party/wajepa && PYTHONPATH=$DATA_DIR/third_party/wajepa:$DATA_DIR/third_party/navsim \
    $DATA_DIR/envs/wajepa/bin/python <repo>/experiments/op_probe/scripts/opb_wajepa.py --data lb_navtest [--limit N] [--rows-file f.npy]

Requests are built per token from the OpenScene log (CAM_L0 / F0 / R0 / B0 of the 4 history frames, time-major, as
experiments/top10/lib/top10_t2/wajepa_run.py) and from op_parity's tab.npz (4-pose history, t0 velocity / acceleration, command), i.e. the
inputs of pp_navhard.py req. The model runs its own predict_trajectory on a batch (bf16 autocast = its HUGSIM adapter precision; the flow noise
of a batch differs from batch-1 calls, which only matters for the plan, not for the encoder features). Taps:
  C   context_scene (V-JEPA 2.1 encoder + scene projector, history tokens) mean-pooled to 32 contiguous token groups x 512
  T   input of predictor.traj_out at the last flow step (8 trajectory tokens x 512)
  H   input of the final Linear of traj_out at the last flow step (8 x 512), the analogue of Cinque's add_54
  traj its plan (8 x 3)
Out: $DATA_DIR/runs/op_probe/feats/WA/<data>.npz (float16 features).
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "experiments/top10/lib/top10_t2"), str(REPO)]
CAMS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")
LOGDIR = {True: "test", False: "trainval"}
N_GROUPS = 32


def requests(data, rows):
    """img (n, 16) paths, hist (n, 4, 3), ego (n, 4), cmd (n,) for the rows of an op_parity cache."""
    tab = np.load(D / "runs/op_parity/cache" / data / "tab.npz")
    test = "navtest" in data
    names, logs = tab["names"][rows], tab["log"][rows]
    img = []
    cache = {}
    for t, lg in zip(names, logs):
        if lg not in cache:
            fr = pickle.load(open(D / "datasets/navsim/navsim_logs" / LOGDIR[test] / f"{lg}.pkl", "rb"))
            cache = {lg: (fr, {f["token"]: i for i, f in enumerate(fr)})}
        fr, at = cache[lg]
        i = at[t]
        root = D / "datasets/navsim/sensor_blobs" / LOGDIR[test]
        img.append([str(root / fr[j]["cams"][c]["data_path"]) if fr[j]["cams"][c]["data_path"] else "" for j in range(i - 3, i + 1) for c in CAMS])
    hist = tab["pose"][rows].astype(np.float32)
    ego = np.concatenate([tab["vel"][rows][:, -1], tab["acc"][rows][:, -1]], -1).astype(np.float32)
    c = tab["cmd"][rows][:, -1]
    cmd = np.where(c.sum(1) > 0, c.argmax(1), 3)
    assert np.allclose(hist[:, -1], 0, atol=1e-4)
    return dict(keys=names, img=np.array(img), hist=hist, ego=ego, cmd=cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rows-file", default="")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from jevdrive.run import Run
    with Run("op_probe", f"wajepa-{a.data}{a.tag}", config=vars(a)) as run:
        msg = extract(a)
        run.summary["result"] = msg


def extract(a):
    import wajepa_run as WR
    n_all = len(np.load(D / "runs/op_parity/cache" / a.data / "tab.npz")["names"])
    rows = np.load(a.rows_file) if a.rows_file else np.arange(n_all)[: a.limit or None]
    t0 = time.time()
    z = requests(a.data, rows)
    print(f"{len(rows)} requests in {time.time() - t0:.0f} s", flush=True)
    agent = WR.build_agent()
    model = agent.model
    cap = {}
    orig = model._encode_scene_context

    def enc(*args, **kw):
        out = orig(*args, **kw)
        cap["C"] = out
        return out
    model._encode_scene_context = enc
    to = model.predictor.traj_out
    to.register_forward_pre_hook(lambda m, x: cap.__setitem__("T", x[0]))
    to[3].register_forward_pre_hook(lambda m, x: cap.__setitem__("H", x[0]))
    dl = torch.utils.data.DataLoader(WR.Req(z, np.arange(len(rows))), batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                                     prefetch_factor=4)
    C, T, H, traj = [], [], [], []
    t1, k = time.time(), 0
    with torch.no_grad():
        for f in dl:
            f = {kk: v.cuda(non_blocking=True) for kk, v in f.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tr = model.predict_trajectory(f).float()
            c = cap["C"].float()
            B, N, Dd = c.shape
            g = torch.tensor_split(torch.arange(N, device=c.device), N_GROUPS)
            C.append(torch.stack([c[:, ix].mean(1) for ix in g], 1).half().cpu().numpy())
            T.append(cap["T"].float().half().cpu().numpy())
            H.append(cap["H"].float().half().cpu().numpy())
            traj.append(tr.cpu().numpy())
            k += B
            if k == B:
                print(f"shapes: context {tuple(cap['C'].shape)}, T {tuple(cap['T'].shape)}, H {tuple(cap['H'].shape)}, traj {tuple(tr.shape)}", flush=True)
            if (k // B) % 50 == 0:
                print(f"{k}/{len(rows)} {k / (time.time() - t1):.2f} samples/s", flush=True)
    out = D / "runs/op_probe/feats/WA"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.data}{a.tag}{'-lim' if a.limit else ''}.npz"
    np.savez(f, tokens=z["keys"], rows=rows, C=np.concatenate(C), T=np.concatenate(T), H=np.concatenate(H), traj=np.concatenate(traj))
    msg = f"{len(rows)} tokens in {time.time() - t1:.0f} s ({len(rows) / (time.time() - t1):.2f}/s) -> {f}"
    if "navtest" in a.data:                     # against its stored navtest export (the run that reproduced 91.71)
        st = pickle.load(open(D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl", "rb"))["trajectories"]
        tj = np.concatenate(traj)
        ade = [np.linalg.norm(tj[i, :, :2] - np.asarray(getattr(st[t], "poses", st[t]))[:, :2], axis=-1).mean() for i, t in enumerate(z["keys"]) if t in st]
        msg += f"; ADE to stored plans mean {np.mean(ade):.3f} m, p95 {np.percentile(ade, 95):.3f} m (n {len(ade)})"
    print(msg, flush=True)
    (out / f"{f.stem}.done").write_text(msg + "\n")
    return msg


if __name__ == "__main__":
    main()
