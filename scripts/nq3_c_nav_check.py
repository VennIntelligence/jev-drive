"""Night queue 3, lane C-nav: rule-8 check of the P6 frame source for the four NAVSIM-trained examinees
(todos/2026-09-26-night-queue-3.md, Q1 [C-nav] entries). Only the frame source is new, so the check is: on N P6 frames,
the new p6 plan / request path gives the same tensors and trajectories as a direct construction from the same JPEGs
through the existing p5 path.

  direct  (jevdrive env) N frames picked with rng 0 -> scratch/direct.npz: the three current JPEGs and the NAVSIM ego of
          each frame looked up by frame_name in the P6 index (T1), and T2 requests for exactly these rows via req_carla
          (the P5 request path) next to the matching rows sliced from the full p6 request files
  t1      (model env, model repo cwd) p6 plan rows + p6 maps + render_fast  vs  direct files / ego + the P5 plan's rig and
          P5 maps + navsim_rig.render (the unoptimised render): images, feature tensors, trajectories
  sanity  (jevdrive env) every finished p6 output: forward share, 2 s mean speed vs the ego speed in past.npy
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ["DATA_DIR"])
S = DATA / "runs/nq3/c/q1_nav/check"
P6 = DATA / "processed/carla_p6"


def pick(names, n):
    return np.sort(np.random.default_rng(0).choice(len(names), n, replace=False))


def direct(n: int):
    import pandas as pd
    from jevdrive import top10_t2 as T2
    from jevdrive.night2_n3 import nav_ego
    S.mkdir(parents=True, exist_ok=True)
    plan = json.loads((DATA / "processed/top10_exam/p6/plan.json").read_text())
    names = np.asarray(plan["frames"]["frame_name"])[pick(plan["frames"]["frame_name"], n)]
    t = pd.read_parquet(P6 / "index.parquet")
    past = np.load(P6 / "past.npy", mmap_mode="r")
    rows = np.flatnonzero(t.frame_name.isin(set(names)).to_numpy())
    tr = t.iloc[rows]
    files = np.array([[f[3], f[7], f[11]] for f in tr.files], object)
    root = T2.root
    T2.root = lambda *p: (S / "t2" / Path(*p), (S / "t2" / Path(*p)).parent.mkdir(parents=True, exist_ok=True))[0]
    try:
        T2.req_carla("direct", t, past, rows)
    finally:
        T2.root = root
    same = {}
    for m in T2.MODELS:
        full = dict(np.load(T2.root("requests", f"p6_{m}.npz")))
        at = {k: i for i, k in enumerate(full["keys"])}
        sub = {k: v[[at[x] for x in tr.frame_name]] for k, v in full.items()}
        np.savez(S / "t2" / f"new_{m}.npz", **sub)
        d = np.load(S / "t2/requests" / f"direct_{m}.npz")
        same[m] = {k: bool(np.array_equal(d[k], sub[k])) for k in d.files}
    np.savez(S / "direct.npz", frame_name=tr.frame_name.to_numpy().astype(str), files=files.astype(str),
             ego=nav_ego(past[rows], tr.intent.to_numpy()))
    print(json.dumps({"n": len(rows), "t2_request_rows_identical": same}, indent=1))
    (S / "t2_requests.json").write_text(json.dumps(same, indent=1))


def _flat(x, pre=""):
    import torch
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return {pre: np.asarray(x.cpu() if torch.is_tensor(x) else x, np.float64)}
    if isinstance(x, dict):
        return {k2: v2 for k, v in x.items() for k2, v2 in _flat(v, f"{pre}/{k}").items()}
    if isinstance(x, (list, tuple)):
        return {k2: v2 for i, v in enumerate(x) for k2, v2 in _flat(v, f"{pre}/{i}").items()}
    return {}


def t1(model: str):
    import torch
    from jevdrive import navsim_rig as R
    spec = importlib.util.spec_from_file_location("infer", REPO / "scripts/top10_exam_infer.py")
    I = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(I)
    M = getattr(I, model)()
    p6 = json.loads((DATA / "processed/top10_exam/p6/plan.json").read_text())
    p5 = json.loads((DATA / "processed/top10_exam/p5/plan.json").read_text())
    I.maps_path("p6", "x").parent.mkdir(parents=True, exist_ok=True)
    for k, r in p6["rigs"].items():
        I.make_maps(("p6", k, r))
    d = np.load(S / "direct.npz")
    at = {k: i for i, k in enumerate(p6["frames"]["frame_name"])}
    ds = I.Frames(p6, M.feats)
    new = [ds[at[f]][1] for f in d["frame_name"]]
    cams5 = dict(zip(R.NAMES, [{n: np.asarray(v) for n, v in c.items()} for c in p5["rigs"]["p5"]["virt"]]))
    mp5 = I.load_maps("p5", "p5")
    old, dimg = [], 0
    for i, f in enumerate(d["frame_name"]):
        src = [I.read_rgb(x) for x in d["files"][i]]
        ref = R.render(src, mp5)
        got = R.render_fast(src, I.load_fast("p6", p6["frames"]["rig"][at[f]], tuple((x.shape[1], x.shape[0]) for x in src)))
        dimg = max(dimg, max(int(np.abs(a.astype(int) - b).max()) for a, b in zip(ref, got)))
        old.append(M.feats(dict(zip(R.NAMES, ref)), cams5, d["ego"][i].astype(np.float32)))
    ego_diff = float(np.abs(np.asarray([p6["frames"]["ego"][at[f]] for f in d["frame_name"]]) - d["ego"]).max())
    dfeat = max(float(np.abs(a - b).max()) for x, y in zip(new, old)
                for a, b in zip(_flat(x).values(), _flat(y).values()))
    dev = torch.device("cuda")
    M.agent.to(dev).eval()
    def fwd(xs):                                  # batches of 16: SparseDriveV2's kernel rejects 32
        with torch.no_grad():
            r = [M.forward(I.to_dev(M.collate(xs[k:k + 16]), dev)) for k in range(0, len(xs), 16)]
        return (np.concatenate([a.float().cpu().numpy() for a, _ in r]),
                None if r[0][1] is None else torch.cat([b.cpu() for _, b in r]))
    (tn, sn), (to, so) = fwd(new), fwd(old)
    res = {"model": model, "n": len(new), "max_abs_image_diff": dimg, "max_abs_ego_diff": ego_diff,
           "max_abs_feature_diff": dfeat, "max_abs_traj_diff_m": float(np.abs(tn - to)[..., :2].max()),
           "same_selected": None if sn is None else bool((sn == so).all().item())}
    print(json.dumps(res, indent=1))
    (S / f"t1_{model}.json").write_text(json.dumps(res, indent=1))


def sanity():
    import pandas as pd
    t = pd.read_parquet(P6 / "index.parquet", columns=["frame_name"])
    past = np.load(P6 / "past.npy", mmap_mode="r")
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    out = {}
    for m in ("sparsedrivev2", "ztrs", "drivor", "wajepa"):
        f = DATA / "processed/top10_exam/p6" / f"{m}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        g = z["grid"]
        v0 = np.linalg.norm(past[pos[z["frame_name"].astype(str)].to_numpy(), -1, 2:4], axis=-1)
        v2 = np.linalg.norm(g[:, 7], axis=-1) / 2.0                     # mean speed over 0-2 s
        mv = v0 > 2.0
        out[m] = {"n": len(g), "finite_to_4s": float(np.isfinite(g[:, :16]).all(axis=(1, 2)).mean()),
                  "forward_3s": float((g[:, 11, 0] > 0.5 * v0 * 3.0 - 1.0).mean()),
                  "x3_gt_0_moving": float((g[mv, 11, 0] > 0).mean()), "median_v2_over_v0_moving": float(np.median(v2[mv] / v0[mv])),
                  "corr_v2_v0": float(np.corrcoef(v2, v0)[0, 1]), "median_abs_y3_m": float(np.median(np.abs(g[:, 11, 1])))}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("direct", "t1", "sanity"))
    ap.add_argument("--model", default="sparsedrivev2")
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    {"direct": lambda: direct(a.n), "t1": lambda: t1(a.model), "sanity": sanity}[a.cmd]()
