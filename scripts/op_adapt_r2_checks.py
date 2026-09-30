"""op-adapt round 2, package C numerical checks (op-train venv; one GPU). Every new path against the one it must equal:

  desire    decomposed teacher (op, op_L, op_R from the desire block) vs the queued Cinque graph stepped at 20 Hz with a
            modeld rising-edge pulse (fp32 port both sides; P5 protocol: 5 Hz frames held 4 steps), one Cosmos C+ stream
  frames    read_mp4_every(::4) and the K+ composite vs cosmos_full.load_pair(...)[::4]
  e1        cache-sim trunks vs E1's stored stage-3 maps (runs/op_cosmos_probe/feats.npz) of the same pairs
  nus       teacher op on nuScenes val keyframes vs round 1's evaluated original plan (train-lam10 eval/nusc.npz)
  identity  offset path with e = psi = 0 (cache-off-identity) vs the round-1 navtrain trunk cache of the same tokens

  CUDA_VISIBLE_DEVICES=1 python scripts/op_adapt_r2_checks.py desire frames e1 nus identity
Output: R2/checks/C_<name>.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_r2_data as C  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

E1 = data_dir() / "runs" / "op_cosmos_probe"


def save(name, r):
    (C.root("checks") / f"C_{name}.json").write_text(json.dumps(r, indent=1, default=float))
    print(name, json.dumps(r, default=float))


def first_pairs(n=3):
    return (E1 / "pairs_frozen.txt").read_text().split()[:n]


class DesireStepper(A.Stepper):
    @torch.no_grad()
    def step_d(self, img2, pulse: np.ndarray):
        n = self.net
        f = {"new_img": self._t("new_img", img2), "desire": self._t("desire", pulse),
             "traffic_convention": self._t("traffic_convention", [[1.0, 0.0]]), "action_t": self._t("action_t", [C.AT]), **self.state}
        out = n.run(f, ["outputs"] + ["next_" + k for k in self.state])
        self.state = {k: out["next_" + k] for k in self.state}
        return out["outputs"].float().reshape(-1).cpu().numpy()


def check_desire():
    import op_adapt_r2_cache as RC
    import cosmos_openpilot as CO
    RC._init(CO.maps())
    pair, fr = RC.sim_frames(first_pairs(1)[0])
    frames = fr[0]                                                     # C+ (24, 2, 6, 128, 256)
    net = A.load("cinque", torch.float32).cuda()
    st = DesireStepper(net, "cinque")
    ps = net.slices["plan"]
    # decomposed
    cur = torch.as_tensor(frames).cuda()
    prev = torch.cat([torch.zeros_like(cur[:1]), cur[:-1]])
    T = RC.trunk(net, prev, cur, 24)
    j = np.arange(C.MIN_SLOT, C.NSLOT)
    mu, sd = RC.teach(net, RC.hidden(net, T), C.stride_ctx(j, 1), np.tile([[1.0, 0.0]], (len(j), 1)).astype(np.float32))
    res = {}
    for di, d in enumerate(C.DESIRES):
        for jj in ((9, 16, 23) if d else (None,)):
            st.reset()
            outs = {}
            for s in range(C.NSLOT):
                for k in range(4):
                    pulse = np.zeros(8, np.float32)
                    if d and s == jj and k == 0:
                        pulse[d] = 1
                    o = st.step_d(frames[s], pulse)
                outs[s] = o
            for s in (j if jj is None else [jj]):
                ref = outs[s][ps.start:ps.start + 495].reshape(33, 15)
                rsd = np.exp(outs[s][ps.start + 495:ps.start + 990]).reshape(33, 15)
                k = s - C.MIN_SLOT
                e = np.abs(mu[k, di] - ref)
                res.setdefault(f"d{d}", []).append({"slot": int(s), "pos_max": float(e[:, :3].max()), "all_max": float(e.max()),
                                                    "std_rel_max": float((np.abs(sd[k, di] - rsd) / rsd).max()),
                                                    "y3s_step": float(np.interp(3.0, A.T_IDXS, ref[:, 1])),
                                                    "y3s_dec": float(np.interp(3.0, A.T_IDXS, mu[k, di][:, 1]))})
    summ = {k: {"pos_max": max(r["pos_max"] for r in v), "std_rel_max": max(r["std_rel_max"] for r in v)} for k, v in res.items()}
    summ["pass"] = all(v["pos_max"] <= 1e-3 for v in summ.values())
    save("desire", {"pair": pair, "summary": summ, "rows": res})


def check_frames():
    import op_adapt_r2_cache as RC
    from jevdrive.cosmos_full import load_pair, main_dir
    p = first_pairs(1)[0]
    rep = {}
    for kind in ("carla", "cosmos"):
        a, b = load_pair(p, kind)
        d = main_dir() / "pairs" / p
        pp, mm = RC.read_mp4_every(d / f"{kind}_plus.mp4"), RC.read_mp4_every(d / f"{kind}_minus.mp4")
        if kind == "cosmos":
            n = RC.H_ * RC.W_ // 8
            sup = np.load(d / "gt.npz")["support"]
            s = np.stack([np.unpackbits(sup[t * n:(t + 1) * n]).reshape(RC.H_, RC.W_, 1) for t in range(0, RC.T_, C.STEP)]).astype(bool)
            pp = np.where(s, pp, mm)
        rep[kind] = {"plus_maxdiff": int(np.abs(pp.astype(int) - a[::4]).max()), "minus_maxdiff": int(np.abs(mm.astype(int) - b[::4]).max())}
    rep["pass"] = all(v["plus_maxdiff"] == 0 and v["minus_maxdiff"] == 0 for v in rep.values() if isinstance(v, dict))
    save("frames", {"pair": p, **rep})


def check_e1():
    z = np.load(E1 / "feats.npz", mmap_mode="r")
    names = (E1 / "pairs_frozen.txt").read_text().split()
    rows = []
    for i, p in enumerate(names[:3]):
        f = C.root("cache-sim") / f"{p}.npy"
        if not f.exists():
            continue
        mine = np.load(f).reshape(4, C.NSLOT, 1024, 8, 16).astype(np.float32)
        ref = np.asarray(z["map3"][i]).astype(np.float32)                      # (4, 24, 1024, 8, 16), same stream order
        d = np.abs(mine - ref)
        rows.append({"pair": p, "max": float(d.max()), "mean": float(d.mean()), "ref_p99": float(np.percentile(np.abs(ref), 99)),
                     "corr": float(np.corrcoef(mine.ravel(), ref.ravel())[0, 1])})
    save("e1", {"rows": rows, "pass": bool(rows) and all(r["corr"] > 0.9999 for r in rows)})


def check_nus():
    ev = sorted((data_dir() / "runs/op_adapt/train-lam10").glob("*/eval/nusc.npz"))[-1]
    e = np.load(ev)
    idx = C.load_index("nus")
    t = C.load_teacher("nus")
    tok = pd.Series(idx.token.to_numpy(), index=idx.uid).reindex(t["uid"]).to_numpy()
    pos = pd.Series(np.arange(len(tok)), index=tok)
    k = pos.reindex(e["key"]).to_numpy()
    ok = ~np.isnan(k)
    mine = t["plan_mu"][k[ok].astype(int), 0]
    ref = e["orig_plan"][ok]
    dr = A.plan_drift(mine, ref)
    save("nus", {"eval": str(ev), "n": int(ok.sum()), "drift_median": float(np.median(dr)), "drift_max": float(dr.max()),
                 "all_max": float(np.abs(mine - ref).max()), "pass": float(np.median(dr)) < 1e-3})


def check_identity():
    from jevdrive import op_adapt_data as D
    t = pd.read_parquet(C.root("offset") / "table.parquet")
    nav = C.load_index("nav").set_index("uid")
    f = C.root("cache-off-identity") / "-1.npy"
    T = np.load(f).astype(np.float32)
    ctx = np.load(C.root("cache-off-identity") / "-1.ctx.npy")
    g = t[t.split == "train"].head(len(ctx))
    rows = []
    for (_, r), c in zip(g.iterrows(), ctx):
        n = nav.loc[r.twin_uid]
        z = np.load(data_dir() / n.file)
        ref = z["trunk"][np.asarray(n.ctx)[np.asarray(n.ctx) >= 0]].astype(np.float32)
        mine = T[c[c >= 0]]
        assert (np.asarray(n.ctx) >= 0).sum() == (c >= 0).sum()
        rows.append(float(np.abs(mine - ref).max()))
    save("identity", {"n": len(rows), "max": max(rows), "median_of_max": float(np.median(rows)), "pass": max(rows) < 0.05})


def check_v6img(out="research/results/op-adapt-r2/offset_v6"):
    """V6 visual check: 5 dev samples per corner (the first 5 dev tokens), current frame. Panel: top = original
    openpilot road | wide model frame (luma), bottom = the same after the offset homography."""
    from PIL import Image, ImageDraw
    from jevdrive import navsim_zs as Z
    from jevdrive.openpilot.frames import unpack_luma
    t = pd.read_parquet(C.root("offset") / "table.parquet")
    d = t[t.split == "dev"]
    toks = sorted(d.token.unique())[:5]
    ent = {e["token"]: e for e in Z.load_index("navtrain", slim=True) if e["token"] in set(toks)}
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    files = []
    for _, r in d[d.token.isin(toks)].iterrows():
        cam = ent[r.token]["cams"][-1]["CAM_F0"]
        m0, m1 = C.OffsetMaps(cam, 0.0, 0.0), C.OffsetMaps(cam, r.e, r.psi)
        ycc = m0.decode(cam["path"])
        a, b = m0(ycc), m1(ycc)
        top = np.concatenate([unpack_luma(a[0]), unpack_luma(a[1])], 1)
        bot = np.concatenate([unpack_luma(b[0]), unpack_luma(b[1])], 1)
        im = Image.fromarray(np.concatenate([top, np.full((4, top.shape[1]), 255, np.uint8), bot], 0)).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.text((6, 4), "original (road | wide)", fill=(255, 255, 0))
        dr.text((6, 264), f"offset e={r.e:+.1f} m, psi={r.psi:+.2f} rad (left +)", fill=(255, 255, 0))
        f = o / f"c{int(r.corner)}_{r.token}.webp"
        im.save(f, quality=70)
        files.append(f.name)
    save("v6img", {"files": files, "out": str(o)})


if __name__ == "__main__":
    for w in sys.argv[1:]:
        globals()[f"check_{w}"]()
