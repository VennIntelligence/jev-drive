"""factor_wm p2op static-history probe (no training; results/p2-static-probe.md). Tests decision 146 point 4.

g1-diag's H1 method (fw_diag.h1) on the P2-family policies P2-F-s0 / PC-s0 / PX-s0: at t0 of WOD clips, read the policy once with
  history real   the 10 logged frames (and logged ego history poses, speed, acceleration)
  history static the t0 frame repeated 10 times (HUGSIM's 5 s static warm-up); two variants:
                 static_img  images only (exactly fw_diag.h1), ego inputs unchanged
                 static_full images + ego history of a standing car (poses all 0, v = 0, a = 0)
on clip types launch (g0b launch, 40) and standing-static (g1s stay, 40: the log never moves for 8 s).
Readouts: action[1] (shipped-distilled head in the P2 family), plan speed at 1 s / 3 s, should_stop = v0 < 0.3 and action[1] < 0.1,
should_stop_plan = v0 < 0.3 and (plan speed at 0.5 s - v0) / 0.5 < 0.1 (the plan-driven readout HUGSIM executes).

  $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_p2_static.py
Output: $DATA_DIR/runs/factor_wm/p2op/static_probe.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402
import fw_p2 as P2  # noqa: E402
from fw_diag import _boot  # noqa: E402

ARMS = {"P2": "P2-F-s0", "C": "PC-s0", "X": "PX-s0"}
SETS = {"launch": ("g0b", "launch"), "standing-static": ("g1s", "stay")}
HIST = ("real", "static_img", "static_full")


def probe(R, S, tab, cs):
    import torch
    A = R.A
    n = len(cs)
    tc = S.t["tc"][cs].astype(np.float32)
    out = {}
    for hist in HIST:
        f = np.asarray(S.imgs[cs][:, : C.NH])
        if hist != "real":
            f = np.repeat(f[:, C.T0: C.T0 + 1], C.NH, axis=1)
        f = f[:, 1:]
        H = R.enc(f[:, :-1].reshape(-1, *f.shape[2:]), f[:, 1:].reshape(-1, *f.shape[2:])).reshape(n, P2.NS, 32, 512)
        v0 = S.t["v"][cs, C.T0].astype(float)
        ego = []
        for b, c in enumerate(cs):
            if hist == "static_full":
                ego.append(P2.ego_feat(np.zeros((C.NH, 3)), 0.0, 0.0, tab["cmd"][c, C.T0]))
            else:
                ego.append(P2.ego_feat(np.array(tab["hist"][c]), v0[b], float(tab["alog"][c, C.T0]), tab["cmd"][c, C.T0]))
        v0 = np.zeros(n) if hist == "static_full" else v0
        with torch.no_grad():
            o = R.m(H, torch.from_numpy(np.asarray(ego, np.float32)).to(R.dev), torch.from_numpy(tc).to(R.dev)).float().cpu().numpy()
        acc = o[:, R.m.net.slices["action"].start + 1]
        plan = o[:, R.pi].reshape(-1, 33, 15)
        v = lambda t: np.array([np.interp(t, A.T_IDXS, p[:, 3]) for p in plan])  # noqa: E731
        ap = (v(0.5) - v0) / 0.5
        out[hist] = dict(acc=acc, v1=v(1.0), v3=v(3.0), stop=((v0 < 0.3) & (acc < 0.1)).astype(float), stop_plan=((v0 < 0.3) & (ap < 0.1)).astype(float))
    return out


def main():
    import torch  # noqa: F401
    raw = {}
    groups = {}
    for ty, (name, cat) in SETS.items():
        S = C.Clips(name)
        tab = dict(np.load(P2.P("logged", name) / "tab.npz"))
        cs = [c for c in range(S.n) if S.t["cat"][c] == cat]
        groups[ty] = np.array([str(S.t["seq"][c]) for c in cs])
        for arm, tag in ARMS.items():
            R = P2.Runner(tag, "cuda", 1)
            raw[(arm, ty)] = probe(R, S, tab, cs)
            R.pool.shutdown(wait=False)
            del R
            torch.cuda.empty_cache()
    res = dict(n={ty: len(g) for ty, g in groups.items()}, cells={}, diffs={})
    for (arm, ty), d in raw.items():
        for h, m in d.items():
            res["cells"][f"{arm}/{h}/{ty}"] = {k: float(np.mean(x)) for k, x in m.items()} | {"acc_median": float(np.median(m["acc"])), "v1_median": float(np.median(m["v1"])), "v3_median": float(np.median(m["v3"]))}
    for ty in SETS:
        for h in HIST:
            for ref in ("P2", "C"):
                res["diffs"][f"X-{ref}/{h}/{ty}"] = {k: _boot(raw[("X", ty)][h][k] - raw[(ref, ty)][h][k], groups[ty]) for k in ("acc", "v1", "v3", "stop", "stop_plan")}
    json.dump(dict(res, per_clip={f"{a}/{t}": {h: {k: x.tolist() for k, x in m.items()} for h, m in d.items()} for (a, t), d in raw.items()}),
              open(C.root("p2op") / "static_probe.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
