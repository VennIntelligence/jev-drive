"""Offline action-head probes for the joint action arms (plans/2026-10-07-joint-action-prereg.md), on the dev rows of the training shards.

Per tag (op_parity checkpoint), moving rows (v0 > 3 m/s), curvatures in the left-positive sign (1/m):
  gain       slope of the commanded curvature k_cmd = -action[0] / max(1, v0)^2 on the logged curvature at t + 0.275 s (pp_train.log_curv `log`),
             all moving rows / turning rows (|k_log| > 0.02) / 3-8 m/s / > 8 m/s; shipped Cinque sits at ~0.63 on navtrain
  plan       slope and correlation of k_cmd on the model's own plan curvature over 0.5-1.5 s (what HUGSIM spec_plan_smooth steers with)
  hist       feedback gain on the vehicle's own past curvature: the 4 history poses (and ay) are bent as if the car had driven an extra
             curvature d over the last 1.5 s (yaw d s, y d s^2 / 2, ay d v^2), g_h = d k_cmd / d (central difference, d = 0.01 1/m); same for
             the plan curvature. g_h >= 1 means the command perpetuates whatever the car did (an integrator in the loop)
  latdrop    slope of k_cmd(vy = ay = 0) on k_cmd(as logged): what the head does with HUGSIM's missing lateral components

  python experiments/op_parity/scripts/pp_joint_probe.py --tags HP-F-s0 JC-F-s0 --out experiments/op_parity/results/joint_action/probe_s0.json
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/op_parity/scripts"), str(_R), str(_R / "lib")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pp_train as T  # noqa: E402

SHARDS = ("navtrain_full.s0of12", "navtrain_full.s1of12")
D = 0.01


def bend(ego: torch.Tensor, v: torch.Tensor, d: float) -> torch.Tensor:
    """Add a constant extra curvature d (left +) to the last 1.5 s of the history in parity_adapter.ego_features (poses at -1.5/-1/-0.5/0 s)."""
    e = ego.clone()
    s = v[:, None] * torch.tensor([-1.5, -1.0, -0.5, 0.0], device=ego.device)            # arc length to each history pose (negative)
    e[:, 9:20:3] += d * s * s / 2 / 10.0                                                  # y / 10
    e[:, 10:20:3] += d * s                                                                # yaw
    e[:, 7] += d * v * v / 3.0                                                            # ay / 3
    return e


def slope(a, b):
    return float(a @ b / max(b @ b, 1e-12))


@torch.no_grad()
def probe(tag: str, S, rows: np.ndarray, Wk, bs=256) -> dict:
    m = T.load_pmodel(tag, S.ego.device)
    pi = torch.as_tensor(S.pi, device=S.ego.device)
    acc = {k: [] for k in ("cmd", "pk", "v", "klog", "ok", "cmd_p", "cmd_m", "pk_p", "pk_m", "cmd_z")}
    for i in range(0, len(rows), bs):
        r = torch.as_tensor(rows[i:i + bs], device=S.ego.device)
        v = S.v0[r]
        def run(ego):  # noqa: E306
            o = m(S.front[r], ego, S.tc[r]).float()
            return -o[:, T.ACT_COL] / v.clamp_min(1) ** 2, -T.plan_curv(o[:, pi].view(-1, 33, 15), Wk)
        c, pk = run(S.ego[r])
        cp, pkp = run(bend(S.ego[r], v, D))
        cm, pkm = run(bend(S.ego[r], v, -D))
        ez = S.ego[r].clone()
        ez[:, T.EGO_LAT] = 0
        cz, _ = run(ez)
        for k, x in zip(acc, (c, pk, v, -S.klab[r], S.klab_ok[r], cp, cm, pkp, pkm, cz)):
            acc[k].append(x.cpu().numpy())
    a = {k: np.concatenate(x).astype(np.float64) for k, x in acc.items()}
    mv = (a["v"] > 3) & a["ok"].astype(bool)
    c, kl, pk = a["cmd"][mv], a["klog"][mv], a["pk"][mv]
    out = {"n": int(mv.sum())}
    for nm, sel in (("all", np.ones_like(kl, bool)), ("turn", np.abs(kl) > 0.02), ("v3_8", a["v"][mv] <= 8), ("v8", a["v"][mv] > 8)):
        out[f"gain_{nm}"] = slope(c[sel], kl[sel])
        out[f"n_{nm}"] = int(sel.sum())
    out["plan_gain_log"] = slope(pk, kl)
    out["cmd_on_plan"] = slope(c, pk)
    out["cmd_plan_corr"] = float(np.corrcoef(c, pk)[0, 1])
    out["cmd_mean_abs"], out["plan_mean_abs"], out["log_mean_abs"] = float(np.abs(c).mean()), float(np.abs(pk).mean()), float(np.abs(kl).mean())
    out["g_hist_cmd"] = float(np.mean((a["cmd_p"] - a["cmd_m"])[mv]) / (2 * D))
    out["g_hist_plan"] = float(np.mean((a["pk_p"] - a["pk_m"])[mv]) / (2 * D))
    out["latdrop_slope"] = slope(a["cmd_z"][mv], c)
    return out


def main(a):
    from jevdrive.run import Run
    dev = torch.device("cuda")
    tabs = np.concatenate([np.load(T.data_dir() / "runs" / "op_parity" / "cache" / d / "tab.npz")["names"] for d in SHARDS])
    _, dv, sp = T.split_rows({"names": tabs}, a.split)
    S = T.Store(SHARDS, dev, need_side=False, frames="warp", host=True)
    S.act_labels("log")
    Wk = torch.as_tensor(T.R2.t_weights(np.asarray(T.ACT_WIN)), device=dev)
    res = {}
    with Run("op_parity", "joint-probe", config=vars(a)) as run:
        run.use_split(sp[1])
        for tag in a.tags:
            res[tag] = probe(tag, S, dv, Wk)
            run.info(f"{tag}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res[tag].items()))
        run.summary.update(res)
    out = _pl.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps(old | res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--split", default="navsim/op-parity-full")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
