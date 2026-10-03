"""Image-command Q3 after the course change: sky-arrow fine-tune (op-train venv, one GPU).
Plan: ../plans/2026-10-04-img-cmd-ft2-prereg.md (addendum). Port, trainable set, loss and loop as img2_train.py; banks from img3_bank.py
(skytrain, skycarla) and img2_bank.py (dist).

Rows of one batch (56):
  I 24   sky arrow for command c on a junction frame (nav 16, CARLA 8). Residual target = the original's own `none` plan +
         [C_c(s(t)) - C_o(s(t))], C_o = the branch centreline nearest the original's 4 s point (the branch it already takes), s(t) =
         the original plan's arc length: a command for the branch the original already takes leaves its plan unchanged.
  D 24   no overlay, distilled to the original: nav junction 6, CARLA junction 4, nav straight 2, L3 nav 4 / WOD 4 / CARLA p6 4
  N  8   straight arrow on lane-keeping frames: plan consistency to the original's `none` plan
The sky_disc / sky_wrong controls are never trained.

  teacher                     the original on the `none` rows of skytrain / skycarla -> ft/teacher2/<bank>.npz (dist: img2_train's)
  train --arm SA|SB|SC [--seed s] -> ft/runs/<arm>-s<seed>/ckpt-final.pt
"""
import sys as _sys, pathlib as _pl, os as _os  # noqa: E401
_R = _pl.Path(_os.environ.get("JEV_REPO", _pl.Path(__file__).resolve().parents[3]))
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_R / "experiments" / "op_adapt_h" / "scripts"), str(_pl.Path(__file__).resolve().parent)]
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import img2_train as Q2  # noqa: E402
import img_overlay as O  # noqa: E402
from img_report import near  # noqa: E402

FT = Q2.FT
SG = _os.environ.get("IMG_ARM", "") == "sg"          # the sky + green-line arm (banks sg*, family sg)
BANKS = ("sgtrain", "sgcarla", "dist") if SG else ("skytrain", "skycarla", "dist")
TB, CB, FAM = BANKS[0], BANKS[1], ("sg" if SG else "sky")
MIX = (("IN", "I", 16), ("IC", "I", 8),
       ("QJ", "D", 6), ("CJ", "D", 4), ("QS", "D", 2), ("DN", "D", 4), ("DW", "D", 4), ("DC", "D", 4),
       ("NS", "N", 8))
ARMS = {"smoke": dict(steps=40, ckpt_every=10 ** 9), "SA": dict(), "SB": dict(lam_d=30.0, lam_c=3.0), "SC": dict(lr=5e-5, steps=2000)}
ARMS |= {"G" + k[1]: v for k, v in ARMS.items() if k.startswith("S")} | {"gsmoke": ARMS["smoke"]}   # sg arm: same configs


def residual_target(s, c, o20):
    """(20, 2): the original's plan o20 shifted by the offset between branch c's centreline and the centreline of the branch it takes."""
    cls = sorted({b["cls"] for b in s["branches"]})
    Q = {k: O.full_centre(s, O.cmd_path(s, k)) for k in cls}
    o = min(cls, key=lambda k: near(Q[k], o20[15])[0])
    if c == o:
        return o20.copy()
    P = np.r_[np.zeros((1, 2)), o20]
    sl = np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))

    def along(Qk):
        t = Qk[-1] - Qk[-2]
        Qk = np.r_[Qk, Qk[-1] + np.outer(0.5 * np.arange(1, 401), t / np.linalg.norm(t))]
        a = O.arc(Qk)
        k0 = a[int(np.argmin(np.linalg.norm(Qk, axis=1)))]
        return O.at(Qk, k0 + sl)
    return (o20 + along(Q[c]) - along(Q[o])).astype(np.float32)


class Pool(Q2.Pool):
    def __init__(self, cfg):
        import img2_bank as QB
        self.B = {b: Q2.bank(b) for b in BANKS}
        self.tea = {b: dict(np.load(FT / "teacher2" / f"{b}.npz")) for b in BANKS}
        self.none_row = {str(self.B[b][1]["token"][r]): (b, k) for b in BANKS for k, r in enumerate(self.tea[b]["rows"])}
        G = {s["token"]: s for s in Q2.FB.samples("train")}
        G |= {s["token"]: s for s in QB.carla_samples()}
        vt, vc, vd = self.B[TB][1], self.B[CB][1], self.B["dist"][1]
        tr = vt["split"] == "train"
        jt, st = vt["kind"] == "junction", vt["kind"] == "straight"
        ct = vc["split"] == "train"
        self.idx = {"IN": (TB, np.flatnonzero(tr & jt & (vt["fam"] == FAM))),
                    "IC": (CB, np.flatnonzero(ct & (vc["fam"] == FAM))),
                    "QJ": (TB, np.flatnonzero(tr & jt & (vt["fam"] == "none"))),
                    "CJ": (CB, np.flatnonzero(ct & (vc["fam"] == "none"))),
                    "QS": (TB, np.flatnonzero(tr & st & (vt["fam"] == "none"))),
                    "DN": ("dist", np.flatnonzero((vd["split"] == "train") & (vd["dom"] == "nav"))),
                    "DW": ("dist", np.flatnonzero((vd["split"] == "train") & (vd["dom"] == "wod"))),
                    "DC": ("dist", np.flatnonzero((vd["split"] == "train") & (vd["dom"] == "carla"))),
                    "NS": (TB, np.flatnonzero(tr & st & (vt["fam"] == FAM)))}
        for k, (b, ix) in self.idx.items():
            assert len(ix), k
        self.hum = {}
        for key in ("IN", "IC"):
            b, ix = self.idx[key]
            vb = self.B[b][1]
            H = np.full((len(vb["fam"]), 16, 3), np.nan, np.float32)
            for j in ix:
                t, c = str(vb["token"][j]), str(vb["cmd"][j])
                nb, k = self.none_row[t]
                o20 = Q2.o_timing(self.tea[nb]["mu"][k], np.asarray(self.B[nb][1]["cam"][self.tea[nb]["rows"][k]], float).reshape(-1))
                H[j] = Q2.L.human_targets(residual_target(G[t], c, o20)[None])[0]
            self.hum[b] = H


def cmd_teacher(a):

    dev = torch.device("cuda")
    m = Q2.L.load_model(None, dev)
    didx, pi = Q2.A.distill_index(m.net.slices), Q2.A.plan_index(m.net.slices)
    for b in BANKS:
        p = FT / "teacher2" / f"{b}.npz"
        if p.exists():
            continue
        T, v = Q2.bank(b)
        nn = np.flatnonzero(v["fam"] == "none")
        out = np.zeros((len(nn), len(didx)), np.float16)
        mu = np.zeros((len(nn), 33, 15), np.float32)
        with torch.no_grad():
            for i in range(0, len(nn), 64):
                j = nn[i:i + 64]
                o = m(torch.from_numpy(np.stack([T[x] for x in j])).to(dev), torch.from_numpy(v["slot_valid"][j]).to(dev),
                      torch.from_numpy(np.asarray(v["tc"][j], np.float32)).to(dev).half())["outputs"].float()
                out[i:i + len(j)] = o[:, didx].cpu().numpy()
                mu[i:i + len(j)] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
        np.savez(p, rows=nn, out=out, mu=mu)
        print(f"teacher {b}: {len(nn)} none rows", flush=True)
    P = Pool(Q2.QCfg("teacher"))
    print("pools", {k: (b, len(ix)) for k, (b, ix) in P.idx.items()}, flush=True)
    for b, H in P.hum.items():
        h = H[~np.isnan(H[:, 0, 0])]
        print(f"I targets {b}: n {len(h)}, 4 s x median {np.median(h[:, 15, 0]):.2f} m, |y| median {np.median(np.abs(h[:, 15, 1])):.2f} m")


def cmd_train(a):
    Q2.Pool, Q2.MIX, Q2.ARMS, Q2.BANKS = Pool, MIX, ARMS, BANKS        # img2_train's loop and Batcher on this lane's pools
    Q2.cmd_train(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("teacher")
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", default="")
    p.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "train": cmd_train}[a.cmd](a)
