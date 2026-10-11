"""LONG-PROBE input-ladder head: BODY1 arm S0's `step` / `blind` head (experiments/body1/lib/contact_head.build, imported, its modules used
unchanged) with three longitudinal logits added to its per-query output and the rung's extra inputs.

Rungs (cumulative):
  blind      ego vector + query poses (S0 arch `blind`, the floor)
  vis        + the frozen Cinque `view_39` tokens of the 8 slots (S0 arch `step`, the S0 input)
  nat        + openpilot's native outputs of the student that made the query (lead, lead_prob, meta, action; the policy's select_4 / mean),
             an embedding added to every step token of that query; the shipped and logged queries get a learned null (lc.QM)
  priv       + privileged memory tokens: 32 logged agent boxes with motion, the hindsight route with the logged speed / timing, 4 stop lines
             with light states, the first >= 45 deg turn (ORACLE ONLY: labels as inputs)
  blind_nat  descriptive: blind + the native outputs (no vision tokens)
Outputs per query: S0's 7 (contact_head.OUT) then the logits of A, B, C. With rung vis and the 7 outputs only, the forward is S0's.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc as LC  # noqa: E402
import contact_head as C  # noqa: E402

RUNGS = ("blind", "vis", "nat", "priv", "blind_nat")
OUT = C.OUT + ("A_logit", "B_logit", "C_logit")
IDX = dict(a=0, b=1, A=7, B=8, C=9)
NAT, HID, OBJ, RTE, LIN, TRN = 206, 1024, 69, 157, 40, 10
HEADS = ("lat", "A", "B", "C")                  # model selection per output group: lat = S0's rule (mean of own agent / boundary AUC)


def build(rung: str, d: int = 256, enc: int = 3, dec: int = 3, drop: float = 0.1):
    import torch
    import torch.nn as nn
    assert rung in RUNGS
    core = C.build("blind" if rung.startswith("blind") else "step", d, enc, dec, drop)
    core.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(OUT)))
    mlp = lambda i: nn.Sequential(nn.LayerNorm(i), nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))  # noqa: E731

    class LHead(nn.Module):
        def __init__(s):
            super().__init__()
            s.rung, s.core = rung, core
            s.vision = core.vision
            s.nat = rung in ("nat", "priv", "blind_nat")
            s.priv = rung == "priv"
            if s.nat:
                s.np_, s.hp = mlp(NAT), mlp(HID)
                s.null = nn.Parameter(torch.zeros(d))
                s.register_buffer("qm", torch.as_tensor(LC.QM.astype(np.int64)))
            if s.priv:
                s.op, s.rp, s.lp, s.tp = mlp(OBJ), mlp(RTE), mlp(LIN), mlp(TRN)
                s.ptype = nn.Parameter(torch.randn(4, d) * 0.02)

        def memory(s, V, valid, E, P):
            c = s.core
            e = c.ep(E)[:, None]
            parts, pads = [e], [E.new_zeros(len(E), 1, dtype=torch.bool)]
            if s.vision:
                parts.append((c.vp(c.vln(V.float())) + c.slot + c.pos).flatten(1, 2))
                pads.append(~valid.repeat_interleave(32, 1))
            if s.priv:
                obj, rte, lin, trn = (x.float() for x in P)
                parts += [s.op(obj) + s.ptype[0], (s.rp(rte) + s.ptype[1])[:, None], s.lp(lin) + s.ptype[2], (s.tp(trn) + s.ptype[3])[:, None]]
                no = E.new_zeros(len(E), 1, dtype=torch.bool)
                pads += [~(obj[..., 6:63:7].amax(-1) > 0), no, lin[..., 0] == 0, no]          # objects without a valid time / absent lines
            if len(parts) == 1:
                return c.enc(e), None
            pad = torch.cat(pads, 1)
            return c.enc(torch.cat(parts, 1), src_key_padding_mask=pad), pad

        def forward(s, V, valid, E, q, N=None, P=None):
            """-> (per-query outputs (B, Q, len(OUT)) fp32, per-step (clearance, margin) (B, Q, 9, 2) fp32 or None)."""
            c = s.core
            B_, Q = q.shape[:2]
            mem, pad = s.memory(V, valid, E, P)
            x = c.sp(c.steps(q)) + c.temb                                      # (B, Q, 9, d)
            if s.nat:
                nat, hid = N
                z = s.np_(nat.float()) + s.hp(hid.float())                     # (B, 2, d): one per student
                g = z[:, s.qm.clamp_min(0)]                                     # (B, Q, d)
                g = torch.where((s.qm >= 0)[None, :, None], g, s.null.to(g.dtype))
                x = x + g[:, :, None]
            x = x.flatten(1, 2)
            for l in c.dec:
                x = l(x, mem, pad, c.T)
            x = x.reshape(B_, Q, c.T, -1)
            return c.out(torch.cat([x.mean(2), x.amax(2)], -1)).float(), (c.so(x).float() if c.T > 1 else None)
    return LHead()


def predict(net, D, idx=None, slots: int = 8, batch: int = 512):
    """Batched eval forward over device tensors D (V, E, Qp, nat, hid, obj, rte, lin, trn; absent = None) -> (n, Q, len(OUT)) numpy."""
    import torch
    net.eval()
    n = len(D["Qp"])
    idx = torch.arange(n, device=D["Qp"].device) if idx is None else idx
    O = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c in range(0, len(idx), batch):
            i = idx[c:c + batch]
            v = (torch.arange(8, device=i.device)[None] >= 8 - slots).expand(len(i), 8)
            N = (D["nat"][i], D["hid"][i]) if net.nat else None
            P = tuple(D[k][i] for k in ("obj", "rte", "lin", "trn")) if net.priv else None
            Vi = None
            if net.vision:
                Vi = D["V"][i.cpu()].to(i.device, non_blocking=True) if D["V"].device.type == "cpu" else D["V"][i]
            o, _ = net(Vi, v, D["E"][i], D["Qp"][i], N, P)
            O.append(o.cpu().numpy())
    net.train()
    return np.concatenate(O)


def load_ckpt(path, dev, head="lat"):
    """ckpt.pt of lc_train.py -> (net in eval mode with the weights selected for `head`, ckpt dict)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    net = build(c["rung"], c["d"], c["enc"], c["dec"], c["drop"]).to(dev)
    net.load_state_dict(ck["best"][head])
    return net.eval(), ck


# ---------------------------------------------------------------- AUC with paired log-cluster bootstrap (contact_head.auc_boot's draws)
def _cluster(y, s, codes, G):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    sp, gp, sn, gn = s[y], codes[y], s[~y], codes[~y]
    W = np.zeros((len(sp), G))
    for h in np.unique(gn):
        x = np.sort(sn[gn == h])
        lo, hi = np.searchsorted(x, sp, "left"), np.searchsorted(x, sp, "right")
        W[:, h] = lo + 0.5 * (hi - lo)
    U = np.zeros((G, G))
    np.add.at(U, gp, W)
    return U, np.bincount(gp, minlength=G).astype(float), np.bincount(gn, minlength=G).astype(float)


def auc_draws(y, scores: dict, groups, n_boot: int = 10_000, seed: int = 0) -> dict:
    """AUC of several scores on the same rows with the SAME log-cluster draws (contact_head.auc_boot's resampling: default_rng(seed)
    .integers(G, size=(n_boot, G))) -> {name: (point AUC, per-draw AUC array (n_boot,), nan where a draw has no positive / negative)}."""
    import pandas as pd
    codes, uniq = pd.factorize(np.asarray(groups))
    G = len(uniq)
    idx = np.random.default_rng(seed).integers(G, size=(n_boot, G))
    c = np.zeros((n_boot, G))
    np.add.at(c, (np.arange(n_boot)[:, None], idx), 1.0)
    out = {}
    for k, s in scores.items():
        U, n1, n0 = _cluster(y, s, codes, G)
        den = (c @ n1) * (c @ n0)
        b = np.where(den > 0, np.einsum("bg,gh,bh->b", c, U, c) / np.where(den > 0, den, 1), np.nan)
        out[k] = (float(U.sum() / (n1.sum() * n0.sum())) if n1.sum() and n0.sum() else np.nan, b)
    return out


def ci(b, alpha=0.05):
    b = b[np.isfinite(b)]
    return (float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))) if len(b) else (np.nan, np.nan)
