"""Heading-profile auxiliary supervision on the policy's plan-pathway hidden state (experiments/corridor HEAD1b step B2,
plans/2026-10-10-head1-prereg.md amendment 2026-10-10; EXPLORATORY: second attempt after the missed HEAD1 gate G3).

Tap     `select_4` of the Cinque graph (node 638): the last token of the off-policy temporal summarizer's transformer output, (512,). It is
        the one state every off-policy head is decoded from: plan = final(select_4 + head_mlp.plan(LN(select_4))) (nodes 639-644, 663). It
        lies downstream of the parity adapter (the adapter biases the 9 context frames the summarizer reads) and all of the trainable plan
        pathway except the plan hydra's own MLP is upstream of it, so the auxiliary gradient reaches the adapter and the summarizer and leaves
        the plan read-out free.
Head    LN -> Linear(512, 512) -> GELU -> Linear(512, 22): the heading (rad, relative to the ego heading at t0) at the 22 arc lengths of
        head1_labels.GRID. Target = label L (logged path re-parameterised by arc length; a TRAINING LABEL, never an input), Huber delta
        0.1 rad over the labelled points of the imitation rows. The head is dropped at inference: the checkpoint is a plain P2 checkpoint,
        the head is saved next to it as aux.pt.
lam     > 0: lam x loss, gradient into the policy. 0: the head is fitted on the DETACHED state with weight 1 (a probe of the plain recipe:
        the policy's gradients, and so its weights, are those of the run without the head, bit for bit).
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TAP, NG, DELTA = "select_4", 22, 0.1


def holdout_mask(logs, salt="head1val") -> np.ndarray:
    """True on rows whose log is in the validation part: sha256("<salt>|" + log) % 10 == 0 (head1_train.is_val)."""
    u, inv = np.unique(np.asarray(logs).astype(str), return_inverse=True)
    return np.array([int(hashlib.sha256(f"{salt}|{l}".encode()).hexdigest(), 16) % 10 == 0 for l in u.tolist()])[inv]


class HeadingAux(nn.Module):
    def __init__(self, label_file, names, dev, d_in=512):
        super().__init__()
        z = np.load(label_file)
        ln = z["names"].astype(str)
        o = np.argsort(ln)
        nm = np.asarray(names).astype(str)
        j = np.clip(np.searchsorted(ln[o], nm), 0, len(ln) - 1)
        hit = ln[o][j] == nm
        Y = np.where(hit[:, None], z["L"][o[j]], np.nan).astype(np.float32)
        assert Y.shape[1] == NG
        self.coverage = float(np.isfinite(Y[:, 0]).mean())
        self.Y = torch.from_numpy(np.nan_to_num(Y)).to(dev)
        self.ok = torch.from_numpy(np.isfinite(Y)).to(dev)
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, NG))
        self.to(dev)

    def forward(self, h):
        return self.net(h.float())

    def loss(self, h, rows, m):
        """Huber over the labelled points of the rows m (B,) bool; 0 when there is none."""
        ok = self.ok[rows] & m[:, None]
        l = F.huber_loss(self(h), self.Y[rows], delta=DELTA, reduction="none") * ok
        return l.sum() / ok.sum().clamp_min(1)

    @staticmethod
    def dose(rest, term, params) -> dict:
        """Gradient norms on the policy's parameters of the auxiliary term (already weighted) and of the rest of the loss."""
        g = lambda x: float(torch.sqrt(sum((q.float() ** 2).sum() for q in torch.autograd.grad(x, params, retain_graph=True, allow_unused=True) if q is not None)))  # noqa: E731
        return {"aux/grad_aux": g(term), "aux/grad_rest": g(rest)}

    @torch.no_grad()
    def dump(self, model, S, rows, W, rear, path, bs=128, pi=None):
        """Plans (8 poses, rear axle) and the head's profile on `rows` of the store -> npz (names, log, poses, prof, fut, hid)."""
        tap, was = model.tap, model.training
        model.tap = TAP
        model.eval()
        self.eval()
        pi = torch.as_tensor(S.pi if pi is None else pi, device=S.ego.device)
        P, Q, Hd = [], [], []
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs], device=S.ego.device)
            out, hid = model(S.front[r], S.ego[r], S.tc[r], None, None, nv=None if S.nv is None else S.nv[r])
            P.append(torch.stack(rear(out.float()[:, pi].view(-1, 33, 15), S.cam_x[r], W), -1).cpu().numpy())
            Q.append(self(hid).cpu().numpy())
            Hd.append(hid.float().cpu().numpy().astype(np.float16))
        model.tap = tap
        model.train(was)
        self.train(was)
        np.savez(path, names=S.tb["names"][rows], log=S.tb["log"][rows], poses=np.concatenate(P).astype(np.float32), prof=np.concatenate(Q).astype(np.float32),
                 fut=S.tb["fut"][rows], hid=np.concatenate(Hd))
