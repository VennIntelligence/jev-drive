"""Route-choice fine-tune as an op_guard command adapter (experiments/op_guard/scripts/cmd_adapter.py protocol), and the registry helper.

RouteCmdAdapter: the candidate's serving ONNX on the training port (op_guard portcand.load: changed initializers copied into LModel, stage 4
recomputed from the stored stage-3 trunks) plus its route adapter (`adapter.npz`, lib/route_adapter.RouteAdapter in torch) whose bias is
added to the hidden tokens of all 9 context frames before the valid mask, as in training (scripts/rft.py RModel.policy). Command:
  none                    zero feature vector -> zero bias (the unconditioned fine-tuned model)
  correct / negative      Command.poly (valid vertices, rear-axle frame) padded to 16 vertices + mask -> route_adapter.features(enc), no noise
Frame sources the guard lines use (the same forwards as their plain-model paths):
  op_img_cmd/ft/bank/<bank>#<row>   line_drift: img2_eval.bank trunks, slot_valid, tc (portcand.plans_from_trunks)
  op_lb/<data>#<i>                   line_negatives: h_prep.NavSrc frames -> op_adapt_h.trunks, slot 0 invalid, tc from lht (port_plans)

Registry (any python; run it on the Mac and commit candidates.json: the box only pulls):
  python experiments/op_route_ft/scripts/guard_adapter.py register <arm>-s0 [--onnx PATH] [--adapter PATH|none] [--note TEXT]
  defaults: onnx $DATA_DIR/runs/op_route_ft/onnx/<arm>-s0.onnx, adapter = the `.adapter.npz` next to it; adapter none -> command_adapter null
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _d in ("", "lib", "experiments/op_guard/scripts", "experiments/op_img_cmd/scripts", "experiments/op_adapt_h/scripts"):
    if str(REPO / _d) not in sys.path:
        sys.path.append(str(REPO / _d))
import route_adapter as RA  # noqa: E402

DOTTED = "experiments.op_route_ft.scripts.guard_adapter:RouteCmdAdapter"


def pad_poly(poly):
    """(k, 2) valid vertices -> (16, 2), (16,) mask."""
    p = np.zeros((16, 2), np.float32)
    m = np.zeros(16, bool)
    q = np.asarray(poly, np.float32).reshape(-1, 2)[:16]
    p[:len(q)], m[:len(q)] = q, True
    return p, m


class RouteCmdAdapter:
    def __init__(self, candidate: dict, gpu: int = 0, cpus: str = ""):
        self.c = candidate
        self.path = candidate.get("route_adapter") or candidate.get("adapter")
        self.enc = str(np.load(self.path)["enc"]) if self.path else None
        self.model = self.ad = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        import portcand
        self.dev = torch.device("cuda")
        self.model, self.info = portcand.load(self.c["onnx"], self.dev)
        if self.path:
            z = np.load(self.path)
            ad = RA.RouteAdapter(self.enc)
            ad.load_state_dict({k: torch.from_numpy(z[k.replace(".", "_")]) for k in ad.state_dict()})
            self.ad = ad.to(self.dev).eval()

    def feats(self, commands):
        F = RA.ENC_DIM[self.enc] if self.enc else 1
        out = np.zeros((len(commands), F), np.float32)
        for i, c in enumerate(commands):
            if c.kind != "none" and c.poly is not None and self.enc:
                out[i] = RA.features(self.enc, *pad_poly(c.poly))
        return out

    def plans(self, frames, commands) -> np.ndarray:
        import torch
        from jevdrive import op_adapt as A
        self._load()
        m = self.model
        f = torch.from_numpy(self.feats(commands)).to(self.dev)
        with torch.no_grad():
            B = self.ad(f) if self.ad is not None else torch.zeros(len(commands), *A.H_SHAPE, device=self.dev)

        class Bias(torch.nn.Module):                        # LModel.policy calls adapter(H, intent): intent = row index into B
            def forward(self, H, idx):
                return H + B[idx][:, None].to(H.dtype)
        m.adapter = Bias()
        pi = A.plan_index(m.net.slices)
        mu = np.zeros((len(frames), 33, 15), np.float32)
        groups = {}
        for i, fr in enumerate(frames):
            src, _, r = fr.source.rpartition("#")
            groups.setdefault(src, []).append((i, int(r)))
        try:
            for src, items in groups.items():
                idx = np.array([i for i, _ in items])
                rows = np.array([r for _, r in items])
                if src.startswith("op_img_cmd/ft/bank/"):
                    self._bank(src.split("/")[-1], idx, rows, mu, pi)
                elif src.startswith("op_lb/"):
                    self._nav(src.split("/", 1)[1], idx, rows, mu, pi)
                else:
                    raise NotImplementedError(f"RouteCmdAdapter: frame source {src!r}")
        finally:
            m.adapter = None
        return mu

    def _run(self, trunk, valid, tc, idx):
        import torch
        o = self.model(trunk, valid, tc, torch.from_numpy(np.asarray(idx)).to(self.dev))["outputs"].float()
        return o

    def _bank(self, bank, idx, rows, mu, pi, bs=96):
        import torch
        import img2_eval as E
        T, v = E.bank(bank)
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                r, j = rows[i:i + bs], idx[i:i + bs]
                o = self._run(torch.from_numpy(np.stack([np.asarray(T[k]) for k in r])).to(self.dev), torch.from_numpy(v["slot_valid"][r]).to(self.dev),
                              torch.from_numpy(np.asarray(v["tc"][r], np.float32)).to(self.dev).half(), j)
                mu[j] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()

    def _nav(self, data, idx, rows, mu, pi, bs=16):
        import torch
        import h_prep as HP
        from experiments.op_adapt_h.lib import op_adapt_h as H
        src = HP.NavSrc(data)
        lht = src.mt["lht"]
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                r, j = rows[i:i + bs], idx[i:i + bs]
                s = np.ones((len(r), 9), bool)
                s[:, 0] = False                             # h_prep nav: slot 0 carries a zero hidden state
                tc = np.array([[0.0, 1.0] if lht[k] else [1.0, 0.0] for k in r], np.float32)
                x = torch.from_numpy(np.stack([src(k) for k in r])).to(self.dev)
                st = torch.from_numpy(s).to(self.dev)
                o = self._run(H.trunks(self.model.net, x) * st[:, :, None, None, None], st, torch.from_numpy(tc).to(self.dev).half(), j)
                mu[j] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()

    def describe(self) -> dict:
        return {"adapter": "route_ft", "enc": self.enc, "adapter_npz": self.path, "onnx": self.c.get("onnx"),
                "runtime": "training port (portcand) + torch RouteAdapter, bias before the valid mask, no noise",
                "port": getattr(self, "info", None)}


# ---------------------------------------------------------------- registry
def register(a):
    import os
    reg_p = REPO / "experiments/op_guard/candidates.json"
    onnx = a.onnx or f"$DATA_DIR/runs/op_route_ft/onnx/{a.name}.onnx"
    adp = a.adapter or onnx[: -len(".onnx")] + ".adapter.npz"
    if Path(os.path.expandvars("$DATA_DIR")).is_dir():   # on the box: the files must exist (on the Mac: written unchecked)
        if adp != "none" and not Path(os.path.expandvars(adp)).is_file():
            raise SystemExit(f"missing adapter {os.path.expandvars(adp)} (use --adapter none for rc-ctl)")
        if not Path(os.path.expandvars(onnx)).is_file():
            raise SystemExit(f"missing ONNX {os.path.expandvars(onnx)} (route_onnx.py build first)")
    reg = json.loads(reg_p.read_text())
    reg[a.name] = {"onnx": onnx, "route_adapter": None if adp == "none" else adp, "command_adapter": None if adp == "none" else DOTTED,
                   "note": a.note or f"op_route_ft {a.name} (route-choice fine-tune; adapter {'none: zero bias' if adp == 'none' else 'route polyline'})"}
    reg_p.write_text(json.dumps(reg, indent=1) + "\n")
    print("registered", a.name, json.dumps(reg[a.name]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register")
    r.add_argument("name", help="candidate name, e.g. rc-bear-s0")
    r.add_argument("--onnx", default="")
    r.add_argument("--adapter", default="", help="adapter.npz path or none")
    r.add_argument("--note", default="")
    a = ap.parse_args()
    register(a)
