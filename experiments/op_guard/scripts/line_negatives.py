"""Guard line `negatives`: open loop, negative route commands (lib/route_neg.py): does a command for a route the scene does not have
pull the plan off the logged path?

Frames: the 158 navtrain negatives of experiments/op_route_cmd/results/negatives_sample.npz (route_neg_sample.py; N1 exit 58 /
N2 side 40 / N3 wrong-way 30 / N4 U-turn 30). All 158 are used in both modes; 135 still await the human visual check (only the 23 N1
tier-A left / right rows are map-certain), reported as a separate info row. Frames on op_lb's NAVSIM protocol (neg_frames.py:
4 keys + 6 GIMM frames), plans through the training port with the candidate's ONNX weights (portcand.py), as h_prep / h_wod.
Offset = |y_plan(4 s) - y_logged(4 s)| (rear-axle frame; the target of a negative is the logged path, results/negatives.md), mean over
frames, against the original (shipped) model without a command. Rule: offset under the negative command <= original + 0.3 m.

  command_adapter null  the model has no command channel: the negative cannot reach it. shipped: offset == original by construction
                        (status ok, trivially satisfied); another candidate: its plain-model offset on these frames vs shipped's.
  command_adapter set   cmd_adapter.load(...).plans(frames, commands) for Command('none'), Command('correct', poly_pos) and
                        Command('negative', poly); the rule row uses the negative, the others are info rows.

  DATA_DIR=... $DATA_DIR/envs/op-train/bin/python experiments/op_guard/scripts/line_negatives.py --candidate it_dw3-s0 --gpu 3
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardlib as G  # noqa: E402
import portcand  # noqa: E402
import cmd_adapter as CA  # noqa: E402

sys.path[:0] = [str(G.REPO / "experiments" / "op_img_cmd" / "scripts"), str(G.REPO / "experiments" / "op_adapt_h" / "scripts")]
LINE = "negatives"
DATA = "lb_guardneg"
MARGIN = 0.3
TAU_IDX = 15                      # 4 s on the 0.25 s grid of h_prep.nav_future


def ensure_frames(a, prov: dict):
    """op_lb data lb_guardneg (keys + GIMM), built once by neg_frames.py in the openpilot / vfi envs."""
    root = G.data_dir() / "runs" / "op_lb" / DATA
    env = dict(os.environ)
    ts = ["taskset", "-c", a.cpus] if a.cpus else []
    for step, envname, extra, done in (("prep", "openpilot", [], root / "keys.npy"),
                                       ("synth", "vfi", ["--gpu", str(a.gpu)], root / "gimm.chunks")):
        if step == "synth" and done.exists() and all((done / f"{k:05d}.done").exists() for k in range(-(-158 // 32))):
            continue
        if step == "prep" and done.exists():
            continue
        cmd = ts + [str(G.data_dir() / "envs" / envname / "bin" / "python"), str(HERE / "neg_frames.py"), step] + extra
        print("+", " ".join(cmd), flush=True)
        t = time.time()
        subprocess.run(cmd, check=True, env=env, cwd=G.REPO)
        prov.setdefault("frames_build_s", {})[step] = round(time.time() - t, 1)


def sample():
    z = np.load(G.REPO / "experiments" / "op_route_cmd" / "results" / "negatives_sample.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def port_plans(c: dict, dev, prov: dict) -> tuple[np.ndarray, list]:
    """Plain-model plans (n, 33, 15) of the lb_guardneg tokens (order of meta.json) via the port."""
    import torch
    import h_prep as HP
    from experiments.op_adapt_h.lib import op_adapt_h as H
    m, info = portcand.load(c["onnx"], dev)
    prov["port"] = info
    src = HP.NavSrc(DATA)
    mt = src.mt
    n = len(mt["names"])
    sv = np.ones((n, 9), bool)
    sv[:, 0] = False                                   # h_prep nav: slot 0 carries a zero hidden state
    tc = np.array([[0.0, 1.0] if l else [1.0, 0.0] for l in mt["lht"]], np.float32)
    mu = np.zeros((n, 33, 15), np.float32)
    from jevdrive import op_adapt as A
    pi = A.plan_index(m.net.slices)
    with torch.no_grad():
        for i in range(0, n, 16):
            j = slice(i, min(i + 16, n))
            x = torch.from_numpy(np.stack([src(r) for r in range(j.start, j.stop)])).to(dev)
            s = torch.from_numpy(sv[j]).to(dev)
            o = m(H.trunks(m.net, x) * s[:, :, None, None, None], s, torch.from_numpy(tc[j]).to(dev).half())["outputs"].float()
            mu[j] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
    return mu, list(mt["names"])


def y4(mu, cams):
    import img_run
    from img_report import interp
    return np.array([interp(img_run.to_rear(p[:, 0:3], p[:, 11], np.asarray(c, float))[0], 4.0)[1] for p, c in zip(mu, cams)])


def cand_arrays(c: dict, a, out: Path, prov: dict) -> dict:
    p = out / "negatives.npz"
    if p.exists() and not a.force:
        prov["cache"].append(f"{p} (this line's own earlier result)")
        return dict(np.load(p, allow_pickle=True))
    import torch
    import h_prep as HP
    dev = torch.device("cuda")
    ensure_frames(a, prov)
    S = sample()
    mt = HP.NavSrc(DATA).mt
    names = list(mt["names"])
    pos = {t: i for i, t in enumerate(names)}
    k = np.array([pos[t] for t in S["id"].astype(str)])
    cams = np.asarray(mt["cam"], float)[k]
    fut = HP.nav_future(list(S["id"].astype(str)))[:, TAU_IDX, 1]
    t = time.time()
    ad = CA.load(c, a.gpu, a.cpus)
    z = {"id": S["id"], "kind": S["kind"], "tier": S["tier"], "needs_visual": S["needs_visual"], "y_logged": fut}
    if ad is None:
        mu, _ = port_plans(c, dev, prov)
        z["y_none"] = y4(mu[k], cams)
    else:
        prov["adapter"] = ad.describe()
        fr = [CA.Frame("nav", str(tk), f"op_lb/{DATA}#{i}") for tk, i in zip(S["id"].astype(str), k)]
        for kind, P, M in (("none", None, None), ("correct", S["poly_pos"], S["pmask_pos"]), ("negative", S["poly"], S["pmask"])):
            cm = [CA.Command(kind, None if P is None else np.asarray(P[i])[np.asarray(M[i], bool)],
                             {"kind": str(S["kind"][i]), "tier": str(S["tier"][i])}) for i in range(len(fr))]
            z[f"y_{kind}"] = y4(ad.plans(fr, cm), cams)
    prov["plans_s"] = round(time.time() - t, 1)
    np.savez(p, **z)
    return z


def main():
    a = G.line_args(LINE, __doc__.splitlines()[0]).parse_args()
    c = G.resolve(a.candidate)
    out = G.run_dir(c["name"], a.mode)
    if G.done(out, LINE) and not a.force:
        print(f"{out}/lines/{LINE}.json exists (use --force)")
        return
    t0 = time.time()
    rule = G.LINES[LINE][2]
    prov = {"cache": [], "frames": f"negatives_sample.npz, 158 navtrain tokens, op_lb {DATA} (4 keys + GIMM)", "subset": "all 158 in both modes",
            "adapter": c.get("command_adapter")}
    try:
        za = cand_arrays(c, a, out, prov)
        if c["name"] == G.SHIPPED:
            zb = za
        else:
            pr = {"cache": []}
            zb = cand_arrays(G.resolve(G.SHIPPED), type(a)(**{**vars(a), "force": False}), G.run_dir(G.SHIPPED, a.mode), pr)
            prov["cache"] += [f"shipped: {x}" for x in pr["cache"]]
        assert (za["id"] == zb["id"]).all()
        ref = np.abs(zb["y_none"] - zb["y_logged"])
        has_cmd = "y_negative" in za
        off = np.abs((za["y_negative"] if has_cmd else za["y_none"]) - za["y_logged"])
        from jevdrive import stats
        r = stats.paired(off, ref)
        if c["name"] == G.SHIPPED:
            note = "no command channel: the negative never reaches the model, offset == original by construction"
        elif not has_cmd:
            note = "no command channel (command_adapter null): plain-model offset on the negative frames vs shipped's"
        else:
            note = "offset under the negative command (adapter) vs shipped without a command"
        rows = [G.row("negatives.offset", "|y(4 s) - logged| on the negative frames, mean (m)", float(off.mean()), float(ref.mean()), rule=rule,
                      ok=G.at_most(float(off.mean()), float(ref.mean()), MARGIN), ci=[r["lo"], r["hi"]],
                      note=note + f"; n {len(off)}, 135 of 158 await the visual check (all used)")]
        cert = (za["kind"].astype(str) == "N1_exit") & (za["tier"].astype(str) == "A") & ~za["needs_visual"].astype(bool)
        rows.append(G.row("negatives.offset_n1a", "same, the 23 map-certain N1 tier-A left / right rows", float(off[cert].mean()),
                          float(ref[cert].mean()), rule="info", note=f"n {int(cert.sum())}"))
        for kd in sorted(set(za["kind"].astype(str))):
            m = za["kind"].astype(str) == kd
            rows.append(G.row(f"negatives.offset_{kd}", f"same, {kd}", float(off[m].mean()), float(ref[m].mean()), rule="info", note=f"n {int(m.sum())}"))
        if has_cmd:
            for kd in ("none", "correct"):
                o = np.abs(za[f"y_{kd}"] - za["y_logged"])
                rows.append(G.row(f"negatives.offset_cmd_{kd}", f"offset with command '{kd}'", float(o.mean()), float(ref.mean()), rule="info"))
            rows.append(G.row("negatives.pull", "|y_negative - y_none| at 4 s, mean", float(np.abs(za["y_negative"] - za["y_none"]).mean()),
                              0.0, rule="info"))
        status = "ok"
    except NotImplementedError as e:
        rows, status = [G.row("negatives.offset", "offset under negative commands", None, rule=rule, note=f"command adapter stub: {e}")], "stub"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rows, status = [G.row("negatives", "error", None, rule=rule, note=f"{type(e).__name__}: {e}")], "error"
    p = G.write_line(out, LINE, c["name"], a.mode, rows, t0, status, prov)
    print(json.dumps(json.loads(p.read_text()), indent=1, default=str))


if __name__ == "__main__":
    main()
