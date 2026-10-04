"""Guard line `drift`: open loop, no command; how far the candidate's plan moves from the ORIGINAL (shipped) model's plan.

Frames = the no-overlay (`none`) rows of the image-command guard (experiments/op_img_cmd, Q3 sky banks, img2_eval.py with IMG_SETS=sky):
  navdev     navtrain junction + straight frames of the Q2 dev logs (bank skytrain)           op_lb protocol (4 keys + GIMM)
  carladev   CARLA B2D junction frames of the dev routes (bank skycarla)
  dist dev   op_adapt_H dev pools nav / wod / carla (bank dist; L3's drift readout)
  full mode adds naveval (bank skyeval, 385 junction / 293 straight) and carlatest (skycarla test routes).
Plans: the stored stage-3 trunks through the training port with the candidate's ONNX weights (portcand.py), same forward as
img2_eval.py `plans`; a candidate with a command adapter (cmd_adapter.py) gets its plans from the adapter with Command('none').
Metrics per (set, frame kind): median over frames of |y_cand - y_orig| at 4 s (rear-axle frame, img_run.to_rear) = the guard's
"4 s lateral drift", and the median op_adapt.plan_drift (0-5 s mean L2, the metric op_img_cmd's 0.10 m line was set on) as info.
Rule: the max over sets / kinds of the 4 s lateral median <= 0.10 m (shipped = 0 by definition).

  DATA_DIR=... $DATA_DIR/envs/op-train/bin/python experiments/op_guard/scripts/line_drift.py --candidate it_dw3-s0 --gpu 3
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("IMG_SETS", "sky")                  # img2_eval: the Q3 sky banks (the guard's frames after the course change)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guardlib as G  # noqa: E402
import portcand  # noqa: E402
import cmd_adapter as CA  # noqa: E402

sys.path[:0] = [str(G.REPO / "experiments" / "op_img_cmd" / "scripts"), str(G.REPO / "experiments" / "op_common_cause" / "scripts"),
                str(G.REPO / "experiments" / "op_adapt_h" / "scripts")]
LINE = "drift"
TAU = 4.0
LINE_M = 0.10
SETS = {"subset": ("navdev", "carladev", "dist:nav", "dist:wod", "dist:carla"),
        "full": ("navdev", "carladev", "naveval", "carlatest", "dist:nav", "dist:wod", "dist:carla")}


def frames(name: str):
    """(bank name, row indices into the bank, kind per row, domain) of the `none` rows of one evaluation set."""
    import img2_eval as E
    if name.startswith("dist:"):
        dom = name[5:]
        T, v = E.bank("dist")
        r = np.flatnonzero((v["dom"] == dom) & (v["split"] == "dev") & (v["fam"].astype(str) == "none"))
        return "dist", r, np.full(len(r), "all"), dom
    b, Gs = E.set_def(name)
    T, v = E.bank(b)
    r = np.flatnonzero((v["fam"].astype(str) == "none") & np.isin(v["token"], list(Gs)))
    return b, r, np.array([Gs[str(t)]["kind"] for t in v["token"][r]]), "nav" if name.startswith("nav") else "carla"


def plans(model, adapter, bank: str, rows, dom: str, dev) -> np.ndarray:
    import img2_eval as E
    T, v = E.bank(bank)
    if adapter is not None:
        fr = [CA.Frame(dom, str(v["token"][r]), f"op_img_cmd/ft/bank/{bank}#{r}") for r in rows]
        return adapter.plans(fr, [CA.Command("none")] * len(fr))
    return portcand.plans_from_trunks(model, T, rows, v["slot_valid"], v["tc"], dev)


def lat4(mu, cams):
    import img_run
    from img_report import interp
    return np.array([interp(img_run.to_rear(m[:, 0:3], m[:, 11], np.asarray(c, float).reshape(-1))[0], TAU)[1] for m, c in zip(mu, cams)])


def set_plans(c: dict, a, name: str, out: Path, prov: dict, model_cache: dict):
    """Plans of one candidate on one set -> out/drift/<set>.npz (resumable; --force recomputes)."""
    p = out / "drift" / f"{name.replace(':', '_')}.npz"
    if p.exists() and not a.force:
        prov["cache"].append(f"{p} (this line's own earlier result)")
        return dict(np.load(p, allow_pickle=True))
    import torch
    import img2_eval as E
    dev = torch.device("cuda")
    if "m" not in model_cache:
        model_cache["ad"] = CA.load(c, a.gpu, a.cpus)
        model_cache["m"], model_cache["info"] = (None, {"port": "adapter"}) if model_cache["ad"] else portcand.load(c["onnx"], dev)
    b, rows, kind, dom = frames(name)
    t = time.time()
    mu = plans(model_cache["m"], model_cache["ad"], b, rows, dom, dev)
    _, v = E.bank(b)
    z = dict(rows=rows, kind=kind, dom=dom, mu=mu, cam=np.asarray(v["cam"])[rows], y4=lat4(mu, np.asarray(v["cam"])[rows]))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **z)
    prov.setdefault("plans_s", {})[name] = round(time.time() - t, 1)
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
    prov = {"cache": [], "frames": "op_img_cmd Q3 sky banks, `none` rows; dist = op_adapt_H dev pools", "sets": SETS[a.mode],
            "adapter": c.get("command_adapter")}
    try:
        from jevdrive import op_adapt as A
        rows, worst, mc, mref = [], [], {}, {}
        ref_dir = G.run_dir(G.SHIPPED, a.mode)
        for s in SETS[a.mode]:
            za = set_plans(c, a, s, out, prov, mc)
            if c["name"] == G.SHIPPED:
                zb = za
            else:
                pr = {"cache": []}
                b = type(a)(**{**vars(a), "force": False})
                zb = set_plans(G.resolve(G.SHIPPED), b, s, ref_dir, pr, mref)
                prov["cache"] += [f"shipped: {x}" for x in pr["cache"]]
            assert np.array_equal(za["rows"], zb["rows"])
            dy = np.abs(za["y4"] - zb["y4"])
            dr = A.plan_drift(za["mu"].astype(np.float32), zb["mu"].astype(np.float32))
            for k in sorted(set(za["kind"].astype(str))):
                m = za["kind"].astype(str) == k
                med = float(np.median(dy[m]))
                worst.append((med, f"{s}/{k}"))
                rows.append(G.row(f"drift.{s}.{k}", f"4 s lateral drift median, {s} {k} ({za['dom']})", med, 0.0, rule="info",
                                  note=f"n {int(m.sum())}; p90 {np.percentile(dy[m], 90):.3f}; plan_drift (0-5 s L2) median "
                                       f"{np.median(dr[m]):.3f}", n=int(m.sum()), plan_drift_median=float(np.median(dr[m]))))
        w, where = max(worst)
        note = f"max at {where}" + ("; shipped vs itself = 0 by definition" if c["name"] == G.SHIPPED else "")
        rows.insert(0, G.row("drift.max", "4 s lateral drift vs shipped, max over sets / kinds (median per set)", w, 0.0, rule=rule,
                             ok=G.at_most(w, 0.0, LINE_M), note=note))
        prov["port"] = mc.get("info")
        status = "ok"
    except NotImplementedError as e:
        rows, status = [G.row("drift.max", "4 s lateral drift vs shipped", None, rule=rule, ok=None, note=f"command adapter stub: {e}")], "stub"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rows, status = [G.row("drift", "error", None, rule=rule, note=f"{type(e).__name__}: {e}")], "error"
    p = G.write_line(out, LINE, c["name"], a.mode, rows, t0, status, prov)
    print(json.dumps(json.loads(p.read_text()), indent=1, default=str))


if __name__ == "__main__":
    main()
