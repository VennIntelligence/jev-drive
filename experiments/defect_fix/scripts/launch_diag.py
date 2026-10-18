#!/usr/bin/env python3
"""LAUNCH-DIAG: what in the JTN stage teaches the late launch on wait set v2 variant b (results/rtja/launch_diag.md). No training.

  $AP launch_diag.py lead                          (box, CPU) the logged lead of every variant-b lead-class window (wait_set.py's lead rule at the
                                                   window start), its movement onset (first frame > 0.25 m from its start, wait_set's REST_R) and
                                                   the human's -> $O/lead.json
  python3 launch_diag.py cl --drv DIR --lead F --tags T ..   (Mac or box, CPU) the stored closed-loop reads' decisions (drive.jsonl of
                                                   reads/<tag>/run/b*): plan 4 s arc and ego speed per decision, binned by time since the lead
                                                   moved -> stdout markdown
  $TR launch_diag.py replay --dump TAG --tags T .. (box, GPU) every tag's plan on the exact inputs TAG's own closed-loop run fed (SH30_DUMP of
                                                   the lead-class b windows), with the dumping run's logged route memory ("own"; only for tags
                                                   with the same memory layout) and with the memory zeroed ("zero", the --rm-drop input)
                                                   -> $O/replay/<dump>/<tag>.json (arc4 per scene / decision)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix/launch_diag"
SET = REPO / "experiments/alpasim_proxy/results/wait_set/set.json"
LEAD = ("other_lead", "red_queue")


def windows(cls=LEAD):
    return {w["id"]: w for w in json.loads(SET.read_text())["windows"] if w["variant"] == "b" and w["cls"] in cls}


def arc(P):
    p = [(0.0, 0.0)] + [(x[0], x[1]) for x in P]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:]))


# ---------------------------------------------------------------- lead
def cmd_lead(a):
    import numpy as np
    sys.path.insert(0, str(REPO / "experiments/alpasim_proxy/scripts"))
    sys.argv = sys.argv[:1]
    import wait_set as WS
    W = windows()
    out, by_log = {}, defaultdict(list)
    for w in W.values():
        by_log[w["log"]].append(w)
    for log, ws in sorted(by_log.items()):
        L = WS.Log(log)
        for w in ws:
            w0 = int(w["start_idx"])
            fr = list(range(w0, min(w0 + WS.WIN, len(L.tok))))
            B = L.boxes(fr)
            lead = None
            for tk, x, y, yw, ln, wd in B[w0]:
                ex, ey = WS.to_ego(L.xy[w0], L.yaw[w0], (x, y))
                if abs(ey) < WS.LEAD_LAT and ex - ln / 2 > WS.FRONT - 0.5 and ex - ln / 2 - WS.FRONT < WS.LEAD_GAP:
                    gap = ex - ln / 2 - WS.FRONT
                    if lead is None or gap < lead[1]:
                        lead = (tk, gap, (x, y))
            if lead is None:
                out[w["id"]] = dict(lead=False)
                continue
            tr = {f: next(((x, y) for tk, x, y, *_ in B[f] if tk == lead[0]), None) for f in fr}
            p0 = np.asarray(lead[2])
            disp = [None if tr[f] is None else float(np.hypot(*(np.asarray(tr[f]) - p0))) for f in fr]
            mv = next((i for i, d in enumerate(disp) if d is not None and d > WS.REST_R), None)
            sp = [None] * len(fr)
            for i in range(2, len(fr) - 2):
                if tr[fr[i - 2]] is not None and tr[fr[i + 2]] is not None:
                    sp[i] = float(np.hypot(*(np.asarray(tr[fr[i + 2]]) - np.asarray(tr[fr[i - 2]])))) * WS.HZ / 4
            go = next((i for i, s in enumerate(sp) if s is not None and s > WS.V_GO), None)
            out[w["id"]] = dict(lead=True, gap0=round(float(lead[1]), 2), lead_onset_s=None if mv is None else mv / WS.HZ,
                                lead_go_s=None if go is None else go / WS.HZ, h_onset_s=w["onset_s"], h_launch_s=w["h_launch_s"],
                                disp=[None if d is None else round(d, 3) for d in disp[::5]], cls=w["cls"])
        L.c.close()
        print(log, len(ws), flush=True)
    O.mkdir(parents=True, exist_ok=True)
    (O / "lead.json").write_text(json.dumps(out, indent=0))
    on = [v["lead_onset_s"] for v in out.values() if v.get("lead_onset_s") is not None]
    print(f"{len(out)} windows, lead found {sum(v['lead'] for v in out.values())}, lead onset known {len(on)}, "
          f"median lead onset {sorted(on)[len(on) // 2]:.2f} s (human onset 3.0 s)")


# ---------------------------------------------------------------- closed loop
def decisions(drv, tag):
    """{scene: {k: (t_s, arc4, v)}} from the stored read's variant-b driver logs."""
    W = windows()
    D = defaultdict(dict)
    for f in glob.glob(f"{drv}/{tag}/run/b*/driver-logs/drive.jsonl") + glob.glob(f"{drv}/{tag}/driver-logs/drive.jsonl"):
        for line in open(f):
            d = json.loads(line)
            if d["kind"] == "drive" and d["scene"] in W:
                D[d["scene"]][d["k"]] = (d["now"] / 1e6, arc(d["poses"]), 10 * d["ego"][4])
    return D


def cmd_cl(a):
    lead = json.loads(Path(a.lead).read_text())
    bins = [(-9, -1.0), (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 9)]
    lab = ["< -1", "-1..-0.5", "-0.5..0", "0..0.5", "0.5..1", "1..1.5", "1.5..2", "> 2"]
    print("| tag | n | " + " | ".join(lab) + " |")
    print("|---|--:|" + "--:|" * len(lab))
    rows = {}
    for t in a.tags:
        D = decisions(a.drv, t)
        acc = defaultdict(list)
        for sc, r in D.items():
            L = lead.get(sc, {})
            if L.get("lead_onset_s") is None:
                continue
            for k, (ts, ar, v) in r.items():
                dt = ts - L["lead_onset_s"]
                for (lo, hi), nm in zip(bins, lab):
                    if lo <= dt < hi:
                        acc[nm].append((ar, v))
        rows[t] = acc
        n = len([s for s in D if lead.get(s, {}).get("lead_onset_s") is not None])
        print(f"| {t} | {n} | " + " | ".join(f"{sum(x for x, _ in acc[nm]) / len(acc[nm]):.2f} / {sum(v for _, v in acc[nm]) / len(acc[nm]):.2f}"
                                             if acc[nm] else "-" for nm in lab) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps({t: {nm: v for nm, v in r.items()} for t, r in rows.items()}))


# ---------------------------------------------------------------- replay
def cmd_replay(a):
    import numpy as np
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO)]
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    src = O / "dump" / a.dump / "all"
    dumps = sorted(glob.glob(str(src / "driver-logs/dump/s*_k*.npz")))
    rm, lay = {}, None
    for line in open(src / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "drive" and "rmem" in r:
            rm[(r["scene"], int(r["k"]))] = np.asarray(r["rmem"], np.float32)
    Dz = []
    for f in dumps:
        d = np.load(f)
        k = int(f.rsplit("_k", 1)[1][:-4])
        Dz.append((str(d["scene"]), k, {q: d[q] for q in ("cur", "valid", "ego", "cam_t", "poses")}))
    print(f"{len(Dz)} dumped decisions from {src}", flush=True)
    out = O / "replay" / a.dump
    out.mkdir(parents=True, exist_ok=True)
    for tag in a.tags:
        if (out / f"{tag}.json").exists() and not a.force:
            continue
        core = SC.Core(tag, "cuda", "backwarp")
        res = defaultdict(dict)
        for sc, k, d in Dz:
            ego = d["ego"].astype(np.float32)
            cur, valid = d["cur"], d["valid"]
            prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
            with torch.no_grad():
                pp, cc = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
                H = core.model.net.run_batched(core.A.vision_feeds(pp, cc), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *core.A.H_SHAPE)
                r_ = {}
                modes = ("zero",) if not core.rm else ("own", "zero")
                for mode in modes:
                    if core.rm:
                        m = rm.get((sc, k))
                        if mode == "own" and (m is None or len(m) != core.rm_dim):
                            continue
                        row = np.r_[ego, m if mode == "own" else np.zeros(core.rm_dim, np.float32)]
                    else:
                        row = ego
                    o = core.model(H, torch.from_numpy(row[None]).to(core.dev), torch.tensor([[1.0, 0.0]], device=core.dev))
                    o = (o[0] if isinstance(o, tuple) else o).float()
                    mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
                    P = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, d["cam_t"][:2].astype(np.float64), Z.T_OUT, "lever")
                    r_[mode] = round(arc(P), 4)
            res[sc][k] = dict(v=round(10 * float(ego[4]), 4), served=round(arc(d["poses"]), 4), **r_)
        (out / f"{tag}.json").write_text(json.dumps(dict(tag=tag, dump=a.dump, rm=bool(core.rm), rm_dim=int(getattr(core, "rm_dim", 0) or 0), res=res)))
        z = [x.get("zero") for r in res.values() for x in r.values()]
        print(tag, "decisions", len(z), "mean arc4 zero-memory", round(float(np.mean(z)), 3), flush=True)
        del core
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("lead")
    p = sp.add_parser("cl")
    p.add_argument("--drv", required=True), p.add_argument("--lead", required=True), p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--json")
    p = sp.add_parser("replay")
    p.add_argument("--dump", required=True), p.add_argument("--tags", nargs="+", required=True), p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dict(lead=cmd_lead, cl=cmd_cl, replay=cmd_replay)[a.cmd](a)
