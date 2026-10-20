#!/usr/bin/env python3
"""LINEAGE-PAI281 table (box, repo .venv; read roots are read, never written). Reads: lineage.sh. Arms dropped on 2026-10-20 (scope): P0 + ACC,
RB0-F-s0 + ACC and + ACC + GAP, NTV0-L18-s0 + ACC without GAP, + LEAD + GAP without ACC, + ACC + GAP at 1.0,4.0 (stored evidence: d330, d381).

  lineage_pai281.py stored                              "<label> <root>" of the stored comparison reads (lineage.sh extracts them)
  lineage_pai281.py table --tax DIR --out FILE.md [--json FILE]

Per read: the scenes it scored without a driver RPC error (decision 327's rule), their mean, zeros by class (pai281.ptype / zclass, i.e.
nx_regress.failure_type), corridor zeros split by rf_tax.classify's primary label (DIR/<label>.pkl, rf_tax.py extract of the read):
overshoot = cor_route_end (the ego passed the end of the human's log), route turn = cor_straight_at_turn / cor_turn_under / cor_turn_over /
cor_wrong_exit, other = cor_drift / cor_lane_change; partial scores 0 < s < 0.98 (rear-hit, slow, other); and the mean on the scenes every
row scored cleanly. Paired steps: jevdrive.stats.paired on the scenes both reads scored cleanly, log-clustered, 10 000 draws, seed 0.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/wait_cap/scripts")]
from jevdrive import stats  # noqa: E402
from pai281 import load, ptype, zclass  # noqa: E402

D = os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs") + "/runs"
L = f"{D}/defect_fix/lineage/read"
NEW = [("P0-NAT", "P0, native"), ("P0-VL", "P0, VC+LEAD"), ("P0-AG", "P0, +ACC+GAP"), ("RB0-NAT", "RB0-F-s0, native"), ("RB0-VL", "RB0-F-s0, VC+LEAD"),
       ("NTV0-NAT", "NTV0-L18-s0, native")]
STORED = [("NTV0-VL", "NTV0-L18-s0, VC+LEAD (stored N0)", "route_follow/pai_input_loop/N0"),
          ("NTV0-AG", "NTV0-L18-s0, +ACC+GAP (stored)", "defect_fix/pai_accgap/NTV0-L18-s0"),
          ("NTV0s1-AG", "NTV0-L18-s1, +ACC+GAP", "defect_fix/pai_accgap/NTV0-L18-s1"),
          ("RSG-VL", "RSG-L18-s0, VC+LEAD", "route_follow/pai_g10/g10"),
          ("RSG-ACC", "RSG-L18-s0, +ACC", "route_follow/pai_g10/acc"),
          ("RSG-AG", "RSG-L18-s0, +ACC+GAP", "route_follow/acc_gap/read/s2"),
          ("JTN-VL", "JTN-L18-s0, VC+LEAD", "defect_fix/pai_nohold/JTN-L18-s0"),
          ("JTN-AG", "JTN-L18-s0, +ACC+GAP", "defect_fix/pai_accgap/JTN-L18-s0"),
          ("JTNs1-AG", "JTN-L18-s1, +ACC+GAP", "defect_fix/pai_accgap/JTN-L18-s1"),
          ("Bs0-AG", "B STANDFMX-L18-s0, +ACC+GAP", "defect_fix/stand_b/pai281/STANDFMX-L18-s0"),
          ("Bs1-AG", "B STANDFMX-L18-s1, +ACC+GAP", "defect_fix/stand_b/pai281/STANDFMX-L18-s1"),
          ("B2s0-AG", "B2 STANDFMXH-L18-s0, +ACC+GAP", "defect_fix/stand_b/pai281/STANDFMXH-L18-s0"),
          ("B2s1-AG", "B2 STANDFMXH-L18-s1, +ACC+GAP", "defect_fix/stand_b/pai281/STANDFMXH-L18-s1")]
ORDER = ["P0-NAT", "P0-VL", "P0-AG", "RB0-NAT", "RB0-VL", "NTV0-NAT", "NTV0-VL", "NTV0-AG"]
ROWS = [(k, lab, f"{L}/{k}") for k, lab in NEW] + [(k, lab, f"{D}/{r}") for k, lab, r in STORED]
ROWS = sorted(ROWS, key=lambda r: ORDER.index(r[0]) if r[0] in ORDER else len(ORDER) + [s[0] for s in STORED].index(r[0]))
STEPS = [("RB0-NAT", "P0-NAT"), ("NTV0-NAT", "RB0-NAT"), ("RB0-VL", "RB0-NAT"), ("RB0-VL", "P0-VL"), ("NTV0-VL", "RB0-VL"), ("NTV0-VL", "P0-VL"), ("NTV0-AG", "P0-AG"), ("NTV0-NAT", "P0-NAT"), ("RSG-VL", "RB0-VL"),
         ("P0-VL", "P0-NAT"), ("P0-AG", "P0-VL"), ("NTV0-VL", "NTV0-NAT"), ("NTV0-AG", "NTV0-VL"),
         ("RSG-ACC", "RSG-VL"), ("RSG-AG", "RSG-ACC")]
COR = {"cor_route_end": "overshoot", "cor_straight_at_turn": "turn", "cor_turn_under": "turn", "cor_turn_over": "turn", "cor_wrong_exit": "turn",
       "cor_drift": "other", "cor_lane_change": "other"}


def primaries(pkl: str) -> dict:
    """rf_tax extract -> {scene: classify primary label}; {} when the extract is missing."""
    if not os.path.exists(pkl):
        return {}
    import rf_tax as TX
    out = {}
    for o in pickle.load(open(pkl, "rb"))["rolls"]:
        if "ctrl" in o:
            try:
                out[o["scene"]] = TX.classify(o)[0].get("primary", "")
            except Exception as e:  # noqa: BLE001  (one bad rollout must not drop the read's split)
                out[o["scene"]] = f"err:{type(e).__name__}"
    return out


def config(root: str) -> dict:
    f = Path(root) / "config.txt"
    kv = dict(x.split("=", 1) for x in f.read_text().splitlines() if "=" in x) if f.exists() else {}
    return dict(serve=kv.get("SERVE", "?") or "(none)", wiz=kv.get("WIZ", "?"), fast=kv.get("ALPASIM_FAST", "unset (pre-switch: official set)"),
                code=kv.get("code", "?"))


def one(job):
    k, lab, root, tax = job
    R, bad = load(root)
    return k, lab, root, R, bad, primaries(f"{tax}/{k}.pkl"), config(root)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["stored", "table"]), a.add_argument("--tax", default=""), a.add_argument("--out", default="")
    a.add_argument("--json", default="")
    a = a.parse_args()
    if a.cmd == "stored":
        print("\n".join(f"{k} {D}/{r}" for k, _, r in STORED))
        return
    with ProcessPoolExecutor(min(len(ROWS), 16)) as ex:
        res = {r[0]: r for r in ex.map(one, [(k, lab, root, a.tax) for k, lab, root in ROWS])}
    clean = {k: set(v[3]) - set(v[4]) for k, v in res.items()}
    common = set.intersection(*clean.values())
    J, T = {}, []
    hdr = ("| read | n | mean | mean (common) | zeros | AF-front | AF-side | offroad | corridor | overshoot / route-turn / other | other zeros "
           "| rear-hit partial | slow partial | other partial | RPC excluded |")
    T += [hdr, "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|"]
    for k, lab, root, R, bad, P, cfg in (res[k] for k, _, _ in ROWS):
        S = sorted(clean[k])
        sc = np.array([float(R[s].get("score") or 0.0) for s in S])
        ty = [ptype(R[s]) for s in S]
        z = Counter(zclass(t) for t, x in zip(ty, sc) if x == 0)
        p = Counter(t for t, x in zip(ty, sc) if 0 < x < 0.98)
        cz = [s for s, t, x in zip(S, ty, sc) if x == 0 and zclass(t) == "corridor"]
        cs = Counter(COR.get(P.get(s, ""), "?") for s in cz) if P else None
        cm = float(np.mean([float(R[s].get("score") or 0.0) for s in sorted(common)]))
        oth = sum(v for c, v in z.items() if c not in ("AF-front", "AF-side", "offroad", "corridor"))
        J[k] = dict(label=lab, root=root, **cfg, n=len(S), mean=float(sc.mean()), mean_common=cm, zeros=int((sc == 0).sum()), zc=dict(z), pc=dict(p),
                    cor_split=dict(cs) if cs is not None else None, rpc=sorted(bad), scores={s: float(R[s].get("score") or 0.0) for s in S},
                    types={s: t for s, t in zip(S, ty)})
        split = "-" if cs is None else f"{cs['overshoot']} / {cs['turn']} / {cs['other']}" + (f" (+{cs['?']} ?)" if cs["?"] else "")
        T.append(f"| {lab} | {len(S)} | {sc.mean():.4f} | {cm:.4f} | {int((sc == 0).sum())} | {z['AF-front']} | {z['AF-side']} | {z['offroad']} | "
                 f"{z['corridor']} | {split} | {oth} | {p['rear-hit']} | {p['slow']} | {sum(p.values()) - p['rear-hit'] - p['slow']} | "
                 f"{', '.join(s[7:15] for s in sorted(bad)) or '-'} |")
    T += ["", f"Mean (common): the {len(common)} scenes every row above scored without a driver RPC error.", "",
          "## Paired steps (scenes both reads scored cleanly)", "", "| step | n | A mean | B mean | A - B, 95 % CI | zeros A - B |", "|---|--:|--:|--:|---|--:|"]
    for x, y in STEPS:
        S = sorted(clean[x] & clean[y])
        sa = np.array([J[x]["scores"][s] for s in S])
        sb = np.array([J[y]["scores"][s] for s in S])
        q = stats.paired(sa, sb, groups=np.array(["_".join(s.split("_")[:2]) for s in S]))
        T.append(f"| {J[x]['label']} - {J[y]['label']} | {len(S)} | {sa.mean():.4f} | {sb.mean():.4f} | {stats.fmt(q, '+.4f')} | "
                 f"{int((sa == 0).sum()) - int((sb == 0).sum()):+d} |")
    T += ["", "## Read roots and serve flags", "", "| read | root | SERVE | WIZ | ALPASIM_FAST | code |", "|---|---|---|---|---|---|"]
    for k, lab, root in ROWS:
        c = J[k]
        T.append(f"| {lab} | `{root.replace(D, '$DATA_DIR/runs')}` | `{c['serve']}` | `{c['wiz']}` | {c['fast']} | {c['code']} |")
    out = "\n".join(T) + "\n"
    print(out)
    if a.out:
        Path(a.out).write_text(out)
    if a.json:
        Path(a.json).write_text(json.dumps(J, default=str))


if __name__ == "__main__":
    main()
