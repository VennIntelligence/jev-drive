#!/usr/bin/env python3
"""Lane ONPOL-NUPLAN pilot gate table (plans/2026-10-11-onpol-prereg.md, "Pilot"): the training-side gates of the four 3 000-step pilots, read from
their logs and scalars (run dirs $DATA_DIR/runs/onpol_nuplan/on-train-<tag>/). The 48-scene closed-loop smoke is the lane's own step and not read here.

  on_pilot_check.py [--tags SIMIT-ON-P-s0 SSELF-ON-P-s0 SBOTH-ON-P-s0] [--none SNONE-ON-P-s0] [--out experiments/onpol_nuplan/results/pilot]
Gates per arm: every logged loss finite and the run complete; composition 102 normal / 13 hinge-only / 13 on-policy in the log line (ON-NONE: 115 / 13 / 0);
dev ADE at the last step <= the ON-NONE pilot's + 0.010 m (the archived S, 0.545, and the step-0 value are shown beside it); dev_drift_off <= 0.30.
Descriptive: on the on-policy rows, hinge activation (share of rows with a non-zero hinge, mean hinge) and the imitation error (loss, mean displacement
to the target) over steps 2701-3000 against 201-400, from the 25-step means of the trainer. Writes <out>/pilot_gates.{json,md}; exit 0 = every arm
passes, 2 = an arm fails (reasons in the json), 3 = the ON-NONE pilot is missing or incomplete.

Round 2 (plans/2026-10-12-onpol2-prereg.md, B "Pilot"; on_train_y.sh): the Y family on the YR10m10-F base, reference = the YNONE pilot:
  on_pilot_check.py --tags YSELFnoL-ON-P-s0 --none YNONE-ON-P-s0 --ho-k 0 --yr-k 13 --archived 0.539 --archived-name YR10m10-F-s0 --out .../results/pilot_y
composition 102 normal / 13 yaw-rate / 0 hinge-only / 13 on-policy (the reference: 115 / 13 / 0 / 0). The defaults are round 1's.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
DESC = ("on/imit", "on/ade", "on/agent_pos", "on/agent_mean", "on/agent_posmean", "on/road_pos", "on/road_mean", "on/road_posmean")
WIN = {"201-400": (201, 400), "2701-3000": (2701, 3000)}


def read(tag, exp, steps):
    runs = sorted((DATA / "runs" / exp).glob(f"on-train-{tag}/*/events.jsonl"))
    if not runs:
        return None
    sc, end = {}, None
    for ln in runs[-1].read_text().splitlines():
        e = json.loads(ln)
        if e.get("kind") == "scalar":
            sc.setdefault(e["tag"], {})[int(e["step"])] = e["value"]
        elif e.get("kind") == "end":
            end = e.get("ok")
    log = (runs[-1].parent / "log.txt").read_text()
    m = re.search(r"per batch of (\d+): (\d+) normal, (?:(\d+) yaw-rate, )?(\d+) hinge-only .*?\), (\d+) on-policy \(mode (\w+)", log)
    yr, m = (int(m.group(3) or 0), re.match(r"(.*)\|(.*)\|(.*)\|(.*)\|(.*)", "|".join(m.group(1, 2, 4, 5, 6)))) if m else (None, None)
    loss = {q[5:]: v for q, v in sc.items() if q.startswith("loss/")}
    last = max(sc.get("dev/ade", {0: None}))
    win = lambda q, lo, hi: [v for s, v in loss.get(q, {}).items() if lo <= s - 12 <= hi] or None  # noqa: E731  a 25-step mean is placed at its centre
    r = dict(tag=tag, run=str(runs[-1].parent), ended_ok=end, composition=[int(x) for x in m.groups()[:4]] if m else None, mode=m.group(5) if m else None,
             steps=max(max(v) for v in loss.values()) if loss else 0, finite=all(math.isfinite(x) for v in loss.values() for x in v.values()),
             dev_step=last, dev_ade=sc.get("dev/ade", {}).get(last), dev_drift_off=sc.get("dev/drift_off", {}).get(last), dev0_ade=sc.get("dev/ade", {}).get(0),
             it_s=(sc.get("throughput/steps_per_s") or {0: None})[max(sc.get("throughput/steps_per_s") or {0: 0})], on_hash=dict(re.findall(r"on_hash @ (\d+): ([0-9a-f]+)", log)),
             ckpt=(DATA / "runs/op_parity/runs" / tag / "ckpt-final.pt").exists(), **({"yaw_rate": yr} if yr else {}),
             on={w: {q[3:]: sum(x) / len(x) for q in DESC if (x := win(q, lo, hi))} for w, (lo, hi) in WIN.items()})
    r["complete"] = bool(end and r["steps"] == steps and last == steps and r["ckpt"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["SIMIT-ON-P-s0", "SSELF-ON-P-s0", "SBOTH-ON-P-s0"])
    ap.add_argument("--none", default="SNONE-ON-P-s0")
    ap.add_argument("--exp", default="onpol_nuplan")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--ho-k", type=int, default=13, help="hinge-only rows per batch of the recipe (S: 13; the YR base: 0)")
    ap.add_argument("--yr-k", type=int, default=0, help="yaw-rate rows per batch of the recipe (the YR base: 13)")
    ap.add_argument("--archived", type=float, default=0.545, help="dev ADE of the archived P2H10S-F-s0 (shown, not a gate)")
    ap.add_argument("--archived-name", default="P2H10S-F-s0")
    ap.add_argument("--ade-tol", type=float, default=0.010)
    ap.add_argument("--drift-max", type=float, default=0.30)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/pilot"))
    a = ap.parse_args()
    none = read(a.none, a.exp, a.steps)
    arms, ref = {}, none["dev_ade"] if none and none["complete"] else None
    for tag in [a.none] + a.tags:
        r = none if tag == a.none else read(tag, a.exp, a.steps)
        if r is None:
            arms[tag] = dict(tag=tag, fail=["no run dir"], **{"pass": False})
            continue
        on = tag != a.none
        want = [128, 128 - a.ho_k - a.yr_k - a.k * on, a.ho_k, a.k * on]
        bad = []
        if not r["complete"]:
            bad.append(f"incomplete: ended_ok {r['ended_ok']}, steps {r['steps']}, dev at {r['dev_step']}, checkpoint {r['ckpt']}")
        if not r["finite"]:
            bad.append("non-finite loss")
        if r["composition"] != want:
            bad.append(f"composition {r['composition']} != {want}")
        if r.get("yaw_rate", 0) != a.yr_k:
            bad.append(f"yaw-rate rows per batch {r.get('yaw_rate', 0)} != {a.yr_k}")
        if on and ref is None:
            bad.append("no complete ON-NONE pilot to compare with")
        elif r["dev_ade"] is None or (on and not r["dev_ade"] <= ref + a.ade_tol):
            bad.append(f"dev ADE {r['dev_ade']} > ON-NONE {ref} + {a.ade_tol}")
        if r["dev_drift_off"] is None or not r["dev_drift_off"] <= a.drift_max:
            bad.append(f"dev_drift_off {r['dev_drift_off']} > {a.drift_max}")
        arms[tag] = r | dict(fail=bad, **{"pass": not bad})
    hs = {json.dumps(arms[t].get("on_hash", {}), sort_keys=True) for t in a.tags if arms[t].get("on_hash")}
    res = dict(none=a.none, none_dev_ade=ref, archived_dev_ade=a.archived, ade_tol=a.ade_tol, drift_max=a.drift_max, same_on_policy_rows=len(hs) == 1, arms=arms)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_gates.json").write_text(json.dumps(res, indent=1) + "\n")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    md = [f"# ONPOL-NUPLAN pilots: training-side gates ({a.steps} steps, seed 0)", "",
          f"Dev ADE gate: <= ON-NONE ({f(ref)}) + {a.ade_tol} m; archived {a.archived_name} {a.archived}; dev_drift_off <= {a.drift_max}. "
          f"On-policy rows identical across the arms (running sha1 of the drawn indices): {res['same_on_policy_rows']}.", "",
          f"| Tag | Mode | Rows per batch (normal / {'yaw-rate / ' if a.yr_k else ''}hinge-only / on-policy) | Finite | Dev ADE at 0 | Dev ADE | - ON-NONE | drift_off | it/s | Gate |", "|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|"]
    for t, r in arms.items():
        if "composition" not in r:
            md.append(f"| {t} | | | | | | | | | FAIL: no run dir |")
            continue
        c = r["composition"]
        comp, dd = " / ".join(map(str, c[1:2] + ([r.get("yaw_rate", 0)] if a.yr_k else []) + c[2:])) if c else "-", "-" if ref is None or r["dev_ade"] is None else f"{r['dev_ade'] - ref:+.3f}"
        md.append(f"| {t} | {r['mode']} | {comp} | {r['finite']} | {f(r['dev0_ade'])} | {f(r['dev_ade'])} | {dd} | {f(r['dev_drift_off'])} | {f(r['it_s'], 2)} | "
                  + ("pass" if r["pass"] else "FAIL: " + "; ".join(r["fail"])) + " |")
    md += ["", "Descriptive, on-policy rows (means of the trainer's 25-step means; `pos` = share of rows with a non-zero hinge, `mean` over all rows, "
           "`posmean` over the active ones; imit = the imitation loss, ade = mean displacement to the target in m):", "",
           "| Tag | Steps | " + " | ".join(q[3:] for q in DESC) + " |", "|:--|:--|" + "--:|" * len(DESC)]
    for t in a.tags:
        for w in WIN:
            md.append(f"| {t} | {w} | " + " | ".join(f(arms[t].get("on", {}).get(w, {}).get(q[3:]), 4) for q in DESC) + " |")
    (out / "pilot_gates.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    sys.exit(3 if ref is None else 0 if all(r["pass"] for r in arms.values()) else 2)


if __name__ == "__main__":
    main()
