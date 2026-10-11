#!/usr/bin/env python3
"""Lane JOINT phase 1 pilot checklist (plans/2026-10-11-joint1-prereg.md, "Pilot"): one pilot checkpoint, its training log and its 48-scene smoke.

  joint1_pilot_check.py --tag JAPS-P300 --driver ap2 --expect 102,13,13 --terms hinge,agent,agent_ho,road_ho --out F.json
Checks: (training) the log's composition line shows the registered rows per batch, the last loss line is finite and carries every listed term,
the checkpoint's `ap2` record matches the driver (ap2 <-> std alpasim); (smoke) 48 rollouts with a score, 480 drive records, no driver error
(ot2_loop already fails on missing scenes / driver errors), every served plan finite. Exit 0 = pass, 2 = fail (the reasons are in the json).
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--driver", required=True, choices=["ap2", "sh30"])
    ap.add_argument("--expect", required=True, help="normal,yaw-rate,hinge-only rows per batch")
    ap.add_argument("--terms", default="hinge", help="loss terms the last loss line must carry")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad, res = [], dict(tag=a.tag, driver=a.driver)
    runs = sorted((DATA / "runs/alpasim").glob(f"joint-train-{a.tag}/*/log.txt"))
    if not runs:
        bad.append("no training run dir")
    else:
        log = runs[-1].read_text()
        n, y, h = (int(x) for x in a.expect.split(","))
        m = re.search(r"per batch of (\d+): (\d+) normal, (\d+) yaw-rate, (\d+) hinge-only", log)
        res["composition"] = m.groups() if m else None
        if not m or (int(m.group(2)), int(m.group(3)), int(m.group(4))) != (n, y, h):
            bad.append(f"composition {res['composition']} != {(n, y, h)}")
        steps = re.findall(r"step (\d+): (.*?); ([\d.]+) it/s, ([\d.]+) GB", log)
        if not steps:
            bad.append("no loss line")
        else:
            st, body, its, gb = steps[-1]
            vals = dict((k.strip(), float(v)) for k, v in (x.rsplit(" ", 1) for x in body.split(", ")))
            res.update(last_step=int(st), losses=vals, it_s=float(its), peak_gb=float(gb))
            miss = [t for t in a.terms.split(",") if t and t not in vals]
            if miss:
                bad.append(f"terms missing from the loss line: {miss}")
            if not all(math.isfinite(v) for v in vals.values()):
                bad.append("non-finite loss")
        dv = re.findall(r"dev @ (\d+): (.*)", log)
        res["dev"] = dv[-1][1] if dv else None
    try:
        import torch
        ck = torch.load(DATA / "runs/op_parity/runs" / a.tag / "ckpt-final.pt", map_location="cpu", weights_only=False)
        fam = "ap2" if (ck.get("ap2") or {}).get("std", "navsim") == "alpasim" else "sh30"
        res["family"] = fam
        if fam != a.driver:
            bad.append(f"checkpoint family {fam} != driver {a.driver}")
    except Exception as e:                                           # noqa: BLE001
        bad.append(f"checkpoint: {e!r}")
    man = DATA / "runs/alpasim/joint1" / f"smoke-{a.tag}" / "manifest.json"
    if not man.exists():
        bad.append("no smoke manifest")
    else:
        dirs = json.loads(man.read_text()).get(f"{a.tag}@smoke48", [])
        n_roll = n_rec = n_bad = 0
        for d in map(Path, dirs):
            s = json.loads((d / "aggregate/results-summary.json").read_text())
            n_roll += sum(r.get("score") is not None for r in s["rollouts"])
            for ln in (d / "driver-logs/drive.jsonl").read_text().splitlines():
                r = json.loads(ln)
                n_rec += 1
                p = r.get("poses") or r.get("plan") or []
                if any(isinstance(v, float) and not math.isfinite(v) for row in p for v in (row if isinstance(row, list) else [row])):
                    n_bad += 1
        res.update(smoke_rollouts=n_roll, drive_records=n_rec, nonfinite_plans=n_bad)
        if n_roll != 48:
            bad.append(f"smoke rollouts {n_roll} != 48")
        if n_rec < 470:
            bad.append(f"drive records {n_rec} < 470")
        if n_bad:
            bad.append(f"{n_bad} non-finite served plans")
    res["fail"], res["pass"] = bad, not bad
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if not bad else 2)


if __name__ == "__main__":
    main()
