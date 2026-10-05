#!/usr/bin/env python
"""factor_wm G1 chain (box, .venv python): submits every stage to the GPU pool up front with `after` (plans/2026-10-05-stage1-prereg.md
section 3.3); the pool places the jobs. Resumable: a job whose output exists is not submitted (and drops out of `after`).

  S1    train 2 400 steps on logged states (no rollouts)                                   | independent
  S2/S3 r1 collect (shipped; 3 shards) -> train r1 800 -> r2 collect (r1 model) -> train r2 1 600 (r1 + r2) -> r3 collect (r2 model)
        -> train final 2 400 (r1 + r2 + r3), all from shipped. S2 = kind closedlat, 10 steps (decision 132's engine); S3 = kind closed, 40 steps
  per arm (S0 = shipped, S1, S2, S3): WOD engine eval (g0b + g1s, kind closed, plane); serving ONNX (op_l_onnx build); guard driver
        (guard.py --mode full --lines hugsim,navtest,navhard) as a small pool job that submits its own units
  report after everything: fw_g1report.py -> $DATA_DIR/runs/factor_wm/report/g1.{md,json}
Each collect / train batch carries a --preflight smoke (2 clips / 5 steps).

  .venv/bin/python experiments/factor_wm/scripts/fw_g1chain.py [--dry-run]
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
PY = str(DD / "envs/op-train/bin/python")
VPY = str(REPO / ".venv/bin/python")
S = "experiments/factor_wm/scripts/"
R = DD / "runs/factor_wm"
J = R / "jobs/g1"
STEPS = {1: 800, 2: 1600, 3: 2400}
SHARDS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from jevdrive.cl import pool as P
    ids = {}

    def sub(name, cmd, done, after=(), vram=14.0, cpu=18, train=False, preflight=None):
        after = [x for x in after if x]
        if done is not None and Path(done).exists():
            print(f"{name}: done ({done})")
            return None
        full = f"{cmd} && test -f {shlex.quote(str(done))}" if done is not None else cmd
        kw = dict(owner="factor_wm G1", vram_gb=vram, cpu=cpu, train=train, after=after, log_dir=str(J / name), cwd=str(REPO),
                  env={"OMP_NUM_THREADS": "8"})
        if a.dry_run:
            print(f"{name}: would submit after {after}\n  {full}" + (f"\n  preflight: {preflight}" if preflight else ""))
            return f"<{name}>"
        if preflight:
            res = {k: kw[k] for k in ("owner", "vram_gb", "cpu", "train", "cwd", "env")}
            pf = P.preflight(preflight, f"fw-{name}", log_dir=str(J / name / "preflight"), **res)
            kw["after"] = kw["after"] + [pf]
            print(f"{name} preflight -> {pf}", flush=True)
        jid = P.submit(full, name=f"fw-{name}", **kw)
        print(f"{name} -> {jid}", flush=True)
        ids[name] = jid
        return jid

    def ck(tag):
        return R / "runs" / tag / "ckpt-final.pt"

    # S1
    last = {"S1": sub("train-S1", f"{PY} {S}fw_train.py --tag S1 --rolls none --steps 2400 --workers 14", ck("S1"), vram=30, cpu=16, train=True,
                      preflight=f"{PY} {S}fw_train.py --tag pf-S1 --rolls none --steps 5 --workers 4")}
    # S2 / S3 DAgger
    for arm, kind, kl in (("S2", "closedlat", 10), ("S3", "closed", 0)):
        low = arm.lower()
        model, prev = "shipped", None
        rolls = []
        for r in (1, 2, 3):
            tag = f"{low}r{r}"
            klarg = f" --kl {kl}" if kl else ""
            col = [sub(f"col-{tag}-{i}", f"{PY} {S}fw_roll.py collect --model {model} --kind {kind}{klarg} --tag {tag} --seed {r} --shard {i}/{SHARDS}",
                       R / "roll" / tag / f"train-collect-{i}of{SHARDS}.npz", after=[prev],
                       preflight=(f"{PY} {S}fw_roll.py collect --model shipped --kind {kind}{klarg} --tag pf-{tag} --seed {r} --limit 2"
                                  if i == 0 and r == 1 else None))
                   for i in range(SHARDS)]
            rolls.append(tag)
            ttag = f"{arm}-r{r}" if r < 3 else arm
            prev = sub(f"train-{ttag}", f"{PY} {S}fw_train.py --tag {ttag} --rolls {','.join(rolls)} --steps {STEPS[r]} --workers 14", ck(ttag),
                       after=col, vram=30, cpu=16, train=True,
                       preflight=(f"{PY} {S}fw_train.py --tag pf-{low} --rolls none --steps 5 --workers 4" if r == 1 else None))
            model = str(ck(ttag))
        last[arm] = prev
    # evals, ONNX, guards
    finals = []
    for arm in ("S0", "S1", "S2", "S3"):
        model = "shipped" if arm == "S0" else str(ck(arm))
        dep = [] if arm == "S0" else [last[arm]]
        finals.append(sub(f"eval-{arm}", f"{PY} {S}fw_roll.py eval --model {model} --tag eval-{arm} --sets g0b,g1s",
                          R / "roll" / f"eval-{arm}" / "g1s-eval-0of1.npz", after=dep))
        if arm == "S0":
            continue
        onnx = R / "onnx" / f"fw-{arm}.onnx"
        o = sub(f"onnx-{arm}", f"{PY} experiments/op_adapt_l/scripts/op_l_onnx.py build --ckpt {ck(arm)} --out {onnx} --no-adapter", onnx,
                after=dep, vram=1, cpu=4)
        finals.append(sub(f"guard-{arm}", f"{VPY} experiments/op_guard/scripts/guard.py --candidate fw-{arm} --mode full --lines hugsim,navtest,navhard",
                          DD / "runs/op_guard" / f"fw-{arm}" / "full/lines/hugsim.json", after=[o], vram=1, cpu=2))
    sub("report", f"{PY} {S}fw_g1report.py --out {R / 'report/g1'}", None, after=finals, vram=1, cpu=8)
    if not a.dry_run:
        (J).mkdir(parents=True, exist_ok=True)
        json.dump(ids, open(J / "ids.json", "w"), indent=1)


if __name__ == "__main__":
    main()
