"""python -m jevdrive.bench {models, sets, run, status, report}   (docs/bench.md)

  run     --model M [M ...] --bench navtest|navhard|hugsim [--preset exam|spec|spec_plan] [--scenarios all64|turn23|...]
          [--shards K] [--workers W] [--jobs K] [--subset SPLIT] [--wait] [--dry]
  status  [--model M ... --bench B --preset P]      (no args: every run dir with a live job)
  report  --bench B --arms A [A ...] [--vs REF ...] [--preset P] [--out DIR] [--scenarios SET]
          arm = label=spec+spec (seeds) or one spec; e.g. --arms P2=P2-F-s0+P2-F-s1 P0 --vs WA-JEPA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jevdrive.bench")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("models")
    sp.add_parser("sets")
    r = sp.add_parser("run")
    r.add_argument("--model", nargs="+", required=True)
    r.add_argument("--bench", nargs="+", required=True, choices=["navtest", "navhard", "hugsim"])
    r.add_argument("--preset", nargs="+", default=["exam"])
    r.add_argument("--scenarios", default="all64")
    r.add_argument("--subset", default="", help="navtest: a jevdrive.data.splits token / log split (default: all of navtest)")
    r.add_argument("--shards", type=int, default=0, help="navtest scoring shards (default: one per ~12 cores)")
    r.add_argument("--workers", type=int, default=6, help="HUGSIM scenario slots per worker job")
    r.add_argument("--jobs", type=int, default=0, help="HUGSIM worker jobs (default: one per card)")
    r.add_argument("--stall-s", type=float, default=None)
    r.add_argument("--timeout-s", type=float, default=None)
    r.add_argument("--retries", type=int, default=None, help="HUGSIM additional scenario attempts (default: 2)")
    r.add_argument("--opts", default="{}", help="HUGSIM agent options (JSON object)")
    r.add_argument("--controller", default="", help="HUGSIM controller tree; defaults to the preset's tree")
    r.add_argument("--controller-env", default="{}", help="HUGSIM OP_CTRL / OP_CTRL_LONG / LOWSPEED_CTRL / LOWSPEED_SEL objects")
    r.add_argument("--onnx", default="", help="HUGSIM custom serving ONNX (part of the run identity)")
    r.add_argument("--repeat", nargs="+", default=[""], help="independent repeat labels; the same label resumes")
    r.add_argument("--publish-out", default="", help="compatibility export: legacy HUGSIM results directory (requires --wait)")
    r.add_argument("--publish-tag", default="", help="compatibility export: legacy results.csv tag")
    r.add_argument("--in-pool", action="store_true", help="legacy leased HUGSIM worker: execute shared stages without resubmitting")
    r.add_argument("--priority", type=float, default=0.0)
    r.add_argument("--gpus", default="", help="restrict to cards (comma list); default: any")
    r.add_argument("--wait", action="store_true")
    r.add_argument("--wait-timeout-s", type=float, default=0, help="orchestrator deadline: cancel this run's pool jobs when waiting expires")
    r.add_argument("--dry", action="store_true")
    s = sp.add_parser("status")
    s.add_argument("--model", nargs="*", default=[])
    s.add_argument("--bench", default="")
    s.add_argument("--preset", default="exam")
    s.add_argument("--wait", action="store_true")
    p = sp.add_parser("report")
    for parser in (s, p):
        parser.add_argument("--opts", default="{}")
        parser.add_argument("--controller", default="")
        parser.add_argument("--controller-env", default="{}")
        parser.add_argument("--repeat", default="")
        parser.add_argument("--onnx", default="")
    p.add_argument("--bench", required=True, choices=["navtest", "navhard", "hugsim"])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--vs", nargs="*", default=[])
    p.add_argument("--preset", default="exam")
    p.add_argument("--out", default="")
    p.add_argument("--scenarios", default="", help="HUGSIM: restrict the tables to a scenario set (e.g. turn23)")
    p.add_argument("--no-strata", action="store_true")
    a = ap.parse_args(argv)

    from . import runner as RN
    if a.cmd == "models":
        from .models import listing
        for x in listing():
            print(f"{x['name']:<22} {x['family']:<7} {x['frames']:<6} {x['weights'][-60:]:<60} {x['note'][:70]}")
        return 0
    if a.cmd == "sets":
        from .sets import HUGSIM_SETS, hugsim_scenarios
        for k in HUGSIM_SETS:
            try:
                print(f"hugsim {k:<8} {len(hugsim_scenarios(k)):>3} scenarios  {HUGSIM_SETS[k]}")
            except OSError as e:
                print(f"hugsim {k:<8} missing ({e})")
        print("navtest / navhard: all tokens, or --subset <jevdrive.data.splits name> (python -c 'from jevdrive.data import splits; print(splits.available(\"navsim\"))')")
        return 0
    if a.cmd == "run":
        from . import run as _run
        if a.wait_timeout_s < 0 or (a.wait_timeout_s and not a.wait):
            ap.error("--wait-timeout-s requires --wait and a positive duration")
        if a.retries is not None and a.retries < 0:
            ap.error("--retries must be non-negative")
        if a.publish_out or a.publish_tag:
            if not (a.publish_out and a.publish_tag and a.wait and a.bench == ["hugsim"] and
                    len(a.model) == len(a.preset) == len(a.repeat) == 1):
                ap.error("--publish-out/--publish-tag require --wait and one HUGSIM model/preset/repeat")
        if a.in_pool:
            from .compat import in_pool
            def _run(model, bench, dry=False, priority=0, gpus=None, **kw):
                return in_pool(model, bench, **kw)
            if a.dry or a.wait_timeout_s:
                ap.error("--in-pool does not support --dry or --wait-timeout-s; run deadline orchestration outside a lease")
        dirs = []
        kw = {k: v for k, v in (("stall_s", a.stall_s), ("timeout_s", a.timeout_s), ("retries", a.retries)) if v is not None}
        gpus = [int(g) for g in a.gpus.split(",") if g] or None
        for b in a.bench:
            for mdl in a.model:
                for pr in (a.preset if b == "hugsim" else ["exam"]):
                    for repeat in (dict.fromkeys(a.repeat) if b == "hugsim" else [""]):
                        kwb = dict(preset=pr, scenarios=a.scenarios, workers=a.workers, jobs=a.jobs, opts=a.opts,
                                   controller=a.controller, controller_env=a.controller_env, repeat=repeat, onnx=a.onnx, **kw) if b == "hugsim" else \
                            dict(subset=a.subset, shards=a.shards)
                        dirs.append(_run(mdl, b, dry=a.dry, priority=a.priority, gpus=gpus, **kwb))
        print("\n".join(map(str, dirs)))
        if a.wait and not a.dry:
            if not a.in_pool and not RN.wait(dirs, timeout_s=a.wait_timeout_s):
                if any((d / "WAIT_TIMEOUT").exists() for d in dirs):
                    from ..cl import pool
                    ids = [j for d in dirs for j in json.loads((d / "jobs.json").read_text()).values() if j != "done"]
                    pool.wait(ids, poll_s=5)
                    if a.publish_out:
                        from .compat import publish_hugsim
                        publish_hugsim(dirs[0], a.publish_out, a.publish_tag, allow_partial=True)
                    return 124
                return 1
            if a.publish_out:
                from .compat import publish_hugsim
                publish_hugsim(dirs[0], a.publish_out, a.publish_tag)
        return 0
    if a.cmd == "status":
        if a.model:
            from . import run_dir
            dirs = [run_dir(m, a.bench, a.preset, opts=a.opts, controller=a.controller,
                            controller_env=a.controller_env, repeat=a.repeat, onnx=a.onnx) for m in a.model]
        else:
            dirs = sorted({f.parent for f in RN.bench_root().glob("*/*/jobs.json")})
            dirs = [d for d in dirs if not (d / "DONE").exists() or (d / "ERROR").exists()] or dirs[-10:]
        if a.wait:
            return 0 if RN.wait(dirs) else 1
        for d in dirs:
            st = RN.state(d)
            line = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else ""
            flag = "DONE" if (d / "DONE").exists() else ("ERROR" if (d / "ERROR").exists() else "running")
            print(f"{d.parent.name}/{d.name} [{flag}] " + " ".join(f"{k}={v}" for k, v in st.items()))
            if line:
                print("   ", line)
            if (d / "DONE").exists() and (d / "summary.json").exists():
                sm = json.loads((d / "summary.json").read_text())
                print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sm.items() if k not in ("failed",)})
        return 0
    if a.cmd == "report":
        from .tables import report
        sub = None
        if a.scenarios:
            from .sets import hugsim_scenarios
            sub = [Path(s).stem for s in hugsim_scenarios(a.scenarios)]
        report(a.bench, a.arms, a.vs, preset=a.preset, out=a.out, strata=not a.no_strata, units_subset=sub,
               opts=a.opts, controller=a.controller, controller_env=a.controller_env, repeat=a.repeat, onnx=a.onnx)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
