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
    r.add_argument("--priority", type=float, default=0.0)
    r.add_argument("--gpus", default="", help="restrict to cards (comma list); default: any")
    r.add_argument("--wait", action="store_true")
    r.add_argument("--dry", action="store_true")
    s = sp.add_parser("status")
    s.add_argument("--model", nargs="*", default=[])
    s.add_argument("--bench", default="")
    s.add_argument("--preset", default="exam")
    s.add_argument("--wait", action="store_true")
    p = sp.add_parser("report")
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
        dirs = []
        kw = {k: v for k, v in (("stall_s", a.stall_s), ("timeout_s", a.timeout_s)) if v is not None}
        gpus = [int(g) for g in a.gpus.split(",") if g] or None
        for b in a.bench:
            for mdl in a.model:
                for pr in (a.preset if b == "hugsim" else ["exam"]):
                    kwb = dict(preset=pr, scenarios=a.scenarios, workers=a.workers, jobs=a.jobs, **kw) if b == "hugsim" else \
                        dict(subset=a.subset, shards=a.shards)
                    dirs.append(_run(mdl, b, dry=a.dry, priority=a.priority, gpus=gpus, **kwb))
        print("\n".join(map(str, dirs)))
        if a.wait and not a.dry:
            return 0 if RN.wait(dirs) else 1
        return 0
    if a.cmd == "status":
        if a.model:
            from . import run_dir
            dirs = [run_dir(m, a.bench, a.preset) for m in a.model]
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
        report(a.bench, a.arms, a.vs, preset=a.preset, out=a.out, strata=not a.no_strata, units_subset=sub)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
