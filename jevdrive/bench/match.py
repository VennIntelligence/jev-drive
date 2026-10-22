"""Paired match: a list of drivers against one reference on several boards, every cell a bench run (docs/bench.md "match").

One command submits the whole matrix and returns. Every allowed driver x board, and the reference on every board, is its own
`bench.run` (so finished cells are reused and a re-run resumes): HUGSIM runs split into one worker job per card each, so N drivers
fill every free card; navtest scoring shards are sized from the box's cores (navsim.default_shards). One CPU report job, chained
`after` every stage of every cell, then writes <out>/<board>_{arms,paired,strata}.{md,csv} (jevdrive.bench.tables.report: HUGSIM
paired over scenarios clustered by scene, with the HD parts) and <out>/match.{md,csv} (one row per driver x board: driver, reference,
difference [95 % CI]), and touches <out>/DONE. A cell the registry does not allow (Model.benches; e.g. route-memory checkpoints are
HUGSIM-only and their lineage trained on navtest logs) is never run; match.md lists it with the reason.

  python -m jevdrive.bench match --drivers A B .. --ref WA-JEPA [--also P0] --boards hugsim[:preset] navtest navhard
                                 [--scenarios all64] [--repeat LABEL] --out DIR [--priority P] [--dry]
  python -m jevdrive.bench match ... --report-only     (what the report job runs: tables of the finished cells)
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

METRIC = {"hugsim": "HD-Score", "navtest": "EPDMS x 100", "navhard": "two-stage EPDMS"}


def board(b: str) -> tuple:
    """'hugsim:exam' / 'hugsim' / 'navtest' -> (bench, preset)."""
    bench, _, preset = b.partition(":")
    if bench not in METRIC:
        raise ValueError(f"board {b!r}: one of {', '.join(METRIC)} (hugsim takes :<preset>)")
    return bench, (preset or "exam")


def cells(drivers, ref: str, also=(), boards=("hugsim",)) -> tuple:
    """-> (runs [(model, bench, preset)], skipped [(model, board, reason)]). A board whose reference is not allowed is skipped whole."""
    from .models import resolve
    runs, skipped = [], []
    for b in boards:
        bench, preset = board(b)
        r = resolve(ref)
        if bench not in r.benches or (r.family == "wajepa" and bench == "hugsim" and preset != "exam"):
            skipped += [(d, b, f"reference {ref} not served on {b}") for d in drivers]
            continue
        ok = []
        for mdl in dict.fromkeys(drivers):
            m = resolve(mdl, check=True)
            if bench not in m.benches:
                skipped.append((mdl, b, m.note or f"registered for {m.benches}"))
            else:
                ok.append(mdl)
        if ok:                                                       # the reference and the extra arms only where a driver runs
            runs += [(mdl, bench, preset) for mdl in dict.fromkeys([*ok, ref, *also]) if bench in resolve(mdl, check=True).benches]
    return runs, skipped


def _kw(bench, preset, scenarios, repeat, workers, jobs):
    return dict(preset=preset, scenarios=scenarios, repeat=repeat, workers=workers, jobs=jobs) if bench == "hugsim" else {}


def submit(drivers, ref, out, also=(), boards=("hugsim",), scenarios="all64", repeat="", workers=6, jobs=0, priority=0.0, dry=False,
           argv=None) -> str:
    """Queue every cell and the report job; returns the report job id."""
    from . import run as bench_run
    from ..cl import pool as P
    from .runner import REPO, py
    runs, skipped = cells(drivers, ref, also, boards)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    after, dirs = [], []
    for mdl, bench, preset in runs:
        d = bench_run(mdl, bench, dry=dry, priority=priority, **_kw(bench, preset, scenarios, repeat, workers, jobs))
        dirs.append(str(d))
        if not dry:
            after += [j for j in json.loads((Path(d) / "jobs.json").read_text()).values() if j != "done"]
    for mdl, b, why in skipped:
        print(f"skip {mdl} on {b}: {why}")
    cmd = [py("jev"), "-m", "jevdrive.bench", "match", *(argv or []), "--report-only"]
    (out / "match.json").write_text(json.dumps(dict(drivers=list(drivers), ref=ref, also=list(also), boards=list(boards), scenarios=scenarios,
                                                    repeat=repeat, runs=dirs, skipped=skipped, report_cmd=shlex.join(cmd)), indent=1))
    if dry:
        print("[dry] report:", shlex.join(cmd))
        return "dry"
    jid = P.submit(cmd, name=f"bn-match-{out.name}"[:60], owner="bench", vram_gb=0.5, cpu=4, ram_gb=16, after=after,
                   log_dir=str(out / "pool"), cwd=str(REPO), priority=priority)
    print(f"{len(runs)} cells, {len(after)} jobs queued; report job {jid} -> {out}/match.md (then {out}/DONE)")
    return jid


def report(drivers, ref, out, also=(), boards=("hugsim",), scenarios="all64", repeat="") -> Path:
    """Tables of the finished cells: per board bench.tables.report, then match.{md,csv}."""
    import pandas as pd
    from .. import stats
    from .tables import report as board_report
    runs, skipped = cells(drivers, ref, also, boards)
    out = Path(out)
    rows, notes = [], [f"skipped {m} on {b}: {why}" for m, b, why in skipped]
    for b in boards:
        bench, preset = board(b)
        arms = [m for m, bb, p in runs if (bb, p) == (bench, preset) and m != ref]
        if not arms:
            continue
        kw = dict(repeat=repeat, cluster="scene") if bench == "hugsim" else {}
        sub = None
        if bench == "hugsim" and scenarios != "all64":
            from .sets import hugsim_scenarios
            sub = [Path(s).stem for s in hugsim_scenarios(scenarios)]
        try:
            res = board_report(bench, arms, [ref], preset=preset, out=str(out), units_subset=sub, **kw)
        except Exception as e:                                       # one broken board must not hide the others
            notes.append(f"{b}: report failed: {type(e).__name__}: {e}")
            continue
        P = res.get("paired")
        if P is None:
            continue
        for _, r in P.iterrows():
            a, _, vs = r["pair"].partition(" - ")
            if vs != ref:
                continue
            row = {"board": b, "metric": METRIC[bench], "driver": a, "reference": ref, "driver mean": r.mean_a, "ref mean": r.mean_b,
                   "diff": r["mean"], "lo": r.lo, "hi": r.hi, "n": r.n}
            row |= {k: r[k] for k in P.columns if k.startswith("d") and k != "driver"}
            rows.append(row)
    df = pd.DataFrame(rows)
    note = (f"driver - reference, paired per unit ({', '.join(f'{k}: {v}' for k, v in METRIC.items())}); "
            "per-board tables beside this file." + "".join(f"\n\n{n}" for n in notes))
    stats.write_table(df, out / "match", floatfmt=".3f", note=note)
    (out / "DONE").write_text(json.dumps(dict(rows=len(rows), notes=notes)))
    print(f"match -> {out}/match.md")
    return out / "match.md"


def main(a) -> int:
    common = dict(also=a.also, boards=a.boards, scenarios=a.scenarios, repeat=a.repeat)
    if a.report_only:
        report(a.drivers, a.ref, a.out, **common)
        return 0
    argv = ["--drivers", *a.drivers, "--ref", a.ref, *(["--also", *a.also] if a.also else []), "--boards", *a.boards,
            "--scenarios", a.scenarios, "--repeat", a.repeat, "--out", str(Path(a.out).resolve())]
    submit(a.drivers, a.ref, a.out, workers=a.workers, jobs=a.jobs, priority=a.priority, dry=a.dry, argv=argv, **common)
    return 0
