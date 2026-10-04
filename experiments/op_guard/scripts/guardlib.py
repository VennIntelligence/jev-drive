"""Shared pieces of the common guard set (experiments/op_guard/README.md): candidate registry, line-result files, pass / fail rules.

A *line* script (`line_<id>.py`) evaluates one guard line for one candidate and writes `<out>/lines/<id>.json`:

    {"line": id, "candidate": name, "mode": "subset|full", "status": "ok|stub|error", "runtime_s": float,
     "rows": [{"id", "metric", "value", "ref", "delta", "ci", "rule", "pass": true|false|null, "note"}],
     "provenance": {...}}                       # pass null = not evaluable (stub or no reference yet)

`out` is `$DATA_DIR/runs/op_guard/<candidate>/<mode>` on the box (large per-token files next to it); `guard.py` merges the line files into
guard.json / guard.md. Reference for paired lines = the `shipped` candidate of the same mode (its lines run first); shipped against itself is
`pass` by definition (delta 0).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GUARD = REPO / "experiments" / "op_guard"
REGISTRY = GUARD / "candidates.json"
SHIPPED = "shipped"


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))


def candidates() -> dict:
    return json.loads(REGISTRY.read_text())


def resolve(candidate: str) -> dict:
    """Registry name or a path to an ONNX file -> {name, onnx (absolute path or None for shipped), command_adapter}."""
    reg = candidates()
    if candidate in reg:
        c = dict(reg[candidate], name=candidate)
    elif candidate.endswith(".onnx"):
        c = dict(onnx=candidate, name=Path(candidate).stem, command_adapter=None)
    else:
        raise SystemExit(f"unknown candidate {candidate!r}: not in {REGISTRY} and not an .onnx path "
                         "(checkpoints: export a serving ONNX first, see README 'Gaps')")
    if c.get("onnx"):
        c["onnx"] = os.path.expandvars(c["onnx"])
        if not Path(c["onnx"]).is_file():
            raise SystemExit(f"missing ONNX {c['onnx']}")
    return c


def run_dir(candidate: str, mode: str) -> Path:
    d = data_dir() / "runs" / "op_guard" / candidate / mode
    (d / "lines").mkdir(parents=True, exist_ok=True)
    return d


def line_args(line: str, desc: str = "") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc or line)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mode", choices=("subset", "full"), default="subset")
    ap.add_argument("--gpu", type=int, default=0, help="physical card index (use the leased one)")
    ap.add_argument("--cpus", default="", help="cpu list of the lease (taskset syntax), empty = unpinned")
    ap.add_argument("--force", action="store_true", help="recompute even if lines/<id>.json exists")
    return ap


def row(id, metric, value, ref=None, rule="", ok=None, ci=None, note="", **extra) -> dict:
    d = dict(id=id, metric=metric, value=value, ref=ref, delta=None if (ref is None or value is None) else value - ref, ci=ci,
             rule=rule, **{"pass": ok}, note=note)
    d.update(extra)
    return d


def at_least(value, ref, margin: float):
    """rule `value >= ref - margin` (margin >= 0: tolerated drop)."""
    return None if value is None or ref is None else bool(value >= ref - margin - 1e-12)


def at_most(value, ref, margin: float):
    """rule `value <= ref + margin`."""
    return None if value is None or ref is None else bool(value <= ref + margin + 1e-12)


def write_line(out: Path, line: str, candidate: str, mode: str, rows: list, t0: float, status: str = "ok", provenance: dict | None = None):
    p = Path(out) / "lines" / f"{line}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(line=line, candidate=candidate, mode=mode, status=status, runtime_s=round(time.time() - t0, 1), rows=rows,
                                   provenance=provenance or {}), indent=1, default=float))
    tmp.rename(p)
    return p


def done(out: Path, line: str) -> bool:
    return (Path(out) / "lines" / f"{line}.json").is_file()


def load_line(out: Path, line: str):
    p = Path(out) / "lines" / f"{line}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def shipped_line(candidate_out: Path, line: str):
    """The shipped candidate's line file in the same mode (sibling dir), or None."""
    return load_line(Path(candidate_out).parent.parent / SHIPPED / Path(candidate_out).name, line)


# the guard lines: id -> (board, what, rule text). Line scripts import the rule text so the table says the same everywhere.
LINES = {
    "navtest": ("NAVSIM navtest", "PDMS paired vs shipped", "not below shipped - 0.3"),
    "navhard": ("NAVSIM navhard two-stage", "official EPDMS + early-turn subset (PDM heading change >= 5 deg in 2 s, decision 110)", "no drop"),
    "wod": ("WOD-E2E val rater frames", "RFS; standstill false-start rate", "RFS no drop; false start <= shipped + 2 pp"),
    "hugsim": ("HUGSIM 11 scenes (decision 124), `spec`", "spins, stuck, HD", "spins not up"),
    "b2d_turns": ("B2D 25 junction turns (decision 127), zones off", "took exit / leaves lane / collisions", "primary readout (no pass line)"),
    "b2d_ds": ("B2D 19 routes x 2 seeds, `spec`", "DS", "DS no drop"),
    "drift": ("open loop, no command", "4 s lateral drift vs the original model at junction / straight frames", "<= 0.10 m"),
    "negatives": ("open loop, negative commands (lib/route_neg.py)", "offset when the commanded exit does not exist", "<= original + 0.3 m"),
}
