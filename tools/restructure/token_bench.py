#!/usr/bin/env python3
"""Retrieval-cost benchmark of the restructure: tokens an agent must read to answer typical lookups, old vs new tree.

Each task lists the minimal path an agent follows in each tree: whole-file reads, line-range reads (after a grep
found the place), directory listings and grep outputs. Cost = bytes of what the agent sees / 4 (a rough token
estimate; Chinese text costs more per byte, so the decision-log numbers are if anything understated). CLAUDE.md is
read every session and is reported separately.

    python tools/token_bench.py --old <unmoved tree> --new <restructured tree> [--md out.md]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def cost(root: Path, step):
    kind, *args = step
    if kind == "read":
        return (root / args[0]).stat().st_size
    if kind == "lines":                      # read lines a..b of a file
        lines = (root / args[0]).read_bytes().split(b"\n")
        return sum(len(x) + 1 for x in lines[args[1] - 1:args[2]])
    if kind == "ls":
        return len("\n".join(sorted(p.name for p in (root / args[0]).iterdir())).encode())
    if kind == "grep":                       # grep -rn PATTERN PATHS (output the agent reads)
        out = subprocess.run(["grep", "-rn", "-I", args[0], *args[1:]], cwd=root, capture_output=True).stdout
        return len(out)
    if kind == "grepi":                      # grep -rni
        out = subprocess.run(["grep", "-rni", "-I", args[0], *args[1:]], cwd=root, capture_output=True).stdout
        return len(out)
    if kind == "grepl":                      # grep -rl (file names only)
        out = subprocess.run(["grep", "-rl", "-I", args[0], *args[1:]], cwd=root, capture_output=True).stdout
        return len(out)
    raise ValueError(kind)


def find_line(root, path, needle):
    for i, l in enumerate((root / path).read_text().split("\n"), 1):
        if l.startswith(needle):
            return i
    raise KeyError(needle)


def entry_lines(root, n):
    """Line range of decision entry n in the old monolithic log."""
    lines = (root / "research/decisions.md").read_text().split("\n")
    s = next(i for i, l in enumerate(lines, 1) if l.startswith(f"## {n}. "))
    e = next((i for i, l in enumerate(lines[s:], s + 1) if l.startswith("## ")), len(lines))
    return ("lines", "research/decisions.md", s, e - 1)


def tasks(old: Path):
    return [
        ("op-adapt L: which script produced the result, what was the conclusion",
         [("read", "README.md"), ("grep", "op-adapt L", "research/decisions.md"), entry_lines(old, 78), ("ls", "scripts")],
         [("grep", "op-adapt L", "experiments/INDEX.md"), ("read", "experiments/op_adapt_l/README.md")]),
        ("default CARLA worker profile and its evidence",
         [("read", "README.md"), ("read", "docs/closed-loop-runbook.md")],
         [("grep", "profile", "research/decisions.md"), ("read", "research/decisions/083.md")]),
        ("where is the nq3 Q1 table",
         [("read", "README.md"), ("ls", "research/results"), ("ls", "research/results/nq3"), ("ls", "research/results/nq3/q1")],
         [("grep", "nq3", "experiments/INDEX.md"), ("ls", "experiments/night_queue_3/results/q1")]),
        ("which controller experiments exist, which are superseded",
         [("ls", "todos"), ("read", "todos/README.md"), ("grep", "^## .*控制器", "research/decisions.md")],
         [("read", "experiments/INDEX.md")]),
        ("status and claim of decision 55",
         [("grep", "^## 55\\.", "research/decisions.md"), entry_lines(old, 55)],
         [("read", "research/decisions/055.md")]),
        ("list every decision still pending (待定)",
         [("grep", "^## ", "research/decisions.md")],
         [("read", "research/decisions.md")]),
        ("which file implements the S_jev rule scorer, what is it",
         [("grepl", "S_jev", "scripts", "jevdrive"), ("lines", "jevdrive/op_adapt_score.py", 1, 12)],
         [("grep", "S_jev", "experiments/INDEX.md"), ("grep", "scorer", "experiments/op_adapt_r2/README.md")]),
        ("WL-2 verdict, results and figures",
         [("read", "README.md"), ("read", "research/wl2-results.md"), ("ls", "research/results/wl2"), ("ls", "research/figs")],
         [("grep", "WL-2", "experiments/INDEX.md"), ("read", "experiments/world_model/README.md"), ("ls", "experiments/world_model/results/wl2")]),
        ("how to run the HUGSIM zero-shot exam",
         [("ls", "scripts"), ("ls", "scripts/hugsim"), ("read", "todos/2026-09-25-hugsim-exam/README.md")],
         [("grep", "HUGSIM", "experiments/INDEX.md"), ("read", "experiments/hugsim/README.md"), ("lines", "experiments/hugsim/archive/zs_exam.sh", 1, 30)]),
        ("what did the zero-shot Bench2Drive exam conclude",
         [("ls", "todos/2026-09-24-zeroshot-exam"), ("read", "todos/2026-09-24-zeroshot-exam/bench2drive.md"), entry_lines(old, 33)],
         [("grep", "zero-shot B2D", "experiments/INDEX.md"), ("read", "experiments/zeroshot_b2d/README.md")]),
        ("which shared module reads WOD-E2E frames",
         [("ls", "jevdrive"), ("lines", "jevdrive/waymo.py", 1, 30)],
         [("ls", "jevdrive"), ("lines", "jevdrive/waymo.py", 1, 30)]),
        ("night queue 4 G (ghost test): conclusion and code",
         [("read", "README.md"), ("grep", "ghost", "research/decisions.md"), entry_lines(old, 58), ("ls", "scripts"), ("ls", "jevdrive")],
         [("grep", "nq4", "experiments/INDEX.md"), ("read", "experiments/night_queue_4/README.md")]),
        ("NAVSIM skill pack N3 score and its run script",
         [("grep", "^## .*N3", "research/decisions.md"), entry_lines(old, 72), ("ls", "scripts")],
         [("grep", "N3", "research/decisions.md"), ("read", "research/decisions/072.md"), ("grep", "n3", "experiments/skill_pack/README.md")]),
        ("TFv6 controller campaign: conclusion and report",
         [("ls", "todos"), ("ls", "todos/2026-09-23-tfv6-controller"), ("read", "todos/2026-09-23-tfv6-controller/report.md")],
         [("grep", "TFv6 W2", "experiments/INDEX.md"), ("read", "experiments/b2d_tfv6/README.md")]),
        ("which experiments are live right now",
         [("ls", "todos"), ("read", "todos/README.md"), ("ls", "tmp")],
         [("lines", "experiments/INDEX.md", 1, 9)]),
    ]


def cold_tasks(old: Path):
    """Starting a session: CLAUDE.md is always read first, then whatever hops the router needs."""
    C = ("read", "CLAUDE.md")
    return [
        ("what rules apply when I add an experiment",
         [C, ("read", "README.md"), ("read", "todos/README.md"), ("read", "todos/TEMPLATE.md")],
         [C, ("read", "experiments/TEMPLATE.md")]),
        ("where do I put a new result and its figure",
         [C, ("read", "research/README.md")],
         [C, ("read", "experiments/TEMPLATE.md")]),
        ("which experiments are live",
         [C, ("ls", "todos"), ("read", "todos/README.md")],
         [C, ("lines", "experiments/INDEX.md", 1, 9)]),
        ("what is the CARLA default worker profile",
         [C, ("read", "docs/closed-loop-runbook.md")],
         [C, ("grep", "profile", "research/decisions.md")]),
        ("how do I start a long job on the box",
         [C, ("read", "docs/long-runs.md")],
         [C, ("read", "docs/long-runs.md")]),
        ("which shared library gives run dirs and splits",
         [C, ("read", "README.md"), ("read", "docs/README.md"), ("ls", "jevdrive"), ("read", "docs/lib.md")],
         [C, ("read", "docs/lib.md")]),
        ("open the op-adapt L experiment page",
         [C, ("ls", "todos"), ("read", "todos/2026-10-01-op-adapt-L-prereg.md")],
         [C, ("grep", "op-adapt L", "experiments/INDEX.md"), ("read", "experiments/op_adapt_l/README.md")]),
        ("how do I log in to the GPU box",
         [C, ("read", "docs/remote-box.md")],
         [C, ("read", "docs/remote-box.md")]),
        ("what are the writing rules for research notes",
         [C, ("read", "research/README.md")],
         [C, ("read", "research/README.md")]),
        ("which experiments touched NAVSIM",
         [C, ("grepl", "NAVSIM", "todos", "research")],
         [C, ("grepi", "navsim", "experiments/INDEX.md")]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    old, new = Path(a.old), Path(a.new)
    rows, to, tn = [], 0, 0
    sets = [("lookup", tasks(old)), ("cold start (CLAUDE.md included)", cold_tasks(old))]
    tot = {}
    for label, ts in sets:
        a0 = b0 = 0
        rows.append(f"| *{label}* | | | | | |")
        for name, so, sn in ts:
            co = sum(cost(old, s) for s in so) // 4
            cn = sum(cost(new, s) for s in sn) // 4
            a0, b0 = a0 + co, b0 + cn
            rows.append(f"| {name} | {len(so)} | {co:,} | {len(sn)} | {cn:,} | {co / max(cn, 1):.1f}x |")
        rows.append(f"| **{label}: {len(ts)} tasks** | | **{a0:,}** | | **{b0:,}** | **{a0 / max(b0, 1):.1f}x** |")
        tot[label] = (a0, b0)
        to, tn = to + a0, tn + b0
    claude = (old / "CLAUDE.md").stat().st_size // 4, (new / "CLAUDE.md").stat().st_size // 4
    out = ["| task | old steps | old tokens | new steps | new tokens | ratio |", "|---|---|---|---|---|---|", *rows,
           f"| **all 25 tasks** | | **{to:,}** | | **{tn:,}** | **{to / max(tn, 1):.1f}x** |", "",
           f"CLAUDE.md, read every session: {claude[0]:,} -> {claude[1]:,} tokens. "
           f"Directory listings: scripts/ {len(list((old / 'scripts').iterdir()))} -> {len(list((new / 'scripts').iterdir()))} entries, "
           f"jevdrive/ {len(list((old / 'jevdrive').iterdir()))} -> {len(list((new / 'jevdrive').iterdir()))}; "
           f"research/decisions.md {(old / 'research/decisions.md').stat().st_size // 4:,} -> {(new / 'research/decisions.md').stat().st_size // 4:,} tokens."]
    tk = lambda r, f: (r / f).stat().st_size // 4 if (r / f).is_file() else 0
    out += ["", "| file | old tokens | new tokens |", "|---|---|---|"]
    for f in ("CLAUDE.md", "README.md", "docs/README.md", "research/README.md", "experiments/INDEX.md", "research/decisions.md",
              "experiments/TEMPLATE.md"):
        out.append(f"| {f} | {tk(old, f):,} | {tk(new, f):,} |")
    rs = [p.stat().st_size // 4 for p in (new / "experiments").glob("*/README.md")]
    if rs:
        out.append(f"| experiments/*/README.md ({len(rs)}) | - | avg {sum(rs) // len(rs)}, max {max(rs)} |")
        line = max((len(l.encode()) for l in (new / "experiments/INDEX.md").read_text().splitlines()), default=0) // 4
        out += ["", f"Cold start to any topic README (CLAUDE.md + one INDEX.md grep line <= {line} tok + README): "
                    f"<= {tk(new, 'CLAUDE.md') + line + max(rs):,} tok; reading the whole INDEX.md instead: "
                    f"<= {tk(new, 'CLAUDE.md') + tk(new, 'experiments/INDEX.md') + max(rs):,} tok."]
    print("\n".join(out))
    if a.md:
        Path(a.md).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
