#!/usr/bin/env python3
"""Final stage of the restructure apply: per-topic READMEs with a generated file index, experiments/INDEX.md, and the
split decision log (tools/split_decisions.py). Re-runnable: README file indexes are regenerated between the
`files:begin` / `files:end` markers, so `python tools/restructure_docs.py` after adding files keeps them current.

    python tools/restructure_docs.py                     # from the repo root, after the moves
    python tools/restructure_docs.py --check             # exit 1 if a README index or INDEX.md is stale
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

EXP = Path("experiments")
DRAFTS = Path("restructure/readmes")
BEGIN, END = "<!-- files:begin -->", "<!-- files:end -->"
CODE = (".py", ".sh")
AREAS = [  # INDEX.md grouping; topics not listed fall under "other"
    ("live", None),
    ("openpilot adaptation and its closed loop", ["op_adapt_l", "op_adapt_r2", "op_adapt_r1", "op_closed_loop", "op_openloop",
                                                 "skill_pack", "log_expert_audit", "feature_adapter"]),
    ("closed-loop harness, controllers and privileged ceilings", ["b2d_privileged", "cl_infra", "b2d_controller", "b2d_controller_eval",
                                                                  "b2d_tcp", "b2d_tfv6", "tfv6_rules", "simlingo_catalogue", "carla_rewind"]),
    ("zero-shot exams and leaderboards", ["zeroshot_openloop", "zeroshot_b2d", "model_smoke", "hugsim", "leaderboard_audit", "top10",
                                          "baselines_latency"]),
    ("frozen features, readouts and the reaction line", ["probe_planner_v0", "prediag", "driving_backbones", "reactivity", "fusion_diag",
                                                         "fastperc", "elicitation", "real_transfer", "statepol"]),
    ("night queues", ["night_queue_2", "night_queue_3", "night_queue_4"]),
    ("real-appearance pairs and world models", ["p3_ped_exam", "cosmos", "controlnet_pair", "world_model"]),
]


ALIASES = {  # names the decision log and old notes use for a topic: INDEX.md carries them so a grep finds the topic
    "op_adapt_l": "op-adapt L", "op_adapt_r2": "op-adapt r2, S_jev", "op_adapt_r1": "op-adapt r1, op_torch",
    "op_closed_loop": "op-arb, op-drive", "op_openloop": "op-interp, op-lb, navhard", "skill_pack": "N0-N4, navsim raise",
    "feature_adapter": "E0, E1 probe", "cl_infra": "cl-lib, infra acceptance", "b2d_controller_eval": "Task 10",
    "b2d_tfv6": "TFv6 W2/W2b/D1-D3", "tfv6_rules": "TFv6 rules x interface", "zeroshot_openloop": "zero-shot exam",
    "zeroshot_b2d": "zero-shot B2D", "model_smoke": "openpilot smoke, rig study", "hugsim": "HUGSIM, I3 pairs",
    "leaderboard_audit": "hack audit, text analysis", "top10": "T1-T3", "prediag": "P0-P4, L0",
    "probe_planner_v0": "probe v0, planner v0, stage A", "reactivity": "P5, M-C, I4", "fusion_diag": "fusion Q1-Q9",
    "elicitation": "E1-E6, I3 exam", "real_transfer": "G0-G3", "night_queue_2": "nq2, N1-N6, P6",
    "night_queue_3": "nq3, lanes A-D, Q1-Q6", "night_queue_4": "nq4, G K X OPL, cx", "p3_ped_exam": "nq4 P3, ped dose",
    "world_model": "W, WL, WL-2", "controlnet_pair": "cn_pair", "statepol": "state-space policies",
}


def describe(path: Path) -> str:
    """First docstring line (or first comment line) of a code file, without path parentheticals."""
    text = path.read_text(errors="replace")
    s = ""
    if path.suffix == ".py":
        try:
            s = (ast.get_docstring(ast.parse(text)) or "").strip().split("\n")[0]
        except SyntaxError:
            s = ""
    if not s:
        for line in text.splitlines()[:15]:
            line = line.strip()
            if line.startswith("#") and not line.startswith("#!") and len(line) > 3 and "restructure:" not in line:
                s = line.lstrip("# ")
                break
    s = re.sub(r"\s*\((?:[^()]*(?:todos/|bcbdde4:|experiments/|research/|docs/|scripts/|jevdrive/)[^()]*)\)", "", s)
    return clip(s.rstrip(" .:;,"), 100)


def clip(s, n):
    """Cut at a word boundary, never inside a path."""
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0].rstrip(" ,;:(") + " …"


_MD_TEXT = None


def md_corpus() -> str:
    """All markdown outside experiments/*/README.md (the READMEs are what we generate)."""
    global _MD_TEXT
    if _MD_TEXT is None:
        fs = subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True).stdout.split()
        _MD_TEXT = "\n".join(Path(f).read_text(errors="replace") for f in fs
                             if Path(f).is_file() and not re.match(r"experiments/[^/]+/README\.md$", f))
    return _MD_TEXT


def file_index(topic: Path) -> list[str]:
    out = []
    for sub in ("scripts", "lib", "archive"):
        d = topic / sub
        if not d.is_dir():
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in CODE and "__pycache__" not in p.parts)
        if not files:
            continue
        label = {"scripts": "entry points", "lib": "library (imported by other code)",
                 "archive": "one-off code of the concluded experiment; reproduce from here, do not extend"}[sub]
        out.append(f"**[{sub}/]({sub}/)** ({label})")
        out += [f"- [{p.relative_to(topic)}]({p.relative_to(topic)}): {describe(p)}" for p in files]
        other = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix not in CODE and "__pycache__" not in p.parts)
        if other:
            out.append(f"- also {len(other)} data/config file(s): " + ", ".join(f"[{p.relative_to(topic)}]({p.relative_to(topic)})" for p in other[:4])
                       + (" …" if len(other) > 4 else ""))
    for sub, what in (("results", "small result files"), ("figs", "figures"), ("plans", "live plan notes (Chinese)")):
        d = topic / sub
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*") if p.is_file() or p.is_symlink()]
        top = sorted({p.relative_to(d).parts[0] + ("/" if len(p.relative_to(d).parts) > 1 else "") for p in files})
        shown = ", ".join(f"[{t}]({sub}/{t})" for t in top[:8]) + (f" … ({len(top)} entries)" if len(top) > 8 else "")
        out.append(f"**[{sub}/]({sub}/)** ({what}, {len(files)} files): {shown}")
        if sub == "figs":   # figures no remaining doc shows (their plan note was removed): collected here
            corpus = md_corpus()
            orphan = sorted(p for p in files if p.name not in corpus)
            if orphan:
                out.append("Figures whose describing plan note was removed (text: `git show bcbdde4:<plan>`): "
                           + ", ".join(f"[{p.relative_to(topic / 'figs')}]({p.relative_to(topic)})" for p in orphan))
    return out


def readme(topic: str, draft: str) -> str:
    body = draft
    if BEGIN not in body:
        body = body.rstrip() + f"\n\n{BEGIN}\n{END}\n"
    head, rest = body.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    idx = file_index(EXP / topic)
    return head + BEGIN + "\n## Files\n\n" + "\n".join(idx) + "\n" + END + tail


def field(text, name):
    m = re.search(rf"^{name}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def headline(text):
    h = field(text, "headline")
    if h:
        return h
    m = re.search(r"\*\*Conclusion\.\*\*\s*(.+?)(?:\n\n|\Z)", text, re.S)
    if not m:
        return ""
    s = " ".join(m.group(1).split())
    first = re.split(r"(?<=[.;])\s+(?=[A-Z(])", s)[0]
    first = re.sub(r"\s*\(decisions[^)]*\)", "", first).rstrip(" .;")
    return clip(first, 170)


def index(readmes: dict[str, str]) -> str:
    lines = ["# Experiments index",
             "",
             "One line per topic; `<topic>/README.md` has the question, conclusion, decision entries and file list.",
             "Status: `live`, `concluded` (code in `archive/`), `superseded-by <topic>`. Template: docs/restructure-design.md.",
             ""]
    placed = set()
    for area, topics in AREAS:
        if topics is None:
            topics = [t for t, r in readmes.items() if field(r, "status") == "live"]
        else:
            topics = [t for t in topics if t in readmes and t not in placed]
        if not topics:
            continue
        lines += [f"## {area}", "", "| topic | status | decisions | headline |", "|---|---|---|---|"]
        for t in topics:
            r = readmes[t]
            al = f" ({ALIASES[t]})" if t in ALIASES else ""
            lines.append(f"| [{t}]({t}/README.md){al} | {field(r, 'status')} | {field(r, 'decisions') or '—'} | "
                         f"{headline(r).replace('|', '/')} |")
            placed.add(t)
        lines.append("")
    rest = [t for t in readmes if t not in placed]
    if rest:
        lines += ["## other", "", "| topic | status | decisions | headline |", "|---|---|---|---|"]
        lines += [f"| [{t}]({t}/README.md) | {field(readmes[t], 'status')} | {field(readmes[t], 'decisions') or '—'} | "
                  f"{headline(readmes[t])} |" for t in rest]
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    topics = sorted(p.name for p in EXP.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    out = {}
    for t in topics:
        src = DRAFTS / f"{t}.md" if (DRAFTS / f"{t}.md").exists() else EXP / t / "README.md"
        if not src.exists():
            print(f"no README draft for {t}", file=sys.stderr)
            return 1
        out[t] = readme(t, src.read_text())
    stale = []
    for t, text in out.items():
        p = EXP / t / "README.md"
        if a.check:
            if not p.exists() or p.read_text() != text:
                stale.append(str(p))
        else:
            p.write_text(text)
    idx = index(out)
    if a.check:
        if not (EXP / "INDEX.md").exists() or (EXP / "INDEX.md").read_text() != idx:
            stale.append("experiments/INDEX.md")
        print("\n".join(stale) or "up to date")
        return 1 if stale else 0
    (EXP / "INDEX.md").write_text(idx)
    if Path("research/decisions.md").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("split_decisions.py")), "--topics", str(EXP)], check=True)
    long = [t for t, x in out.items() if x.count("\n") > 60]
    print(f"{len(out)} READMEs, INDEX.md {idx.count(chr(10))} lines; READMEs over 60 lines: {long}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
