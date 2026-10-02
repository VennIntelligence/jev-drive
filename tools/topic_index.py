#!/usr/bin/env python3
"""Topic READMEs' generated file list and experiments/INDEX.md, from the README header fields.

Each `experiments/<topic>/README.md` carries `status:`, `decisions:`, `index:` and (until the first run) `key:` with
the 8-12 files an agent opens first; the block between `files:begin` / `files:end` is generated from them (later runs
read the key files back from the block). INDEX.md is one line per topic. Run after adding a topic or a key file:

    python tools/topic_index.py            # from the repo root
    python tools/topic_index.py --check    # exit 1 if a README block or INDEX.md is stale
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

EXP = Path("experiments")
DRAFTS = Path("restructure/readmes")
BEGIN, END = "<!-- files:begin -->", "<!-- files:end -->"
AREAS = [  # INDEX.md order; live topics come first
    ("openpilot adaptation", ["vlm_arb", "op_adapt_l", "op_adapt_h", "op_adapt_r2", "op_adapt_r1", "op_closed_loop", "op_openloop", "skill_pack",
                              "log_expert_audit", "feature_adapter", "op_common_cause", "op_img_cmd"]),
    ("closed-loop harness and controllers", ["b2d_privileged", "cl_infra", "b2d_controller", "b2d_controller_eval", "b2d_tcp",
                                             "b2d_tfv6", "tfv6_rules", "simlingo_catalogue", "carla_rewind"]),
    ("zero-shot exams and leaderboards", ["zeroshot_openloop", "zeroshot_b2d", "model_smoke", "hugsim", "leaderboard_audit",
                                          "top10", "baselines_latency"]),
    ("frozen features and the reaction line", ["probe_planner_v0", "prediag", "driving_backbones", "reactivity", "fusion_diag",
                                               "fastperc", "elicitation", "real_transfer", "statepol"]),
    ("night queues", ["night_queue_2", "night_queue_3", "night_queue_4"]),
    ("real-appearance pairs and world models", ["p3_ped_exam", "cosmos", "controlnet_pair", "world_model"]),
]
ALIASES = {  # names the decision log and old notes use: a grep on INDEX.md finds the topic
    "vlm_arb": "vlm-arb, vlm_arb",
    "op_adapt_l": "op-adapt L", "op_adapt_r2": "op-adapt r2, S_jev", "op_adapt_r1": "op-adapt r1, op_torch",
    "op_closed_loop": "op-arb, op-drive", "op_openloop": "op-interp, op-lb, navhard", "skill_pack": "N0-N4, navsim raise",
    "feature_adapter": "E0, E1", "cl_infra": "cl-lib, infra", "b2d_controller_eval": "Task 10", "b2d_tfv6": "TFv6 W2, D1-D3",
    "tfv6_rules": "TFv6 rules", "zeroshot_openloop": "zero-shot", "zeroshot_b2d": "zero-shot B2D",
    "model_smoke": "openpilot smoke, rigs", "hugsim": "HUGSIM, I3", "leaderboard_audit": "hack audit", "top10": "T1-T3",
    "prediag": "P0-P4, L0", "probe_planner_v0": "probe v0, stage A", "reactivity": "P5, M-C, I4", "fusion_diag": "fusion Q1-Q9",
    "elicitation": "E1-E6", "real_transfer": "G0-G3", "night_queue_2": "nq2, N1-N6, P6", "night_queue_3": "nq3, Q1-Q6",
    "night_queue_4": "nq4, G K X OPL", "p3_ped_exam": "P3, ped dose", "world_model": "W, WL, WL-2", "controlnet_pair": "cn_pair",
    "statepol": "state-space",
}


def clause(path: Path) -> str:
    """One clause from a code file's docstring (or first comment): up to the first period / colon, <= 40 chars."""
    text = path.read_text(errors="replace") if path.is_file() else ""
    s = ""
    if path.suffix == ".py":
        try:
            s = (ast.get_docstring(ast.parse(text)) or "").strip().split("\n")[0]
        except SyntaxError:
            pass
    if not s:
        s = next((l.strip().lstrip("# ") for l in text.splitlines()[:15]
                  if l.strip().startswith("#") and not l.startswith("#!") and len(l.strip()) > 3 and "restructure:" not in l), "")
    s = re.sub(r"\s*\([^()]*\)", "", s)
    s = re.sub(r"\s*(?:of|from|in|for|see|on)?\s*`?[\w.\-$]*/[\w.\-/*{}]+`?", "", s)   # paths say nothing here
    parts = [x.strip(" .:;,") for x in re.split(r"(?<=[a-z0-9\]])[.;]\s|:\s|\s-{1,2}\s", s) if x.strip(" .:;,")]
    s = parts[1] if len(parts) > 1 and len(parts[0]) <= 25 else (parts[0] if parts else "")
    return s if len(s) <= 40 else s[:40].rsplit(" ", 1)[0].rstrip(" ,;:") + " …"


def field(text, name):
    m = re.search(rf"^{name}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def key_files(topic: Path, text: str) -> list[Path]:
    """Key files: the `key:` line (repo-root paths) or, on later runs, the links of the generated block."""
    k = field(text, "key")
    if k:
        out = [Path(p.strip()) for p in k.split(",") if p.strip()]
    else:
        block = text.split(BEGIN, 1)[1].split(END, 1)[0] if BEGIN in text else ""
        out = []
        for name, where in re.findall(r"^- `([^`]+)` \(([^)]+)\)", block, re.M):
            cand = [topic / where / name, Path(where) / name] + list(topic.rglob(name))
            out += [next((c for c in cand if c.is_file()), topic / where / name)]
    return [p for p in out if p.exists()]


def block(topic: Path, keys: list[Path]) -> str:
    lines = ["## Files", ""]
    for p in keys:
        rel = os.path.relpath(p, topic)
        where = rel.split("/")[0] if not rel.startswith("..") else str(p.parent)   # outside the topic: repo-root dir
        lines.append(f"- `{p.name}` ({where}): {clause(p)}")
    tail = []
    for sub, what in (("archive", "one-off code"), ("results", "result files"), ("figs", "figures"), ("plans", "live plans"),
                      ("lib", "library"), ("scripts", "entry points")):
        d = topic / sub
        if d.is_dir():
            n = sum(1 for x in d.rglob("*") if (x.is_file() or x.is_symlink()) and "__pycache__" not in x.parts)
            tail.append(f"[{sub}/]({sub}/) {n} {what}")
    if tail:
        lines += ["", " · ".join(tail)]
    return "\n".join(lines)


def readme(topic: str, src: str) -> str:
    keys = key_files(EXP / topic, src)
    body = re.sub(r"^key:.*\n", "", src, flags=re.M)
    if BEGIN not in body:
        body = body.rstrip() + f"\n\n{BEGIN}\n{END}\n"
    head, rest = body.split(BEGIN, 1)
    return head + BEGIN + "\n" + block(EXP / topic, keys) + "\n" + END + rest.split(END, 1)[1]


def ranges(dec: str) -> str:
    """'77, 78, 79, 80, 81' -> '77-81' (non-numeric entries such as 3d stay as they are)."""
    toks = [t.strip() for t in re.split(r"[,\s]+", dec) if t.strip()]
    out, run = [], []
    for t in toks + [None]:
        if t is not None and t.isdigit() and run and int(t) == int(run[-1]) + 1:
            run.append(t)
            continue
        if run:
            out.append(run[0] if len(run) == 1 else f"{run[0]}-{run[-1]}")
        run = [t] if t is not None and t.isdigit() else []
        if t is not None and not t.isdigit():
            out.append(t)
    return ",".join(out)


def index(readmes: dict[str, str]) -> str:
    lines = ["# Experiments index", "",
             "One line per topic: name (aliases): status; key finding [d decision entries]. Open `<topic>/README.md`.", ""]
    live = [t for t, r in readmes.items() if field(r, "status") == "live"]
    groups = [("live", live)] + [(a, [t for t in ts if t in readmes and t not in live]) for a, ts in AREAS]
    placed = set(live)
    for _, ts in AREAS:
        placed |= set(ts)
    other = [t for t in readmes if t not in placed]
    if other:
        groups.append(("other", other))
    for area, ts in groups:
        if not ts:
            continue
        lines += [f"**{area}**", ""]
        for t in ts:
            r = readmes[t]
            al = f" ({ALIASES[t]})" if t in ALIASES else ""
            st = field(r, "status")
            lines.append(f"- [{t}]({t}/README.md){al}: {'' if st == 'concluded' else st + '; '}"
                         f"{field(r, 'index') or field(r, 'headline')} [d{ranges(field(r, 'decisions')) or '-'}]")
        lines.append("")
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
            print(f"no README for {t}", file=sys.stderr)
            return 1
        out[t] = readme(t, src.read_text())
    idx = index(out)
    if a.check:
        stale = [f"experiments/{t}/README.md" for t, x in out.items() if (EXP / t / "README.md").read_text() != x]
        if not (EXP / "INDEX.md").exists() or (EXP / "INDEX.md").read_text() != idx:
            stale.append("experiments/INDEX.md")
        print("\n".join(stale) or "up to date")
        return 1 if stale else 0
    for t, x in out.items():
        (EXP / t / "README.md").write_text(x)
    (EXP / "INDEX.md").write_text(idx)
    sd = [p for p in (Path(__file__).with_name("split_decisions.py"), Path("tools/restructure/split_decisions.py")) if p.exists()]
    if Path("research/decisions.md").exists() and sd:
        subprocess.run([sys.executable, str(sd[0]), "--topics", str(EXP)], check=True)
    big = {t: len(x.encode()) // 4 for t, x in out.items() if len(x.encode()) // 4 > 300 or x.count("\n") > 60}
    print(f"{len(out)} READMEs (avg {sum(len(x.encode()) for x in out.values()) // 4 // len(out)} tok), "
          f"INDEX.md {len(idx.encode()) // 4} tok; over 300 tok / 60 lines: {big}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
