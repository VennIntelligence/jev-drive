#!/usr/bin/env python3
"""Split research/decisions.md into a one-read index plus one file per entry (research/decisions/NNN.md).

The index keeps the path research/decisions.md, so every existing link to it still resolves; entry numbers stay the
same, so prose references ("decisions 55", "第 55 条") stay valid. Entries are copied verbatim with headings promoted
one level and relative links re-based. Part of the restructure apply (docs/restructure-design.md); idempotent: an
already split log (no `## N.` sections left) is left alone.

    python tools/split_decisions.py [--topics restructure/readmes]   # run from the repo root
"""
from __future__ import annotations

import argparse
import collections
import posixpath as pp
import re
from pathlib import Path

LOG = Path("research/decisions.md")
DIR = Path("research/decisions")
HEAD_RE = re.compile(r"^## (\d+[a-z]?)\. (.*)$")
LINK_RE = re.compile(r"(!?\[[^\]\n]*\]\()([^)\s]+)(\))")
STATUS_WORDS = ["已确认", "已定", "已决定", "已完成", "待定", "关闭", "暂停", "过期"]
CLAIM_MAX = 50


def topics_by_entry(readmes: Path):
    out = collections.defaultdict(list)
    for f in sorted(list(readmes.glob("*.md")) + list(readmes.glob("*/README.md"))):
        m = re.search(r"^decisions:\s*(.+)$", f.read_text(), re.M)
        if not m:
            continue
        stem = f.parent.name if f.name == "README.md" else f.stem
        for tok in re.split(r"[,\s]+", m.group(1)):
            tok = tok.strip()
            if re.fullmatch(r"\d+[a-z]?", tok):
                out[tok].append(stem)
    return out


def status_of(title):
    bold = " ".join(re.findall(r"\*\*([^*]+)\*\*", title))
    for w in STATUS_WORDS:
        if w in bold:
            return w
    for w in STATUS_WORDS:
        if w in title:
            return w
    return "—"


def claim_of(title):
    t = re.sub(r"（[^（）]*(?:（[^（）]*）[^（）]*)*）\s*$", "", title).strip()   # trailing (status, evidence) parenthetical
    t = re.sub(r"\*\*", "", t)
    return t if len(t) <= CLAIM_MAX else t[:CLAIM_MAX].rstrip() + "…"


def rebase_links(text, src_dir, dst_dir):
    def sub(m):
        t = m.group(2)
        if re.match(r"^[a-z][\w+.-]*:", t) or t.startswith(("#", "/")):
            return m.group(0)
        path, sep, frag = t.partition("#")
        new = pp.relpath(pp.normpath(pp.join(src_dir, path)), dst_dir)
        return m.group(1) + new + sep + frag + m.group(3)
    return LINK_RE.sub(sub, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default="experiments", help="topic READMEs whose `decisions:` line maps entries to topics")
    a = ap.parse_args()
    lines = LOG.read_text().split("\n")
    starts = [i for i, l in enumerate(lines) if HEAD_RE.match(l)]
    if not starts:
        print("already split")
        return
    head = lines[:starts[0]]
    seen = collections.Counter()
    entries = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        num, title = HEAD_RE.match(lines[s]).groups()
        seen[num] += 1
        digits = re.match(r"\d+", num).group()
        name = f"{int(digits):03d}{num[len(digits):]}" + (f"-{seen[num]}" if seen[num] > 1 else "")
        body = lines[s:e]
        while body and body[-1].strip() in ("", "---"):
            body.pop()
        body = [re.sub(r"^(#{2,6}) ", lambda m: m.group(1)[1:] + " ", l) if l.startswith("##") else l for l in body]
        entries.append((num, name, title, "\n".join(body) + "\n"))
    DIR.mkdir(parents=True, exist_ok=True)
    topics = topics_by_entry(Path(a.topics))
    for num, name, title, body in entries:
        (DIR / f"{name}.md").write_text(rebase_links(body, "research", "research/decisions"))

    def key(e):
        m = re.match(r"(\d+)([a-z]?)", e[0])
        return int(m.group(1)), m.group(2), e[1]

    keep = "\n".join(head).rstrip().rstrip("-").rstrip()
    keep = keep.replace("跨 session 的共同决定都记在这里，一条一个小节。",
                        "跨 session 的共同决定都记在这里：本页是索引，一条一行；每条的全文在 `decisions/NNN.md`。")
    keep = keep.replace("- 每条决定被推翻时，检查引用它的文档（`research/`、`todos/`）有没有跟着改。",
                        "- 每条决定被推翻时，检查引用它的文档（`research/`、`experiments/*/README.md`）有没有跟着改。\n"
                        "- **新条目**：写 `decisions/NNN.md`（编号接着最大的往下排），再在下表加一行；"
                        "状态或结论变了，两处一起改。表里的结论截断到 50 字，全文以条目文件为准。")
    out = [keep, "", "## 索引", "",
           "第 N 条的全文在 `decisions/NNN.md`（三位编号，如 `decisions/083.md`、`decisions/003d.md`）；"
           "主题 T 的说明在 `../experiments/T/README.md`。同号两条（8、9、10）是早期并存的两版，后写的一版文件名带 `-2`。", "",
           "| # | 状态 | 主题 | 结论（截断） |", "|---|---|---|---|"]
    for num, name, title, _ in sorted(entries, key=key):
        tp = ", ".join(topics.get(num, [])) or "—"
        claim = claim_of(title).replace("|", "\\|")
        out.append(f"| {num}{'-' + name.split('-')[1] if '-' in name else ''} | {status_of(title)} | {tp} | {claim} |")
    LOG.write_text("\n".join(out) + "\n")
    print(f"split {len(entries)} entries into {DIR}/, index {len(out)} lines")


if __name__ == "__main__":
    main()
