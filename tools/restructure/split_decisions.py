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
    ap.add_argument("--curated", default=next((p for p in ("restructure/decisions_index.tsv", "tools/restructure/decisions_index.tsv")
                                               if Path(p).exists()), "restructure/decisions_index.tsv"),
                    help="hot/archive tiers and short claims")
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

    cur = {}   # curated rows: num -> (tier, claim, evidence, status, note)
    if Path(a.curated).is_file():
        import csv
        for r in csv.DictReader(open(a.curated), delimiter="\t"):
            cur[r["num"]] = (r["tier"], r["claim"], r["evidence"], r["status"], r.get("note", ""))
    head_txt = ["# 决定记录", "",
                "跨 session 的共同决定：本页一行一条，只列仍然成立、可以照着做的条目；全文在 [decisions/](decisions/)`<NNN>.md`"
                "（三位编号，同号的后一版带 `-2`）。撤回、被取代或已关闭的条目在 [decisions/ARCHIVE.md](decisions/ARCHIVE.md)。",
                "",
                "维护：有中间结果就落盘；关键测量先预登记（判据写死、结果留空）；写错就地改并写明原来说了什么、为什么变；"
                "证据变弱就降级。新条目写 `decisions/NNN.md`（编号接最大值）并在下表加一行；条目被推翻时把它移到 ARCHIVE.md，"
                "并检查引用它的主题 README。证据：强 / 中 / 弱；状态：已确认 / 待定。", "",
                "| # | 结论 | 证据 | 状态 |", "|---|---|---|---|"]
    arch = ["# 决定记录：归档", "", "撤回、被取代或已关闭的条目；全文仍在 `<NNN>.md`。主索引：[../decisions.md](../decisions.md)。", "",
            "| # | 结论 | 状态 | 原因 |", "|---|---|---|---|"]
    out = list(head_txt)
    for num, name, title, _ in sorted(entries, key=key):
        n = f"{num}{'-' + name.split('-')[1] if '-' in name else ''}"
        tier, claim, ev, st, note = cur.get(n, ("hot", claim_of(title), "—", status_of(title), ""))
        claim = claim.replace("|", "\\|")
        if tier == "archive":
            arch.append(f"| {n} | {claim} | {st} | {note} |")
        else:
            out.append(f"| {n} | {claim} | {ev} | {st} |")
    LOG.write_text("\n".join(out) + "\n")
    (DIR / "ARCHIVE.md").write_text("\n".join(arch) + "\n")
    print(f"split {len(entries)} entries into {DIR}/, index {len(out) - len(head_txt)} hot, archive {len(arch) - 6}")


if __name__ == "__main__":
    main()
