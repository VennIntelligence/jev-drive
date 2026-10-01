"""多代理安全更新共享任务板的一行。"""
import fcntl
import re
import sys
from pathlib import Path

board, slug, owner, status = sys.argv[1:5]
path = Path('/data/hack_audit/out/INDEX.md')
with path.open('r+') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    data = f.read()
    pattern = rf'^\| {re.escape(board)} \| ([^|]+) \| [^|]+ \| [^|]+ \| \[审计页\]\(audits/{re.escape(board)}__{re.escape(slug)}\.md\) \|$'
    repl = rf'| {board} | \1 | {owner} | {status} | [审计页](audits/{board}__{slug}.md) |'
    updated, n = re.subn(pattern, repl, data, flags=re.M)
    if n != 1: raise SystemExit(f'expected one row, got {n}: {board}/{slug}')
    f.seek(0);f.write(updated);f.truncate()
