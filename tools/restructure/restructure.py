#!/usr/bin/env python3
"""Apply the topic restructure described by restructure/manifest.tsv (design: docs/restructure-design.md).

Driven only by the manifest. Moves with `git mv`, deletes with `git rm`, rewrites every reference it can resolve
(markdown links and inline paths, `jevdrive.x` module names, bare script-module imports, `__file__` / `$0` relative
paths, path strings in code), and fails loudly on references it cannot resolve. Deterministic and idempotent: a second
`apply` on an applied tree finds nothing to move and nothing to rewrite.

    python tools/restructure.py check                       # validate the manifest against the tree
    python tools/restructure.py apply --dry-run              # plan + rewrite report, no change to the tree
    python tools/restructure.py apply                        # do it (run on a clean tree)
    python tools/restructure.py verify --out v.json [--baseline base.json]   # static + smoke checks, compare
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import os
import posixpath as pp
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path.cwd()
def _head():
    try:
        return subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "bcbdde4"


BASE_SHA = _head()   # removed files stay readable at the commit the apply starts from
KINDS = {"shared", "topic", "archive", "data", "doc", "delete"}
STATUSES = {"live", "one-off", "superseded"}
TEXT_EXT = {".md", ".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".tsv"}
CODE_EXT = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
REPO_TOPS = ("scripts", "jevdrive", "research", "todos", "tmp", "docs", "results", "tests", "patches", "tools", "experiments")
SCRATCH_DIRS = {"tmp"}
PERMALINK = f"https://github.com/VennIntelligence/jev-drive/blob/{BASE_SHA}/"   # the apply commit's parent is pushed first


def git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd or ROOT, check=check, text=True, capture_output=True).stdout


def tracked(cwd=None):
    return git("ls-files", cwd=cwd).splitlines()


# ------------------------------------------------------------------------------------------------ manifest

@dataclass
class Plan:
    rows: list
    filemap: dict            # old file -> new file ('' = deleted); only files that move or are deleted
    kind: dict               # old file -> kind
    dirmap: dict             # old dir -> new dir ('' = deleted) for dirs moved wholesale
    old_files: set
    new_files: set
    old_dirs: set
    new_dirs: set
    norewrite: set = field(default_factory=set)   # old files whose content is never rewritten (data, vendored)


def ancestors(p):
    out = []
    while "/" in p:
        p = p.rsplit("/", 1)[0]
        out.append(p)
    return out


def load_manifest(path, files):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    errors, filemap, kind, norewrite = [], {}, {}, set()
    ftset = set(files)
    for i, r in enumerate(rows, 2):
        o, n, k, s = r["old_path"], r["new_path"], r["kind"], r["status"]
        if k not in KINDS:
            errors.append(f"line {i}: kind {k!r}")
        if s not in STATUSES:
            errors.append(f"line {i}: status {s!r}")
        if (k == "delete") != (n == ""):
            errors.append(f"line {i}: delete rows (and only they) have an empty new_path")
        if o.startswith("#"):
            continue
        members = [f for f in files if f.startswith(o)] if o.endswith("/") else ([o] if o in ftset else [])
        if not members:
            done = (k == "delete") or (n.endswith("/") and any(f.startswith(n) for f in files)) or n in ftset
            if done:              # already applied: idempotent re-run; files keep the row's kind
                for f in ([x for x in files if x.startswith(n)] if n.endswith("/") else ([n] if n in ftset else [])):
                    kind[f] = k
                    if k == "data" or "vendored" in r["note"]:
                        norewrite.add(f)
                continue
            if (ROOT / o).exists():
                errors.append(f"line {i}: {o} exists but is untracked; list untracked files explicitly is not supported - track or drop them first")
            else:
                errors.append(f"line {i}: {o} matches no tracked file")
        for f in members:
            if f in filemap or f in kind:
                errors.append(f"line {i}: {f} listed twice")
            nf = "" if k == "delete" else (n + f[len(o):] if o.endswith("/") else n)
            kind[f] = k
            if nf != f:
                filemap[f] = nf
            if k == "data" or "vendored" in r["note"]:
                norewrite.add(f)
    new_files = {filemap.get(f, f) for f in files} - {""}
    dests = collections.Counter(filemap.get(f, f) for f in files if filemap.get(f, f))
    errors += [f"two files land on {d}" for d, c in dests.items() if c > 1]
    # untracked files inside directories that move would be left behind
    for o in {r["old_path"] for r in rows if r["old_path"].endswith("/") and r["kind"] != "delete"}:
        if (ROOT / o).is_dir():
            ut = git("ls-files", "--others", "--exclude-standard", o).split()
            if ut:
                errors.append(f"{o} holds untracked files that would be left behind: {ut[:5]}{' ...' if len(ut) > 5 else ''}")
    old_dirs = {a for f in files for a in ancestors(f)}
    new_dirs = {a for f in new_files for a in ancestors(f)}
    dirmap = {}
    under = collections.defaultdict(list)
    for f in files:
        for a in ancestors(f):
            under[a].append(f)
    for d, fs in under.items():
        if d in new_dirs:
            continue  # still exists (identity)
        targets = {filemap.get(f, f) for f in fs}
        if targets == {""}:
            dirmap[d] = ""
            continue
        roots = {filemap.get(f, f)[: len(filemap.get(f, f)) - len(f) + len(d)] if filemap.get(f, f).endswith(f[len(d):]) else None
                 for f in fs if filemap.get(f, f)}
        if len(roots) == 1 and None not in roots:
            dirmap[d] = roots.pop()
    plan = Plan(rows, filemap, kind, dirmap, set(files), new_files, old_dirs, new_dirs, norewrite)
    return plan, errors


# ------------------------------------------------------------------------------------------------ resolution

class Unresolved(Exception):
    pass


def norm(p):
    p = pp.normpath(p)
    return "" if p == "." else p


def resolve(plan, old):
    """Old repo-relative path (file or dir) -> new path. '' = deleted. None = not a repo path. Raises Unresolved."""
    o = norm(old)
    if not o or o.startswith(".."):
        return None
    if o in plan.old_files:
        return plan.filemap.get(o, o)
    if o in plan.old_dirs:
        if o in plan.new_dirs:
            return o
        if o in plan.dirmap:
            return plan.dirmap[o]
        raise Unresolved(f"directory {o}/ is split across several new places")
    if o.split("/")[0] in SCRATCH_DIRS:
        return o  # gitignored scratch space keeps existing on disk; only its tracked files are removed
    # a path below a tracked dir that is not itself tracked (runtime outputs, untracked files); a top-level ancestor
    # alone is too weak a signal ("results/x" is usually relative prose, not the old top-level results/)
    for a in ancestors(o):
        if a.count("/") == 0 and a not in ("todos", "tmp"):
            break
        if a in plan.old_dirs:
            if a in plan.new_dirs:
                return o
            if a in plan.dirmap:
                return (plan.dirmap[a] + o[len(a):]) if plan.dirmap[a] else ""
            raise Unresolved(f"{o}: its directory {a}/ is split across several new places")
    return None


def new_location(plan, old):
    return plan.filemap.get(old, old)


# ------------------------------------------------------------------------------------------------ rewriting

@dataclass
class Ctx:
    plan: Plan
    old: str                 # old path of the file being rewritten
    new: str                 # its new path
    log: list
    unresolved: list

    def note(self, rule, before, after, line=0):
        self.log.append((self.new, line, rule, before, after))

    def fail(self, rule, text, why, line=0):
        self.unresolved.append((self.old, line, rule, text, why))


PATH_RE = re.compile(r"(?<![\w.\-$@])(?:\./)?((?:" + "|".join(REPO_TOPS) + r")/[\w.\-/*{}]*[\w*}/])")
ROOT_PREFIX_RE = re.compile(r"(?:\$\{?[A-Za-z_]\w*\}?|jev-drive|~)/$")   # $repo/, ${REPO}/, ~/data/jev-drive/
LINK_RE = re.compile(r"(!?\[[^\]\n]*\]\()(<?)([^)\s>]+)(>?)((?:\s+\"[^\"\n]*\")?\))")
REFDEF_RE = re.compile(r"^(\s*\[[^\]\n]+\]:\s+)(\S+)", re.M)
HTML_SRC_RE = re.compile(r"((?:src|href)=\")([^\"\s]+)(\")")
MOD_RE = re.compile(r"(?<![\w.])jevdrive\.([A-Za-z_]\w*)")
SMOD_RE = re.compile(r"(?<![\w./])scripts\.([A-Za-z_]\w*)(?=\s+import\b|\s*$|\s+as\b)", re.M)


def module_of(path):
    return path[:-3].replace("/", ".").removesuffix(".__init__") if path.endswith(".py") else None


def lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def deleted_ref(old_target, as_link):
    return PERMALINK + old_target if as_link else f"{BASE_SHA}:{old_target}"


def rewrite_path_token(ctx, tok, line):
    """A repo-root-relative path token. Returns the replacement (or tok unchanged)."""
    if any(c in tok for c in "*{}"):
        cut = min(i for i in (tok.find("*"), tok.find("{"), tok.find("}")) if i >= 0)
        head = tok[:cut].rsplit("/", 1)[0] if "/" in tok[:cut] else ""
        if not head:
            return tok
        try:
            nh = resolve(ctx.plan, head)
        except Unresolved as e:
            ctx.fail("path-glob", tok, str(e), line)
            return tok
        if nh is None or nh == head:
            return tok
        if nh == "":
            ctx.fail("path-glob", tok, f"{head} is deleted", line)
            return tok
        rep = nh + tok[len(head):]
        ctx.note("path-glob", tok, rep, line)
        return rep
    trail = "/" if tok.endswith("/") else ""
    try:
        new = resolve(ctx.plan, tok)
    except Unresolved as e:
        ctx.fail("path", tok, str(e), line)
        return tok
    if new is None or new == norm(tok):
        return tok
    if new == "":
        rep = deleted_ref(norm(tok), False)
    else:
        rep = new + trail
    ctx.note("path", tok, rep, line)
    return rep


def rewrite_md(ctx, text):
    old_dir, new_dir = pp.dirname(ctx.old), pp.dirname(ctx.new)
    masks = []

    def link_target(target, line):
        if re.match(r"^[a-z][\w+.-]*:", target) or target.startswith("#") or target.startswith("/"):
            return target
        path, frag = (target.split("#", 1) + [""])[:2]
        frag = "#" + frag if "#" in target else ""
        if not path:
            return target
        trail = "/" if path.endswith("/") else ""
        cand_rel = norm(pp.join(old_dir, path))
        try:
            new = resolve(ctx.plan, cand_rel)
            base = cand_rel
            if new is None or (new == cand_rel and not (ROOT / cand_rel).exists() and cand_rel not in ctx.plan.old_files
                               and cand_rel not in ctx.plan.old_dirs):
                alt = norm(path)
                n2 = resolve(ctx.plan, alt)
                if n2 is not None and (alt in ctx.plan.old_files or alt in ctx.plan.old_dirs):
                    new, base = n2, alt
        except Unresolved as e:
            if not (ROOT / cand_rel).exists():   # already broken before the move: keep the same (missing) target
                new, base = cand_rel, cand_rel
            else:
                ctx.fail("md-link", target, str(e), line)
                return target
        if new is None:
            if ctx.new == ctx.old or cand_rel.startswith(".."):
                return target
            new = cand_rel
        if new == "":
            rep = deleted_ref(base, True) + frag
        else:
            relp = pp.relpath(new, new_dir or ".")
            rep = relp + (trail if not relp.endswith("/") else "") + frag
            if rep == target:
                return target
        ctx.note("md-link", target, rep, line)
        return rep

    def sub_link(m):
        line = lineno(text, m.start())
        rep = rewrite_tokens(ctx, m.group(1)) + m.group(2) + link_target(m.group(3), line) + m.group(4) + m.group(5)
        masks.append(rep)
        return f"\x00{len(masks) - 1}\x00"

    def sub_ref(m):
        rep = m.group(1) + link_target(m.group(2), lineno(text, m.start()))
        masks.append(rep)
        return f"\x00{len(masks) - 1}\x00"

    def sub_html(m):
        rep = m.group(1) + link_target(m.group(2), lineno(text, m.start())) + m.group(3)
        masks.append(rep)
        return f"\x00{len(masks) - 1}\x00"

    t = LINK_RE.sub(sub_link, text)
    t = REFDEF_RE.sub(sub_ref, t)
    t = HTML_SRC_RE.sub(sub_html, t)
    t = rewrite_tokens(ctx, t)
    return re.sub(r"\x00(\d+)\x00", lambda m: masks[int(m.group(1))], t)


def rewrite_tokens(ctx, text):
    """Repo-root path tokens and `jevdrive.x` module names in any text."""
    def sub_path(m):
        return rewrite_path_token(ctx, m.group(1), lineno(text, m.start())) if m.group(1) else m.group(0)

    def keep(m):
        before = text[max(0, m.start() - 41):m.start()]
        if before.endswith("/") and not ROOT_PREFIX_RE.search(before):
            return True               # a path inside some other tree (third_party/x/scripts/...), not ours
        return bool(re.search(r"[0-9a-f]{7,40}:$", before))

    t = PATH_RE.sub(lambda m: m.group(0) if keep(m) else m.group(0).replace(m.group(1), sub_path(m)), text)

    def sub_mod(m):
        name = m.group(1)
        old = f"jevdrive/{name}.py"
        if old not in ctx.plan.old_files:
            return m.group(0)
        new = new_location(ctx.plan, old)
        if new == old:
            return m.group(0)
        if not new:
            ctx.fail("module", m.group(0), "module is deleted", lineno(text, m.start()))
            return m.group(0)
        rep = module_of(new)
        ctx.note("module", m.group(0), rep, lineno(text, m.start()))
        return rep

    t = MOD_RE.sub(sub_mod, t)
    if Path(ctx.old).suffix != ".py":
        return t

    def sub_smod(m):
        old = f"scripts/{m.group(1)}.py"
        new = new_location(ctx.plan, old) if old in ctx.plan.old_files else old
        if new == old or not new:
            return m.group(0)
        rep = module_of(new)
        ctx.note("module", m.group(0), rep, lineno(t, m.start()))
        return rep

    return SMOD_RE.sub(sub_smod, t)


# --- python -----------------------------------------------------------------------------------------------

ANCHOR_RE = re.compile(
    r"Path\(\s*__file__\s*\)(?P<res>\.resolve\(\)|\.absolute\(\))?(?P<up>(?:\.parent(?!s))+|\.parents\[(?P<k>\d+)\])?(?P<wn>\.with_name\(\s*(?P<q0>[\"'])(?P<wname>[^\"']+)(?P=q0)\s*\))?")
OSANCHOR_RE = re.compile(r"(?P<dn>(?:os\.path\.dirname\(\s*)+)os\.path\.(?:abspath|realpath)\(\s*__file__\s*\)")
CHAIN_RE = re.compile(r"(?:\s*/\s*(?P<q>[\"'])(?P<lit>[^\"'{}\n]+)(?P=q))+")
ASSIGN_RE = re.compile(r"^(?P<ind>[ \t]*)(?P<name>[A-Za-z_]\w*)\s*=\s*$")


def py_root_expr(newfile, res):
    depth = newfile.count("/")
    return f"Path(__file__){res}.parents[{depth}]" if depth else f"Path(__file__){res}.parent"


def py_dir_expr(newfile, target, res):
    """Expression for repo dir `target` ('' = root) from a file now at `newfile`."""
    own = pp.dirname(newfile)
    if target == own:
        return f"Path(__file__){res}.parent"
    if own.startswith(target + "/") or target == "":
        up = own.count("/") + 1 - (target.count("/") + 1 if target else 0)
        return f"Path(__file__){res}.parents[{up}]" if up else f"Path(__file__){res}.parent"
    return f'{py_root_expr(newfile, res)} / "{target}"'


def rewrite_py_anchors(ctx, text):
    """`Path(__file__)...` and os.path.dirname(...__file__) anchors plus a literal join chain after them."""
    plan, oldf, newf = ctx.plan, ctx.old, ctx.new
    bound = {}  # variable name -> old dir it points at

    def target_of(m):
        if m.group("wn"):
            return pp.dirname(oldf), m.group("wname")
        up = m.group("up")
        if not up:
            return None, None  # the file itself
        lv = int(m.group("k")) + 1 if m.group("k") is not None else up.count(".parent")
        d = oldf
        for _ in range(lv):
            d = pp.dirname(d)
        return d, None

    out, pos = [], 0
    for m in ANCHOR_RE.finditer(text):
        line = lineno(text, m.start())
        xdir, sib = target_of(m)
        if xdir is None:
            continue
        res = m.group("res") or ""
        tail = text[m.end():]
        cm = CHAIN_RE.match(tail)
        lits = [x.group("lit") for x in re.finditer(r"(?P<q>[\"'])(?P<lit>[^\"'{}\n]+)(?P=q)", cm.group(0))] if cm else []
        if sib:
            lits = [sib] + lits
        end = m.end() + (cm.end() if cm else 0)
        try:
            if lits:
                oldp = norm(pp.join(xdir, *lits))
                newp = resolve(plan, oldp)
                if newp is None:  # not a repo path (outside the repo or a pure runtime name): keep the meaning of the dir
                    newp = oldp
                if newp == "":
                    ctx.fail("py-anchor", text[m.start():end], f"points at deleted {oldp}", line)
                    continue
                lm = ASSIGN_RE.match(text[text.rfind("\n", 0, m.start()) + 1:m.start()])
                if lm:
                    bound[lm.group("name")] = (oldp, newp)
                if newp == oldp and newf == oldf:
                    continue
                own = pp.dirname(newf)
                if newp.startswith(own + "/") and own:
                    rep = f'Path(__file__){res}.parent / "{newp[len(own) + 1:]}"'
                else:
                    rep = f'{py_root_expr(newf, res)} / "{newp}"'
            else:
                if newf == oldf:
                    lm = ASSIGN_RE.match(text[text.rfind("\n", 0, m.start()) + 1:m.start()])
                    if lm:
                        bound[lm.group("name")] = (xdir, xdir)
                    continue
                newx = resolve(plan, xdir) if xdir else ""
                if newx is None:
                    newx = xdir
                if newx == "" and xdir:
                    ctx.fail("py-anchor", m.group(0), f"dir {xdir} is deleted", line)
                    continue
                lm = ASSIGN_RE.match(text[text.rfind("\n", 0, m.start()) + 1:m.start()])
                name = lm.group("name") if lm else None
                own_new = pp.dirname(newf)
                if name and xdir == pp.dirname(oldf) and own_bindable(plan, text, name, xdir, own_new, ctx, line):
                    newx = own_new          # the file moved together with what it reaches through this name
                rep = py_dir_expr(newf, newx, res)
                if name:
                    bound[name] = (xdir, newx)
        except Unresolved as e:
            ctx.fail("py-anchor", text[m.start():end], str(e), line)
            continue
        if rep != text[m.start():end]:
            ctx.note("py-anchor", text[m.start():end], rep, line)
            out.append(text[pos:m.start()] + rep)
            pos = end
    text = "".join(out) + text[pos:]
    # os.path.dirname(os.path.abspath(__file__)) style
    out, pos = [], 0
    for m in OSANCHOR_RE.finditer(text):
        if newf == oldf:
            break
        line = lineno(text, m.start())
        lv = m.group("dn").count("dirname")
        cl = re.match(r"(?:\s*\)){%d}" % lv, text[m.end():])
        if not cl:
            ctx.fail("py-anchor", m.group(0), "unbalanced os.path anchor", line)
            continue
        mend = m.end() + cl.end()
        d = oldf
        for _ in range(lv):
            d = pp.dirname(d)
        try:
            newx = resolve(plan, d) if d else ""
        except Unresolved as e:
            ctx.fail("py-anchor", m.group(0), str(e), line)
            continue
        newx = d if newx is None else newx
        depth = newf.count("/")
        up = depth - (newx.count("/") + 1 if newx else 0)
        if newx and not (pp.dirname(newf) + "/").startswith(newx + "/"):
            rep = "os.path.join(" + "os.path.dirname(" * depth + "os.path.abspath(__file__)" + ")" * depth + f', "{newx}")'
        else:
            ctx.note("py-anchor", text[m.start():mend], rep, line)
        out.append(text[pos:m.start()] + rep)
        pos = mend
    text = "".join(out) + text[pos:]
    return text, bound


def own_bindable(plan, text, name, xdir, own_new, ctx, line):
    """True if every use of NAME (bound to the file's own old dir) is a literal join whose target now lives in the
    file's new dir. A non-literal use (NAME / var, f"{NAME}/...") is reported and keeps the old meaning."""
    uses = list(re.finditer(r"(?<![\w.])" + re.escape(name) + r"(?![\w])(?!\s*=[^=])", text))
    ok = True
    for u in uses:
        after = text[u.end():u.end() + 200]
        cm = re.match(r"(?:\s*/\s*([\"'])([^\"'{}\n]+)\1)+", after)
        if cm:
            lits = [x.group(2) for x in re.finditer(r"([\"'])([^\"'{}\n]+)\1", cm.group(0))]
            try:
                tgt = resolve(plan, norm(pp.join(xdir, *lits)))
            except Unresolved:
                return False
            if tgt is None or pp.dirname(tgt) != own_new and not tgt.startswith(own_new + "/"):
                ok = False
        elif re.match(r"\s*/", after) or re.match(r"\.(parent|parents|with_name|joinpath|glob|iterdir)", after):
            return False
    if re.search(r"\{" + re.escape(name) + r"\}[/\\]", text):
        return False
    return ok


LIT = r"(?P<q>[\"'])[^\"'{}\n]+(?P=q)"
CHAIN_LITS = re.compile(r"([\"'])([^\"'{}\n]+)\1")


def derive_names(orig, bound):
    """Names derived from anchor-bound names, in source order: `B = A / "x" / "y"`, `B = A.parent`, `B = A.parents[k]`.
    Returns name -> (old dir, new value dir or None when the value is a dir that no longer exists)."""
    names = dict(bound)
    rx = re.compile(r"^[ \t]*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<src>[A-Za-z_]\w*)(?P<attr>\.parent(?!s)|\.parents\[(?P<k>\d+)\])?"
                    r"(?P<chain>(?:\s*/\s*" + LIT + r")*)\s*(?:#.*)?$", re.M)
    for m in rx.finditer(orig):
        src = m.group("src")
        if src not in names or m.group("name") in bound:
            continue
        x, y = names[src]
        up = 1 if m.group("attr") == ".parent" else (int(m.group("k")) + 1 if m.group("k") is not None else 0)
        for _ in range(up):
            x, y = pp.dirname(x), (pp.dirname(y) if y is not None else None)
        lits = [c.group(2) for c in CHAIN_LITS.finditer(m.group("chain") or "")]
        # value after the rewrite: follows the source; a literal chain may still move it (pass 1 of the chain rewrite)
        names[m.group("name")] = (norm(pp.join(x, *lits)), norm(pp.join(y, *lits)) if y is not None else None)
    return names


def rewrite_py_bound_chains(ctx, text, orig, bound):
    """`NAME / "a" / "b.py"` where NAME is (derived from) a __file__ anchor: point the chain at the new place.
    Two passes: the value every name holds after the rewrite is settled first, then each use is rewritten against it.
    A name assigned twice in a file is ambiguous (scopes); its uses are reported if they reach a moved path."""
    plan = ctx.plan
    names = derive_names(orig, bound)
    if not names:
        return text
    counts = collections.Counter()
    try:
        for n in ast.walk(ast.parse(orig)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.For)) else []
            for t in tg:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        counts[x.id] += 1
    except SyntaxError:
        counts.update(m.group(1) for m in re.finditer(r"^[ \t]*([A-Za-z_]\w*)\s*=[^=]", orig, re.M))
    ambiguous = {n for n in names if counts[n] > 1}
    val = {n: v[1] for n, v in names.items() if n in bound}
    rx = re.compile(r"(?<![\w.])(?P<name>" + "|".join(map(re.escape, names)) + r")(?P<attr>\.parent(?!s)|\.parents\[(?P<k>\d+)\])?"
                    r"(?P<chain>(?:\s*/\s*" + LIT + r")+)")
    assign = re.compile(r"^[ \t]*(?P<lhs>[A-Za-z_]\w*)\s*=\s*$")

    def target(m):
        name = m.group("name")
        xdir = names[name][0]
        up = 1 if m.group("attr") == ".parent" else (int(m.group("k")) + 1 if m.group("k") is not None else 0)
        for _ in range(up):
            xdir = pp.dirname(xdir)
        lits = [c.group(2) for c in CHAIN_LITS.finditer(m.group("chain"))]
        return up, lits, norm(pp.join(xdir, *lits))

    def lhs_of(m):
        lm = assign.match(text[text.rfind("\n", 0, m.start()) + 1:m.start()])
        return lm.group("lhs") if lm and lm.group("lhs") in names and lm.group("lhs") not in bound else None

    # pass 1: values of derived names (in source order; their sources are settled before them)
    for m in rx.finditer(text):
        lhs = lhs_of(m)
        if not lhs:
            continue
        _, _, oldp = target(m)
        try:
            newp = resolve(plan, oldp)
        except Unresolved:
            val[lhs] = None       # a base dir that no longer exists as one place: uses are rewritten one by one
            continue
        val[lhs] = oldp if newp is None else newp
    # pass 2: rewrite
    out, pos = [], 0
    for m in rx.finditer(text):
        line = lineno(text, m.start())
        name = m.group("name")
        up, lits, oldp = target(m)
        ydir = val.get(name, names[name][1])
        for _ in range(up):
            ydir = pp.dirname(ydir) if ydir is not None else None
        try:
            newp = resolve(plan, oldp)
        except Unresolved as e:
            if lhs_of(m):
                continue
            ctx.fail("py-chain", m.group(0), str(e), line)
            continue
        if newp is None:
            newp = oldp
        if newp == "":
            ctx.fail("py-chain", m.group(0), f"points at deleted {oldp}", line)
            continue
        cur = norm(pp.join(ydir, *lits)) if ydir is not None else None
        if cur == newp:
            continue
        if name in ambiguous:
            ctx.fail("py-chain", m.group(0), f"{name} is assigned more than once in this file; rewrite by hand", line)
            continue
        base = name + (m.group("attr") or "")
        xd = pp.dirname(oldp) if False else None
        if ydir is None:      # value is a vanished dir: climb to the root through the OLD depth, which is what it holds
            old_base = names[name][0]
            for _ in range(up):
                old_base = pp.dirname(old_base)
            d = old_base.count("/") + 1 if old_base else 0
            rep = f'{base} / "{newp}"' if d == 0 else f'{base}.parents[{d - 1}] / "{newp}"'
        elif ydir and newp.startswith(ydir + "/"):
            rep = f'{base} / "{newp[len(ydir) + 1:]}"'
        elif not ydir:
            rep = f'{base} / "{newp}"'
        else:
            rep = f'{base}.parents[{ydir.count("/")}] / "{newp}"'
        if rep == m.group(0):
            continue
        ctx.note("py-chain", m.group(0), rep, line)
        out.append(text[pos:m.start()] + rep)
        pos = m.end()
    text = "".join(out) + text[pos:]
    # non-literal joins on a name whose directory moved or vanished cannot be rewritten
    for n, (x, _) in names.items():
        y = val.get(n, names[n][1])
        try:
            want = resolve(plan, x) if x else ""
        except Unresolved:
            want = "\0split"
        if want is None:          # not a path the restructure knows (untracked, outside the repo): unchanged
            want = x
        if y is not None and want == y and n not in ambiguous:
            continue
        if n in ambiguous and want == x:
            continue
        for u in re.finditer(r"(?<![\w.])" + re.escape(n) + r"\s*/\s*(?![\"'\s])([^\n]{0,40})", text):
            ctx.fail("py-dynamic", f"{n} / {u.group(1)}", f"non-literal join on {n} (old {x}/, moved or split)", lineno(text, u.start()))
        for u in re.finditer(r"\{" + re.escape(n) + r"\}[/\\]([^\n]{0,40})", text):
            ctx.fail("py-dynamic", u.group(0), f"f-string path on {n} (old {x}/, moved or split)", lineno(text, u.start()))
    return text


def mod_file(plan, dotted):
    """Old repo file of a dotted module (module.py or package/__init__.py), or None."""
    p = dotted.replace(".", "/")
    for f in (p + ".py", p + "/__init__.py"):
        if f in plan.old_files:
            return f
    return None


def rewrite_py_imports(ctx, text):
    """Imports of moved modules: `from jevdrive import a as A, b` (one statement per destination package) and relative
    imports inside the old jevdrive package (`from .x import y`, `from . import x`), made absolute when the importer
    or the imported module moved."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    plan = ctx.plan
    moved_self = ctx.new != ctx.old
    lines = text.split("\n")
    edits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom):
            continue
        if n.level == 0 and n.module == "jevdrive":
            base = "jevdrive"
        elif n.level > 0 and ctx.old.startswith("jevdrive/"):
            pkg = ctx.old[:-3].split("/")[:-n.level]
            base = ".".join(pkg + ([n.module] if n.module else []))
        else:
            continue
        groups = collections.OrderedDict()
        changed = n.level > 0 and moved_self
        if n.module and n.level > 0:              # from .x import a, b: one target module
            f = mod_file(plan, base)
            new = new_location(plan, f) if f else None
            if f and new != f:
                changed = True
            target = module_of(new) if new else base
            groups[target] = [a.name + (f" as {a.asname}" if a.asname else "") for a in n.names]
        else:                                      # from jevdrive import a / from . import a: a may be a module
            for a in n.names:
                f = mod_file(plan, f"{base}.{a.name}")
                new = new_location(plan, f) if f else None
                pkg = base
                if f and new and new != f:
                    pkg, changed = module_of(new).rsplit(".", 1)[0], True
                    if module_of(new).rsplit(".", 1)[1] != a.name:
                        ctx.fail("py-import", a.name, "module renamed", n.lineno)
                groups.setdefault(pkg, []).append(a.name + (f" as {a.asname}" if a.asname else ""))
        if changed:
            edits.append((n.lineno, n.end_lineno, n.col_offset, groups))
    for lo, hi, col, groups in sorted(edits, reverse=True):
        ind = lines[lo - 1][:col]
        trailing = re.search(r"\s+#.*$", lines[hi - 1])
        comment = trailing.group(0) if trailing else ""
        new = [f"{ind}from {pkg} import {', '.join(ns)}{comment}" for pkg, ns in groups.items()]
        before = "\n".join(lines[lo - 1:hi])
        ctx.note("py-import", before.strip(), " | ".join(x.strip() for x in new), lo)
        lines[lo - 1:hi] = [";".join([new[0]] + [x.strip() for x in new[1:]])] if lines[lo - 1][:col].strip() else new
    return "\n".join(lines)


def bare_imports(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(a.name.split(".")[0], n.lineno) for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.append((n.module.split(".")[0], n.lineno))
    return out


def script_modules(plan):
    """stem -> old .py files outside jevdrive/ that code can import by bare name (data packages excluded)."""
    st = collections.defaultdict(list)
    for f in plan.old_files:
        if f.endswith(".py") and not f.startswith("jevdrive/") and f not in plan.norewrite and plan.kind.get(f) != "data" \
                and not f.startswith(("todos/", "tmp/", "research/results/")):
            st[Path(f).parent.name if f.endswith("/__init__.py") else Path(f).stem].append(f)
    return st


def import_root(f):
    """The sys.path entry that makes module file f importable by its bare name (a package's parent dir)."""
    return pp.dirname(pp.dirname(f)) if f.endswith("/__init__.py") else pp.dirname(f)


INJECT_TAG = "# restructure: dirs of the script modules this file imports by bare name"


def inject_paths(ctx, text, stems):
    """Bare imports of repo script modules: make sure each module's NEW directory is on sys.path."""
    if INJECT_TAG in text:
        return text
    need = []
    for name, line in bare_imports(text):
        cands = stems.get(name)
        if not cands:
            continue
        same = [c for c in cands if import_root(c) == pp.dirname(ctx.old)]
        pick = same or ([c for c in cands if import_root(c) == "scripts"]) or (cands if len(cands) == 1 else [])
        if not pick:
            ctx.fail("bare-import", name, f"ambiguous module name: {cands}", line)
            continue
        o = pick[0]
        n = new_location(ctx.plan, o)
        if not n:
            ctx.fail("bare-import", name, f"imports deleted {o}", line)
            continue
        if n == o and ctx.new == ctx.old:
            continue
        need.append(import_root(n))
    need = sorted(set(need))
    if not need or (need == [pp.dirname(ctx.new)] and ctx.new == ctx.old):
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    body = tree.body
    at = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        at = body[0].end_lineno
    for n in body:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            at = max(at, n.end_lineno)
    if at == 0:
        lines0 = text.split("\n")
        while at < len(lines0) and (lines0[at].startswith("#!") or re.match(r"#.*coding[:=]", lines0[at])):
            at += 1
    depth = ctx.new.count("/")
    dirs = ", ".join(f'"{d}"' for d in need)
    block = (f"import sys as _sys, pathlib as _pl  {INJECT_TAG}\n"
             f"_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[{depth}] / _d) for _d in ({dirs},)]")
    if depth == 0:
        block = block.replace(f".parents[0] / _d", ".parent / _d")
    lines = text.split("\n")
    lines[at:at] = [block]
    ctx.note("bare-import", ",".join(sorted({n for n, _ in bare_imports(text) if n in stems})), "sys.path += " + dirs, at + 1)
    return "\n".join(lines)


# --- shell ------------------------------------------------------------------------------------------------

SH_ANCHOR_RE = re.compile(r"\$\(\s*dirname\s+\"?\$(?:0|\{BASH_SOURCE\[0\]\}|BASH_SOURCE)\"?\s*\)(?P<rel>(?:/[\w.\-]+)*)")


def rewrite_sh(ctx, text):
    plan, oldf, newf = ctx.plan, ctx.old, ctx.new
    orig = text
    if newf != oldf:
        out, pos = [], 0
        for m in SH_ANCHOR_RE.finditer(text):
            line = lineno(text, m.start())
            rel = m.group("rel")
            oldp = norm(pp.join(pp.dirname(oldf), rel.lstrip("/"))) if rel else pp.dirname(oldf)
            try:
                newp = resolve(plan, oldp) if oldp else ""
            except Unresolved as e:
                ctx.fail("sh-anchor", m.group(0), str(e), line)
                continue
            newp = oldp if newp is None else newp
            if newp == "" and oldp:
                ctx.fail("sh-anchor", m.group(0), f"points at deleted {oldp}", line)
                continue
            r = pp.relpath(newp or ".", pp.dirname(newf) or ".")
            rep = m.group(0)[: m.start("rel") - m.start()] + ("" if r == "." else "/" + r)
            if rep != m.group(0):
                ctx.note("sh-anchor", m.group(0), rep, line)
                out.append(text[pos:m.start()] + rep)
                pos = m.end()
        text = "".join(out) + text[pos:]
        text = rewrite_sh_bound(ctx, orig, text)
    return rewrite_tokens(ctx, text)


SH_BIND_RE = re.compile(r"^[ \t]*(?:local[ \t]+|export[ \t]+)?(?P<name>[A-Za-z_]\w*)=\$\((?:cd[ \t]+)?\"?\$\(\s*dirname\s+\"?\$(?:0|\{BASH_SOURCE\[0\]\}|BASH_SOURCE)\"?\s*\)(?P<rel>(?:/[\w.\-]+)*)\"?(?:[ \t]*&&[ \t]*pwd)?\)", re.M)


def rewrite_sh_bound(ctx, orig, text):
    """`$here/<lit>` where here=$(cd "$(dirname "$0")..." && pwd): point the literal at the target's new place,
    relative to the value the variable holds after the anchor rewrite (its old meaning)."""
    plan = ctx.plan
    for b in SH_BIND_RE.finditer(orig):
        name, rel = b.group("name"), b.group("rel")
        x = norm(pp.join(pp.dirname(ctx.old), rel.lstrip("/"))) if rel else pp.dirname(ctx.old)
        try:
            y = resolve(plan, x) if x else ""
        except Unresolved:
            continue
        y = x if y is None else y
        rx = re.compile(r"\$\{?" + re.escape(name) + r"\}?/(?P<lit>[\w.\-/]+[\w])")
        out, pos = [], 0
        for m in rx.finditer(text):
            oldp = norm(pp.join(x, m.group("lit")))
            try:
                newp = resolve(plan, oldp)
            except Unresolved as e:
                ctx.fail("sh-chain", m.group(0), str(e), lineno(text, m.start()))
                continue
            if newp is None or newp == norm(pp.join(y, m.group("lit"))):
                continue
            if newp == "":
                ctx.fail("sh-chain", m.group(0), f"points at deleted {oldp}", lineno(text, m.start()))
                continue
            rep = m.group(0)[: m.start("lit") - m.start()] + pp.relpath(newp, y or ".")
            ctx.note("sh-chain", m.group(0), rep, lineno(text, m.start()))
            out.append(text[pos:m.start()] + rep)
            pos = m.end()
        text = "".join(out) + text[pos:]
    return text


# --- driver -----------------------------------------------------------------------------------------------

def rewrite_file(plan, old, text, stems, all_bound):
    new = new_location(plan, old)
    ctx = Ctx(plan, old, new, [], [])
    ext = Path(old).suffix
    if ext == ".md":
        text = rewrite_md(ctx, text)
    elif ext == ".py":
        orig = text
        text, bound = rewrite_py_anchors(ctx, text)
        text = rewrite_py_bound_chains(ctx, text, orig, bound)
        text = rewrite_py_imports(ctx, text)
        text = rewrite_tokens(ctx, text)
        text = inject_paths(ctx, text, stems)
    elif ext == ".sh":
        text = rewrite_sh(ctx, text)
    else:
        text = rewrite_tokens(ctx, text)
    return text, ctx


def results_map(plan):
    """research/results/<x> -> its new directory (for manual edits that resolve result paths at run time)."""
    out = {}
    for d, n in sorted(plan.dirmap.items()):
        if d.count("/") == 2 and d.startswith("research/results/") and n:
            out[d.split("/")[2]] = n
    return out


def load_manual(path, plan):
    """restructure/manual_edits.tsv: old_path, find, replace (\\n = newline; {{results_map}} = dict literal).
    Applied after the automatic rewrite; each find must occur exactly once."""
    if not path or not Path(path).exists():
        return []
    rm = "{" + ", ".join(f'"{k}": "{v}"' for k, v in results_map(plan).items()) + "}"
    rows = []
    for r in csv.reader(open(path), delimiter="\t"):
        if not r or r[0].startswith("#") or r[0] == "old_path":
            continue
        raw_b = r[2].replace("\\n", "\n").replace("{{results_map}}", rm)
        f, a, b = r[0], r[1].replace("\\n", "\n").replace("{{base_sha}}", BASE_SHA), raw_b.replace("{{base_sha}}", BASE_SHA)
        rows.append((f, a, b, re.compile(re.escape(raw_b).replace(re.escape("{{base_sha}}"), "[0-9a-f]{7,40}"))))
    return rows


NO_REWRITE = ("restructure/summaries/", "restructure/overlay", "restructure/append/", "restructure/decisions_index.tsv",
              "restructure/untracked.tsv", "restructure/manifest.tsv", "restructure/acknowledged.tsv", "restructure/manual_edits.tsv", "restructure/report/",
              "tools/", "docs/restructure-design.md", "docs/path-map.tsv")
NO_CHECK = ("docs/restructure-design.md", "tools/restructure/")   # describes old and planned paths on purpose


def overlays():
    """restructure/overlay.tsv: path -> (expected blob of the file the overlay replaces, overlay file); 'new' = new file."""
    f = ROOT / "restructure/overlay.tsv"
    if not f.is_file():
        return {}
    rows = [r for r in csv.reader(open(f), delimiter="\t") if r and r[0] != "path" and not r[0].startswith("#")]
    return {r[0]: (r[1], ROOT / "restructure/overlay" / r[0]) for r in rows}


def rewrite_all(plan, read, manual=()):
    stems = script_modules(plan)
    results, log, unresolved = {}, [], []
    for old in sorted(plan.old_files):
        if old.startswith(NO_REWRITE):
            continue
        new = new_location(plan, old)
        if not new or old in plan.norewrite and not old.endswith(".md"):
            continue
        if Path(old).suffix not in TEXT_EXT:
            continue
        if Path(old).suffix != ".md" and plan.kind.get(old) == "data":
            continue
        try:
            text = read(old)
        except UnicodeDecodeError:
            continue
        src, mlog = text, []
        ov = overlays().get(old)
        if ov:                          # whole-file replacement, guarded by the blob the overlay was written against
            want, path = ov
            have = git("hash-object", "--", old).strip()
            if path.read_text() == src:
                pass
            elif want != have:
                unresolved.append((old, 0, "overlay", want[:10], f"{old} changed since the overlay was written (blob {have[:10]}); rewrite the overlay"))
            else:
                src = path.read_text()
                mlog.append((new, 0, "overlay", old, str(path)))
        app = ROOT / "restructure/append" / old
        if app.is_file() and app.read_text().strip().split("\n")[0] not in src:
            body = app.read_text().strip()
            lines = src.rstrip("\n").split("\n")
            at = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("Last verified")), len(lines))
            lines[at:at] = [body, ""] if at < len(lines) else ["", body]
            src = "\n".join(lines) + "\n"
            mlog.append((new, at + 1, "append", "", str(app)))
        summ = ROOT / "restructure/summaries" / old
        if summ.is_file() and "**Summary.**" not in src:
            lines = src.split("\n")
            h1 = next((i for i, l in enumerate(lines) if l.startswith("# ")), None)
            if h1 is None:
                unresolved.append((old, 0, "summary", "", "no H1 to insert the summary after"))
            else:
                lines[h1 + 1:h1 + 1] = ["", summ.read_text().strip()]
                src = "\n".join(lines)
                mlog.append((new, h1 + 2, "summary", "", str(summ)))
        for f, a, b, done in manual:
            if f == old:
                if done.search(src) and (a not in src or a in b):
                    continue          # already applied (the replacement may contain the find text)
                if src.count(a) != 1:
                    unresolved.append((old, 0, "manual", a[:60], f"manual edit text found {src.count(a)} times (want 1)"))
                    continue
                src = src.replace(a, b)
                mlog.append((new, 0, "manual", a, b))
        out, ctx = rewrite_file(plan, old, src, stems, {})
        ctx.log[:0] = mlog
        log += ctx.log
        unresolved += ctx.unresolved
        if out != text:
            results[old] = out
    return results, log, unresolved


def load_acks(path):
    if not path or not Path(path).exists():
        return []
    rows = [r for r in csv.reader(open(path), delimiter="\t") if r and not r[0].startswith("#")]
    return [(r[0], r[1], re.compile(r[2])) for r in rows]  # file glob, rule, text regex (a 4th column says why)


def acked(acks, u):
    import fnmatch
    f, line, rule, text, why = u
    return any(fnmatch.fnmatch(f, g) and (r == "*" or r == rule) and rx.search(text) for g, r, rx in acks)


def companions(plan):
    """Ignored files that travel with tracked ones: same dir + same stem (e.g. the ignored PDF next to a moved PNG), and
    everything left inside a directory that moves wholesale. Untracked, not-ignored files are refused by `check`."""
    ign = git("ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
    by_dir_stem = {}
    for o, n in plan.filemap.items():
        if n:
            by_dir_stem[(pp.dirname(o), Path(o).stem)] = n
    out = []
    for f in ign:
        if "__pycache__" in f or f.endswith((".pyc", ".DS_Store")):
            continue
        key = (pp.dirname(f), Path(f).stem)
        if key in by_dir_stem:
            n = by_dir_stem[key]
            out.append((f, pp.join(pp.dirname(n), Path(f).name)))
            continue
        for d in ancestors(f):
            if d in plan.dirmap:
                out.append((f, (plan.dirmap[d] + f[len(d):]) if plan.dirmap[d] else ""))
                break
    return out


STAGING_KEEP = ("manifest.tsv", "manual_edits.tsv", "acknowledged.tsv", "untracked.tsv", "decisions_index.tsv", "overlay.tsv")


def stash_staging():
    """Take the restructure machinery off the hot path: records go to tools/restructure/, staging copies are removed."""
    dst = ROOT / "tools/restructure"
    dst.mkdir(parents=True, exist_ok=True)
    st = ROOT / "restructure"
    if not st.is_dir():
        return
    for name in STAGING_KEEP:
        if (st / name).exists() and git("ls-files", f"restructure/{name}").strip():
            git("mv", "-k", f"restructure/{name}", f"tools/restructure/{name}")
    for d in ("readmes", "overlay", "append", "summaries"):
        if (st / d).is_dir():
            git("rm", "-r", "-q", "--cached", "--ignore-unmatch", f"restructure/{d}")
            shutil.rmtree(st / d)
    if (st / "report").is_dir():
        (dst / "report").mkdir(parents=True, exist_ok=True)
        for f in (st / "report").iterdir():
            shutil.move(str(f), str(dst / "report" / f.name))
    for f, to in (("docs/restructure-design.md", "tools/restructure/design.md"), ("tools/restructure.py", "tools/restructure/restructure.py"),
                  ("tools/restructure_plan.py", "tools/restructure/restructure_plan.py"),
                  ("tools/split_decisions.py", "tools/restructure/split_decisions.py"), ("tools/token_bench.py", "tools/restructure/token_bench.py")):
        if git("ls-files", f).strip():
            git("mv", "-k", f, to)
    shutil.rmtree(st, ignore_errors=True)
    git("add", "-A", "--", *[p for p in ("restructure", "tools") if (ROOT / p).exists() or git("ls-files", p).strip()])


def cmd_check(a):
    files = tracked()
    plan, errors = load_manifest(a.manifest, files)
    for e in errors:
        print("ERROR", e)
    print(f"{len(plan.rows)} rows, {len(plan.filemap)} files move or go, {len(plan.dirmap)} dirs move wholesale")
    return 1 if errors else 0


def cmd_apply(a):
    files = tracked()
    plan, errors = load_manifest(a.manifest, files)
    if errors:
        for e in errors:
            print("ERROR", e)
        return 1
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty and not a.dry_run:
        print("ERROR: tracked files have uncommitted changes; commit or stash first")
        return 1
    results, log, unresolved = rewrite_all(plan, lambda f: (ROOT / f).read_text(), load_manual(a.manual, plan))
    acks = load_acks(a.acks)
    hard = [u for u in unresolved if not acked(acks, u)]
    rep = Path(a.report)
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "rewrites.tsv", "w") as fh:
        fh.write("file\tline\trule\tbefore\tafter\n")
        for r in log:
            fh.write("\t".join(str(x).replace("\t", " ").replace("\n", "\\n") for x in r) + "\n")
    with open(rep / "unresolved.tsv", "w") as fh:
        fh.write("file\tline\trule\ttext\twhy\tacknowledged\n")
        for u in unresolved:
            fh.write("\t".join(str(x).replace("\t", " ").replace("\n", "\\n") for x in u) + f"\t{'yes' if acked(acks, u) else 'NO'}\n")
    by = collections.Counter(r[2] for r in log)
    print(f"rewrites: {len(log)} in {len(results)} files ({dict(by)}); unresolved: {len(unresolved)} ({len(hard)} not acknowledged)")
    if hard:
        for u in hard[:40]:
            print("UNRESOLVED", *u)
        print(f"see {rep / 'unresolved.tsv'}; acknowledge deliberate cases in {a.acks}")
        if not a.force:
            return 2
    moves = sorted((o, n) for o, n in plan.filemap.items() if n)
    dels = sorted(o for o, n in plan.filemap.items() if not n)
    comp = companions(plan)
    with open(rep / "companions.tsv", "w") as fh:
        fh.write("ignored file\tmoves to (empty = stays, its dir is deleted)\n")
        fh.writelines(f"{o}\t{n}\n" for o, n in comp)
    print(f"moves: {len(moves)}, deletions: {len(dels)}, ignored companions: {len(comp)} ({sum(1 for _, n in comp if not n)} left in deleted dirs)")
    if a.dry_run:
        return 0
    for o, n in moves:
        (ROOT / n).parent.mkdir(parents=True, exist_ok=True)
        git("mv", "-k", o, n)
    for i in range(0, len(dels), 200):
        git("rm", "-q", "--", *dels[i:i + 200])
    for old, text in results.items():
        (ROOT / new_location(plan, old)).write_text(text)
    for o, n in comp:
        if n and (ROOT / o).exists() and not (ROOT / n).exists():
            (ROOT / n).parent.mkdir(parents=True, exist_ok=True)
            os.rename(ROOT / o, ROOT / n)
    # path map for old references in history and notes
    pm = ROOT / "docs/path-map.tsv"
    with open(pm, "w") as fh:
        fh.write("# old path -> new path of the 2026-10 restructure (rows ending in / cover a whole dir); empty = removed:\n"
                 "# read it with `git show <last commit before the restructure>:<old path>`\n")
        for r in plan.rows:
            if r["old_path"] != r["new_path"]:
                fh.write(f"{r['old_path']}\t{r['new_path']}\n")
    for path, (want, src) in overlays().items():   # new files from the overlay (replacements were done above)
        if want == "new" and not (ROOT / path).exists():
            (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, ROOT / path)
    ti = next(p for p in (ROOT / "tools/topic_index.py", Path(__file__).with_name("topic_index.py")) if p.exists())
    subprocess.run([sys.executable, str(ti)], cwd=ROOT, check=True)
    git("add", "-A", "--", *sorted({str(Path(new_location(plan, o)).parts[0]) for o in results}
                                   | {"docs", "experiments", "research"}))
    stash_staging()
    for d in sorted({a for o in plan.filemap for a in ancestors(o)}, key=lambda d: -d.count("/")):
        try:
            (ROOT / d).rmdir()          # empty leftovers only, deepest first
        except OSError:
            pass
    print("applied; review `git status`, then run `verify` and `compare`")
    return 0


# ------------------------------------------------------------------------------------------------ verify

SANDBOX = "(version 1)(allow default)(deny network-outbound (remote ip))(deny file-write* (subpath \"/Users\"))"


def sandboxed(cmd, cwd, timeout):
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(cwd), "HOME": "/tmp/restructure-home",
           "DATA_DIR": "/nonexistent-data-dir", "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
           "LANG": "en_US.UTF-8"}
    os.makedirs(env["HOME"], exist_ok=True)
    if shutil.which("sandbox-exec"):
        cmd = ["sandbox-exec", "-p", SANDBOX, *cmd]
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return "timeout", ""


def classify_error(out, cwd):
    if not out:
        return "ok"
    m = re.findall(r"(ModuleNotFoundError: No module named '([\w.]+)'|ImportError: [^\n]+|FileNotFoundError: [^\n]+|\w+Error: [^\n]*)", out)
    if not m:
        return "error"
    last, mod = m[-1]
    if re.search(r"sched_getaffinity|sched_setaffinity|/proc/|No module named '(carla|agents|leaderboard|srunner)'", last + out[-300:]):
        return "env-missing"
    if last.startswith("ModuleNotFoundError"):
        top = mod.split(".")[0]
        repo_like = (Path(cwd) / top).exists() or (Path(cwd) / "scripts" / f"{top}.py").exists() or top in {"jevdrive", "experiments"}
        return f"repo-module-missing:{mod}" if repo_like else "env-missing"
    return re.sub(r"(/[\w./-]+)", "<path>", last)[:160]


def import_safe(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    ok = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test):
            continue
        if isinstance(n, ast.Try) and all(isinstance(x, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr)) for x in n.body):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and "sys.path" in ast.unparse(n.value):
            continue
        if isinstance(n, ok):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
                    isinstance(c, ast.Call) and not re.match(r"^(Path|str|int|float|dict|list|set|tuple|frozenset|re\.compile|os\.environ\.get|"
                                                              r"os\.path\.\w+|getattr|range|sorted|collections\.\w+|np\.\w+|field|"
                                                              r"logging\.getLogger|namedtuple|TypeVar|.*Path.*|.*\.resolve|.*environ.*)$",
                                                              ast.unparse(c.func)) for c in ast.walk(n)):
                return False
            continue
        return False
    return True


def anchor_dir(f, expr):
    """Repo dir an anchor expression evaluates to for a file at repo path f ('' = root), or None."""
    m = re.match(r"Path\(\s*__file__\s*\)(?:\.resolve\(\)|\.absolute\(\))?(?P<up>(?:\.parent(?!s))+|\.parents\[(?P<k>\d+)\])", expr)
    if not m:
        m2 = re.match(r"(?P<dn>(?:os\.path\.dirname\(\s*)+)os\.path\.(?:abspath|realpath)\(\s*__file__", expr)
        if not m2:
            return None
        lv = m2.group("dn").count("dirname")
    else:
        lv = int(m.group("k")) + 1 if m.group("k") is not None else m.group("up").count(".parent")
    d = f
    for _ in range(lv):
        if d == "":
            return None
        d = pp.dirname(d)
    return d


def path_dirs(f, text):
    """Dirs a file puts on sys.path through __file__ anchors, names bound to them, and literal joins (static)."""
    names = {}
    for m in re.finditer(r"^[ \t]*([A-Za-z_]\w*)\s*=\s*(Path\(\s*__file__[^\n]*|os\.path\.dirname\([^\n]*__file__[^\n]*)$", text, re.M):
        d = anchor_dir(f, m.group(2).strip())
        if d is not None:
            lits = re.findall(r"/\s*[\"']([^\"']+)[\"']", m.group(2))
            names[m.group(1)] = norm(pp.join(d, *lits))
    for m in re.finditer(r"^[ \t]*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)(\.parent(?!s)|\.parents\[(\d+)\])?((?:\s*/\s*[\"'][^\"']+[\"'])*)\s*$", text, re.M):
        if m.group(2) in names and m.group(1) not in names:
            d = names[m.group(2)]
            up = 1 if m.group(3) == ".parent" else (int(m.group(4)) + 1 if m.group(4) else 0)
            for _ in range(up):
                d = pp.dirname(d)
            names[m.group(1)] = norm(pp.join(d, *re.findall(r"[\"']([^\"']+)[\"']", m.group(5) or "")))
    out = set()
    for m in re.finditer(r"(?:sys\.path\.(?:insert|append)|sys\.path\[:0\]|_sys\.path\[:0\])[^\n]*", text):
        line = m.group(0)
        inj = re.search(r"parents\[(\d+)\] / _d\) for _d in \(([^)]*)\)", line)
        if inj:
            base = anchor_dir(f, f"Path(__file__).parents[{inj.group(1)}]")
            if base is not None:
                out |= {norm(pp.join(base, d)) for d in re.findall(r"\"([^\"]+)\"", inj.group(2))}
            continue
        for am in re.finditer(r"(Path\(\s*__file__\s*\)[\w.()\[\]]*|(?:os\.path\.dirname\(\s*)+os\.path\.(?:abspath|realpath)\(\s*__file__\s*\)\)*)((?:\s*/\s*[\"'][^\"']+[\"'])*)", line):
            d = anchor_dir(f, am.group(1))
            if d is not None:
                out.add(norm(pp.join(d, *re.findall(r"[\"']([^\"']+)[\"']", am.group(2) or ""))))
        for nm in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)(\.parent(?!s))?((?:\s*/\s*[\"'][^\"']+[\"'])*)", line):
            if nm.group(1) in names:
                d = names[nm.group(1)]
                if nm.group(2):
                    d = pp.dirname(d)
                out.add(norm(pp.join(d, *re.findall(r"[\"']([^\"']+)[\"']", nm.group(3) or ""))))
    return out


def static_imports(cwd, files):
    """Every import of a repo module resolves to a file in this tree (jevdrive.x / experiments.x and bare names that
    exist somewhere in the repo are checked against sys.path as the file sets it: own dir, injected dirs, scripts/, research/)."""
    stems = collections.defaultdict(set)
    for f in files:
        if f.endswith(".py"):
            stems[Path(f).stem].add(pp.dirname(f))
    bad = {}
    for f in files:
        if not f.endswith(".py") or f.startswith(("todos/", "tmp/")) or "/results/" in f:
            continue
        text = (cwd / f).read_text(errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        reach = {pp.dirname(f)} | path_dirs(f, text)
        missing = []
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module] + ([f"{n.module}.{a.name}" for a in n.names] if n.module.split(".")[0] in {"jevdrive", "experiments"} else [])
            for nm in names:
                top = nm.split(".")[0]
                if top in {"jevdrive", "experiments"}:
                    p = cwd / nm.replace(".", "/")
                    par = cwd / nm.rsplit(".", 1)[0].replace(".", "/")
                    if not (p.with_suffix(".py").exists() or p.is_dir() or par.with_suffix(".py").exists() or (par / "__init__.py").exists()):
                        missing.append(nm)
                elif top in stems and top not in sys.stdlib_module_names:
                    dirs = stems[top]
                    if not (dirs & reach) and not any(d.startswith(tuple(r + "/" for r in reach if r)) and False for d in dirs):
                        missing.append(nm)
        if missing:
            bad[f] = sorted(set(missing))
    return bad


def md_check(cwd, files):
    """Broken relative links and repo-path tokens in markdown; figures that nothing references."""
    fs = set(files)
    dirs = {a for f in files for a in ancestors(f)}
    broken, referenced = [], set()
    edges = collections.defaultdict(set)
    for f in files:
        if not f.endswith(".md") or f.startswith(NO_CHECK):
            continue
        text = (cwd / f).read_text(errors="replace")
        for m in list(LINK_RE.finditer(text)) + list(HTML_SRC_RE.finditer(text)):
            target = m.group(3) if m.re is LINK_RE else m.group(2)
            if re.match(r"^[a-z][\w+.-]*:", target) or target.startswith(("#", "/")):
                continue
            path = norm(pp.join(pp.dirname(f), target.split("#")[0]))
            if not target.split("#")[0]:
                continue
            referenced.add(path)
            edges[f].add(path)
            if path not in fs and path not in dirs:
                broken.append((f, "link", target))
        bare = HTML_SRC_RE.sub("", LINK_RE.sub("", text))
        for m in PATH_RE.finditer(bare):
            tok = m.group(1)
            before = bare[max(0, m.start() - 41):m.start()]
            if re.search(r"[0-9a-f]{7,40}:$", before) or (before.endswith("/") and not ROOT_PREFIX_RE.search(before)):
                continue
            if any(c in tok for c in "*{}"):
                continue
            p = norm(tok)
            referenced.add(p)
            edges[f].add(p)          # a repo-root path in prose is a reference an agent can follow
            if p not in fs and p not in dirs and not any(a in dirs and a.split("/")[0] in ("experiments",) for a in []) and \
                    p.split("/")[0] in REPO_TOPS and not (cwd / p).exists():
                broken.append((f, "token", tok))
    under = lambda d: [x for x in files if x.startswith(d + "/")] if d in dirs else []
    seen, todo = set(), ["README.md"]          # every doc reachable from README.md through links (a dir link covers it)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for t in edges.get(f, ()):
            for x in ([t] if t in fs else under(t)):
                referenced.add(x)
                if x.endswith(".md") and x not in seen:
                    todo.append(x)
    figs = [f for f in files if re.search(r"\.(png|webp|jpg|jpeg|gif|svg)$", f) and ("/figs/" in f or f.startswith("research/figs/"))]
    unref = [f for f in figs if f not in referenced]
    unreach = sorted(f for f in files if f.endswith(".md") and f not in seen and not f.startswith(NO_CHECK))
    return broken, unref, unreach


def cmd_verify(a):
    cwd = ROOT
    files = [f for f in tracked() if (cwd / f).is_file() and not (cwd / f).is_symlink()]
    out = {"tree": str(cwd), "files": len(files)}
    py = [f for f in files if f.endswith(".py") and not f.startswith(("todos/", "tmp/")) and "/results/" not in f
          and not f.startswith("research/results/")]
    # 1 compileall (in memory, no pyc written)
    comp = {}
    for f in py:
        try:
            compile((cwd / f).read_text(errors="replace"), f, "exec")
        except SyntaxError as e:
            comp[f] = f"{e.msg} line {e.lineno}"
    out["compile_fail"] = comp
    # 2 static import resolution
    out["static_import_missing"] = static_imports(cwd, files)
    # 3 import every import-safe module, 4 --help on argparse scripts; sandboxed (no network, no writes under /Users)
    jobs = []
    if not a.quick:
        for f in py:
            if f.startswith("tools/"):
                continue
            text = (cwd / f).read_text(errors="replace")
            mod = module_of(f)
            if import_safe(text) and mod and "-" not in Path(f).stem:
                if f.startswith("jevdrive/"):   # library: imported as a package module
                    code = f"import importlib; importlib.import_module({mod!r})"
                else:                           # script module: imported the way `python file.py` sees it (own dir first)
                    code = f"import sys, importlib; sys.path.insert(0, {pp.dirname(f) or '.'!r}); importlib.import_module({Path(f).stem!r})"
                jobs.append(("import", f, [sys.executable, "-c", code], 60))
            if "ArgumentParser" in text and "__main__" in text:
                jobs.append(("help", f, [sys.executable, f, "--help"], 30))
        for f in files:
            if re.search(r"(^|/)test_[\w]+\.py$", f) and not f.startswith(("todos/", "tmp/", "research/results/")) and "/results/" not in f:
                jobs.append(("tests", f, [sys.executable, f], 180))
    res = {"import": {}, "help": {}, "tests": {}}

    def run(job):
        kind, f, cmd, to = job
        rc, o = sandboxed(cmd, cwd, to)
        return kind, f, "ok" if rc == 0 else ("timeout" if rc == "timeout" else classify_error(o, cwd) or f"rc={rc}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(4, (os.cpu_count() or 8) - 2)) as ex:
        for kind, f, r in ex.map(run, jobs):
            res[kind][f] = r
    imp, hl = res["import"], res["help"]
    out["import"], out["help"] = imp, hl
    # 5 bash -n
    out["bash_n_fail"] = {f: o for f in files if f.endswith(".sh") and not f.startswith(("todos/", "tmp/"))
                          for rc, o in [sandboxed(["bash", "-n", f], cwd, 10)] if rc != 0}
    # 6 markdown links / paths, figures
    broken, unref, unreach = md_check(cwd, files)
    out["md_broken"] = [list(b) for b in broken]
    out["fig_unreferenced"] = unref
    out["md_unreachable"] = unreach
    tests = res["tests"]
    out["tests"] = tests
    out["file_hashes"] = {l.split("\t", 1)[1]: l.split()[1] for l in git("ls-files", "-s", cwd=cwd).splitlines()}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"compile_fail {len(comp)}, static_import_missing {len(out['static_import_missing'])}, import {sum(v == 'ok' for v in imp.values())}/{len(imp)} ok, "
          f"help {sum(v == 'ok' for v in hl.values())}/{len(hl)} ok, bash_n_fail {len(out['bash_n_fail'])}, md_broken {len(broken)}, "
          f"fig_unreferenced {len(unref)}, md_unreachable {len(unreach)}, tests {sum(v == 'ok' for v in tests.values())}/{len(tests)} ok")
    return 0


def cmd_compare(a):
    """Compare a verify result on the moved tree with one on the unmoved tree, mapping old paths through the manifest."""
    base, after = json.loads(Path(a.baseline).read_text()), json.loads(Path(a.after).read_text())
    files = list(base["file_hashes"])
    plan, _ = load_manifest(a.manifest, files) if False else (None, None)
    rows = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    fm = {}
    for r in rows:
        o, n = r["old_path"], r["new_path"]
        for f in ([x for x in files if x.startswith(o)] if o.endswith("/") else [o]):
            fm[f] = (n + f[len(o):] if o.endswith("/") else n) if n else ""
    m = lambda f: fm.get(f, f)
    rep, ok = [], True

    def per_file(key, good=("ok",)):
        nonlocal ok
        b, c = base[key], after[key]
        reg, fixed, same_bad = [], [], 0
        for f, v in b.items():
            n = m(f)
            if not n:
                continue
            w = c.get(n, "absent")
            vn, wn = re.sub(r"<path>", "", str(v)), re.sub(r"<path>", "", str(w))
            if vn == wn:
                same_bad += v not in good
                continue
            rank = lambda x: 0 if x in good else 1 if x == "env-missing" else 3 if str(x).startswith(("repo-module-missing", "absent")) else 2
            if rank(w) < rank(v):
                fixed.append(f)
            elif rank(w) > rank(v) or rank(w) == 2:
                reg.append((f, n, v, w))
            else:
                same_bad += 1
        new_only = [n for n in c if n not in {m(f) for f in b}]
        if reg:
            ok = False
        rep.append(f"{key}: baseline {sum(v in good for v in b.values())}/{len(b)} ok, after {sum(v in good for v in c.values())}/{len(c)} ok; "
                   f"regressions {len(reg)}, fixed {len(fixed)}, unchanged non-ok {same_bad}, new entries {len(new_only)}")
        for r in reg[:30]:
            rep.append(f"   REGRESSION {r[0]} -> {r[1]}: {r[2]} => {r[3]}")
        return reg

    per_file("import")
    per_file("help")
    per_file("tests")
    for key in ("compile_fail", "bash_n_fail"):
        b = {m(f) for f in base[key]}
        c = set(after[key])
        new = c - b
        ok &= not new
        rep.append(f"{key}: baseline {len(b)}, after {len(c)}, new {sorted(new)[:20]}")
    b = {m(f): v for f, v in base["static_import_missing"].items()}
    new = {f: v for f, v in after["static_import_missing"].items() if f not in b or set(v) - set(b[f])}
    ok &= not new
    rep.append(f"static_import_missing: baseline {len(b)} files, after {len(after['static_import_missing'])}, new {len(new)}")
    for f, v in list(new.items())[:30]:
        rep.append(f"   NEW {f}: {v}")
    split = lambda f: re.sub(r"^research/decisions/\d+[a-z]?(-\d+)?\.md$", "research/decisions.md", f)  # the split log

    def keyb(f, k, t):   # a broken reference, mapped to the new tree for comparison
        return (split(m(f)), Path(t.split("#")[0]).name)
    bb = {keyb(*b) for b in base["md_broken"]}
    acks = load_acks(a.acks)
    newb = [b for b in after["md_broken"] if (split(b[0]), Path(b[2].split("#")[0]).name) not in bb
            and not acked(acks, (b[0], 0, "md-broken", b[2], ""))]
    ok &= not newb
    rep.append(f"md_broken: baseline {len(base['md_broken'])}, after {len(after['md_broken'])}, new {len(newb)}")
    rep += [f"   NEW BROKEN {f} [{k}] {t}" for f, k, t in newb[:40]]
    rep.append(f"fig_unreferenced: baseline {len(base['fig_unreferenced'])}, after {len(after['fig_unreferenced'])} "
               f"(new: {sorted(set(after['fig_unreferenced']) - {m(f) for f in base['fig_unreferenced']})[:10]})")
    ub = {m(f) for f in base.get("md_unreachable", [])}
    newu = [f for f in after.get("md_unreachable", []) if f not in ub]
    ok &= not newu
    rep.append(f"md_unreachable from README.md: baseline {len(base.get('md_unreachable', []))}, after "
               f"{len(after.get('md_unreachable', []))}, new {len(newu)}")
    rep += [f"   NEW UNREACHABLE {f}" for f in newu[:30]]
    # conservation: every baseline file is present at its new path (or deleted on purpose); data files byte-identical
    lost, changed_data = [], []
    for f, h in base["file_hashes"].items():
        n = m(f)
        if n == "":
            continue
        if n not in after["file_hashes"]:
            lost.append(f)
        elif fm.get(f) is not None and h != after["file_hashes"][n] and not re.search(r"\.(md|py|sh|json|yaml|yml|toml|txt|tsv|cfg)$", f):
            changed_data.append(f)
    deleted = sorted(f for f in base["file_hashes"] if m(f) == "")
    ok &= not lost and not changed_data
    rep.append(f"files: baseline {len(base['file_hashes'])}, after {len(after['file_hashes'])}, deleted on purpose {len(deleted)} "
               f"(todos/ {sum(f.startswith('todos/') for f in deleted)}, tmp/ {sum(f.startswith('tmp/') for f in deleted)}), "
               f"lost {len(lost)}, binary changed {len(changed_data)}")
    rep += [f"   LOST {f}" for f in lost[:20]]
    Path(a.out).write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def staged(path):
    """restructure/<x> before the apply, tools/restructure/<x> after it."""
    alt = "tools/restructure/" + path.split("/", 1)[1]
    return path if Path(path).exists() or not Path(alt).exists() else alt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "apply"):
        p = sp.add_parser(name)
        p.add_argument("--manifest", default="restructure/manifest.tsv")
        if name == "apply":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--force", action="store_true", help="apply although unacknowledged unresolved references exist")
            p.add_argument("--acks", default="restructure/acknowledged.tsv")
            p.add_argument("--manual", default="restructure/manual_edits.tsv")
            p.add_argument("--report", default="restructure/report")
    p = sp.add_parser("verify")
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true", help="static checks only (no imports, --help, tests)")
    p = sp.add_parser("compare")
    p.add_argument("--baseline", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--manifest", default="restructure/manifest.tsv")
    p.add_argument("--out", default="restructure/report/compare.txt")
    p.add_argument("--acks", default="restructure/acknowledged.tsv")
    a = ap.parse_args()
    for k in ("manifest", "acks", "manual", "report"):
        if getattr(a, k, None):
            setattr(a, k, staged(getattr(a, k)))
    return {"check": cmd_check, "apply": cmd_apply, "verify": cmd_verify, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
