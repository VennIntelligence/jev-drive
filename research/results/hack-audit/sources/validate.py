#!/usr/bin/env python3
"""静态验收审计交付物；仅读取 CSV/JSON/Markdown 和固定 commit 的 Git 对象。

最终交付默认检查全部五项：
    python3 out/sources/validate.py

可在阶段 2/3 尚未结束时单独核查：
    python3 out/sources/validate.py --checks sampling,findings

复核约定：matrix.csv 是 board,repo,category,present 的长表；review.csv 是
board,repo,category,original,verdict,reviewer,selection_seed，可另加 reason/evidence。
repo 可填 repos.csv 的 slug、method、repo_url 或 repo_path。随机抽样由本脚本定义：
对 yes/no 单元格分别按 (board,slug,category) 排序，使用同一个
random.Random(selection_seed) 先 sample(yes, ceil(20%))，再
sample(no, ceil(10%))。下面的命令只向 stdout 打印待复核 CSV，不写文件：
    python3 out/sources/validate.py --draw-review 20260924 > out/review.csv

独立代理不继承上下文一事无法从文件中证明；脚本只核对 reviewer 与阶段 1
负责人不同。脚本离线验证 URL 的 repo、commit、文件和行号对应固定 Git 对象
中的原文；加 --check-http 可并发尝试访问链接，网络故障只产生 WARN。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


BOARDS = {
    'bench2drive', 'carla_lb2', 'navsim_v1', 'navsim_v2',
    'nuscenes', 'wod_e2e', 'hugsim',
}
FINDING_FIELDS = {
    'id', 'board', 'repo', 'commit', 'category', 'title', 'location',
    'snippet', 'description', 'deploy_valid', 'disclosed', 'paper_quote',
    'impact_evidence', 'confidence',
}
CHECKS = ('sampling', 'index', 'findings', 'matrix', 'review')
CATEGORY_RE = re.compile(r'^###\s+`?([a-z][a-z0-9_]*)`?(?:\s|$)')
INDEX_LINK_RE = re.compile(r'^\[[^]]+\]\((audits/([^/)]+)\.md)\)$')
ANCHOR_RE = re.compile(r'^L([1-9]\d*)(?:-L([1-9]\d*))?$')


def read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = required - set(columns)
        if missing:
            raise ValueError(f'{path}: 缺列 {sorted(missing)}')
        if len(columns) != len(set(columns)):
            raise ValueError(f'{path}: 列名重复')
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        if None in row:
            raise ValueError(f'{path}:{line}: 字段数超过表头')
        if any(value is None for value in row.values()):
            raise ValueError(f'{path}:{line}: 字段数少于表头')
    return rows, columns


def repo_url(value: str) -> str:
    return value.rstrip('/').removesuffix('.git').lower()


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {'https', 'http'} and bool(parsed.netloc)


def nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class AuditData:
    def __init__(self, root: Path, findings_override: Path | None):
        self.root = root
        self.out = root / 'out'
        self.findings_path = findings_override or (self.out / 'findings.jsonl')
        self.sampling: list[dict[str, str]] | None = None
        self.repos: list[dict[str, str]] | None = None
        self.aliases: dict[tuple[str, str], tuple[str, str]] = {}
        self.units: dict[tuple[str, str], dict[str, str]] = {}
        self.index: dict[tuple[str, str], dict[str, str]] | None = None
        self.findings: list[dict] | None = None
        self.matrix: dict[tuple[str, str, str], str] | None = None
        self.stage1_owners: set[str] = set()

    def load_sampling_repos(self):
        if self.sampling is None:
            self.sampling, _ = read_csv(
                self.out / 'sampling.csv',
                {'board', 'rank', 'method', 'score', 'ranking_source_url',
                 'snapshot_date', 'repo_url', 'code_grade', 'selected',
                 'not_selected_reason'},
            )
        if self.repos is None:
            self.repos, _ = read_csv(
                self.out / 'repos.csv',
                {'board', 'method', 'slug', 'repo_url', 'repo_path', 'commit',
                 'clone_status', 'paper_path', 'paper_status'},
            )
            for row in self.repos:
                key = (row['board'], row['slug'])
                if key in self.units:
                    raise ValueError(f'repos.csv 单元重复: {key}')
                self.units[key] = row
                for alias in {
                    row['slug'], row['method'], row['repo_url'],
                    row['repo_path'], f"{row['board']}__{row['slug']}",
                }:
                    canonical = repo_url(alias) if is_url(alias) else alias.lower()
                    alias_key = (row['board'], canonical)
                    prior = self.aliases.get(alias_key)
                    if prior is not None and prior != key:
                        raise ValueError(f'repo 别名冲突: {alias_key}: {prior}/{key}')
                    self.aliases[alias_key] = key

    def unit(self, board: str, value: str) -> tuple[str, str] | None:
        self.load_sampling_repos()
        alias = repo_url(value) if is_url(value) else value.lower()
        return self.aliases.get((board, alias))

    def load_index(self):
        if self.index is not None:
            return
        self.load_sampling_repos()
        self.index = {}
        for number, line in enumerate((self.out / 'INDEX.md').read_text(encoding='utf-8').splitlines(), 1):
            if not line.startswith('|') or line.startswith('| ---'):
                continue
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if len(cells) != 5 or cells[0] not in BOARDS:
                continue
            board, method, owner, status, link = cells
            matched = INDEX_LINK_RE.fullmatch(link)
            if matched is None:
                raise ValueError(f'INDEX.md:{number}: 审计页链接格式错误: {link}')
            filename = matched[2]
            prefix = board + '__'
            if not filename.startswith(prefix):
                raise ValueError(f'INDEX.md:{number}: 审计页与榜单不符: {link}')
            slug = filename[len(prefix):]
            key = (board, slug)
            if key in self.index:
                raise ValueError(f'INDEX.md:{number}: 单元重复: {key}')
            self.index[key] = dict(method=method, owner=owner, status=status,
                                   link=matched[1], line=str(number))
            self.stage1_owners.add(owner)

    def load_findings(self):
        if self.findings is not None:
            return
        self.findings = []
        for number, line in enumerate(self.findings_path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{self.findings_path}:{number}: JSON 错误: {exc}') from exc
            if not isinstance(item, dict):
                raise ValueError(f'{self.findings_path}:{number}: 发现须为 JSON object')
            item['_line'] = number
            self.findings.append(item)

    def load_matrix(self):
        if self.matrix is not None:
            return
        rows, columns = read_csv(self.out / 'matrix.csv',
                                 {'board', 'repo', 'category', 'present'})
        if any(char in (self.out / 'matrix.csv').read_text(encoding='utf-8') for char in (' ', '\t')):
            raise ValueError('matrix.csv 含空格或制表符')
        self.matrix = {}
        for number, row in enumerate(rows, 2):
            unit = self.unit(row['board'], row['repo'])
            if unit is None:
                raise ValueError(f'matrix.csv:{number}: 未知单元 {row["board"]}/{row["repo"]}')
            key = (*unit, row['category'])
            if key in self.matrix:
                raise ValueError(f'matrix.csv:{number}: 重复单元格 {key}')
            self.matrix[key] = row['present']


def check_sampling(data: AuditData) -> tuple[str, list[str]]:
    data.load_sampling_repos()
    errors: list[str] = []
    rows = data.sampling or []
    repos = data.repos or []
    coverage = {row['board'] for row in rows}
    if coverage != BOARDS:
        errors.append(f'榜单覆盖 {sorted(coverage)}，应为 {sorted(BOARDS)}')
    selected = {}
    for line, row in enumerate(rows, 2):
        label = f'sampling.csv:{line} {row["board"]}/{row["method"]}'
        if row['board'] not in BOARDS:
            errors.append(f'{label}: 未知榜单')
        if not row['rank'].isdigit() or int(row['rank']) < 1:
            errors.append(f'{label}: rank 非正整数')
        if not all(nonblank(row[name]) for name in ('method', 'score', 'snapshot_date')):
            errors.append(f'{label}: 方法/分数/快照日期缺失')
        if not is_url(row['ranking_source_url']):
            errors.append(f'{label}: 排行来源 URL 缺失或无效')
        if row['code_grade'] not in {'A', 'B', 'C'}:
            errors.append(f'{label}: code_grade 非 A/B/C')
        if row['selected'] not in {'yes', 'no'}:
            errors.append(f'{label}: selected 非 yes/no')
        if row['selected'] == 'no' and not nonblank(row['not_selected_reason']):
            errors.append(f'{label}: 未选原因缺失')
        if row['selected'] == 'yes':
            key = (row['board'], row['method'])
            if key in selected:
                errors.append(f'{label}: 被选方法重复')
            selected[key] = row
            if row['code_grade'] not in {'A', 'B'}:
                errors.append(f'{label}: C 档不应入选')
            if not is_url(row['repo_url']):
                errors.append(f'{label}: 入选但 repo URL 无效')
    repo_keys = {(row['board'], row['method']) for row in repos}
    if set(selected) != repo_keys:
        errors.append(f'入选行与 repos.csv 不一致：sampling 独有 {sorted(set(selected)-repo_keys)}；repos 独有 {sorted(repo_keys-set(selected))}')
    for line, row in enumerate(repos, 2):
        label = f'repos.csv:{line} {row["board"]}/{row["slug"]}'
        if not re.fullmatch(r'[0-9a-f]{40}', row['commit']):
            errors.append(f'{label}: commit 不是完整 40 位 hash')
        sample = selected.get((row['board'], row['method']))
        if sample and repo_url(sample['repo_url']) != repo_url(row['repo_url']):
            errors.append(f'{label}: 仓库 URL 与抽样表不同')
        if row['clone_status'] == 'ok' and not (data.root / row['repo_path'] / '.git').exists():
            errors.append(f'{label}: clone_status=ok 但本地 Git 仓库缺失')
        if row['paper_status'] == 'ok' and not (data.root / row['paper_path']).is_file():
            errors.append(f'{label}: paper_status=ok 但 PDF 缺失')
        if not nonblank(row['clone_status']) or not nonblank(row['paper_status']):
            errors.append(f'{label}: clone/paper 状态缺失')
    counts = Counter(row['board'] for row in rows if row['selected'] == 'yes')
    detail = f'{len(rows)} 个抽样条目、{len(repos)} 个入选单元、7 榜覆盖；入选数 {dict(sorted(counts.items()))}'
    return detail, errors


def check_index(data: AuditData) -> tuple[str, list[str]]:
    data.load_index()
    errors: list[str] = []
    expected = set(data.units)
    actual = set(data.index or {})
    if expected != actual:
        errors.append(f'INDEX 与 repos.csv 单元不一致：缺 {sorted(expected-actual)}；多 {sorted(actual-expected)}')
    for key, row in (data.index or {}).items():
        label = f'INDEX.md:{row["line"]} {key[0]}/{key[1]}'
        repo = data.units.get(key)
        if repo and row['method'] != repo['method']:
            errors.append(f'{label}: 方法名与 repos.csv 不符')
        if row['status'] not in {'done', 'verified'}:
            errors.append(f'{label}: 交付时状态仍为 {row["status"]}')
        if not nonblank(row['owner']):
            errors.append(f'{label}: 负责人为空')
        audit = data.out / row['link']
        if not audit.is_file():
            errors.append(f'{label}: 审计页缺失 {audit}')
            continue
        page = audit.read_text(encoding='utf-8')
        if not re.search('已读|读过', page):
            errors.append(f'{label}: 审计页未列已读文件')
        if '未读' not in page:
            errors.append(f'{label}: 审计页未列未读内容')
        if '整体印象' not in page:
            errors.append(f'{label}: 审计页缺整体印象')
        if '发现' not in page:
            errors.append(f'{label}: 审计页缺发现列表或无发现说明')
    detail = f'{len(actual)} 个任务板单元、{sum((data.out / row["link"]).is_file() for row in (data.index or {}).values())} 页审计'
    return detail, errors


def parse_permalink(location: str):
    parsed = urlparse(location)
    if parsed.scheme != 'https' or parsed.hostname != 'github.com' or parsed.query:
        raise ValueError('须为 GitHub HTTPS permalink')
    parts = parsed.path.lstrip('/').split('/')
    if len(parts) < 5 or parts[2] != 'blob' or not parts[0] or not parts[1]:
        raise ValueError('路径须包含 owner/repo/blob/commit/file')
    owner, name, _, commit = parts[:4]
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('URL 未固定到完整 commit')
    path = unquote('/'.join(parts[4:]))
    if not path or path.startswith('/') or any(part in {'.', '..'} for part in path.split('/')):
        raise ValueError('文件路径为空或含路径跳转')
    anchor = ANCHOR_RE.fullmatch(parsed.fragment)
    if not anchor:
        raise ValueError('锚点须为 #L起始-L结束 或 #L单行')
    start = int(anchor[1]); end = int(anchor[2] or anchor[1])
    if end < start or end - start + 1 > 15:
        raise ValueError('行范围倒置或超过 15 行')
    return f'https://github.com/{owner}/{name}', commit, path, start, end


def git_lines(repo_path: Path, commit: str, path: str) -> list[str]:
    proc = subprocess.run(
        ['git', '-C', str(repo_path), 'show', f'{commit}:{path}'],
        capture_output=True, timeout=20, check=False,
    )
    if proc.returncode:
        message = proc.stderr.decode('utf-8', errors='replace').strip()
        raise ValueError(f'Git 对象不可读: {message[:240]}')
    return proc.stdout.decode('utf-8').splitlines()


def check_findings(data: AuditData) -> tuple[str, list[str]]:
    data.load_findings(); data.load_sampling_repos()
    errors: list[str] = []
    ids = set()
    cache: dict[tuple[Path, str, str], list[str]] = {}
    for item in data.findings or []:
        number = item['_line']; fid = item.get('id', '?')
        label = f'{data.findings_path.name}:{number} {fid}'
        missing = [field for field in FINDING_FIELDS if not nonblank(item.get(field))]
        if missing:
            errors.append(f'{label}: 缺字段或空值 {sorted(missing)}')
            continue
        if fid in ids:
            errors.append(f'{label}: id 重复')
        ids.add(fid)
        if item['disclosed'] not in {'yes', 'partial', 'no'}:
            errors.append(f'{label}: disclosed 非 yes/partial/no')
        if item['confidence'] not in {'high', 'medium', 'low'}:
            errors.append(f'{label}: confidence 非 high/medium/low')
        if not re.match(r'^(yes|partial|no)\s*[:：]', item['deploy_valid']):
            errors.append(f'{label}: deploy_valid 须有 yes/partial/no 和理由')
        if not re.fullmatch(r'[a-z][a-z0-9_]*', item['category']):
            errors.append(f'{label}: category 须为 snake_case')
        unit = data.unit(item['board'], item['repo'])
        if unit is None:
            errors.append(f'{label}: repo 未在对应榜单 repos.csv 中')
            continue
        repo = data.units[unit]
        if item['commit'] != repo['commit']:
            errors.append(f'{label}: commit 与 repos.csv 不同')
        try:
            link_repo, link_commit, path, start, end = parse_permalink(item['location'])
            if repo_url(link_repo) != repo_url(repo['repo_url']):
                errors.append(f'{label}: permalink 仓库与 repos.csv 不同')
            if link_commit != item['commit']:
                errors.append(f'{label}: permalink commit 与记录不同')
            repo_path = data.root / repo['repo_path']
            cache_key = (repo_path, link_commit, path)
            if cache_key not in cache:
                cache[cache_key] = git_lines(repo_path, link_commit, path)
            lines = cache[cache_key]
            if end > len(lines):
                errors.append(f'{label}: 行范围超出 Git 文件 {len(lines)} 行')
            else:
                exact = '\n'.join(lines[start - 1:end])
                if exact != item['snippet']:
                    errors.append(f'{label}: snippet 与固定 Git 行范围不完全一致')
        except (ValueError, UnicodeDecodeError, subprocess.TimeoutExpired) as exc:
            errors.append(f'{label}: {exc}')
        audit = data.out / 'audits' / f'{unit[0]}__{unit[1]}.md'
        if audit.is_file() and fid not in audit.read_text(encoding='utf-8'):
            errors.append(f'{label}: 对应审计页未列该发现 ID')
    detail = f'{len(data.findings or [])} 条发现、{len(cache)} 个固定 Git 文件、全部检查原文与论文披露字段'
    return detail, errors


def check_http_links(data: AuditData) -> tuple[int, list[str]]:
    """额外验证 GitHub 可达性；网络错误不影响离线验收退出码。"""
    data.load_findings()
    links = sorted({item.get('location', '') for item in data.findings or []
                    if isinstance(item.get('location'), str) and
                    item['location'].startswith('https://github.com/')})

    def open_link(link: str) -> tuple[str, str | None]:
        last_error = ''
        for method in ('HEAD', 'GET'):
            request = Request(link, method=method,
                              headers={'User-Agent': 'hack-audit-static-validator/1.0'})
            try:
                with urlopen(request, timeout=8) as response:
                    if method == 'GET':
                        response.read(1)
                    return link, None
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = f'{method}: {exc}'
        return link, last_error

    warnings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for link, error in pool.map(open_link, links):
            if error:
                warnings.append(f'{link}: {error}')
    return len(links), warnings


def taxonomy_categories(path: Path) -> set[str]:
    categories = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if re.match(r'^##\s*(?:提议|变更日志|变更记录|Changelog|Proposals)(?:\s|$)',
                    line, flags=re.I):
            break
        matched = CATEGORY_RE.match(line)
        if matched:
            categories.add(matched[1])
    return categories


def check_matrix(data: AuditData) -> tuple[str, list[str]]:
    data.load_matrix(); data.load_findings()
    errors: list[str] = []
    matrix = data.matrix or {}
    taxonomy = taxonomy_categories(data.out / 'taxonomy.md')
    if not taxonomy:
        errors.append('taxonomy.md 未找到最终类别标题；应在“提议/变更日志”前写 `### `category``')
    expected = {(*unit, category) for unit in data.units for category in taxonomy}
    actual = set(matrix)
    if expected != actual:
        errors.append(f'矩阵非完整 repo×类别：缺 {len(expected-actual)} 格 {sorted(expected-actual)[:8]}；多 {len(actual-expected)} 格 {sorted(actual-expected)[:8]}')
    for key, value in matrix.items():
        if value not in {'yes', 'no', 'NA'}:
            errors.append(f'{key}: present={value!r}，须为 yes/no/NA')
    finding_cells = set()
    for item in data.findings or []:
        unit = data.unit(item.get('board', ''), item.get('repo', ''))
        if unit and nonblank(item.get('category')):
            finding_cells.add((*unit, item['category']))
    uncovered = sorted(key for key, value in matrix.items()
                       if value == 'yes' and key not in finding_cells)
    if uncovered:
        errors.append(f'{len(uncovered)} 个 yes 无对应发现：{uncovered[:10]}')
    counts = Counter(matrix.values())
    detail = f'{len(data.units)} 单元×{len(taxonomy)} 最终类别，{len(matrix)} 格；{dict(sorted(counts.items()))}'
    return detail, errors


def review_sample(matrix: dict[tuple[str, str, str], str], seed: int):
    yes = sorted(key for key, value in matrix.items() if value == 'yes')
    no = sorted(key for key, value in matrix.items() if value == 'no')
    count_yes = math.ceil(len(yes) * .20)
    count_no = math.ceil(len(no) * .10)
    rng = random.Random(seed)
    return rng.sample(yes, count_yes) + rng.sample(no, count_no), count_yes, count_no


def check_review(data: AuditData) -> tuple[str, list[str]]:
    data.load_matrix(); data.load_index()
    errors: list[str] = []
    rows, _ = read_csv(data.out / 'review.csv',
                       {'board', 'repo', 'category', 'original', 'verdict',
                        'reviewer', 'selection_seed'})
    if not rows:
        errors.append('review.csv 无复核条目')
        return '0 条复核', errors
    seeds = {row['selection_seed'] for row in rows}
    if len(seeds) != 1:
        errors.append(f'selection_seed 不一致：{sorted(seeds)}')
    try:
        seed = int(next(iter(seeds)))
    except ValueError:
        errors.append('selection_seed 须为十进制整数')
        seed = 0
    expected_sample, need_yes, need_no = review_sample(data.matrix or {}, seed)
    observed = {}
    for number, row in enumerate(rows, 2):
        label = f'review.csv:{number}'
        unit = data.unit(row['board'], row['repo'])
        if unit is None:
            errors.append(f'{label}: 未知单元 {row["board"]}/{row["repo"]}')
            continue
        key = (*unit, row['category'])
        if key in observed:
            errors.append(f'{label}: 重复复核单元格 {key}')
        observed[key] = row
        original = (data.matrix or {}).get(key)
        if original not in {'yes', 'no'}:
            errors.append(f'{label}: 抽到非 yes/no 矩阵单元格 {key}')
        if row['original'] != original:
            errors.append(f'{label}: original={row["original"]!r} 与矩阵 {original!r} 不同')
        if row['verdict'] not in {'yes', 'no', 'NA'}:
            errors.append(f'{label}: verdict 须为 yes/no/NA')
        if not nonblank(row['reviewer']):
            errors.append(f'{label}: reviewer 为空')
        elif row['reviewer'].split('/')[-1] in data.stage1_owners:
            errors.append(f'{label}: reviewer 与阶段 1 负责人相同')
    if set(observed) != set(expected_sample):
        errors.append(f'随机复核集合与 seed={seed} 不符：缺 {sorted(set(expected_sample)-set(observed))[:8]}；多 {sorted(set(observed)-set(expected_sample))[:8]}')
    selected_counts = Counter((data.matrix or {}).get(key) for key in observed)
    if selected_counts['yes'] != need_yes or selected_counts['no'] != need_no:
        errors.append(f'抽样比例不足/超出：yes {selected_counts["yes"]}/{need_yes}，no {selected_counts["no"]}/{need_no}')
    comparable = [(key, row) for key, row in observed.items()
                  if row['original'] in {'yes', 'no'} and row['verdict'] in {'yes', 'no', 'NA'}]
    agreements = sum(row['original'] == row['verdict'] for _, row in comparable)
    if comparable and agreements * 10 < len(comparable) * 9:
        errors.append(f'一致率 {agreements}/{len(comparable)}={agreements/len(comparable):.1%}，低于 90%')
    disagreements = [key for key, row in comparable if row['original'] != row['verdict']]
    if disagreements:
        report_path = data.out / 'report.md'
        report = report_path.read_text(encoding='utf-8') if report_path.is_file() else ''
        for board, slug, category in disagreements:
            token = f'{board}__{slug}:{category}'
            if token not in report:
                errors.append(f'复核不一致 {token} 未逐条写入 report.md')
    rate = f'{agreements}/{len(comparable)}={agreements/len(comparable):.1%}' if comparable else '0/0'
    detail = f'{len(rows)} 条复核，yes {selected_counts["yes"]}/{need_yes}、no {selected_counts["no"]}/{need_no}；一致率 {rate}；{len(disagreements)} 条不一致'
    return detail, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help='工作目录，默认脚本所在工作区 /data/hack_audit')
    parser.add_argument('--findings', type=Path,
                        help='仅调试时覆写 findings.jsonl，最终验收请用默认路径')
    parser.add_argument('--checks', default=','.join(CHECKS),
                        help='逗号分隔的检查集合，默认全部五项')
    parser.add_argument('--draw-review', type=int, metavar='SEED',
                        help='按约定随机种子把待复核 review.csv 模板打印到 stdout')
    parser.add_argument('--check-http', action='store_true',
                        help='额外以 HEAD/GET 检查 GitHub 链接可访问；网络错误仅 WARN')
    args = parser.parse_args()
    requested = [name.strip() for name in args.checks.split(',') if name.strip()]
    unknown = set(requested) - set(CHECKS)
    if unknown:
        parser.error(f'未知检查：{sorted(unknown)}')
    root = args.root.resolve()
    findings = args.findings
    if findings is not None and not findings.is_absolute():
        findings = root / findings
    data = AuditData(root, findings)
    if args.draw_review is not None:
        try:
            data.load_matrix()
            sample, _, _ = review_sample(data.matrix or {}, args.draw_review)
        except (OSError, ValueError) as exc:
            print(f'FAIL draw-review: {exc}', file=sys.stderr)
            return 1
        writer = csv.writer(sys.stdout, lineterminator='\n')
        writer.writerow(['board', 'repo', 'category', 'original', 'verdict',
                         'reviewer', 'selection_seed'])
        for board, slug, category in sample:
            writer.writerow([board, data.units[(board, slug)]['repo_url'], category,
                             data.matrix[(board, slug, category)], '', '', args.draw_review])
        return 0
    functions = {
        'sampling': check_sampling, 'index': check_index,
        'findings': check_findings, 'matrix': check_matrix,
        'review': check_review,
    }
    pass_count = fail_count = skip_count = 0
    for name in CHECKS:
        if name not in requested:
            print(f'SKIP {name}: 未列入 --checks')
            skip_count += 1
            continue
        try:
            detail, errors = functions[name](data)
        except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired) as exc:
            detail, errors = '', [str(exc)]
        if errors:
            fail_count += 1
            print(f'FAIL {name}: {detail}; {len(errors)} 项问题')
            for error in errors:
                print(f'  - {error}')
        else:
            pass_count += 1
            print(f'PASS {name}: {detail}')
    print(f'SUMMARY PASS={pass_count} FAIL={fail_count} SKIP={skip_count}')
    if 'findings' in requested:
        if args.check_http:
            try:
                count, warnings = check_http_links(data)
                print(f'HTTP 检查 {count} 个唯一链接；WARN={len(warnings)}')
                for warning in warnings:
                    print(f'  - WARN {warning}')
            except (OSError, ValueError) as exc:
                print(f'WARN HTTP 检查无法完成: {exc}')
        else:
            print('NOTE: permalink 已按固定 Git 对象核验；可用 --check-http 验证网络可达性。')
    if 'review' in requested:
        print('NOTE: 代理是否继承上下文无法从静态文件证明，需由主代理确认。')
    return 1 if fail_count else 0


if __name__ == '__main__':
    raise SystemExit(main())
