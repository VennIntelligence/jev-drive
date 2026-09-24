"""Validate the second-round handoff without running any model or evaluator."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urldefrag

import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out2"
REQUIRED = {
    "w1_ablation_ledger.csv": [
        "method", "board", "split", "metric", "component", "baseline_config",
        "variant_config", "baseline_score", "variant_score", "delta",
        "same_weights_or_retrained", "seeds_or_runs", "n_samples", "source",
        "hack_finding_id",
    ],
    "w2_cross_board.csv": [
        "method", "board", "protocol/split", "metric", "score", "source",
        "is_author_reported", "same_checkpoint",
    ],
    "w3_issues.csv": [
        "repo", "issue_url", "issue_date", "category", "user_reported_numbers",
        "paper_numbers", "paper_url", "author_replied", "author_reply_summary",
        "author_quote", "author_reply_url", "resolved", "issue_state", "case_summary",
        "related_methods", "paper_locator",
    ],
    "w4_attack_surface.csv": [
        "board", "attack_surface", "code_evidence", "observed_in_round1", "finding_id",
    ],
    "w5_paper_only.csv": [
        "id", "board", "method", "category", "design_disclosed", "reported_score_effect",
        "source", "disclosure_scope", "code_status", "code_status_date", "code_status_source",
    ],
}
URL_FIELDS = {
    "w1_ablation_ledger.csv": ["source"],
    "w2_cross_board.csv": ["source"],
    "w3_issues.csv": ["issue_url", "paper_url", "author_reply_url"],
    "w4_attack_surface.csv": ["code_evidence"],
    "w5_paper_only.csv": ["source", "code_status_source"],
}
URL_RE = re.compile(r"https?://[^\s;,)]+")
PAGE_RE = re.compile(r"PDF p\.(\d+)")
PDF_RE = re.compile(r"(?:papers/)?([A-Za-z0-9_.+-]+\.pdf)")


def read_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pdf_pages(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True)
    return int(re.search(r"^Pages:\s+(\d+)", info.stdout, re.M).group(1))


def check_url(url: str) -> tuple[str, str]:
    last_status = "untried"
    for attempt in range(3):
        try:
            response = requests.head(url, allow_redirects=True, timeout=(7, 18))
            if response.status_code in {403, 405, 429}:
                response.close()
                response = requests.get(url, allow_redirects=True, timeout=(7, 18), stream=True)
            last_status = str(response.status_code)
            response.close()
            if last_status == "200" or last_status in {"401", "404"}:
                return url, last_status
        except requests.RequestException as exc:
            last_status = type(exc).__name__
        if attempt < 2:
            time.sleep(0.5 * (attempt + 1))
    return url, last_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-url", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    counts = {}
    urls: set[str] = set()
    page_checks = 0
    pdf_cache: dict[Path, int] = {}

    for name, required in REQUIRED.items():
        rows = read_rows(name)
        counts[name] = len(rows)
        for number, row in enumerate(rows, 2):
            for key in required:
                if key not in row:
                    errors.append(f"{name}:{number}: missing column {key}")
                elif not (row[key] or "").strip():
                    errors.append(f"{name}:{number}: blank {key}")
            for key in URL_FIELDS[name]:
                value = row.get(key)
                if value:
                    urls.update(URL_RE.findall(value))
            if name == "w4_attack_surface.csv":
                if not re.search(r"github\.com/.+/blob/[a-f0-9]{40}/.+#L\d+", row["code_evidence"]):
                    errors.append(f"{name}:{number}: code_evidence lacks fixed commit and line")
            if name in {"w1_ablation_ledger.csv", "w2_cross_board.csv"}:
                source = row["source"]
                if not re.search(r"(?:Table|Figure|Fig\.|Section|Appendix)\s*[^,)]*", source):
                    errors.append(f"{name}:{number}: source lacks table/figure/section label")
                page = PAGE_RE.search(source)
                if not page:
                    errors.append(f"{name}:{number}: source lacks PDF page")
                    continue
                pdf_match = PDF_RE.search(source)
                if pdf_match:
                    pdf = ROOT / "papers" / pdf_match.group(1)
                    if not pdf.is_file():
                        errors.append(f"{name}:{number}: missing {pdf}")
                    else:
                        if pdf not in pdf_cache:
                            pdf_cache[pdf] = pdf_pages(pdf)
                        if int(page.group(1)) > pdf_cache[pdf]:
                            errors.append(f"{name}:{number}: page exceeds {pdf.name} ({pdf_cache[pdf]})")
                        page_checks += 1

    url_results: list[tuple[str, str]] = []
    url_resources = {urldefrag(url)[0] for url in urls}
    if not args.skip_url:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(check_url, url) for url in sorted(url_resources)]
            for future in as_completed(futures):
                url_results.append(future.result())
        for url, status in url_results:
            if status != "200":
                errors.append(f"URL {status}: {url}")

    lines = [
        "# 结构与链接验收",
        "",
        "运行：`python3 out2/sources/validate.py`。脚本只读 CSV、PDF 元数据与公开 URL；不运行模型或评测。",
        "",
        "| 文件 | 数据行 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in counts.items())
    lines.extend([
        "",
        f"- 必填字段与来源页码检查：{page_checks} 条本地 PDF 页面，{len(errors)} 项总错误（含下述 URL）。",
        f"- URL 检查：{len(urls)} 条含页锚引用，{len(url_results)} 个不同 HTTP 资源，" + (
            "已跳过" if args.skip_url else f"{sum(status == '200' for _, status in url_results)} 个 HTTP 200"
        ) + "。",
        "",
        "## 错误",
        "",
    ])
    lines.extend(f"- {error}" for error in errors)
    if not errors:
        lines.append("无。")
    (OUT / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rows={counts} pdf_page_checks={page_checks} urls={len(url_results)} errors={len(errors)}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
