"""Render the per-board top-10 CSVs of todos/2026-09-26-top10-intersection/boards as compact markdown tables."""
import csv, re, sys
from pathlib import Path

BOARDS = ['nuscenes', 'wod_e2e', 'navsim_v1', 'navsim_v2_navhard', 'navsim_v2_navtest',
          'bench2drive', 'longest6', 'carla_lb2', 'nuplan_val14', 'nuplan_test14hard', 'interplan', 'hugsim']
ROOT = Path(__file__).resolve().parents[1] / 'todos/2026-09-26-top10-intersection/boards'


def cell(s, n=70):
    s = re.sub(r'\s+', ' ', s or '').replace('|', '/').strip()
    return s if len(s) <= n else s[:n - 1] + '…'


def avail(s):
    """Shorten a code/weights field to its verdict plus first URL."""
    s = s or ''
    verdict = (re.match(r'\s*(yes|partial|no|n/a|not checked|not found|see repo)', s, re.I) or [None, '?'])[1]
    url = re.search(r'https?://\S+', s)
    return f'{verdict.lower()} {url.group(0).rstrip(").,;")}' if url else verdict.lower()


def main(boards):
    for b in boards:
        rows = list(csv.DictReader(open(ROOT / f'{b}.csv')))
        print(f'\n**{b}** ({rows[0]["primary_metric"]})\n')
        print('| # | method | family | score | secondary | date | source | code | weights | inputs | hack mechanisms | new |')
        print('|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|')
        for r in rows:
            print('| ' + ' | '.join([cell(r['rank'], 12), cell(r['method'], 40), cell(r['family'], 30), cell(r['score'], 24),
                                     cell(r['secondary'], 40), cell(r['date'], 8), cell(r['source'], 80), cell(avail(r['code']), 70),
                                     cell(avail(r['weights']), 70), cell(r['inputs'], 40), cell(r['hack_mechanisms'], 50),
                                     cell(r['newly_added'], 4)]) + ' |')


if __name__ == '__main__':
    main(sys.argv[1:] or BOARDS)
