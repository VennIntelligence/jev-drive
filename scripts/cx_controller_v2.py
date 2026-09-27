#!/usr/bin/env python3
"""Successor for the final-arm blank QUEUE case; preserve the original controller.

The original B shell writes a newline for an empty array. Blank records mean no
pending arms, whereas nonempty malformed records remain errors. No live chain,
scientific gate, resource grant, or worker ownership policy changes here.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cx_controller as original


class Controller(original.Controller):
    def b_ready_queue(self):
        if not self.config.get('b_handoff'):
            return []
        base = self.data / 'runs/nq3/b'
        queue = base / 'QUEUE'
        if not queue.exists():
            return []
        approved = set((base / 'APPROVED').read_text().splitlines()) if (base / 'APPROVED').exists() else set()
        skipped = set((base / 'SKIP').read_text().splitlines()) if (base / 'SKIP').exists() else set()
        ready = []
        for line in queue.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError('malformed original B queue')
            arm, seed = fields[:2]
            if f'{arm} {seed}' in skipped or (base / f'arms/{arm}/s{seed}/DONE').exists():
                continue
            verdict = self.data / f'runs/sched/pilot/b/arms/{arm}/s0/verdict.json'
            if arm in approved or (verdict.exists() and json.loads(verdict.read_text()).get('verdict') == 'PASS'):
                ready.append(f'B:{arm}:{seed}')
        return ready


def supervise():
    out = Path(os.environ['DATA_DIR']) / 'runs/sched/controller'
    lock = original.claim(out / 'supervisor.lock')
    backoff = 5
    with lock, (out / 'supervisor.log').open('a', buffering=1) as log:
        while True:
            start = time.monotonic()
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'run'],
                                    stdout=log, stderr=subprocess.STDOUT)
            if time.monotonic() - start > 300:
                backoff = 5
            log.write(f'{time.time()} controller exited rc={result.returncode}; restart in {backoff}s\n')
            time.sleep(backoff)
            backoff = min(60, backoff * 2)


if __name__ == '__main__':
    if sys.argv[1:] == ['supervise']:
        supervise()
    else:
        original.Controller = Controller
        raise SystemExit(original.main())
