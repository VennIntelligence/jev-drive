import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/night_queue_4/archive", "scripts",)]
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cx_gk_batch_v2 as batch
import sch_table as sch


class RefillRegressionTest(unittest.TestCase):
    def test_real_7437_pid_snapshot_can_fill_three_idle_cards(self):
        probe = dict(pids=7437, cores_used=35)
        device = dict(used_gb=.03)
        self.assertEqual([batch.choose_workers(probe, pending, device) for pending in (0, 6, 12)], [6, 6, 6])
        self.assertEqual(batch.choose_workers(probe, 18, device), 3)
        self.assertEqual(batch.choose_workers(dict(pids=15900, cores_used=20), 0, device), 0)

    def test_idle_tail_workers_are_not_pending_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = dict(runtime=str(root/'runtime'), out=str(root/'out'), workers=6)
            self.assertEqual(batch.pending_workers(job, 0), 6)
            (root/'runtime').mkdir();(root/'runtime/owned-runners').touch()
            (root/'out/claims').mkdir(parents=True);(root/'out/claims/route.lock').touch()
            self.assertEqual(batch.pending_workers(job, 1), 0)
            self.assertEqual(batch.pending_workers(job, 0), 1)

    def test_complete_B_claim_released_but_live_writer_protected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(batch, 'DATA', Path(tmp)), patch.object(batch, 'event'):
            root = Path(tmp);base=root/'runs/nq3/b';(base/'results').mkdir(parents=True)
            (base/'DONE').write_text('complete');(base/'QUEUE').write_text('\n')
            for name in ('arms.csv','per_route.csv','summary.md'):(base/'results'/name).write_text('result')
            table=root/'runs/sched/table.tsv';table.parent.mkdir(parents=True)
            row=dict(lane='nq3-b',gpus='2',workers='6',idx0='60',idx_span='30',cpus='0-15',status='batch',go='-')
            with patch.object(sch,'TABLE',table):
                sch.save([row])
                batch.release_completed_b({42:dict(argv=['python','--out',str(base/'arms/simlingo/s0')],ppid=1)})
                self.assertEqual(sch.load()[0]['gpus'],'2')
                batch.release_completed_b({43:dict(argv=['bash','experiments/night_queue_3/archive/nq3_b.sh','chain'],ppid=1)})
                self.assertEqual(sch.load()[0]['gpus'],'-')
                self.assertEqual(sch.load()[0]['idx0'],'-')

    def test_no_port_wait_is_durable_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(batch,'OUT',Path(tmp)), patch.object(batch,'event') as event:
            batch._WAIT_REASONS.clear()
            batch.wait_reason('2','NO_LEGAL_INDEX_BLOCK')
            batch.wait_reason('2','NO_LEGAL_INDEX_BLOCK')
            self.assertEqual(event.call_count,1)
            self.assertEqual(json.loads((Path(tmp)/'admission.json').read_text())['2']['reason'],'NO_LEGAL_INDEX_BLOCK')


if __name__ == '__main__':
    unittest.main()
