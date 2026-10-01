import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/night_queue_4/archive", "scripts",)]
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cx_gk_batch as batch
import sch_table as sch


class BatchRefillTest(unittest.TestCase):
    def test_new_block_avoids_registered_plus_minus_120_and_bound_ports(self):
        row = dict(lane='old', gpus='1', workers='2', idx0='120', idx_span='10', cpus='0', status='pilot', go='-')
        first = batch.free_block([row], {2000 + 50 * 10})
        self.assertGreaterEqual(first, 11)
        new = dict(row, lane='new', gpus='0', idx0=str(first), idx_span='8')
        self.assertEqual(sch.conflicts([row, new]), [])

    def test_seed_zero_reuses_completed_pilot_routes_and_other_seeds_do_not(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(batch, 'G', Path(tmp)):
            source = Path(tmp) / 'cx_pilot_20260927_k/k0_unseen/s0'
            (source / 'done').mkdir(parents=True)
            (source / 'requested.json').write_text(json.dumps([str(i) for i in range(220)]))
            (source / 'done/0.json').write_text('{"status":"finished","attempt":1}')
            (source / 'attempts/0/1').mkdir(parents=True)
            (source / 'attempts/0/1/route_result.json').write_text('{}')
            zero = batch.prepare('k0_unseen', 0)
            one = batch.prepare('k0_unseen', 1)
            self.assertTrue((zero / 'done/0.json').exists())
            self.assertFalse((one / 'done/0.json').exists())
            self.assertEqual(len(json.loads((one / 'requested.json').read_text())), 220)
            with self.assertRaises(RuntimeError):
                batch.prepare('k0_unseen', 0)

    def test_pass_marker_without_full_ten_route_evidence_is_not_a_gate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(batch, 'G', Path(tmp)/'g'), patch.object(batch, 'DATA', Path(tmp)), patch.object(batch, 'OUT', Path(tmp)/'out'):
            ready = Path(tmp)/'runs/nq4/k/READY';ready.parent.mkdir(parents=True);ready.touch()
            pilot = Path(tmp)/'g/cx_pilot_20260927_pilot/k0_unseen.k';pilot.mkdir(parents=True)
            (pilot/'PASS').touch()
            self.assertFalse(batch.gate('k0_unseen'))


if __name__ == '__main__':
    unittest.main()
