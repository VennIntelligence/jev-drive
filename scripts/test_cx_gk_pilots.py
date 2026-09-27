import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cx_gk_pilots as pilots
import cx_owned_process as owned


class PilotSafetyTest(unittest.TestCase):
    def test_pending_B_workers_are_charged_before_admission(self):
        probe = dict(pids=14900, cores_used=60, gpus=[dict(gpu=1, used_gb=0, carla=0)])
        self.assertEqual(pilots.admission(probe, 0, 2, [], 10), [])
        self.assertTrue(pilots.admission(probe, 2, 2, [], 10))
        self.assertTrue(pilots.admission(probe, 0, 2, ['overlap'], 10))
        self.assertTrue(pilots.admission(probe, 0, 2, [], 2))

    def test_reused_root_pid_does_not_adopt_unrelated_children(self):
        def row(pid, ticks, parent=1):
            return dict(pid=pid, start_ticks=ticks, ppid=parent, pgid=pid, sid=pid)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'owned.json'
            with patch.object(owned, 'process_snapshot', return_value={10: row(10, 'old'), 11: row(11, 'child', 10)}):
                owned.refresh(path, 10)
            with patch.object(owned, 'process_snapshot', return_value={10: row(10, 'new'), 12: row(12, 'unrelated', 10)}):
                record, _ = owned.refresh(path)
            self.assertEqual({m['pid'] for m in record['members']}, {10, 11})

    def test_partial_and_failed_pilots_do_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilots, 'G', Path(tmp)):
            d = Path(tmp) / f'{pilots.ARMS}_pilot/k0_unseen.k'
            d.mkdir(parents=True)
            self.assertEqual(pilots.result('k0_unseen', 'k'), 'TECHNICAL_WAIT')
            (d / 'PASS').touch()
            (d / 'stage1.json').write_text(json.dumps({'pass': True}))
            (d / 'stage2.json').write_text(json.dumps({'pass': False}))
            self.assertEqual(pilots.result('k0_unseen', 'k'), 'INVALID_PASS_EVIDENCE')

    def test_two_route_smoke_cannot_skip_ten_route_pilot(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilots, 'G', Path(tmp)):
            d = Path(tmp) / 'smoke_pilot/blue.ghost'; d.mkdir(parents=True)
            (d / 'PASS').touch()
            for i in (1, 2):
                (d / f'stage{i}.json').write_text(json.dumps({'pass': True, 'facts': {'routes': i}}))
            self.assertFalse(pilots.historical_pilot('blue', 'ghost'))


if __name__ == '__main__':
    unittest.main()
