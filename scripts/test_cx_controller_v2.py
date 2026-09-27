"""Regression tests for the real final-arm queue and durable state refresh."""
import json
from unittest.mock import patch

import cx_controller_v2 as v2
from test_cx_controller import ControllerTest


class FinalArmTest(ControllerTest):
    def controller(self, jobs, **extra):
        return v2.Controller(self.data, dict(jobs=jobs, deadline='2099-01-01T00:00:00+00:00', **extra))

    def test_final_arm_blank_queue_refreshes_status_without_relaunch(self):
        c = self.controller([dict(id='B', process_match=[['job']])], b_handoff=dict(gpus=[0], workers=6))
        self.write('runs/nq3/b/QUEUE', '\n')
        with patch.object(c, 'b_grant'), patch.object(c, 'b_recover') as recovery:
            c.tick({42: self.proc()}, self.gpu([42]))
        self.assertEqual(c.state['queue']['status'], 'NO_READY_WORK')
        self.assertEqual(c.state['jobs']['B']['status'], 'RUNNING')
        self.assertEqual(json.loads((c.out / 'heartbeat.json').read_text())['node'], 'ONLINE')
        self.assertIn('NO_READY_WORK', (c.out / 'STATUS.md').read_text())
        recovery.assert_called_once()

    def test_blank_records_preserve_pass_skip_done_gates(self):
        c = self.controller([], b_handoff=dict(gpus=[0], workers=6))
        self.write('runs/nq3/b/QUEUE', '\nblue 0 all 3\n \nsimlingo 0 all 8\ncl1 0 all 1\ncl3 0 all 2\n')
        self.write('runs/nq3/b/APPROVED', 'blue\nsimlingo\ncl1\n')
        self.write('runs/nq3/b/SKIP', 'blue 0\n')
        self.write('runs/nq3/b/arms/cl1/s0/DONE')
        self.assertEqual(c.b_ready_queue(), ['B:simlingo:0'])

    def test_nonempty_malformed_queue_still_fails(self):
        c = self.controller([], b_handoff=dict(gpus=[0], workers=6))
        self.write('runs/nq3/b/QUEUE', '\nsimlingo 0\n')
        with self.assertRaisesRegex(ValueError, 'malformed'):
            c.b_ready_queue()


if __name__ == '__main__':
    import unittest
    unittest.main()
