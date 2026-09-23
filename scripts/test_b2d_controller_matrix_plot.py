"""Prevent partial-run and selection bias in frozen matrix figures."""
import json
from pathlib import Path
import tempfile
import unittest

from b2d_controller_matrix_plot import ROUTES, require_complete, stats


class FigureSelectionTests(unittest.TestCase):
    def test_partial_duplicate_and_failed_campaign_are_not_final_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(route_id=r, variant=v, preset=p, gate_pass=False)
                    for r in ROUTES for v, p in [('baseline', 'carla'), ('baseline', 'tcp'),
                                                ('baseline', 'pursuit'), ('max', 'pursuit')]]
            (root/'events.jsonl').write_text(json.dumps(dict(kind='end', status='completed', cases=20)))
            for invalid in [rows[:-1], rows[:-1]+[rows[0]]]:
                (root/'summary.json').write_text(json.dumps(invalid))
                with self.assertRaises(RuntimeError):
                    require_complete(root)
            (root/'summary.json').write_text(json.dumps(rows))
            # Gate failures must remain included: complete means all cases ran.
            self.assertEqual(len(require_complete(root)), 20)
            (root/'events.jsonl').write_text(json.dumps(dict(kind='end', status='failed', cases=20)))
            with self.assertRaises(RuntimeError):
                require_complete(root)

    def test_signed_errors_and_missing_values(self):
        self.assertAlmostEqual(stats([-3., 4., None, float('nan')])['rms'], 5./2**.5)
        self.assertIsNone(stats([])['rms'])


if __name__ == '__main__':
    unittest.main()
