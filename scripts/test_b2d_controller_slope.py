"""Offline metric/selection fixtures for the optional CARLA slope helper."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from b2d_controller_slope import _select_candidates, hold_summary, run


def hold_rows(count=101, signed_speed=0.):
    return [dict(sim_time=100. + tick * .05, rear_xyz=[0., 0., 0.],
                 signed_speed_mps=signed_speed, speed_mps=abs(signed_speed), pitch_deg=4.)
            for tick in range(count)]


def waypoint(pitch, x=0., width=3.5, junction=False, lane_type='Driving'):
    return SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=x, y=0., z=0.),
                                                    rotation=SimpleNamespace(pitch=pitch)),
                           lane_width=width, is_junction=junction, lane_type=lane_type,
                           road_id=1, lane_id=1, s=x)


class SlopeMetricTests(unittest.TestCase):
    def test_complete_five_seconds_and_grade(self):
        result = hold_summary(hold_rows())
        self.assertTrue(result['gate_pass'])
        self.assertEqual(result['duration_s'], 5.)
        self.assertAlmostEqual(result['grade_percent'], 6.992681194, places=7)
        self.assertFalse(hold_summary(hold_rows(100))['gates']['duration'])

    def test_reverse_is_preserved_and_drift_fails(self):
        rows = hold_rows(signed_speed=-.04)
        for tick, row in enumerate(rows):
            row['rear_xyz'][0] = -.002 * tick
        result = hold_summary(rows)
        self.assertEqual(result['min_signed_speed_mps'], -.04)
        self.assertEqual(result['reverse_samples'], 101)
        self.assertAlmostEqual(result['displacement_m'], .2)
        self.assertFalse(result['gate_pass'])
        self.assertTrue(result['gates']['speed'])
        self.assertFalse(result['gates']['displacement'])

    def test_return_motion_and_high_speed_are_not_hidden(self):
        rows = hold_rows()
        rows[20]['rear_xyz'][2] = .12
        rows[20]['speed_mps'] = .2
        result = hold_summary(rows)
        self.assertEqual(result['endpoint_displacement_m'], 0.)
        self.assertEqual(result['displacement_m'], .12)
        self.assertFalse(result['gates']['speed'])
        self.assertFalse(result['gates']['displacement'])
        rows = hold_rows()
        rows[10]['pitch_deg'] = 2.9
        self.assertFalse(hold_summary(rows)['gates']['measured_slope'])

    def test_static_drivable_selection(self):
        candidates = [waypoint(4.), waypoint(5., x=5.), waypoint(-6., x=40.),
                      waypoint(7., x=70., junction=True), waypoint(8., x=100., width=2.),
                      waypoint(9., x=130., lane_type='Sidewalk'), waypoint(2., x=200.)]
        world_map = SimpleNamespace(generate_waypoints=lambda spacing: candidates)
        selected = _select_candidates(world_map)
        self.assertEqual([w.transform.rotation.pitch for w in selected], [-6., 5.])

    def test_no_slope_is_explicitly_untested_without_server_ownership(self):
        world_map = SimpleNamespace(name='TownFake', generate_waypoints=lambda spacing: [])
        world = SimpleNamespace(get_map=lambda: world_map)
        # There are deliberately no start/stop/load/tick methods on this client.
        # An untested result must not require or assume server process ownership.
        client = SimpleNamespace(get_world=lambda: world, get_client_version=lambda: '0.9.15',
                                 get_server_version=lambda: '0.9.15')
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'carla': SimpleNamespace()}):
                result = run(client, {'slope': {'allow_town12': False}}, directory)
            self.assertEqual(result['status'], 'untested')
            self.assertEqual(result['reason'], 'no_suitable_slope')
            self.assertFalse(result['gate_pass'])
            self.assertFalse(result['server_owned'])
            saved = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(saved['attempts'], [])
            self.assertTrue((Path(directory) / 'slope.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
