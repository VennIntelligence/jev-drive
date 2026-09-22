"""Input isolation and case ordering for the diagnostic matrix; no simulator."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from b2d_controller_validate import archive_matrix, load_matrix


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = self.write('a.json', {'adapter': {'rear_axle_offset_m': -1.4,
                                                  'route_stop_deceleration': 2.}})
        self.b = self.write('b.json', {'adapter': {'rear_axle_offset_m': -1.4},
                                     'rear_axle_offset_m': -1.5, 'route_stop_deceleration': 3.})

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return str(path)

    def variants(self):
        return self.write('variants.json', {
            'max': {'controller_config': self.a, 'presets': ['carla', 'tcp', 'pursuit']},
            'additive': {'controller_config': self.b, 'presets': ['pursuit']}})

    def test_order_and_physics_are_per_case(self):
        cases, cruises = load_matrix(None, '', self.variants(),
                                      self.write('speeds.json', {'17563': 6}), 8.)
        self.assertEqual([(c['variant'], c['preset']) for c in cases],
                         [('max', 'carla'), ('max', 'tcp'), ('max', 'pursuit'), ('additive', 'pursuit')])
        self.assertEqual(cases[0]['rear_axle_offset_m'], -1.4)
        self.assertEqual(cases[3]['rear_axle_offset_m'], -1.5)
        self.assertEqual(cases[3]['stop_deceleration_mps2'], 3.)
        self.assertEqual(cruises, {'17563': 6, 'default': 8.})

    def test_legacy_and_explicit_default(self):
        cases, cruises = load_matrix(self.a, 'tcp,pursuit', cruise_mps=7.)
        self.assertEqual([(c['variant'], c['preset']) for c in cases],
                         [('default', 'tcp'), ('default', 'pursuit')])
        self.assertEqual(cruises['default'], 7.)
        _, cruises = load_matrix(self.a, 'carla', route_cruises_path=self.write('speeds.json', {'default': 6}))
        self.assertEqual(cruises['default'], 6)

    def test_archive_runs_exact_loaded_bytes_after_source_changes(self):
        variant_path = self.variants()
        cases, cruises = load_matrix(None, '', variant_path)
        loaded = Path(self.a).read_bytes()
        Path(self.a).write_text('{}')
        routes = self.root / 'routes.xml'
        routes.write_text('<routes/>')
        manifest = archive_matrix(self.root / 'out', routes, cases, cruises, variant_path)
        self.assertEqual(Path(cases[0]['archived_controller_config']).read_bytes(), loaded)
        self.assertEqual(cases[0]['controller_config_sha256'], hashlib.sha256(loaded).hexdigest())
        self.assertEqual(manifest['ordering'], 'route_then_variant_then_preset')
        self.assertEqual(len(manifest['cases']), 4)
        json.dumps(manifest, allow_nan=False)

    def test_invalid_matrix_inputs_fail_before_server_import(self):
        bad = [
            {},
            {'../escape': {'controller_config': self.a, 'presets': ['pursuit']}},
            {'x': {'controller_config': 'relative.json', 'presets': ['pursuit']}},
            {'x': {'controller_config': self.a, 'presets': ['pursuit', 'pursuit']}},
            {'x': {'controller_config': self.a, 'presets': ['unknown']}},
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_matrix(None, '', self.write('bad.json', value))
        for value in [0, -1, float('nan'), True, '6']:
            with self.subTest(speed=value), self.assertRaises(ValueError):
                load_matrix(self.a, 'pursuit', route_cruises_path=self.write('speeds.json', {'17563': value}))


if __name__ == '__main__':
    unittest.main()
