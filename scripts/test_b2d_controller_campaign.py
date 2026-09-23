"""Configuration isolation tests; no git, CARLA or external processes."""
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import Mock, patch
import b2d_controller_campaign as campaign
import json
from pathlib import Path
import tempfile
import unittest
from b2d_controller_campaign import resolve_configs, archived_configs


class CampaignConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.default = self.write('default.json', {'longitudinal_mode': 'vendor'})
        self.pi = self.write('pi.json', {'longitudinal_mode': 'pi', 'pi_kp': .5})

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def snapshot(self, resolved):
        archive = self.root / 'archive'
        archive.mkdir()
        rows = []
        for index, source in enumerate(resolved['inputs']):
            target = archive / ('%d.json' % index)
            target.write_bytes(source.read_bytes())
            rows.append(dict(original=str(source), archived=target.name,
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        return {'inputs': rows}, archive

    def test_fallback_and_override_are_archived_not_live_inputs(self):
        mapping = self.write('mapping.json', {'pursuit': str(self.pi)})
        resolved = resolve_configs(self.default, 'carla,tcp,pursuit', mapping)
        manifest, archive = self.snapshot(resolved)
        self.pi.write_text('{"longitudinal_mode":"vendor"}')
        self.default.unlink()
        mapping.unlink()
        bindings, default = archived_configs(resolved, manifest, archive)
        self.assertEqual(bindings['carla']['archived_path'], default['archived_path'])
        self.assertTrue(bindings['tcp']['fallback'])
        self.assertFalse(bindings['pursuit']['fallback'])
        self.assertEqual(json.loads(Path(bindings['pursuit']['archived_path']).read_text())['longitudinal_mode'], 'pi')
        self.assertEqual(json.loads(Path(default['archived_path']).read_text())['longitudinal_mode'], 'vendor')

    def test_no_mapping_preserves_common_default(self):
        resolved = resolve_configs(self.default, 'carla,pursuit')
        self.assertEqual(set(resolved['selected'].values()), {self.default})
        self.assertEqual(resolved['inputs'], [self.default])

    def test_unknown_missing_relative_and_nonobject_rejected(self):
        for mapping in ({'unknown': str(self.pi)}, {'pursuit': 'relative.json'},
                        {'pursuit': str(self.root / 'missing.json')}, {'pursuit': None}):
            with self.subTest(mapping=mapping), self.assertRaises((ValueError, OSError)):
                resolve_configs(self.default, 'pursuit', self.write('bad.json', mapping))
        for presets in ('', 'typo', 'carla,carla'):
            with self.assertRaises(ValueError):
                resolve_configs(self.default, presets)
        with self.assertRaises(ValueError):
            resolve_configs(self.write('array.json', []), 'pursuit')

    def test_main_uses_archived_bytes_for_every_group_on_one_server(self):
        mapping = self.write('mapping.json', {'pursuit': str(self.pi)})
        routes = self.root / 'routes.xml'; routes.write_text('<routes/>')
        out = self.root / 'run'
        seen = []
        server = SimpleNamespace(index=70, proc=SimpleNamespace(pid=123))
        server.start = lambda: None
        server.stop = lambda: None
        server.alive = lambda: True

        def fake_snapshot(directory, inputs):
            directory.mkdir(parents=True)
            records = []
            for index, source in enumerate(inputs):
                target = directory / ('%d-input' % index)
                target.write_bytes(source.read_bytes())
                records.append(dict(original=str(source), archived=target.name,
                                    sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            # Mutating all original configs before any runner must have no effect.
            self.default.write_text('{}'); self.pi.write_text('{}'); mapping.unlink()
            return {'inputs': records}

        def fake_runner(args, selected, servers):
            self.assertEqual(servers, [server])
            path = Path(args.controller_config)
            self.assertTrue(str(path).startswith(str(out / 'provenance')))
            seen.append((args.controller_preset, json.loads(path.read_text())))
            Path(args.out).mkdir(parents=True)
            return SimpleNamespace(learn_maps=lambda _: None, run=lambda: 0,
                                   events=io.StringIO(), stop_flag=False)

        def fake_args(values):
            def option(name):
                return values[values.index(name) + 1]
            return SimpleNamespace(controller_config=option('--controller-config'),
                                   controller_preset=option('--controller-preset'),
                                   out=option('--out'), routes=option('--routes'),
                                   towns='all', route_ids=None, limit=None)

        constructor = Mock(return_value=server)
        runtime = (fake_runner, constructor, fake_args, Mock(return_value=[]))
        argv = ['campaign', '--routes', str(routes), '--out', str(out),
                '--controller-config', str(self.default), '--preset-configs', str(mapping),
                '--seeds', '0,1']
        with patch.object(campaign.sys, 'argv', argv), patch.object(campaign, 'snapshot', fake_snapshot), \
             patch.object(campaign, 'load_runtime', return_value=runtime), \
             patch.object(campaign, 'assert_sources_unchanged'), \
             patch.object(campaign.signal, 'signal'), \
             patch.object(campaign.subprocess, 'check_output', return_value='mock'), \
             patch.object(campaign.subprocess, 'run'), patch.object(campaign.sys, 'stdout', io.StringIO()):
            self.assertEqual(campaign.main(), 0)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual([row[1]['longitudinal_mode'] for row in seen],
                         ['vendor', 'vendor', 'pi'] * 2)
        manifest = json.loads((out / 'manifest.json').read_text())
        groups = json.loads((out / 'groups.json').read_text())
        self.assertEqual(len(groups), 6)
        for row in groups:
            self.assertEqual(row['controller_config'], manifest['preset_controller_configs'][row['preset']])

    def test_changed_during_snapshot_and_corrupt_archive_rejected(self):
        resolved = resolve_configs(self.default, 'pursuit')
        self.default.write_text('{"changed": true}')
        manifest, archive = self.snapshot(resolved)
        with self.assertRaisesRegex(ValueError, 'changed while snapshotting'):
            archived_configs(resolved, manifest, archive)
        resolved = resolve_configs(self.default, 'pursuit')
        (archive / manifest['inputs'][0]['archived']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed while snapshotting'):
            archived_configs(resolved, manifest, archive)


if __name__ == '__main__':
    unittest.main()
