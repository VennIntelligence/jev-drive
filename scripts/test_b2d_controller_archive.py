import json
from pathlib import Path

import unittest
import tempfile
from unittest.mock import patch
import b2d_controller_archive as archive


class ArchiveTests(unittest.TestCase):
    def test_snapshot_preserves_dirty_bytes_and_detects_runtime_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / 'repo'
            (repo / 'scripts' / 'drive_runtime').mkdir(parents=True)
            runtime = repo / 'scripts' / 'b2d_controller.py'
            runtime.write_text('# captured working bytes\n')
            sensor = repo / 'scripts' / 'drive_runtime' / 'sensors.py'
            sensor.write_text('# sensor version\n')
            config = tmp_path / 'config.json'
            config.write_text('{"wheelbase": 2.86}')
            out = tmp_path / 'snapshot'
            with patch.object(archive, 'REPO', repo), patch.object(archive, 'git', return_value='fixture'):
                manifest = archive.snapshot(out, [config])
                self.assertEqual((out / 'source/scripts/b2d_controller.py').read_bytes(), runtime.read_bytes())
                self.assertEqual(len(manifest['sources']), 2)
                archive.assert_sources_unchanged(manifest)
                sensor.write_text('# changed sensor version\n')
                with self.assertRaisesRegex(RuntimeError, 'sensors.py'):
                    archive.assert_sources_unchanged(manifest)
                with self.assertRaises(FileExistsError):
                    archive.snapshot(out, [config])

    def test_inventory_preserves_failed_and_partial_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root = tmp_path / 'run'
            root.mkdir()
            (root / 'failed.jsonl').write_text('{"status":"failed"}\n{"partial":')
            (root / 'empty.log').write_bytes(b'')
            output = tmp_path / 'inventory.json'
            result = archive.inventory(root, output)
            self.assertEqual(result['file_count'], 2)
            self.assertEqual({row['path'] for row in result['files']}, {'empty.log', 'failed.jsonl'})
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaises(FileExistsError):
                archive.inventory(root, output)


if __name__ == '__main__':
    unittest.main()
