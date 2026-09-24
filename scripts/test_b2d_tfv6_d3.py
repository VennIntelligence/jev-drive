"""D3 runner must import from the same sparse environment used in tmux."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class D3LaunchTests(unittest.TestCase):
    def test_runner_help_imports_without_inherited_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['OPENBLAS_CORETYPE'] = 'Barcelona'
        result = subprocess.run([sys.executable, str(root/'scripts/b2d_tfv6_d3.py'), '--help'],
                                cwd=root, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('factorial', result.stdout)
        self.assertIn('kalman', result.stdout)


if __name__ == '__main__':
    unittest.main()
